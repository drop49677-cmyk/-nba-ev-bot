import os
import time
import pandas as pd
import logging
import numpy as np
from statistics import NormalDist
from dotenv import load_dotenv
from supabase import create_client, Client

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("EV_Bot_Analyzer")

# ─────────────────────────────────────────────
# CONFIGURACIÓN DEL MODELO
# ─────────────────────────────────────────────
SEASON        = '2025-26'
MIN_GAMES     = 8        # Mínimo de partidos para que el análisis sea válido
MIN_PROB      = 0.90     # Probabilidad estadística mínima real
MIN_EDGE      = 0.05     # Edge mínimo sobre la prob. implícita sin vig
MIN_MINUTES   = 18.0     # Descartar jugadores con < 18 min promedio (garbage time)
WEIGHTS       = (0.50, 0.30, 0.20)  # Temporada / Últimos 10 / Últimos 5
# Línea mínima como % de la media proyectada (filtra líneas exóticas/alternativas muy bajas)
MIN_LINE_RATIO = 0.60    # La línea debe ser al menos el 60% de la media del jugador


class EVAnalyzer:
    def __init__(self):
        load_dotenv()
        url = os.getenv("SUPABASE_URL")
        key = os.getenv("SUPABASE_KEY")
        self.supabase: Client = create_client(url, key) if url and key else None

        # Caches precargados al inicio (una sola llamada por categoría)
        self._usage_cache = {}   # player_name → usage_rate (float)
        self._pace_cache  = {}   # team_name   → pace (float)
        self._injured     = set()  # set de nombres de jugadores lesionados

    # ═══════════════════════════════════════════
    # PRE-CARGA MASIVA (una sola vez al arrancar)
    # ═══════════════════════════════════════════

    def preload_league_stats(self):
        """
        Descarga en UNA sola llamada las estadísticas avanzadas de TODA la liga:
        - Usage Rate de todos los jugadores
        - Pace de todos los equipos
        Esto es mucho más eficiente que una llamada por jugador.
        """
        from nba_api.stats.endpoints import leaguedashplayerstats, leaguedashteamstats

        # ── Usage Rate de jugadores ────────────────────────────────────────
        logger.info("Pre-cargando Usage Rate de todos los jugadores...")
        try:
            time.sleep(1)
            player_stats = leaguedashplayerstats.LeagueDashPlayerStats(
                season=SEASON,
                measure_type_detailed_defense='Advanced',
                per_mode_detailed='PerGame'
            )
            df_players = player_stats.get_data_frames()[0]
            for _, row in df_players.iterrows():
                name = row.get('PLAYER_NAME', '')
                usg  = row.get('USG_PCT', 0.20)
                if name:
                    self._usage_cache[name] = float(usg) if usg else 0.20
            logger.info(f"  → Usage cargado para {len(self._usage_cache)} jugadores.")
        except Exception as e:
            logger.warning(f"  → No se pudo cargar Usage Rate: {e}. Se usará 20% por defecto.")

        # ── Pace de equipos ────────────────────────────────────────────────
        logger.info("Pre-cargando Pace de todos los equipos...")
        try:
            time.sleep(1)
            team_stats = leaguedashteamstats.LeagueDashTeamStats(
                season=SEASON,
                measure_type_detailed_defense='Advanced',
                per_mode_detailed='PerGame'
            )
            df_teams = team_stats.get_data_frames()[0]
            for _, row in df_teams.iterrows():
                name = row.get('TEAM_NAME', '')
                pace = row.get('PACE', 100.0)
                if name:
                    self._pace_cache[name] = float(pace) if pace else 100.0
            logger.info(f"  → Pace cargado para {len(self._pace_cache)} equipos.")
        except Exception as e:
            logger.warning(f"  → No se pudo cargar Pace: {e}. Se usará 100 por defecto.")

    def preload_injury_report(self):
        """
        Descarga el injury report directamente del CDN público de NBA.com.
        Descarta jugadores marcados como Out o Doubtful.
        """
        import requests as req
        logger.info("Pre-cargando injury report (NBA CDN)...")
        try:
            url = "https://cdn.nba.com/static/json/liveData/injuryreport/injuryreport.json"
            r = req.get(url, headers={"User-Agent": "Mozilla/5.0"}, timeout=10)
            r.raise_for_status()
            players = r.json().get("injuryReport", {}).get("InjuredPlayers", [])
            for p in players:
                status = p.get("Status", "")
                pname  = p.get("PlayerName", "")
                if status in ("Out", "Doubtful") and pname:
                    self._injured.add(pname)
            logger.info(f"  → {len(self._injured)} jugadores Out/Doubtful en injury report.")
        except Exception as e:
            logger.warning(f"  → Injury report no disponible: {e}. No se aplicará filtro de lesiones.")

    # ═══════════════════════════════════════════
    # OBTENCIÓN DE DATOS
    # ═══════════════════════════════════════════

    def get_todays_odds(self):
        """Descarga las líneas de apuestas activas desde Supabase."""
        logger.info("Obteniendo cuotas activas desde Supabase...")
        if not self.supabase:
            return []
        try:
            response = self.supabase.table("player_odds").select("*").execute()
            data = response.data
            logger.info(f"  → {len(data)} líneas encontradas.")
            return data
        except Exception as e:
            logger.error(f"Error consultando Supabase: {e}")
            return []

    def get_player_gamelog(self, player_name: str):
        """Retorna el historial completo de partidos del jugador en la temporada."""
        from nba_api.stats.static import players as nba_players_static
        from nba_api.stats.endpoints import playergamelog

        nba_players = nba_players_static.find_players_by_full_name(player_name)
        if not nba_players:
            return None

        player_id = nba_players[0]['id']
        try:
            time.sleep(0.6)
            gamelog = playergamelog.PlayerGameLog(player_id=player_id, season=SEASON)
            df = gamelog.get_data_frames()[0]
            if df.empty or len(df) < MIN_GAMES:
                return None
            df['MIN'] = pd.to_numeric(df['MIN'], errors='coerce')
            if df['MIN'].mean() < MIN_MINUTES:
                logger.info(f"    → {player_name} promedia {df['MIN'].mean():.1f} min. Ignorado.")
                return None
            return df
        except Exception as e:
            logger.error(f"Error gamelog {player_name}: {e}")
            return None

    def get_team_gamelog(self, team_name: str):
        """Retorna el historial completo de partidos del equipo."""
        from nba_api.stats.static import teams as nba_teams_static
        from nba_api.stats.endpoints import teamgamelog

        nba_teams = nba_teams_static.find_teams_by_full_name(team_name)
        if not nba_teams:
            return None
        team_id = nba_teams[0]['id']
        try:
            time.sleep(0.6)
            gamelog = teamgamelog.TeamGameLog(team_id=team_id, season=SEASON)
            return gamelog.get_data_frames()[0]
        except Exception as e:
            logger.error(f"Error gamelog equipo {team_name}: {e}")
            return None

    # ═══════════════════════════════════════════
    # MATEMÁTICA CENTRAL
    # ═══════════════════════════════════════════

    @staticmethod
    def remove_vig(over_odds: float, under_odds: float) -> tuple:
        """
        Elimina el margen del casino (vig) usando ambas cuotas.
        Retorna las probabilidades reales implícitas sin vig.

        Si solo tenemos una cuota, asumimos vig estándar del 6%.
        """
        if over_odds and under_odds and over_odds > 0 and under_odds > 0:
            # Método de normalización bilateral (más preciso)
            p_over_raw  = 1 / over_odds
            p_under_raw = 1 / under_odds
            total       = p_over_raw + p_under_raw  # > 1.0 por el vig
            p_over_novig  = p_over_raw  / total
            p_under_novig = p_under_raw / total
        else:
            VIG = 0.06
            p_over_novig  = (1 / over_odds)  * (1 - VIG) if over_odds  else 0
            p_under_novig = (1 / under_odds) * (1 - VIG) if under_odds else 0
        return p_over_novig, p_under_novig

    @staticmethod
    def is_back_to_back(season_df: pd.DataFrame) -> bool:
        """True si el último partido registrado fue ayer (B2B)."""
        try:
            season_df = season_df.copy()
            season_df['GAME_DATE'] = pd.to_datetime(season_df['GAME_DATE'])
            last_game = season_df['GAME_DATE'].max()
            diff = (pd.Timestamp.now() - last_game).days
            return diff <= 1
        except:
            return False

    def projected_stat(self, season_df: pd.DataFrame, stat_col: str,
                       usage: float, pace: float, b2b: bool) -> tuple:
        """
        Proyección ponderada del stat con todos los factores contextuales.
        Retorna (proj_mean, std_dev, median, consistency_pct).
        """
        l10 = season_df.head(10)
        l5  = season_df.head(5)

        m_season = season_df[stat_col].mean()
        m_10     = l10[stat_col].mean()
        m_5      = l5[stat_col].mean()

        base = (m_season * WEIGHTS[0]) + (m_10 * WEIGHTS[1]) + (m_5 * WEIGHTS[2])

        # Factores de ajuste (normalizados)
        usage_factor = usage / 0.20          # 20% = promedio de la liga
        pace_factor  = pace  / 100.0         # 100 = promedio de la liga
        b2b_penalty  = 0.92 if b2b else 1.0  # -8% en B2B (dato empírico NBA)

        proj_mean = base * usage_factor * pace_factor * b2b_penalty

        std_dev = season_df[stat_col].std()
        median  = season_df[stat_col].median()
        if pd.isna(std_dev) or std_dev < 0.1:
            std_dev = 0.5

        # Consistencia: 100 = extremadamente consistente, 0 = muy errático
        cv = std_dev / max(proj_mean, 0.1)
        consistency = max(0.0, 100 - cv * 100)

        return proj_mean, std_dev, median, consistency

    def ensemble_prob(self, proj_mean: float, std_dev: float, line: float) -> tuple:
        """
        Ensemble: Normal (60%) + Monte Carlo (40%) para mayor robustez.
        Retorna (prob_over, prob_under).
        """
        # Distribución Normal
        dist = NormalDist(mu=proj_mean, sigma=std_dev)
        n_under = dist.cdf(line)
        n_over  = 1.0 - n_under

        # Monte Carlo (10 000 simulaciones, seed fijo para reproducibilidad)
        rng     = np.random.default_rng(seed=42)
        samples = rng.normal(loc=proj_mean, scale=std_dev, size=10_000)
        mc_over  = float(np.mean(samples > line))
        mc_under = float(np.mean(samples < line))

        prob_over  = n_over  * 0.60 + mc_over  * 0.40
        prob_under = n_under * 0.60 + mc_under * 0.40
        return prob_over, prob_under

    # ═══════════════════════════════════════════
    # ANÁLISIS PRINCIPAL
    # ═══════════════════════════════════════════

    def calculate_ev(self):
        # 1. Precarga masiva (una sola vez)
        self.preload_league_stats()
        self.preload_injury_report()

        # 2. Cuotas del día
        odds_data = self.get_todays_odds()
        if not odds_data:
            logger.warning("No hay cuotas. Ejecuta odds.py primero.")
            return

        entities = list(set(o["player_name"] for o in odds_data))
        logger.info(f"\nAnalizando {len(entities)} entidades con Modelo Cuantitativo Pro...")

        ev_bets = []

        for name in entities:
            entity_odds = [o for o in odds_data if o["player_name"] == name]
            is_team     = any(o["market"] == "h2h" for o in entity_odds)

            # ── EQUIPO (Moneyline) ────────────────────────────────────────
            if is_team:
                logger.info(f"  [EQUIPO] {name}")
                season_df = self.get_team_gamelog(name)
                if season_df is None or len(season_df) < MIN_GAMES:
                    continue

                l10 = season_df.head(10)
                l5  = season_df.head(5)
                win_s   = (season_df['WL'] == 'W').mean()
                win_10  = (l10['WL'] == 'W').mean()
                win_5   = (l5['WL'] == 'W').mean()
                proj_w  = win_s * 0.50 + win_10 * 0.30 + win_5 * 0.20

                for odds in entity_odds:
                    if odds["market"] != "h2h":
                        continue
                    o_odds = odds.get("over_odds")
                    u_odds = odds.get("under_odds")
                    if not o_odds:
                        continue

                    matchup   = odds.get("matchup", "?")
                    game_time = odds.get("game_time", "?")

                    # Probabilidad implícita sin vig
                    p_win_implied, _ = self.remove_vig(o_odds, u_odds or 0)
                    edge = proj_w - p_win_implied

                    if edge >= MIN_EDGE and proj_w >= 0.85:
                        ev_bets.append({
                            "player/team": name,
                            "market": "Moneyline",
                            "bet": "WIN",
                            "odds": o_odds,
                            "bookie": odds["bookmaker"],
                            "prob_real": f"{proj_w*100:.1f}%",
                            "prob_casino_novig": f"{p_win_implied*100:.1f}%",
                            "edge": f"+{edge*100:.1f}%",
                            "info": f"Win%: S{win_s*100:.0f}% L10:{win_10*100:.0f}% L5:{win_5*100:.0f}%",
                            "matchup": matchup,
                            "game_time": game_time
                        })
                continue

            # ── JUGADOR ───────────────────────────────────────────────────
            logger.info(f"  [JUGADOR] {name}")

            # Filtro de lesiones
            if name in self._injured:
                logger.info(f"    → {name} está en injury report (Out/Doubtful). Ignorado.")
                continue

            season_df = self.get_player_gamelog(name)
            if season_df is None:
                continue

            usage = self._usage_cache.get(name, 0.20)
            b2b   = self.is_back_to_back(season_df)

            if b2b:
                logger.info(f"    → B2B detectado para {name}.")

            for odds in entity_odds:
                market  = odds["market"]
                line    = odds["line"]
                o_odds  = odds.get("over_odds")
                u_odds  = odds.get("under_odds")
                bookie  = odds["bookmaker"]
                matchup = odds.get("matchup", "?")
                gtime   = odds.get("game_time", "?")

                stat_map = {
                    "player_points": "PTS",
                    "player_rebounds": "REB",
                    "player_assists": "AST"
                }
                stat_col = stat_map.get(market)
                if not stat_col or stat_col not in season_df.columns:
                    continue

                # Encontrar el equipo del jugador para obtener el Pace
                try:
                    player_team = matchup.split(" vs ")[0].strip() if " vs " in matchup else ""
                    pace = self._pace_cache.get(player_team, 100.0)
                except:
                    pace = 100.0

                # Proyección con todos los factores
                proj_mean, std_dev, median, consistency = self.projected_stat(
                    season_df, stat_col, usage, pace, b2b
                )

                # ── Filtro de líneas exóticas/alternativas muy bajas ──────
                # Si la línea es menor al 60% del promedio proyectado,
                # es casi imposible de NO superar → dato inútil estadísticamente.
                if proj_mean > 0 and (line / proj_mean) < MIN_LINE_RATIO:
                    logger.info(f"    → Línea {line} muy baja vs media {proj_mean:.1f} para {name}. Ignorada.")
                    continue

                # Probabilidades sin vig (comparación justa)
                p_over_implied, p_under_implied = self.remove_vig(o_odds, u_odds)

                # Probabilidades reales (Ensemble Normal + Monte Carlo)
                prob_over, prob_under = self.ensemble_prob(proj_mean, std_dev, line)

                context = (
                    f"Media:{proj_mean:.1f} Med:{median:.1f} Dev:{std_dev:.1f} "
                    f"Consist:{consistency:.0f}% USG:{usage*100:.1f}% "
                    f"Pace:{pace:.1f} B2B:{b2b}"
                )

                # ── OVER ──────────────────────────────────────────────────
                if o_odds and prob_over >= MIN_PROB:
                    edge_over = prob_over - p_over_implied
                    if edge_over >= MIN_EDGE:
                        ev_bets.append({
                            "player/team": name,
                            "market": market,
                            "bet": f"OVER {line}",
                            "odds": o_odds,
                            "bookie": bookie,
                            "prob_real": f"{prob_over*100:.1f}%",
                            "prob_casino_novig": f"{p_over_implied*100:.1f}%",
                            "edge": f"+{edge_over*100:.1f}%",
                            "info": context,
                            "matchup": matchup,
                            "game_time": gtime
                        })

                # ── UNDER ─────────────────────────────────────────────────
                if u_odds and prob_under >= MIN_PROB:
                    edge_under = prob_under - p_under_implied
                    if edge_under >= MIN_EDGE:
                        ev_bets.append({
                            "player/team": name,
                            "market": market,
                            "bet": f"UNDER {line}",
                            "odds": u_odds,
                            "bookie": bookie,
                            "prob_real": f"{prob_under*100:.1f}%",
                            "prob_casino_novig": f"{p_under_implied*100:.1f}%",
                            "edge": f"+{edge_under*100:.1f}%",
                            "info": context,
                            "matchup": matchup,
                            "game_time": gtime
                        })

        # ═══════════════════════════════════════
        # REPORTE FINAL
        # ═══════════════════════════════════════
        if ev_bets:
            ev_df = pd.DataFrame(ev_bets)

            # Mejor cuota por apuesta (sin duplicados por bookie)
            ev_df = ev_df.sort_values("odds", ascending=False)
            ev_df = ev_df.drop_duplicates(subset=["player/team", "market", "bet"], keep="first")

            # Un solo resultado por jugador+mercado (la mejor línea matemáticamente)
            ev_df = ev_df.sort_values("edge", ascending=False)
            ev_df = ev_df.drop_duplicates(subset=["player/team", "market"], keep="first")

            # Orden final: mayor Edge primero
            ev_df = ev_df.sort_values("edge", ascending=False).reset_index(drop=True)

            print("\n" + "="*90)
            print("=== MODELO CUANTITATIVO PRO v2 — APUESTAS +EV DEL DÍA ===")
            print("="*90)
            print(ev_df.to_string())
            print(f"\nTotal apuestas únicas: {len(ev_df)}")

            self.send_telegram_alert(ev_df.to_dict("records"))
        else:
            msg = "No se encontraron apuestas que cumplan todos los criterios hoy."
            logger.info(msg)
            self._send_telegram_msg(f"🤖 <b>Bot EV Pro</b>\n\n{msg}")

    # ═══════════════════════════════════════════
    # TELEGRAM
    # ═══════════════════════════════════════════

    def _send_telegram_msg(self, text: str):
        """Envía un mensaje simple a Telegram."""
        import requests
        token   = os.getenv("TELEGRAM_BOT_TOKEN")
        chat_id = os.getenv("TELEGRAM_CHAT_ID")
        if not token or not chat_id:
            return
        try:
            requests.post(
                f"https://api.telegram.org/bot{token}/sendMessage",
                json={"chat_id": chat_id, "text": text, "parse_mode": "HTML"},
                timeout=10
            )
        except Exception as e:
            logger.error(f"Telegram error: {e}")

    def send_telegram_alert(self, ev_bets: list):
        """Envía todas las apuestas del día a Telegram (auto-dividido si supera 4000 chars)."""
        import requests
        token   = os.getenv("TELEGRAM_BOT_TOKEN")
        chat_id = os.getenv("TELEGRAM_CHAT_ID")
        if not token or not chat_id:
            return

        url = f"https://api.telegram.org/bot{token}/sendMessage"

        msg  = "🤖 <b>MODELO PRO v2 — APUESTAS +EV DEL DÍA</b> 🤖\n"
        msg += f"<i>Filtros: Prob≥{int(MIN_PROB*100)}% | Edge≥{int(MIN_EDGE*100)}% | Min≥{MIN_MINUTES}min | Vig eliminado | Lesiones excluidas</i>\n\n"

        for idx, bet in enumerate(ev_bets, 1):
            b2b_tag = "⚠️ <b>B2B</b> " if "B2B:True" in bet.get("info", "") else ""
            inj_tag = ""
            text  = f"<b>{idx}. {bet['player/team']}</b> {b2b_tag}{inj_tag}\n"
            text += f"🏟️ {bet.get('matchup','?')} — {bet.get('game_time','?')}\n"
            text += f"🎯 <b>{bet['bet']}</b> ({bet['market']})\n"
            text += f"💰 Cuota: <b>{bet['odds']}</b> en {bet['bookie']}\n"
            text += f"📊 Prob Real: <b>{bet['prob_real']}</b> | Sin vig casino: {bet['prob_casino_novig']}\n"
            text += f"📈 Edge (sin vig): <b>{bet['edge']}</b>\n"
            text += f"🔬 {bet.get('info','')}\n\n"

            if len(msg) + len(text) > 4000:
                requests.post(url, json={"chat_id": chat_id, "text": msg, "parse_mode": "HTML"}, timeout=10)
                msg = text
            else:
                msg += text

        if msg.strip():
            try:
                res = requests.post(url, json={"chat_id": chat_id, "text": msg, "parse_mode": "HTML"}, timeout=10)
                res.raise_for_status()
                logger.info(f"Telegram enviado. Total apuestas: {len(ev_bets)}")
            except Exception as e:
                logger.error(f"Error Telegram: {e}")


if __name__ == "__main__":
    analyzer = EVAnalyzer()
    analyzer.calculate_ev()
