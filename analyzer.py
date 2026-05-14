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
MIN_GAMES     = 8        # Mínimo de partidos para análisis válido
MIN_PROB      = 0.90     # Probabilidad estadística mínima real
MIN_EDGE      = 0.05     # Edge mínimo sobre la prob. implícita sin vig
MIN_MINUTES   = 18.0     # Descartar jugadores con < 18 min promedio
WEIGHTS       = (0.50, 0.30, 0.20)  # Temporada / Últimos 10 / Últimos 5
MIN_LINE_RATIO = 0.60    # Línea mínima como % de la media (filtra líneas exóticas)


class EVAnalyzer:
    def __init__(self):
        load_dotenv()
        url = os.getenv("SUPABASE_URL")
        key = os.getenv("SUPABASE_KEY")
        self.supabase: Client = create_client(url, key) if url and key else None

        # Caches
        self._usage_cache = {}  # player_name → usage_rate
        self._pace_cache  = {}  # team_name   → pace

    # ═══════════════════════════════════════════
    # PRE-CARGA MASIVA (con fallback silencioso)
    # ═══════════════════════════════════════════

    def preload_league_stats(self):
        """
        Intenta cargar Usage Rate y Pace desde la NBA API.
        Si falla (IP bloqueada en la nube), usa defaults silenciosamente.
        """
        from nba_api.stats.endpoints import leaguedashplayerstats, leaguedashteamstats

        logger.info("Pre-cargando Usage Rate...")
        try:
            time.sleep(1)
            ps = leaguedashplayerstats.LeagueDashPlayerStats(
                season=SEASON,
                measure_type_detailed_defense='Advanced',
                per_mode_detailed='PerGame'
            )
            df = ps.get_data_frames()[0]
            for _, row in df.iterrows():
                name = row.get('PLAYER_NAME', '')
                usg  = row.get('USG_PCT', 0.20)
                if name:
                    self._usage_cache[name] = float(usg) if usg else 0.20
            logger.info(f"  → Usage Rate: {len(self._usage_cache)} jugadores cargados.")
        except Exception as e:
            logger.warning(f"  → Usage Rate no disponible ({type(e).__name__}). Usando 20% por defecto.")

        logger.info("Pre-cargando Pace de equipos...")
        try:
            time.sleep(1)
            ts = leaguedashteamstats.LeagueDashTeamStats(
                season=SEASON,
                measure_type_detailed_defense='Advanced',
                per_mode_detailed='PerGame'
            )
            df = ts.get_data_frames()[0]
            for _, row in df.iterrows():
                name = row.get('TEAM_NAME', '')
                pace = row.get('PACE', 100.0)
                if name:
                    self._pace_cache[name] = float(pace) if pace else 100.0
            logger.info(f"  → Pace: {len(self._pace_cache)} equipos cargados.")
        except Exception as e:
            logger.warning(f"  → Pace no disponible ({type(e).__name__}). Usando 100 por defecto.")

    # ═══════════════════════════════════════════
    # LECTURA DESDE SUPABASE (sin NBA API)
    # ═══════════════════════════════════════════

    def get_todays_odds(self):
        """Descarga las líneas de apuestas activas desde Supabase."""
        logger.info("Obteniendo cuotas desde Supabase...")
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

    def get_player_gamelog_from_db(self, player_name: str) -> pd.DataFrame | None:
        """
        Lee el historial de partidos del jugador directamente desde Supabase.
        No llama a la NBA API — funciona en cualquier entorno (nube incluida).
        """
        try:
            response = (
                self.supabase.table("player_stats")
                .select("*")
                .eq("player_name", player_name)
                .order("game_date", desc=True)
                .execute()
            )
            if not response.data:
                return None

            df = pd.DataFrame(response.data)
            df['PTS'] = pd.to_numeric(df['pts'], errors='coerce').fillna(0)
            df['REB'] = pd.to_numeric(df['reb'], errors='coerce').fillna(0)
            df['AST'] = pd.to_numeric(df['ast'], errors='coerce').fillna(0)
            df['MIN'] = pd.to_numeric(df['min'], errors='coerce').fillna(0)

            # Filtro de minutos mínimos
            if df['MIN'].mean() < MIN_MINUTES:
                logger.info(f"    → {player_name}: {df['MIN'].mean():.1f} min prom. Ignorado.")
                return None

            if len(df) < MIN_GAMES:
                logger.info(f"    → {player_name}: solo {len(df)} partidos en DB. Ejecuta update_stats.py.")
                return None

            return df

        except Exception as e:
            logger.error(f"Error leyendo player_stats para {player_name}: {e}")
            return None

    def get_team_gamelog_from_db(self, team_name: str) -> pd.DataFrame | None:
        """Lee el historial de W/L del equipo desde Supabase."""
        try:
            response = (
                self.supabase.table("team_game_logs")
                .select("*")
                .eq("team_name", team_name)
                .order("game_date", desc=True)
                .execute()
            )
            if not response.data:
                return None
            df = pd.DataFrame(response.data)
            if len(df) < MIN_GAMES:
                return None
            return df
        except Exception as e:
            logger.error(f"Error leyendo team_game_logs para {team_name}: {e}")
            return None

    # ═══════════════════════════════════════════
    # MATEMÁTICA CENTRAL
    # ═══════════════════════════════════════════

    @staticmethod
    def remove_vig(over_odds: float, under_odds: float) -> tuple:
        """Elimina el margen del casino. Retorna probs reales sin vig."""
        if over_odds and under_odds and over_odds > 0 and under_odds > 0:
            p_o = 1 / over_odds
            p_u = 1 / under_odds
            total = p_o + p_u
            return p_o / total, p_u / total
        VIG = 0.06
        p_o = (1 / over_odds)  * (1 - VIG) if over_odds  else 0
        p_u = (1 / under_odds) * (1 - VIG) if under_odds else 0
        return p_o, p_u

    @staticmethod
    def is_back_to_back(df: pd.DataFrame) -> bool:
        """True si el último partido fue ayer."""
        try:
            df = df.copy()
            df['game_date'] = pd.to_datetime(df['game_date'])
            last = df['game_date'].max()
            return (pd.Timestamp.now() - last).days <= 1
        except:
            return False

    def projected_stat(self, df: pd.DataFrame, stat_col: str,
                       usage: float, pace: float, b2b: bool) -> tuple:
        """Proyección ponderada con todos los factores."""
        l10 = df.head(10)
        l5  = df.head(5)

        m_s  = df[stat_col].mean()
        m_10 = l10[stat_col].mean()
        m_5  = l5[stat_col].mean()

        base = (m_s * WEIGHTS[0]) + (m_10 * WEIGHTS[1]) + (m_5 * WEIGHTS[2])

        usage_f = usage / 0.20
        pace_f  = pace  / 100.0
        b2b_pen = 0.92 if b2b else 1.0

        proj = base * usage_f * pace_f * b2b_pen

        std    = df[stat_col].std()
        median = df[stat_col].median()
        if pd.isna(std) or std < 0.1:
            std = 0.5

        cv = std / max(proj, 0.1)
        consistency = max(0.0, 100 - cv * 100)

        return proj, std, median, consistency

    def ensemble_prob(self, proj: float, std: float, line: float) -> tuple:
        """Ensemble Normal (60%) + Monte Carlo (40%)."""
        dist    = NormalDist(mu=proj, sigma=std)
        n_under = dist.cdf(line)
        n_over  = 1.0 - n_under

        rng     = np.random.default_rng(seed=42)
        samples = rng.normal(loc=proj, scale=std, size=10_000)
        mc_over  = float(np.mean(samples > line))
        mc_under = float(np.mean(samples < line))

        return (n_over * 0.60 + mc_over * 0.40), (n_under * 0.60 + mc_under * 0.40)

    # ═══════════════════════════════════════════
    # ANÁLISIS PRINCIPAL
    # ═══════════════════════════════════════════

    def calculate_ev(self):
        # Intentar pre-cargar Usage/Pace (falla silenciosamente si estamos en la nube)
        self.preload_league_stats()

        odds_data = self.get_todays_odds()
        if not odds_data:
            logger.warning("No hay cuotas. Ejecuta odds.py primero.")
            return

        entities = list(set(o["player_name"] for o in odds_data))
        logger.info(f"\nAnalizando {len(entities)} entidades desde Supabase (sin NBA API)...")

        ev_bets = []

        for name in entities:
            entity_odds = [o for o in odds_data if o["player_name"] == name]
            is_team     = any(o["market"] == "h2h" for o in entity_odds)

            # ── EQUIPO ────────────────────────────────────────────────────
            if is_team:
                logger.info(f"  [EQUIPO] {name}")
                season_df = self.get_team_gamelog_from_db(name)
                if season_df is None:
                    logger.warning(f"    → Sin datos en team_game_logs. Ejecuta update_stats.py.")
                    continue

                l10 = season_df.head(10)
                l5  = season_df.head(5)
                win_s  = (season_df['wl'] == 'W').mean()
                win_10 = (l10['wl'] == 'W').mean()
                win_5  = (l5['wl'] == 'W').mean()
                proj_w = win_s * 0.50 + win_10 * 0.30 + win_5 * 0.20

                for odds in entity_odds:
                    if odds["market"] != "h2h" or not odds.get("over_odds"):
                        continue
                    o_odds    = odds["over_odds"]
                    u_odds    = odds.get("under_odds")
                    matchup   = odds.get("matchup", "?")
                    game_time = odds.get("game_time", "?")

                    p_win_impl, _ = self.remove_vig(o_odds, u_odds or 0)
                    edge = proj_w - p_win_impl

                    if edge >= MIN_EDGE and proj_w >= 0.85:
                        ev_bets.append({
                            "player/team": name,
                            "market": "Moneyline",
                            "bet": "WIN",
                            "odds": o_odds,
                            "bookie": odds["bookmaker"],
                            "prob_real": f"{proj_w*100:.1f}%",
                            "prob_casino_novig": f"{p_win_impl*100:.1f}%",
                            "edge": f"+{edge*100:.1f}%",
                            "info": f"Win%: S{win_s*100:.0f}% L10:{win_10*100:.0f}% L5:{win_5*100:.0f}%",
                            "matchup": matchup,
                            "game_time": game_time
                        })
                continue

            # ── JUGADOR ───────────────────────────────────────────────────
            logger.info(f"  [JUGADOR] {name}")
            season_df = self.get_player_gamelog_from_db(name)
            if season_df is None:
                continue

            usage = self._usage_cache.get(name, 0.20)
            b2b   = self.is_back_to_back(season_df)
            if b2b:
                logger.info(f"    → B2B detectado.")

            for odds in entity_odds:
                market  = odds["market"]
                line    = odds.get("line")
                o_odds  = odds.get("over_odds")
                u_odds  = odds.get("under_odds")
                bookie  = odds["bookmaker"]
                matchup = odds.get("matchup", "?")
                gtime   = odds.get("game_time", "?")

                stat_map = {
                    "player_points":   "PTS",
                    "player_rebounds": "REB",
                    "player_assists":  "AST"
                }
                stat_col = stat_map.get(market)
                if not stat_col or stat_col not in season_df.columns:
                    continue

                # Pace del equipo del jugador
                try:
                    player_team = matchup.split(" vs ")[0].strip() if " vs " in matchup else ""
                    pace = self._pace_cache.get(player_team, 100.0)
                except:
                    pace = 100.0

                proj, std, median, consistency = self.projected_stat(
                    season_df, stat_col, usage, pace, b2b
                )

                # Filtrar líneas exóticas muy bajas
                if proj > 0 and line and (line / proj) < MIN_LINE_RATIO:
                    logger.info(f"    → Línea {line} muy baja vs media {proj:.1f}. Ignorada.")
                    continue

                p_over_imp, p_under_imp = self.remove_vig(o_odds, u_odds)
                prob_over, prob_under   = self.ensemble_prob(proj, std, line)

                ctx = (
                    f"Media:{proj:.1f} Med:{median:.1f} Dev:{std:.1f} "
                    f"Consist:{consistency:.0f}% USG:{usage*100:.1f}% "
                    f"Pace:{pace:.1f} B2B:{b2b}"
                )

                if o_odds and prob_over >= MIN_PROB:
                    edge_over = prob_over - p_over_imp
                    if edge_over >= MIN_EDGE:
                        ev_bets.append({
                            "player/team": name,
                            "market": market,
                            "bet": f"OVER {line}",
                            "odds": o_odds,
                            "bookie": bookie,
                            "prob_real": f"{prob_over*100:.1f}%",
                            "prob_casino_novig": f"{p_over_imp*100:.1f}%",
                            "edge": f"+{edge_over*100:.1f}%",
                            "info": ctx,
                            "matchup": matchup,
                            "game_time": gtime
                        })

                if u_odds and prob_under >= MIN_PROB:
                    edge_under = prob_under - p_under_imp
                    if edge_under >= MIN_EDGE:
                        ev_bets.append({
                            "player/team": name,
                            "market": market,
                            "bet": f"UNDER {line}",
                            "odds": u_odds,
                            "bookie": bookie,
                            "prob_real": f"{prob_under*100:.1f}%",
                            "prob_casino_novig": f"{p_under_imp*100:.1f}%",
                            "edge": f"+{edge_under*100:.1f}%",
                            "info": ctx,
                            "matchup": matchup,
                            "game_time": gtime
                        })

        # ═══════════════════════════════════════
        # REPORTE FINAL
        # ═══════════════════════════════════════
        if ev_bets:
            ev_df = pd.DataFrame(ev_bets)
            ev_df = ev_df.sort_values("odds", ascending=False)
            ev_df = ev_df.drop_duplicates(subset=["player/team", "market", "bet"], keep="first")
            ev_df = ev_df.sort_values("edge", ascending=False)
            ev_df = ev_df.drop_duplicates(subset=["player/team", "market"], keep="first")
            ev_df = ev_df.sort_values("edge", ascending=False).reset_index(drop=True)

            print("\n" + "="*90)
            print("=== MODELO CUANTITATIVO PRO — APUESTAS +EV DEL DÍA ===")
            print("="*90)
            print(ev_df.to_string())
            print(f"\nTotal apuestas únicas: {len(ev_df)}")
            self.send_telegram_alert(ev_df.to_dict("records"))
        else:
            msg = "No se encontraron apuestas que cumplan los criterios hoy."
            logger.info(msg)
            self._send_telegram_msg(f"🤖 <b>Bot EV Pro</b>\n\n{msg}")

    # ═══════════════════════════════════════════
    # TELEGRAM
    # ═══════════════════════════════════════════

    def _send_telegram_msg(self, text: str):
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
        import requests
        token   = os.getenv("TELEGRAM_BOT_TOKEN")
        chat_id = os.getenv("TELEGRAM_CHAT_ID")
        if not token or not chat_id:
            return

        url = f"https://api.telegram.org/bot{token}/sendMessage"

        msg  = "🤖 <b>MODELO PRO — APUESTAS +EV DEL DÍA</b> 🤖\n"
        msg += f"<i>Prob≥{int(MIN_PROB*100)}% | Edge≥{int(MIN_EDGE*100)}% | Min≥{MIN_MINUTES}min | Vig eliminado</i>\n\n"

        for idx, bet in enumerate(ev_bets, 1):
            b2b = "⚠️ <b>B2B</b> " if "B2B:True" in bet.get("info", "") else ""
            text  = f"<b>{idx}. {bet['player/team']}</b> {b2b}\n"
            text += f"🏟️ {bet.get('matchup','?')} — {bet.get('game_time','?')}\n"
            text += f"🎯 <b>{bet['bet']}</b> ({bet['market']})\n"
            text += f"💰 Cuota: <b>{bet['odds']}</b> en {bet['bookie']}\n"
            text += f"📊 Prob Real: <b>{bet['prob_real']}</b> | Sin vig: {bet['prob_casino_novig']}\n"
            text += f"📈 Edge: <b>{bet['edge']}</b>\n"
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
                logger.info(f"Telegram enviado. {len(ev_bets)} apuestas.")
            except Exception as e:
                logger.error(f"Telegram error: {e}")


if __name__ == "__main__":
    analyzer = EVAnalyzer()
    analyzer.calculate_ev()
