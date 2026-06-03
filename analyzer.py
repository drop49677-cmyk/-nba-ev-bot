"""
analyzer.py — Motor de análisis +EV de élite.
Mejoras v3:
  - Penalización por volatilidad (CV > 0.45)
  - Filtro de hit rate empírico (≥50% en últimos 10 juegos)
  - Cap de EV al 20% (evita picks con EV ficticio)
  - Ensemble de 5 modelos (Normal, Empírico, Monte Carlo, Poisson, Quantile)
  - Ajuste por fortaleza defensiva del oponente
  - Book consensus como señal adicional
  - Scoring adaptativo por tipo de mercado
"""
import json
import logging
import math
import os
import sys
from collections import Counter
from datetime import datetime, timezone
from statistics import NormalDist

import numpy as np
import pandas as pd
import requests
from dotenv import load_dotenv
from supabase import Client, create_client



load_dotenv()

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger("EV_Bot_Analyzer")


SEASON = os.getenv("NBA_SEASON", "2025-26")
BANKROLL = float(os.getenv("BANKROLL", 1000))
ALLOWED_BOOKIES = [
    b.strip().lower()
    for b in os.getenv("ALLOWED_BOOKIES", "").split(",")
    if b.strip()
]

# ── Configuración de LOCK_MODE ("Apuestas Fijas") ──────────────────────────────
# Si LOCK_MODE=1 en .env, aplicamos filtros de élite para seleccionar solo apuestas ultra seguras
LOCK_MODE = os.getenv("LOCK_MODE", "1").lower() not in {"0", "false", "no"}

# ── Thresholds del modelo ──────────────────────────────────────────────────────
# Preferimos pocos picks de alta confianza sobre muchos picks mediocres.
MIN_GAMES         = int(os.getenv("MIN_GAMES", 8))
MIN_MINUTES       = float(os.getenv("MIN_MINUTES", 18.0))

if LOCK_MODE:
    logger.info("🔒 MODO LOCK (FIJAS) ACTIVO: Se aplicarán filtros de élite ultra selectivos.")
    MIN_PROB          = float(os.getenv("MIN_PROB", 0.84))       # Probabilidad mínima matemática (84%+)
    MIN_EDGE          = float(os.getenv("MIN_EDGE", 0.06))       # EV mínimo (6%+)
    MIN_SCORE         = float(os.getenv("MIN_SCORE", 78.0))      # Solo picks de grado S / A+
    MAX_CV_NO_PENALTY     = float(os.getenv("MAX_CV_NO_PENALTY", 0.38)) # Volatilidad baja (CV <= 0.38)
    MIN_OVER_HIT_RATE     = float(os.getenv("MIN_OVER_HIT_RATE", 0.70))  # Hit rate últimos 10 >= 70%
    MIN_UNDER_HIT_RATE    = float(os.getenv("MIN_UNDER_HIT_RATE", 0.70)) # Hit rate últimos 10 >= 70%
    
    # IMPORTANTE: Al subir la cuota mínima a 1.65 y bajar la máxima a 2.20,
    # descartamos líneas alternativas que solo se ofrecen en ciertas casas de apuestas
    # y forzamos al bot a jugar únicamente en la LÍNEA PRINCIPAL (la estándar con cuotas balanceadas ~1.85).
    MIN_ODDS              = float(os.getenv("MIN_ODDS", 1.65))
    MAX_ODDS              = float(os.getenv("MAX_ODDS", 2.20))
else:
    MIN_PROB          = float(os.getenv("MIN_PROB", 0.78))
    MIN_EDGE          = float(os.getenv("MIN_EDGE", 0.04))
    MIN_SCORE         = float(os.getenv("MIN_SCORE", 62.0))
    MAX_CV_NO_PENALTY     = float(os.getenv("MAX_CV_NO_PENALTY", 0.45))
    MIN_OVER_HIT_RATE     = float(os.getenv("MIN_OVER_HIT_RATE", 0.45))
    MIN_UNDER_HIT_RATE    = float(os.getenv("MIN_UNDER_HIT_RATE", 0.45))
    MIN_ODDS              = float(os.getenv("MIN_ODDS", 1.35))
    MAX_ODDS              = float(os.getenv("MAX_ODDS", 3.00))

MIN_KELLY         = float(os.getenv("MIN_KELLY", 1.5))
MAX_PLAYER_STAKE_PCT  = float(os.getenv("MAX_PLAYER_STAKE_PCT", 0.03))
MAX_GAME_STAKE_PCT    = float(os.getenv("MAX_GAME_STAKE_PCT", 0.04))
MAX_PICKS_TOTAL       = int(os.getenv("MAX_PICKS_TOTAL", 5))
MAX_PICKS_PER_GAME    = int(os.getenv("MAX_PICKS_PER_GAME", 2))
MAX_PICKS_PER_PLAYER  = int(os.getenv("MAX_PICKS_PER_PLAYER", 1))
MIN_MARKET_CONFIDENCE_PROB  = float(os.getenv("MIN_MARKET_CONFIDENCE_PROB", 0.68))
MAX_MARKET_CONFIDENCE_ODDS  = float(os.getenv("MAX_MARKET_CONFIDENCE_ODDS", 1.55))
KELLY_FRACTION  = float(os.getenv("KELLY_FRACTION", 0.25))
SEND_TELEGRAM   = os.getenv("SEND_TELEGRAM", "1").lower() not in {"0", "false", "no"}

# ── Nuevos parámetros de calidad ──────────────────────────────────────────────
# EV máximo creíble. EV > 20% casi siempre indica sobreestimación del modelo.
MAX_CREDIBLE_EV       = float(os.getenv("MAX_CREDIBLE_EV", 0.20))
# Cuántos libros deben coincidir para book consensus (señal alcista/bajista)
MIN_CONSENSUS_BOOKS   = int(os.getenv("MIN_CONSENSUS_BOOKS", 3))



PINNACLE_KEY = "pinnacle"


MARKET_STATS = {
    "player_points":                  ["pts", "PTS"],
    "player_rebounds":                ["reb", "REB"],
    "player_assists":                 ["ast", "AST"],
    "player_threes":                  ["fg3m", "FG3M"],
    "player_blocks":                  ["blk", "BLK"],
    "player_steals":                  ["stl", "STL"],
    "player_turnovers":               ["tov", "TOV"],
    "player_points_rebounds_assists": ["pra"],
}

COUNT_MARKETS = {
    "player_assists",
    "player_threes",
    "player_blocks",
    "player_steals",
    "player_turnovers",
}

# Multiplicador de umbral MIN_PROB por mercado.
# Stats de conteo (robos, bloqueos) son más erráticos → exigir más confianza.
MARKET_PROB_MULT = {
    "player_points":   1.00,
    "player_rebounds": 1.00,
    "player_assists":  1.02,
    "player_threes":   1.04,
    "player_blocks":   1.06,
    "player_steals":   1.06,
    "player_turnovers": 1.04,
    "player_points_rebounds_assists": 0.98,
}

MARKET_TRANSLATIONS = {
    "player_points":                  "Puntos",
    "player_rebounds":                "Rebotes",
    "player_assists":                 "Asistencias",
    "player_points_rebounds_assists": "Puntos + Rebotes + Asistencias",
    "player_threes":                  "Triples",
    "player_steals":                  "Robos",
    "player_blocks":                  "Bloqueos",
    "player_turnovers":               "Perdidas",
    "h2h":     "Ganador del Partido",
    "spreads": "Handicap",
    "totals":  "Total del Partido",
}

ABBR_TO_TEAM_MAP = {
    'ATL': 'Atlanta Hawks', 'BOS': 'Boston Celtics', 'CLE': 'Cleveland Cavaliers',
    'NOP': 'New Orleans Pelicans', 'CHI': 'Chicago Bulls', 'DAL': 'Dallas Mavericks',
    'DEN': 'Denver Nuggets', 'GSW': 'Golden State Warriors', 'HOU': 'Houston Rockets',
    'LAC': 'Los Angeles Clippers', 'LAL': 'Los Angeles Lakers', 'MIA': 'Miami Heat',
    'MIL': 'Milwaukee Bucks', 'MIN': 'Minnesota Timberwolves', 'BKN': 'Brooklyn Nets',
    'NYK': 'New York Knicks', 'ORL': 'Orlando Magic', 'IND': 'Indiana Pacers',
    'PHI': 'Philadelphia 76ers', 'PHX': 'Phoenix Suns', 'POR': 'Portland Trail Blazers',
    'SAC': 'Sacramento Kings', 'SAS': 'San Antonio Spurs', 'OKC': 'Oklahoma City Thunder',
    'TOR': 'Toronto Raptors', 'UTA': 'Utah Jazz', 'MEM': 'Memphis Grizzlies',
    'WAS': 'Washington Wizards', 'DET': 'Detroit Pistons', 'CHA': 'Charlotte Hornets'
}



class EVAnalyzer:
    def __init__(self):
        load_dotenv()
        url = os.getenv("SUPABASE_URL")
        key = os.getenv("SUPABASE_KEY")
        self.supabase: Client | None = create_client(url, key) if url and key else None
        self._odds_df: pd.DataFrame | None = None
        self._player_cache: dict[str, pd.DataFrame] = {}
        self._team_cache: dict[str, pd.DataFrame] = {}
        self._defense_cache: dict[str, dict] = {}   # team → {pts_allowed, reb_allowed, ast_allowed}
        self._injured: set[str] = set()
        self._reject_reasons: Counter[str] = Counter()
        self._near_misses: list[dict] = []
        self.is_fallback = False
        self.fallback_date = None
        self._coverage_summary = "Cobertura no calculada."
        self._learning_scores: dict = {}
        self.load_learning_scores()

    def load_learning_scores(self):
        path = "learning_scores.json"
        if os.path.exists(path):
            try:
                with open(path, "r", encoding="utf-8") as f:
                    self._learning_scores = json.load(f).get("player_market_scores", {})
                logger.info("  -> Puntos de aprendizaje (IA) cargados.")
            except Exception as e:
                logger.error("Error cargando learning_scores.json: %s", e)
        else:
            self._learning_scores = {}

    def run_pipeline(self):
        return self.calculate_ev()

    # ── Supabase ──────────────────────────────────────────────────────────────

    def get_todays_odds(self) -> list[dict]:
        if not self.supabase:
            logger.error("Faltan SUPABASE_URL o SUPABASE_KEY.")
            return []
        today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
        logger.info("Obteniendo cuotas activas desde Supabase...")
        try:
            response = (
                self.supabase.table("player_odds")
                .select("*")
                .gte("game_date", today)
                .execute()
            )
            data = response.data or []
            logger.info("  -> %s lineas encontradas para hoy (%s).", len(data), today)
            
            if not data:
                logger.warning("No hay cuotas para hoy. Buscando la fecha más reciente con datos...")
                recent_res = (
                    self.supabase.table("player_odds")
                    .select("game_date")
                    .order("game_date", desc=True)
                    .limit(1)
                    .execute()
                )
                recent_rows = recent_res.data or []
                if recent_rows:
                    recent_date = recent_rows[0].get("game_date")
                    if recent_date != today:
                        self.is_fallback = True
                        self.fallback_date = recent_date
                    logger.info("  -> Usando fallback de cuotas para la fecha reciente: %s", recent_date)
                    fallback_res = (
                        self.supabase.table("player_odds")
                        .select("*")
                        .eq("game_date", recent_date)
                        .execute()
                    )
                    data = fallback_res.data or []
                    logger.info("  -> %s lineas encontradas para %s.", len(data), recent_date)
                else:
                    self.log_odds_table_status()
            return data
        except Exception as exc:
            logger.warning("Filtro por fecha falló (%s). Leyendo tabla completa.", exc)
            try:
                response = self.supabase.table("player_odds").select("*").execute()
                data = response.data or []
                logger.info("  -> %s lineas encontradas.", len(data))
                if not data:
                    self.log_odds_table_status()
                return data
            except Exception as exc2:
                logger.error("Error consultando Supabase: %s", exc2)
                return []


    def log_odds_table_status(self):
        if not self.supabase:
            return
        try:
            response = (
                self.supabase.table("player_odds")
                .select("game_date,market,bookmaker")
                .order("game_date", desc=True)
                .limit(20)
                .execute()
            )
            rows = response.data or []
            if not rows:
                logger.error("player_odds esta vacia en Supabase. odds.py no guardo filas.")
                return
            dates = sorted({str(row.get("game_date")) for row in rows}, reverse=True)
            markets = sorted({str(row.get("market")) for row in rows})
            logger.error(
                "player_odds tiene datos, pero no para hoy. Fechas recientes: %s | mercados: %s",
                ", ".join(dates[:5]),
                ", ".join(markets[:8]),
            )
        except Exception as exc:
            logger.error("No pude diagnosticar player_odds: %s", exc)

    def preload_injury_report(self):
        logger.info("Pre-cargando injury report NBA...")
        try:
            url = "https://cdn.nba.com/static/json/liveData/injuryreport/injuryreport.json"
            r = requests.get(url, headers={"User-Agent": "Mozilla/5.0"}, timeout=10)
            r.raise_for_status()
            players = r.json().get("injuryReport", {}).get("InjuredPlayers", [])
            for player in players:
                status = (player.get("Status") or "").lower()
                name = player.get("PlayerName")
                if name and status in {"out", "doubtful"}:
                    self._injured.add(name)
            logger.info("  -> %s jugadores Out/Doubtful.", len(self._injured))
        except Exception as exc:
            logger.warning("Injury report no disponible: %s", exc)

    def _parse_defense_rows(self, rows: list[dict]):
        for row in rows:
            team = row.get("team_name", "")
            if team:
                self._defense_cache[team] = {
                    "pts_allowed":  float(row.get("pts_allowed", 112.0)),
                    "reb_allowed":  float(row.get("reb_allowed", 43.5)),
                    "ast_allowed":  float(row.get("ast_allowed", 25.0)),
                    "fg3m_allowed": float(row.get("fg3m_allowed", 12.5)),
                    "blk_allowed":  float(row.get("blk_allowed", 5.0)),
                    "stl_allowed":  float(row.get("stl_allowed", 7.5)),
                }

    def preload_defense_stats(self):
        """Carga estadísticas defensivas por equipo desde Supabase (si existe la tabla) o fallback local."""
        loaded = False
        if self.supabase:
            try:
                response = self.supabase.table("team_defense_stats").select("*").execute()
                rows = response.data or []
                if rows:
                    self._parse_defense_rows(rows)
                    logger.info("  -> Estadisticas defensivas cargadas desde Supabase.")
                    loaded = True
            except Exception as exc:
                logger.info("team_defense_stats en Supabase no disponible (%s). Probando local.", exc)
        
        if not loaded:
            local_path = "team_defense_stats.json"
            if os.path.exists(local_path):
                try:
                    with open(local_path, "r", encoding="utf-8") as f:
                        rows = json.load(f)
                    if rows:
                        self._parse_defense_rows(rows)
                        logger.info("  -> Estadisticas defensivas cargadas desde fallback local.")
                        loaded = True
                except Exception as exc:
                    logger.warning("Error leyendo fallback local de defensas: %s", exc)
            else:
                logger.info("Sin archivo local de fallback defensas.")

        if not loaded:
            logger.info("Ajuste por oponente desactivado (sin datos).")


    def get_player_gamelog(self, player_name: str) -> pd.DataFrame | None:
        if player_name in self._player_cache:
            return self._player_cache[player_name]
        if not self.supabase:
            return None
        try:
            response = (
                self.supabase.table("player_stats")
                .select("*")
                .eq("player_name", player_name)
                .order("game_date", desc=True)
                .execute()
            )
            df = pd.DataFrame(response.data or [])
        except Exception as exc:
            logger.error("Error player_stats %s: %s", player_name, exc)
            return None
        if df.empty or len(df) < MIN_GAMES:
            return None
        df = self._normalize_player_df(df)
        if "min" in df.columns and df["min"].mean() < MIN_MINUTES:
            logger.info("    -> %s promedia %.1f min. Ignorado.", player_name, df["min"].mean())
            return None
        self._player_cache[player_name] = df
        return df

    def get_team_gamelog(self, team_name: str) -> pd.DataFrame | None:
        if team_name in self._team_cache:
            return self._team_cache[team_name]
        if not self.supabase:
            return None
        try:
            response = (
                self.supabase.table("team_game_logs")
                .select("*")
                .eq("team_name", team_name)
                .order("game_date", desc=True)
                .execute()
            )
            df = pd.DataFrame(response.data or [])
        except Exception as exc:
            logger.error("Error team_game_logs %s: %s", team_name, exc)
            return None
        if df.empty or len(df) < MIN_GAMES:
            return None
        self._team_cache[team_name] = df
        return df

    # ── Normalización ──────────────────────────────────────────────────────────

    @staticmethod
    def _normalize_player_df(df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        for col in df.columns:
            lower = col.lower()
            if lower != col and lower not in df.columns:
                df[lower] = df[col]
        numeric_cols = ["pts", "reb", "ast", "fg3m", "stl", "blk", "tov", "min"]
        for col in numeric_cols:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0)
        if {"pts", "reb", "ast"}.issubset(df.columns):
            df["pra"] = df["pts"] + df["reb"] + df["ast"]
        if "game_date" in df.columns:
            df["game_date"] = pd.to_datetime(df["game_date"], errors="coerce")
            df = df.sort_values("game_date", ascending=False)
        return df.reset_index(drop=True)

    # ── Matemática central ────────────────────────────────────────────────────

    @staticmethod
    def remove_vig(over_odds: float | None, under_odds: float | None) -> tuple[float, float]:
        over_odds = float(over_odds or 0)
        under_odds = float(under_odds or 0)
        if over_odds > 1 and under_odds > 1:
            p_over = 1 / over_odds
            p_under = 1 / under_odds
            total = p_over + p_under
            return p_over / total, p_under / total
        vig = 0.06
        return (
            (1 / over_odds) * (1 - vig) if over_odds > 1 else 0,
            (1 / under_odds) * (1 - vig) if under_odds > 1 else 0,
        )

    @staticmethod
    def kelly_criterion(prob: float, odds: float) -> float:
        if odds <= 1 or prob <= 0:
            return 0.0
        b = odds - 1
        raw = (b * prob - (1 - prob)) / b
        return max(0.0, raw * KELLY_FRACTION)

    @staticmethod
    def _grade(score: float) -> str:
        if score >= 78:
            return "S"
        if score >= 68:
            return "A"
        if score >= 60:
            return "B"
        return "C"

    @staticmethod
    def _stat_column(df: pd.DataFrame, market: str) -> str | None:
        for col in MARKET_STATS.get(market, []):
            if col in df.columns:
                return col
        return None

    @staticmethod
    def _is_b2b(df: pd.DataFrame) -> bool:
        if "game_date" not in df.columns or df["game_date"].isna().all():
            return False
        last_game = df["game_date"].max()
        if pd.isna(last_game):
            return False
        now = pd.Timestamp.now(tz="UTC").tz_localize(None)
        if getattr(last_game, "tzinfo", None) is not None:
            last_game = last_game.tz_localize(None)
        return (now - last_game).days <= 1

    def _opponent_adjustment(self, opponent_team: str, stat_col: str) -> float:
        """
        Ajusta la proyección según la fortaleza defensiva del oponente.
        Si el rival defiende mal el stat en cuestión → subir proyección.
        Si defiende bien → bajar proyección.
        Retorna el factor multiplicador (1.0 = neutro).
        """
        if not self._defense_cache or not opponent_team:
            return 1.0

        try:
            d = None
            for team, stats in self._defense_cache.items():
                if opponent_team.lower() in team.lower() or team.lower() in opponent_team.lower():
                    d = stats
                    break
            
            if not d:
                return 1.0

            league_avgs = {
                "pts":  112.0, "PTS": 112.0,
                "reb":  43.5,  "REB": 43.5,
                "ast":  25.0,  "AST": 25.0,
                "fg3m": 12.5,  "FG3M": 12.5,
                "blk":  5.0,   "BLK": 5.0,
                "stl":  7.5,   "STL": 7.5,
                "pra":  145.0,
            }
            stat_key_map = {
                "pts": "pts_allowed",  "PTS": "pts_allowed",
                "reb": "reb_allowed",  "REB": "reb_allowed",
                "ast": "ast_allowed",  "AST": "ast_allowed",
                "fg3m": "fg3m_allowed","FG3M": "fg3m_allowed",
                "blk": "blk_allowed",  "BLK": "blk_allowed",
                "stl": "stl_allowed",  "STL": "stl_allowed",
                "pra": "pts_allowed",
            }
            league_avg = league_avgs.get(stat_col, 0)
            defense_key = stat_key_map.get(stat_col)
            if not defense_key or league_avg == 0:
                return 1.0

            allowed = d[defense_key]
            factor = allowed / league_avg
            # Limitar el ajuste a ±10% para no distorsionar demasiado
            factor = max(0.90, min(1.10, factor))
            return factor
        except Exception as exc:
            logger.warning("Error en opponent adjustment: %s", exc)
            return 1.0

    def projected_stat(self, df: pd.DataFrame, stat_col: str, opponent_team: str = "") -> dict:
        values = pd.to_numeric(df[stat_col], errors="coerce").dropna().astype(float)
        if len(values) < MIN_GAMES:
            return {}

        l10 = values.head(10)
        l5  = values.head(5)
        l3  = values.head(3)
        season_mean = values.mean()
        recent_mean = (season_mean * 0.40) + (l10.mean() * 0.35) + (l5.mean() * 0.25)
        median = values.median()
        trend  = l5.mean() - values.tail(min(10, len(values))).mean()

        # Shrink toward median to avoid chasing outlier streaks
        proj = (recent_mean * 0.72) + (median * 0.28)
        if abs(season_mean) > 0.01:
            trend_factor = max(0.93, min(1.07, 1 + (trend / max(abs(season_mean), 1)) * 0.07))
            proj *= trend_factor
        if self._is_b2b(df):
            proj *= 0.96

        # Ajuste por oponente
        opp_factor = self._opponent_adjustment(opponent_team, stat_col)
        proj *= opp_factor


        std = max(values.std(ddof=1), math.sqrt(max(proj, 0.5)) * 0.65, 0.75)
        consistency = float((values > values.median()).mean())

        # Coeficiente de variación: mide qué tan errático es el jugador
        cv = std / max(proj, 0.1)

        # Hit rate empírico en últimos 10 juegos (calculado por línea en el caller)
        return {
            "mean":         float(proj),
            "std":          float(std),
            "median":       float(median),
            "season_mean":  float(season_mean),
            "l10":          float(l10.mean()),
            "l5":           float(l5.mean()),
            "l3":           float(l3.mean()),
            "consistency":  consistency,
            "games":        int(len(values)),
            "cv":           float(cv),
            "opp_factor":   float(opp_factor),
            "values":       values,   # para hit rate empírico
        }

    @staticmethod
    def poisson_cdf(k: int, lam: float) -> float:
        if lam <= 0:
            return 1.0
        term = math.exp(-lam)
        total = term
        for i in range(1, max(k, 0) + 1):
            term *= lam / i
            total += term
        return max(0.0, min(1.0, total))

    @classmethod
    def ensemble_prob(
        cls,
        values: pd.Series,
        projection: dict,
        line: float,
        market: str,
    ) -> tuple[float, float]:
        """
        Ensemble de 5 modelos:
          - Normal distribution (25%)
          - Empírica histórica (30%)
          - Monte Carlo (20%)
          - Poisson para stats de conteo (15%)
          - Quantile bootstrap (10%)
        """
        mean = projection["mean"]
        std  = projection["std"]
        dist = NormalDist(mu=mean, sigma=std)
        normal_under = dist.cdf(line)
        normal_over  = 1 - normal_under

        clean = pd.to_numeric(values, errors="coerce").dropna().astype(float)
        empirical_over  = float((clean > line).mean())
        empirical_under = float((clean < line).mean())

        rng = np.random.default_rng(seed=42)
        samples = rng.normal(loc=mean, scale=std, size=12000)
        mc_over  = float((samples > line).mean())
        mc_under = float((samples < line).mean())

        # Modelo 5: Bootstrap quantile
        if len(clean) >= 10:
            boot_means = np.array([
                rng.choice(clean.values, size=len(clean), replace=True).mean()
                for _ in range(3000)
            ])
            q_over  = float((boot_means > line).mean())
            q_under = float((boot_means < line).mean())
        else:
            q_over  = normal_over
            q_under = normal_under

        if market in COUNT_MARKETS:
            # Para stats discretas, Poisson es más preciso
            poisson_under = cls.poisson_cdf(math.floor(line), max(mean, 0.01))
            poisson_over  = 1 - poisson_under
            prob_over  = (normal_over  * 0.20 + empirical_over  * 0.30
                          + mc_over  * 0.15 + poisson_over  * 0.25 + q_over  * 0.10)
            prob_under = (normal_under * 0.20 + empirical_under * 0.30
                          + mc_under * 0.15 + poisson_under * 0.25 + q_under * 0.10)
        else:
            prob_over  = (normal_over  * 0.25 + empirical_over  * 0.30
                          + mc_over  * 0.20 + q_over  * 0.10
                          + normal_over  * 0.15)   # normal tiene doble peso en continuo
            prob_under = (normal_under * 0.25 + empirical_under * 0.30
                          + mc_under * 0.20 + q_under * 0.10
                          + normal_under * 0.15)

        return max(0.0, min(0.995, prob_over)), max(0.0, min(0.995, prob_under))

    @staticmethod
    def _empirical_hit_rate(values: pd.Series, line: float, side: str, n: int = 10) -> float:
        """Hit rate real del jugador sobre/bajo esta línea en sus últimos n juegos."""
        recent = pd.to_numeric(values, errors="coerce").dropna().head(n)
        if recent.empty:
            return 0.5
        if side == "OVER":
            return float((recent > line).mean())
        return float((recent < line).mean())

    @staticmethod
    def _volatility_penalty(cv: float) -> float:
        """
        Penalización al score por inconsistencia del jugador.
        CV > 0.45 → penalización gradual hasta -15 puntos al score.
        """
        if cv <= MAX_CV_NO_PENALTY:
            return 0.0
        excess = cv - MAX_CV_NO_PENALTY
        return -min(15.0, excess * 35.0)

    @staticmethod
    def _book_consensus_bonus(line_group: pd.DataFrame, side: str, line: float) -> tuple[float, str]:
        """
        Mide si la mayoría de bookies tiene sus odds orientadas al mismo lado.
        Retorna (bonus_score, descripcion).
        """
        if line_group.empty or len(line_group) < MIN_CONSENSUS_BOOKS:
            return 0.0, ""

        books = line_group[
            ~line_group["bookmaker"].str.lower().str.contains(PINNACLE_KEY, na=False)
        ]
        if len(books) < 2:
            return 0.0, ""

        over_odds  = pd.to_numeric(books["over_odds"], errors="coerce").dropna()
        under_odds = pd.to_numeric(books["under_odds"], errors="coerce").dropna()
        if over_odds.empty or under_odds.empty:
            return 0.0, ""

        avg_over  = over_odds.mean()
        avg_under = under_odds.mean()

        # Si el mercado pone la cuota del lado que apostamos más baja → consenso en contra
        # Si la cuota de nuestro lado es más alta → consenso a favor
        if side == "OVER":
            if avg_over < avg_under and avg_over < 1.85:
                return -3.0, f"Consenso_contra_OVER({len(books)} libros)"
            if avg_over > avg_under:
                return 2.5, f"Consenso_OVER({len(books)} libros)"
        else:  # UNDER
            if avg_under < avg_over and avg_under < 1.85:
                return -3.0, f"Consenso_contra_UNDER({len(books)} libros)"
            if avg_under > avg_over:
                return 2.5, f"Consenso_UNDER({len(books)} libros)"
        return 0.0, ""

    def _sharp_signal(self, row: pd.Series, side: str, line_group: pd.DataFrame) -> tuple[float, str]:
        pin = line_group[line_group["bookmaker"].str.lower().str.contains(PINNACLE_KEY, na=False)]
        if pin.empty:
            return 0.0, ""
        pin_row = pin.iloc[0]
        p_over, p_under = self.remove_vig(pin_row.get("over_odds"), pin_row.get("under_odds"))
        odds = float(row.get("over_odds") if side == "OVER" else row.get("under_odds") or 0)
        pin_prob = p_over if side == "OVER" else p_under
        if odds <= 1 or pin_prob <= 0:
            return 0.0, ""
        market_edge = (pin_prob * odds) - 1
        if market_edge >= 0.015:
            return min(6.0, market_edge * 100), f"Sharp:+{market_edge*100:.1f}%"
        if market_edge <= -0.04:
            return -8.0, f"Sharp:{market_edge*100:.1f}%"
        return 0.0, ""

    def _track_near_miss(
        self,
        row: pd.Series,
        side: str,
        prob: float,
        implied: float,
        edge: float,
        stake: float,
        score: float,
        reasons: list[str],
        projection: dict,
    ):
        for reason in reasons:
            self._reject_reasons[reason] += 1
        if edge < 0:
            return
        line = float(row.get("line") or 0)
        self._near_misses.append(
            {
                "player/team": row.get("player_name"),
                "market":      row.get("market", ""),
                "bet":         f"{side} {line:g}",
                "odds":        float(row.get("over_odds") if side == "OVER" else row.get("under_odds") or 0),
                "bookie":      row.get("bookmaker"),
                "prob_real":   f"{prob*100:.1f}%",
                "prob_novig":  f"{implied*100:.1f}%",
                "edge":        f"{edge*100:+.1f}%",
                "kelly_bet":   f"${stake:.2f}",
                "score":       round(score, 1),
                "flags":       "WATCHLIST",
                "info": (
                    f"No oficial: {','.join(reasons)} | "
                    f"Proj:{projection['mean']:.1f} Med:{projection['median']:.1f} "
                    f"L10:{projection['l10']:.1f} L5:{projection['l5']:.1f} "
                    f"CV:{projection['cv']:.2f}"
                ),
                "matchup":    row.get("matchup", "?"),
                "game_time":  row.get("game_time", "?"),
                "_prob_raw":  prob,
                "_score_raw": score,
                "_edge_raw":  edge,
            }
        )

    def _build_bet(
        self,
        row: pd.Series,
        side: str,
        prob: float,
        implied: float,
        projection: dict,
        sharp_bonus: float,
        sharp_text: str,
        hit_rate: float,
        vol_penalty: float,
        consensus_bonus: float,
        consensus_text: str,
        line_group: pd.DataFrame,
    ) -> dict | None:
        odds = float(row.get("over_odds") if side == "OVER" else row.get("under_odds") or 0)
        if odds <= 1:
            self._reject_reasons["bad_odds"] += 1
            return None

        # Filtro de cuota (evita picks demasiado cortos tipo 1.20 o demasiado largos / alternativos tipo 3.50)
        if odds < MIN_ODDS:
            self._reject_reasons["odds_too_short"] += 1
            return None
        if odds > MAX_ODDS:
            self._reject_reasons["odds_too_long"] += 1
            return None

        market = row.get("market", "")

        # Threshold de probabilidad adaptivo por mercado
        mkt_prob_threshold = MIN_PROB * MARKET_PROB_MULT.get(market, 1.0)

        # EV real = (prob * odds) - 1
        edge_raw = (prob * odds) - 1
        # Cap de EV: si el edge supera MAX_CREDIBLE_EV, es casi siempre un artefacto del modelo.
        # Reducimos la probabilidad efectiva para calcular stake pero mantenemos el pick si pasa filtros.
        if edge_raw > MAX_CREDIBLE_EV:
            credible_prob = (1 + MAX_CREDIBLE_EV) / odds
            effective_prob = min(prob, credible_prob)
        else:
            effective_prob = prob

        edge        = (effective_prob * odds) - 1
        prob_gap    = effective_prob - implied
        kelly       = self.kelly_criterion(effective_prob, odds)
        stake       = min(kelly * BANKROLL, BANKROLL * MAX_PLAYER_STAKE_PCT)

        # ── Reinforcement Learning: Ajuste por Puntos de Aprendizaje ───────────────
        player_name = row.get("player_name")
        learning_key = f"{player_name} || {market}"
        learning_points = 0
        learning_adjustment = 0.0
        ai_adjustment = 0.0
        
        if learning_key in self._learning_scores:
            entry = self._learning_scores[learning_key]
            learning_points = entry.get("points", 0)
            if learning_points > 0:
                learning_adjustment = min(10.0, learning_points * 0.2)
            elif learning_points < 0:
                learning_adjustment = max(-15.0, learning_points * 0.3)
            
            ai_adjustment = entry.get("ai_adjustment", 0.0)

        # Score compuesto con todos los ajustes
        score = (
            min(edge / 0.15, 1.0) * 32          # Peso del edge
            + min(prob_gap / 0.20, 1.0) * 24    # Peso del gap vs mercado
            + min(effective_prob / 0.85, 1.0) * 20  # Peso de la probabilidad
            + min(hit_rate / 0.70, 1.0) * 12    # Peso del hit rate empírico
            + min(stake / max(BANKROLL * 0.02, 1), 1.0) * 6
            + sharp_bonus                        # Señal sharp (±6)
            + vol_penalty                        # Penalización volatilidad (0 a -15)
            + consensus_bonus                    # Book consensus (±3)
            + learning_adjustment                # Ajuste de reputación por Puntos (±15)
            + ai_adjustment                      # Ajuste inteligente de la IA (hasta -15)
        )

        reasons = []
        if prob < mkt_prob_threshold:
            reasons.append("prob")
        if edge < MIN_EDGE:
            reasons.append("ev")
        if stake < MIN_KELLY:
            reasons.append("kelly")
        if score < MIN_SCORE:
            reasons.append("score")

        # Filtro de hit rate empírico
        min_hr = MIN_OVER_HIT_RATE if side == "OVER" else MIN_UNDER_HIT_RATE
        if hit_rate < min_hr:
            reasons.append(f"hit_rate_{hit_rate*100:.0f}%")

        # Filtros adicionales de ultra precisión si es LOCK_MODE (Apuestas Fijas)
        if LOCK_MODE:
            l5 = projection.get("l5", 0)
            l3 = projection.get("l3", 0)
            median = projection.get("median", 0)
            line = float(row.get("line") or 0)
            
            # 1. Consistencia de Tendencia
            if side == "OVER":
                if l5 <= line:
                    reasons.append(f"trend_L5_{l5:.1f}<=line")
                if l3 <= line:
                    reasons.append(f"trend_L3_{l3:.1f}<=line")
                if median <= line:
                    reasons.append(f"median_{median:.1f}<=line")
            elif side == "UNDER":
                if l5 >= line:
                    reasons.append(f"trend_L5_{l5:.1f}>=line")
                if l3 >= line:
                    reasons.append(f"trend_L3_{l3:.1f}>=line")
                if median >= line:
                    reasons.append(f"median_{median:.1f}>=line")
            
            # 2. Desacuerdo de Pinnacle (Sharp disagree filter)
            pin = line_group[line_group["bookmaker"].str.lower().str.contains(PINNACLE_KEY, na=False)]
            if not pin.empty:
                pin_row = pin.iloc[0]
                p_over, p_under = self.remove_vig(pin_row.get("over_odds"), pin_row.get("under_odds"))
                pin_prob = p_over if side == "OVER" else p_under
                market_edge = (pin_prob * odds) - 1
                if market_edge < 0.0:
                    reasons.append("sharp_disagree")

        if reasons:
            self._track_near_miss(row, side, prob, implied, edge, stake, score, reasons, projection)
            return None


        cv = projection.get("cv", 0)
        line = float(row.get("line") or 0)
        opp_info = f" OppAdj:{projection.get('opp_factor', 1.0):.2f}" if projection.get("opp_factor", 1.0) != 1.0 else ""
        
        ia_text = ""
        if learning_points != 0 or ai_adjustment != 0.0:
            adj_sign = "+" if (learning_adjustment + ai_adjustment) >= 0 else ""
            ia_text = f"🧠 IA:{learning_points:g}pts/AI:{ai_adjustment:.1f}({adj_sign}{learning_adjustment + ai_adjustment:.1f})"

        info = (
            f"Proj:{projection['mean']:.1f} Med:{projection['median']:.1f} "
            f"L10:{projection['l10']:.1f} L5:{projection['l5']:.1f} L3:{projection.get('l3', 0):.1f} "
            f"Std:{projection['std']:.1f} CV:{cv:.2f} HR:{hit_rate*100:.0f}% G:{projection['games']}"
            f"{opp_info}"
        )
        extras = [t for t in [sharp_text, consensus_text, ia_text] if t]
        if extras:
            info = f"{info} | {' | '.join(extras)}"

        flags = "MATH+EV"
        if sharp_text and sharp_bonus > 0:
            flags += " SHARP"
        if consensus_bonus > 0:
            flags += " CONSENSUS"

        return {
            "grade":       self._grade(score),
            "player/team": row.get("player_name"),
            "market":      market,
            "bet":         f"{side} {line:g}",
            "odds":        odds,
            "bookie":      row.get("bookmaker"),
            "prob_real":   f"{prob*100:.1f}%",
            "prob_novig":  f"{implied*100:.1f}%",
            "edge":        f"+{edge*100:.1f}%",
            "kelly_bet":   f"${stake:.2f}",
            "score":       round(score, 1),
            "hit_rate":    f"{hit_rate*100:.0f}%",
            "flags":       flags,
            "info":        info,
            "matchup":     row.get("matchup", "?"),
            "game_time":   row.get("game_time", "?"),
            "_prob_raw":   prob,
            "_score_raw":  score,
            "_stake_raw":  stake,
            "_edge_raw":   edge,
        }

    def _moneyline_bets(self, odds_df: pd.DataFrame) -> list[dict]:
        picks = []
        teams = odds_df[odds_df["market"] == "h2h"]
        for team, group in teams.groupby("player_name"):
            df = self.get_team_gamelog(team)
            if df is None or "wl" not in df.columns:
                continue
            wins = (df["wl"].str.upper() == "W").astype(float)
            if len(wins) < MIN_GAMES:
                continue
            proj = wins.mean() * 0.45 + wins.head(10).mean() * 0.35 + wins.head(5).mean() * 0.20
            for _, row in group.iterrows():
                if PINNACLE_KEY in str(row.get("bookmaker", "")).lower():
                    continue
                odds = float(row.get("over_odds") or 0)
                if odds <= 1 or odds < MIN_ODDS:
                    continue
                implied, _ = self.remove_vig(odds, row.get("under_odds"))
                edge = (proj * odds) - 1
                if proj < MIN_PROB or edge < MIN_EDGE:
                    continue
                stake = min(self.kelly_criterion(proj, odds) * BANKROLL, BANKROLL * 0.02)
                if stake < MIN_KELLY:
                    continue
                score = min(100, edge / 0.18 * 45 + proj / 0.70 * 35 + stake / (BANKROLL * 0.02) * 20)
                if score < MIN_SCORE:
                    continue
                picks.append(
                    {
                        "grade":       self._grade(score),
                        "player/team": team,
                        "market":      "h2h",
                        "bet":         "WIN",
                        "odds":        odds,
                        "bookie":      row.get("bookmaker"),
                        "prob_real":   f"{proj*100:.1f}%",
                        "prob_novig":  f"{implied*100:.1f}%",
                        "edge":        f"+{edge*100:.1f}%",
                        "kelly_bet":   f"${stake:.2f}",
                        "score":       round(score, 1),
                        "hit_rate":    "N/A",
                        "flags":       "TEAM+EV",
                        "info": (
                            f"Win% S:{wins.mean()*100:.0f} "
                            f"L10:{wins.head(10).mean()*100:.0f} "
                            f"L5:{wins.head(5).mean()*100:.0f}"
                        ),
                        "matchup":     row.get("matchup", "?"),
                        "game_time":   row.get("game_time", "?"),
                        "_prob_raw":   proj,
                        "_score_raw":  score,
                        "_stake_raw":  stake,
                        "_edge_raw":   edge,
                    }
                )
        return picks

    def _market_confidence_bets(self, odds_df: pd.DataFrame) -> list[dict]:
        picks = []
        h2h = odds_df[
            (odds_df["market"] == "h2h")
            & ~odds_df["bookmaker"].str.lower().str.contains(PINNACLE_KEY, na=False)
        ]
        if h2h.empty:
            return picks
        for (_, bookie), group in h2h.groupby(["matchup", "bookmaker"], dropna=False):
            group = group.copy()
            group["over_odds"] = pd.to_numeric(group["over_odds"], errors="coerce")
            group = group[group["over_odds"] > 1]
            if len(group) < 2:
                continue
            implied_raw = 1 / group["over_odds"]
            total_raw = implied_raw.sum()
            if total_raw <= 0:
                continue
            group["_market_prob"] = implied_raw / total_raw
            favorite = group.sort_values(
                ["_market_prob", "over_odds"], ascending=[False, True]
            ).iloc[0]
            prob = float(favorite["_market_prob"])
            odds = float(favorite["over_odds"])
            if prob < MIN_MARKET_CONFIDENCE_PROB or odds > MAX_MARKET_CONFIDENCE_ODDS or odds < MIN_ODDS:
                continue
            stake = min(BANKROLL * 0.01, BANKROLL * MAX_PLAYER_STAKE_PCT)
            score = min(100.0, prob * 100)
            if score < MIN_SCORE:
                continue
            picks.append(
                {
                    "grade":       self._grade(score),
                    "player/team": favorite.get("player_name"),
                    "market":      "h2h",
                    "bet":         "WIN",
                    "odds":        odds,
                    "bookie":      bookie,
                    "prob_real":   f"{prob*100:.1f}%",
                    "prob_novig":  f"{prob*100:.1f}%",
                    "edge":        "+0.0%",
                    "kelly_bet":   f"${stake:.2f}",
                    "score":       round(score, 1),
                    "hit_rate":    "N/A",
                    "flags":       "MARKET-HIGH-PROB",
                    "info":        "Favorito fuerte por consenso de cuotas; sin edge estadistico interno",
                    "matchup":     favorite.get("matchup", "?"),
                    "game_time":   favorite.get("game_time", "?"),
                    "_prob_raw":   prob,
                    "_score_raw":  score,
                    "_stake_raw":  stake,
                    "_edge_raw":   0.0,
                }
            )
        return picks

    def calculate_ev(self):
        self._reject_reasons.clear()
        self._near_misses.clear()
        self.preload_injury_report()
        self.preload_defense_stats()

        odds_data = self.get_todays_odds()
        if not odds_data:
            msg = "No hay cuotas. Ejecuta odds.py primero."
            logger.warning(msg)
            self._send_telegram_msg(f"<b>Bot EV Elite</b>\n\n{msg}")
            return []

        odds_df = pd.DataFrame(odds_data)
        odds_df["bookmaker"] = odds_df["bookmaker"].fillna("").astype(str)
        odds_df["line"]      = pd.to_numeric(odds_df["line"], errors="coerce").fillna(0)
        odds_df["over_odds"] = pd.to_numeric(odds_df["over_odds"], errors="coerce")
        odds_df["under_odds"]= pd.to_numeric(odds_df["under_odds"], errors="coerce")
        raw_odds_df = odds_df.copy()

        if ALLOWED_BOOKIES:
            allowed = odds_df["bookmaker"].str.lower().apply(
                lambda b: any(book in b for book in ALLOWED_BOOKIES) or PINNACLE_KEY in b
            )
            odds_df = odds_df[allowed].copy()
            has_bettable = (
                ~odds_df["bookmaker"].str.lower().str.contains(PINNACLE_KEY, na=False)
            ).any()
            if not has_bettable:
                logger.warning(
                    "ALLOWED_BOOKIES no coincide con casas apostables. "
                    "Usando fallback: todas las casas no-Pinnacle disponibles."
                )
                odds_df = raw_odds_df.copy()

        self._odds_df = odds_df
        self._coverage_summary = self._odds_coverage_summary(odds_df)
        logger.info("Cobertura de cuotas: %s", self._coverage_summary)
        logger.info("Analizando %s lineas con modelo matematico +EV v3...", len(odds_df))

        picks: list[dict] = []
        picks.extend(self._moneyline_bets(odds_df))
        picks.extend(self._market_confidence_bets(odds_df))

        prop_df = odds_df[odds_df["market"].isin(MARKET_STATS)]
        unsupported = len(odds_df) - len(prop_df) - len(odds_df[odds_df["market"] == "h2h"])
        if unsupported > 0:
            self._reject_reasons["unsupported_market"] += int(unsupported)

        line_groups = {
            key: group
            for key, group in prop_df.groupby(
                ["matchup", "player_name", "market", "line"], dropna=False
            )
        }
        bettable_prop_df = prop_df[
            ~prop_df["bookmaker"].str.lower().str.contains(PINNACLE_KEY, na=False)
        ]

        for player, player_lines in bettable_prop_df.groupby("player_name"):
            if player in self._injured:
                logger.info("  [LESION] %s Out/Doubtful. Ignorado.", player)
                self._reject_reasons["injury"] += len(player_lines)
                continue
            stats = self.get_player_gamelog(player)
            if stats is None:
                self._reject_reasons["missing_stats"] += len(player_lines)
                continue

            # Obtener el nombre del equipo del jugador desde su gamelog
            player_team_name = None
            if not stats.empty and "opponent" in stats.columns:
                sample_opp = str(stats["opponent"].iloc[0])
                tokens = sample_opp.replace(".", "").split()
                if tokens:
                    player_abbr = tokens[0]  # El primer token es el equipo del jugador (p. ej. "DET")
                    player_team_name = ABBR_TO_TEAM_MAP.get(player_abbr)

            for _, row in player_lines.iterrows():
                market  = row.get("market")
                stat_col = self._stat_column(stats, market)
                if not stat_col:
                    self._reject_reasons["missing_stat_column"] += 1
                    continue

                matchup = row.get("matchup", "")
                opponent_team = ""
                if matchup and player_team_name:
                    core_matchup = matchup
                    if core_matchup.startswith("NBA "):
                        core_matchup = core_matchup[4:]
                    teams_in_match = [t.strip() for t in core_matchup.split(" vs ")]
                    if len(teams_in_match) == 2:
                        if player_team_name.lower() in teams_in_match[0].lower():
                            opponent_team = teams_in_match[1]
                        elif player_team_name.lower() in teams_in_match[1].lower():
                            opponent_team = teams_in_match[0]

                projection = self.projected_stat(stats, stat_col, opponent_team)

                if not projection:
                    self._reject_reasons["thin_sample"] += 1
                    continue

                line = float(row.get("line") or 0)
                if line <= 0:
                    self._reject_reasons["bad_line"] += 1
                    continue

                # Filtrar líneas extremas vs proyección
                ratio = line / max(projection["mean"], 0.1)
                if side_line_is_too_extreme(ratio):
                    logger.info(
                        "    -> %s %s linea %.1f extrema vs proj %.1f. Ignorada.",
                        player, market, line, projection["mean"],
                    )
                    self._reject_reasons["extreme_line"] += 1
                    continue

                values = projection.pop("values")  # extraer antes de pasar projection
                prob_over, prob_under = self.ensemble_prob(values, projection, line, market)
                implied_over, implied_under = self.remove_vig(row.get("over_odds"), row.get("under_odds"))

                key = (row.get("matchup"), row.get("player_name"), row.get("market"), row.get("line"))
                line_group = line_groups.get(key, pd.DataFrame())

                # Calcular todos los ajustes de score
                cv = projection.get("cv", 0)
                vol_penalty = self._volatility_penalty(cv)

                # ── OVER ──────────────────────────────────────────────────────
                sharp_bonus_o, sharp_text_o = self._sharp_signal(row, "OVER", line_group)
                cons_bonus_o, cons_text_o   = self._book_consensus_bonus(line_group, "OVER", line)
                hit_rate_over = self._empirical_hit_rate(values, line, "OVER")
                bet_over = self._build_bet(
                    row, "OVER", prob_over, implied_over, projection,
                    sharp_bonus_o, sharp_text_o,
                    hit_rate_over, vol_penalty, cons_bonus_o, cons_text_o,
                    line_group,
                )
                if bet_over:
                    picks.append(bet_over)

                # ── UNDER ─────────────────────────────────────────────────────
                sharp_bonus_u, sharp_text_u = self._sharp_signal(row, "UNDER", line_group)
                cons_bonus_u, cons_text_u   = self._book_consensus_bonus(line_group, "UNDER", line)
                hit_rate_under = self._empirical_hit_rate(values, line, "UNDER")
                bet_under = self._build_bet(
                    row, "UNDER", prob_under, implied_under, projection,
                    sharp_bonus_u, sharp_text_u,
                    hit_rate_under, vol_penalty, cons_bonus_u, cons_text_u,
                    line_group,
                )
                if bet_under:
                    picks.append(bet_under)

                # Restaurar values al projection dict (por si se reutiliza)
                projection["values"] = values

        final = self._post_process(picks)
        try:
            from track_results import save_todays_picks
            save_todays_picks(self.supabase, final)
        except Exception as exc:
            logger.warning("No se pudieron guardar los picks de hoy en bet_history: %s", exc)
        self._report(final)
        return final


    def _post_process(self, picks: list[dict]) -> list[dict]:
        if not picks:
            return []
        df = pd.DataFrame(picks)
        df = df.sort_values(["_score_raw", "_prob_raw", "_edge_raw", "odds"], ascending=False)
        df = df.drop_duplicates(subset=["player/team", "market", "bet"], keep="first")
        df = df.groupby(["player/team", "market"], group_keys=False).head(1)
        df = df.groupby("player/team", group_keys=False).head(MAX_PICKS_PER_PLAYER)
        df = df.groupby("matchup", group_keys=False).head(MAX_PICKS_PER_GAME)

        capped = []
        for _, group in df.groupby("matchup", dropna=False):
            max_game = BANKROLL * MAX_GAME_STAKE_PCT
            total = group["_stake_raw"].sum()
            scale = min(1.0, max_game / total) if total > 0 else 1.0
            for item in group.to_dict("records"):
                stake = item["_stake_raw"] * scale
                item["_stake_raw"] = stake
                item["kelly_bet"]  = f"${stake:.2f}"
                capped.append(item)

        capped_df = pd.DataFrame(capped)
        capped_df = capped_df[capped_df["_score_raw"] >= MIN_SCORE]
        capped_df = capped_df.sort_values(
            ["_score_raw", "_prob_raw", "_edge_raw"], ascending=False
        ).head(MAX_PICKS_TOTAL)
        return capped_df.to_dict("records")

    @staticmethod
    def _odds_coverage_summary(odds_df: pd.DataFrame) -> str:
        if odds_df.empty or "matchup" not in odds_df.columns:
            return "0 partidos, 0 lineas."
        matchups = odds_df["matchup"].fillna("?").astype(str)
        sports   = matchups.str.split(" ", n=1).str[0].replace({"?": "UNKNOWN"})
        sport_counts = sports.value_counts().to_dict()
        sport_text   = ", ".join(f"{sport}:{count}" for sport, count in sport_counts.items())
        return f"{matchups.nunique()} partidos, {len(odds_df)} lineas ({sport_text})"

    def _report(self, picks: list[dict]):
        if self.is_fallback:
            msg = (
                f"⚠️ <b>Alerta Bot EV Elite</b>\n\n"
                f"No se encontraron cuotas nuevas para hoy. Los créditos de la API podrían estar agotados, "
                f"o no hay partidos hoy.\n"
                f"<i>(Se omitió el envío de picks antiguos de la fecha {self.fallback_date} para evitar errores).</i>"
            )
            logger.warning(msg)
            self._send_telegram_msg(msg)
            return

        if not picks:
            msg = "No se encontraron apuestas +EV con los filtros matematicos de hoy."
            logger.info(msg)
            logger.info("Diagnostico filtros: %s", self.diagnostics_summary())
            watchlist = self._watchlist(limit=5)
            if watchlist:
                watch_df = pd.DataFrame(watchlist).fillna("")
                print("\n" + "=" * 110)
                print("=== RADAR DE LEANS CERCANOS - NO SON PICKS OFICIALES ===")
                print(watch_df.to_string(index=False))
                print(f"\nDiagnostico filtros: {self.diagnostics_summary()}")
            if watchlist:
                self._send_telegram_msg(
                    f"<b>Bot EV Elite</b>\n\n{msg}\n\n"
                    f"<b>Radar de leans cercanos</b>\n{format_watchlist_html(watchlist)}"
                )
            else:
                self._send_telegram_msg(f"<b>Bot EV Elite</b>\n\n{msg}")
            return

        df = pd.DataFrame(picks)
        clean = df.drop(
            columns=["_score_raw", "_stake_raw", "_edge_raw", "_prob_raw"], errors="ignore"
        )
        clean = clean.reset_index(drop=True).fillna("")
        print("\n" + "=" * 110)
        print("=== BOT EV ELITE v3 - MODELO MATEMATICO +EV DEL DIA ===")
        print(clean.to_string())
        print(f"\nTotal apuestas unicas: {len(clean)}")
        print(f"Diagnostico filtros: {self.diagnostics_summary()}")
        self.send_telegram_alert(clean.to_dict("records"))

    def _watchlist(self, limit: int = 5) -> list[dict]:
        if not self._near_misses:
            return []
        df = pd.DataFrame(self._near_misses)
        df = df.sort_values(["_score_raw", "_edge_raw"], ascending=False)
        df = df.drop_duplicates(subset=["player/team", "market", "bet"], keep="first")
        return (
            df.head(limit)
            .drop(columns=["_score_raw", "_edge_raw", "_prob_raw"], errors="ignore")
            .to_dict("records")
        )

    def diagnostics_summary(self) -> str:
        if not self._reject_reasons:
            return "Sin rechazos registrados."
        parts = [f"{reason}:{count}" for reason, count in self._reject_reasons.most_common()]
        return " | ".join(parts)

    def _send_telegram_msg(self, text: str):
        if not SEND_TELEGRAM:
            logger.info("SEND_TELEGRAM=0; mensaje Telegram omitido.")
            return
        token   = os.getenv("TELEGRAM_BOT_TOKEN")
        chat_id = os.getenv("TELEGRAM_CHAT_ID")
        if not token or not chat_id:
            return
        try:
            requests.post(
                f"https://api.telegram.org/bot{token}/sendMessage",
                json={"chat_id": chat_id, "text": text, "parse_mode": "HTML"},
                timeout=10,
            )
        except Exception as exc:
            logger.error("Telegram error: %s", exc)

    def send_telegram_alert(self, ev_bets: list[dict]):
        if not SEND_TELEGRAM:
            logger.info("SEND_TELEGRAM=0; alerta Telegram omitida.")
            return
        token   = os.getenv("TELEGRAM_BOT_TOKEN")
        chat_id = os.getenv("TELEGRAM_CHAT_ID")
        if not token or not chat_id:
            return

        url = f"https://api.telegram.org/bot{token}/sendMessage"
        msg = (
            "<b>🏀 BOT EV ELITE v3 — APUESTAS MATEMÁTICAS DEL DÍA</b>\n"
            f"<i>Filtros: Prob>={MIN_PROB*100:.0f}% | EV>={MIN_EDGE*100:.0f}% | "
            f"HR>={MIN_OVER_HIT_RATE*100:.0f}% | MaxCV={MAX_CV_NO_PENALTY} | "
            f"Max {MAX_PICKS_TOTAL} picks</i>\n"
            f"<i>Analizado: {self._coverage_summary}</i>\n"
            f"───────────────────────────\n\n"
        )

        for idx, bet in enumerate(ev_bets, 1):
            market_es = MARKET_TRANSLATIONS.get(bet["market"], bet["market"])
            name = bet["player/team"]
            if bet["market"] == "h2h":
                instruction = f"<b>{name}</b> gana el partido"
            else:
                side, line_val = bet["bet"].split(" ", 1)
                side_es = "MÁS DE" if side == "OVER" else "MENOS DE"
                instruction = f"<b>{name}</b> {side_es} <b>{line_val}</b> {market_es}"

            grade = bet["grade"]
            grade_emoji = "🔥" if grade == "S" else ("⭐" if grade == "A" else "⚡")
            hit_rate_str = bet.get("hit_rate", "N/A")
            
            info_str = bet.get("info", "")

            text = (
                f"{grade_emoji} <b>{idx}. [{grade}] {bet['flags']}</b>\n"
                f"🏀 <b>Partido:</b> {bet.get('matchup', '?')} ({bet.get('game_time', '?')})\n"
                f"🎯 <b>Pick:</b> {instruction}\n"
                f"💰 <b>Cuota:</b> <code>{bet['odds']}</code> en {display_bookie(bet['bookie'])} | Stake: <code>{bet['kelly_bet']}</code>\n"
                f"📊 <b>Métricas:</b> Prob: <code>{bet['prob_real']}</code> | Consenso: <code>{bet['prob_novig']}</code> | EV: <code>{bet['edge']}</code>\n"
                f"📈 <b>Hit Rate L10:</b> <code>{hit_rate_str}</code>\n"
                f"ℹ️ <i>{info_str}</i>\n"
                f"───────────────────────────\n\n"
            )

            if len(msg) + len(text) > 3900:
                requests.post(
                    url,
                    json={"chat_id": chat_id, "text": msg, "parse_mode": "HTML"},
                    timeout=10,
                )
                msg = text
            else:
                msg += text

        if msg.strip():
            try:
                requests.post(
                    url,
                    json={"chat_id": chat_id, "text": msg, "parse_mode": "HTML"},
                    timeout=10,
                )
                logger.info("Telegram enviado: %s apuestas.", len(ev_bets))
            except Exception as exc:
                logger.error("Telegram error: %s", exc)


# ── Helpers ───────────────────────────────────────────────────────────────────

def side_line_is_too_extreme(ratio: float) -> bool:
    return ratio < 0.50 or ratio > 1.80


def display_bookie(bookie: str) -> str:
    raw = (bookie or "").lower()
    if any(b in raw for b in ["unibet", "betrivers", "888sport"]):
        return "BETPLAY"
    if any(b in raw for b in ["bet365", "williamhill", "betsson", "betfair"]):
        return "WPLAY"
    return (bookie or "?").upper()


def format_watchlist_html(watchlist: list[dict]) -> str:
    lines = []
    for idx, bet in enumerate(watchlist, 1):
        market_es = MARKET_TRANSLATIONS.get(bet["market"], bet["market"])
        parts = bet["bet"].split(" ", 1)
        if len(parts) == 2:
            side, line_val = parts
            side_es = "MÁS DE" if side == "OVER" else "MENOS DE"
            instruction = f"<b>{bet['player/team']}</b> {side_es} {line_val} {market_es}"
        else:
            instruction = f"<b>{bet['player/team']}</b> {bet['bet']}"
            
        lines.append(
            f"👀 <b>{idx}. {instruction}</b>\n"
            f"  └ Cuota: <code>{bet['odds']}</code> | EV: <code>{bet['edge']}</code> | HR: <code>{bet.get('hit_rate', '?')}</code> | Score: <code>{bet['score']}</code>"
        )
    return "\n\n".join(lines)


if __name__ == "__main__":
    analyzer = EVAnalyzer()
    analyzer.calculate_ev()
