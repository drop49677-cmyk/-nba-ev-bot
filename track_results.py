"""
track_results.py — Verifica resultados del dia anterior y calcula ROI acumulado.
Corre automaticamente al inicio del pipeline (run_bot.py lo llama antes de odds.py).

Flujo:
  1. Lee picks de ayer desde Supabase (tabla bet_history)
  2. Consulta resultados reales via NBA API
  3. Actualiza outcome en bet_history
  4. Imprime resumen de ROI acumulado y hit rate por mercado
"""

import logging
import os
import sys
import time
from datetime import datetime, timedelta, timezone

import pandas as pd
import requests
from dotenv import load_dotenv

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger("ResultTracker")

SEASON = os.getenv("NBA_SEASON", "2025-26")


def get_supabase():
    from supabase import create_client
    url = os.getenv("SUPABASE_URL")
    key = os.getenv("SUPABASE_KEY")
    if not url or not key:
        return None
    return create_client(url, key)


def ensure_bet_history_exists(supabase) -> bool:
    """Verifica si existe la tabla bet_history. Si no, la crea via SQL RPC (si está disponible)."""
    try:
        supabase.table("bet_history").select("id").limit(1).execute()
        return True
    except Exception as exc:
        logger.warning("Tabla bet_history no existe o no es accesible: %s", exc)
        logger.info(
            "Crea la tabla manualmente en Supabase con este SQL:\n"
            "CREATE TABLE bet_history (\n"
            "  id BIGSERIAL PRIMARY KEY,\n"
            "  pick_date DATE NOT NULL,\n"
            "  player_name TEXT,\n"
            "  market TEXT,\n"
            "  bet TEXT,\n"
            "  line FLOAT,\n"
            "  odds FLOAT,\n"
            "  bookie TEXT,\n"
            "  stake FLOAT,\n"
            "  prob_real FLOAT,\n"
            "  edge FLOAT,\n"
            "  score FLOAT,\n"
            "  hit_rate FLOAT,\n"
            "  matchup TEXT,\n"
            "  outcome TEXT DEFAULT 'pending',\n"
            "  actual_value FLOAT,\n"
            "  profit FLOAT,\n"
            "  created_at TIMESTAMPTZ DEFAULT NOW()\n"
            ");"
        )
        return False


def save_todays_picks(supabase, picks: list[dict]) -> int:
    """Guarda los picks de hoy en bet_history para trackear mañana."""
    if not picks or not supabase:
        return 0
    if not ensure_bet_history_exists(supabase):
        return 0

    today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    records = []
    for pick in picks:
        try:
            # Extraer line del string "OVER 18.5" o "UNDER 10.5"
            bet_str = pick.get("bet", "")
            parts = bet_str.split(" ")
            line = float(parts[1]) if len(parts) >= 2 else 0.0

            edge_str = pick.get("edge", "0%").replace("+", "").replace("%", "")
            prob_str = pick.get("prob_real", "0%").replace("%", "")
            hr_str = pick.get("hit_rate", "0%").replace("%", "")
            stake_str = pick.get("kelly_bet", "$0").replace("$", "")

            records.append({
                "pick_date":   today,
                "player_name": pick.get("player/team", ""),
                "market":      pick.get("market", ""),
                "bet":         bet_str,
                "line":        line,
                "odds":        float(pick.get("odds", 0)),
                "bookie":      pick.get("bookie", ""),
                "stake":       float(stake_str) if stake_str else 0.0,
                "prob_real":   float(prob_str) / 100 if prob_str else 0.0,
                "edge":        float(edge_str) / 100 if edge_str else 0.0,
                "score":       float(pick.get("score", 0)),
                "hit_rate":    float(hr_str) / 100 if hr_str else 0.0,
                "matchup":     pick.get("matchup", ""),
                "outcome":     "pending",
            })
        except Exception as exc:
            logger.warning("No pude parsear pick para bet_history: %s — %s", pick, exc)

    if records:
        try:
            supabase.table("bet_history").insert(records).execute()
            logger.info("  -> %s picks guardados en bet_history.", len(records))
            return len(records)
        except Exception as exc:
            logger.error("Error guardando picks en bet_history: %s", exc)
    return 0


def get_nba_boxscore(player_name: str, game_date: str) -> float | None:
    """
    Intenta obtener el valor real de una estadística vía NBA CDN o stats.nba.com.
    Retorna None si no encuentra el dato.
    """
    try:
        # Primero intentar via nba_api si está instalada
        from nba_api.stats.static import players as nba_players_static
        from nba_api.stats.endpoints import playergamelog

        nba_players = nba_players_static.find_players_by_full_name(player_name)
        if not nba_players:
            return None
        player_id = nba_players[0]["id"]
        time.sleep(0.7)
        gl = playergamelog.PlayerGameLog(player_id=player_id, season=SEASON)
        df = gl.get_data_frames()[0]
        if df.empty:
            return None
        # Convertir GAME_DATE al formato yyyy-mm-dd
        df["GAME_DATE_FMT"] = pd.to_datetime(
            df["GAME_DATE"], format="%b %d, %Y", errors="coerce"
        ).dt.strftime("%Y-%m-%d")
        match = df[df["GAME_DATE_FMT"] == game_date]
        if match.empty:
            return None
        return match.iloc[0].to_dict()
    except ImportError:
        logger.info("nba_api no disponible — no se puede verificar resultado automaticamente.")
        return None
    except Exception as exc:
        logger.warning("Error consultando gamelog de %s: %s", player_name, exc)
        return None


def _stat_for_market(row_data: dict, market: str) -> float | None:
    """Extrae el valor estadístico relevante del boxscore según el mercado."""
    market_col = {
        "player_points":    "PTS",
        "player_rebounds":  "REB",
        "player_assists":   "AST",
        "player_threes":    "FG3M",
        "player_blocks":    "BLK",
        "player_steals":    "STL",
        "player_turnovers": "TOV",
        "player_points_rebounds_assists": None,
    }
    if market == "player_points_rebounds_assists":
        pts = row_data.get("PTS", 0) or 0
        reb = row_data.get("REB", 0) or 0
        ast = row_data.get("AST", 0) or 0
        return float(pts) + float(reb) + float(ast)
    col = market_col.get(market)
    if col and col in row_data:
        return float(row_data[col])
    return None


def verify_yesterday_picks(supabase) -> dict:
    """
    Verifica los picks de ayer con resultados reales.
    Retorna stats de rendimiento.
    """
    if not supabase:
        return {}
    if not ensure_bet_history_exists(supabase):
        return {}

    yesterday = (datetime.now(timezone.utc) - timedelta(days=1)).strftime("%Y-%m-%d")
    logger.info("Verificando picks del %s...", yesterday)

    try:
        response = (
            supabase.table("bet_history")
            .select("*")
            .eq("pick_date", yesterday)
            .eq("outcome", "pending")
            .execute()
        )
        pending = response.data or []
    except Exception as exc:
        logger.error("Error leyendo bet_history: %s", exc)
        return {}

    if not pending:
        logger.info("  -> Sin picks pendientes de ayer.")
        return _roi_summary(supabase)

    logger.info("  -> %s picks pendientes de verificar.", len(pending))
    wins = losses = voids = 0

    for pick in pending:
        player  = pick.get("player_name", "")
        market  = pick.get("market", "")
        bet_str = pick.get("bet", "")
        line    = float(pick.get("line") or 0)
        odds    = float(pick.get("odds") or 1)
        stake   = float(pick.get("stake") or 0)

        parts = bet_str.split(" ")
        side  = parts[0] if parts else ""

        boxscore = get_nba_boxscore(player, yesterday)
        if boxscore is None:
            logger.info("    -> %s: sin boxscore disponible. Quedara como pending.", player)
            continue

        actual = _stat_for_market(boxscore, market)
        if actual is None:
            logger.info("    -> %s: stat no encontrada para %s.", player, market)
            continue

        won = (side == "OVER" and actual > line) or (side == "UNDER" and actual < line)
        if side not in {"OVER", "UNDER"}:
            # moneyline
            wl = str(boxscore.get("WL", "")).upper()
            won = (wl == "W")

        outcome = "win" if won else "loss"
        profit  = (stake * (odds - 1)) if won else -stake

        try:
            supabase.table("bet_history").update({
                "outcome":      outcome,
                "actual_value": float(actual),
                "profit":       round(float(profit), 2),
            }).eq("id", pick["id"]).execute()

            result_emoji = "✅" if won else "❌"
            logger.info(
                "    %s %s %s %.1f | Real: %.1f | %s | P&L: %+.2f",
                result_emoji, player, side, line, actual, outcome.upper(), profit,
            )
            if won:
                wins += 1
            else:
                losses += 1
        except Exception as exc:
            logger.error("    -> Error actualizando %s: %s", player, exc)

    logger.info(
        "Ayer: %s/%-2s ganados | %s perdidos | %s sin verificar.",
        wins, wins + losses, losses, voids,
    )
    return _roi_summary(supabase)


def _roi_summary(supabase) -> dict:
    """Calcula y muestra el resumen de ROI acumulado."""
    try:
        response = supabase.table("bet_history").select(
            "market,outcome,stake,profit,odds,prob_real"
        ).neq("outcome", "pending").neq("outcome", "void").execute()
        rows = response.data or []
    except Exception as exc:
        logger.error("Error leyendo historial: %s", exc)
        return {}

    if not rows:
        logger.info("Sin historial completo aun.")
        return {}

    df = pd.DataFrame(rows)
    df["profit"] = pd.to_numeric(df["profit"], errors="coerce").fillna(0)
    df["stake"]  = pd.to_numeric(df["stake"], errors="coerce").fillna(0)

    total_bets    = len(df)
    total_staked  = df["stake"].sum()
    total_profit  = df["profit"].sum()
    roi_pct       = (total_profit / total_staked * 100) if total_staked > 0 else 0
    wins          = (df["outcome"] == "win").sum()
    hit_rate      = wins / total_bets if total_bets > 0 else 0

    print("\n" + "=" * 70)
    print("=== RENDIMIENTO ACUMULADO DEL BOT ===")
    print(f"  Total picks : {total_bets}")
    print(f"  Ganados     : {wins} ({hit_rate*100:.1f}%)")
    print(f"  Total staked: ${total_staked:.2f}")
    print(f"  P&L total   : ${total_profit:+.2f}")
    print(f"  ROI         : {roi_pct:+.1f}%")

    # Por mercado
    if "market" in df.columns:
        by_market = df.groupby("market").agg(
            picks=("outcome", "count"),
            wins=("outcome", lambda x: (x == "win").sum()),
            profit=("profit", "sum"),
            staked=("stake", "sum"),
        )
        by_market["hit%"]  = (by_market["wins"] / by_market["picks"] * 100).round(1)
        by_market["roi%"]  = (by_market["profit"] / by_market["staked"] * 100).round(1)
        print("\nRendimiento por mercado:")
        print(by_market[["picks", "wins", "hit%", "profit", "roi%"]].to_string())
    print("=" * 70 + "\n")

    return {
        "total_bets":   total_bets,
        "hit_rate":     hit_rate,
        "total_profit": total_profit,
        "roi_pct":      roi_pct,
    }


def send_roi_telegram(stats: dict):
    """Envía el resumen de ROI a Telegram si hay datos."""
    if not stats:
        return
    token   = os.getenv("TELEGRAM_BOT_TOKEN")
    chat_id = os.getenv("TELEGRAM_CHAT_ID")
    if not token or not chat_id:
        return

    total = stats.get("total_bets", 0)
    if total == 0:
        return

    hit  = stats.get("hit_rate", 0)
    roi  = stats.get("roi_pct", 0)
    pnl  = stats.get("total_profit", 0)
    emoji = "📈" if pnl >= 0 else "📉"

    msg = (
        f"{emoji} <b>Rendimiento Acumulado Bot EV Elite</b>\n\n"
        f"<b>Total picks:</b> {total}\n"
        f"<b>Hit rate:</b> {hit*100:.1f}%\n"
        f"<b>P&L:</b> ${pnl:+.2f}\n"
        f"<b>ROI:</b> {roi:+.1f}%\n"
    )
    try:
        requests.post(
            f"https://api.telegram.org/bot{token}/sendMessage",
            json={"chat_id": chat_id, "text": msg, "parse_mode": "HTML"},
            timeout=10,
        )
    except Exception as exc:
        logger.error("Telegram ROI error: %s", exc)


if __name__ == "__main__":
    sb = get_supabase()
    if sb:
        stats = verify_yesterday_picks(sb)
        send_roi_telegram(stats)
    else:
        logger.error("No se pudo conectar a Supabase.")
