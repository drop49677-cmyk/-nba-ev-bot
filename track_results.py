"""
track_results.py — Verifica resultados del dia anterior y calcula ROI acumulado.
Corre automaticamente al inicio del pipeline (run_bot.py lo llama antes de odds.py).

Flujo:
  1. Lee picks de ayer desde Supabase (tabla bet_history)
  2. Consulta resultados reales via NBA API
  3. Actualiza outcome en bet_history
  4. Imprime resumen de ROI acumulado y hit rate por mercado
"""

import json
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


# ── Helpers de Aprendizaje por Refuerzo y IA ──────────────────────────────────

def load_learning_scores() -> dict:
    path = "learning_scores.json"
    if os.path.exists(path):
        try:
            with open(path, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception as e:
            logger.error("Error cargando learning_scores.json: %s", e)
    return {"player_market_scores": {}}


def save_learning_scores(scores: dict):
    path = "learning_scores.json"
    try:
        with open(path, "w", encoding="utf-8") as f:
            json.dump(scores, f, indent=2, ensure_ascii=False)
        logger.info("learning_scores.json guardado con éxito.")
    except Exception as e:
        logger.error("Error guardando learning_scores.json: %s", e)


def update_learning_scores(player: str, market: str, won: bool, ai_adjustment: float = 0.0, ai_notes: str = "") -> tuple[int, int, float]:
    """Actualiza los puntos de aprendizaje para un jugador y mercado. Retorna (puntos_nuevos, cambio, ai_adjustment_total)."""
    scores = load_learning_scores()
    key = f"{player} || {market}"
    if "player_market_scores" not in scores:
        scores["player_market_scores"] = {}
        
    db = scores["player_market_scores"]
    if key not in db:
        db[key] = {"points": 0, "wins": 0, "losses": 0, "ai_adjustment": 0.0, "ai_notes": ""}
        
    change = 10 if won else -10
    db[key]["points"] += change
    if won:
        db[key]["wins"] += 1
        # Reducir paulatinamente el castigo de la IA si el jugador se recupera
        db[key]["ai_adjustment"] = min(0.0, db[key].get("ai_adjustment", 0.0) + 2.0)
    else:
        db[key]["losses"] += 1
        # Acumular el ajuste de la IA (hasta un tope de -15.0)
        db[key]["ai_adjustment"] = max(-15.0, db[key].get("ai_adjustment", 0.0) + ai_adjustment)
        if ai_notes:
            db[key]["ai_notes"] = ai_notes
            
    db[key]["last_updated"] = datetime.now().strftime("%Y-%m-%d")
    
    save_learning_scores(scores)
    return db[key]["points"], change, db[key]["ai_adjustment"]


def display_bookie(bookie: str) -> str:
    raw = (bookie or "").lower()
    if any(b in raw for b in ["unibet", "betrivers", "888sport"]):
        return "BETPLAY"
    if any(b in raw for b in ["bet365", "williamhill", "betsson", "betfair"]):
        return "WPLAY"
    return (bookie or "?").upper()


MARKET_TRANSLATIONS = {
    "player_points":                  "Puntos",
    "player_rebounds":                "Rebotes",
    "player_assists":                 "Asistencias",
    "player_points_rebounds_assists": "Puntos+Rebotes+Asistencias",
    "player_threes":                  "Triples",
    "player_steals":                  "Robos",
    "player_blocks":                  "Bloqueos",
    "player_turnovers":               "Pérdidas",
    "h2h":                            "Ganador del Partido",
}


def analyze_loss_with_gemini(player: str, market: str, bet: str, actual: float, line: float, odds: float, boxscore: dict) -> tuple[float, str]:
    """Usa la API de Gemini por debajo para evaluar si el fallo fue por varianza o sistemático y calibrar."""
    api_key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
    if not api_key:
        return 0.0, ""
    
    minutos = boxscore.get("MIN", "N/A") if isinstance(boxscore, dict) else "N/A"
    pts = boxscore.get("PTS", 0) if isinstance(boxscore, dict) else 0
    reb = boxscore.get("REB", 0) if isinstance(boxscore, dict) else 0
    ast = boxscore.get("AST", 0) if isinstance(boxscore, dict) else 0
    wl = boxscore.get("WL", "?") if isinstance(boxscore, dict) else "?"
    
    market_es = MARKET_TRANSLATIONS.get(market, market)
    
    prompt = (
        f"Analiza por qué falló el pick deportivo de la NBA:\n"
        f"- Jugador: {player}\n"
        f"- Mercado: {market_es}\n"
        f"- Pick: {bet} (Cuota: {odds})\n"
        f"- Resultado real: {actual} (Línea: {line})\n"
        f"- Minutos: {minutos}\n"
        f"- Stats: PTS {pts}, REB {reb}, AST {ast}, Resultado {wl}\n\n"
        f"Determina si el fallo fue por varianza/mala suerte (ej. Curry tiró 1/10 triples abiertos) "
        f"o si es un fallo sistemático del modelo (ej. cambio de rol del jugador, defensa rival muy dura, reducción de minutos consistente, lesión).\n"
        f"Devuelve obligatoriamente un objeto JSON con este formato exacto:\n"
        f"{{\n"
        f"  \"ai_adjustment\": <número entre -15.0 y 0.0, donde 0.0 es varianza/mala suerte y -15.0 es un grave error sistemático que exige penalizar mucho al jugador>,\n"
        f"  \"reason\": \"<breve explicación técnica en una frase de por qué aplicas este ajuste (evita comillas dobles dentro del texto, usa comillas simples si es necesario)>\"\n"
        f"}}\n"
        f"No devuelvas ningún texto antes ni después del JSON."
    )
    
    url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-2.5-flash:generateContent?key={api_key}"
    headers = {"Content-Type": "application/json"}
    payload = {
        "contents": [{
            "parts": [{"text": prompt}]
        }],
        "generationConfig": {
            "maxOutputTokens": 2000,
            "temperature": 0.2,
            "responseMimeType": "application/json"
        }
    }
    
    try:
        response = requests.post(url, json=payload, headers=headers, timeout=25)
        if response.status_code == 200:
            res_json = response.json()
            text = res_json["candidates"][0]["content"]["parts"][0]["text"].strip()
            data = json.loads(text)
            ai_adjustment = float(data.get("ai_adjustment", 0.0))
            reason = data.get("reason", "")
            return ai_adjustment, reason
        else:
            logger.warning("Error de API Gemini: %s - %s", response.status_code, response.text)
    except Exception as e:
        logger.error("Error al conectar con Gemini API o parsear JSON: %s", e)
    return 0.0, ""


def send_daily_summary_telegram(graded_results: list[dict], date_str: str):
    token   = os.getenv("TELEGRAM_BOT_TOKEN")
    chat_id = os.getenv("TELEGRAM_CHAT_ID")
    if not token or not chat_id:
        return

    url = f"https://api.telegram.org/bot{token}/sendMessage"

    wins = sum(1 for r in graded_results if r["outcome"] == "win")
    total = len(graded_results)
    hit_rate = (wins / total * 100) if total > 0 else 0
    total_staked = sum(r["stake"] for r in graded_results)
    total_profit = sum(r["profit"] for r in graded_results)

    pnl_emoji = "📈" if total_profit >= 0 else "📉"
    pnl_sign = "+" if total_profit >= 0 else ""
    roi = (total_profit / total_staked * 100) if total_staked > 0 else 0

    msg = (
        f"🏆 <b>BOT EV ELITE — REPORTE DIARIO DE RESULTADOS</b>\n"
        f"📅 <b>Fecha evaluada:</b> {date_str}\n"
        f"───────────────────────────\n\n"
    )

    for idx, r in enumerate(graded_results, 1):
        market_es = MARKET_TRANSLATIONS.get(r["market"], r["market"])
        bookie_es = display_bookie(r["bookie"])
        emoji = "✅" if r["outcome"] == "win" else "❌"
        profit_sign = "+" if r["profit"] >= 0 else ""
        
        # Traducir instrucción
        parts = r["bet"].split(" ", 1)
        if len(parts) == 2:
            side, line_val = parts
            side_es = "MÁS DE" if side == "OVER" else "MENOS DE"
            instruction = f"<b>{r['player']}</b> {side_es} <b>{line_val}</b> {market_es}"
        else:
            instruction = f"<b>{r['player']}</b> {r['bet']}"

        actual_str = f"{r['actual']:g}" if r["actual"] is not None else "N/A"
        line_str = f"{r['line']:g}" if r["line"] is not None else "N/A"

        # Formateo de reputación de aprendizaje (puntos estándar e IA de forma numérica simple)
        change_sign = "+" if r["learning_change"] > 0 else ""
        ai_adj_str = f" | IA: {r['ai_adjustment']:.1f}" if r["ai_adjustment"] != 0.0 else ""
        rep_str = f"🧠 <b>Reputación:</b> <code>{r['learning_points']} pts{ai_adj_str}</code> ({change_sign}{r['learning_change']})"

        msg += (
            f"{emoji} <b>{idx}. {instruction}</b>\n"
            f"  ├ <b>Real:</b> <code>{actual_str}</code> (Línea: {line_str})\n"
            f"  ├ <b>Cuota:</b> <code>{r['odds']}</code> en {bookie_es} | Stake: ${r['stake']:.2f}\n"
            f"  ├ <b>P&L:</b> <code>{profit_sign}${r['profit']:.2f}</code>\n"
            f"  └ {rep_str}\n\n"
        )

    msg += (
        f"───────────────────────────\n"
        f"📊 <b>RESUMEN DEL DÍA:</b>\n"
        f"🔹 <b>Aciertos:</b> <code>{wins}/{total} ({hit_rate:.1f}%)</code>\n"
        f"🔹 <b>Inversión:</b> <code>${total_staked:.2f}</code>\n"
        f"🔹 <b>P&L del Día:</b> <code>{pnl_sign}${total_profit:.2f}</code> {pnl_emoji}\n"
        f"🔹 <b>ROI diario:</b> <code>{roi:+.1f}%</code>\n"
        f"───────────────────────────\n"
        f"🧠 <i>El bot ha actualizado su memoria y aprendido de estos resultados.</i>"
    )

    try:
        requests.post(
            url,
            json={"chat_id": chat_id, "text": msg, "parse_mode": "HTML"},
            timeout=15,
        )
        logger.info("Reporte diario enviado a Telegram.")
    except Exception as exc:
        logger.error("Error enviando reporte diario a Telegram: %s", exc)


def verify_yesterday_picks(supabase) -> dict:
    """
    Verifica los picks de ayer con resultados reales.
    Retorna stats de rendimiento y envía reporte de resultados.
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
    wins = losses = 0
    graded_results = []

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
            logger.info("    -> %s: sin boxscore disponible. Quedará como pending.", player)
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

        # ── IA: Analizar pérdida con Gemini REST de forma silenciosa para obtener el ajuste y la nota ──
        ai_adjustment = 0.0
        gemini_reason = ""
        if not won:
            ai_adjustment, gemini_reason = analyze_loss_with_gemini(player, market, bet_str, actual, line, odds, boxscore)

        # ── Reinforcement Learning: Actualizar puntos de aprendizaje ──
        new_points, change, ai_adj_total = update_learning_scores(player, market, won, ai_adjustment, gemini_reason)

        try:
            supabase.table("bet_history").update({
                "outcome":      outcome,
                "actual_value": float(actual),
                "profit":       round(float(profit), 2),
            }).eq("id", pick["id"]).execute()

            result_emoji = "✅" if won else "❌"
            logger.info(
                "    %s %s %s %.1f | Real: %.1f | %s | P&L: %+.2f | IA Adj: %.1f (Motivo: %s) | Puntos: %s",
                result_emoji, player, side, line, actual, outcome.upper(), profit,
                ai_adjustment, gemini_reason or "N/A", new_points
            )
            
            graded_results.append({
                "player": player,
                "market": market,
                "bet": bet_str,
                "line": line,
                "actual": actual,
                "outcome": outcome,
                "odds": odds,
                "stake": stake,
                "profit": profit,
                "bookie": pick.get("bookie", ""),
                "learning_points": new_points,
                "learning_change": change,
                "ai_adjustment": ai_adj_total
            })

            if won:
                wins += 1
            else:
                losses += 1
        except Exception as exc:
            logger.error("    -> Error actualizando %s: %s", player, exc)

    logger.info(
        "Ayer: %s ganados | %s perdidos.",
        wins, losses,
    )

    if graded_results:
        send_daily_summary_telegram(graded_results, yesterday)

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
        logger.info("Sin historial completo aún.")
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
    pnl_sign = "+" if pnl >= 0 else ""

    msg = (
        f"📈 <b>ESTADÍSTICAS HISTÓRICAS ACUMULADAS</b>\n"
        f"<i>Rendimiento total del Bot EV Elite</i>\n"
        f"───────────────────────────\n"
        f"🔹 <b>Total picks evaluados:</b> {total}\n"
        f"🔹 <b>Hit Rate acumulado:</b> <code>{hit*100:.1f}%</code>\n"
        f"🔹 <b>P&L acumulado:</b> <code>{pnl_sign}${pnl:,.2f}</code>\n"
        f"🔹 <b>ROI global:</b> <code>{roi:+.1f}%</code>\n"
        f"───────────────────────────\n"
        f"🚀 <i>El bot sigue optimizando y aprendiendo de cada resultado.</i>"
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
