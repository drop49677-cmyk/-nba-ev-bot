"""
verify_nightly.py — Polling nocturno para calificar partidos en cuanto terminen.
Este script se ejecuta periódicamente (ej. cada 30 min) durante las horas de partidos.
Lógica:
  1. Busca si hay picks en bet_history con outcome 'pending'.
  2. Para cada fecha de picks pendientes, consulta el scoreboard de la NBA.
  3. Si TODOS los partidos de esa fecha están finalizados (Status = 3 o "Final"),
     corre la calificación (verify_yesterday_picks) e informa a Telegram de forma inmediata.
"""

import os
import sys
import logging
from datetime import datetime, timezone
from dotenv import load_dotenv

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger("NightlyVerifier")

load_dotenv()

# Asegurar encoding utf-8
try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass


def main():
    from track_results import get_supabase, verify_yesterday_picks, send_roi_telegram
    
    supabase = get_supabase()
    if not supabase:
        logger.error("No se pudo conectar a Supabase.")
        return

    # 1. Buscar picks pendientes
    logger.info("Buscando picks con estado 'pending' en Supabase...")
    try:
        response = (
            supabase.table("bet_history")
            .select("pick_date")
            .eq("outcome", "pending")
            .execute()
        )
        rows = response.data or []
    except Exception as exc:
        logger.error("Error consultando bet_history: %s", exc)
        return

    if not rows:
        logger.info("No hay picks pendientes por calificar.")
        return

    # Obtener fechas únicas ordenadas de los picks pendientes
    pending_dates = sorted(list({row["pick_date"] for row in rows}))
    logger.info("Fechas con picks pendientes de verificación: %s", pending_dates)

    from nba_api.stats.endpoints import scoreboardv3

    for date_str in pending_dates:
        logger.info("Consultando estado de partidos para la fecha: %s", date_str)
        try:
            sb = scoreboardv3.ScoreboardV3(game_date=date_str)
            games = sb.get_dict().get("scoreboard", {}).get("games", [])
        except Exception as exc:
            logger.error("Error al consultar scoreboard para la fecha %s: %s", date_str, exc)
            continue

        if not games:
            logger.warning("No se encontraron partidos en la API para la fecha %s.", date_str)
            continue

        # Verificar si todos los partidos están finalizados
        all_finished = True
        for g in games:
            status = g.get("gameStatus", 0)
            status_text = str(g.get("gameStatusText", "")).upper()
            home = g.get("homeTeam", {}).get("teamTricode", "")
            away = g.get("awayTeam", {}).get("teamTricode", "")
            
            # Status 3 = Final en la API de la NBA
            is_final = (status == 3) or ("FINAL" in status_text)
            if not is_final:
                all_finished = False
                logger.info("  -> Partido en juego o no iniciado: %s vs %s (Estado: %s, '%s')", 
                            home, away, status, status_text)
                break
            else:
                logger.info("  -> Partido finalizado: %s vs %s", home, away)

        if all_finished:
            logger.info("🎉 ¡Todos los partidos del %s han finalizado! Iniciando calificación...", date_str)
            stats = verify_yesterday_picks(supabase, target_date=date_str)
            if stats:
                send_roi_telegram(stats)
            logger.info("Calificación para la fecha %s completada.", date_str)
        else:
            logger.info("Aún hay partidos activos para la fecha %s. Se omitió la calificación.", date_str)


if __name__ == "__main__":
    main()
