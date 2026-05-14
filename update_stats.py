"""
update_stats.py — Script de actualización de estadísticas (corre LOCAL en tu PC).
Ejecuta este script una vez por semana (o antes de la temporada de playoffs)
para mantener Supabase actualizado con los gamelogs más recientes de cada jugador y equipo.

Flujo:
  1. Lee player_odds de Supabase para saber qué jugadores/equipos necesita
  2. Descarga el gamelog completo de la temporada de cada jugador → player_stats
  3. Descarga el gamelog de W/L de cada equipo → team_game_logs
"""

import os
import time
import logging
import pandas as pd
from dotenv import load_dotenv
from supabase import create_client, Client

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger("Stats_Updater")

SEASON = '2025-26'
MIN_MINUTES = 18.0  # Ignorar jugadores de garbage time


def main():
    load_dotenv()
    supabase: Client = create_client(os.getenv("SUPABASE_URL"), os.getenv("SUPABASE_KEY"))

    # ── 1. Obtener jugadores y equipos desde player_odds ──────────────────────
    logger.info("Leyendo entidades activas desde player_odds en Supabase...")
    response = supabase.table("player_odds").select("player_name, market").execute()
    odds_data = response.data

    players = list(set(o["player_name"] for o in odds_data if o["market"] != "h2h"))
    teams   = list(set(o["player_name"] for o in odds_data if o["market"] == "h2h"))

    logger.info(f"  → {len(players)} jugadores y {len(teams)} equipos para actualizar.")

    # ── 2. Gamelogs de jugadores → player_stats ───────────────────────────────
    from nba_api.stats.static import players as nba_players_static
    from nba_api.stats.endpoints import playergamelog

    total_player_rows = 0
    for player_name in players:
        logger.info(f"  [JUGADOR] {player_name}")
        nba_players = nba_players_static.find_players_by_full_name(player_name)
        if not nba_players:
            logger.warning(f"    → No encontrado en NBA API. Saltando.")
            continue

        player_id = nba_players[0]['id']
        try:
            time.sleep(0.7)
            gamelog = playergamelog.PlayerGameLog(player_id=player_id, season=SEASON)
            df = gamelog.get_data_frames()[0]

            if df.empty:
                continue

            # Filtrar minutos mínimos
            df['MIN'] = pd.to_numeric(df['MIN'], errors='coerce')
            avg_min = df['MIN'].mean()
            if avg_min < MIN_MINUTES:
                logger.info(f"    → {avg_min:.1f} min promedio — ignorado.")
                continue

            # Preparar registros para Supabase
            records = []
            for _, row in df.iterrows():
                records.append({
                    "player_id":   int(player_id),
                    "player_name": player_name,
                    "game_date":   str(row.get("GAME_DATE", ""))[:10],
                    "opponent":    str(row.get("MATCHUP", "")),
                    "pts":         int(row.get("PTS", 0)),
                    "reb":         int(row.get("REB", 0)),
                    "ast":         int(row.get("AST", 0)),
                    "min":         float(row.get("MIN", 0)),
                    "league":      "NBA"
                })

            if records:
                supabase.table("player_stats").upsert(
                    records, on_conflict="player_id,game_date"
                ).execute()
                total_player_rows += len(records)
                logger.info(f"    → {len(records)} partidos guardados en Supabase.")

        except Exception as e:
            logger.error(f"    → Error: {e}")

    logger.info(f"Jugadores: {total_player_rows} filas totales guardadas en player_stats.")

    # ── 3. Gamelogs de equipos → team_game_logs ───────────────────────────────
    from nba_api.stats.static import teams as nba_teams_static
    from nba_api.stats.endpoints import teamgamelog

    total_team_rows = 0
    for team_name in teams:
        logger.info(f"  [EQUIPO] {team_name}")
        nba_teams = nba_teams_static.find_teams_by_full_name(team_name)
        if not nba_teams:
            logger.warning(f"    → No encontrado. Saltando.")
            continue

        team_id = nba_teams[0]['id']
        try:
            time.sleep(0.7)
            tgl = teamgamelog.TeamGameLog(team_id=team_id, season=SEASON)
            df = tgl.get_data_frames()[0]

            if df.empty:
                continue

            records = []
            for _, row in df.iterrows():
                records.append({
                    "team_name": team_name,
                    "game_date": str(row.get("GAME_DATE", ""))[:10],
                    "wl":        str(row.get("WL", ""))
                })

            if records:
                supabase.table("team_game_logs").upsert(
                    records, on_conflict="team_name,game_date"
                ).execute()
                total_team_rows += len(records)
                logger.info(f"    → {len(records)} partidos guardados en Supabase.")

        except Exception as e:
            logger.error(f"    → Error: {e}")

    logger.info(f"Equipos: {total_team_rows} filas totales guardadas en team_game_logs.")
    logger.info("✅ update_stats.py completado. Supabase actualizado.")


if __name__ == "__main__":
    main()
