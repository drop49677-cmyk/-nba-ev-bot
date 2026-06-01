"""
update_stats.py — Script de actualización de estadísticas (corre LOCAL en tu PC).
Ejecuta este script una vez por semana (o antes de la temporada de playoffs)
para mantener Supabase y los archivos locales actualizados con los gamelogs más recientes
de cada jugador y equipo, además de las estadísticas defensivas.

Flujo:
  1. Descarga estadísticas defensivas de la liga -> team_defense_stats.json y Supabase
  2. Lee player_odds de Supabase para saber qué jugadores/equipos necesita
  3. Descarga el gamelog completo de la temporada de cada jugador → player_stats
  4. Descarga el gamelog de W/L de cada equipo → team_game_logs
"""

import json
import logging
import os
import time
import pandas as pd
from dotenv import load_dotenv
from supabase import create_client, Client

load_dotenv()

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger("Stats_Updater")

SEASON = os.getenv("NBA_SEASON", "2025-26")
MIN_MINUTES = float(os.getenv("MIN_MINUTES", 18.0))  # Ignorar jugadores de garbage time

BASE_PLAYER_COLUMNS = {
    "player_id",
    "player_name",
    "game_date",
    "opponent",
    "pts",
    "reb",
    "ast",
    "min",
    "league",
}


def safe_int(row, key: str) -> int:
    try:
        value = row.get(key, 0)
        if pd.isna(value):
            return 0
        return int(value)
    except Exception:
        return 0


def safe_float(row, key: str) -> float:
    try:
        value = row.get(key, 0)
        if pd.isna(value):
            return 0.0
        return float(value)
    except Exception:
        return 0.0


def upsert_player_records(supabase: Client, records: list[dict]) -> None:
    if getattr(upsert_player_records, "base_only", False):
        compact_records = [
            {key: value for key, value in record.items() if key in BASE_PLAYER_COLUMNS}
            for record in records
        ]
        supabase.table("player_stats").upsert(
            compact_records, on_conflict="player_id,game_date"
        ).execute()
        return

    try:
        supabase.table("player_stats").upsert(
            records, on_conflict="player_id,game_date"
        ).execute()
        return
    except Exception as exc:
        logger.warning(
            "    -> Upsert extendido fallo (%s). Reintentando con columnas base.",
            exc,
        )
        setattr(upsert_player_records, "base_only", True)

    compact_records = [
        {key: value for key, value in record.items() if key in BASE_PLAYER_COLUMNS}
        for record in records
    ]
    supabase.table("player_stats").upsert(
        compact_records, on_conflict="player_id,game_date"
    ).execute()


def update_team_defense_stats(supabase: Client) -> None:
    """Descarga estadísticas defensivas de la NBA (puntos/rebotes/asistencias/etc. permitidos)."""
    logger.info("Obteniendo estadísticas defensivas por equipo desde NBA API...")
    try:
        from nba_api.stats.endpoints import leaguedashteamstats
        
        time.sleep(0.7)
        team_stats = leaguedashteamstats.LeagueDashTeamStats(
            season=SEASON,
            measure_type_detailed_defense='Opponent',
            per_mode_detailed='PerGame'
        )
        df = team_stats.get_data_frames()[0]
        
        if df.empty:
            logger.warning("No se recibieron estadísticas defensivas.")
            return

        records = []
        for _, row in df.iterrows():
            records.append({
                "team_name":    row.get("TEAM_NAME"),
                "pts_allowed":  safe_float(row, "OPP_PTS"),
                "reb_allowed":  safe_float(row, "OPP_REB"),
                "ast_allowed":  safe_float(row, "OPP_AST"),
                "fg3m_allowed": safe_float(row, "OPP_FG3M"),
                "blk_allowed":  safe_float(row, "OPP_BLK"),
                "stl_allowed":  safe_float(row, "OPP_STL"),
            })

        # 1. Guardar localmente como fallback
        local_path = "team_defense_stats.json"
        with open(local_path, "w", encoding="utf-8") as f:
            json.dump(records, f, indent=2, ensure_ascii=False)
        logger.info(f"  -> Guardadas estadísticas defensivas localmente en {local_path}.")

        # 2. Intentar guardar en Supabase
        if supabase:
            try:
                supabase.table("team_defense_stats").upsert(
                    records, on_conflict="team_name"
                ).execute()
                logger.info("  -> Subidas estadísticas defensivas a Supabase (tabla team_defense_stats).")
            except Exception as exc:
                logger.info("  -> Supabase table 'team_defense_stats' no accesible. Usando fallback local.")

    except Exception as e:
        logger.error(f"Error actualizando estadísticas defensivas: {e}")


def main():
    load_dotenv()
    supabase: Client = create_client(os.getenv("SUPABASE_URL"), os.getenv("SUPABASE_KEY"))

    # ── 1. Estadísticas defensivas por equipo ─────────────────────────────────
    update_team_defense_stats(supabase)

    # ── 2. Obtener jugadores y equipos desde player_odds ──────────────────────
    logger.info("Leyendo entidades activas desde player_odds en Supabase...")
    response = supabase.table("player_odds").select("player_name, market").execute()
    odds_data = response.data

    players = list(set(o["player_name"] for o in odds_data if o["market"] != "h2h"))
    teams   = list(set(o["player_name"] for o in odds_data if o["market"] == "h2h"))

    logger.info(f"  → {len(players)} jugadores y {len(teams)} equipos para actualizar.")

    # ── 3. Gamelogs de jugadores → player_stats ───────────────────────────────
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
                    "pts":         safe_int(row, "PTS"),
                    "reb":         safe_int(row, "REB"),
                    "ast":         safe_int(row, "AST"),
                    "fg3m":        safe_int(row, "FG3M"),
                    "stl":         safe_int(row, "STL"),
                    "blk":         safe_int(row, "BLK"),
                    "tov":         safe_int(row, "TOV"),
                    "min":         safe_float(row, "MIN"),
                    "league":      "NBA"
                })

            if records:
                upsert_player_records(supabase, records)
                total_player_rows += len(records)
                logger.info(f"    → {len(records)} partidos guardados en Supabase.")

        except Exception as e:
            logger.error(f"    → Error: {e}")

    logger.info(f"Jugadores: {total_player_rows} filas totales guardadas en player_stats.")

    # ── 4. Gamelogs de equipos → team_game_logs ───────────────────────────────
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
