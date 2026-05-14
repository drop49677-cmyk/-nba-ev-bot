import os
import time
import logging
import datetime
from typing import List, Dict, Any

import requests
import pandas as pd
from dotenv import load_dotenv
from supabase import create_client, Client

# ==========================================
# 1. CONFIGURACIÓN DEL LOGGER
# ==========================================
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("bot_ingestion.log", mode='a')
    ]
)
logger = logging.getLogger("EV_Bot_Ingestion")

# Constantes de Ligas (Fase 1: Filtrado de las 5 ligas objetivo)
TARGET_LEAGUES = ["NBA", "EuroLeague", "ACB", "BSL", "ABA"]

from nba_api.stats.endpoints import scoreboardv3, boxscoretraditionalv3

# ==========================================
# 2. CAPA DE INTERACCIÓN CON APIs (NBA_API)
# ==========================================
class SportsDataAPI:
    """
    Clase para manejar las llamadas a stats.nba.com mediante la librería nba_api.
    """
    def __init__(self):
        # nba_api maneja las sesiones y headers internamente
        pass
        
    def get_todays_games(self, target_date=None) -> List[Dict[str, Any]]:
        """
        Obtiene los partidos de una fecha específica. Si target_date es None, usa hoy.
        """
        if not target_date:
            target_date = datetime.date.today().strftime("%Y-%m-%d")
            
        logger.info(f"Consultando partidos para la fecha: {target_date}")
        
        try:
            sb = scoreboardv3.ScoreboardV3(game_date=target_date)
            games_dict = sb.get_dict().get("scoreboard", {}).get("games", [])
            
            games = []
            for g in games_dict:
                games.append({
                    "game_id": g["gameId"],
                    "league": "NBA",
                    "home_team": g.get("homeTeam", {}).get("teamTricode", "HOME"),
                    "away_team": g.get("awayTeam", {}).get("teamTricode", "AWAY"),
                    "game_date": target_date
                })
            return games
        except Exception as e:
            logger.error(f"Error consultando partidos en nba_api: {e}")
            return []

    def get_player_stats_for_game(self, game_id: str, home_team: str, away_team: str, game_date: str) -> List[Dict[str, Any]]:
        """
        Extrae las estadísticas de todos los jugadores de ambos equipos para un partido dado usando nba_api.
        """
        stats = []
        try:
            # Rate limit preventivo para nba_api
            time.sleep(1) 
            bx = boxscoretraditionalv3.BoxScoreTraditionalV3(game_id=game_id)
            bx_dict = bx.get_dict().get("boxScoreTraditional", {})
            
            for team_type, opponent_tricode in [("homeTeam", f"vs {away_team}"), ("awayTeam", f"@ {home_team}")]:
                team_data = bx_dict.get(team_type, {})
                players = team_data.get("players", [])
                
                for p in players:
                    p_stats = p.get("statistics", {})
                    # Si el jugador no jugó, a veces minutes está vacío o en 0
                    min_played = p_stats.get("minutes", "0.0")
                    if not min_played or min_played == "00:00" or min_played == "0.0":
                        continue
                        
                    first = p.get("firstName", "")
                    last = p.get("familyName", "")
                    player_name = f"{first} {last}".strip()
                    if not player_name:
                        player_name = p.get("nameI", "Unknown")
                        
                    stats.append({
                        "player_id": p.get("personId"),
                        "player_name": player_name,
                        "game_date": game_date,
                        "opponent": opponent_tricode,
                        "pts": p_stats.get("points", 0),
                        "reb": p_stats.get("reboundsTotal", 0),
                        "ast": p_stats.get("assists", 0),
                        "min": min_played
                    })
        except Exception as e:
            logger.error(f"Error consultando boxscore del partido {game_id}: {e}")
            
        return stats

# ==========================================
# 3. PIPELINE DE INGESTA DE DATOS
# ==========================================
class DataIngestionPipeline:
    def __init__(self):
        load_dotenv()
        self.supabase_url = os.getenv("SUPABASE_URL")
        self.supabase_key = os.getenv("SUPABASE_KEY")
        
        if not self.supabase_url or not self.supabase_key:
            logger.warning("Credenciales SUPABASE_URL o SUPABASE_KEY no detectadas. Solo se imprimirá en local.")
            self.supabase = None
        else:
            try:
                self.supabase: Client = create_client(self.supabase_url, self.supabase_key)
            except Exception as e:
                logger.error(f"Error al conectar con Supabase: {e}. Asegúrate de usar la Anon Key o Service Role Key (empiezan con eyJ). Solo se imprimirá en local.")
                self.supabase = None
            
        self.api = SportsDataAPI()
        
    def run(self):
        logger.info("=== INICIANDO FASE 1: INGESTA DE DATOS ===")
        
        # 1. Obtener partidos de ayer (Para prueba inicial)
        target_date = (datetime.date.today() - datetime.timedelta(days=1)).strftime("%Y-%m-%d")
        games = self.api.get_todays_games(target_date)
        # Filtrar a las ligas objetivo
        target_games = [g for g in games if g["league"] in TARGET_LEAGUES]
        logger.info(f"Partidos programados para {target_date} en ligas objetivo: {len(target_games)}")
        
        all_player_stats = []
        
        # 2. Iterar sobre los partidos
        for game in target_games:
            logger.info(f"Procesando partido {game['game_id']} de la liga {game['league']}")
            
            try:
                stats = self.api.get_player_stats_for_game(
                    game["game_id"], 
                    game["home_team"], 
                    game["away_team"],
                    game["game_date"]
                )
                
                for s in stats:
                    s["league"] = game["league"]
                    
                all_player_stats.extend(stats)
            except Exception as e:
                 logger.error(f"Error procesando el partido {game['game_id']}: {e}")
                 continue

        if not all_player_stats:
            logger.warning("No se recolectaron estadísticas de jugadores. Terminando proceso.")
            return
        # ==========================================
        # 4. LIMPIEZA Y NORMALIZACIÓN (PANDAS)
        # ==========================================
        logger.info("Iniciando limpieza y normalización con Pandas...")
        df = pd.DataFrame(all_player_stats)
        
        # Conversión de tipos y manejo de valores nulos
        df["pts"] = pd.to_numeric(df["pts"], errors="coerce").fillna(0).astype(int)
        df["reb"] = pd.to_numeric(df["reb"], errors="coerce").fillna(0).astype(int)
        df["ast"] = pd.to_numeric(df["ast"], errors="coerce").fillna(0).astype(int)
        df["min"] = pd.to_numeric(df["min"], errors="coerce").fillna(0.0)
        df["game_date"] = pd.to_datetime(df["game_date"]).dt.strftime('%Y-%m-%d')
        
        # Limpieza de duplicados: Un jugador no puede tener 2 registros el mismo día
        df = df.drop_duplicates(subset=["player_id", "game_date"], keep="last")
        
        logger.info(f"Dataset final preparado: {len(df)} registros procesados.")

        # ==========================================
        # 5. CONEXIÓN Y UPSERT A SUPABASE
        # ==========================================
        if self.supabase:
            logger.info("Realizando UPSERT en la tabla 'player_stats' de Supabase...")
            records = df.to_dict(orient="records")
            
            try:
                # Nota para Supabase:
                # La tabla `player_stats` debe tener una Primary Key o un Constraint Unique
                # (por ejemplo compuesto por player_id y game_date) para que 'upsert' no duplique registros.
                response = self.supabase.table("player_stats").upsert(records).execute()
                
                # Dependiendo de la versión de supabase-py, response.data contiene los registros insertados
                inserted_count = len(response.data) if hasattr(response, 'data') else "N/A"
                logger.info(f"UPSERT exitoso. Registros afectados/verificados: {inserted_count}")
                
            except Exception as e:
                logger.error(f"Fallo crítico durante el UPSERT en Supabase: {e}")
        else:
            logger.info("Modo local finalizado. Muestra de los primeros 5 registros normalizados:")
            print(df.head())


if __name__ == "__main__":
    pipeline = DataIngestionPipeline()
    pipeline.run()
