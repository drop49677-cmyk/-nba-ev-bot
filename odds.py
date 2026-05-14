import os
import requests
import pandas as pd
from datetime import datetime
import logging
from dotenv import load_dotenv
from supabase import create_client, Client

# Configuración básica de logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("EV_Bot_Odds")

class OddsAPIClient:
    """
    Cliente para extraer cuotas de apuestas usando The Odds API.
    Mercados objetivo: Puntos, Rebotes, Asistencias de jugadores.
    """
    def __init__(self):
        load_dotenv()
        self.api_key = os.getenv("ODDS_API_KEY")
        if not self.api_key:
            logger.warning("No se encontró ODDS_API_KEY en .env. Consigue una en the-odds-api.com")
            
        self.base_url = "https://api.the-odds-api.com/v4/sports"
        self.sport = "basketball_nba"
        
        # Mercados: Player Props + Ganador del Partido (h2h)
        self.markets = "player_points,player_rebounds,player_assists,h2h"
        # Regiones (us, uk, eu, au) - Depende de las casas de apuestas que uses
        self.regions = "us,eu" 
        
    def get_player_props(self):
        """
        Descarga las cuotas actuales para los partidos próximos de la NBA (Jugadores y Equipos).
        """
        if not self.api_key:
            return []
            
        # 1. Obtener los eventos (partidos)
        events_url = f"{self.base_url}/{self.sport}/events"
        try:
            logger.info("Obteniendo eventos próximos de la NBA...")
            events_res = requests.get(events_url, params={"apiKey": self.api_key})
            events_res.raise_for_status()
            events = events_res.json()
        except Exception as e:
            logger.error(f"Error obteniendo eventos: {e}")
            return []
            
        all_odds = []
        
        # 2. Por cada evento, pedir las cuotas (Player Props y H2H)
        for event in events:
            event_id = event["id"]
            commence_time_raw = event["commence_time"] # ej: 2026-05-15T23:10:00Z
            game_date = commence_time_raw.split("T")[0]
            
            # Formatear la hora legible
            try:
                dt_obj = datetime.strptime(commence_time_raw, "%Y-%m-%dT%H:%M:%SZ")
                game_time = dt_obj.strftime("%H:%M UTC")
            except:
                game_time = commence_time_raw
                
            matchup = f"{event['home_team']} vs {event['away_team']}"
            
            logger.info(f"Descargando cuotas para: {matchup}")
            
            odds_url = f"{self.base_url}/{self.sport}/events/{event_id}/odds"
            params = {
                "apiKey": self.api_key,
                "regions": self.regions,
                "markets": self.markets,
                "oddsFormat": "decimal"
            }
            
            try:
                odds_res = requests.get(odds_url, params=params)
                odds_res.raise_for_status()
                data = odds_res.json()
                
                # Procesar la respuesta
                bookmakers = data.get("bookmakers", [])
                for bookie in bookmakers:
                    bookie_name = bookie["key"]
                    
                    for market in bookie.get("markets", []):
                        market_name = market["key"] # ej: player_points o h2h
                        
                        if market_name == "h2h":
                            # Mercado de Equipos (Ganador)
                            for outcome in market.get("outcomes", []):
                                team_name = outcome.get("name")
                                price = outcome.get("price")
                                all_odds.append({
                                    "player_name": team_name,
                                    "market": "h2h",
                                    "line": 0,
                                    "over_odds": price,
                                    "under_odds": 0,
                                    "bookmaker": bookie_name,
                                    "game_date": game_date,
                                    "matchup": matchup,
                                    "game_time": game_time
                                })
                        elif market_name in ["player_points", "player_rebounds", "player_assists"]:
                            # Mercados de Jugadores
                            player_lines = {}
                            
                            for outcome in market.get("outcomes", []):
                                player = outcome.get("description", "Unknown")
                                tipo = outcome.get("name") # "Over" o "Under"
                                price = outcome.get("price")
                                point = outcome.get("point") # La línea (ej: 24.5)
                                
                                # Evitar líneas nulas por fallos de la API
                                if point is None:
                                    continue
                                    
                                if player not in player_lines:
                                    player_lines[player] = {"line": point, "over_odds": None, "under_odds": None}
                                    
                                if tipo == "Over":
                                    player_lines[player]["over_odds"] = price
                                elif tipo == "Under":
                                    player_lines[player]["under_odds"] = price
                                    
                            # Agregar a la lista final
                            for player, lines in player_lines.items():
                                all_odds.append({
                                    "player_name": player,
                                    "market": market_name,
                                    "line": lines["line"],
                                    "over_odds": lines["over_odds"],
                                    "under_odds": lines["under_odds"],
                                    "bookmaker": bookie_name,
                                    "game_date": game_date,
                                    "matchup": matchup,
                                    "game_time": game_time
                                })
                            
            except Exception as e:
                logger.error(f"Error extrayendo cuotas del evento {event_id}: {e}")
                
        logger.info(f"Total de líneas (cuotas) extraídas: {len(all_odds)}")
        return all_odds

    def save_to_supabase(self, odds_data):
        """
        Sube los datos limpios a la tabla player_odds en Supabase
        """
        if not odds_data:
            return
            
        supabase_url = os.getenv("SUPABASE_URL")
        supabase_key = os.getenv("SUPABASE_KEY")
        
        if not supabase_url or not supabase_key:
            logger.warning("Credenciales de Supabase no encontradas.")
            return
            
        try:
            supabase: Client = create_client(supabase_url, supabase_key)
            logger.info("Realizando UPSERT de cuotas en Supabase (tabla 'player_odds')...")
            
            # Usar Pandas para limpiar duplicados si los hubiera
            df = pd.DataFrame(odds_data)
            
            # Eliminamos duplicados exactos en Python antes de enviar (por seguridad)
            df = df.drop_duplicates(subset=["player_name", "market", "bookmaker", "game_date"], keep="last")
            # Convertir NaN a None para que Supabase (JSON) lo acepte como null
            import numpy as np
            df = df.replace({np.nan: None})
            records = df.to_dict(orient="records")
            
            # Usamos upsert con on_conflict para que, si el bot corre 2 veces el mismo día, 
            # simplemente actualice las cuotas en lugar de fallar por duplicados.
            response = supabase.table("player_odds").upsert(
                records, 
                on_conflict="player_name,market,bookmaker,game_date"
            ).execute()
            
            logger.info(f"Upsert exitoso. Registros subidos/actualizados: {len(records)}")
        except Exception as e:
            logger.error(f"Fallo crítico al subir cuotas a Supabase: {e}")

if __name__ == "__main__":
    client = OddsAPIClient()
    # 1. Extraer cuotas
    odds_data = client.get_player_props()
    # 2. Imprimir muestra
    if odds_data:
        print("\nMuestra de cuotas extraídas:")
        print(pd.DataFrame(odds_data).head(10).to_string())
    # 3. Guardar en Supabase
    client.save_to_supabase(odds_data)
