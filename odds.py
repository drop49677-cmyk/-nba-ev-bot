import logging
import os
import sys
import time
from datetime import datetime, timezone

import numpy as np
import pandas as pd
import requests
from dotenv import load_dotenv
from supabase import create_client



try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger("EV_Bot_Odds")


BASIC_MARKETS = "h2h,spreads,totals"
PLAYER_PROP_MARKETS = [
    "player_points",
    "player_rebounds",
    "player_assists",
    "player_threes",
    "player_blocks",
    "player_steals",
    "player_turnovers",
    "player_points_rebounds_assists",
]


class OddsAPIClient:
    def __init__(self):
        load_dotenv()
        self.api_keys = []
        # Cargar clave primaria
        prim = os.getenv("ODDS_API_KEY")
        if prim:
            self.api_keys.append(prim)
        # Cargar claves de respaldo (ODDS_API_KEY_1 hasta 9)
        for i in range(1, 10):
            k = os.getenv(f"ODDS_API_KEY_{i}")
            if k:
                self.api_keys.append(k)
        
        # Eliminar duplicados manteniendo orden
        seen = set()
        self.api_keys = [x for x in self.api_keys if not (x in seen or seen.add(x))]
        self.current_key_idx = 0

        self.base_url = "https://api.the-odds-api.com/v4/sports"
        self.regions = os.getenv("ODDS_REGIONS", "us,eu")
        self.odds_format = "decimal"
        default_sports = (
            "soccer_epl,"
            "soccer_spain_la_liga,"
            "soccer_italy_serie_a,"
            "soccer_germany_bundesliga,"
            "soccer_france_ligue_one,"
            "soccer_usa_mls,"
            "basketball_nba"
        )
        sports = os.getenv("ODDS_SPORTS", default_sports)
        self.sports = [sport.strip() for sport in sports.split(",") if sport.strip()]
        self.fetch_nba_props = os.getenv("ODDS_FETCH_NBA_PROPS", "1").lower() not in {"0", "false", "no"}
        self.out_of_credits = False

    @property
    def api_key(self) -> str | None:
        if not self.api_keys or self.current_key_idx >= len(self.api_keys):
            return None
        return self.api_keys[self.current_key_idx]

    def rotate_key(self) -> bool:
        self.current_key_idx += 1
        if self.current_key_idx >= len(self.api_keys):
            logger.error("Se agotaron todas las API keys de The Odds API configuradas.")
            self.out_of_credits = True
            return False
        logger.info(
            "Rotando a la siguiente API Key (clave %s de %s)...", 
            self.current_key_idx + 1, len(self.api_keys)
        )
        return True

    @staticmethod
    def _response_detail(exc: Exception) -> str:
        response = getattr(exc, "response", None)
        if response is None:
            return str(exc)
        try:
            body = response.text[:700]
        except Exception:
            body = ""
        remaining = response.headers.get("x-requests-remaining")
        used = response.headers.get("x-requests-used")
        quota = f" | used={used} remaining={remaining}" if used or remaining else ""
        return f"{response.status_code} {response.reason}: {body}{quota}"

    @staticmethod
    def _is_out_of_credits(exc: Exception) -> bool:
        response = getattr(exc, "response", None)
        if response is None:
            return False
        try:
            return "OUT_OF_USAGE_CREDITS" in response.text
        except Exception:
            return False

    @staticmethod
    def _event_time(raw_time: str) -> tuple[str, str] | None:
        if not raw_time or "T" not in raw_time:
            return None
        try:
            dt = datetime.strptime(raw_time, "%Y-%m-%dT%H:%M:%SZ").replace(tzinfo=timezone.utc)
        except ValueError:
            return None
        if dt < datetime.now(timezone.utc):
            return None
        return dt.strftime("%Y-%m-%d"), dt.strftime("%H:%M UTC")

    @staticmethod
    def _matchup(data: dict, sport: str) -> str:
        icon = "NBA" if sport.startswith("basketball") else "SOCCER"
        return f"{icon} {data.get('home_team')} vs {data.get('away_team')}"

    def _get(self, url: str, params: dict, timeout: int = 20):
        while True:
            current_key = self.api_key
            if not current_key:
                raise ValueError("No API Key disponible en las configuraciones de .env.")
            params["apiKey"] = current_key
            try:
                response = requests.get(url, params=params, timeout=timeout)
                response.raise_for_status()
                return response
            except requests.exceptions.HTTPError as exc:
                if exc.response is not None:
                    # 401 Unauthorized, 429 Too Many Requests, o error de créditos
                    if exc.response.status_code in {401, 429} or "OUT_OF_USAGE_CREDITS" in exc.response.text:
                        logger.warning(f"Error con la clave actual: {exc.response.status_code} — {exc.response.text[:200]}")
                        if self.rotate_key():
                            time.sleep(1.0)
                            continue
                raise exc

    def get_player_props(self) -> list[dict]:
        if not self.api_key:
            logger.error("No hay ninguna ODDS_API_KEY configurada en el archivo .env.")
            return []


        all_odds: list[dict] = []
        for sport in self.sports:
            if self.out_of_credits:
                logger.error("Sin creditos de The Odds API. Deteniendo descarga para no gastar llamadas.")
                break
            logger.info("Escaneando deporte: %s", sport)
            before = len(all_odds)

            self._download_basic_markets(sport, all_odds)
            if self.out_of_credits:
                logger.error("Sin creditos tras mercados basicos. Saltando llamadas restantes.")
                break

            # Player props are mostly useful for NBA in this bot because update_stats.py
            # pulls NBA player logs. Keep it focused to save API credits.
            if sport == "basketball_nba" and self.fetch_nba_props:
                self._download_nba_props(sport, all_odds)
                if self.out_of_credits:
                    logger.error("Sin creditos tras props NBA. Saltando llamadas restantes.")
                    break

            logger.info("  -> %s cuotas nuevas para %s.", len(all_odds) - before, sport)

        logger.info("Total cuotas parseadas antes de guardar: %s", len(all_odds))
        return all_odds

    def _download_basic_markets(self, sport: str, all_odds: list[dict]):
        try:
            response = self._get(
                f"{self.base_url}/{sport}/odds",
                params={
                    "apiKey": self.api_key,
                    "regions": self.regions,
                    "markets": BASIC_MARKETS,
                    "oddsFormat": self.odds_format,
                },
                timeout=25,
            )
            events = response.json()
            logger.info("  Mercados basicos: %s eventos.", len(events))
        except Exception as exc:
            logger.error("  Error mercados basicos %s: %s", sport, self._response_detail(exc))
            if self._is_out_of_credits(exc):
                self.out_of_credits = True
            return

        for event in events:
            parsed_time = self._event_time(event.get("commence_time", ""))
            if not parsed_time:
                continue
            game_date, game_time = parsed_time
            matchup = self._matchup(event, sport)
            self._parse_bookmakers(event.get("bookmakers", []), matchup, game_date, game_time, all_odds)

    def _download_nba_props(self, sport: str, all_odds: list[dict]):
        try:
            response = self._get(
                f"{self.base_url}/{sport}/events",
                params={"apiKey": self.api_key},
                timeout=20,
            )
            events = response.json()
            logger.info("  Eventos NBA para props: %s.", len(events))
        except Exception as exc:
            logger.error("  Error eventos %s: %s", sport, self._response_detail(exc))
            if self._is_out_of_credits(exc):
                self.out_of_credits = True
            return

        for event in events:
            parsed_time = self._event_time(event.get("commence_time", ""))
            if not parsed_time:
                continue
            game_date, game_time = parsed_time
            matchup = self._matchup(event, sport)
            event_id = event.get("id")
            if not event_id:
                continue

            for market in PLAYER_PROP_MARKETS:
                try:
                    response = self._get(
                        f"{self.base_url}/{sport}/events/{event_id}/odds",
                        params={
                            "apiKey": self.api_key,
                            "regions": self.regions,
                            "markets": market,
                            "oddsFormat": self.odds_format,
                        },
                        timeout=20,
                    )
                    data = response.json()
                except Exception as exc:
                    logger.warning("    Sin mercado %s para %s: %s", market, matchup, self._response_detail(exc))
                    if self._is_out_of_credits(exc):
                        self.out_of_credits = True
                        return
                    continue

                before = len(all_odds)
                self._parse_bookmakers(data.get("bookmakers", []), matchup, game_date, game_time, all_odds)
                added = len(all_odds) - before
                if added:
                    logger.info("    %s: +%s lineas en %s.", matchup, added, market)

    def _parse_bookmakers(self, bookmakers, matchup, game_date, game_time, all_odds):
        for bookie in bookmakers:
            bookie_name = bookie.get("key")
            if not bookie_name:
                continue
            for market in bookie.get("markets", []):
                market_name = market.get("key")
                if market_name == "h2h":
                    self._parse_h2h(market, bookie_name, matchup, game_date, game_time, all_odds)
                elif market_name in {"spreads", "totals"}:
                    self._parse_game_line(market, bookie_name, matchup, game_date, game_time, all_odds)
                elif market_name in PLAYER_PROP_MARKETS:
                    self._parse_player_prop(market, bookie_name, matchup, game_date, game_time, all_odds)

    @staticmethod
    def _parse_h2h(market, bookie_name, matchup, game_date, game_time, all_odds):
        for outcome in market.get("outcomes", []):
            name = outcome.get("name")
            price = outcome.get("price")
            if not name or not price:
                continue
            all_odds.append(
                {
                    "player_name": name,
                    "market": "h2h",
                    "line": 0,
                    "over_odds": price,
                    "under_odds": 0,
                    "bookmaker": bookie_name,
                    "game_date": game_date,
                    "matchup": matchup,
                    "game_time": game_time,
                }
            )

    @staticmethod
    def _parse_game_line(market, bookie_name, matchup, game_date, game_time, all_odds):
        market_name = market.get("key")
        lines = {}
        for outcome in market.get("outcomes", []):
            name = outcome.get("name")
            point = outcome.get("point")
            price = outcome.get("price")
            if not name or point is None or not price:
                continue

            if market_name == "totals":
                key = ("Total Match Points", point)
                side = "over" if name.lower() == "over" else "under"
            else:
                key = (name, point)
                side = "over"

            lines.setdefault(key, {"over": None, "under": None})
            lines[key][side] = price

        for (name, point), prices in lines.items():
            if not prices["over"]:
                continue
            all_odds.append(
                {
                    "player_name": name,
                    "market": market_name,
                    "line": point,
                    "over_odds": prices["over"],
                    "under_odds": prices.get("under") or prices["over"],
                    "bookmaker": bookie_name,
                    "game_date": game_date,
                    "matchup": matchup,
                    "game_time": game_time,
                }
            )

    @staticmethod
    def _parse_player_prop(market, bookie_name, matchup, game_date, game_time, all_odds):
        market_name = market.get("key")
        lines = {}
        for outcome in market.get("outcomes", []):
            name = outcome.get("description")
            point = outcome.get("point")
            price = outcome.get("price")
            side = (outcome.get("name") or "").lower()
            if not name or point is None or not price or side not in {"over", "under"}:
                continue

            key = (name, point)
            lines.setdefault(key, {"over": None, "under": None})
            lines[key][side] = price

        for (name, point), prices in lines.items():
            if not prices["over"] or not prices["under"]:
                continue
            all_odds.append(
                {
                    "player_name": name,
                    "market": market_name,
                    "line": point,
                    "over_odds": prices["over"],
                    "under_odds": prices["under"],
                    "bookmaker": bookie_name,
                    "game_date": game_date,
                    "matchup": matchup,
                    "game_time": game_time,
                }
            )

    def save_to_supabase(self, odds_data: list[dict]):
        if not odds_data:
            logger.error(
                "No se descargaron cuotas. player_odds quedo vacia; revisa ODDS_API_KEY, "
                "creditos de The Odds API o deportes activos."
            )
            return

        supabase_url = os.getenv("SUPABASE_URL")
        supabase_key = os.getenv("SUPABASE_KEY")
        if not supabase_url or not supabase_key:
            logger.error("Faltan SUPABASE_URL o SUPABASE_KEY en .env. No se puede guardar.")
            return

        try:
            supabase = create_client(supabase_url, supabase_key)
            df = pd.DataFrame(odds_data)
            df = df.dropna(subset=["player_name", "market", "bookmaker", "game_date", "matchup", "line"])
            df = df.drop_duplicates(
                subset=["player_name", "market", "bookmaker", "game_date", "matchup", "line"],
                keep="last",
            )
            df = df.replace({np.nan: None})
            records = df.to_dict(orient="records")
            supabase.table("player_odds").upsert(
                records,
                on_conflict="player_name,market,bookmaker,game_date,matchup,line",
            ).execute()
            logger.info("Guardadas/actualizadas %s cuotas en Supabase player_odds.", len(records))
        except Exception as exc:
            logger.error("Error guardando en Supabase player_odds: %s", exc)


if __name__ == "__main__":
    client = OddsAPIClient()
    odds = client.get_player_props()
    client.save_to_supabase(odds)
