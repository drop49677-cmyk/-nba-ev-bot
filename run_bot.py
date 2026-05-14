"""
run_bot.py — Script maestro del pipeline completo.
Ejecuta en orden:
  1. odds.py    → Descarga cuotas frescas de los casinos y las guarda en Supabase
  2. analyzer.py → Analiza los datos y envía las apuestas +EV a Telegram

Este es el archivo que GitHub Actions ejecuta todos los días a la 1 AM (Colombia).
"""

import subprocess
import sys
import logging
from datetime import datetime

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("BOT_RUNNER")


def run_script(script_name: str) -> bool:
    """Ejecuta un script Python y retorna True si tuvo éxito."""
    logger.info(f"{'='*50}")
    logger.info(f"INICIANDO: {script_name}")
    logger.info(f"{'='*50}")
    
    result = subprocess.run(
        [sys.executable, script_name],
        capture_output=False,  # Muestra output en tiempo real
        text=True
    )
    
    if result.returncode == 0:
        logger.info(f"✅ {script_name} completado exitosamente.")
        return True
    else:
        logger.error(f"❌ {script_name} falló con código {result.returncode}.")
        return False


if __name__ == "__main__":
    start = datetime.now()
    logger.info(f"🤖 BOT NBA +EV — Ejecución iniciada: {start.strftime('%Y-%m-%d %H:%M:%S')}")

    # Paso 1: Descargar cuotas de casinos
    if not run_script("odds.py"):
        logger.error("El pipeline se detuvo porque odds.py falló.")
        sys.exit(1)

    # Paso 2: Analizar y enviar alertas a Telegram
    if not run_script("analyzer.py"):
        logger.error("El pipeline se detuvo porque analyzer.py falló.")
        sys.exit(1)

    end = datetime.now()
    elapsed = (end - start).seconds
    logger.info(f"🏁 Pipeline completo en {elapsed}s — {end.strftime('%H:%M:%S')}")
