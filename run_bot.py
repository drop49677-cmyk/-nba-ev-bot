"""
run_bot.py — Pipeline maestro v3.
Orden: track_results.py → odds.py → analyzer.py
"""
import subprocess
import sys
import time
import logging
from datetime import datetime

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger("BOT_RUNNER")


def run_script(script_name: str, max_retries: int = 2, delay: int = 30) -> bool:
    """Ejecuta un script con reintentos automáticos en caso de fallo."""
    logger.info("=" * 60)
    logger.info("INICIANDO: %s", script_name)
    logger.info("=" * 60)
    for attempt in range(1, max_retries + 1):
        result = subprocess.run(
            [sys.executable, script_name],
            capture_output=False,
            text=True,
        )
        if result.returncode == 0:
            logger.info("✅ %s completado exitosamente.", script_name)
            return True
        logger.warning("⚠️  %s falló (intento %s/%s).", script_name, attempt, max_retries)
        if attempt < max_retries:
            logger.info("   Reintentando en %ss...", delay)
            time.sleep(delay)
    logger.error("❌ %s falló tras %s intentos.", script_name, max_retries)
    return False


if __name__ == "__main__":
    start = datetime.now()
    logger.info("🤖 BOT NBA +EV v3 — Inicio: %s", start.strftime("%Y-%m-%d %H:%M:%S"))
    results = {}

    # Paso 0: Verificar resultados de ayer y calcular ROI
    logger.info("=" * 60)
    logger.info("PASO 0: Verificando resultados de ayer...")
    logger.info("=" * 60)
    results["track_results.py"] = run_script("track_results.py", max_retries=1, delay=5)

    # Paso 1: Descargar cuotas
    results["odds.py"] = run_script("odds.py", max_retries=3, delay=20)
    if not results["odds.py"]:
        logger.error("Pipeline detenido: odds.py falló.")
        sys.exit(1)

    # Paso 2: Análisis +EV y envío a Telegram
    results["analyzer.py"] = run_script("analyzer.py", max_retries=2, delay=15)
    if not results["analyzer.py"]:
        logger.error("analyzer.py falló.")
        sys.exit(1)

    elapsed = (datetime.now() - start).seconds
    logger.info("\n" + "=" * 60)
    logger.info("🏁 Pipeline completo en %ss", elapsed)
    for script, ok in results.items():
        status = "✅" if ok else "❌"
        logger.info("  %s %s", status, script)
    logger.info("=" * 60)
