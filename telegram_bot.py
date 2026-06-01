import os
import asyncio
import logging
from dotenv import load_dotenv
from telegram import Update
from telegram.ext import Application, CommandHandler, ContextTypes

import pandas as pd
from analyzer import EVAnalyzer

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger("NBA_Telegram_Bot")

load_dotenv()

async def start_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE):
    msg = (
        "🏀 *Bot NBA Elite Activado* 🏀\n\n"
        "Comandos disponibles:\n"
        "/status - Resumen del sistema y picks de hoy\n"
        "/bankroll <monto> - Actualizar Bankroll\n"
        "/backtest <días> - Correr simulación histórica\n"
        "/settings - Ver configuración actual"
    )
    await update.message.reply_text(msg, parse_mode="Markdown")

async def status_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text("⏳ Calculando predicciones y buscando valor en el mercado...")
    # Correr el analyzer de forma síncrona/asíncrona
    try:
        analyzer = EVAnalyzer()
        await asyncio.to_thread(analyzer.run_pipeline)
        await update.message.reply_text("✅ Análisis completado. Las apuestas se enviaron al canal/chat configurado.")
    except Exception as e:
        await update.message.reply_text(f"❌ Error en el pipeline: {e}")

async def bankroll_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if not context.args:
        await update.message.reply_text("Uso: /bankroll <cantidad>\nEjemplo: `/bankroll 1500`", parse_mode="Markdown")
        return
    try:
        new_br = float(context.args[0])
        # Actualizaríamos la BD o variable de entorno
        # Por ahora solo respondemos
        await update.message.reply_text(f"💰 Bankroll actualizado a: ${new_br:,.2f}\n*(Nota: requiere actualizar el archivo .env o la BD para ser persistente)*")
    except ValueError:
        await update.message.reply_text("❌ Por favor ingresa un número válido.")

async def backtest_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE):
    days = context.args[0] if context.args else "30"
    await update.message.reply_text(f"⏳ Corriendo backtest Walk-Forward de los últimos {days} días...\n*(Calculando Sharpe, ROI, Drawdown)*")
    
    # Aquí llamaríamos a backtest.py
    # Para la prueba, corremos el backtest de forma asíncrona
    try:
        import subprocess
        result = await asyncio.to_thread(subprocess.run, ["python", "backtest.py"], capture_output=True, text=True)
        out = result.stdout
        
        # Parsear las líneas finales del output de backtest
        lines = out.split('\n')
        res_lines = [l for l in lines if "Yield" in l or "Hit Rate" in l or "Total P&L" in l]
        
        if res_lines:
            report = "📊 *RESULTADOS BACKTEST*\n" + "\n".join(res_lines)
            await update.message.reply_text(report, parse_mode="Markdown")
        else:
            await update.message.reply_text("✅ Backtest completado, revisa los logs para más detalles.")
            
    except Exception as e:
        await update.message.reply_text(f"❌ Error corriendo backtest: {e}")

async def settings_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE):
    from analyzer import MIN_PROB, MIN_EDGE, MIN_MINUTES, MIN_SCORE, KELLY_FRACTION
    msg = (
        "⚙️ *CONFIGURACIÓN ACTUAL DEL MOTOR NBA*\n\n"
        f"🔸 Probabilidad Mínima: {MIN_PROB*100}%\n"
        f"🔸 Edge Mínimo: {MIN_EDGE*100}%\n"
        f"🔸 Fracción de Kelly: {KELLY_FRACTION}x\n"
        f"🔸 Minutos Jugados (Filtro): {MIN_MINUTES} min\n"
        f"🔸 Score Compuesto Mínimo: {MIN_SCORE} (Grados S/A)\n\n"
        "Modificaciones avanzadas: editar `analyzer.py`"
    )
    await update.message.reply_text(msg, parse_mode="Markdown")

def main():
    token = os.getenv("TELEGRAM_BOT_TOKEN")
    if not token:
        logger.error("No TELEGRAM_BOT_TOKEN en .env")
        return
        
    app = Application.builder().token(token).build()
    
    app.add_handler(CommandHandler("start", start_cmd))
    app.add_handler(CommandHandler("status", status_cmd))
    app.add_handler(CommandHandler("bankroll", bankroll_cmd))
    app.add_handler(CommandHandler("backtest", backtest_cmd))
    app.add_handler(CommandHandler("settings", settings_cmd))
    
    logger.info("Bot de Telegram iniciado. Escuchando comandos...")
    app.run_polling()

if __name__ == "__main__":
    main()
