from fastapi import FastAPI, HTTPException
import httpx
import asyncio
import os
import time
import pandas as pd
import numpy as np
from datetime import datetime
import logging
from typing import Dict, Any, List
import json

# =============================================================================
# إعدادات البوت الجديد
# =============================================================================

TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN", "")
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID", "")
EXECUTOR_BOT_URL = os.getenv("EXECUTOR_BOT_URL", "")
EXECUTOR_BOT_API_KEY = os.getenv("EXECUTOR_BOT_API_KEY", "")
EXECUTE_TRADES = os.getenv("EXECUTE_TRADES", "false").lower() == "true"

# إعدادات التداول
SCAN_INTERVAL = 300  # 5 دقائق بين كل فحص
CONFIDENCE_THRESHOLD = 61  # عتبة الثقة

# العملات المدعومة
SUPPORTED_COINS = {
    'btc': {'name': 'Bitcoin', 'binance_symbol': 'BTCUSDT', 'symbol': 'BTC'},
    'eth': {'name': 'Ethereum', 'binance_symbol': 'ETHUSDT', 'symbol': 'ETH'},
    'bnb': {'name': 'Binance Coin', 'binance_symbol': 'BNBUSDT', 'symbol': 'BNB'},
    'sol': {'name': 'Solana', 'binance_symbol': 'SOLUSDT', 'symbol': 'SOL'},
    'xrp': {'name': 'Ripple', 'binance_symbol': 'XRPUSDT', 'symbol': 'XRP'},
}

TIMEFRAMES = ['1h', '15m', '5m']  # الأطر الزمنية للمسح

# إعداد التسجيل
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("simple_signal_generator")

app = FastAPI(title="Simple Crypto Signal Generator")

# إحصائيات النظام
system_stats = {
    "start_time": time.time(),
    "total_scans": 0,
    "signals_generated": 0,
    "signals_sent": 0
}

class SimpleSignalGenerator:
    """مولد إشارات مبسط يعتمد على استراتيجية المتوسطات + RSI + MACD"""
    
    @staticmethod
    def calculate_ema(prices: List[float], period: int) -> float:
        """حساب المتوسط المتحرك الأسي"""
        if len(prices) < period:
            return prices[-1] if prices else 0
        series = pd.Series(prices)
        ema = series.ewm(span=period, adjust=False).mean()
        return round(ema.iloc[-1], 4)
    
    @staticmethod
    def calculate_rsi(prices: List[float], period: int = 14) -> float:
        """حساب مؤشر RSI"""
        if len(prices) < period + 1:
            return 50.0
        
        deltas = np.diff(prices)
        gains = np.where(deltas > 0, deltas, 0)
        losses = np.where(deltas < 0, -deltas, 0)
        
        avg_gains = pd.Series(gains).rolling(period).mean().dropna().values
        avg_losses = pd.Series(losses).rolling(period).mean().dropna().values
        
        if len(avg_gains) == 0 or len(avg_losses) == 0:
            return 50.0
        
        if avg_losses[-1] == 0:
            return 100.0
        
        rs = avg_gains[-1] / avg_losses[-1]
        rsi = 100 - (100 / (1 + rs))
        return round(min(max(rsi, 0), 100), 2)
    
    @staticmethod
    def calculate_macd(prices: List[float]) -> Dict[str, float]:
        """حساب مؤشر MACD"""
        if len(prices) < 26:
            return {'macd': 0, 'signal': 0, 'histogram': 0}
        
        ema_12 = pd.Series(prices).ewm(span=12, adjust=False).mean().values
        ema_26 = pd.Series(prices).ewm(span=26, adjust=False).mean().values
        
        macd_line = ema_12[-1] - ema_26[-1]
        signal_line = pd.Series([ema_12[i] - ema_26[i] for i in range(len(prices))]).ewm(span=9, adjust=False).mean().values[-1]
        histogram = macd_line - signal_line
        
        return {
            'macd': round(macd_line, 4),
            'signal': round(signal_line, 4),
            'histogram': round(histogram, 4)
        }
    
    def analyze_trend(self, prices: List[float], current_price: float) -> Dict[str, Any]:
        """تحليل الاتجاه باستخدام المتوسطات المتحركة"""
        ema_9 = self.calculate_ema(prices, 9)
        ema_21 = self.calculate_ema(prices, 21)
        ema_50 = self.calculate_ema(prices, 50)
        
        # تحديد الترتيب والاتجاه
        ma_order = "صاعد" if ema_9 > ema_21 > ema_50 else "هابط" if ema_9 < ema_21 < ema_50 else "متذبذب"
        
        # قوة الاتجاه بناء على المسافات
        trend_strength = 0
        if ma_order == "صاعد":
            distance_9_21 = abs(ema_9 - ema_21) / current_price
            distance_21_50 = abs(ema_21 - ema_50) / current_price
            if distance_9_21 > 0.02 and distance_21_50 > 0.03:
                trend_strength = 10
            elif distance_9_21 > 0.01 and distance_21_50 > 0.015:
                trend_strength = 7
            else:
                trend_strength = 4
        elif ma_order == "هابط":
            distance_9_21 = abs(ema_9 - ema_21) / current_price
            distance_21_50 = abs(ema_21 - ema_50) / current_price
            if distance_9_21 > 0.02 and distance_21_50 > 0.03:
                trend_strength = 10
            elif distance_9_21 > 0.01 and distance_21_50 > 0.015:
                trend_strength = 7
            else:
                trend_strength = 4
        
        return {
            'ema_9': ema_9,
            'ema_21': ema_21,
            'ema_50': ema_50,
            'order': ma_order,
            'strength': trend_strength,
            'price_above_21': current_price > ema_21,
            'price_above_50': current_price > ema_50
        }
    
    def generate_signal(self, prices: List[float], volumes: List[float]) -> Dict[str, Any]:
        """توليد إشارة تداول مبسطة"""
        if len(prices) < 50:
            return {"signal": "none", "confidence": 0, "reason": "بيانات غير كافية"}
        
        current_price = prices[-1]
        rsi = self.calculate_rsi(prices)
        macd = self.calculate_macd(prices)
        trend = self.analyze_trend(prices, current_price)
        
        # حساب نقاط الثقة
        confidence_score = 0
        reasons = []
        
        # 1. تحليل المتوسطات (40 نقطة كحد أقصى)
        ma_score = 0
        if trend['order'] == "صاعد" and trend['price_above_21'] and trend['price_above_50']:
            ma_score = trend['strength'] * 4  # 10 * 4 = 40 نقطة
            reasons.append(f"المتوسطات صاعدة (قوة: {trend['strength']}/10)")
        elif trend['order'] == "هابط" and not trend['price_above_21'] and not trend['price_above_50']:
            ma_score = trend['strength'] * 4
            reasons.append(f"المتوسطات هابطة (قوة: {trend['strength']}/10)")
        
        # 2. تحليل RSI (30 نقطة كحد أقصى)
        rsi_score = 0
        if 40 <= rsi <= 65:  # منطقة مناسبة للشراء
            distance_from_50 = abs(rsi - 50)
            rsi_score = max(0, 30 - (distance_from_50 * 1.5))
            reasons.append(f"RSI في منطقة مناسبة: {rsi}")
        elif 35 <= rsi <= 60:  # منطقة مناسبة للبيع
            distance_from_50 = abs(rsi - 50)
            rsi_score = max(0, 30 - (distance_from_50 * 1.5))
            reasons.append(f"RSI في منطقة مناسبة: {rsi}")
        
        # 3. تحليل MACD (30 نقطة كحد أقصى)
        macd_score = 0
        if macd['histogram'] > 0 and macd['macd'] > macd['signal']:
            macd_score = min(30, abs(macd['histogram']) * 1000)
            reasons.append(f"MACD إيجابي: {macd['histogram']:.4f}")
        elif macd['histogram'] < 0 and macd['macd'] < macd['signal']:
            macd_score = min(30, abs(macd['histogram']) * 1000)
            reasons.append(f"MACD سلبي: {macd['histogram']:.4f}")
        
        confidence_score = ma_score + rsi_score + macd_score
        
        # تحديد اتجاه الإشارة
        signal_type = "none"
        if confidence_score >= CONFIDENCE_THRESHOLD:
            if trend['order'] == "صاعد" and 40 <= rsi <= 65 and macd['histogram'] > 0:
                signal_type = "BUY"
            elif trend['order'] == "هابط" and 35 <= rsi <= 60 and macd['histogram'] < 0:
                signal_type = "SELL"
        
        return {
            "signal": signal_type,
            "confidence": round(confidence_score),
            "price": current_price,
            "indicators": {
                "rsi": rsi,
                "macd": macd,
                "trend": trend,
                "scores": {
                    "moving_averages": round(ma_score),
                    "rsi": round(rsi_score),
                    "macd": round(macd_score)
                }
            },
            "reasons": reasons,
            "timestamp": time.time()
        }

class BinanceDataFetcher:
    """جلب البيانات من Binance"""
    
    def __init__(self):
        self.client = httpx.AsyncClient(timeout=30.0)
        self.signal_generator = SimpleSignalGenerator()
    
    async def get_coin_data(self, coin_symbol: str, timeframe: str) -> Dict[str, Any]:
        """جلب بيانات العملة وتحليلها"""
        try:
            url = f"https://api.binance.com/api/v3/klines?symbol={coin_symbol}&interval={timeframe}&limit=100"
            response = await self.client.get(url)
            
            if response.status_code == 200:
                data = response.json()
                prices = [float(item[4]) for item in data]  # أسعار الإغلاق
                volumes = [float(item[5]) for item in data]  # أحجام التداول
                
                # توليد الإشارة
                signal = self.signal_generator.generate_signal(prices, volumes)
                signal['prices'] = prices
                signal['volumes'] = volumes
                signal['timeframe'] = timeframe
                
                return signal
            else:
                return {"signal": "none", "confidence": 0, "reason": "فشل جلب البيانات"}
                
        except Exception as e:
            logger.error(f"خطأ في جلب بيانات {coin_symbol}: {e}")
            return {"signal": "none", "confidence": 0, "reason": f"خطأ: {str(e)}"}

class TelegramNotifier:
    """إشعارات التليجرام"""
    
    def __init__(self, token: str, chat_id: str):
        self.token = token
        self.chat_id = chat_id
        self.base_url = f"https://api.telegram.org/bot{token}"
    
    async def send_signal_alert(self, coin: str, timeframe: str, signal_data: Dict[str, Any]) -> bool:
        """إرسال إشعار بالإشارة"""
        if signal_data["signal"] == "none":
            return False
        
        try:
            message = self._build_signal_message(coin, timeframe, signal_data)
            
            payload = {
                'chat_id': self.chat_id,
                'text': message,
                'parse_mode': 'Markdown'
            }
            
            async with httpx.AsyncClient() as client:
                response = await client.post(f"{self.base_url}/sendMessage", 
                                           json=payload, timeout=10.0)
            
            if response.status_code == 200:
                logger.info(f"تم إرسال إشعار إشارة لـ {coin} ({timeframe})")
                return True
            else:
                logger.error(f"فشل إرسال الإشعار: {response.status_code}")
                return False
                
        except Exception as e:
            logger.error(f"خطأ في إرسال الإشعار: {e}")
            return False
    
    def _build_signal_message(self, coin: str, timeframe: str, signal_data: Dict[str, Any]) -> str:
        """بناء رسالة الإشارة"""
        signal_type = signal_data["signal"]
        confidence = signal_data["confidence"]
        price = signal_data["price"]
        indicators = signal_data["indicators"]
        reasons = signal_data["reasons"]
        
        if signal_type == "BUY":
            emoji = "🟢"
            action = "شراء"
            color = "#00C851"
        else:  # SELL
            emoji = "🔴"
            action = "بيع"
            color = "#FF4444"
        
        message = f"{emoji} **إشارة {action} - {coin.upper()}** {emoji}\n"
        message += "═" * 40 + "\n\n"
        
        message += f"💰 **السعر الحالي:** `${price:,.2f}`\n"
        message += f"⏰ **الإطار الزمني:** `{timeframe}`\n"
        message += f"🎯 **قوة الإشارة:** `{confidence}/100`\n"
        message += f"🕒 **الوقت:** `{datetime.now().strftime('%H:%M %d/%m/%Y')}`\n\n"
        
        message += "📊 **تفاصيل المؤشرات:**\n"
        message += f"• 📈 **RSI:** `{indicators['rsi']}`\n"
        message += f"• 🔄 **MACD Hist:** `{indicators['macd']['histogram']:.4f}`\n"
        message += f"• 📶 **اتجاه المتوسطات:** `{indicators['trend']['order']}`\n"
        message += f"• 💪 **قوة الاتجاه:** `{indicators['trend']['strength']}/10`\n\n"
        
        message += "📈 **توزيع النقاط:**\n"
        scores = indicators['scores']
        message += f"• المتوسطات: `{scores['moving_averages']}/40`\n"
        message += f"• RSI: `{scores['rsi']}/30`\n"
        message += f"• MACD: `{scores['macd']}/30`\n\n"
        
        message += "🔍 **أسباب الإشارة:**\n"
        for reason in reasons:
            message += f"• {reason}\n"
        
        message += "\n"
        message += "⚡ **مولد الإشارات المبسط**\n"
        message += "🎯 **استراتيجية:** المتوسطات + RSI + MACD"
        
        return message

class ExecutorBotClient:
    """عميل للتواصل مع بوت التنفيذ"""
    
    def __init__(self, base_url: str, api_key: str):
        self.base_url = base_url
        self.api_key = api_key
        self.client = httpx.AsyncClient(timeout=30.0)
    
    async def send_trade_signal(self, signal_data: Dict[str, Any]) -> bool:
        """إرسال إشارة تداول إلى البوت المنفذ"""
        if not EXECUTE_TRADES:
            logger.info("تنفيذ الصفقات معطل في الإعدادات")
            return False
            
        try:
            headers = {
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json"
            }
            
            payload = {
                "signal": signal_data,
                "timestamp": time.time(),
                "source": "simple_signal_generator",
                "version": "1.0.0"
            }
            
            response = await self.client.post(
                f"{self.base_url}/api/trade/signal",
                json=payload,
                headers=headers
            )
            
            if response.status_code == 200:
                logger.info(f"تم إرسال إشارة للتنفيذ: {signal_data['coin']} - {signal_data['signal']}")
                system_stats["signals_sent"] += 1
                return True
            else:
                logger.error(f"فشل إرسال الإشارة: {response.status_code}")
                return False
                
        except Exception as e:
            logger.error(f"خطأ في التواصل مع البوت المنفذ: {e}")
            return False

# =============================================================================
# التهيئة
# =============================================================================

data_fetcher = BinanceDataFetcher()
notifier = TelegramNotifier(TELEGRAM_BOT_TOKEN, TELEGRAM_CHAT_ID)
executor_client = ExecutorBotClient(EXECUTOR_BOT_URL, EXECUTOR_BOT_API_KEY)

# =============================================================================
# المهام الأساسية
# =============================================================================

async def market_scanner_task():
    """المهمة الرئيسية للمسح الضوئي"""
    logger.info("بدء مهمة مسح السوق كل 5 دقائق")
    
    while True:
        try:
            signals_found = 0
            
            for coin_key, coin_data in SUPPORTED_COINS.items():
                for timeframe in TIMEFRAMES:
                    try:
                        # جلب البيانات وتحليلها
                        signal_data = await data_fetcher.get_coin_data(coin_data['binance_symbol'], timeframe)
                        
                        # إذا كانت هناك إشارة قوية
                        if (signal_data["signal"] != "none" and 
                            signal_data["confidence"] >= CONFIDENCE_THRESHOLD):
                            
                            logger.info(f"إشارة {signal_data['signal']} لـ {coin_key} ({timeframe}) - ثقة: {signal_data['confidence']}")
                            
                            # إرسال إشعار التليجرام
                            await notifier.send_signal_alert(coin_key, timeframe, signal_data)
                            
                            # إرسال إشارة التنفيذ
                            trade_signal = {
                                "coin": coin_key,
                                "symbol": coin_data['binance_symbol'],
                                "action": signal_data["signal"],
                                "timeframe": timeframe,
                                "price": signal_data["price"],
                                "confidence": signal_data["confidence"],
                                "reasons": signal_data["reasons"],
                                "indicators": signal_data["indicators"]
                            }
                            
                            await executor_client.send_trade_signal(trade_signal)
                            signals_found += 1
                            
                            # انتظار بين الإشارات
                            await asyncio.sleep(2)
                            
                    except Exception as e:
                        logger.error(f"خطأ في معالجة {coin_key} ({timeframe}): {e}")
                        continue
            
            system_stats["total_scans"] += 1
            system_stats["signals_generated"] += signals_found
            
            if signals_found > 0:
                logger.info(f"اكتملت دورة المسح - تم العثور على {signals_found} إشارة")
            else:
                logger.info("اكتملت دورة المسح - لا توجد إشارات قوية")
            
            await asyncio.sleep(SCAN_INTERVAL)
            
        except Exception as e:
            logger.error(f"خطأ في المهمة الرئيسية: {e}")
            await asyncio.sleep(60)

# =============================================================================
# واجهات API
# =============================================================================

@app.get("/")
async def root():
    return {
        "message": "Simple Crypto Signal Generator",
        "status": "running",
        "version": "1.0.0",
        "strategy": "EMA + RSI + MACD",
        "confidence_threshold": CONFIDENCE_THRESHOLD,
        "supported_coins": list(SUPPORTED_COINS.keys())
    }

@app.get("/scan/{coin}")
async def scan_coin(coin: str, timeframe: str = "1h"):
    if coin not in SUPPORTED_COINS:
        raise HTTPException(404, "العملة غير مدعومة")
    if timeframe not in TIMEFRAMES:
        raise HTTPException(404, "الإطار الزمني غير مدعوم")
    
    coin_data = SUPPORTED_COINS[coin]
    signal_data = await data_fetcher.get_coin_data(coin_data['binance_symbol'], timeframe)
    
    return {
        "coin": coin,
        "timeframe": timeframe,
        "signal": signal_data
    }

@app.get("/system-stats")
async def get_system_stats():
    uptime = time.time() - system_stats["start_time"]
    return {
        "uptime_seconds": uptime,
        "total_scans": system_stats["total_scans"],
        "signals_generated": system_stats["signals_generated"],
        "signals_sent": system_stats["signals_sent"],
        "confidence_threshold": CONFIDENCE_THRESHOLD,
        "scan_interval": SCAN_INTERVAL
    }

@app.get("/test-signal/{coin}")
async def test_signal(coin: str, timeframe: str = "1h"):
    """اختبار توليد إشارة لعملة معينة"""
    if coin not in SUPPORTED_COINS:
        raise HTTPException(404, "العملة غير مدعومة")
    
    coin_data = SUPPORTED_COINS[coin]
    signal_data = await data_fetcher.get_coin_data(coin_data['binance_symbol'], timeframe)
    
    # إرسال إشعار تجريبي
    await notifier.send_signal_alert(coin, timeframe, signal_data)
    
    return {
        "coin": coin,
        "timeframe": timeframe,
        "signal": signal_data,
        "test_alert_sent": True
    }

# =============================================================================
# تشغيل التطبيق
# =============================================================================

@app.on_event("startup")
async def startup_event():
    logger.info("بدء تشغيل مولد الإشارات المبسط")
    logger.info(f"العملات المدعومة: {list(SUPPORTED_COINS.keys())}")
    logger.info(f"الأطر الزمنية: {TIMEFRAMES}")
    logger.info(f"عتبة الثقة: {CONFIDENCE_THRESHOLD}")
    logger.info(f"فاصل المسح: {SCAN_INTERVAL} ثانية")
    
    # بدء المهمة الرئيسية
    asyncio.create_task(market_scanner_task())

if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)
