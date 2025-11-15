# ====================== المكتبات ======================
import os
import pandas as pd
import numpy as np
import requests
from typing import Dict, Any, List, Optional, Tuple
from datetime import datetime, timedelta
import telebot
import warnings
import logging
from tenacity import retry, stop_after_attempt, wait_exponential
from dotenv import load_dotenv
import matplotlib.pyplot as plt
import io
from collections import Counter
import asyncio
import httpx
import time
from fastapi import FastAPI, HTTPException
import uvicorn

warnings.filterwarnings('ignore')
load_dotenv()

# ====================== إعدادات FastAPI ======================
app = FastAPI(title="Crypto Signals Scanner", version="2.0.0")

# ====================== إعدادات التفعيل / الإلغاء ======================
ENABLE_TRAILING_STOP = False
ENABLE_DYNAMIC_POSITION_SIZING = False
ENABLE_MARKET_REGIME_FILTER = False
ENABLE_ATR_SL_TP = False
ENABLE_SUPPORT_RESISTANCE_FILTER = True
ENABLE_TIME_FILTER = True
ENABLE_WALK_FORWARD = False
ENABLE_LOGGING = True
ENABLE_DETAILED_REPORT = False
ENABLE_FUTURES_TRADING = True
ENABLE_SIGNAL_SENDING = True

# ====================== إعدادات البوت المنفذ ======================
EXECUTOR_BOT_URL = os.getenv("EXECUTOR_BOT_URL", "https://your-executor-bot.onrender.com")
EXECUTOR_BOT_API_KEY = os.getenv("EXECUTOR_BOT_API_KEY", "")
EXECUTE_TRADES = os.getenv("EXECUTE_TRADES", "false").lower() == "true"

# ====================== إعدادات المسح ======================
SCAN_INTERVAL = 600  # 10 دقائق بين كل فحص
CONFIDENCE_THRESHOLD = 60  # عتبة الثقة للإشارات

# ====================== دالة مساعدة ======================
def interval_to_hours(interval):
    mapping = {
        '1m': 1/60, '3m': 3/60, '5m': 5/60, '15m': 15/60,
        '30m': 30/60, '1h': 1, '2h': 2, '4h': 4, '6h': 6,
        '8h': 8, '12h': 12, '1d': 24, '3d': 72, '1w': 168, '1M': 720
    }
    return mapping.get(interval, 4)

# ====================== الإعدادات الأساسية ======================
TRADE_CONFIG = {
    'symbol': 'BNBUSDT',
    'timeframe': '4h',
    'initial_balance': 200,
    'leverage': 10,
    'base_stop_loss': 0.015,
    'base_take_profit': 0.045,
    'base_position_size': 0.25,
    'max_positions': 4,
    'paper_trading': True,
    'use_trailing_stop': ENABLE_TRAILING_STOP,
    'trailing_stop_percent': 0.01,
    'trailing_activation': 0.015,
    'max_trade_duration': 48,
    'atr_multiplier_sl': 1.5,
    'atr_multiplier_tp': 3.0,
    'atr_period': 14,
    'support_resistance_window': 20,
    'peak_hours': [0, 4, 8, 12, 16, 20],
    'min_volume_ratio': 1.2,
    'market_type': 'FUTURES',
    'margin_mode': 'ISOLATED'
}

INDICATOR_CONFIG = {
    'rsi_period': 21,
    'rsi_overbought': 70,
    'rsi_oversold': 30,
    'ema_fast': 9,
    'ema_slow': 21,
    'ema_trend': 50,
    'ema_regime': 200,
    'macd_fast': 12,
    'macd_slow': 26,
    'macd_signal': 9
}

SIGNAL_CONFIG = {
    'min_conditions': 3,
    'use_trend_filter': True,
    'use_volume_filter': True,
    'prevent_conflicts': True,
    'min_signal_strength': 6,
    'max_signal_strength': 10,
    'require_trend_confirmation': True,
    'min_volume_ratio': 1.0
}

TELEGRAM_CONFIG = {
    'bot_token': os.getenv('TELEGRAM_BOT_TOKEN', ''),
    'chat_id': os.getenv('TELEGRAM_CHAT_ID', '')
}

# ====================== إعداد التسجيل ======================
if ENABLE_LOGGING:
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s | %(levelname)s | %(message)s',
        handlers=[logging.StreamHandler()]
    )
logger = logging.getLogger(__name__) if ENABLE_LOGGING else None

# ====================== إحصائيات النظام ======================
system_stats = {
    "start_time": time.time(),
    "total_scans": 0,
    "total_signals_sent": 0,
    "last_scan_time": None,
    "executor_connected": False,
    "last_signal_time": None
}

# ====================== عميل البوت المنفذ ======================
class ExecutorBotClient:
    """عميل للتواصل مع بوت التنفيذ"""
    
    def __init__(self, base_url: str, api_key: str):
        self.base_url = base_url
        self.api_key = api_key
        self.client = httpx.AsyncClient(timeout=30.0)

    async def send_trade_signal(self, signal_data: Dict[str, Any]) -> bool:
        """إرسال إشارة تداول إلى البوت المنفذ"""
        if not EXECUTE_TRADES:
            safe_log_info("تنفيذ الصفقات معطل في الإعدادات", "executor", "trade")
            return False
            
        try:
            headers = {
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json"
            }
            
            payload = {
                "signal": signal_data,
                "timestamp": time.time(),
                "source": "advanced_crypto_bot",
                "system_stats": system_stats
            }
            
            response = await self.client.post(
                f"{self.base_url}/api/trade/signal",
                json=payload,
                headers=headers
            )
            
            if response.status_code == 200:
                result = response.json()
                safe_log_info(f"✅ تم إرسال إشارة للتنفيذ: {result.get('message', '')}", 
                            signal_data.get('symbol', 'unknown'), "executor")
                system_stats["total_signals_sent"] += 1
                system_stats["last_signal_time"] = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                return True
            else:
                safe_log_error(f"❌ فشل إرسال الإشارة: {response.status_code} - {response.text}", 
                             signal_data.get('symbol', 'unknown'), "executor")
                return False
                
        except Exception as e:
            safe_log_error(f"❌ خطأ في التواصل مع البوت المنفذ: {e}", 
                         signal_data.get('symbol', 'unknown'), "executor")
            return False

    async def health_check(self) -> bool:
        """فحص حالة البوت المنفذ"""
        try:
            response = await self.client.get(f"{self.base_url}/health", timeout=10.0)
            system_stats["executor_connected"] = (response.status_code == 200)
            return response.status_code == 200
        except Exception as e:
            system_stats["executor_connected"] = False
            safe_log_error(f"فحص صحة البوت المنفذ فشل: {e}", "system", "executor")
            return False

    async def close(self):
        await self.client.aclose()

# ====================== الكلاس الرئيسي المطور كماسح للإشارات ======================
class AdvancedCryptoScanner:
    def __init__(self, trade_config, indicator_config, signal_config):
        self.trade_config = trade_config
        self.indicator_config = indicator_config
        self.signal_config = signal_config
        self.data = None
        self.executor_client = ExecutorBotClient(EXECUTOR_BOT_URL, EXECUTOR_BOT_API_KEY)

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, max=10))
    async def fetch_binance_data(self, symbol: str, timeframe: str, days: int = 30):
        """جلب البيانات من Binance Futures"""
        try:
            limit = 500
            all_data = []
            end_time = int(datetime.now().timestamp() * 1000)
            interval_h = interval_to_hours(timeframe)
            required_candles = int(days * 24 / interval_h) + 50

            if ENABLE_LOGGING:
                logger.info(f"جلب {required_candles} شمعة من العقود الآجلة {symbol} ({timeframe})")

            while len(all_data) < required_candles:
                params = {
                    'symbol': symbol,
                    'interval': timeframe,
                    'limit': min(limit, required_candles - len(all_data)),
                    'endTime': end_time
                }
                
                async with httpx.AsyncClient() as client:
                    response = await client.get("https://fapi.binance.com/fapi/v1/klines", params=params, timeout=15)
                    response.raise_for_status()
                    data = response.json()
                
                if not data or len(data) == 0:
                    break
                all_data = data + all_data
                end_time = data[0][0] - 1

            df = pd.DataFrame(all_data, columns=[
                'timestamp', 'open', 'high', 'low', 'close', 'volume',
                'close_time', 'quote_asset_volume', 'number_of_trades',
                'taker_buy_base_asset_volume', 'taker_buy_quote_asset_volume', 'ignore'
            ])

            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
            for col in ['open', 'high', 'low', 'close', 'volume']:
                df[col] = pd.to_numeric(df[col], errors='coerce')

            df = df[['timestamp', 'open', 'high', 'low', 'close', 'volume']]
            df = df.sort_values('timestamp').reset_index(drop=True)
            df = df.drop_duplicates(subset='timestamp')

            self.data = df
            self.calculate_indicators()
            
            if ENABLE_LOGGING:
                logger.info(f"تم جلب {len(self.data)} شمعة من العقود الآجلة بنجاح")
            return True

        except Exception as e:
            if ENABLE_LOGGING:
                logger.error(f"خطأ في جلب بيانات العقود الآجلة: {e}")
            return False

    def calculate_atr(self, period=14):
        """حساب ATR"""
        high = self.data['high']
        low = self.data['low']
        close = self.data['close']
        tr0 = abs(high - low)
        tr1 = abs(high - close.shift())
        tr2 = abs(low - close.shift())
        tr = pd.concat([tr0, tr1, tr2], axis=1).max(axis=1)
        return tr.rolling(period).mean()

    def calculate_indicators(self):
        """حساب المؤشرات الفنية"""
        df = self.data
        p = self.indicator_config
        t = self.trade_config

        # RSI
        delta = df['close'].diff()
        gain = (delta.where(delta > 0, 0)).fillna(0)
        loss = (-delta.where(delta < 0, 0)).fillna(0)
        
        avg_gain = gain.rolling(window=p['rsi_period'], min_periods=1).mean()
        avg_loss = loss.rolling(window=p['rsi_period'], min_periods=1).mean()
        
        rs = avg_gain / avg_loss
        rs = rs.replace([np.inf, -np.inf], 0).fillna(0)
        df['rsi'] = 100 - (100 / (1 + rs))

        # المتوسطات المتحركة
        df['ema_fast'] = df['close'].ewm(span=p['ema_fast'], adjust=False, min_periods=1).mean()
        df['ema_slow'] = df['close'].ewm(span=p['ema_slow'], adjust=False, min_periods=1).mean()
        df['ema_trend'] = df['close'].ewm(span=p['ema_trend'], adjust=False, min_periods=1).mean()
        df['ema_regime'] = df['close'].ewm(span=p['ema_regime'], adjust=False, min_periods=1).mean()

        # MACD
        ema_fast = df['close'].ewm(span=p['macd_fast'], adjust=False, min_periods=1).mean()
        ema_slow = df['close'].ewm(span=p['macd_slow'], adjust=False, min_periods=1).mean()
        df['macd'] = ema_fast - ema_slow
        df['macd_signal'] = df['macd'].ewm(span=p['macd_signal'], adjust=False, min_periods=1).mean()
        df['macd_histogram'] = df['macd'] - df['macd_signal']

        # Volume MA
        df['volume_ma'] = df['volume'].rolling(20, min_periods=1).mean()

        # ATR
        if ENABLE_ATR_SL_TP:
            df['atr'] = self.calculate_atr(t['atr_period'])

        # Support & Resistance
        if ENABLE_SUPPORT_RESISTANCE_FILTER:
            window = t['support_resistance_window']
            df['resistance'] = df['high'].rolling(window).max()
            df['support'] = df['low'].rolling(window).min()

        self.data = df
        if ENABLE_LOGGING:
            logger.info("تم حساب جميع المؤشرات للعقود الآجلة")

    def get_market_regime(self, row):
        """تحديد نظام السوق"""
        if not ENABLE_MARKET_REGIME_FILTER:
            return "NEUTRAL"
        price = row['close']
        ema200 = row['ema_regime']
        if price > ema200 * 1.05:
            return "BULL"
        elif price < ema200 * 0.95:
            return "BEAR"
        else:
            return "SIDEWAYS"

    def calculate_signal_strength(self, buy_conditions, sell_conditions, row):
        """حساب قوة الإشارة المتدرجة من 1-10"""
        try:
            base_conditions = max(buy_conditions, sell_conditions)
            
            if base_conditions == 0:
                return 1
            
            rsi = row['rsi'] if not pd.isna(row['rsi']) else 50
            ema_trend_position = 1 if row['close'] > row['ema_trend'] else 0
            macd_strength = abs(row['macd_histogram']) / row['close'] * 1000 if not pd.isna(row['macd_histogram']) else 0
            volume_strength = min(row['volume'] / row['volume_ma'], 3) if not pd.isna(row['volume_ma']) and row['volume_ma'] > 0 else 1
            
            strength_points = 0
            
            # قوة RSI
            if (buy_conditions > sell_conditions and rsi < 25) or (sell_conditions > buy_conditions and rsi > 75):
                strength_points += 2
            elif (buy_conditions > sell_conditions and rsi < 30) or (sell_conditions > buy_conditions and rsi > 70):
                strength_points += 1
            
            # قوة الاتجاه
            if (buy_conditions > sell_conditions and ema_trend_position == 1) or \
               (sell_conditions > buy_conditions and ema_trend_position == 0):
                strength_points += 1
            
            # قوة MACD
            if macd_strength > 0.8:
                strength_points += 1
            elif macd_strength > 0.5:
                strength_points += 0.5
            
            # قوة الحجم
            if volume_strength > 2.0:
                strength_points += 1.5
            elif volume_strength > 1.5:
                strength_points += 1
            elif volume_strength > 1.2:
                strength_points += 0.5
            
            # نظام السوق
            regime = self.get_market_regime(row)
            if regime == "BULL" and buy_conditions > sell_conditions:
                strength_points += 0.5
            elif regime == "BEAR" and sell_conditions > buy_conditions:
                strength_points += 0.5
            
            total_strength = min(base_conditions + strength_points, 10)
            total_strength = max(total_strength, 1)
            
            return total_strength
            
        except Exception as e:
            if ENABLE_LOGGING:
                logger.error(f"خطأ في حساب قوة الإشارة: {e}")
            return 1

    def generate_signal(self, row):
        """توليد إشارات تداول"""
        try:
            required_columns = ['rsi', 'ema_fast', 'ema_slow', 'macd', 'ema_trend', 'volume_ma']
            if any(pd.isna(row[col]) for col in required_columns):
                return 'HOLD', 1, "بيانات ناقصة"

            buy_conditions = 0
            sell_conditions = 0
            condition_details = []

            # 1. شرط RSI
            if row['rsi'] < self.indicator_config['rsi_oversold']:
                buy_conditions += 1
                condition_details.append("RSI منخفض")
            elif row['rsi'] > self.indicator_config['rsi_overbought']:
                sell_conditions += 1
                condition_details.append("RSI مرتفع")

            # 2. شرط EMA
            if row['ema_fast'] > row['ema_slow']:
                buy_conditions += 1
                condition_details.append("EMA صاعد")
            else:
                sell_conditions += 1
                condition_details.append("EMA هابط")

            # 3. شرط MACD
            macd_strength = abs(row['macd_histogram']) > (row['close'] * 0.001)
            if row['macd'] > row['macd_signal'] and macd_strength:
                buy_conditions += 1
                condition_details.append("MACD صاعد")
            elif row['macd'] < row['macd_signal'] and macd_strength:
                sell_conditions += 1
                condition_details.append("MACD هابط")

            # 4. فلتر الاتجاه
            if self.signal_config['use_trend_filter']:
                if row['close'] > row['ema_trend']:
                    buy_conditions += 1
                    condition_details.append("فوق المتوسط 50")
                else:
                    sell_conditions += 1
                    condition_details.append("تحت المتوسط 50")

            # 5. فلتر الحجم
            volume_ratio = row['volume'] / row['volume_ma'] if row['volume_ma'] > 0 else 1
            volume_ok = volume_ratio > self.signal_config.get('min_volume_ratio', 1.0)

            # حساب قوة الإشارة
            signal_strength = self.calculate_signal_strength(buy_conditions, sell_conditions, row)

            # الفلاتر الإضافية
            regime = self.get_market_regime(row)
            regime_ok = not ENABLE_MARKET_REGIME_FILTER or \
                       (regime != "BEAR" if buy_conditions > sell_conditions else regime != "BULL")

            hour = row['timestamp'].hour
            time_ok = not ENABLE_TIME_FILTER or hour in self.trade_config['peak_hours']

            near_level = False
            if ENABLE_SUPPORT_RESISTANCE_FILTER and 'resistance' in row:
                dist_r = abs(row['close'] - row['resistance']) / row['close']
                dist_s = abs(row['close'] - row['support']) / row['close']
                near_level = min(dist_r, dist_s) < 0.003

            # القرار النهائي
            signal = 'HOLD'
            min_conditions = self.signal_config['min_conditions']
            min_strength = self.signal_config.get('min_signal_strength', 5)
            max_strength = self.signal_config.get('max_signal_strength', 10)

            strength_in_range = min_strength <= signal_strength <= max_strength

            if (buy_conditions >= min_conditions and 
                strength_in_range and
                volume_ok and regime_ok and time_ok and not near_level):
                signal = 'BUY'

            elif (sell_conditions >= min_conditions and 
                  strength_in_range and
                  volume_ok and regime_ok and time_ok and not near_level):
                signal = 'SELL'

            details = " | ".join(condition_details) if condition_details else "لا توجد إشارات قوية"
            
            if signal != 'HOLD':
                details += f" | قوة: {signal_strength:.1f}/10"
            
            return signal, signal_strength, details
            
        except Exception as e:
            if ENABLE_LOGGING:
                logger.error(f"خطأ في توليد الإشارة: {e}")
            return 'HOLD', 1, f"خطأ: {str(e)}"

    async def scan_and_send_signals(self):
        """المسح الضوئي وإرسال الإشارات"""
        try:
            # جلب البيانات
            success = await self.fetch_binance_data(
                self.trade_config['symbol'], 
                self.trade_config['timeframe'], 
                days=30
            )
            
            if not success or self.data is None or len(self.data) < 50:
                if ENABLE_LOGGING:
                    logger.error("فشل في جلب البيانات أو البيانات غير كافية")
                return None

            # الحصول على آخر صف (البيانات الأحدث)
            latest_row = self.data.iloc[-1]
            
            # توليد الإشارة
            signal, strength, details = self.generate_signal(latest_row)
            
            if signal in ['BUY', 'SELL'] and strength >= CONFIDENCE_THRESHOLD:
                # تحضير بيانات الإشارة
                signal_data = {
                    "symbol": self.trade_config['symbol'],
                    "action": signal,
                    "signal_type": "ENTRY",
                    "timeframe": self.trade_config['timeframe'],
                    "price": float(latest_row['close']),
                    "confidence_score": strength * 10,  # تحويل من 1-10 إلى 10-100
                    "reason": details,
                    "analysis": {
                        "rsi": float(latest_row['rsi']),
                        "ema_fast": float(latest_row['ema_fast']),
                        "ema_slow": float(latest_row['ema_slow']),
                        "macd_histogram": float(latest_row['macd_histogram']),
                        "volume_ratio": float(latest_row['volume'] / latest_row['volume_ma']) if latest_row['volume_ma'] > 0 else 1.0,
                        "signal_strength": strength,
                        "market_regime": self.get_market_regime(latest_row)
                    },
                    "timestamp": time.time(),
                    "system_version": "2.0.0"
                }
                
                # إرسال الإشارة إلى البوت المنفذ
                if ENABLE_SIGNAL_SENDING:
                    sent = await self.executor_client.send_trade_signal(signal_data)
                    if sent:
                        if ENABLE_LOGGING:
                            logger.info(f"✅ تم إرسال إشارة {signal} لـ {self.trade_config['symbol']} - قوة: {strength}/10")
                        return signal_data
                    else:
                        if ENABLE_LOGGING:
                            logger.error(f"❌ فشل إرسال الإشارة لـ {self.trade_config['symbol']}")
                else:
                    if ENABLE_LOGGING:
                        logger.info(f"📡 إشارة مكتشفة ولكن الإرسال معطل: {signal} لـ {self.trade_config['symbol']} - قوة: {strength}/10")
                    return signal_data
            else:
                if ENABLE_LOGGING:
                    logger.info(f"⏸️ لا توجد إشارات قوية لـ {self.trade_config['symbol']} - الإشارة: {signal} - القوة: {strength}/10")
                return None
                
        except Exception as e:
            if ENABLE_LOGGING:
                logger.error(f"❌ خطأ في المسح الضوئي: {e}")
            return None

# ====================== تهيئة الماسح ======================
scanner = AdvancedCryptoScanner(TRADE_CONFIG, INDICATOR_CONFIG, SIGNAL_CONFIG)

# ====================== واجهات API ======================
@app.get("/")
async def root():
    return {
        "message": "Crypto Signals Scanner - النظام المتقدم",
        "version": "2.0.0",
        "status": "running",
        "symbol": TRADE_CONFIG['symbol'],
        "timeframe": TRADE_CONFIG['timeframe'],
        "signal_sending_enabled": ENABLE_SIGNAL_SENDING,
        "confidence_threshold": CONFIDENCE_THRESHOLD
    }

@app.get("/health")
async def health_check():
    """فحص صحة النظام"""
    executor_health = await scanner.executor_client.health_check()
    
    return {
        "status": "healthy",
        "executor_connected": executor_health,
        "system_stats": system_stats,
        "timestamp": datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    }

@app.post("/scan")
async def scan_signals():
    """المسح الضوئي اليدوي للإشارات"""
    try:
        signal_data = await scanner.scan_and_send_signals()
        
        if signal_data:
            return {
                "status": "success",
                "signal_found": True,
                "signal_data": signal_data,
                "message": "تم اكتشاف إشارة وإرسالها بنجاح"
            }
        else:
            return {
                "status": "success", 
                "signal_found": False,
                "message": "لا توجد إشارات قوية حالياً"
            }
            
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"خطأ في المسح الضوئي: {str(e)}")

@app.get("/system-stats")
async def get_system_stats():
    """الحصول على إحصائيات النظام"""
    uptime_seconds = time.time() - system_stats["start_time"]
    
    days = int(uptime_seconds // 86400)
    hours = int((uptime_seconds % 86400) // 3600)
    minutes = int((uptime_seconds % 3600) // 60)
    
    if days > 0:
        uptime_str = f"{days} يوم, {hours} ساعة, {minutes} دقيقة"
    elif hours > 0:
        uptime_str = f"{hours} ساعة, {minutes} دقيقة"
    else:
        uptime_str = f"{minutes} دقيقة"
    
    return {
        "system_stats": system_stats,
        "uptime": uptime_str,
        "config": {
            "symbol": TRADE_CONFIG['symbol'],
            "timeframe": TRADE_CONFIG['timeframe'],
            "confidence_threshold": CONFIDENCE_THRESHOLD,
            "scan_interval": SCAN_INTERVAL,
            "signal_sending_enabled": ENABLE_SIGNAL_SENDING,
            "trade_execution_enabled": EXECUTE_TRADES
        },
        "current_time": datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    }

# ====================== المهام الدورية ======================
async def periodic_scanner_task():
    """المهمة الدورية للمسح الضوئي"""
    if ENABLE_LOGGING:
        logger.info("بدء المهمة الدورية للمسح الضوئي للإشارات")
    
    while True:
        try:
            await scanner.scan_and_send_signals()
            system_stats["total_scans"] += 1
            system_stats["last_scan_time"] = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            
            await asyncio.sleep(SCAN_INTERVAL)
            
        except Exception as e:
            if ENABLE_LOGGING:
                logger.error(f"خطأ في المهمة الدورية: {e}")
            await asyncio.sleep(60)

# ====================== التشغيل ======================
@app.on_event("startup")
async def startup_event():
    """حدث بدء التشغيل"""
    if ENABLE_LOGGING:
        logger.info("بدء تشغيل ماسح الإشارات المتقدم")
        logger.info(f"الرمز: {TRADE_CONFIG['symbol']}")
        logger.info(f"الإطار الزمني: {TRADE_CONFIG['timeframe']}")
        logger.info(f"فاصل المسح: {SCAN_INTERVAL} ثانية")
        logger.info(f"عتبة الثقة: {CONFIDENCE_THRESHOLD}")
        logger.info(f"إرسال الإشارات: {'مفعل' if ENABLE_SIGNAL_SENDING else 'معطل'}")
        logger.info(f"تنفيذ الصفقات: {'مفعل' if EXECUTE_TRADES else 'معطل'}")
    
    # بدء المهمة الدورية
    asyncio.create_task(periodic_scanner_task())

@app.on_event("shutdown")
async def shutdown_event():
    """حدث إيقاف التشغيل"""
    if ENABLE_LOGGING:
        logger.info("إيقاف ماسح الإشارات المتقدم")
    await scanner.executor_client.close()

def safe_log_info(message: str, source: str = "app"):
    """تسجيل آمن للمعلومات"""
    try:
        if ENABLE_LOGGING:
            logger.info(f"{message} - Source: {source}")
    except Exception as e:
        print(f"خطأ في التسجيل: {e} - الرسالة: {message}")

def safe_log_error(message: str, source: str = "app"):
    """تسجيل آمن للأخطاء"""
    try:
        if ENABLE_LOGGING:
            logger.error(f"{message} - Source: {source}")
    except Exception as e:
        print(f"خطأ في تسجيل الخطأ: {e} - الرسالة: {message}")

if __name__ == "__main__":
    port = int(os.getenv("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)
