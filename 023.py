# ====================== المكتبات ======================
import os
import pandas as pd
import numpy as np
import requests
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
from typing import Dict, Any, List, Optional

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
ENABLE_TELEGRAM_ALERTS = True  # تفعيل التلغرام

# ====================== إعدادات البوت المنفذ ======================
EXECUTOR_BOT_URL = os.getenv("EXECUTOR_BOT_URL", "https://your-executor-bot.onrender.com")
EXECUTOR_BOT_API_KEY = os.getenv("EXECUTOR_BOT_API_KEY", "")
EXECUTE_TRADES = os.getenv("EXECUTE_TRADES", "false").lower() == "true"

# ====================== إعدادات المسح ======================
SCAN_INTERVAL = 600  # 10 دقائق بين كل فحص
CONFIDENCE_THRESHOLD = 5  # عتبة الثقة للإشارات (من 1-10)

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
    'symbols': ["BNBUSDT", "ETHUSDT", "BTCUSDT", "XRPUSDT"],  # قائمة الرموز للمسح
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
    'min_signal_strength': 5,  # زيادة الحد الأدنى لقوة الإشارة
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
    "last_signal_time": None,
    "signals_by_symbol": {},
    "buy_signals": 0,
    "sell_signals": 0
}

# ====================== إشعارات التلغرام ======================
class TelegramNotifier:
    """إشعارات التلغرام للإشارات المكتشفة"""
    
    def __init__(self, token: str, chat_id: str):
        self.token = token
        self.chat_id = chat_id
        self.base_url = f"https://api.telegram.org/bot{token}"

    async def send_signal_alert(self, signal_data: Dict[str, Any]) -> bool:
        """إرسال تنبيه إشارة عبر التلغرام"""
        if not ENABLE_TELEGRAM_ALERTS or not self.token or not self.chat_id:
            return False
            
        try:
            message = self._build_signal_message(signal_data)
            
            payload = {
                'chat_id': self.chat_id,
                'text': message,
                'parse_mode': 'Markdown',
                'disable_web_page_preview': True
            }
            
            async with httpx.AsyncClient() as client:
                response = await client.post(f"{self.base_url}/sendMessage", 
                                           json=payload, timeout=10.0)
            
            if response.status_code == 200:
                safe_log_info(f"✅ تم إرسال تنبيه التلغرام لإشارة {signal_data['symbol']}", 
                            signal_data['symbol'], "telegram")
                return True
            else:
                safe_log_error(f"❌ فشل إرسال تنبيه التلغرام: {response.status_code}", 
                             signal_data['symbol'], "telegram")
                return False
                
        except Exception as e:
            safe_log_error(f"❌ خطأ في إرسال تنبيه التلغرام: {e}", 
                         signal_data.get('symbol', 'unknown'), "telegram")
            return False

    def _build_signal_message(self, signal_data: Dict[str, Any]) -> str:
        """بناء رسالة الإشارة"""
        symbol = signal_data['symbol']
        action = signal_data['action']
        price = signal_data['price']
        strength = signal_data['analysis']['signal_strength']
        confidence = signal_data['confidence_score']
        reason = signal_data['reason']
        timeframe = signal_data['timeframe']
        
        analysis = signal_data['analysis']
        rsi = analysis.get('rsi', 0)
        ema_fast = analysis.get('ema_fast', 0)
        ema_slow = analysis.get('ema_slow', 0)
        macd_hist = analysis.get('macd_histogram', 0)
        volume_ratio = analysis.get('volume_ratio', 1.0)
        market_regime = analysis.get('market_regime', 'NEUTRAL')
        
        # تحديد الرموز بناءً على نوع الإشارة
        if action == 'BUY':
            action_emoji = "🟢"
            action_text = "شراء"
            action_type = "قاع سعري"
        else:  # SELL
            action_emoji = "🔴" 
            action_text = "بيع"
            action_type = "قمة سعرية"
        
        # تحديد قوة الإشارة
        if strength >= 9:
            strength_emoji = "💥💥💥"
            strength_text = "قوية جداً"
        elif strength >= 8:
            strength_emoji = "💥💥"
            strength_text = "قوية"
        elif strength >= 7:
            strength_emoji = "💥"
            strength_text = "جيدة"
        else:
            strength_emoji = "⚡"
            strength_text = "متوسطة"
        
        # بناء الرسالة
        message = f"""
{action_emoji} **إشارة {action_text} - {symbol}** {action_emoji}

💰 **السعر الحالي:** `${price:,.4f}`
⏰ **الإطار الزمني:** `{timeframe}`
🎯 **نوع الإشارة:** `{action_type}`
📊 **قوة الإشارة:** {strength_emoji} `{strength}/10` ({strength_text})
🔢 **درجة الثقة:** `{confidence}%`

📈 **المؤشرات الفنية:**
• 📊 **RSI:** `{rsi:.2f}` {'(تشبع بيعي)' if rsi < 30 else '(تشبع شرائي)' if rsi > 70 else '(محايد)'}
• 📈 **MACD Hist:** `{macd_hist:.6f}` {'(صاعد)' if macd_hist > 0 else '(هابط)'}
• 📉 **EMA 9/21:** `{ema_fast:.4f}/{ema_slow:.4f}` {'(صاعد)' if ema_fast > ema_slow else '(هابط)'}
• 🔊 **نسبة الحجم:** `{volume_ratio:.2f}x` {'(مرتفع)' if volume_ratio > 1.5 else '(طبيعي)'}
• 🌐 **نظام السوق:** `{market_regime}`

📝 **التفاصيل:** {reason}

⏳ **الوقت:** `{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}`
🔧 **المصدر:** `ماسح الإشارات المتقدم v2.0`

💡 **التوصية:** {'الدخول في صفقة شراء مع إدارة المخاطر' if action == 'BUY' else 'الدخول في صفقة بيع مع إدارة المخاطر'}
        """
        
        return message

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
                
                # تحديث الإحصائيات
                system_stats["total_signals_sent"] += 1
                system_stats["last_signal_time"] = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                
                symbol = signal_data.get('symbol', 'unknown')
                if symbol not in system_stats["signals_by_symbol"]:
                    system_stats["signals_by_symbol"][symbol] = 0
                system_stats["signals_by_symbol"][symbol] += 1
                
                # تحديث إحصائيات البيع/الشراء
                if signal_data.get('action') == 'BUY':
                    system_stats["buy_signals"] += 1
                else:
                    system_stats["sell_signals"] += 1
                
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
        self.data = {}
        self.executor_client = ExecutorBotClient(EXECUTOR_BOT_URL, EXECUTOR_BOT_API_KEY)
        self.telegram_notifier = TelegramNotifier(TELEGRAM_CONFIG['bot_token'], TELEGRAM_CONFIG['chat_id'])

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

            self.data[symbol] = df
            self.calculate_indicators(symbol)
            
            if ENABLE_LOGGING:
                logger.info(f"تم جلب {len(df)} شمعة من العقود الآجلة بنجاح لـ {symbol}")
            return True

        except Exception as e:
            if ENABLE_LOGGING:
                logger.error(f"خطأ في جلب بيانات العقود الآجلة لـ {symbol}: {e}")
            return False

    def calculate_indicators(self, symbol: str):
        """حساب المؤشرات الفنية"""
        df = self.data[symbol]
        p = self.indicator_config

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

        self.data[symbol] = df
        if ENABLE_LOGGING:
            logger.info(f"تم حساب جميع المؤشرات للعقود الآجلة لـ {symbol}")

    def get_market_regime(self, symbol: str, row):
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

    def calculate_signal_strength(self, buy_conditions, sell_conditions, symbol: str, row):
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
            regime = self.get_market_regime(symbol, row)
            if regime == "BULL" and buy_conditions > sell_conditions:
                strength_points += 0.5
            elif regime == "BEAR" and sell_conditions > buy_conditions:
                strength_points += 0.5
            
            total_strength = min(base_conditions + strength_points, 10)
            total_strength = max(total_strength, 1)
            
            return total_strength
            
        except Exception as e:
            if ENABLE_LOGGING:
                logger.error(f"خطأ في حساب قوة الإشارة لـ {symbol}: {e}")
            return 1

    def generate_signal(self, symbol: str, row):
        """توليد إشارات تداول - إصدار محسن"""
        try:
            required_columns = ['rsi', 'ema_fast', 'ema_slow', 'macd', 'ema_trend', 'volume_ma']
            if any(pd.isna(row[col]) for col in required_columns):
                return 'HOLD', 1, "بيانات ناقصة"

            buy_conditions = 0
            sell_conditions = 0
            condition_details = []

            # 1. شرط RSI - محسن
            rsi = row['rsi']
            if rsi < self.indicator_config['rsi_oversold']:
                buy_conditions += 2  # زيادة الوزن
                condition_details.append(f"RSI منخفض ({rsi:.1f})")
            elif rsi > self.indicator_config['rsi_overbought']:
                sell_conditions += 2  # زيادة الوزن
                condition_details.append(f"RSI مرتفع ({rsi:.1f})")
            elif rsi < 35:
                buy_conditions += 1
                condition_details.append(f"RSي قريب من التشبع البيعي ({rsi:.1f})")
            elif rsi > 65:
                sell_conditions += 1
                condition_details.append(f"RSI قريب من التشبع الشرائي ({rsi:.1f})")

            # 2. شرط EMA - محسن
            ema_fast = row['ema_fast']
            ema_slow = row['ema_slow']
            if ema_fast > ema_slow:
                buy_conditions += 2
                condition_details.append("EMA صاعد بقوة")
            else:
                sell_conditions += 2
                condition_details.append("EMA هابط بقوة")

            # 3. شرط MACD - محسن
            macd_histogram = row['macd_histogram']
            macd_strength = abs(macd_histogram) > (row['close'] * 0.001)
            
            if macd_histogram > 0.002 and macd_strength:  # زيادة الحساسية
                buy_conditions += 2
                condition_details.append("MACD صاعد بقوة")
            elif macd_histogram < -0.002 and macd_strength:
                sell_conditions += 2
                condition_details.append("MACD هابط بقوة")
            elif macd_histogram > 0:
                buy_conditions += 1
                condition_details.append("MACD إيجابي")
            else:
                sell_conditions += 1
                condition_details.append("MACD سلبي")

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
            if volume_ratio > 1.5:
                buy_conditions += 1
                sell_conditions += 1
                condition_details.append(f"حجم مرتفع ({volume_ratio:.1f}x)")

            # حساب قوة الإشارة
            signal_strength = self.calculate_signal_strength(buy_conditions, sell_conditions, symbol, row)

            # الفلاتر الإضافية
            regime = self.get_market_regime(symbol, row)
            regime_ok = not ENABLE_MARKET_REGIME_FILTER or \
                       (regime != "BEAR" if buy_conditions > sell_conditions else regime != "BULL")

            hour = row['timestamp'].hour
            time_ok = not ENABLE_TIME_FILTER or hour in self.trade_config['peak_hours']

            near_level = False
            if ENABLE_SUPPORT_RESISTANCE_FILTER:
                # إضافة تحليل الدعم والمقاومة
                if 'resistance' in row and 'support' in row:
                    current_price = row['close']
                    distance_to_resistance = abs(current_price - row['resistance']) / current_price
                    distance_to_support = abs(current_price - row['support']) / current_price
                    
                    if distance_to_resistance < 0.01:  # قريب من المقاومة
                        sell_conditions += 1
                        condition_details.append("قريب من المقاومة")
                    elif distance_to_support < 0.01:  # قريب من الدعم
                        buy_conditions += 1
                        condition_details.append("قريب من الدعم")

            # القرار النهائي - محسن
            signal = 'HOLD'
            min_conditions = self.signal_config['min_conditions']

            # تحديد الإشارة بناءً على الفرق بين الشروط
            condition_diff = buy_conditions - sell_conditions
            
            if condition_diff >= min_conditions and signal_strength >= CONFIDENCE_THRESHOLD and regime_ok and time_ok:
                signal = 'BUY'
            elif condition_diff <= -min_conditions and signal_strength >= CONFIDENCE_THRESHOLD and regime_ok and time_ok:
                signal = 'SELL'

            details = " | ".join(condition_details) if condition_details else "لا توجد إشارات قوية"
            
            if signal != 'HOLD':
                details += f" | قوة: {signal_strength:.1f}/10"
                details += f" | شروط: {buy_conditions}-{sell_conditions}"
            
            return signal, signal_strength, details
            
        except Exception as e:
            if ENABLE_LOGGING:
                logger.error(f"خطأ في توليد الإشارة لـ {symbol}: {e}")
            return 'HOLD', 1, f"خطأ: {str(e)}"

    async def scan_symbol(self, symbol: str):
        """المسح الضوئي لرمز معين"""
        try:
            # جلب البيانات
            success = await self.fetch_binance_data(
                symbol, 
                self.trade_config['timeframe'], 
                days=30
            )
            
            if not success or symbol not in self.data or len(self.data[symbol]) < 50:
                if ENABLE_LOGGING:
                    logger.error(f"فشل في جلب البيانات أو البيانات غير كافية لـ {symbol}")
                return None

            # الحصول على آخر صف (البيانات الأحدث)
            latest_row = self.data[symbol].iloc[-1]
            
            # توليد الإشارة
            signal, strength, details = self.generate_signal(symbol, latest_row)
            
            if signal in ['BUY', 'SELL'] and strength >= CONFIDENCE_THRESHOLD:
                # تحضير بيانات الإشارة
                signal_data = {
                    "symbol": symbol,
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
                        "market_regime": self.get_market_regime(symbol, latest_row)
                    },
                    "timestamp": time.time(),
                    "system_version": "2.0.0"
                }
                
                # إرسال الإشارة إلى البوت المنفذ
                if ENABLE_SIGNAL_SENDING:
                    sent = await self.executor_client.send_trade_signal(signal_data)
                    if sent:
                        if ENABLE_LOGGING:
                            logger.info(f"✅ تم إرسال إشارة {signal} لـ {symbol} - قوة: {strength}/10")
                        
                        # إرسال تنبيه التلغرام
                        await self.telegram_notifier.send_signal_alert(signal_data)
                        
                        return signal_data
                    else:
                        if ENABLE_LOGGING:
                            logger.error(f"❌ فشل إرسال الإشارة لـ {symbol}")
                else:
                    if ENABLE_LOGGING:
                        logger.info(f"📡 إشارة مكتشفة ولكن الإرسال معطل: {signal} لـ {symbol} - قوة: {strength}/10")
                    
                    # إرسال تنبيه التلغرام حتى لو كان الإرسال معطل
                    await self.telegram_notifier.send_signal_alert(signal_data)
                    
                    return signal_data
            else:
                if ENABLE_LOGGING:
                    logger.info(f"⏸️ لا توجد إشارات قوية لـ {symbol} - الإشارة: {signal} - القوة: {strength}/10")
                return None
                
        except Exception as e:
            if ENABLE_LOGGING:
                logger.error(f"❌ خطأ في المسح الضوئي لـ {symbol}: {e}")
            return None

    async def scan_all_symbols(self):
        """المسح الضوئي لجميع الرموز"""
        signals_found = []
        
        for symbol in self.trade_config['symbols']:
            try:
                signal_data = await self.scan_symbol(symbol)
                if signal_data:
                    signals_found.append(signal_data)
                # انتظار بين كل رمز لتجنب حظر API
                await asyncio.sleep(2)
            except Exception as e:
                if ENABLE_LOGGING:
                    logger.error(f"❌ خطأ في معالجة {symbol}: {e}")
                continue
                
        return signals_found

# ====================== تهيئة الماسح ======================
scanner = AdvancedCryptoScanner(TRADE_CONFIG, INDICATOR_CONFIG, SIGNAL_CONFIG)

# ====================== واجهات API ======================
@app.get("/")
async def root():
    return {
        "message": "Crypto Signals Scanner - النظام المتقدم",
        "version": "2.0.0",
        "status": "running",
        "symbols": TRADE_CONFIG['symbols'],
        "timeframe": TRADE_CONFIG['timeframe'],
        "signal_sending_enabled": ENABLE_SIGNAL_SENDING,
        "confidence_threshold": CONFIDENCE_THRESHOLD,
        "telegram_alerts": ENABLE_TELEGRAM_ALERTS
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
        signals_found = await scanner.scan_all_symbols()
        
        if signals_found:
            return {
                "status": "success",
                "signals_found": len(signals_found),
                "signals": signals_found,
                "message": f"تم اكتشاف {len(signals_found)} إشارة وإرسالها بنجاح"
            }
        else:
            return {
                "status": "success", 
                "signals_found": 0,
                "message": "لا توجد إشارات قوية حالياً"
            }
            
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"خطأ في المسح الضوئي: {str(e)}")

@app.post("/scan/{symbol}")
async def scan_single_symbol(symbol: str):
    """المسح الضوئي لرمز معين"""
    try:
        if symbol not in TRADE_CONFIG['symbols']:
            raise HTTPException(status_code=400, detail=f"الرمز {symbol} غير مدعوم")
            
        signal_data = await scanner.scan_symbol(symbol)
        
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
                "message": "لا توجد إشارات قوية حالياً لهذا الرمز"
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
            "symbols": TRADE_CONFIG['symbols'],
            "timeframe": TRADE_CONFIG['timeframe'],
            "confidence_threshold": CONFIDENCE_THRESHOLD,
            "scan_interval": SCAN_INTERVAL,
            "signal_sending_enabled": ENABLE_SIGNAL_SENDING,
            "trade_execution_enabled": EXECUTE_TRADES,
            "telegram_alerts": ENABLE_TELEGRAM_ALERTS
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
            signals_found = await scanner.scan_all_symbols()
            system_stats["total_scans"] += 1
            system_stats["last_scan_time"] = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            
            if signals_found:
                if ENABLE_LOGGING:
                    logger.info(f"✅ اكتمل المسح الدوري - تم العثور على {len(signals_found)} إشارة وإرسالها")
            else:
                if ENABLE_LOGGING:
                    logger.info(f"⏸️ اكتمل المسح الدوري - لا توجد إشارات قوية")
            
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
        logger.info(f"الرموز: {TRADE_CONFIG['symbols']}")
        logger.info(f"الإطار الزمني: {TRADE_CONFIG['timeframe']}")
        logger.info(f"فاصل المسح: {SCAN_INTERVAL} ثانية")
        logger.info(f"عتبة الثقة: {CONFIDENCE_THRESHOLD}")
        logger.info(f"إرسال الإشارات: {'مفعل' if ENABLE_SIGNAL_SENDING else 'معطل'}")
        logger.info(f"تنفيذ الصفقات: {'مفعل' if EXECUTE_TRADES else 'معطل'}")
        logger.info(f"تنبيهات التلغرام: {'مفعل' if ENABLE_TELEGRAM_ALERTS else 'معطل'}")
    
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
