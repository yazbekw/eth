from fastapi import FastAPI, HTTPException
import httpx
import asyncio
import os
import time
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import logging
from typing import Dict, Any, List, Optional
import json
import random
import hmac
import hashlib
import base64

# =============================================================================
# إعدادات البوت المتقدم
# =============================================================================

TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN", "")
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID", "")
EXECUTOR_BOT_URL = os.getenv("EXECUTOR_BOT_URL", "")
EXECUTOR_BOT_API_KEY = os.getenv("EXECUTOR_BOT_API_KEY", "")
EXECUTE_TRADES = os.getenv("EXECUTE_TRADES", "false").lower() == "true"

# مفاتيح CoinEx
COINEX_ACCESS_ID = os.getenv("COINEX_ACCESS_ID", "")
COINEX_SECRET_KEY = os.getenv("COINEX_SECRET_KEY", "")

# إعدادات التداول
SCAN_INTERVAL = 900  # 15 دقيقة بين كل فحص
CONFIDENCE_THRESHOLD_SINGLE = 65  # عتبة الإشارة الواحدة
CONFIDENCE_THRESHOLD_MULTIPLE = 61  # عتبة الإشارات المتعددة
MIN_STRATEGY_CONFIDENCE = 25  # أقل ثقة للاستراتيجيات المحتسبة

# العملات المدعومة
SUPPORTED_COINS = {
    'eth': {'name': 'Ethereum', 'coinex_symbol': 'ETHUSDT', 'binance_symbol': 'ETHUSDT', 'symbol': 'ETH'},
    'bnb': {'name': 'Binance Coin', 'coinex_symbol': 'BNBUSDT', 'binance_symbol': 'BNBUSDT', 'symbol': 'BNB'},
    'btc': {'name': 'Bitcoin', 'coinex_symbol': 'BTCUSDT', 'binance_symbol': 'BTCUSDT', 'symbol': 'BTC'},
    'sol': {'name': 'Solana', 'coinex_symbol': 'SOLUSDT', 'binance_symbol': 'SOLUSDT', 'symbol': 'SOL'},
    'xrp': {'name': 'Ripple', 'coinex_symbol': 'XRPUSDT', 'binance_symbol': 'XRPUSDT', 'symbol': 'XRP'},
    'ada': {'name': 'Cardano', 'coinex_symbol': 'ADAUSDT', 'binance_symbol': 'ADAUSDT', 'symbol': 'ADA'},
    'avax': {'name': 'Avalanche', 'coinex_symbol': 'AVAXUSDT', 'binance_symbol': 'AVAXUSDT', 'symbol': 'AVAX'},
    'dot': {'name': 'Polkadot', 'coinex_symbol': 'DOTUSDT', 'binance_symbol': 'DOTUSDT', 'symbol': 'DOT'},
    'link': {'name': 'Chainlink', 'coinex_symbol': 'LINKUSDT', 'binance_symbol': 'LINKUSDT', 'symbol': 'LINK'},
}

TIMEFRAME = '1h'  # إطار زمني موحد لجميع الاستراتيجيات

# إعداد التسجيل
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger("advanced_signal_generator")

app = FastAPI(title="Advanced Crypto Signal Generator")

# إحصائيات النظام
system_stats = {
    "start_time": time.time(),
    "total_scans": 0,
    "signals_generated": 0,
    "signals_sent": 0,
    "last_heartbeat": None,
    "data_source_stats": {
        "coinex": {"success": 0, "failed": 0},
        "binance": {"success": 0, "failed": 0}
    },
    "strategies_performance": {
        "ema_rsi_macd": {"calls": 0, "signals": 0},
        "volume_divergence": {"calls": 0, "signals": 0},
        "smart_money": {"calls": 0, "signals": 0}
    },
    "conflict_penalties_applied": 0,
    "trend_alignment_applied": 0,
    "enhanced_signals_sent": 0,
    "detailed_reports_sent": 0
}

# =============================================================================
# مصادر البيانات المتعددة المحسنة مع CoinEx كمصدر رئيسي
# =============================================================================

class EnhancedMultiSourceDataFetcher:
    """مصادر متعددة محسنة مع إدارة ذكية للطلبات - CoinEx كمصدر رئيسي"""
    
    def __init__(self):
        self.client = httpx.AsyncClient(timeout=30.0)
        self.request_times = {}
        self.min_request_interval = 2.0
        
    async def get_coinex_data(self, symbol: str, interval: str, limit: int = 100) -> Optional[Dict]:
        """جلب البيانات من CoinEx كمصدر رئيسي"""
        try:
            current_time = time.time()
            last_request = self.request_times.get('coinex', 0)
            if current_time - last_request < self.min_request_interval:
                wait_time = self.min_request_interval - (current_time - last_request)
                await asyncio.sleep(wait_time)
            
            # تحويل الإطار الزمني ليتوافق مع CoinEx بشكل صحيح
            interval_mapping = {
                '1m': '1min',
                '5m': '5min',
                '15m': '15min',
                '30m': '30min',
                '1h': '1hour',
                '4h': '4hour',
                '1d': '1day',
                '1w': '1week'
            }
            
            coinex_interval = interval_mapping.get(interval, '1hour')
            
            # بناء URL بشكل صحيح لـ CoinEx
            url = f"https://api.coinex.com/v1/market/kline"
            params = {
                'market': symbol,
                'type': coinex_interval,
                'limit': limit
            }
            
            headers = {
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36',
                'Accept': 'application/json',
            }
            
            logger.info(f"🔍 جلب البيانات من CoinEx لـ {symbol} بالإطار {coinex_interval}...")
            
            # استخدام params بدلاً من إضافتها مباشرة في URL
            response = await self.client.get(url, params=params, headers=headers)
            self.request_times['coinex'] = time.time()
            
            if response.status_code == 200:
                data = response.json()
                
                if data.get('code') == 0 and data.get('data'):
                    kline_data = data['data']
                    if len(kline_data) > 0:
                        system_stats["data_source_stats"]["coinex"]["success"] += 1
                        logger.info(f"✅ نجح جلب البيانات من CoinEx لـ {symbol} ({len(kline_data)} شمعة)")
                        return kline_data
                else:
                    error_msg = data.get('message', 'Unknown error')
                    logger.warning(f"⚠️ CoinEx returned error: {error_msg}")
            else:
                logger.warning(f"⚠️ فشل جلب البيانات من CoinEx لـ {symbol}: {response.status_code} - {response.text}")
            
            system_stats["data_source_stats"]["coinex"]["failed"] += 1
            return None
            
        except Exception as e:
            system_stats["data_source_stats"]["coinex"]["failed"] += 1
            logger.error(f"❌ خطأ في جلب البيانات من CoinEx لـ {symbol}: {e}")
            return None
    
    async def get_binance_data(self, symbol: str, interval: str, limit: int = 100) -> Optional[Dict]:
        """جلب البيانات من Binance كمصدر احتياطي"""
        try:
            current_time = time.time()
            last_request = self.request_times.get('binance', 0)
            if current_time - last_request < self.min_request_interval:
                wait_time = self.min_request_interval - (current_time - last_request)
                await asyncio.sleep(wait_time)
            
            # تحويل الإطار الزمني ليتوافق مع Binance
            interval_mapping = {
                '1m': '1m',
                '5m': '5m',
                '15m': '15m',
                '30m': '30m',
                '1h': '1h',
                '4h': '4h',
                '1d': '1d',
                '1w': '1w'
            }
            
            binance_interval = interval_mapping.get(interval, '1h')
            
            url = f"https://api.binance.com/api/v3/klines"
            params = {
                'symbol': symbol,
                'interval': binance_interval,
                'limit': limit
            }
            
            headers = {
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36',
                'Accept': 'application/json',
            }
            
            logger.info(f"🔍 جلب البيانات من Binance لـ {symbol}...")
            response = await self.client.get(url, params=params, headers=headers)
            self.request_times['binance'] = time.time()
            
            if response.status_code == 200:
                data = response.json()
                if data and len(data) > 0:
                    system_stats["data_source_stats"]["binance"]["success"] += 1
                    logger.info(f"✅ نجح جلب البيانات من Binance لـ {symbol} ({len(data)} شمعة)")
                    return data
            
            system_stats["data_source_stats"]["binance"]["failed"] += 1
            if response.status_code == 418:
                wait_time = random.uniform(10, 20)
                logger.warning(f"⏳ تم حظر الطلبات من Binance، انتظار {wait_time:.1f} ثانية...")
                await asyncio.sleep(wait_time)
            
            return None
            
        except Exception as e:
            system_stats["data_source_stats"]["binance"]["failed"] += 1
            logger.error(f"❌ خطأ في جلب البيانات من Binance لـ {symbol}: {e}")
            return None
    
    def _process_coinex_kline_data(self, kline_data: List) -> List:
        """معالجة بيانات Kline من CoinEx لتصبح متوافقة مع النظام"""
        processed_data = []
        for kline in kline_data:
            # تنسيق بيانات CoinEx: [time, open, close, high, low, volume, amount]
            # نحتاج لتحويلها إلى تنسيق مشابه لـ Binance
            try:
                processed_data.append([
                    int(kline[0]) * 1000,  # timestamp (convert to ms)
                    str(kline[1]),         # open
                    str(kline[3]),         # high
                    str(kline[4]),         # low
                    str(kline[2]),         # close
                    str(kline[5]),         # volume
                    int(kline[0]) * 1000,  # close time
                    str(kline[5]),         # quote asset volume (same as volume)
                    "1",                   # number of trades
                    str(kline[6]),         # taker buy base asset volume (amount)
                    str(kline[6]),         # taker buy quote asset volume
                    "0"                    # ignore
                ])
            except (IndexError, ValueError) as e:
                logger.warning(f"⚠️ خطأ في معالجة بيانات CoinEx: {kline} - {e}")
                continue
        
        return processed_data
    
    def _process_binance_kline_data(self, kline_data: List) -> List:
        """معالجة بيانات Kline من Binance لتصبح متوافقة مع النظام"""
        processed_data = []
        for kline in kline_data:
            try:
                # تنسيق بيانات Binance كما هو
                processed_data.append([
                    kline[0],    # open time
                    kline[1],    # open
                    kline[2],    # high
                    kline[3],    # low
                    kline[4],    # close
                    kline[5],    # volume
                    kline[6],    # close time
                    kline[7],    # quote asset volume
                    kline[8],    # number of trades
                    kline[9],    # taker buy base asset volume
                    kline[10],   # taker buy quote asset volume
                    kline[11]    # ignore
                ])
            except (IndexError, ValueError) as e:
                logger.warning(f"⚠️ خطأ في معالجة بيانات Binance: {kline} - {e}")
                continue
        
        return processed_data
    
    async def get_coin_data(self, symbol: str, interval: str) -> Dict[str, Any]:
        """جلب البيانات من مصادر متعددة مع CoinEx كمصدر رئيسي وBinance كاحتياطي"""
        logger.info(f"🔍 بدء جلب بيانات {symbol} من CoinEx (رئيسي) وBinance (احتياطي)...")
        
        # المحاولة مع CoinEx أولاً
        coinex_symbol = symbol
        sources = [
            ("coinex", self.get_coinex_data, coinex_symbol),
            ("binance", self.get_binance_data, symbol),
        ]
        
        for source_name, source_func, source_symbol in sources:
            try:
                logger.info(f"🔄 المحاولة مع {source_name} لـ {source_symbol}...")
                data = await source_func(source_symbol, interval)
                
                if data is not None and len(data) >= 20:
                    # معالجة البيانات حسب المصدر
                    if source_name == "coinex":
                        processed_data = self._process_coinex_kline_data(data)
                    else:
                        processed_data = self._process_binance_kline_data(data)
                    
                    if len(processed_data) < 20:
                        logger.warning(f"⚠️ بيانات غير كافية من {source_name} بعد المعالجة: {len(processed_data)}")
                        continue
                    
                    prices = [float(item[4]) for item in processed_data]
                    volumes = [float(item[5]) for item in processed_data]
                    
                    logger.info(f"✅ نجح جلب البيانات من {source_name} لـ {symbol} - {len(prices)} سعر")
                    return {
                        "prices": prices,
                        "volumes": volumes, 
                        "data_source": source_name,
                        "success": True
                    }
                else:
                    logger.warning(f"⚠️ {source_name} لم يعيد بيانات كافية لـ {symbol}")
                
                await asyncio.sleep(1)
                
            except Exception as e:
                logger.error(f"❌ خطأ في {source_name} لـ {symbol}: {e}")
                await asyncio.sleep(1)
                continue
        
        logger.error(f"❌ فشل جميع مصادر البيانات لـ {symbol}")
        return {
            "signal": "none", 
            "confidence": 0, 
            "reasons": ["فشل جلب البيانات من جميع المصادر"],
            "data_source": "none",
            "success": False
        }

# =============================================================================
# الاستراتيجيات المحسنة مع تفاصيل التحليل
# =============================================================================

class EmaRsiMacdStrategy:
    """الاستراتيجية الأساسية: المتوسطات + RSI + MACD"""
    
    def __init__(self):
        self.name = "ema_rsi_macd"
    
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
        
        ma_order = "صاعد" if ema_9 > ema_21 > ema_50 else "هابط" if ema_9 < ema_21 < ema_50 else "متذبذب"
        
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
        """توليد إشارة تداول مع تفاصيل التحليل"""
        system_stats["strategies_performance"][self.name]["calls"] += 1
        
        if len(prices) < 20:
            return {
                "signal": "none", 
                "confidence": 0, 
                "reasons": ["بيانات غير كافية"],
                "analysis_details": {"error": "بيانات غير كافية"}
            }
        
        current_price = prices[-1]
        rsi = self.calculate_rsi(prices)
        macd = self.calculate_macd(prices)
        trend = self.analyze_trend(prices, current_price)
        
        confidence_score = 0
        reasons = []
        analysis_details = {
            "rsi_value": rsi,
            "macd_histogram": macd['histogram'],
            "macd_line": macd['macd'],
            "macd_signal": macd['signal'],
            "ema_trend": trend['order'],
            "trend_strength": trend['strength'],
            "ema_9": trend['ema_9'],
            "ema_21": trend['ema_21'],
            "ema_50": trend['ema_50'],
            "price_vs_ema_21": "فوق" if trend['price_above_21'] else "تحت",
            "price_vs_ema_50": "فوق" if trend['price_above_50'] else "تحت"
        }
        
        # تحليل المتوسطات
        ma_score = 0
        if trend['order'] == "صاعد" and trend['price_above_21'] and trend['price_above_50']:
            ma_score = trend['strength'] * 4
            reasons.append(f"المتوسطات صاعدة (قوة: {trend['strength']}/10)")
            analysis_details["ma_signal"] = "صاعد"
            analysis_details["ma_score"] = ma_score
        elif trend['order'] == "هابط" and not trend['price_above_21'] and not trend['price_above_50']:
            ma_score = trend['strength'] * 4
            reasons.append(f"المتوسطات هابطة (قوة: {trend['strength']}/10)")
            analysis_details["ma_signal"] = "هابط"
            analysis_details["ma_score"] = ma_score
        else:
            analysis_details["ma_signal"] = "محايد"
            analysis_details["ma_score"] = 0
        
        # تحليل RSI
        rsi_score = 0
        if 40 <= rsi <= 65:
            distance_from_50 = abs(rsi - 50)
            rsi_score = max(0, 30 - (distance_from_50 * 1.5))
            reasons.append(f"RSI في منطقة مناسبة: {rsi}")
            analysis_details["rsi_signal"] = "متعادل"
            analysis_details["rsi_score"] = rsi_score
        elif 35 <= rsi <= 60:
            distance_from_50 = abs(rsi - 50)
            rsi_score = max(0, 30 - (distance_from_50 * 1.5))
            reasons.append(f"RSI في منطقة مناسبة: {rsi}")
            analysis_details["rsi_signal"] = "متعادل"
            analysis_details["rsi_score"] = rsi_score
        elif rsi < 30:
            analysis_details["rsi_signal"] = "تشبع بيع"
            analysis_details["rsi_score"] = 0
        elif rsi > 70:
            analysis_details["rsi_signal"] = "تشبع شراء"
            analysis_details["rsi_score"] = 0
        else:
            analysis_details["rsi_signal"] = "محايد"
            analysis_details["rsi_score"] = 0
        
        # تحليل MACD
        macd_score = 0
        if macd['histogram'] > 0 and macd['macd'] > macd['signal']:
            macd_score = min(30, abs(macd['histogram']) * 1000)
            reasons.append(f"MACD إيجابي: {macd['histogram']:.4f}")
            analysis_details["macd_signal"] = "صاعد"
            analysis_details["macd_score"] = macd_score
        elif macd['histogram'] < 0 and macd['macd'] < macd['signal']:
            macd_score = min(30, abs(macd['histogram']) * 1000)
            reasons.append(f"MACD سلبي: {macd['histogram']:.4f}")
            analysis_details["macd_signal"] = "هابط"
            analysis_details["macd_score"] = macd_score
        else:
            analysis_details["macd_signal"] = "محايد"
            analysis_details["macd_score"] = 0
        
        confidence_score = ma_score + rsi_score + macd_score
        analysis_details["total_score"] = confidence_score
        
        signal_type = "none"
        if confidence_score >= 40:
            if trend['order'] == "صاعد" and 40 <= rsi <= 65 and macd['histogram'] > 0:
                signal_type = "BUY"
            elif trend['order'] == "هابط" and 35 <= rsi <= 60 and macd['histogram'] < 0:
                signal_type = "SELL"
        
        if signal_type != "none":
            system_stats["strategies_performance"][self.name]["signals"] += 1
        
        return {
            "signal": signal_type,
            "confidence": round(confidence_score),
            "price": current_price,
            "reasons": reasons,
            "timestamp": time.time(),
            "analysis_details": analysis_details
        }

class VolumeDivergenceStrategy:
    """استراتيجية الانزياح بين السعر والحجم"""
    
    def __init__(self):
        self.name = "volume_divergence"
    
    @staticmethod
    def calculate_divergence(prices: List[float], volumes: List[float], 
                           lookback_period: int = 20) -> Dict[str, Any]:
        """حساب الانزياح بين حركة السعر والحجم"""
        if len(prices) < lookback_period * 2:
            return {"divergence": "none", "strength": 0}
        
        recent_prices = prices[-lookback_period:]
        older_prices = prices[-lookback_period*2:-lookback_period]
        
        price_trend_recent = (recent_prices[-1] - recent_prices[0]) / recent_prices[0]
        price_trend_older = (older_prices[-1] - older_prices[0]) / older_prices[0]
        
        recent_volumes = volumes[-lookback_period:]
        older_volumes = volumes[-lookback_period*2:-lookback_period]
        
        volume_trend_recent = (recent_volumes[-1] - np.mean(recent_volumes)) / np.mean(recent_volumes)
        volume_trend_older = (older_volumes[-1] - np.mean(older_volumes)) / np.mean(older_volumes)
        
        # كشف الانزياح الإيجابي
        if (price_trend_recent < -0.03 and price_trend_older < -0.03 and
            volume_trend_recent > -0.2 and volume_trend_older < -0.3):
            strength = min(80, int(abs(price_trend_recent) * 1000 + abs(volume_trend_recent) * 100))
            return {"divergence": "positive_bullish", "strength": strength}
        
        # كشف الانزياح السلبي
        elif (price_trend_recent > 0.03 and price_trend_older > 0.03 and
              volume_trend_recent < 0.2 and volume_trend_older > 0.3):
            strength = min(80, int(abs(price_trend_recent) * 1000 + abs(volume_trend_recent) * 100))
            return {"divergence": "negative_bearish", "strength": strength}
        
        # كشف التأكيد الحجمي
        elif ((price_trend_recent > 0.02 and volume_trend_recent > 0.3) or
              (price_trend_recent < -0.02 and volume_trend_recent > 0.3)):
            strength = min(70, int(abs(price_trend_recent) * 800 + volume_trend_recent * 50))
            return {"divergence": "volume_confirmation", "strength": strength}
        
        return {"divergence": "none", "strength": 0}
    
    def generate_signal(self, prices: List[float], volumes: List[float]) -> Dict[str, Any]:
        """توليد إشارة بناء على الانزياح الحجمي السعري مع تفاصيل التحليل"""
        system_stats["strategies_performance"][self.name]["calls"] += 1
        
        if len(prices) < 40:
            return {
                "signal": "none", 
                "confidence": 0, 
                "reasons": ["بيانات غير كافية"],
                "analysis_details": {"error": "بيانات غير كافية"}
            }
        
        current_price = prices[-1]
        divergence_data = self.calculate_divergence(prices, volumes)
        
        confidence_score = 0
        signal_type = "none"
        reasons = []
        
        analysis_details = {
            "divergence_type": divergence_data["divergence"],
            "divergence_strength": divergence_data["strength"],
            "price_change_recent": (prices[-1] - prices[-20]) / prices[-20] * 100,
            "volume_change_recent": (volumes[-1] - np.mean(volumes[-20:])) / np.mean(volumes[-20:]) * 100,
            "volume_avg": np.mean(volumes[-20:])
        }
        
        if divergence_data["divergence"] == "positive_bullish":
            confidence_score = divergence_data["strength"]
            signal_type = "BUY"
            reasons = [
                "انزياح إيجابي: هبوط الأسعار مع ضعف حجم البيع",
                "تشير إلى استنفاد البائعين واستعداد للارتداد"
            ]
            analysis_details["signal_reason"] = "انزياح إيجابي - استنفاد البائعين"
        
        elif divergence_data["divergence"] == "negative_bearish":
            confidence_score = divergence_data["strength"]
            signal_type = "SELL"
            reasons = [
                "انزياح سلبي: صعود الأسعار مع ضعف حجم الشراء",
                "تشير إلى استنفاد المشترين واستعداد للهبوط"
            ]
            analysis_details["signal_reason"] = "انزياح سلبي - استنفاد المشترين"
        
        elif divergence_data["divergence"] == "volume_confirmation":
            price_trend = "صاعد" if prices[-1] > prices[-10] else "هابط"
            
            if price_trend == "صاعد":
                confidence_score = divergence_data["strength"]
                signal_type = "BUY"
                reasons = [
                    "تأكيد حجمي قوي للصعود",
                    "حجم الشراء يدعم استمرار الاتجاه الصاعد"
                ]
                analysis_details["signal_reason"] = "تأكيد حجمي صاعد"
            else:
                confidence_score = divergence_data["strength"]
                signal_type = "SELL"
                reasons = [
                    "تأكيد حجمي قوي للهبوط", 
                    "حجم البيع يدعم استمرار الاتجاه الهابط"
                ]
                analysis_details["signal_reason"] = "تأكيد حجمي هابط"
        else:
            analysis_details["signal_reason"] = "لا يوجد انزياح ملحوظ"
        
        if signal_type != "none" and confidence_score >= 40:
            system_stats["strategies_performance"][self.name]["signals"] += 1
        
        return {
            "signal": signal_type,
            "confidence": confidence_score,
            "price": current_price,
            "reasons": reasons,
            "timestamp": time.time(),
            "analysis_details": analysis_details
        }

class SmartMoneyStrategy:
    """استراتيجية تراكم وتوزيع ذكية"""
    
    def __init__(self):
        self.name = "smart_money"
    
    @staticmethod
    def detect_smart_money_patterns(prices: List[float], volumes: List[float], 
                                  window: int = 10) -> Dict[str, Any]:
        """كشف أنماط الأموال الذكية"""
        if len(prices) < window * 2:
            return {"pattern": "unknown", "confidence": 0}
        
        price_change = (prices[-1] - prices[-window]) / prices[-window]
        volume_change = (volumes[-1] - np.mean(volumes[-window*2:-window])) / np.mean(volumes[-window*2:-window])
        
        if price_change < -0.02 and volume_change > 0.5:
            return {"pattern": "accumulation", "confidence": min(80, int(volume_change * 30))}
        
        elif price_change > 0.02 and volume_change > 0.5:
            return {"pattern": "distribution", "confidence": min(80, int(volume_change * 30))}
        
        elif abs(price_change) < 0.01 and volume_change > 1.0:
            return {"pattern": "absorption", "confidence": min(70, int(volume_change * 25))}
        
        return {"pattern": "no_pattern", "confidence": 0}
    
    def generate_signal(self, prices: List[float], volumes: List[float]) -> Dict[str, Any]:
        """توليد إشارة بناء على تحليل التراكم والتوزيع مع تفاصيل التحليل"""
        system_stats["strategies_performance"][self.name]["calls"] += 1
        
        if len(prices) < 40:
            return {
                "signal": "none", 
                "confidence": 0, 
                "reasons": ["بيانات غير كافية"],
                "analysis_details": {"error": "بيانات غير كافية"}
            }
        
        current_price = prices[-1]
        smart_pattern = self.detect_smart_money_patterns(prices, volumes)
        
        volume_ma_20 = np.mean(volumes[-20:])
        current_volume = volumes[-1]
        volume_ratio = current_volume / volume_ma_20 if volume_ma_20 > 0 else 1
        
        confidence_score = 0
        signal_type = "none"
        reasons = []
        
        analysis_details = {
            "smart_pattern": smart_pattern["pattern"],
            "pattern_confidence": smart_pattern["confidence"],
            "volume_ratio": volume_ratio,
            "price_change_10": (prices[-1] - prices[-10]) / prices[-10] * 100,
            "volume_change": (volumes[-1] - volume_ma_20) / volume_ma_20 * 100,
            "current_volume": current_volume,
            "avg_volume_20": volume_ma_20
        }
        
        if (smart_pattern["pattern"] == "accumulation" and 
            volume_ratio > 1.5 and
            smart_pattern["confidence"] > 40):
            
            confidence_score = smart_pattern["confidence"]
            signal_type = "BUY"
            reasons = [
                "نمط تراكم الأموال الذكية",
                "حجم التداول مرتفع يشير إلى تراكم"
            ]
            analysis_details["signal_reason"] = "تراكم الأموال الذكية"
        
        elif (smart_pattern["pattern"] == "distribution" and 
              volume_ratio > 1.5 and
              smart_pattern["confidence"] > 40):
            
            confidence_score = smart_pattern["confidence"]
            signal_type = "SELL"
            reasons = [
                "نمط توزيع الأموال الذكية",
                "حجم التداول مرتفع يشير إلى توزيع"
            ]
            analysis_details["signal_reason"] = "توزيع الأموال الذكية"
        
        elif (smart_pattern["pattern"] == "absorption" and 
              volume_ratio > 2.0 and
              smart_pattern["confidence"] > 40):
            
            price_trend = "صاعد" if prices[-1] > prices[-20] else "هابط"
            
            if price_trend == "صاعد":
                confidence_score = smart_pattern["confidence"]
                signal_type = "BUY"
                reasons = [
                    "امتصاص بيع قوي من قبل المشترين الأقوياء",
                    "حجم امتصاص مرتفع"
                ]
                analysis_details["signal_reason"] = "امتصاص شراء"
            else:
                confidence_score = smart_pattern["confidence"]
                signal_type = "SELL" 
                reasons = [
                    "امتصاص شراء قوي من قبل البائعين الأقوياء",
                    "حجم امتصاص مرتفع"
                ]
                analysis_details["signal_reason"] = "امتصاص بيع"
        else:
            analysis_details["signal_reason"] = "لا نمط ذكي واضح"
        
        if signal_type != "none" and confidence_score >= 40:
            system_stats["strategies_performance"][self.name]["signals"] += 1
        
        return {
            "signal": signal_type,
            "confidence": confidence_score,
            "price": current_price,
            "reasons": reasons,
            "timestamp": time.time(),
            "analysis_details": analysis_details
        }

# =============================================================================
# محرك الإشارات المتقدم مع نظام التقارير التحليلية التفصيلية
# =============================================================================

class AdvancedSignalEngine:
    """محرك الإشارات المتقدم الذي يدير الاستراتيجيات الثلاث"""
    
    def __init__(self):
        self.strategies = {
            "ema_rsi_macd": EmaRsiMacdStrategy(),
            "volume_divergence": VolumeDivergenceStrategy(),
            "smart_money": SmartMoneyStrategy()
        }
        self.data_fetcher = EnhancedMultiSourceDataFetcher()
    
    def process_strategy_signals(self, strategy_signals: Dict[str, Dict]) -> Dict[str, Any]:
        """معالجة إشارات الاستراتيجيات وتطبيق قواعد الدمج"""
        
        valid_signals = {}
        for strategy_name, signal in strategy_signals.items():
            if signal["signal"] != "none" and signal["confidence"] >= MIN_STRATEGY_CONFIDENCE:
                valid_signals[strategy_name] = signal
        
        if not valid_signals:
            return {"signal": "none", "confidence": 0, "reasons": ["لا توجد إشارات قوية"]}
        
        signals_list = list(valid_signals.values())
        buy_signals = [s for s in signals_list if s["signal"] == "BUY"]
        sell_signals = [s for s in signals_list if s["signal"] == "SELL"]
        
        if buy_signals and sell_signals:
            system_stats["conflict_penalties_applied"] += 1
            return {"signal": "none", "confidence": 0, "reasons": ["تضارب في الإشارات - تم الإلغاء"]}
        
        final_signal = "BUY" if buy_signals else "SELL" if sell_signals else "none"
        
        if final_signal == "none":
            return {"signal": "none", "confidence": 0, "reasons": ["لا توجد إشارات واضحة"]}
        
        active_signals = buy_signals if final_signal == "BUY" else sell_signals
        confidences = [s["confidence"] for s in active_signals]
        
        if len(active_signals) == 1:
            if confidences[0] >= CONFIDENCE_THRESHOLD_SINGLE:
                final_confidence = confidences[0]
            else:
                return {"signal": "none", "confidence": 0, "reasons": [f"إشارة واحدة ضعيفة ({confidences[0]}%)"]}
        else:
            avg_confidence = sum(confidences) / len(confidences)
            if avg_confidence >= CONFIDENCE_THRESHOLD_MULTIPLE:
                final_confidence = avg_confidence
            else:
                return {"signal": "none", "confidence": 0, "reasons": [f"متوسط الثقة ضعيف ({avg_confidence:.1f}%)"]}
        
        all_reasons = []
        for strategy_name, signal in valid_signals.items():
            if signal["signal"] == final_signal:
                all_reasons.extend(signal["reasons"])
        
        system_stats["trend_alignment_applied"] += 1
        system_stats["enhanced_signals_sent"] += 1
        
        return {
            "signal": final_signal,
            "confidence": round(final_confidence, 1),
            "price": active_signals[0]["price"],
            "strategies_analysis": strategy_signals,
            "winning_strategies": len(active_signals),
            "total_strategies": len(self.strategies),
            "reasons": all_reasons,
            "timestamp": time.time()
        }
    
    async def analyze_coin(self, coin_key: str, binance_symbol: str) -> Dict[str, Any]:
        """تحليل عملة باستخدام جميع الاستراتيجيات"""
        strategy_signals = {}
        
        # استخدام رمز CoinEx للعملة
        coin_data = SUPPORTED_COINS.get(coin_key, {})
        coinex_symbol = coin_data.get('coinex_symbol', binance_symbol)
        
        data_result = await self.data_fetcher.get_coin_data(coinex_symbol, TIMEFRAME)
        
        if not data_result.get("success", False):
            return {
                "signal": "none",
                "confidence": 0,
                "reasons": data_result.get("reasons", ["فشل جلب البيانات"]),
                "data_source": data_result.get("data_source", "none"),
                "strategies_analysis": {},
                "success": False
            }
        
        prices = data_result["prices"]
        volumes = data_result["volumes"]
        data_source = data_result["data_source"]
        
        logger.info(f"📊 تطبيق الاستراتيجيات على {coin_key} ({len(prices)} سعر) من {data_source}")
        
        for strategy_name, strategy in self.strategies.items():
            try:
                signal = strategy.generate_signal(prices, volumes)
                strategy_signals[strategy_name] = signal
                if signal['signal'] != 'none':
                    logger.info(f"📈 {strategy_name} لـ {coin_key}: {signal['signal']} ({signal['confidence']}%)")
            except Exception as e:
                logger.error(f"خطأ في استراتيجية {strategy_name} لـ {coin_key}: {e}")
                strategy_signals[strategy_name] = {
                    "signal": "none", 
                    "confidence": 0, 
                    "reasons": [f"خطأ: {str(e)}"],
                    "analysis_details": {"error": str(e)}
                }
        
        final_signal = self.process_strategy_signals(strategy_signals)
        final_signal["coin_key"] = coin_key
        final_signal["data_source"] = data_source
        final_signal["current_price"] = prices[-1] if prices else 0
        final_signal["prices_count"] = len(prices)
        final_signal["success"] = True
        
        return final_signal

    async def generate_strategy_analysis_report(self) -> Dict[str, Any]:
        """توليد تقرير تحليلي تفصيلي يظهر كيفية توليد الإشارات لكل استراتيجية"""
        logger.info("🔍 بدء توليد التقرير التحليلي التفصيلي...")
        
        analysis_report = {
            "timestamp": datetime.now().isoformat(),
            "total_coins": len(SUPPORTED_COINS),
            "timeframe": TIMEFRAME,
            "coin_analysis": {},
            "strategies_summary": {
                "ema_rsi_macd": {"total_signals": 0, "buy_signals": 0, "sell_signals": 0},
                "volume_divergence": {"total_signals": 0, "buy_signals": 0, "sell_signals": 0},
                "smart_money": {"total_signals": 0, "buy_signals": 0, "sell_signals": 0}
            }
        }
        
        # تحليل كل عملة
        for coin_key, coin_data in SUPPORTED_COINS.items():
            try:
                logger.info(f"🔍 تحليل {coin_key} للتقرير التحليلي...")
                analysis = await self.analyze_coin(coin_key, coin_data['binance_symbol'])
                
                if analysis.get('success'):
                    coin_analysis = {
                        "coin_name": coin_data['name'],
                        "current_price": analysis.get('current_price', 0),
                        "data_source": analysis.get('data_source', 'unknown'),
                        "final_signal": analysis.get('signal', 'none'),
                        "final_confidence": analysis.get('confidence', 0),
                        "strategies": {}
                    }
                    
                    # تفاصيل كل استراتيجية
                    strategies_analysis = analysis.get('strategies_analysis', {})
                    for strategy_name, strategy_data in strategies_analysis.items():
                        coin_analysis["strategies"][strategy_name] = {
                            "signal": strategy_data.get('signal', 'none'),
                            "confidence": strategy_data.get('confidence', 0),
                            "reasons": strategy_data.get('reasons', []),
                            "analysis_details": strategy_data.get('analysis_details', {})
                        }
                        
                        # تحديث إحصائيات الاستراتيجيات
                        if strategy_data.get('signal') != 'none':
                            analysis_report["strategies_summary"][strategy_name]["total_signals"] += 1
                            if strategy_data.get('signal') == 'BUY':
                                analysis_report["strategies_summary"][strategy_name]["buy_signals"] += 1
                            elif strategy_data.get('signal') == 'SELL':
                                analysis_report["strategies_summary"][strategy_name]["sell_signals"] += 1
                    
                    analysis_report["coin_analysis"][coin_key] = coin_analysis
                        
            except Exception as e:
                logger.error(f"❌ خطأ في تحليل {coin_key} للتقرير: {e}")
                analysis_report["coin_analysis"][coin_key] = {
                    "error": str(e),
                    "success": False
                }
        
        logger.info("✅ تم إنشاء التقرير التحليلي التفصيلي بنجاح")
        return analysis_report

# =============================================================================
# نظام التقارير التحليلية التفصيلية
# =============================================================================

class StrategyAnalysisReportGenerator:
    """مولد التقارير التحليلية التفصيلية"""
    
    def __init__(self, notifier):
        self.notifier = notifier
    
    def create_strategy_analysis_report(self, analysis_report: Dict[str, Any]) -> str:
        """إنشاء تقرير تحليلي تفصيلي يظهر كيفية توليد الإشارات"""
        try:
            report_time = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            
            message = f"🔍 **تقرير التحليل التفصيلي - كيفية توليد الإشارات**\n"
            message += "═" * 55 + "\n"
            message += f"⏰ **وقت التقرير:** `{report_time}`\n"
            message += f"📈 **الإطار الزمني:** `{analysis_report['timeframe']}`\n"
            message += f"💰 **إجمالي العملات:** `{analysis_report['total_coins']}`\n\n"
            
            # ملخص الاستراتيجيات
            message += "📊 **ملخص أداء الاستراتيجيات:**\n"
            strategies_summary = analysis_report['strategies_summary']
            
            for strategy_name, stats in strategies_summary.items():
                display_name = self._get_strategy_display_name(strategy_name)
                total = stats['total_signals']
                buy = stats['buy_signals']
                sell = stats['sell_signals']
                
                message += f"• **{display_name}:** إجمالي {total} إشارة (🟢 {buy} شراء | 🔴 {sell} بيع)\n"
            
            message += "\n"
            message += "🔎 **التفاصيل حسب العملة والاستراتيجية:**\n"
            message += "─" * 45 + "\n"
            
            # تفاصيل كل عملة
            for coin_key, coin_analysis in analysis_report['coin_analysis'].items():
                if not coin_analysis or 'error' in coin_analysis:
                    continue
                    
                coin_name = coin_analysis.get('coin_name', coin_key.upper())
                current_price = coin_analysis.get('current_price', 0)
                data_source = coin_analysis.get('data_source', 'unknown')
                final_signal = coin_analysis.get('final_signal', 'none')
                final_confidence = coin_analysis.get('final_confidence', 0)
                
                source_emoji = "🟠" if data_source == "coinex" else "🔵" if data_source == "binance" else "⚪"
                
                message += f"\n**{coin_name} ({coin_key.upper()})** {source_emoji}\n"
                message += f"💰 **السعر:** `${current_price:,.2f}`\n"
                
                # الإشارة النهائية
                if final_signal != 'none':
                    signal_emoji = "🟢" if final_signal == 'BUY' else "🔴"
                    message += f"🎯 **الإشارة النهائية:** {signal_emoji} **{final_signal}** ({final_confidence}%)\n"
                else:
                    message += f"🎯 **الإشارة النهائية:** ⚪ **لا توجد إشارة واضحة**\n"
                
                message += "📊 **تفاصيل الاستراتيجيات:**\n"
                
                # تفاصيل كل استراتيجية
                strategies = coin_analysis.get('strategies', {})
                for strategy_name, strategy_data in strategies.items():
                    signal = strategy_data.get('signal', 'none')
                    confidence = strategy_data.get('confidence', 0)
                    analysis_details = strategy_data.get('analysis_details', {})
                    
                    strategy_display = self._get_strategy_display_name(strategy_name)
                    
                    if signal != 'none':
                        signal_emoji = "🟢" if signal == 'BUY' else "🔴"
                        message += f"  {signal_emoji} **{strategy_display}:** {signal} ({confidence}%)\n"
                        
                        # عرض تفاصيل التحليل
                        analysis_text = self._format_analysis_details(strategy_name, analysis_details)
                        if analysis_text:
                            message += f"    📈 {analysis_text}\n"
                    else:
                        message += f"  ⚪ **{strategy_display}:** لا إشارة ({confidence}%)\n"
                
                message += "─" * 35 + "\n"
            
            message += "\n💡 **كيفية قراءة التقرير:**\n"
            message += "• 🟢 إشارة شراء | 🔴 إشارة بيع | ⚪ لا إشارة\n"
            message += "• **المتوسطات:** تحليل EMA + RSI + MACD\n"
            message += "• **الحجم:** تحليل الانزياح الحجمي السعري\n"
            message += "• **الذكية:** تحليل أنماط الأموال الذكية\n"
            
            message += "═" * 55 + "\n"
            message += "⚡ **المحرك المتقدم للإشارات - التحليل التفصيلي**"
            
            return message
            
        except Exception as e:
            logger.error(f"❌ خطأ في إنشاء التقرير التحليلي: {e}")
            return f"❌ خطأ في إنشاء التقرير التحليلي: {e}"
    
    def _get_strategy_display_name(self, strategy_name: str) -> str:
        """الحصول على اسم عرضي للاستراتيجية"""
        names = {
            "ema_rsi_macd": "المتوسطات",
            "volume_divergence": "الحجم", 
            "smart_money": "الذكية"
        }
        return names.get(strategy_name, strategy_name)
    
    def _format_analysis_details(self, strategy_name: str, analysis_details: Dict) -> str:
        """تنسيق تفاصيل التحليل حسب الاستراتيجية"""
        try:
            if strategy_name == "ema_rsi_macd":
                details = []
                if analysis_details.get('rsi_value'):
                    details.append(f"RSI: {analysis_details['rsi_value']:.1f}")
                if analysis_details.get('ema_trend'):
                    details.append(f"الاتجاه: {analysis_details['ema_trend']}")
                if analysis_details.get('macd_signal') != 'محايد':
                    details.append(f"MACD: {analysis_details['macd_signal']}")
                return " | ".join(details) if details else ""
            
            elif strategy_name == "volume_divergence":
                if analysis_details.get('divergence_type') != 'none':
                    return f"انزياح: {analysis_details['divergence_type']} ({analysis_details['divergence_strength']}%)"
                return "لا انزياح ملحوظ"
            
            elif strategy_name == "smart_money":
                if analysis_details.get('smart_pattern') != 'no_pattern':
                    return f"نمط: {analysis_details['smart_pattern']} (ثقة: {analysis_details['pattern_confidence']}%)"
                return "لا نمط ذكي واضح"
            
            return ""
        except Exception:
            return ""
    
    async def send_strategy_analysis_report(self, analysis_report: Dict[str, Any]) -> bool:
        """إرسال التقرير التحليلي"""
        try:
            report_message = self.create_strategy_analysis_report(analysis_report)
            
            if len(report_message) > 4096:
                # تقسيم الرسالة إذا كانت طويلة جداً
                parts = self._split_message(report_message)
                for part in parts:
                    success = await self._send_single_message(part)
                    if not success:
                        return False
                    await asyncio.sleep(1)
                return True
            else:
                return await self._send_single_message(report_message)
                
        except Exception as e:
            logger.error(f"❌ خطأ في إرسال التقرير التحليلي: {e}")
            return False
    
    async def _send_single_message(self, message: str) -> bool:
        """إرسال رسالة واحدة"""
        try:
            payload = {
                'chat_id': TELEGRAM_CHAT_ID,
                'text': message,
                'parse_mode': 'Markdown',
                'disable_web_page_preview': True
            }
            
            async with httpx.AsyncClient() as client:
                response = await client.post(
                    f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage",
                    json=payload,
                    timeout=30.0
                )
            
            if response.status_code == 200:
                logger.info("✅ تم إرسال التقرير التحليلي بنجاح")
                system_stats["detailed_reports_sent"] += 1
                return True
            else:
                logger.error(f"❌ فشل إرسال التقرير: {response.status_code} - {response.text}")
                return False
                
        except Exception as e:
            logger.error(f"❌ خطأ في إرسال الرسالة: {e}")
            return False
    
    def _split_message(self, message: str, max_length: int = 4096) -> List[str]:
        """تقسيم الرسالة الطويلة إلى أجزاء"""
        parts = []
        while len(message) > max_length:
            split_index = message.rfind('\n', 0, max_length)
            if split_index == -1:
                split_index = max_length
            parts.append(message[:split_index])
            message = message[split_index:].lstrip()
        parts.append(message)
        return parts

# =============================================================================
# باقي المكونات (مختصرة)
# =============================================================================

class TelegramNotifier:
    """إشعارات التليجرام المحدثة"""
    
    def __init__(self, token: str, chat_id: str):
        self.token = token
        self.chat_id = chat_id
        self.base_url = f"https://api.telegram.org/bot{token}"
    
    async def send_heartbeat(self, executor_connected: bool, signals_count: int = 0, 
                        recent_analysis: Dict[str, Any] = None) -> bool:
        """إرسال نبضة اتصال مع تحليل قوة الإشارات"""
        try:
            current_time = datetime.now().strftime('%H:%M %d/%m/%Y')
            uptime_seconds = time.time() - system_stats["start_time"]
            uptime_str = self._format_uptime(uptime_seconds)
        
            status_emoji = "✅" if executor_connected else "❌"
            status_text = "متصل" if executor_connected else "غير متصل"
        
            message = f"💓 *نبضة النظام المتقدم*\n"
            message += "─" * 40 + "\n"
            message += f"⏰ *الوقت:* `{current_time}`\n"
            message += f"⏱️ *مدة التشغيل:* `{uptime_str}`\n"
            message += f"🔗 *الاتصال بالمنفذ:* {status_emoji} `{status_text}`\n"
            message += f"📊 *الإشارات المرسلة:* `{signals_count}`\n"
            message += f"🔍 *المسحات الكلية:* `{system_stats['total_scans']}`\n"
            message += f"📈 *التقارير التحليلية:* `{system_stats['detailed_reports_sent']}`\n"
        
            message += "─" * 40 + "\n"
            message += "✅ *جميع الأنظمة تعمل بشكل طبيعي*"
        
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
                logger.info("💓 تم إرسال نبضة النظام بنجاح")
                system_stats["last_heartbeat"] = current_time
                return True
            else:
                logger.error(f"❌ فشل إرسال النبضة: {response.status_code} - {response.text}")
                return False
            
        except Exception as e:
            logger.error(f"❌ خطأ في إرسال النبضة: {e}")
            return False
    
    def _format_uptime(self, seconds: float) -> str:
        """تنسيق مدة التشغيل"""
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        
        if hours > 0:
            return f"{hours} ساعة, {minutes} دقيقة"
        else:
            return f"{minutes} دقيقة"

class ExecutorBotClient:
    """عميل للتواصل مع بوت التنفيذ"""
    
    def __init__(self, base_url: str, api_key: str):
        self.base_url = base_url
        self.api_key = api_key
        self.client = httpx.AsyncClient(timeout=30.0)

    async def health_check(self) -> bool:
        """فحص حالة البوت المنفذ"""
        try:
            response = await self.client.get(f"{self.base_url}/health", timeout=10.0)
            return response.status_code == 200
        except Exception as e:
            logger.error(f"❌ فحص صحة البوت المنفذ فشل: {e}")
            return False

# =============================================================================
# التهيئة
# =============================================================================

signal_engine = AdvancedSignalEngine()
notifier = TelegramNotifier(TELEGRAM_BOT_TOKEN, TELEGRAM_CHAT_ID)
executor_client = ExecutorBotClient(EXECUTOR_BOT_URL, EXECUTOR_BOT_API_KEY)
analysis_report_generator = StrategyAnalysisReportGenerator(notifier)

# =============================================================================
# المهام الأساسية مع نظام التقارير التحليلية
# =============================================================================

recent_analysis = {}
last_analysis_report_time = 0
ANALYSIS_REPORT_INTERVAL = 1800  # 30 دقيقة بين التقارير التحليلية

async def advanced_market_scanner_task():
    """المهمة الرئيسية للمسح الضوئي المتقدم مع التقارير التحليلية"""
    global recent_analysis, last_analysis_report_time
    logger.info("🚀 بدء مهمة مسح السوق المتقدم كل 15 دقيقة مع التقارير التحليلية")
    
    while True:
        try:
            current_analysis = {}
            
            logger.info(f"🔍 بدء مسح {len(SUPPORTED_COINS)} عملة...")
            
            for coin_key, coin_data in SUPPORTED_COINS.items():
                try:
                    logger.info(f"📊 معالجة {coin_key} ({coin_data['coinex_symbol']})...")
                    
                    analysis_result = await signal_engine.analyze_coin(coin_key, coin_data['binance_symbol'])
                    current_analysis[coin_key] = analysis_result
                    
                    # تسجيل تفاصيل الإشارات
                    if analysis_result.get('success'):
                        strategies_analysis = analysis_result.get('strategies_analysis', {})
                        for strat_name, strat_data in strategies_analysis.items():
                            if strat_data.get('signal') != 'none':
                                logger.info(f"📈 {coin_key} - {strat_name}: {strat_data['signal']} ({strat_data['confidence']}%)")
                        
                except Exception as e:
                    logger.error(f"❌ خطأ في معالجة {coin_key}: {e}")
                    await asyncio.sleep(2)
                    continue
            
            recent_analysis = current_analysis
            logger.info(f"💾 تم حفظ تحليل {len(current_analysis)} عملة")
            
            # إرسال التقارير التحليلية
            current_time = time.time()
            if current_time - last_analysis_report_time >= ANALYSIS_REPORT_INTERVAL:
                logger.info("🔍 إنشاء التقرير التحليلي التفصيلي...")
                analysis_report = await signal_engine.generate_strategy_analysis_report()
                await analysis_report_generator.send_strategy_analysis_report(analysis_report)
                last_analysis_report_time = current_time
            
            system_stats["total_scans"] += 1
            
            logger.info(f"✅ اكتملت دورة المسح - انتظار {SCAN_INTERVAL} ثانية للمسح التالي...")
            await asyncio.sleep(SCAN_INTERVAL)
            
        except Exception as e:
            logger.error(f"❌ خطأ في المهمة الرئيسية: {e}")
            logger.info("⏳ انتظار 60 ثانية قبل إعادة المحاولة...")
            await asyncio.sleep(60)

async def heartbeat_task():
    """مهمة إرسال النبضات الدورية"""
    global recent_analysis
    logger.info("💓 بدء مهمة النبضات الدورية كل ساعتين")
    
    await asyncio.sleep(300)
    
    while True:
        try:
            executor_health = await executor_client.health_check()
            
            success = await notifier.send_heartbeat(
                executor_connected=executor_health,
                signals_count=system_stats["signals_sent"],
                recent_analysis=recent_analysis
            )
            
            if success:
                logger.info("✅ تم إرسال النبضة الدورية بنجاح")
            else:
                logger.error("❌ فشل إرسال النبضة الدورية")
                
            await asyncio.sleep(7200)
                
        except Exception as e:
            logger.error(f"❌ خطأ في مهمة النبضات: {e}")
            await asyncio.sleep(300)

# =============================================================================
# واجهات API الجديدة للتقارير التحليلية
# =============================================================================

@app.get("/")
async def root():
    return {
        "message": "Advanced Crypto Signal Generator",
        "status": "running",
        "version": "2.4.0",
        "strategies": list(signal_engine.strategies.keys()),
        "data_sources": ["coinex", "binance"],
        "confidence_threshold_single": CONFIDENCE_THRESHOLD_SINGLE,
        "confidence_threshold_multiple": CONFIDENCE_THRESHOLD_MULTIPLE,
        "supported_coins": list(SUPPORTED_COINS.keys()),
        "timeframe": TIMEFRAME,
        "mode": "strategy_analysis_reports",
        "analysis_reports_sent": system_stats["detailed_reports_sent"]
    }

@app.get("/strategy-analysis")
async def get_strategy_analysis():
    """الحصول على التحليل التفصيلي للاستراتيجيات"""
    try:
        analysis_report = await signal_engine.generate_strategy_analysis_report()
        return {
            "status": "success",
            "timestamp": analysis_report['timestamp'],
            "analysis": analysis_report
        }
    except Exception as e:
        return {"status": "error", "message": str(e)}

@app.get("/strategy-analysis-report")
async def generate_strategy_analysis_report():
    """توليد وإرسال تقرير تحليلي يدوي"""
    try:
        logger.info("🔍 بدء توليد التقرير التحليلي يدوياً...")
        analysis_report = await signal_engine.generate_strategy_analysis_report()
        success = await analysis_report_generator.send_strategy_analysis_report(analysis_report)
        
        return {
            "status": "success" if success else "error",
            "message": "تم إرسال التقرير التحليلي بنجاح" if success else "فشل إرسال التقرير التحليلي",
            "report_generated": True,
            "coins_analyzed": len(analysis_report['coin_analysis']),
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        logger.error(f"❌ خطأ في توليد التقرير التحليلي: {e}")
        return {"status": "error", "message": str(e)}

@app.get("/coin-analysis/{coin}")
async def get_coin_analysis(coin: str):
    """الحصول على التحليل التفصيلي لعملة محددة"""
    if coin not in SUPPORTED_COINS:
        raise HTTPException(404, "العملة غير مدعومة")
    
    coin_data = SUPPORTED_COINS[coin]
    analysis_result = await signal_engine.analyze_coin(coin, coin_data['binance_symbol'])
    
    return {
        "coin": coin,
        "timeframe": TIMEFRAME,
        "analysis": analysis_result
    }

# =============================================================================
# تشغيل التطبيق
# =============================================================================

@app.on_event("startup")
async def startup_event():
    logger.info("🚀 بدء تشغيل مولد الإشارات المتقدم مع التقارير التحليلية")
    logger.info(f"🎯 الاستراتيجيات: {list(signal_engine.strategies.keys())}")
    logger.info(f"💰 العملات المدعومة: {list(SUPPORTED_COINS.keys())}")
    logger.info(f"📡 مصادر البيانات: CoinEx (رئيسي) + Binance (احتياطي)")
    logger.info(f"⏰ الإطار الزمني: {TIMEFRAME}")
    logger.info(f"🔍 فاصل المسح: {SCAN_INTERVAL} ثانية")
    logger.info(f"📋 فاصل التقارير التحليلية: {ANALYSIS_REPORT_INTERVAL} ثانية")
    logger.info(f"💡 الوضع: تقارير تحليلية تفصيلية للاستراتيجيات")
    
    try:
        executor_health = await executor_client.health_check()
        await notifier.send_heartbeat(
            executor_connected=executor_health, 
            signals_count=system_stats["signals_sent"]
        )
    except Exception as e:
        logger.error(f"❌ خطأ في إرسال نبضة البدء: {e}")
    
    asyncio.create_task(advanced_market_scanner_task())
    asyncio.create_task(heartbeat_task())
    
    logger.info("✅ تم بدء جميع المهام بنجاح")

@app.on_event("shutdown")
async def shutdown_event():
    logger.info("🛑 إيقاف مولد الإشارات المتقدم")

if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)
