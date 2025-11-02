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

# =============================================================================
# إعدادات البوت المتقدم المحسنة
# =============================================================================

TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN", "")
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID", "")
EXECUTOR_BOT_URL = os.getenv("EXECUTOR_BOT_URL", "")
EXECUTOR_BOT_API_KEY = os.getenv("EXECUTOR_BOT_API_KEY", "")
EXECUTE_TRADES = os.getenv("EXECUTE_TRADES", "false").lower() == "true"

# إعدادات التداول المحسنة
SCAN_INTERVAL = 900  # 5 دقائق بين كل فحص
CONFIDENCE_THRESHOLD_SINGLE = 65  # عتبة الإشارة الواحدة (مخفضة)
CONFIDENCE_THRESHOLD_MULTIPLE = 60  # عتبة الإشارات المتعددة (مخفضة)
MIN_STRATEGY_CONFIDENCE = 50  # أقل ثقة للاستراتيجيات المحتسبة (مخفضة)

# العملات المدعومة
SUPPORTED_COINS = {
    'eth': {'name': 'Ethereum', 'binance_symbol': 'ETHUSDT', 'symbol': 'ETH'},
    'bnb': {'name': 'Binance Coin', 'binance_symbol': 'BNBUSDT', 'symbol': 'BNB'},
    'btc': {'name': 'Bitcoin', 'binance_symbol': 'BTCUSDT', 'symbol': 'BTC'},
}

TIMEFRAME = '1h'  # إطار زمني موحد لجميع الاستراتيجيات

# إعداد التسجيل المحسن
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger("advanced_signal_generator")

app = FastAPI(title="Advanced Crypto Signal Generator")

# إحصائيات النظام المحسنة
system_stats = {
    "start_time": time.time(),
    "total_scans": 0,
    "signals_generated": 0,
    "signals_sent": 0,
    "signals_failed": 0,
    "last_heartbeat": None,
    "last_signal_sent": None,
    "data_source_stats": {
        "binance": {"success": 0, "failed": 0},
        "coingecko": {"success": 0, "failed": 0},
        "yahoo_finance": {"success": 0, "failed": 0},
        "cryptocompare": {"success": 0, "failed": 0}
    },
    "strategies_performance": {
        "ema_rsi_macd": {"calls": 0, "signals": 0},
        "volume_divergence": {"calls": 0, "signals": 0},
        "smart_money": {"calls": 0, "signals": 0}
    },
    "execution_stats": {
        "total_attempts": 0,
        "successful": 0,
        "failed": 0,
        "last_error": None
    }
}

# =============================================================================
# مصادر البيانات المتعددة المحسنة
# =============================================================================

class EnhancedMultiSourceDataFetcher:
    """مصادر متعددة محسنة مع إدارة ذكية للطلبات"""
    
    def __init__(self):
        self.client = httpx.AsyncClient(timeout=30.0)
        self.request_times = {}
        self.min_request_interval = 2.0
        
    async def get_binance_data(self, symbol: str, interval: str, limit: int = 50) -> Optional[Dict]:
        """جلب البيانات من Binance مع إدارة محسنة للطلبات"""
        try:
            current_time = time.time()
            last_request = self.request_times.get('binance', 0)
            if current_time - last_request < self.min_request_interval:
                wait_time = self.min_request_interval - (current_time - last_request)
                await asyncio.sleep(wait_time)
            
            url = f"https://api.binance.com/api/v3/klines?symbol={symbol}&interval={interval}&limit={limit}"
            
            headers = {
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36',
                'Accept': 'application/json',
            }
            
            logger.info(f"🔍 جلب البيانات من Binance لـ {symbol}...")
            response = await self.client.get(url, headers=headers)
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
    
    async def get_coingecko_data(self, symbol: str, interval: str, limit: int = 50) -> Optional[Dict]:
        """جلب البيانات من CoinGecko"""
        try:
            current_time = time.time()
            last_request = self.request_times.get('coingecko', 0)
            if current_time - last_request < 3.0:
                wait_time = 3.0 - (current_time - last_request)
                await asyncio.sleep(wait_time)
            
            symbol_mapping = {
                'BTCUSDT': 'bitcoin',
                'ETHUSDT': 'ethereum', 
                'BNBUSDT': 'binancecoin'
            }
            
            coin_id = symbol_mapping.get(symbol)
            if not coin_id:
                return None
            
            days = 7
            url = f"https://api.coingecko.com/api/v3/coins/{coin_id}/market_chart?vs_currency=usd&days={days}"
            
            headers = {
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36',
                'Accept': 'application/json',
            }
            
            logger.info(f"🔍 جلب البيانات من CoinGecko لـ {symbol}...")
            response = await self.client.get(url, headers=headers, timeout=30.0)
            self.request_times['coingecko'] = time.time()
            
            if response.status_code == 200:
                data = response.json()
                if 'prices' in data and len(data['prices']) > 0:
                    system_stats["data_source_stats"]["coingecko"]["success"] += 1
                    
                    prices_data = []
                    prices = data['prices']
                    volumes = data.get('total_volumes', [[p[0], 1000] for p in prices])
                    
                    step = max(1, len(prices) // limit)
                    selected_prices = prices[::step][:limit]
                    selected_volumes = volumes[::step][:limit]
                    
                    for i, (price_point, volume_point) in enumerate(zip(selected_prices, selected_volumes)):
                        timestamp, price = price_point
                        _, volume = volume_point
                        
                        prices_data.append([
                            timestamp,
                            str(price),
                            str(price * 1.01),
                            str(price * 0.99),  
                            str(price),
                            str(volume),
                            timestamp,
                            str(volume),
                            "1",
                            str(volume),
                            str(volume),
                            "0"
                        ])
                    
                    logger.info(f"✅ تم جلب {len(prices_data)} نقطة سعرية من CoinGecko لـ {symbol}")
                    return prices_data
            
            system_stats["data_source_stats"]["coingecko"]["failed"] += 1
            if response.status_code == 429:
                wait_time = random.uniform(30, 60)
                logger.warning(f"⏳ تم حظر الطلبات من CoinGecko، انتظار {wait_time:.1f} ثانية...")
                await asyncio.sleep(wait_time)
            
            return None
            
        except Exception as e:
            system_stats["data_source_stats"]["coingecko"]["failed"] += 1
            logger.error(f"❌ خطأ في جلب البيانات من CoinGecko لـ {symbol}: {e}")
            return None
    
    async def get_cryptocompare_data(self, symbol: str, interval: str, limit: int = 50) -> Optional[Dict]:
        """جلب البيانات من CryptoCompare"""
        try:
            current_time = time.time()
            last_request = self.request_times.get('cryptocompare', 0)
            if current_time - last_request < 2.0:
                wait_time = 2.0 - (current_time - last_request)
                await asyncio.sleep(wait_time)
            
            symbol_mapping = {
                'BTCUSDT': 'BTC',
                'ETHUSDT': 'ETH', 
                'BNBUSDT': 'BNB'
            }
            
            fsym = symbol_mapping.get(symbol)
            if not fsym:
                return None
            
            url = f"https://min-api.cryptocompare.com/data/v2/histohour?fsym={fsym}&tsym=USD&limit={limit}"
            
            headers = {
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36',
                'Accept': 'application/json',
            }
            
            logger.info(f"🔍 جلب البيانات من CryptoCompare لـ {symbol}...")
            response = await self.client.get(url, headers=headers, timeout=30.0)
            self.request_times['cryptocompare'] = time.time()
            
            if response.status_code == 200:
                data = response.json()
                if data.get('Data') and data['Data'].get('Data'):
                    raw_data = data['Data']['Data']
                    
                    prices_data = []
                    for item in raw_data:
                        prices_data.append([
                            item['time'] * 1000,
                            str(item['open']),
                            str(item['high']),
                            str(item['low']),  
                            str(item['close']),
                            str(item['volumeto']),
                            item['time'] * 1000,
                            str(item['volumeto']),
                            "1",
                            str(item['volumeto']),
                            str(item['volumeto']),
                            "0"
                        ])
                    
                    if prices_data:
                        system_stats["data_source_stats"]["cryptocompare"]["success"] += 1
                        logger.info(f"✅ تم جلب {len(prices_data)} نقطة سعرية من CryptoCompare لـ {symbol}")
                        return prices_data
            
            system_stats["data_source_stats"]["cryptocompare"]["failed"] += 1
            return None
            
        except Exception as e:
            system_stats["data_source_stats"]["cryptocompare"]["failed"] += 1
            logger.error(f"❌ خطأ في جلب البيانات من CryptoCompare لـ {symbol}: {e}")
            return None
    
    async def get_coin_data(self, symbol: str, interval: str) -> Dict[str, Any]:
        """جلب البيانات من مصادر متعددة مع fallback ذكي"""
        logger.info(f"🔍 بدء جلب بيانات {symbol} من مصادر متعددة...")
        
        sources = [
            ("binance", self.get_binance_data),
            ("cryptocompare", self.get_cryptocompare_data),
            ("coingecko", self.get_coingecko_data),
        ]
        
        for source_name, source_func in sources:
            try:
                logger.info(f"🔄 المحاولة مع {source_name} لـ {symbol}...")
                data = await source_func(symbol, interval)
                
                if data is not None and len(data) >= 20:
                    prices = [float(item[4]) for item in data]
                    volumes = [float(item[5]) for item in data]
                    
                    logger.info(f"✅ نجح جلب البيانات من {source_name} لـ {symbol} - {len(prices)} سعر")
                    return {
                        "prices": prices,
                        "volumes": volumes, 
                        "data_source": source_name,
                        "success": True
                    }
                
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
# الاستراتيجيات (بدون تغيير)
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
        """توليد إشارة تداول"""
        system_stats["strategies_performance"][self.name]["calls"] += 1
        
        if len(prices) < 20:
            return {"signal": "none", "confidence": 0, "reasons": ["بيانات غير كافية"]}
        
        current_price = prices[-1]
        rsi = self.calculate_rsi(prices)
        macd = self.calculate_macd(prices)
        trend = self.analyze_trend(prices, current_price)
        
        confidence_score = 0
        reasons = []
        
        # تحليل المتوسطات
        ma_score = 0
        if trend['order'] == "صاعد" and trend['price_above_21'] and trend['price_above_50']:
            ma_score = trend['strength'] * 4
            reasons.append(f"المتوسطات صاعدة (قوة: {trend['strength']}/10)")
        elif trend['order'] == "هابط" and not trend['price_above_21'] and not trend['price_above_50']:
            ma_score = trend['strength'] * 4
            reasons.append(f"المتوسطات هابطة (قوة: {trend['strength']}/10)")
        
        # تحليل RSI
        rsi_score = 0
        if 40 <= rsi <= 65:
            distance_from_50 = abs(rsi - 50)
            rsi_score = max(0, 30 - (distance_from_50 * 1.5))
            reasons.append(f"RSI في منطقة مناسبة: {rsi}")
        elif 35 <= rsi <= 60:
            distance_from_50 = abs(rsi - 50)
            rsi_score = max(0, 30 - (distance_from_50 * 1.5))
            reasons.append(f"RSI في منطقة مناسبة: {rsi}")
        
        # تحليل MACD
        macd_score = 0
        if macd['histogram'] > 0 and macd['macd'] > macd['signal']:
            macd_score = min(30, abs(macd['histogram']) * 1000)
            reasons.append(f"MACD إيجابي: {macd['histogram']:.4f}")
        elif macd['histogram'] < 0 and macd['macd'] < macd['signal']:
            macd_score = min(30, abs(macd['histogram']) * 1000)
            reasons.append(f"MACD سلبي: {macd['histogram']:.4f}")
        
        confidence_score = ma_score + rsi_score + macd_score
        
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
            "timestamp": time.time()
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
        """توليد إشارة بناء على الانزياح الحجمي السعري"""
        system_stats["strategies_performance"][self.name]["calls"] += 1
        
        if len(prices) < 40:
            return {"signal": "none", "confidence": 0, "reasons": ["بيانات غير كافية"]}
        
        current_price = prices[-1]
        divergence_data = self.calculate_divergence(prices, volumes)
        
        confidence_score = 0
        signal_type = "none"
        reasons = []
        
        if divergence_data["divergence"] == "positive_bullish":
            confidence_score = divergence_data["strength"]
            signal_type = "BUY"
            reasons = [
                "انزياح إيجابي: هبوط الأسعار مع ضعف حجم البيع",
                "تشير إلى استنفاد البائعين واستعداد للارتداد"
            ]
        
        elif divergence_data["divergence"] == "negative_bearish":
            confidence_score = divergence_data["strength"]
            signal_type = "SELL"
            reasons = [
                "انزياح سلبي: صعود الأسعار مع ضعف حجم الشراء",
                "تشير إلى استنفاد المشترين واستعداد للهبوط"
            ]
        
        elif divergence_data["divergence"] == "volume_confirmation":
            price_trend = "صاعد" if prices[-1] > prices[-10] else "هابط"
            
            if price_trend == "صاعد":
                confidence_score = divergence_data["strength"]
                signal_type = "BUY"
                reasons = [
                    "تأكيد حجمي قوي للصعود",
                    "حجم الشراء يدعم استمرار الاتجاه الصاعد"
                ]
            else:
                confidence_score = divergence_data["strength"]
                signal_type = "SELL"
                reasons = [
                    "تأكيد حجمي قوي للهبوط", 
                    "حجم البيع يدعم استمرار الاتجاه الهابط"
                ]
        
        if signal_type != "none" and confidence_score >= 40:
            system_stats["strategies_performance"][self.name]["signals"] += 1
        
        return {
            "signal": signal_type,
            "confidence": confidence_score,
            "price": current_price,
            "reasons": reasons,
            "timestamp": time.time()
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
        """توليد إشارة بناء على تحليل التراكم والتوزيع"""
        system_stats["strategies_performance"][self.name]["calls"] += 1
        
        if len(prices) < 40:
            return {"signal": "none", "confidence": 0, "reasons": ["بيانات غير كافية"]}
        
        current_price = prices[-1]
        smart_pattern = self.detect_smart_money_patterns(prices, volumes)
        
        volume_ma_20 = np.mean(volumes[-20:])
        current_volume = volumes[-1]
        volume_ratio = current_volume / volume_ma_20 if volume_ma_20 > 0 else 1
        
        confidence_score = 0
        signal_type = "none"
        reasons = []
        
        if (smart_pattern["pattern"] == "accumulation" and 
            volume_ratio > 1.5 and
            smart_pattern["confidence"] > 40):
            
            confidence_score = smart_pattern["confidence"]
            signal_type = "BUY"
            reasons = [
                "نمط تراكم الأموال الذكية",
                "حجم التداول مرتفع يشير إلى تراكم"
            ]
        
        elif (smart_pattern["pattern"] == "distribution" and 
              volume_ratio > 1.5 and
              smart_pattern["confidence"] > 40):
            
            confidence_score = smart_pattern["confidence"]
            signal_type = "SELL"
            reasons = [
                "نمط توزيع الأموال الذكية",
                "حجم التداول مرتفع يشير إلى توزيع"
            ]
        
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
            else:
                confidence_score = smart_pattern["confidence"]
                signal_type = "SELL" 
                reasons = [
                    "امتصاص شراء قوي من قبل البائعين الأقوياء",
                    "حجم امتصاص مرتفع"
                ]
        
        if signal_type != "none" and confidence_score >= 40:
            system_stats["strategies_performance"][self.name]["signals"] += 1
        
        return {
            "signal": signal_type,
            "confidence": confidence_score,
            "price": current_price,
            "reasons": reasons,
            "timestamp": time.time()
        }

# =============================================================================
# محرك الإشارات المتقدم المحسن
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
        
        data_result = await self.data_fetcher.get_coin_data(binance_symbol, TIMEFRAME)
        
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
                strategy_signals[strategy_name] = {"signal": "none", "confidence": 0, "reasons": [f"خطأ: {str(e)}"]}
        
        final_signal = self.process_strategy_signals(strategy_signals)
        final_signal["coin_key"] = coin_key
        final_signal["data_source"] = data_source
        final_signal["current_price"] = prices[-1] if prices else 0
        final_signal["prices_count"] = len(prices)
        final_signal["success"] = True
        
        return final_signal

# =============================================================================
# نظام الإشعارات والإرسال المحسن
# =============================================================================

class TelegramNotifier:
    """إشعارات التليجرام المحدثة"""
    
    def __init__(self, token: str, chat_id: str):
        self.token = token
        self.chat_id = chat_id
        self.base_url = f"https://api.telegram.org/bot{token}"
    
    async def send_advanced_signal_alert(self, coin: str, signal_data: Dict[str, Any]) -> bool:
        """إرسال إشعار إشارة متقدم"""
        if signal_data["signal"] == "none":
            return False
        
        try:
            message = self._build_advanced_signal_message(coin, signal_data)
            
            payload = {
                'chat_id': self.chat_id,
                'text': message,
                'parse_mode': 'Markdown'
            }
            
            async with httpx.AsyncClient() as client:
                response = await client.post(f"{self.base_url}/sendMessage", 
                                           json=payload, timeout=10.0)
            
            if response.status_code == 200:
                logger.info(f"📨 تم إرسال إشعار إشارة متقدم لـ {coin}")
                return True
            else:
                logger.error(f"❌ فشل إرسال الإشعار: {response.status_code} - {response.text}")
                return False
                
        except Exception as e:
            logger.error(f"❌ خطأ في إرسال الإشعار: {e}")
            return False
    
    def _build_advanced_signal_message(self, coin: str, signal_data: Dict[str, Any]) -> str:
        """بناء رسالة إشارة متقدمة"""
        signal_type = signal_data["signal"]
        confidence = signal_data["confidence"]
        price = signal_data["current_price"]
        winning_strategies = signal_data["winning_strategies"]
        total_strategies = signal_data["total_strategies"]
        strategies_analysis = signal_data["strategies_analysis"]
        data_source = signal_data.get("data_source", "unknown")
        
        if signal_type == "BUY":
            emoji = "🟢"
            action = "شراء"
            action_emoji = "📈"
        else:
            emoji = "🔴" 
            action = "بيع"
            action_emoji = "📉"
        
        source_emoji = "🔵" if data_source == "binance" else "🟡" if data_source == "coingecko" else "🟢"
        
        message = f"{emoji} *إشارة {action} {action_emoji} - {coin.upper()}*\n"
        message += "─" * 40 + "\n"
        message += f"💰 *السعر:* `${price:,.2f}`\n"
        message += f"🎯 *الثقة النهائية:* `{confidence}%`\n"
        message += f"📊 *الاستراتيجيات:* `{winning_strategies}/{total_strategies}`\n"
        message += f"📡 *مصدر البيانات:* {source_emoji} `{data_source}`\n"
        message += f"⏰ *الإطار:* `{TIMEFRAME}`\n\n"
        
        message += "*تحليل الاستراتيجيات:*\n"
        for strategy_name, analysis in strategies_analysis.items():
            status_emoji = "✅" if analysis["signal"] == signal_type else "➖" if analysis["signal"] == "none" else "❌"
            display_name = strategy_name.replace('_', ' ').title()
            signal_emoji = "🟢" if analysis["signal"] == "BUY" else "🔴" if analysis["signal"] == "SELL" else "⚪"
            message += f"{status_emoji} *{display_name}:* {signal_emoji} `{analysis['confidence']}%`"
            if analysis["signal"] != "none" and analysis["signal"] != signal_type:
                message += f" (⚠️ {analysis['signal']})"
            message += "\n"
        
        message += "\n*الأسباب:*\n"
        for i, reason in enumerate(signal_data["reasons"][:5], 1):
            message += f"• {reason}\n"
        
        message += "─" * 40 + "\n"
        message += f"🕒 *الوقت:* `{datetime.now().strftime('%H:%M %d/%m')}`\n"
        message += "⚡ *المحرك المتقدم للإشارات*"
        
        return message
    
    async def send_heartbeat(self, executor_connected: bool, signals_count: int = 0, 
                        recent_analysis: Dict[str, Any] = None) -> bool:
        """إرسال نبضة اتصال مع تحليل قوة الإشارات"""
        try:
            current_time = datetime.now().strftime('%H:%M %d/%m/%Y')
            uptime_seconds = time.time() - system_stats["start_time"]
            uptime_str = self._format_uptime(uptime_seconds)
        
            status_emoji = "✅" if executor_connected else "❌"
            status_text = "متصل" if executor_connected else "غير متصل"
        
            strategies_stats = system_stats["strategies_performance"]
            data_stats = system_stats["data_source_stats"]
        
            message = f"💓 *نبضة النظام المتقدم*\n"
            message += "─" * 40 + "\n"
            message += f"⏰ *الوقت:* `{current_time}`\n"
            message += f"⏱️ *مدة التشغيل:* `{uptime_str}`\n"
            message += f"🔗 *الاتصال بالمنفذ:* {status_emoji} `{status_text}`\n"
            message += f"📊 *الإشارات المرسلة:* `{signals_count}`\n"
            message += f"🔍 *المسحات الكلية:* `{system_stats['total_scans']}`\n\n"
        
            message += "*أداء الاستراتيجيات:*\n"
            for strategy_name, stats in strategies_stats.items():
                success_rate = (stats["signals"] / stats["calls"] * 100) if stats["calls"] > 0 else 0
                display_name = strategy_name.replace('_', ' ').title()
                message += f"• *{display_name}:* `{stats['signals']}/{stats['calls']}` ({success_rate:.1f}%)\n"
            
            message += "\n*مصادر البيانات:*\n"
            for source_name, stats in data_stats.items():
                total = stats["success"] + stats["failed"]
                success_rate = (stats["success"] / total * 100) if total > 0 else 0
                source_emoji = "🔵" if source_name == "binance" else "🟡" if source_name == "coingecko" else "🟢"
                clean_source_name = source_name.replace('_', ' ').title()
                message += f"• {source_emoji} *{clean_source_name}:* `{stats['success']}/{total}` ({success_rate:.1f}%)\n"
        
            if recent_analysis:
                message += "\n*📈 تحليل قوة الإشارات الأخيرة:*\n"
                
                signals_summary = []
                for coin, analysis in recent_analysis.items():
                    if analysis and analysis.get('success') and analysis.get('strategies_analysis'):
                        strategies_data = analysis['strategies_analysis']
                        data_source = analysis.get('data_source', 'unknown')
                        source_emoji = "🔵" if data_source == "binance" else "🟡" if data_source == "coingecko" else "🟢"
                        
                        buy_signals = []
                        sell_signals = []
                        
                        for strategy_name, strat_data in strategies_data.items():
                            if strat_data['signal'] == 'BUY' and strat_data['confidence'] > 0:
                                buy_signals.append(strat_data['confidence'])
                            elif strat_data['signal'] == 'SELL' and strat_data['confidence'] > 0:
                                sell_signals.append(strat_data['confidence'])
                        
                        if buy_signals and sell_signals:
                            buy_avg = sum(buy_signals) / len(buy_signals) if buy_signals else 0
                            sell_avg = sum(sell_signals) / len(sell_signals) if sell_signals else 0
                            buy_count = len(buy_signals)
                            sell_count = len(sell_signals)
                            
                            if buy_count > sell_count:
                                dominant_signal = "🟢 شراء"
                                dominant_strength = buy_avg
                            elif sell_count > buy_count:
                                dominant_signal = "🔴 بيع" 
                                dominant_strength = sell_avg
                            else:
                                dominant_signal = "⚖️ متعادل"
                                dominant_strength = max(buy_avg, sell_avg)
                            
                            signals_summary.append(f"⚡ *{coin.upper()}:* {dominant_signal} ({buy_count}/{sell_count}) - {dominant_strength:.1f}% {source_emoji}")
                            
                        elif buy_signals:
                            avg_confidence = sum(buy_signals) / len(buy_signals)
                            strength_emoji = "💪" if avg_confidence >= 60 else "👍" if avg_confidence >= 40 else "👎"
                            signals_summary.append(f"🟢 *{coin.upper()}:* شراء ({len(buy_signals)}/3) - {avg_confidence:.1f}% {strength_emoji} {source_emoji}")
                            
                        elif sell_signals:
                            avg_confidence = sum(sell_signals) / len(sell_signals)
                            strength_emoji = "💪" if avg_confidence >= 60 else "👍" if avg_confidence >= 40 else "👎"
                            signals_summary.append(f"🔴 *{coin.upper()}:* بيع ({len(sell_signals)}/3) - {avg_confidence:.1f}% {strength_emoji} {source_emoji}")
                            
                        else:
                            max_confidence = max([strat_data.get('confidence', 0) for strat_data in strategies_data.values()])
                            if max_confidence > 20:
                                signals_summary.append(f"⚪ *{coin.upper()}:* إشارات ضعيفة (أعلى: {max_confidence}%) {source_emoji}")
                            else:
                                signals_summary.append(f"⚫ *{coin.upper()}:* لا توجد إشارات {source_emoji}")
                    else:
                        signals_summary.append(f"⚫ *{coin.upper()}:* بيانات غير متوفرة")
                
                if signals_summary:
                    for signal_line in signals_summary:
                        message += f"{signal_line}\n"
                else:
                    message += "⚫ لا توجد بيانات للعرض\n"
            else:
                message += "\n*📈 تحليل قوة الإشارات الأخيرة:*\n"
                message += "⚫ لا توجد بيانات حديثة\n"
        
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
    """عميل محسن للتواصل مع بوت التنفيذ"""
    
    def __init__(self, base_url: str, api_key: str):
        self.base_url = base_url
        self.api_key = api_key
        self.client = httpx.AsyncClient(timeout=30.0)
    
    async def send_trade_signal(self, signal_data: Dict[str, Any]) -> bool:
        """إرسال إشارة تداول إلى البوت المنفذ مع تتبع مفصل"""
        logger.info(f"🔗 بدء إرسال إشارة إلى البوت المنفذ: {signal_data['coin']} - {signal_data['action']}")
        system_stats["execution_stats"]["total_attempts"] += 1
        
        if not EXECUTE_TRADES:
            logger.info("⏸️ تنفيذ الصفقات معطل في الإعدادات - لن يتم إرسال الإشارة")
            system_stats["execution_stats"]["last_error"] = "تنفيذ الصفقات معطل"
            return False
            
        try:
            headers = {
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json"
            }
            
            payload = {
                "signal": signal_data,
                "timestamp": time.time(),
                "source": "advanced_signal_generator",
                "version": "2.0.0"
            }
            
            logger.info(f"🌐 إرسال طلب إلى: {self.base_url}/api/trade/signal")
            logger.info(f"📦 بيانات الإشارة: {signal_data['coin']} - {signal_data['action']} - {signal_data['confidence']}%")
            
            response = await self.client.post(
                f"{self.base_url}/api/trade/signal",
                json=payload,
                headers=headers
            )
            
            logger.info(f"📡 استجابة البوت المنفذ: {response.status_code}")
            
            if response.status_code == 200:
                logger.info(f"✅ تم إرسال إشارة للتنفيذ: {signal_data['coin']} - {signal_data['action']}")
                system_stats["signals_sent"] += 1
                system_stats["execution_stats"]["successful"] += 1
                system_stats["last_signal_sent"] = datetime.now().isoformat()
                return True
            else:
                logger.error(f"❌ فشل إرسال الإشارة: {response.status_code} - {response.text}")
                system_stats["signals_failed"] += 1
                system_stats["execution_stats"]["failed"] += 1
                system_stats["execution_stats"]["last_error"] = f"HTTP {response.status_code}: {response.text}"
                return False
                
        except Exception as e:
            logger.error(f"❌ خطأ في التواصل مع البوت المنفذ: {e}")
            system_stats["signals_failed"] += 1
            system_stats["execution_stats"]["failed"] += 1
            system_stats["execution_stats"]["last_error"] = str(e)
            return False

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

# =============================================================================
# المهام الأساسية المحسنة
# =============================================================================

recent_analysis = {}

async def check_executor_connection():
    """فحص مفصل لاتصال البوت المنفذ"""
    logger.info("🔍 فحص اتصال البوت المنفذ...")
    
    if not EXECUTOR_BOT_URL:
        logger.error("❌ عنوان البوت المنفذ غير مضبوط")
        return False
    
    if not EXECUTOR_BOT_API_KEY:
        logger.error("❌ مفتاح API للبوت المنفذ غير مضبوط")
        return False
    
    if not EXECUTE_TRADES:
        logger.warning("⚠️ وضع تنفيذ الصفقات معطل")
        return False
    
    try:
        health = await executor_client.health_check()
        logger.info(f"🏥 حالة البوت المنفذ: {'✅ متصل' if health else '❌ غير متصل'}")
        return health
    except Exception as e:
        logger.error(f"❌ خطأ في فحص اتصال البوت المنفذ: {e}")
        return False

async def advanced_market_scanner_task():
    """المهمة الرئيسية للمسح الضوئي المتقدم مع تتبع محسن"""
    global recent_analysis
    logger.info("🚀 بدء مهمة مسح السوق المتقدم كل 5 دقائق")
    
    # فحص اتصال البوت المنفذ في البداية
    executor_health = await check_executor_connection()
    
    while True:
        try:
            signals_found = 0
            scan_results = []
            current_analysis = {}
            
            logger.info(f"🔍 بدء مسح {len(SUPPORTED_COINS)} عملة...")
            
            for coin_key, coin_data in SUPPORTED_COINS.items():
                try:
                    logger.info(f"📊 معالجة {coin_key} ({coin_data['binance_symbol']})...")
                    
                    analysis_result = await signal_engine.analyze_coin(coin_key, coin_data['binance_symbol'])
                    
                    current_analysis[coin_key] = analysis_result
                    
                    data_source = analysis_result.get('data_source', 'unknown')
                    source_emoji = "🔵" if data_source == "binance" else "🟡" if data_source == "coingecko" else "🟢"
                    
                    if analysis_result.get('success'):
                        strategies_analysis = analysis_result.get('strategies_analysis', {})
                        active_strategies = []
                        for strat_name, strat_data in strategies_analysis.items():
                            if strat_data.get('signal') != 'none' and strat_data.get('confidence', 0) > 0:
                                signal_emoji = "🟢" if strat_data['signal'] == 'BUY' else "🔴"
                                active_strategies.append(f"{strat_name}: {signal_emoji}({strat_data['confidence']}%)")
                        
                        if active_strategies:
                            logger.info(f"📈 {coin_key} - إشارات: {', '.join(active_strategies)} {source_emoji}")
                        else:
                            logger.info(f"➖ {coin_key} - لا توجد إشارات نشطة {source_emoji}")
                    
                    # التحقق من الإشارة وإضافتها للنتائج
                    if (analysis_result["signal"] != "none" and 
                        analysis_result["confidence"] >= CONFIDENCE_THRESHOLD_SINGLE):
                        
                        logger.info(f"🎯 إشارة قوية {analysis_result['signal']} لـ {coin_key} - ثقة: {analysis_result['confidence']}% {source_emoji}")
                        logger.info(f"📝 إضافة إشارة {coin_key} إلى قائمة النتائج للإرسال")
                        
                        scan_results.append({
                            'coin': coin_key,
                            'coin_data': coin_data,
                            'analysis': analysis_result
                        })
                        signals_found += 1
                    else:
                        logger.info(f"⏭️ إشارة {coin_key} غير مؤهلة للإرسال: إشارة={analysis_result['signal']}, ثقة={analysis_result['confidence']}%")
                        
                except Exception as e:
                    logger.error(f"❌ خطأ في معالجة {coin_key}: {e}")
                    await asyncio.sleep(2)
                    continue
            
            recent_analysis = current_analysis
            logger.info(f"💾 تم حفظ تحليل {len(current_analysis)} عملة")
            logger.info(f"📊 إجمالي الإشارات المؤهلة للإرسال: {signals_found}")
            
            # الجزء الحاسم: إرسال الإشارات
            if signals_found > 0:
                logger.info(f"🚀 بدء إرسال {signals_found} إشارة إلى البوت المنفذ")
                await send_unified_alert(scan_results)
                
                for i, result in enumerate(scan_results):
                    logger.info(f"📤 معالجة الإشارة {i+1}/{signals_found}: {result['coin']}")
                    
                    trade_signal = {
                        "coin": result['coin'],
                        "symbol": result['coin_data']['binance_symbol'],
                        "action": result['analysis']["signal"],
                        "timeframe": TIMEFRAME,
                        "price": result['analysis']["current_price"],
                        "confidence": result['analysis']["confidence"],
                        "winning_strategies": result['analysis']["winning_strategies"],
                        "total_strategies": result['analysis']["total_strategies"],
                        "reasons": result['analysis']["reasons"],
                        "strategies_analysis": result['analysis']["strategies_analysis"],
                        "data_source": result['analysis'].get("data_source", "unknown")
                    }
                    
                    logger.info(f"🔄 إرسال إشارة {result['coin']} إلى البوت المنفذ...")
                    success = await executor_client.send_trade_signal(trade_signal)
                    
                    if success:
                        logger.info(f"✅ تم إرسال إشارة {result['coin']} بنجاح")
                    else:
                        logger.error(f"❌ فشل إرسال إشارة {result['coin']}")
                    
                    await asyncio.sleep(1)
            else:
                logger.info("📭 لا توجد إشارات مؤهلة للإرسال في هذه الدورة")
            
            system_stats["total_scans"] += 1
            system_stats["signals_generated"] += signals_found
            
            if signals_found > 0:
                logger.info(f"✅ اكتملت دورة المسح - تم العثور على {signals_found} إشارة وإرسالها")
            else:
                logger.info("✅ اكتملت دورة المسح - لا توجد إشارات قوية")
            
            logger.info(f"⏳ انتظار {SCAN_INTERVAL} ثانية للمسح التالي...")
            await asyncio.sleep(SCAN_INTERVAL)
            
        except Exception as e:
            logger.error(f"❌ خطأ في المهمة الرئيسية: {e}")
            logger.info("⏳ انتظار 60 ثانية قبل إعادة المحاولة...")
            await asyncio.sleep(60)

async def heartbeat_task():
    """مهمة إرسال النبضات الدورية مع تحليل الإشارات"""
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

async def send_unified_alert(scan_results: List[Dict]):
    """إرسال إشعار موحد بجميع الإشارات"""
    if not scan_results:
        logger.info("📭 لا توجد إشارات لإرسالها في التقرير الموحد")
        return
    
    try:
        message = "📊 **تقرير المسح المتقدم**\n"
        message += "─" * 40 + "\n"
        message += f"⏰ **الوقت:** `{datetime.now().strftime('%H:%M %d/%m')}`\n"
        message += f"🔍 **العملات المفحوصة:** `{len(SUPPORTED_COINS)}`\n"
        message += f"🎯 **الإشارات المكتشفة:** `{len(scan_results)}`\n\n"
        
        for i, result in enumerate(scan_results, 1):
            signal_type = result['analysis']["signal"]
            confidence = result['analysis']["confidence"]
            winning_strategies = result['analysis']["winning_strategies"]
            data_source = result['analysis'].get("data_source", "unknown")
            source_emoji = "🔵" if data_source == "binance" else "🟡" if data_source == "coingecko" else "🟢"
            
            emoji = "🟢" if signal_type == "BUY" else "🔴"
            action_emoji = "📈" if signal_type == "BUY" else "📉"
            message += f"{emoji} **{result['coin'].upper()}:** {signal_type} {action_emoji} ({confidence}%) - {winning_strategies}/3 استراتيجيات {source_emoji}\n"
        
        message += "─" * 40 + "\n"
        message += "⚡ **المحرك المتقدم للإشارات**"
        
        payload = {
            'chat_id': TELEGRAM_CHAT_ID,
            'text': message,
            'parse_mode': 'Markdown'
        }
        
        async with httpx.AsyncClient() as client:
            await client.post(f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage", 
                            json=payload, timeout=10.0)
            
        logger.info(f"📨 تم إرسال التقرير الموحد ({len(scan_results)} إشارة)")
        
    except Exception as e:
        logger.error(f"❌ خطأ في إرسال التقرير الموحد: {e}")

# =============================================================================
# واجهات API المحسنة
# =============================================================================

@app.get("/")
async def root():
    return {
        "message": "Advanced Crypto Signal Generator",
        "status": "running",
        "version": "2.3.0",
        "strategies": list(signal_engine.strategies.keys()),
        "data_sources": ["binance", "coingecko", "cryptocompare"],
        "confidence_threshold_single": CONFIDENCE_THRESHOLD_SINGLE,
        "confidence_threshold_multiple": CONFIDENCE_THRESHOLD_MULTIPLE,
        "supported_coins": list(SUPPORTED_COINS.keys()),
        "timeframe": TIMEFRAME,
        "execute_trades": EXECUTE_TRADES
    }

@app.get("/scan/{coin}")
async def scan_coin(coin: str):
    if coin not in SUPPORTED_COINS:
        raise HTTPException(404, "العملة غير مدعومة")
    
    coin_data = SUPPORTED_COINS[coin]
    analysis_result = await signal_engine.analyze_coin(coin, coin_data['binance_symbol'])
    
    return {
        "coin": coin,
        "timeframe": TIMEFRAME,
        "analysis": analysis_result
    }

@app.get("/system-stats")
async def get_system_stats():
    uptime = time.time() - system_stats["start_time"]
    hours = int(uptime // 3600)
    minutes = int((uptime % 3600) // 60)
    
    return {
        "uptime": f"{hours} ساعة, {minutes} دقيقة",
        "uptime_seconds": uptime,
        "total_scans": system_stats["total_scans"],
        "signals_generated": system_stats["signals_generated"],
        "signals_sent": system_stats["signals_sent"],
        "signals_failed": system_stats["signals_failed"],
        "last_heartbeat": system_stats["last_heartbeat"],
        "last_signal_sent": system_stats["last_signal_sent"],
        "data_source_stats": system_stats["data_source_stats"],
        "strategies_performance": system_stats["strategies_performance"],
        "execution_stats": system_stats["execution_stats"],
        "confidence_thresholds": {
            "single_signal": CONFIDENCE_THRESHOLD_SINGLE,
            "multiple_signals": CONFIDENCE_THRESHOLD_MULTIPLE,
            "min_strategy_confidence": MIN_STRATEGY_CONFIDENCE
        },
        "supported_coins_count": len(SUPPORTED_COINS),
        "timeframe": TIMEFRAME,
        "execute_trades": EXECUTE_TRADES
    }

@app.get("/test-signal/{coin}")
async def test_signal(coin: str):
    """اختبار توليد وإرسال إشارة لعملة معينة"""
    if coin not in SUPPORTED_COINS:
        raise HTTPException(404, "العملة غير مدعومة")
    
    coin_data = SUPPORTED_COINS[coin]
    analysis_result = await signal_engine.analyze_coin(coin, coin_data['binance_symbol'])
    
    test_result = {
        "coin": coin,
        "timeframe": TIMEFRAME,
        "analysis": analysis_result,
        "test_alert_sent": False,
        "test_execution_sent": False
    }
    
    if analysis_result["signal"] != "none":
        # إرسال إشعار تيليجرام
        alert_sent = await notifier.send_advanced_signal_alert(coin, analysis_result)
        test_result["test_alert_sent"] = alert_sent
        
        # إرسال إشارة تنفيذ
        if analysis_result["confidence"] >= CONFIDENCE_THRESHOLD_SINGLE:
            trade_signal = {
                "coin": coin,
                "symbol": coin_data['binance_symbol'],
                "action": analysis_result["signal"],
                "timeframe": TIMEFRAME,
                "price": analysis_result["current_price"],
                "confidence": analysis_result["confidence"],
                "winning_strategies": analysis_result["winning_strategies"],
                "total_strategies": analysis_result["total_strategies"],
                "reasons": analysis_result["reasons"],
                "strategies_analysis": analysis_result["strategies_analysis"],
                "data_source": analysis_result.get("data_source", "unknown")
            }
            execution_sent = await executor_client.send_trade_signal(trade_signal)
            test_result["test_execution_sent"] = execution_sent
    
    return test_result

@app.get("/test-heartbeat")
async def test_heartbeat():
    """اختبار إرسال نبضة يدوية"""
    global recent_analysis
    try:
        executor_health = await executor_client.health_check()
        success = await notifier.send_heartbeat(
            executor_connected=executor_health,
            signals_count=system_stats["signals_sent"],
            recent_analysis=recent_analysis
        )
        
        return {
            "status": "success" if success else "error",
            "executor_connected": executor_health,
            "message": "تم إرسال النبضة بنجاح" if success else "فشل إرسال النبضة"
        }
    except Exception as e:
        return {"status": "error", "message": str(e)}

@app.get("/health")
async def health_check():
    """فحص صحة النظام"""
    try:
        executor_health = await executor_client.health_check()
        
        return {
            "status": "healthy",
            "timestamp": datetime.now().isoformat(),
            "executor_connected": executor_health,
            "execute_trades_enabled": EXECUTE_TRADES,
            "system_stats": {
                "uptime_seconds": time.time() - system_stats["start_time"],
                "total_scans": system_stats["total_scans"],
                "signals_sent": system_stats["signals_sent"],
                "signals_failed": system_stats["signals_failed"]
            }
        }
    except Exception as e:
        return {"status": "unhealthy", "error": str(e)}

@app.get("/debug/execution")
async def debug_execution():
    """تشخيص مشاكل التنفيذ"""
    execution_info = {
        "execute_trades": EXECUTE_TRADES,
        "executor_url": EXECUTOR_BOT_URL,
        "executor_api_key_set": bool(EXECUTOR_BOT_API_KEY),
        "recent_signals_count": len([v for v in recent_analysis.values() 
                                   if v and v.get('signal') != 'none' and v.get('confidence', 0) >= CONFIDENCE_THRESHOLD_SINGLE]),
        "signals_sent": system_stats["signals_sent"],
        "signals_generated": system_stats["signals_generated"],
        "signals_failed": system_stats["signals_failed"],
        "last_heartbeat": system_stats["last_heartbeat"],
        "last_signal_sent": system_stats["last_signal_sent"],
        "execution_stats": system_stats["execution_stats"]
    }
    
    # فحص اتصال البوت المنفذ
    try:
        executor_health = await executor_client.health_check()
        execution_info["executor_connected"] = executor_health
    except Exception as e:
        execution_info["executor_connected"] = False
        execution_info["executor_error"] = str(e)
    
    return execution_info

@app.get("/debug/signals")
async def debug_signals():
    """واجهة تشخيص للإشارات"""
    recent_signals = []
    
    for coin_key, analysis in recent_analysis.items():
        if analysis and analysis.get('success'):
            recent_signals.append({
                'coin': coin_key,
                'signal': analysis.get('signal'),
                'confidence': analysis.get('confidence'),
                'threshold_single': CONFIDENCE_THRESHOLD_SINGLE,
                'threshold_multiple': CONFIDENCE_THRESHOLD_MULTIPLE,
                'above_threshold': analysis.get('confidence', 0) >= CONFIDENCE_THRESHOLD_SINGLE,
                'strategies': analysis.get('strategies_analysis', {})
            })
    
    return {
        "recent_signals": recent_signals,
        "execute_trades": EXECUTE_TRADES,
        "executor_url": EXECUTOR_BOT_URL,
        "confidence_thresholds": {
            "single": CONFIDENCE_THRESHOLD_SINGLE,
            "multiple": CONFIDENCE_THRESHOLD_MULTIPLE
        }
    }

@app.get("/test-executor")
async def test_executor_connection():
    """اختبار اتصال البوت المنفذ"""
    try:
        health = await executor_client.health_check()
        return {
            "executor_connected": health,
            "execute_trades_enabled": EXECUTE_TRADES,
            "executor_url": EXECUTOR_BOT_URL,
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        return {"status": "error", "message": str(e)}

# =============================================================================
# تشغيل التطبيق المحسن
# =============================================================================

@app.on_event("startup")
async def startup_event():
    logger.info("🚀 بدء تشغيل مولد الإشارات المتقدم المحسن")
    logger.info(f"🎯 الاستراتيجيات: {list(signal_engine.strategies.keys())}")
    logger.info(f"💰 العملات المدعومة: {list(SUPPORTED_COINS.keys())}")
    logger.info(f"📡 مصادر البيانات: Binance + CoinGecko + CryptoCompare")
    logger.info(f"⏰ الإطار الزمني: {TIMEFRAME}")
    logger.info(f"📊 عتبة الإشارة الواحدة: {CONFIDENCE_THRESHOLD_SINGLE}%")
    logger.info(f"📈 عتبة الإشارات المتعددة: {CONFIDENCE_THRESHOLD_MULTIPLE}%")
    logger.info(f"🔍 فاصل المسح: {SCAN_INTERVAL} ثانية")
    logger.info(f"⚡ تنفيذ الصفقات: {'✅ مفعل' if EXECUTE_TRADES else '❌ معطل'}")
    
    # فحص اتصال البوت المنفذ
    await check_executor_connection()
    
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
