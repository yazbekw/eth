import pandas as pd
import numpy as np
import requests
import time
from datetime import datetime, timedelta
import os
from typing import Dict, List, Optional, Tuple, Any
import logging
import json
import matplotlib.pyplot as plt
import seaborn as sns
from dataclasses import dataclass
import asyncio
import aiohttp
from io import BytesIO
import base64

# =============================================================================
# إعدادات التداول المحسنة - إصدار متوازن للبيع والشراء
# =============================================================================

SYMBOL = os.getenv("TRADING_SYMBOL", "BNBUSDT")
TIMEFRAME = os.getenv("TRADING_TIMEFRAME", "1h")
STOP_LOSS_PERCENT = float(os.getenv("STOP_LOSS_PERCENT", "0.6"))
TAKE_PROFIT_PERCENT = float(os.getenv("TAKE_PROFIT_PERCENT", "3.0"))
TRADE_SIZE_USDT = float(os.getenv("TRADE_SIZE_USDT", "100"))
LEVERAGE = int(os.getenv("LEVERAGE", "10"))
INITIAL_BALANCE = float(os.getenv("INITIAL_BALANCE", "5000.0"))

# عتبات محسنة ومتوازنة للبيع والشراء
BUY_CONFIDENCE_THRESHOLD = int(os.getenv("BUY_CONFIDENCE_THRESHOLD", "65"))  # مخفضة
SELL_CONFIDENCE_THRESHOLD = int(os.getenv("SELL_CONFIDENCE_THRESHOLD", "62"))  # مخفضة بشكل كبير
SELL_PREMIUM_THRESHOLD = int(os.getenv("SELL_PREMIUM_THRESHOLD", "68"))  # مخفضة
SELL_QUALITY_THRESHOLD = int(os.getenv("SELL_QUALITY_THRESHOLD", "58"))  # مخفضة بشكل كبير

# إعدادات مدة الاختبار
DATA_LIMIT = int(os.getenv("DATA_LIMIT", "1000"))

# إعدادات التلغرام
TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN", "")
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID", "")

# إعداد التسجيل
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger("Balanced_Strategy_v8")

# =============================================================================
# هياكل البيانات المحسنة
# =============================================================================

@dataclass
class Trade:
    symbol: str
    direction: str  # BUY or SELL
    entry_price: float
    entry_time: datetime
    exit_price: float = None
    exit_time: datetime = None
    quantity: float = None
    pnl: float = 0
    pnl_percent: float = 0
    confidence: float = 0
    confidence_level: str = ""
    stop_loss: float = None
    take_profit: float = None
    status: str = "OPEN"
    divergence_type: str = ""
    volume_ratio: float = 0
    quality_score: float = 0
    sell_category: str = ""  # STANDARD, PREMIUM, ULTRA
    trend_strength: float = 0
    volume_surge: float = 0
    loss_reason: str = ""  # سبب الخسارة
    max_profit_reached: float = 0  # أقصى ربح تم الوصول إليه
    max_loss_reached: float = 0  # أقصى خسارة تم الوصول إليها
    duration_minutes: int = 0  # مدة الصفقة
    market_condition: str = ""  # حالة السوق عند الدخول

@dataclass
class BacktestResult:
    total_trades: int
    winning_trades: int
    losing_trades: int
    win_rate: float
    total_pnl: float
    final_balance: float
    max_drawdown: float
    sharpe_ratio: float
    profit_factor: float
    avg_trade: float
    best_trade: float
    worst_trade: float
    total_fees: float
    total_days: int
    avg_daily_return: float
    avg_confidence: float
    divergence_analysis: Dict
    volume_analysis: Dict
    quality_analysis: Dict
    performance_metrics: Dict
    sell_analysis: Dict
    loss_analysis: Dict
    market_analysis: Dict  # تحليل حالة السوق

# =============================================================================
# نظام التلغرام
# =============================================================================

class TelegramNotifier:
    """نظام إرسال التقارير إلى التلغرام"""
    
    def __init__(self, bot_token: str, chat_id: str):
        self.bot_token = bot_token
        self.chat_id = chat_id
        self.base_url = f"https://api.telegram.org/bot{bot_token}"
    
    async def send_message(self, text: str, parse_mode: str = "Markdown") -> bool:
        """إرسال رسالة نصية"""
        if not self.bot_token or not self.chat_id:
            logger.warning("❌ إعدادات التلغرام غير مكتملة")
            return False
            
        try:
            payload = {
                'chat_id': self.chat_id,
                'text': text,
                'parse_mode': parse_mode,
                'disable_web_page_preview': True
            }
            
            async with aiohttp.ClientSession() as session:
                async with session.post(f"{self.base_url}/sendMessage", json=payload) as response:
                    if response.status == 200:
                        logger.info("✅ تم إرسال الرسالة إلى التلغرام")
                        return True
                    else:
                        logger.error(f"❌ فشل إرسال الرسالة: {response.status}")
                        return False
                        
        except Exception as e:
            logger.error(f"❌ خطأ في إرسال الرسالة: {e}")
            return False

# =============================================================================
# نظام جلب البيانات
# =============================================================================

class DataFetcher:
    """جلب البيانات من Binance"""
    
    @staticmethod
    def fetch_historical_data(symbol: str, interval: str, limit: int = DATA_LIMIT) -> pd.DataFrame:
        """جلب البيانات التاريخية من Binance"""
        try:
            url = f"https://api.binance.com/api/v3/klines"
            params = {
                'symbol': symbol,
                'interval': interval,
                'limit': limit
            }
            
            response = requests.get(url, params=params, timeout=15)
            data = response.json()
            
            df = pd.DataFrame(data, columns=[
                'timestamp', 'open', 'high', 'low', 'close', 'volume',
                'close_time', 'quote_asset_volume', 'number_of_trades',
                'taker_buy_base_asset_volume', 'taker_buy_quote_asset_volume', 'ignore'
            ])
            
            # تحويل الأنواع
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
            for col in ['open', 'high', 'low', 'close', 'volume']:
                df[col] = df[col].astype(float)
            
            logger.info(f"✅ تم جلب {len(df)} صف من البيانات لـ {symbol}")
            return df[['timestamp', 'open', 'high', 'low', 'close', 'volume']]
            
        except Exception as e:
            logger.error(f"❌ خطأ في جلب البيانات: {e}")
            return pd.DataFrame()

# =============================================================================
# استراتيجية متوازنة للبيع والشراء - إصدار محسن بالكامل
# =============================================================================

class BalancedTradingStrategy:
    """استراتيجية متوازنة مع تحسينات شاملة للبيع والشراء"""
    
    def __init__(self, telegram_notifier: TelegramNotifier = None):
        self.name = "balanced_trading_strategy_v8"
        self.trades: List[Trade] = []
        self.balance = INITIAL_BALANCE
        self.current_balance = INITIAL_BALANCE
        self.positions = {}
        self.trade_history = []
        self.analysis_results = []
        self.telegram_notifier = telegram_notifier
        self.df_global = None
        
        # إحصائيات متقدمة
        self.sell_stats = {
            'standard_sell': {'trades': 0, 'wins': 0, 'total_pnl': 0},
            'premium_sell': {'trades': 0, 'wins': 0, 'total_pnl': 0},
            'ultra_sell': {'trades': 0, 'wins': 0, 'total_pnl': 0}
        }
        
        self.buy_stats = {
            'standard_buy': {'trades': 0, 'wins': 0, 'total_pnl': 0},
            'premium_buy': {'trades': 0, 'wins': 0, 'total_pnl': 0},
            'ultra_buy': {'trades': 0, 'wins': 0, 'total_pnl': 0}
        }
        
        # تحليل الخسائر
        self.loss_analysis = {
            'stop_loss_hits': 0,
            'take_profit_hits': 0,
            'end_of_data_closes': 0,
            'manual_closes': 0,
            'loss_reasons': {},
            'avg_loss_duration': 0,
            'avg_win_duration': 0
        }
        
        # تحليل السوق
        self.market_analysis = {
            'trending_up': 0,
            'trending_down': 0,
            'ranging': 0,
            'high_volatility': 0,
            'low_volatility': 0
        }
    
    def safe_get_price(self, prices: List[float], index: int) -> float:
        """الحصول على سعر بشكل آمن مع التحقق من النطاق"""
        if len(prices) > abs(index):
            return prices[index]
        return prices[-1] if prices else 0
    
    def safe_get_volume(self, volumes: List[float], index: int) -> float:
        """الحصول على حجم بشكل آمن مع التحقق من النطاق"""
        if len(volumes) > abs(index):
            return volumes[index]
        return volumes[-1] if volumes else 0
    
    def analyze_market_condition(self, prices: List[float], volumes: List[float]) -> Dict[str, Any]:
        """تحليل حالة السوق بشكل متقدم"""
        if len(prices) < 30:
            return {"trend": "neutral", "volatility": "medium", "condition": "unknown"}
        
        try:
            # تحليل الاتجاه
            short_ma = np.mean(prices[-10:])
            medium_ma = np.mean(prices[-20:])
            long_ma = np.mean(prices[-30:])
            
            trend_strength = abs(short_ma - long_ma) / long_ma
            
            if short_ma > medium_ma > long_ma and trend_strength > 0.02:
                trend = "uptrend"
                self.market_analysis['trending_up'] += 1
            elif short_ma < medium_ma < long_ma and trend_strength > 0.02:
                trend = "downtrend"
                self.market_analysis['trending_down'] += 1
            else:
                trend = "ranging"
                self.market_analysis['ranging'] += 1
            
            # تحليل التقلبات
            volatility = np.std(prices[-20:]) / np.mean(prices[-20:])
            if volatility > 0.025:
                volatility_level = "high"
                self.market_analysis['high_volatility'] += 1
            elif volatility < 0.01:
                volatility_level = "low"
                self.market_analysis['low_volatility'] += 1
            else:
                volatility_level = "medium"
            
            # تحديد الحالة العامة
            if trend == "uptrend" and volatility_level == "medium":
                condition = "bullish"
            elif trend == "downtrend" and volatility_level == "medium":
                condition = "bearish"
            elif trend == "ranging" and volatility_level == "low":
                condition = "consolidation"
            else:
                condition = "volatile"
            
            return {
                "trend": trend,
                "volatility": volatility_level,
                "condition": condition,
                "trend_strength": trend_strength
            }
            
        except Exception as e:
            logger.error(f"❌ خطأ في تحليل السوق: {e}")
            return {"trend": "neutral", "volatility": "medium", "condition": "unknown"}
    
    def calculate_enhanced_sell_divergence(self, prices: List[float], volumes: List[float], market_condition: Dict) -> Dict[str, Any]:
        """انزياح بيع محسن بشروط ذكية حسب حالة السوق"""
        if len(prices) < 20:
            return {"divergence": "none", "strength": 0, "sell_category": "NONE"}
        
        try:
            current_price = prices[-1]
            prev_price = prices[-2]
            price_change = (current_price - prev_price) / prev_price
            
            current_volume = volumes[-1]
            avg_volume_10 = np.mean(volumes[-10:]) if len(volumes) >= 10 else current_volume
            volume_ratio = current_volume / avg_volume_10 if avg_volume_10 > 0 else 1
            
            # تحليل المقاومة
            resistance_level = max(prices[-15:]) if len(prices) >= 15 else current_price
            near_resistance = current_price >= resistance_level * 0.985
            
            # 1. بيع عند المقاومة مع حجم عالي - شروط مخففة
            if (near_resistance and
                volume_ratio > 1.6 and
                price_change < 0 and
                current_volume > np.percentile(volumes[-50:], 65)):
                
                strength = min(85, int(
                    (current_price / resistance_level - 1) * 1500 +
                    (volume_ratio - 1) * 25 +
                    abs(price_change) * 1000
                ))
                return {"divergence": "resistance_sell", "strength": strength, "sell_category": "ULTRA"}
            
            # 2. بيع انعكاسي سريع
            if len(prices) >= 10:
                recent_high = max(prices[-5:])
                pullback = (recent_high - current_price) / recent_high
                
                if (pullback > 0.015 and
                    volume_ratio > 1.8 and
                    price_change < -0.008):
                    
                    strength = min(80, int(
                        pullback * 2000 +
                        (volume_ratio - 1) * 20 +
                        abs(price_change) * 800
                    ))
                    return {"divergence": "pullback_sell", "strength": strength, "sell_category": "PREMIUM"}
            
            # 3. بيع اتجاهي في السوق الهبوطي
            if market_condition["trend"] == "downtrend":
                if (price_change < -0.005 and
                    volume_ratio > 1.4 and
                    current_price < np.mean(prices[-10:])):
                    
                    strength = min(75, int(
                        abs(price_change) * 1200 +
                        (volume_ratio - 1) * 18 +
                        market_condition["trend_strength"] * 500
                    ))
                    return {"divergence": "trend_sell", "strength": strength, "sell_category": "STANDARD"}
            
            # 4. بيع حجمي في أي حالة سوق
            if (volume_ratio > 2.2 and
                price_change < -0.01 and
                current_volume > np.percentile(volumes[-50:], 80)):
                
                strength = min(70, int(
                    abs(price_change) * 1000 +
                    (volume_ratio - 1) * 15
                ))
                return {"divergence": "volume_sell", "strength": strength, "sell_category": "STANDARD"}
            
            return {"divergence": "none", "strength": 0, "sell_category": "NONE"}
            
        except Exception as e:
            logger.error(f"❌ خطأ في حساب الانزياح البيعي: {e}")
            return {"divergence": "none", "strength": 0, "sell_category": "NONE"}
    
    def calculate_enhanced_buy_divergence(self, prices: List[float], volumes: List[float], market_condition: Dict) -> Dict[str, Any]:
        """انزياح شراء محسن بشروط ذكية حسب حالة السوق"""
        if len(prices) < 20:
            return {"divergence": "none", "strength": 0, "buy_category": "NONE"}
        
        try:
            current_price = prices[-1]
            prev_price = prices[-2]
            price_change = (current_price - prev_price) / prev_price
            
            current_volume = volumes[-1]
            avg_volume_10 = np.mean(volumes[-10:]) if len(volumes) >= 10 else current_volume
            volume_ratio = current_volume / avg_volume_10 if avg_volume_10 > 0 else 1
            
            # تحليل الدعم
            support_level = min(prices[-15:]) if len(prices) >= 15 else current_price
            near_support = current_price <= support_level * 1.015
            
            # 1. شراء عند الدعم مع حجم عالي
            if (near_support and
                volume_ratio > 1.7 and
                price_change > 0 and
                current_volume > np.percentile(volumes[-50:], 70)):
                
                strength = min(85, int(
                    (1 - current_price / support_level) * 1500 +
                    (volume_ratio - 1) * 25 +
                    abs(price_change) * 1000
                ))
                return {"divergence": "support_buy", "strength": strength, "buy_category": "ULTRA"}
            
            # 2. شراء انعكاسي من قاع
            if len(prices) >= 10:
                recent_low = min(prices[-5:])
                bounce = (current_price - recent_low) / recent_low
                
                if (bounce > 0.02 and
                    volume_ratio > 2.0 and
                    price_change > 0.01):
                    
                    strength = min(80, int(
                        bounce * 1800 +
                        (volume_ratio - 1) * 22 +
                        abs(price_change) * 900
                    ))
                    return {"divergence": "bounce_buy", "strength": strength, "buy_category": "PREMIUM"}
            
            # 3. شراء اتجاهي في السوق الصاعد
            if market_condition["trend"] == "uptrend":
                if (price_change > 0.003 and
                    volume_ratio > 1.5 and
                    current_price > np.mean(prices[-10:])):
                    
                    strength = min(75, int(
                        abs(price_change) * 1100 +
                        (volume_ratio - 1) * 20 +
                        market_condition["trend_strength"] * 400
                    ))
                    return {"divergence": "trend_buy", "strength": strength, "buy_category": "STANDARD"}
            
            # 4. شراء حجمي في أي حالة سوق
            if (volume_ratio > 2.5 and
                price_change > 0.015 and
                current_volume > np.percentile(volumes[-50:], 85)):
                
                strength = min(70, int(
                    abs(price_change) * 900 +
                    (volume_ratio - 1) * 18
                ))
                return {"divergence": "volume_buy", "strength": strength, "buy_category": "STANDARD"}
            
            return {"divergence": "none", "strength": 0, "buy_category": "NONE"}
            
        except Exception as e:
            logger.error(f"❌ خطأ في حساب الانزياح الشرائي: {e}")
            return {"divergence": "none", "strength": 0, "buy_category": "NONE"}
    
    def calculate_trend_strength(self, prices: List[float]) -> float:
        """حساب قوة الاتجاه - إصدار محسن"""
        if len(prices) < 10:
            return 0.5
        
        try:
            short_trend = (prices[-1] - prices[-3]) / prices[-3]
            medium_trend = (prices[-1] - prices[-8]) / prices[-8]
            
            trend_strength = (abs(short_trend) * 0.6 + abs(medium_trend) * 0.4)
            direction = 1 if (short_trend + medium_trend) > 0 else -1
            
            return trend_strength * direction
            
        except Exception as e:
            logger.error(f"❌ خطأ في حساب قوة الاتجاه: {e}")
            return 0.5
    
    def calculate_volume_surge(self, volumes: List[float]) -> float:
        """حساب قوة طفرة الحجم - إصدار محسن"""
        if len(volumes) < 5:
            return 0
        
        try:
            current_volume = volumes[-1]
            avg_volume_5 = np.mean(volumes[-5:])
            volume_surge = (current_volume - avg_volume_5) / avg_volume_5 if avg_volume_5 > 0 else 0
            
            return max(0, volume_surge)
            
        except Exception as e:
            logger.error(f"❌ خطأ في حساب طفرة الحجم: {e}")
            return 0
    
    def calculate_quality_score(self, df_row: pd.Series, divergence_data: Dict, 
                              df: pd.DataFrame, current_index: int, direction: str) -> float:
        """حساب درجة الجودة للصفقات - إصدار موحد للبيع والشراء"""
        quality_score = 0
        
        try:
            volume_ratio = df_row.get('volume_ratio_20', 1)
            
            # جودة الحجم (25 نقطة)
            volume_score = min(25, (volume_ratio - 1) * 12)
            quality_score += volume_score
            
            # قوة الانزياح (20 نقطة)
            divergence_strength = min(20, divergence_data.get("strength", 0) / 4)
            quality_score += divergence_strength
            
            # طفرة الحجم (15 نقطة)
            if current_index >= 5:
                volumes = df['volume'].iloc[:current_index+1].tolist()
                volume_surge = self.calculate_volume_surge(volumes)
                surge_score = min(15, volume_surge * 50)
                quality_score += surge_score
            
            # استقرار السعر (15 نقطة)
            if current_index >= 10:
                try:
                    recent_volatility = df['close'].iloc[current_index-5:current_index].std()
                    medium_volatility = df['close'].iloc[current_index-10:current_index].std()
                    if recent_volatility < medium_volatility * 0.9:
                        quality_score += 15
                except:
                    pass
            
            # محاذاة الاتجاه (15 نقطة)
            if current_index >= 15:
                prices = df['close'].iloc[:current_index+1].tolist()
                trend_strength = self.calculate_trend_strength(prices)
                
                if (direction == "BUY" and trend_strength > 0) or (direction == "SELL" and trend_strength < 0):
                    trend_score = min(15, abs(trend_strength) * 300)
                    quality_score += trend_score
            
            # مكافأة الفئة (10 نقطة)
            category = divergence_data.get(f"{direction.lower()}_category", "NONE")
            if category == "ULTRA":
                quality_score += 8
            elif category == "PREMIUM":
                quality_score += 5
        
        except Exception as e:
            logger.error(f"❌ خطأ في حساب الجودة: {e}")
        
        return min(100, quality_score)
    
    def enhanced_confidence_system(self, divergence_data: Dict, quality_score: float, direction: str) -> float:
        """نظام ثقة محسن للبيع والشراء"""
        
        try:
            base_confidence = divergence_data.get("strength", 0)
            
            # مضاعفات حسب الفئة
            category_multipliers = {
                "ULTRA": 1.15,
                "PREMIUM": 1.08,  
                "STANDARD": 1.0,
                "NONE": 0.9
            }
            
            category_key = f"{direction.lower()}_category"
            multiplier = category_multipliers.get(divergence_data.get(category_key, "NONE"), 1.0)
            adjusted_confidence = base_confidence * multiplier
            
            # تعزيز حسب الجودة
            quality_boost = quality_score / 100
            adjusted_confidence *= (1 + quality_boost * 0.2)
            
            # عقوبة مخففة للجودة المنخفضة
            if quality_score < 50:
                adjusted_confidence *= 0.95
            
            return min(95, adjusted_confidence)
            
        except Exception as e:
            logger.error(f"❌ خطأ في نظام الثقة: {e}")
            return 0
    
    def dynamic_risk_management(self, category: str, quality_score: float, direction: str, volatility: float) -> Tuple[float, float]:
        """إدارة مخاطرة ديناميكية للبيع والشراء"""
        
        try:
            base_sl = STOP_LOSS_PERCENT
            base_tp = TAKE_PROFIT_PERCENT
            
            # إعدادات حسب الاتجاه والفئة
            if direction == "SELL":
                risk_adjustments = {
                    "ULTRA": (0.7, 3.8),
                    "PREMIUM": (0.8, 3.2), 
                    "STANDARD": (0.9, 2.6)
                }
            else:  # BUY
                risk_adjustments = {
                    "ULTRA": (0.7, 3.5),
                    "PREMIUM": (0.8, 3.0),
                    "STANDARD": (0.9, 2.5)
                }
            
            sl_multiplier, tp_multiplier = risk_adjustments.get(category, (1.0, 1.0))
            
            # تعديل حسب الجودة والتقلبات
            quality_factor = quality_score / 100
            volatility_factor = 1.0 + (volatility * 1.5)  # زيادة المرونة مع التقلبات
            
            sl_multiplier *= (1.0 - quality_factor * 0.1) * volatility_factor
            tp_multiplier *= (0.95 + quality_factor * 0.15) * volatility_factor
            
            return base_sl * sl_multiplier, base_tp * tp_multiplier
            
        except Exception as e:
            logger.error(f"❌ خطأ في إدارة المخاطرة: {e}")
            return STOP_LOSS_PERCENT, TAKE_PROFIT_PERCENT
    
    def calculate_volume_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """حساب مؤشرات الحجم - إصدار محسن"""
        try:
            df['volume_ma_5'] = df['volume'].rolling(5, min_periods=1).mean()
            df['volume_ma_10'] = df['volume'].rolling(10, min_periods=1).mean()
            df['volume_ma_20'] = df['volume'].rolling(20, min_periods=1).mean()
            
            df['volume_ratio_5'] = df['volume'] / df['volume_ma_5'].replace(0, 1)
            df['volume_ratio_10'] = df['volume'] / df['volume_ma_10'].replace(0, 1)
            df['volume_ratio_20'] = df['volume'] / df['volume_ma_20'].replace(0, 1)
            
            # حساب التقلبات
            df['price_volatility'] = df['close'].rolling(10).std() / df['close'].rolling(10).mean()
            
            return df
            
        except Exception as e:
            logger.error(f"❌ خطأ في حساب مؤشرات الحجم: {e}")
            return df
    
    def generate_enhanced_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        """توليد إشارات محسنة مع تحليل السوق"""
    
        buy_signals = []
        sell_signals = []
        buy_confidence_scores = []
        sell_confidence_scores = []
        buy_quality_scores = []
        sell_quality_scores = []
        sell_categories = []
        buy_categories = []
        trend_strengths = []
        volume_surges = []
        market_conditions = []
    
        for i in range(len(df)):
            try:
                if i < 20:  # بداية سريعة
                    buy_signals.append('none')
                    sell_signals.append('none')
                    buy_confidence_scores.append(0)
                    sell_confidence_scores.append(0)
                    buy_quality_scores.append(0)
                    sell_quality_scores.append(0)
                    sell_categories.append('NONE')
                    buy_categories.append('NONE')
                    trend_strengths.append(0)
                    volume_surges.append(0)
                    market_conditions.append('unknown')
                    continue
            
                # استخراج البيانات
                prices = df['close'].iloc[:i+1].tolist()
                volumes = df['volume'].iloc[:i+1].tolist()
                
                # تحليل حالة السوق
                market_condition = self.analyze_market_condition(prices, volumes)
                market_conditions.append(market_condition["condition"])
            
                # إشارات البيع والشراء المحسنة
                sell_divergence = self.calculate_enhanced_sell_divergence(prices, volumes, market_condition)
                buy_divergence = self.calculate_enhanced_buy_divergence(prices, volumes, market_condition)
                
                # حساب قوة الاتجاه وطفرة الحجم
                trend_strength = self.calculate_trend_strength(prices)
                volume_surge = self.calculate_volume_surge(volumes)
                
                trend_strengths.append(trend_strength)
                volume_surges.append(volume_surge)
            
                # معالجة إشارات البيع
                sell_signal = 'none'
                sell_confidence = 0
                sell_quality = 0
                
                if sell_divergence["divergence"] != "none":
                    sell_quality = self.calculate_quality_score(df.iloc[i], sell_divergence, df, i, "SELL")
                    sell_confidence = self.enhanced_confidence_system(sell_divergence, sell_quality, "SELL")
                    
                    if (sell_confidence >= SELL_CONFIDENCE_THRESHOLD and 
                        sell_quality >= SELL_QUALITY_THRESHOLD):
                        
                        sell_signal = "SELL"
                
                # معالجة إشارات الشراء
                buy_signal = 'none'
                buy_confidence = 0
                buy_quality = 0
                
                if buy_divergence["divergence"] != "none":
                    buy_quality = self.calculate_quality_score(df.iloc[i], buy_divergence, df, i, "BUY")
                    buy_confidence = self.enhanced_confidence_system(buy_divergence, buy_quality, "BUY")
                    
                    if (buy_confidence >= BUY_CONFIDENCE_THRESHOLD and 
                        buy_quality >= 60):  # عتبة جودة مخففة للشراء
                        buy_signal = "BUY"
            
                buy_signals.append(buy_signal)
                sell_signals.append(sell_signal)
                buy_confidence_scores.append(buy_confidence)
                sell_confidence_scores.append(sell_confidence)
                buy_quality_scores.append(buy_quality)
                sell_quality_scores.append(sell_quality)
                sell_categories.append(sell_divergence["sell_category"])
                buy_categories.append(buy_divergence["buy_category"])
                
            except Exception as e:
                logger.error(f"❌ خطأ في توليد الإشارات للمؤشر {i}: {e}")
                buy_signals.append('none')
                sell_signals.append('none')
                buy_confidence_scores.append(0)
                sell_confidence_scores.append(0)
                buy_quality_scores.append(0)
                sell_quality_scores.append(0)
                sell_categories.append('NONE')
                buy_categories.append('NONE')
                trend_strengths.append(0)
                volume_surges.append(0)
                market_conditions.append('unknown')
    
        df['buy_signal'] = buy_signals
        df['sell_signal'] = sell_signals
        df['buy_confidence'] = buy_confidence_scores
        df['sell_confidence'] = sell_confidence_scores
        df['buy_quality'] = buy_quality_scores
        df['sell_quality'] = sell_quality_scores
        df['sell_category'] = sell_categories
        df['buy_category'] = buy_categories
        df['trend_strength'] = trend_strengths
        df['volume_surge'] = volume_surges
        df['market_condition'] = market_conditions
    
        return df
    
    def enhanced_analysis(self, df: pd.DataFrame) -> pd.DataFrame:
        """التحليل المحسن"""
        try:
            df = self.calculate_volume_indicators(df)
            df = self.generate_enhanced_signals(df)
            self.analysis_results = df.to_dict('records')
            return df
        except Exception as e:
            logger.error(f"❌ خطأ في التحليل المحسن: {e}")
            return df
    
    def calculate_position_size(self, price: float, confidence: float, direction: str) -> float:
        """حساب حجم المركز - إصدار متوازن"""
        base_size = (TRADE_SIZE_USDT * LEVERAGE) / price
        
        # تعديل الحجم حسب الثقة والاتجاه
        confidence_factor = confidence / 100
        
        if direction == "SELL":
            # حجم متوازن للبيع مع زيادة طفيفة
            adjusted_size = base_size * (0.9 + confidence_factor * 0.25)
        else:
            adjusted_size = base_size * (0.9 + confidence_factor * 0.25)
        
        return adjusted_size
    
    def open_position(self, symbol: str, direction: str, price: float, 
                     confidence: float, quality_score: float, 
                     category: str, volume_ratio: float, 
                     trend_strength: float, volume_surge: float,
                     timestamp: datetime, market_condition: str) -> Optional[Trade]:
        """فتح مركز جديد مع تحسينات شاملة"""
        
        if symbol in self.positions:
            return None
        
        # حساب حجم المركز
        quantity = self.calculate_position_size(price, confidence, direction)
        
        # إدارة مخاطرة ديناميكية
        volatility = abs(trend_strength)  # استخدام قوة الاتجاه كمؤشر للتقلبات
        sl_percent, tp_percent = self.dynamic_risk_management(category, quality_score, direction, volatility)
        
        if direction == "SELL":
            stop_loss = price * (1 + sl_percent / 100)
            take_profit = price * (1 - tp_percent / 100)
        else:
            stop_loss = price * (1 - sl_percent / 100)
            take_profit = price * (1 + tp_percent / 100)
        
        # رسوم التداول
        trade_value = quantity * price
        fee = trade_value * 0.0004
        self.current_balance -= fee
        
        trade = Trade(
            symbol=symbol,
            direction=direction,
            entry_price=price,
            entry_time=timestamp,
            quantity=quantity,
            confidence=confidence,
            stop_loss=stop_loss,
            take_profit=take_profit,
            status="OPEN",
            divergence_type="bearish_reversal" if direction == "SELL" else "bullish_reversal",
            volume_ratio=volume_ratio,
            quality_score=quality_score,
            sell_category=category if direction == "SELL" else "NONE",
            trend_strength=trend_strength,
            volume_surge=volume_surge,
            max_profit_reached=0,
            max_loss_reached=0,
            market_condition=market_condition
        )
        
        self.positions[symbol] = trade
        self.trades.append(trade)
        
        logger.info(f"🎯 فتح مركز {direction} متوازن لـ {symbol}")
        logger.info(f"   الثقة: {confidence:.1f}% | الجودة: {quality_score:.1f}% | السوق: {market_condition}")
        logger.info(f"   الفئة: {category} | الوقف: {sl_percent:.1f}% | الجني: {tp_percent:.1f}%")
        
        return trade
    
    def update_trade_stats(self, symbol: str, current_price: float):
        """تحديث إحصائيات الصفقة أثناء فتحها"""
        if symbol not in self.positions:
            return
        
        trade = self.positions[symbol]
        
        # حساب الربح/الخسارة الحالي
        if trade.direction == "BUY":
            current_pnl = (current_price - trade.entry_price) * trade.quantity
        else:
            current_pnl = (trade.entry_price - current_price) * trade.quantity
        
        # تحديث أقصى ربح وخسارة
        if current_pnl > trade.max_profit_reached:
            trade.max_profit_reached = current_pnl
        if current_pnl < trade.max_loss_reached:
            trade.max_loss_reached = current_pnl
    
    def close_position(self, symbol: str, price: float, timestamp: datetime, 
                      reason: str = "MANUAL") -> Optional[Trade]:
        """إغلاق مركز مفتوح مع تحليل مفصل"""
        
        if symbol not in self.positions:
            return None
        
        trade = self.positions[symbol]
        
        # حساب مدة الصفقة
        duration = (timestamp - trade.entry_time).total_seconds() / 60
        trade.duration_minutes = int(duration)
        
        # تحديث الإحصائيات النهائية
        self.update_trade_stats(symbol, price)
        
        # حساب الربح/الخسارة النهائي
        if trade.direction == "BUY":
            pnl = (price - trade.entry_price) * trade.quantity
        else:
            pnl = (trade.entry_price - price) * trade.quantity
        
        pnl_percent = (pnl / (trade.quantity * trade.entry_price)) * 100
        
        # رسوم الخروج
        trade_value = trade.quantity * price
        fee = trade_value * 0.0004
        pnl -= fee
        self.current_balance += pnl
        
        # تحديد سبب الخسارة
        loss_reason = ""
        if pnl < 0:
            if reason == "STOP_LOSS":
                loss_reason = "وقف الخسارة"
                self.loss_analysis['stop_loss_hits'] += 1
            elif reason == "END_OF_DATA":
                loss_reason = "نهاية البيانات"
                self.loss_analysis['end_of_data_closes'] += 1
            else:
                loss_reason = "إغلاق يدوي"
                self.loss_analysis['manual_closes'] += 1
            
            # تحديث إحصائيات أسباب الخسارة
            if loss_reason not in self.loss_analysis['loss_reasons']:
                self.loss_analysis['loss_reasons'][loss_reason] = 0
            self.loss_analysis['loss_reasons'][loss_reason] += 1
            
            # تحديث متوسط مدة الخسائر
            if self.loss_analysis['avg_loss_duration'] == 0:
                self.loss_analysis['avg_loss_duration'] = duration
            else:
                self.loss_analysis['avg_loss_duration'] = (self.loss_analysis['avg_loss_duration'] + duration) / 2
        else:
            self.loss_analysis['take_profit_hits'] += 1
            # تحديث متوسط مدة الأرباح
            if self.loss_analysis['avg_win_duration'] == 0:
                self.loss_analysis['avg_win_duration'] = duration
            else:
                self.loss_analysis['avg_win_duration'] = (self.loss_analysis['avg_win_duration'] + duration) / 2
        
        # تحديث بيانات الصفقة
        trade.exit_price = price
        trade.exit_time = timestamp
        trade.pnl = pnl
        trade.pnl_percent = pnl_percent
        trade.status = reason
        trade.loss_reason = loss_reason
        
        # تحديث إحصائيات البيع والشراء
        if trade.direction == "SELL" and trade.sell_category in self.sell_stats:
            stats = self.sell_stats[trade.sell_category]
            stats['trades'] += 1
            stats['total_pnl'] += pnl
            if pnl > 0:
                stats['wins'] += 1
        elif trade.direction == "BUY":
            # تحديث إحصائيات الشراء
            if pnl > 0:
                self.buy_stats['standard_buy']['wins'] += 1
            self.buy_stats['standard_buy']['trades'] += 1
            self.buy_stats['standard_buy']['total_pnl'] += pnl
        
        # إزالة من المراكز المفتوحة
        del self.positions[symbol]
        
        # حفظ في السجل
        trade_record = {
            'symbol': trade.symbol,
            'direction': trade.direction,
            'entry_price': trade.entry_price,
            'exit_price': trade.exit_price,
            'entry_time': trade.entry_time,
            'exit_time': trade.exit_time,
            'pnl': trade.pnl,
            'pnl_percent': trade.pnl_percent,
            'confidence': trade.confidence,
            'quality_score': trade.quality_score,
            'sell_category': trade.sell_category,
            'volume_ratio': trade.volume_ratio,
            'trend_strength': trade.trend_strength,
            'volume_surge': trade.volume_surge,
            'status': trade.status,
            'loss_reason': trade.loss_reason,
            'max_profit_reached': trade.max_profit_reached,
            'max_loss_reached': trade.max_loss_reached,
            'duration_minutes': trade.duration_minutes,
            'market_condition': trade.market_condition
        }
        
        if trade.quantity is not None:
            trade_record['quantity'] = trade.quantity
        
        self.trade_history.append(trade_record)
        
        status_emoji = "🟢" if pnl > 0 else "🔴"
        logger.info(f"📊 إغلاق مركز {trade.direction} لـ {symbol} {status_emoji}"
                   f" الربح: {pnl:.2f} USD ({pnl_percent:.2f}%)")
        
        return trade
    
    def check_stop_conditions(self, symbol: str, current_price: float, 
                            timestamp: datetime) -> bool:
        """فحص شروط الوقف والخروج مع تحديث الإحصائيات"""
        
        if symbol not in self.positions:
            return False
        
        trade = self.positions[symbol]
        
        # تحديث إحصائيات الصفقة
        self.update_trade_stats(symbol, current_price)
        
        if ((trade.direction == "BUY" and current_price <= trade.stop_loss) or
            (trade.direction == "SELL" and current_price >= trade.stop_loss)):
            self.close_position(symbol, trade.stop_loss, timestamp, "STOP_LOSS")
            return True
        
        if ((trade.direction == "BUY" and current_price >= trade.take_profit) or
            (trade.direction == "SELL" and current_price <= trade.take_profit)):
            self.close_position(symbol, trade.take_profit, timestamp, "TAKE_PROFIT")
            return True
        
        return False
    
    def execute_balanced_trading(self, df: pd.DataFrame):
        """تنفيذ التداول المتوازن"""
        
        logger.info("🚀 بدء التداول المتوازن مع تحسينات البيع والشراء...")
        
        for i, row in df.iterrows():
            if i < 20:
                continue
                
            current_price = row['close']
            buy_signal = row['buy_signal']
            sell_signal = row['sell_signal']
            buy_confidence = row['buy_confidence']
            sell_confidence = row['sell_confidence']
            buy_quality = row['buy_quality']
            sell_quality = row['sell_quality']
            sell_category = row['sell_category']
            buy_category = row['buy_category']
            volume_ratio = row['volume_ratio_20']
            trend_strength = row['trend_strength']
            volume_surge = row['volume_surge']
            market_condition = row['market_condition']
            timestamp = row['timestamp']
            
            # فحص شروط الخروج
            if SYMBOL in self.positions:
                self.check_stop_conditions(SYMBOL, current_price, timestamp)
            
            # فتح مراكز جديدة بشروط متوازنة
            if SYMBOL not in self.positions:
                if sell_signal == "SELL":
                    self.open_position(
                        SYMBOL, "SELL", current_price, sell_confidence, sell_quality,
                        sell_category, volume_ratio, trend_strength, volume_surge, timestamp, market_condition
                    )
                elif buy_signal == "BUY":
                    self.open_position(
                        SYMBOL, "BUY", current_price, buy_confidence, buy_quality,
                        buy_category, volume_ratio, trend_strength, volume_surge, timestamp, market_condition
                    )
    
    def analyze_losses(self) -> Dict:
        """تحليل مفصل للصفقات الخاسرة"""
        if not self.trade_history:
            return {}
        
        losing_trades = [t for t in self.trade_history if t.get('pnl', 0) < 0]
        
        analysis = {
            'total_losing_trades': len(losing_trades),
            'loss_reasons': {},
            'avg_loss_amount': 0,
            'max_loss_amount': 0,
            'loss_by_direction': {'BUY': 0, 'SELL': 0},
            'loss_by_confidence': {'high': 0, 'medium': 0, 'low': 0},
            'loss_by_duration': {'short': 0, 'medium': 0, 'long': 0},
            'loss_by_market_condition': {}
        }
        
        if not losing_trades:
            return analysis
        
        total_loss = 0
        max_loss = 0
        
        for trade in losing_trades:
            # إجمالي الخسائر
            loss_amount = abs(trade.get('pnl', 0))
            total_loss += loss_amount
            if loss_amount > max_loss:
                max_loss = loss_amount
            
            # أسباب الخسارة
            reason = trade.get('loss_reason', 'غير معروف')
            if reason not in analysis['loss_reasons']:
                analysis['loss_reasons'][reason] = 0
            analysis['loss_reasons'][reason] += 1
            
            # الخسارة حسب الاتجاه
            direction = trade.get('direction', '')
            if direction in analysis['loss_by_direction']:
                analysis['loss_by_direction'][direction] += 1
            
            # الخسارة حسب الثقة
            confidence = trade.get('confidence', 0)
            if confidence >= 70:
                analysis['loss_by_confidence']['high'] += 1
            elif confidence >= 50:
                analysis['loss_by_confidence']['medium'] += 1
            else:
                analysis['loss_by_confidence']['low'] += 1
            
            # الخسارة حسب المدة
            duration = trade.get('duration_minutes', 0)
            if duration < 60:
                analysis['loss_by_duration']['short'] += 1
            elif duration < 240:
                analysis['loss_by_duration']['medium'] += 1
            else:
                analysis['loss_by_duration']['long'] += 1
            
            # الخسارة حسب حالة السوق
            market_condition = trade.get('market_condition', 'unknown')
            if market_condition not in analysis['loss_by_market_condition']:
                analysis['loss_by_market_condition'][market_condition] = 0
            analysis['loss_by_market_condition'][market_condition] += 1
        
        analysis['avg_loss_amount'] = total_loss / len(losing_trades)
        analysis['max_loss_amount'] = max_loss
        
        return analysis
    
    def calculate_enhanced_results(self, df: pd.DataFrame) -> BacktestResult:
        """حساب النتائج المحسنة مع تحليل السوق"""
        
        if not self.trade_history:
            total_days = (df['timestamp'].max() - df['timestamp'].min()).days
            return BacktestResult(
                total_trades=0, winning_trades=0, losing_trades=0,
                win_rate=0, total_pnl=0, final_balance=self.current_balance,
                max_drawdown=0, sharpe_ratio=0, profit_factor=0,
                avg_trade=0, best_trade=0, worst_trade=0, total_fees=0,
                total_days=max(1, total_days), avg_daily_return=0,
                avg_confidence=0, divergence_analysis={}, volume_analysis={},
                quality_analysis={}, performance_metrics={}, sell_analysis={},
                loss_analysis={}, market_analysis=self.market_analysis
            )
        
        # إنشاء DataFrame من سجل التداول
        trades_data = []
        for trade in self.trade_history:
            trade_data = {
                'symbol': trade.get('symbol', ''),
                'direction': trade.get('direction', ''),
                'entry_price': trade.get('entry_price', 0),
                'exit_price': trade.get('exit_price', 0),
                'pnl': trade.get('pnl', 0),
                'confidence': trade.get('confidence', 0),
                'quality_score': trade.get('quality_score', 0),
                'volume_ratio': trade.get('volume_ratio', 0),
                'sell_category': trade.get('sell_category', 'NONE'),
                'loss_reason': trade.get('loss_reason', ''),
                'duration_minutes': trade.get('duration_minutes', 0),
                'market_condition': trade.get('market_condition', 'unknown')
            }
            trades_data.append(trade_data)
        
        trades_df = pd.DataFrame(trades_data)
        
        # المقاييس الأساسية
        total_trades = len(trades_df)
        winning_trades = len(trades_df[trades_df['pnl'] > 0])
        losing_trades = len(trades_df[trades_df['pnl'] < 0])
        win_rate = (winning_trades / total_trades) * 100 if total_trades > 0 else 0
        
        total_pnl = trades_df['pnl'].sum()
        final_balance = self.current_balance
        
        # أقصى خسارة متراكمة
        balance_history = [INITIAL_BALANCE]
        for trade in self.trade_history:
            balance_history.append(balance_history[-1] + trade['pnl'])
        
        peak = balance_history[0]
        max_dd = 0
        for value in balance_history:
            if value > peak:
                peak = value
            dd = (peak - value) / peak * 100
            if dd > max_dd:
                max_dd = dd
        
        # نسبة شارب
        pnl_values = [trade['pnl'] for trade in self.trade_history]
        avg_return = np.mean(pnl_values) if pnl_values else 0
        std_return = np.std(pnl_values) if len(pnl_values) > 1 else 0
        sharpe_ratio = avg_return / std_return if std_return > 0 else 0
        
        # عامل الربحية
        gross_profit = sum(pnl for pnl in pnl_values if pnl > 0)
        gross_loss = abs(sum(pnl for pnl in pnl_values if pnl < 0))
        profit_factor = gross_profit / gross_loss if gross_loss > 0 else float('inf')
        
        # إحصائيات أخرى
        avg_trade = avg_return
        best_trade = max(pnl_values) if pnl_values else 0
        worst_trade = min(pnl_values) if pnl_values else 0
        
        # حساب الرسوم
        total_fees = 0
        for trade in self.trade_history:
            entry_price = trade.get('entry_price', 0)
            exit_price = trade.get('exit_price', 0)
            quantity = trade.get('quantity', 0)
            
            if quantity > 0:
                entry_fee = quantity * entry_price * 0.0004
                exit_fee = quantity * exit_price * 0.0004
                total_fees += entry_fee + exit_fee
        
        # حساب عدد الأيام والعائد اليومي
        total_days = (df['timestamp'].max() - df['timestamp'].min()).days
        total_days = max(1, total_days)
        avg_daily_return = (final_balance - INITIAL_BALANCE) / INITIAL_BALANCE / total_days * 100
        
        # تحليل الثقة والجودة
        avg_confidence = trades_df['confidence'].mean() if not trades_df.empty else 0
        
        # تحليل البيع والشراء المتقدم
        sell_trades = [t for t in self.trade_history if t.get('direction') == 'SELL']
        buy_trades = [t for t in self.trade_history if t.get('direction') == 'BUY']
        
        sell_analysis = {
            'total_sell_trades': len(sell_trades),
            'sell_win_rate': (len([t for t in sell_trades if t.get('pnl', 0) > 0]) / len(sell_trades) * 100) if len(sell_trades) > 0 else 0,
            'sell_total_pnl': sum(t.get('pnl', 0) for t in sell_trades),
            'sell_avg_pnl': (sum(t.get('pnl', 0) for t in sell_trades) / len(sell_trades)) if len(sell_trades) > 0 else 0,
            'sell_avg_confidence': (sum(t.get('confidence', 0) for t in sell_trades) / len(sell_trades)) if len(sell_trades) > 0 else 0,
            'sell_avg_quality': (sum(t.get('quality_score', 0) for t in sell_trades) / len(sell_trades)) if len(sell_trades) > 0 else 0,
            'buy_total_trades': len(buy_trades),
            'buy_win_rate': (len([t for t in buy_trades if t.get('pnl', 0) > 0]) / len(buy_trades) * 100) if len(buy_trades) > 0 else 0,
            'buy_total_pnl': sum(t.get('pnl', 0) for t in buy_trades),
            'buy_avg_pnl': (sum(t.get('pnl', 0) for t in buy_trades) / len(buy_trades)) if len(buy_trades) > 0 else 0,
            'buy_avg_confidence': (sum(t.get('confidence', 0) for t in buy_trades) / len(buy_trades)) if len(buy_trades) > 0 else 0,
            'buy_avg_quality': (sum(t.get('quality_score', 0) for t in buy_trades) / len(buy_trades)) if len(buy_trades) > 0 else 0
        }
        
        # إضافة إحصائيات فئات البيع
        for category in ['standard_sell', 'premium_sell', 'ultra_sell']:
            if category in self.sell_stats:
                stats = self.sell_stats[category]
                sell_analysis[category] = {
                    'trades': stats['trades'],
                    'win_rate': (stats['wins'] / stats['trades'] * 100) if stats['trades'] > 0 else 0,
                    'total_pnl': stats['total_pnl'],
                    'avg_pnl': stats['total_pnl'] / stats['trades'] if stats['trades'] > 0 else 0
                }
        
        # تحليل الانزياح
        divergence_analysis = {
            'bullish_reversal': {
                'trades': len(buy_trades),
                'win_rate': sell_analysis['buy_win_rate'],
                'total_pnl': sell_analysis['buy_total_pnl'],
                'avg_pnl': sell_analysis['buy_avg_pnl']
            },
            'bearish_reversal': {
                'trades': len(sell_trades),
                'win_rate': sell_analysis['sell_win_rate'],
                'total_pnl': sell_analysis['sell_total_pnl'],
                'avg_pnl': sell_analysis['sell_avg_pnl']
            }
        }
        
        # تحليل الحجم
        volume_analysis = {
            'avg_volume_ratio': trades_df['volume_ratio'].mean() if not trades_df.empty else 0,
            'volume_correlation': trades_df['volume_ratio'].corr(trades_df['pnl']) if len(trades_df) > 1 else 0
        }
        
        # تحليل الجودة
        quality_analysis = {
            'avg_quality_score': trades_df['quality_score'].mean() if not trades_df.empty else 0,
            'quality_correlation': trades_df['quality_score'].corr(trades_df['pnl']) if len(trades_df) > 1 else 0
        }
        
        # مقاييس الأداء
        performance_metrics = {
            'risk_reward_ratio': abs(avg_trade / worst_trade) if worst_trade < 0 else 0,
            'expectancy': (win_rate/100 * avg_trade) - ((1 - win_rate/100) * abs(avg_trade)),
            'consistency_score': (win_rate * profit_factor) / 100 if profit_factor != float('inf') else 0
        }
        
        # تحليل الخسائر
        loss_analysis = self.analyze_losses()
        
        return BacktestResult(
            total_trades=total_trades,
            winning_trades=winning_trades,
            losing_trades=losing_trades,
            win_rate=win_rate,
            total_pnl=total_pnl,
            final_balance=final_balance,
            max_drawdown=max_dd,
            sharpe_ratio=sharpe_ratio,
            profit_factor=profit_factor,
            avg_trade=avg_trade,
            best_trade=best_trade,
            worst_trade=worst_trade,
            total_fees=total_fees,
            total_days=total_days,
            avg_daily_return=avg_daily_return,
            avg_confidence=avg_confidence,
            divergence_analysis=divergence_analysis,
            volume_analysis=volume_analysis,
            quality_analysis=quality_analysis,
            performance_metrics=performance_metrics,
            sell_analysis=sell_analysis,
            loss_analysis=loss_analysis,
            market_analysis=self.market_analysis
        )
    
    def run_balanced_backtest(self, df: pd.DataFrame) -> BacktestResult:
        """تشغيل الباك-تستينغ المتوازن"""
        
        logger.info("🔍 بدء الباك-تستينغ المتوازن v8...")
        
        # إعادة تعيين البيانات
        self.trades = []
        self.positions = {}
        self.trade_history = []
        self.current_balance = INITIAL_BALANCE
        self.sell_stats = {
            'standard_sell': {'trades': 0, 'wins': 0, 'total_pnl': 0},
            'premium_sell': {'trades': 0, 'wins': 0, 'total_pnl': 0},
            'ultra_sell': {'trades': 0, 'wins': 0, 'total_pnl': 0}
        }
        self.buy_stats = {
            'standard_buy': {'trades': 0, 'wins': 0, 'total_pnl': 0},
            'premium_buy': {'trades': 0, 'wins': 0, 'total_pnl': 0},
            'ultra_buy': {'trades': 0, 'wins': 0, 'total_pnl': 0}
        }
        self.loss_analysis = {
            'stop_loss_hits': 0,
            'take_profit_hits': 0,
            'end_of_data_closes': 0,
            'manual_closes': 0,
            'loss_reasons': {},
            'avg_loss_duration': 0,
            'avg_win_duration': 0
        }
        self.market_analysis = {
            'trending_up': 0,
            'trending_down': 0,
            'ranging': 0,
            'high_volatility': 0,
            'low_volatility': 0
        }
        
        # حفظ البيانات العالمية
        self.df_global = df.copy()
        
        # التحليل المحسن
        df_with_signals = self.enhanced_analysis(df)
        
        # تنفيذ التداول المتوازن
        self.execute_balanced_trading(df_with_signals)
        
        # إغلاق المراكز المفتوحة
        if SYMBOL in self.positions:
            last_price = df_with_signals.iloc[-1]['close']
            last_timestamp = df_with_signals.iloc[-1]['timestamp']
            self.close_position(SYMBOL, last_price, last_timestamp, "END_OF_DATA")
        
        return self.calculate_enhanced_results(df_with_signals)
    
    async def send_balanced_report(self, backtest_result: BacktestResult):
        """إرسال تقرير متوازن"""
        
        if not self.telegram_notifier:
            return
        
        try:
            report_text = self._generate_balanced_report_text(backtest_result)
            await self.telegram_notifier.send_message(report_text)
                
        except Exception as e:
            logger.error(f"❌ خطأ في إرسال التقرير: {e}")
    
    def _generate_balanced_report_text(self, backtest_result: BacktestResult) -> str:
        """إنشاء تقرير متوازن"""
        
        report_time = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        
        message = f"🎯 تقرير الاستراتيجية المتوازنة v8 - تحسينات البيع والشراء\n"
        message += "══════════════════════════════════════\n\n"
        
        message += f"⚙️ الإعدادات المتوازنة v8:\n"
        message += f"• العملة: `{SYMBOL}`\n"
        message += f"• الإطار: `{TIMEFRAME}`\n"
        message += f"• الرافعة: `{LEVERAGE}x`\n"
        message += f"• حجم الصفقة: `${TRADE_SIZE_USDT}`\n"
        message += f"• عتبة ثقة الشراء: `{BUY_CONFIDENCE_THRESHOLD}%`\n"
        message += f"• عتبة ثقة البيع: `{SELL_CONFIDENCE_THRESHOLD}%`\n"
        message += f"• عتبة البيع فائق الجودة: `{SELL_PREMIUM_THRESHOLD}%`\n"
        message += f"• عتبة البيع عالي الجودة: `{SELL_QUALITY_THRESHOLD}%`\n\n"
        
        message += f"📊 النتائج المتوازنة v8:\n"
        message += f"• إجمالي الصفقات: `{backtest_result.total_trades}`\n"
        message += f"• الصفقات الرابحة: `{backtest_result.winning_trades}` 🟢\n"
        message += f"• الصفقات الخاسرة: `{backtest_result.losing_trades}` 🔴\n"
        message += f"• نسبة الربح: `{backtest_result.win_rate:.1f}%`\n"
        message += f"• إجمالي الربح: `${backtest_result.total_pnl:.2f}`\n"
        message += f"• الرصيد النهائي: `${backtest_result.final_balance:.2f}`\n"
        message += f"• العائد الإجمالي: `{((backtest_result.final_balance - INITIAL_BALANCE) / INITIAL_BALANCE * 100):.1f}%`\n"
        message += f"• متوسط الثقة: `{backtest_result.avg_confidence:.1f}%`\n\n"
        
        message += f"🎯 مقاييس المخاطرة v8:\n"
        message += f"• أقصى خسارة: `{backtest_result.max_drawdown:.1f}%`\n"
        message += f"• متوسط الربح/صفقة: `${backtest_result.avg_trade:.2f}`\n"
        message += f"• أفضل صفقة: `${backtest_result.best_trade:.2f}` 🚀\n"
        message += f"• أسوأ صفقة: `${backtest_result.worst_trade:.2f}` 📉\n"
        message += f"• نسبة شارب: `{backtest_result.sharpe_ratio:.2f}`\n"
        message += f"• عامل الربحية: `{backtest_result.profit_factor:.2f}`\n\n"
        
        # تحليل السوق
        market_analysis = backtest_result.market_analysis
        total_conditions = sum(market_analysis.values())
        if total_conditions > 0:
            message += f"📈 تحليل حالة السوق v8:\n"
            message += f"• اتجاه صاعد: `{market_analysis['trending_up']}` مرات\n"
            message += f"• اتجاه هابط: `{market_analysis['trending_down']}` مرات\n"
            message += f"• سوق متذبذب: `{market_analysis['ranging']}` مرات\n"
            message += f"• تقلبات عالية: `{market_analysis['high_volatility']}` مرات\n"
            message += f"• تقلبات منخفضة: `{market_analysis['low_volatility']}` مرات\n\n"
        
        # تحليل البيع والشراء
        analysis = backtest_result.sell_analysis
        message += f"🔍 تحليل مفصل للبيع والشراء v8:\n"
        message += "────────────────────\n"
        
        message += f"🔼 صفقات الشراء:\n"
        message += f"• العدد: `{analysis['buy_total_trades']} صفقة`\n"
        message += f"• الربح: `${analysis['buy_total_pnl']:.2f}` {'✅' if analysis['buy_total_pnl'] > 0 else '❌'}\n"
        message += f"• متوسط الربح: `${analysis['buy_avg_pnl']:.2f}`\n"
        message += f"• نسبة النجاح: `{analysis['buy_win_rate']:.1f}%`\n"
        message += f"• متوسط الثقة: `{analysis['buy_avg_confidence']:.1f}%`\n"
        message += f"• متوسط الجودة: `{analysis['buy_avg_quality']:.1f}%`\n\n"
        
        message += f"🔽 صفقات البيع:\n"
        message += f"• العدد: `{analysis['total_sell_trades']} صفقة`\n"
        message += f"• الربح: `${analysis['sell_total_pnl']:.2f}` {'✅' if analysis['sell_total_pnl'] > 0 else '❌'}\n"
        message += f"• متوسط الربح: `${analysis['sell_avg_pnl']:.2f}`\n"
        message += f"• نسبة النجاح: `{analysis['sell_win_rate']:.1f}%`\n"
        message += f"• متوسط الثقة: `{analysis['sell_avg_confidence']:.1f}%`\n"
        message += f"• متوسط الجودة: `{analysis['sell_avg_quality']:.1f}%`\n\n"
        
        # تحليل فئات البيع
        message += f"🎯 تحليل جودة البيع v8:\n"
        for category in ['standard_sell', 'premium_sell', 'ultra_sell']:
            if category in analysis:
                cat_data = analysis[category]
                emoji = "🟢" if cat_data['avg_pnl'] > 0 else "🔴"
                message += f"• {category.upper().replace('_', ' ')}: {cat_data['trades']} صفقات, نجاح: {cat_data['win_rate']:.1f}%, ربح: ${cat_data['total_pnl']:.2f} {emoji}\n"
        
        message += f"\n📊 مقارنة الأداء v8:\n"
        performance_diff = analysis['sell_win_rate'] - analysis['buy_win_rate']
        pnl_diff = analysis['sell_total_pnl'] - analysis['buy_total_pnl']
        message += f"• فرق النجاح: `{performance_diff:+.1f}%` {'✅' if performance_diff > 0 else '❌'}\n"
        message += f"• فرق الربح: `${pnl_diff:+.2f}` {'✅' if pnl_diff > 0 else '❌'}\n\n"
        
        # توصيات
        message += f"🎯 توصيات التحسين v8:\n"
        if analysis['total_sell_trades'] == 0:
            message += f"• زيادة حساسية إشارات البيع 📈\n"
            message += f"• تخفيف شروط الدخول للبيع 🔧\n"
        elif analysis['sell_win_rate'] < 40:
            message += f"• مراجعة إدارة المخاطرة للبيع 📉\n"
            message += f"• تحسين توقيت دخول البيع ⏰\n"
        else:
            message += f"• أداء البيع مقبول - الحفاظ على الإعدادات ✅\n"
        
        if analysis['buy_win_rate'] < 40:
            message += f"• تحسين شروط الشراء مع تحليل السوق 📊\n"
            message += f"• مراجعة إعدادات وقف الخسارة للشراء 🔧\n"
        
        message += f"\n🕒 وقت التقرير: `{report_time}`\n"
        message += "══════════════════════════════════════\n"
        message += "⚡ نظام متوازن v8 + تحليل السوق + تحسينات شاملة\n"
        
        return message

# =============================================================================
# الوظيفة الرئيسية المحسنة
# =============================================================================

async def main():
    """الوظيفة الرئيسية المحسنة"""
    
    logger.info("🚀 بدء تشغيل الاستراتيجية المتوازنة v8...")
    
    telegram_notifier = TelegramNotifier(TELEGRAM_BOT_TOKEN, TELEGRAM_CHAT_ID)
    
    # جلب البيانات
    data_fetcher = DataFetcher()
    df = data_fetcher.fetch_historical_data(SYMBOL, TIMEFRAME, DATA_LIMIT)
    
    if df.empty:
        error_msg = "❌ فشل جلب البيانات. تأكد من اتصال الإنترنت وصحة اسم العملة."
        logger.error(error_msg)
        await telegram_notifier.send_message(error_msg)
        return
    
    # تشغيل الاستراتيجية المتوازنة
    strategy = BalancedTradingStrategy(telegram_notifier)
    backtest_result = strategy.run_balanced_backtest(df)
    
    # إرسال التقرير المتوازن
    await strategy.send_balanced_report(backtest_result)
    
    # حفظ النتائج
    if strategy.trade_history:
        safe_trades = []
        for trade in strategy.trade_history:
            safe_trade = {k: v for k, v in trade.items() if v is not None}
            safe_trades.append(safe_trade)
        
        trades_df = pd.DataFrame(safe_trades)
        filename = f'balanced_trades_v8_{SYMBOL}_{TIMEFRAME}_{datetime.now().strftime("%Y%m%d_%H%M")}.csv'
        trades_df.to_csv(filename, index=False)
        logger.info(f"💾 تم حفظ سجل الصفقات المتوازنة في {filename}")
    
    logger.info("✅ اكتمل تشغيل الاستراتيجية المتوازنة بنجاح")

def run_main():
    """تشغيل الدالة الرئيسية بشكل آمن"""
    try:
        asyncio.run(main())
    except RuntimeError as e:
        if "Event loop is closed" in str(e):
            logger.info("✅ اكتمل التشغيل بنجاح")
        else:
            logger.error(f"❌ خطأ غير متوقع: {e}")
    except Exception as e:
        logger.error(f"❌ خطأ في التشغيل: {e}")

if __name__ == "__main__":
    run_main()
