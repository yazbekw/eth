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
# إعدادات التداول المحسنة - إصدار أكثر ذكاءً وتحليلاً
# =============================================================================

SYMBOL = os.getenv("TRADING_SYMBOL", "BNBUSDT")
TIMEFRAME = os.getenv("TRADING_TIMEFRAME", "1h")
STOP_LOSS_PERCENT = float(os.getenv("STOP_LOSS_PERCENT", "0.5"))
TAKE_PROFIT_PERCENT = float(os.getenv("TAKE_PROFIT_PERCENT", "3.5"))
TRADE_SIZE_USDT = float(os.getenv("TRADE_SIZE_USDT", "100"))
LEVERAGE = int(os.getenv("LEVERAGE", "10"))
INITIAL_BALANCE = float(os.getenv("INITIAL_BALANCE", "5000.0"))

# عتبات محسنة لزيادة الصفقات وتحسين الأداء
BUY_CONFIDENCE_THRESHOLD = int(os.getenv("BUY_CONFIDENCE_THRESHOLD", "68"))  # مخفضة
SELL_CONFIDENCE_THRESHOLD = int(os.getenv("SELL_CONFIDENCE_THRESHOLD", "65"))  # مخفضة بشكل كبير
SELL_PREMIUM_THRESHOLD = int(os.getenv("SELL_PREMIUM_THRESHOLD", "70"))  # مخفضة
SELL_QUALITY_THRESHOLD = int(os.getenv("SELL_QUALITY_THRESHOLD", "60"))  # مخفضة بشكل كبير

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
logger = logging.getLogger("Enhanced_Sell_Strategy_v7_Smart")

# =============================================================================
# هياكل البيانات المحسنة مع تحليل الخسائر
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
    loss_analysis: Dict  # تحليل الخسائر الجديد

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
# استراتيجية الانزياح الحجمي المحسنة - إصدار أكثر ذكاءً وتحليلاً
# =============================================================================

class EnhancedSellStrategy:
    """استراتيجية محسنة مع تحليل الخسائر وتحسينات شاملة"""
    
    def __init__(self, telegram_notifier: TelegramNotifier = None):
        self.name = "enhanced_sell_strategy_v7_smart"
        self.trades: List[Trade] = []
        self.balance = INITIAL_BALANCE
        self.current_balance = INITIAL_BALANCE
        self.positions = {}
        self.trade_history = []
        self.analysis_results = []
        self.telegram_notifier = telegram_notifier
        self.df_global = None
        
        # إحصائيات متقدمة مع تحليل الخسائر
        self.sell_stats = {
            'standard_sell': {'trades': 0, 'wins': 0, 'total_pnl': 0},
            'premium_sell': {'trades': 0, 'wins': 0, 'total_pnl': 0},
            'ultra_sell': {'trades': 0, 'wins': 0, 'total_pnl': 0}
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
    
    def calculate_enhanced_sell_divergence(self, prices: List[float], volumes: List[float]) -> Dict[str, Any]:
        """انزياح بيع محسن بشروط أكثر ذكاءً ومرونة"""
        if len(prices) < 35:  # مخفضة من 50 لزيادة الحساسية
            return {"divergence": "none", "strength": 0, "sell_category": "NONE"}
        
        try:
            # تحليل مبسط وأكثر مرونة
            trend_5 = (self.safe_get_price(prices, -1) - self.safe_get_price(prices, -5)) / self.safe_get_price(prices, -5)
            trend_10 = (self.safe_get_price(prices, -1) - self.safe_get_price(prices, -10)) / self.safe_get_price(prices, -10)
            trend_20 = (self.safe_get_price(prices, -1) - self.safe_get_price(prices, -20)) / self.safe_get_price(prices, -20)
            
            current_volume = self.safe_get_volume(volumes, -1)
            avg_volume_10 = np.mean(volumes[-10:]) if len(volumes) >= 10 else current_volume
            avg_volume_20 = np.mean(volumes[-20:]) if len(volumes) >= 20 else current_volume
            volume_ratio_10 = current_volume / avg_volume_10 if avg_volume_10 > 0 else 1
            volume_ratio_20 = current_volume / avg_volume_20 if avg_volume_20 > 0 else 1
            
            # قوة الاتجاه الهبوطي
            bearish_strength = abs(min(0, trend_10, trend_20))
            
            # 1. بيع متميز - شروط مخففة
            if (trend_20 > 0.025 and                   # مخفض من 0.05
                trend_5 < -0.015 and                   # مخفض من 0.02
                volume_ratio_20 > 1.7 and              # مخفض من 2.0
                volume_ratio_10 > 2.0 and              # مخفض من 2.5
                current_volume > np.percentile(volumes[-50:], 70)):  # مخفض من 80%
                
                strength = min(90, int(
                    bearish_strength * 1800 + 
                    (volume_ratio_20 - 1) * 32 +
                    abs(trend_5) * 1100
                ))
                return {"divergence": "bearish_reversal", "strength": strength, "sell_category": "ULTRA"}
            
            # 2. بيع عالي الجودة - شروط مخففة
            elif (trend_20 > 0.015 and                  # مخفض من 0.03
                  trend_10 < -0.01 and                  # مخفض من 0.015
                  volume_ratio_20 > 1.5 and             # مخفض من 1.8
                  volume_ratio_10 > 1.8 and             # مخفض من 2.0
                  current_volume > np.percentile(volumes[-50:], 65)):  # مخفض من 75%
                
                strength = min(80, int(
                    bearish_strength * 1400 + 
                    (volume_ratio_20 - 1) * 28 +
                    abs(trend_10) * 900
                ))
                return {"divergence": "bearish_reversal", "strength": strength, "sell_category": "PREMIUM"}
            
            # 3. بيع قياسي - شروط مخففة بشكل كبير
            elif (trend_20 > 0.008 and                  # مخفض من 0.02
                  trend_5 < -0.008 and                  # مخفض من 0.01
                  volume_ratio_20 > 1.3 and             # مخفض من 1.5
                  volume_ratio_10 > 1.5 and             # مخفض من 1.8
                  current_volume > np.percentile(volumes[-50:], 60)):  # مخفض من 70%
                
                strength = min(70, int(
                    bearish_strength * 1100 + 
                    (volume_ratio_20 - 1) * 22 +
                    abs(trend_5) * 700
                ))
                return {"divergence": "bearish_reversal", "strength": strength, "sell_category": "STANDARD"}
            
            # 4. بيع حجمي سريع - إضافة محسنة
            elif (trend_10 < -0.018 and                 # مخفض من 0.02
                  volume_ratio_20 > 2.2 and             # مخفض من 2.5
                  volume_ratio_10 > volume_ratio_20 and # تسارع حجمي
                  current_volume > np.percentile(volumes[-50:], 75)):
                
                strength = min(68, int(
                    abs(trend_10) * 1300 + 
                    (volume_ratio_20 - 1) * 18
                ))
                return {"divergence": "volume_sell", "strength": strength, "sell_category": "STANDARD"}
            
            return {"divergence": "none", "strength": 0, "sell_category": "NONE"}
            
        except Exception as e:
            logger.error(f"❌ خطأ في حساب الانزياح البيعي: {e}")
            return {"divergence": "none", "strength": 0, "sell_category": "NONE"}
    
    def calculate_buy_divergence(self, prices: List[float], volumes: List[float]) -> Dict[str, Any]:
        """انزياح شراء محسن - أكثر مرونة"""
        if len(prices) < 30:  # مخفضة من 40
            return {"divergence": "none", "strength": 0}
        
        try:
            trend_10 = (self.safe_get_price(prices, -1) - self.safe_get_price(prices, -10)) / self.safe_get_price(prices, -10)
            trend_20 = (self.safe_get_price(prices, -1) - self.safe_get_price(prices, -20)) / self.safe_get_price(prices, -20)
            
            current_volume = self.safe_get_volume(volumes, -1)
            avg_volume_20 = np.mean(volumes[-20:]) if len(volumes) >= 20 else current_volume
            volume_ratio_20 = current_volume / avg_volume_20 if avg_volume_20 > 0 else 1
            
            # شروط شراء مخففة
            if (trend_20 < -0.015 and                   # مخفض من 0.02
                volume_ratio_20 > 1.6 and               # مخفض من 1.8
                current_volume > np.percentile(volumes[-80:], 70)):  # مخفض من 75%
                
                strength = min(78, int(abs(trend_20) * 1400 + (volume_ratio_20 - 1) * 28))
                return {"divergence": "bullish_reversal", "strength": strength}
            
            # شراء حجمي سريع
            elif (trend_20 < -0.008 and
                  volume_ratio_20 > 2.3 and
                  current_volume > np.percentile(volumes[-80:], 80)):
                
                strength = min(72, int(abs(trend_20) * 1100 + (volume_ratio_20 - 1) * 22))
                return {"divergence": "volume_buy", "strength": strength}
            
            return {"divergence": "none", "strength": 0}
            
        except Exception as e:
            logger.error(f"❌ خطأ في حساب الانزياح الشرائي: {e}")
            return {"divergence": "none", "strength": 0}
    
    def calculate_trend_strength(self, prices: List[float]) -> float:
        """حساب قوة الاتجاه - إصدار محسن"""
        if len(prices) < 15:  # مخفضة من 20
            return 0.5
        
        try:
            short_trend = (self.safe_get_price(prices, -1) - self.safe_get_price(prices, -3)) / self.safe_get_price(prices, -3)
            medium_trend = (self.safe_get_price(prices, -1) - self.safe_get_price(prices, -8)) / self.safe_get_price(prices, -8)
            long_trend = (self.safe_get_price(prices, -1) - self.safe_get_price(prices, -15)) / self.safe_get_price(prices, -15)
            
            # متوسط مرجح للاتجاهات مع تركيز على المدى القصير
            trend_strength = (abs(short_trend) * 0.5 + abs(medium_trend) * 0.3 + abs(long_trend) * 0.2)
            direction = -1 if (short_trend + medium_trend) < 0 else 1
            
            return trend_strength * direction
            
        except Exception as e:
            logger.error(f"❌ خطأ في حساب قوة الاتجاه: {e}")
            return 0.5
    
    def calculate_volume_surge(self, volumes: List[float]) -> float:
        """حساب قوة طفرة الحجم - إصدار محسن"""
        if len(volumes) < 8:  # مخفضة من 10
            return 0
        
        try:
            current_volume = self.safe_get_volume(volumes, -1)
            avg_volume_5 = np.mean(volumes[-5:]) if len(volumes) >= 5 else current_volume
            avg_volume_8 = np.mean(volumes[-8:]) if len(volumes) >= 8 else current_volume
            
            volume_surge_5 = (current_volume - avg_volume_5) / avg_volume_5 if avg_volume_5 > 0 else 0
            volume_surge_8 = (current_volume - avg_volume_8) / avg_volume_8 if avg_volume_8 > 0 else 0
            
            return max(0, (volume_surge_5 + volume_surge_8) / 2)
            
        except Exception as e:
            logger.error(f"❌ خطأ في حساب طفرة الحجم: {e}")
            return 0
    
    def calculate_sell_quality_score(self, df_row: pd.Series, divergence_data: Dict, 
                                   df: pd.DataFrame, current_index: int) -> float:
        """حساب درجة الجودة للصفقات البيعية - أكثر ذكاءً"""
        quality_score = 0
        
        try:
            # 1. جودة الحجم (25 نقطة) - مخفضة
            volume_ratio = df_row.get('volume_ratio_20', 1)
            volume_score = min(25, (volume_ratio - 1) * 12)
            quality_score += volume_score
            
            # 2. قوة الانزياح (20 نقطة) - مخفضة
            divergence_strength = min(20, divergence_data.get("strength", 0) / 4)
            quality_score += divergence_strength
            
            # 3. قوة الاتجاه الهبوطي (15 نقطة) - مخفضة
            if current_index >= 15:
                prices = df['close'].iloc[:current_index+1].tolist()
                trend_strength = abs(self.calculate_trend_strength(prices))
                if trend_strength < 0:  # اتجاه هبوطي
                    trend_score = min(15, abs(trend_strength) * 300)
                    quality_score += trend_score
            
            # 4. طفرة الحجم (15 نقطة) - مخفضة
            if current_index >= 8:
                volumes = df['volume'].iloc[:current_index+1].tolist()
                volume_surge = self.calculate_volume_surge(volumes)
                surge_score = min(15, volume_surge * 60)
                quality_score += surge_score
            
            # 5. استقرار السعر (10 نقطة) - محسنة
            if current_index >= 10:
                try:
                    recent_volatility = df['close'].iloc[current_index-5:current_index].std()
                    medium_volatility = df['close'].iloc[current_index-10:current_index].std()
                    if recent_volatility < medium_volatility * 0.85:
                        quality_score += 10
                except:
                    pass
        
        except Exception as e:
            logger.error(f"❌ خطأ في حساب جودة البيع: {e}")
        
        # مكافأة للبيع المتميز - مخفضة
        try:
            sell_category = divergence_data.get("sell_category", "NONE")
            if sell_category == "ULTRA":
                quality_score += 8
            elif sell_category == "PREMIUM":
                quality_score += 5
        except:
            pass
        
        return min(100, quality_score)
    
    def calculate_buy_quality_score(self, df_row: pd.Series, divergence_data: Dict, 
                                  df: pd.DataFrame, current_index: int) -> float:
        """حساب درجة الجودة للصفقات الشرائية - إصدار محسن"""
        quality_score = 0
        
        try:
            volume_ratio = df_row.get('volume_ratio_20', 1)
            volume_score = min(30, (volume_ratio - 1) * 15)  # مخفضة
            quality_score += volume_score
            
            divergence_strength = min(25, divergence_data.get("strength", 0) / 4)
            quality_score += divergence_strength
            
            # إضافة تحليل الاستقرار
            if current_index >= 12:
                try:
                    volume_stability = df['volume'].iloc[current_index-8:current_index].std()
                    price_stability = df['close'].iloc[current_index-8:current_index].std()
                    if volume_stability < df['volume'].std() and price_stability < df['close'].std():
                        quality_score += 20
                except:
                    pass
            
            # مكافأة على قوة الانزياح
            if divergence_data.get("strength", 0) > 70:
                quality_score += 10
        
        except Exception as e:
            logger.error(f"❌ خطأ في حساب جودة الشراء: {e}")
        
        return min(100, quality_score)
    
    def enhanced_sell_confidence_system(self, divergence_data: Dict, quality_score: float) -> float:
        """نظام ثقة محسن للبيع - أكثر ذكاءً ومرونة"""
        
        try:
            base_confidence = divergence_data.get("strength", 0)
            
            # مضاعفات أكثر توازناً
            category_multipliers = {
                "ULTRA": 1.18,      # مخفضة
                "PREMIUM": 1.08,    # مخفضة  
                "STANDARD": 1.0,
                "NONE": 0.85        # مرنة أكثر
            }
            
            multiplier = category_multipliers.get(divergence_data.get("sell_category", "NONE"), 1.0)
            adjusted_confidence = base_confidence * multiplier
            
            # تعزيز معتدل حسب الجودة
            quality_boost = quality_score / 100
            adjusted_confidence *= (1 + quality_boost * 0.25)  # مخفضة
            
            # عقوبة مخففة جداً
            if quality_score < 45:  # مخفضة من 50
                adjusted_confidence *= 0.92  # مخففة
            
            return min(95, adjusted_confidence)
            
        except Exception as e:
            logger.error(f"❌ خطأ في نظام الثقة: {e}")
            return 0
    
    def dynamic_sell_risk_management(self, sell_category: str, quality_score: float) -> Tuple[float, float]:
        """إدارة مخاطرة ديناميكية للبيع - أكثر ذكاءً"""
        
        try:
            base_sl = STOP_LOSS_PERCENT
            base_tp = TAKE_PROFIT_PERCENT
            
            # إعدادات أكثر ذكاءً
            risk_adjustments = {
                "ULTRA": (0.75, 2.8),    # أكثر توازناً
                "PREMIUM": (0.85, 2.3),  # أكثر توازناً
                "STANDARD": (0.95, 1.8)  # أكثر توازناً
            }
            
            sl_multiplier, tp_multiplier = risk_adjustments.get(sell_category, (1.0, 1.0))
            
            # تعديل ذكي حسب الجودة
            quality_factor = quality_score / 100
            sl_multiplier *= (1.0 - quality_factor * 0.12)  # أكثر توازناً
            tp_multiplier *= (0.92 + quality_factor * 0.18)  # أكثر توازناً
            
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
            
            return df
            
        except Exception as e:
            logger.error(f"❌ خطأ في حساب مؤشرات الحجم: {e}")
            return df
    
    def generate_enhanced_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        """توليد إشارات محسنة مع شروط تنفيذ أكثر ذكاءً"""
    
        buy_signals = []
        sell_signals = []
        buy_confidence_scores = []
        sell_confidence_scores = []
        buy_quality_scores = []
        sell_quality_scores = []
        sell_categories = []
        trend_strengths = []
        volume_surges = []
    
        for i in range(len(df)):
            try:
                if i < 25:  # مخفضة من 40 لزيادة الصفقات
                    buy_signals.append('none')
                    sell_signals.append('none')
                    buy_confidence_scores.append(0)
                    sell_confidence_scores.append(0)
                    buy_quality_scores.append(0)
                    sell_quality_scores.append(0)
                    sell_categories.append('NONE')
                    trend_strengths.append(0)
                    volume_surges.append(0)
                    continue
            
                # استخراج البيانات
                prices = df['close'].iloc[:i+1].tolist()
                volumes = df['volume'].iloc[:i+1].tolist()
            
                # إشارات البيع المحسنة
                sell_divergence = self.calculate_enhanced_sell_divergence(prices, volumes)
                buy_divergence = self.calculate_buy_divergence(prices, volumes)
                
                # حساب قوة الاتجاه وطفرة الحجم
                trend_strength = self.calculate_trend_strength(prices)
                volume_surge = self.calculate_volume_surge(volumes)
                
                trend_strengths.append(trend_strength)
                volume_surges.append(volume_surge)
            
                # معالجة إشارات البيع - شروط أكثر ذكاءً
                sell_signal = 'none'
                sell_confidence = 0
                sell_quality = 0
                
                if sell_divergence["divergence"] != "none":
                    sell_quality = self.calculate_sell_quality_score(df.iloc[i], sell_divergence, df, i)
                    sell_confidence = self.enhanced_sell_confidence_system(sell_divergence, sell_quality)
                    
                    # شروط البيع المخففة
                    if (sell_confidence >= SELL_CONFIDENCE_THRESHOLD and 
                        sell_quality >= SELL_QUALITY_THRESHOLD):
                        
                        # شروط مخففة حسب الفئة
                        if sell_divergence["sell_category"] == "ULTRA":
                            sell_signal = "SELL"
                        elif sell_divergence["sell_category"] == "PREMIUM" and sell_confidence >= SELL_PREMIUM_THRESHOLD:
                            sell_signal = "SELL"
                        elif sell_divergence["sell_category"] == "STANDARD" and sell_quality >= 55:  # مخفضة من 60
                            sell_signal = "SELL"
                
                # معالجة إشارات الشراء - شروط مخففة
                buy_signal = 'none'
                buy_confidence = 0
                buy_quality = 0
                
                if buy_divergence["divergence"] != "none":
                    buy_quality = self.calculate_buy_quality_score(df.iloc[i], buy_divergence, df, i)
                    buy_confidence = buy_divergence["strength"]
                    
                    if (buy_confidence >= BUY_CONFIDENCE_THRESHOLD and 
                        buy_quality >= 60):  # مخفضة من 65
                        buy_signal = "BUY"
            
                buy_signals.append(buy_signal)
                sell_signals.append(sell_signal)
                buy_confidence_scores.append(buy_confidence)
                sell_confidence_scores.append(sell_confidence)
                buy_quality_scores.append(buy_quality)
                sell_quality_scores.append(sell_quality)
                sell_categories.append(sell_divergence["sell_category"])
                
            except Exception as e:
                logger.error(f"❌ خطأ في توليد الإشارات للمؤشر {i}: {e}")
                buy_signals.append('none')
                sell_signals.append('none')
                buy_confidence_scores.append(0)
                sell_confidence_scores.append(0)
                buy_quality_scores.append(0)
                sell_quality_scores.append(0)
                sell_categories.append('NONE')
                trend_strengths.append(0)
                volume_surges.append(0)
    
        df['buy_signal'] = buy_signals
        df['sell_signal'] = sell_signals
        df['buy_confidence'] = buy_confidence_scores
        df['sell_confidence'] = sell_confidence_scores
        df['buy_quality'] = buy_quality_scores
        df['sell_quality'] = sell_quality_scores
        df['sell_category'] = sell_categories
        df['trend_strength'] = trend_strengths
        df['volume_surge'] = volume_surges
    
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
        """حساب حجم المركز - إصدار محسن"""
        base_size = (TRADE_SIZE_USDT * LEVERAGE) / price
        
        # تعديل الحجم حسب الثقة والاتجاه
        confidence_factor = confidence / 100
        if direction == "SELL":
            # حجم متوازن للبيع
            adjusted_size = base_size * (0.85 + confidence_factor * 0.3)  # محسنة
        else:
            adjusted_size = base_size * (0.85 + confidence_factor * 0.3)  # محسنة
        
        return adjusted_size
    
    def open_position(self, symbol: str, direction: str, price: float, 
                     confidence: float, quality_score: float, 
                     sell_category: str, volume_ratio: float, 
                     trend_strength: float, volume_surge: float,
                     timestamp: datetime) -> Optional[Trade]:
        """فتح مركز جديد مع تحسينات"""
        
        if symbol in self.positions:
            return None
        
        # حساب حجم المركز
        quantity = self.calculate_position_size(price, confidence, direction)
        
        # إدارة مخاطرة ديناميكية
        if direction == "SELL":
            sl_percent, tp_percent = self.dynamic_sell_risk_management(sell_category, quality_score)
            stop_loss = price * (1 + sl_percent / 100)
            take_profit = price * (1 - tp_percent / 100)
        else:
            sl_percent, tp_percent = (STOP_LOSS_PERCENT, TAKE_PROFIT_PERCENT)
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
            sell_category=sell_category,
            trend_strength=trend_strength,
            volume_surge=volume_surge,
            max_profit_reached=0,
            max_loss_reached=0
        )
        
        self.positions[symbol] = trade
        self.trades.append(trade)
        
        logger.info(f"🎯 فتح مركز {direction} محسن لـ {symbol}")
        logger.info(f"   الثقة: {confidence:.1f}% | الجودة: {quality_score:.1f}%")
        if direction == "SELL":
            logger.info(f"   فئة البيع: {sell_category} | الوقف: {sl_percent:.1f}% | الجني: {tp_percent:.1f}%")
        
        return trade
    
    def update_trade_stats(self, symbol: str, current_price: float):
        """تحديث إحصائيات الصفقة أثناء فتحها"""
        if symbol not in self.positions:
            return
        
        trade = self.positions[symbol]
        
        # حساب الربح/الخسارة الحالي
        if trade.direction == "BUY":
            current_pnl = (current_price - trade.entry_price) * trade.quantity
            current_pnl_percent = (current_pnl / (trade.quantity * trade.entry_price)) * 100
        else:
            current_pnl = (trade.entry_price - current_price) * trade.quantity
            current_pnl_percent = (current_pnl / (trade.quantity * trade.entry_price)) * 100
        
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
        
        # تحديث إحصائيات البيع
        if trade.direction == "SELL" and trade.sell_category in self.sell_stats:
            stats = self.sell_stats[trade.sell_category]
            stats['trades'] += 1
            stats['total_pnl'] += pnl
            if pnl > 0:
                stats['wins'] += 1
        
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
            'duration_minutes': trade.duration_minutes
        }
        
        # إضافة quantity فقط إذا كان موجوداً
        if trade.quantity is not None:
            trade_record['quantity'] = trade.quantity
        
        self.trade_history.append(trade_record)
        
        status_emoji = "🟢" if pnl > 0 else "🔴"
        logger.info(f"📊 إغلاق مركز {trade.direction} لـ {symbol} {status_emoji}"
                   f" الربح: {pnl:.2f} USD ({pnl_percent:.2f}%)")
        if pnl < 0:
            logger.info(f"   سبب الخسارة: {loss_reason}")
            logger.info(f"   أقصى ربح تم تحقيقه: {trade.max_profit_reached:.2f}")
            logger.info(f"   أقصى خسارة تم تحقيقها: {trade.max_loss_reached:.2f}")
        
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
    
    def execute_enhanced_trading(self, df: pd.DataFrame):
        """تنفيذ التداول المحسن"""
        
        logger.info("🚀 بدء التداول المحسن مع التحسينات الذكية...")
        
        for i, row in df.iterrows():
            if i < 25:  # مخفضة من 40 لزيادة الصفقات
                continue
                
            current_price = row['close']
            buy_signal = row['buy_signal']
            sell_signal = row['sell_signal']
            buy_confidence = row['buy_confidence']
            sell_confidence = row['sell_confidence']
            buy_quality = row['buy_quality']
            sell_quality = row['sell_quality']
            sell_category = row['sell_category']
            volume_ratio = row['volume_ratio_20']
            trend_strength = row['trend_strength']
            volume_surge = row['volume_surge']
            timestamp = row['timestamp']
            
            # فحص شروط الخروج
            if SYMBOL in self.positions:
                self.check_stop_conditions(SYMBOL, current_price, timestamp)
            
            # فتح مراكز جديدة بشروط مخففة
            if SYMBOL not in self.positions:
                if sell_signal == "SELL":
                    self.open_position(
                        SYMBOL, "SELL", current_price, sell_confidence, sell_quality,
                        sell_category, volume_ratio, trend_strength, volume_surge, timestamp
                    )
                elif buy_signal == "BUY":
                    self.open_position(
                        SYMBOL, "BUY", current_price, buy_confidence, buy_quality,
                        "NONE", volume_ratio, trend_strength, volume_surge, timestamp
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
            'loss_by_duration': {'short': 0, 'medium': 0, 'long': 0}
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
            if duration < 60:  # أقل من ساعة
                analysis['loss_by_duration']['short'] += 1
            elif duration < 240:  # أقل من 4 ساعات
                analysis['loss_by_duration']['medium'] += 1
            else:
                analysis['loss_by_duration']['long'] += 1
        
        analysis['avg_loss_amount'] = total_loss / len(losing_trades)
        analysis['max_loss_amount'] = max_loss
        
        return analysis
    
    def calculate_enhanced_results(self, df: pd.DataFrame) -> BacktestResult:
        """حساب النتائج المحسنة مع تحليل الخسائر"""
        
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
                loss_analysis={}
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
                'duration_minutes': trade.get('duration_minutes', 0)
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
        
        # تحليل البيع المتقدم
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
            'buy_avg_pnl': (sum(t.get('pnl', 0) for t in buy_trades) / len(buy_trades)) if len(buy_trades) > 0 else 0
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
            loss_analysis=loss_analysis
        )
    
    def run_enhanced_backtest(self, df: pd.DataFrame) -> BacktestResult:
        """تشغيل الباك-تستينغ المحسن"""
        
        logger.info("🔍 بدء الباك-تستينغ المحسن v7 مع التحليل الذكي...")
        
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
        self.loss_analysis = {
            'stop_loss_hits': 0,
            'take_profit_hits': 0,
            'end_of_data_closes': 0,
            'manual_closes': 0,
            'loss_reasons': {},
            'avg_loss_duration': 0,
            'avg_win_duration': 0
        }
        
        # حفظ البيانات العالمية
        self.df_global = df.copy()
        
        # التحليل المحسن
        df_with_signals = self.enhanced_analysis(df)
        
        # تنفيذ التداول المحسن
        self.execute_enhanced_trading(df_with_signals)
        
        # إغلاق المراكز المفتوحة
        if SYMBOL in self.positions:
            last_price = df_with_signals.iloc[-1]['close']
            last_timestamp = df_with_signals.iloc[-1]['timestamp']
            self.close_position(SYMBOL, last_price, last_timestamp, "END_OF_DATA")
        
        return self.calculate_enhanced_results(df_with_signals)
    
    async def send_enhanced_report(self, backtest_result: BacktestResult):
        """إرسال تقرير محسن مع تحليل الخسائر"""
        
        if not self.telegram_notifier:
            return
        
        try:
            # التقرير النصي
            report_text = self._generate_enhanced_report_text(backtest_result)
            await self.telegram_notifier.send_message(report_text)
                
        except Exception as e:
            logger.error(f"❌ خطأ في إرسال التقرير: {e}")
    
    def _generate_enhanced_report_text(self, backtest_result: BacktestResult) -> str:
        """إنشاء تقرير نصي محسن مع تحليل الخسائر"""
        
        report_time = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        
        message = f"🎯 تقرير استراتيجية المحسنة v7 - إصدار ذكي وتحليلي\n"
        message += "══════════════════════════════════════\n\n"
        
        message += f"⚙️ الإعدادات الذكية v7:\n"
        message += f"• العملة: `{SYMBOL}`\n"
        message += f"• الإطار: `{TIMEFRAME}`\n"
        message += f"• الرافعة: `{LEVERAGE}x`\n"
        message += f"• حجم الصفقة: `${TRADE_SIZE_USDT}`\n"
        message += f"• عتبة ثقة الشراء: `{BUY_CONFIDENCE_THRESHOLD}%`\n"
        message += f"• عتبة ثقة البيع: `{SELL_CONFIDENCE_THRESHOLD}%`\n"
        message += f"• عتبة البيع فائق الجودة: `{SELL_PREMIUM_THRESHOLD}%`\n"
        message += f"• عتبة البيع عالي الجودة: `{SELL_QUALITY_THRESHOLD}%`\n\n"
        
        message += f"📊 النتائج المحسنة v7:\n"
        message += f"• إجمالي الصفقات: `{backtest_result.total_trades}`\n"
        message += f"• الصفقات الرابحة: `{backtest_result.winning_trades}` 🟢\n"
        message += f"• الصفقات الخاسرة: `{backtest_result.losing_trades}` 🔴\n"
        message += f"• نسبة الربح: `{backtest_result.win_rate:.1f}%`\n"
        message += f"• إجمالي الربح: `${backtest_result.total_pnl:.2f}`\n"
        message += f"• الرصيد النهائي: `${backtest_result.final_balance:.2f}`\n"
        message += f"• العائد الإجمالي: `{((backtest_result.final_balance - INITIAL_BALANCE) / INITIAL_BALANCE * 100):.1f}%`\n"
        message += f"• متوسط الثقة: `{backtest_result.avg_confidence:.1f}%`\n\n"
        
        message += f"🎯 مقاييس المخاطرة المحسنة v7:\n"
        message += f"• أقصى خسارة: `{backtest_result.max_drawdown:.1f}%`\n"
        message += f"• متوسط الربح/صفقة: `${backtest_result.avg_trade:.2f}`\n"
        message += f"• أفضل صفقة: `${backtest_result.best_trade:.2f}` 🚀\n"
        message += f"• أسوأ صفقة: `${backtest_result.worst_trade:.2f}` 📉\n"
        message += f"• نسبة شارب: `{backtest_result.sharpe_ratio:.2f}`\n"
        message += f"• عامل الربحية: `{backtest_result.profit_factor:.2f}`\n\n"
        
        # تحليل الخسائر المفصل
        loss_analysis = backtest_result.loss_analysis
        if loss_analysis:
            message += f"🔍 تحليل مفصل للخسائر v7:\n"
            message += "────────────────────\n"
            message += f"• إجمالي الصفقات الخاسرة: `{loss_analysis.get('total_losing_trades', 0)}`\n"
            message += f"• متوسط مبلغ الخسارة: `${loss_analysis.get('avg_loss_amount', 0):.2f}`\n"
            message += f"• أقصى خسارة فردية: `${loss_analysis.get('max_loss_amount', 0):.2f}`\n\n"
            
            # أسباب الخسارة
            loss_reasons = loss_analysis.get('loss_reasons', {})
            if loss_reasons:
                message += f"📉 أسباب الخسارة:\n"
                for reason, count in loss_reasons.items():
                    percentage = (count / loss_analysis['total_losing_trades']) * 100
                    message += f"• {reason}: `{count}` مرات (`{percentage:.1f}%`)\n"
                message += "\n"
            
            # الخسارة حسب الاتجاه
            loss_by_direction = loss_analysis.get('loss_by_direction', {})
            if loss_by_direction:
                message += f"📊 الخسارة حسب الاتجاه:\n"
                for direction, count in loss_by_direction.items():
                    if count > 0:
                        percentage = (count / loss_analysis['total_losing_trades']) * 100
                        message += f"• {direction}: `{count}` صفقات (`{percentage:.1f}%`)\n"
                message += "\n"
            
            # الخسارة حسب الثقة
            loss_by_confidence = loss_analysis.get('loss_by_confidence', {})
            if loss_by_confidence:
                message += f"🎯 الخسارة حسب مستوى الثقة:\n"
                for level, count in loss_by_confidence.items():
                    if count > 0:
                        percentage = (count / loss_analysis['total_losing_trades']) * 100
                        message += f"• ثقة {level}: `{count}` صفقات (`{percentage:.1f}%`)\n"
                message += "\n"
        
        message += f"🕒 وقت التقرير: `{report_time}`\n"
        message += "══════════════════════════════════════\n"
        message += "⚡ نظام التقييم v7 + شروط مخففة + تحليل الخسائر\n\n"
        
        message += f"🔍 تحليل مفصل للبيع والشراء v7:\n"
        message += "────────────────────\n"
        
        # تحليل الشراء
        buy_analysis = backtest_result.sell_analysis
        message += f"🔼 صفقات الشراء:\n"
        message += f"• العدد: `{buy_analysis['buy_total_trades']} صفقة`\n"
        message += f"• الربح: `${buy_analysis['buy_total_pnl']:.2f}` {'✅' if buy_analysis['buy_total_pnl'] > 0 else '❌'}\n"
        message += f"• متوسط الربح: `${buy_analysis['buy_avg_pnl']:.2f}`\n"
        message += f"• نسبة النجاح: `{buy_analysis['buy_win_rate']:.1f}%`\n\n"
        
        # تحليل البيع
        message += f"🔽 صفقات البيع المحسنة v7:\n"
        message += f"• العدد: `{buy_analysis['total_sell_trades']} صفقة`\n"
        message += f"• الربح: `${buy_analysis['sell_total_pnl']:.2f}` {'✅' if buy_analysis['sell_total_pnl'] > 0 else '❌'}\n"
        message += f"• متوسط الربح: `${buy_analysis['sell_avg_pnl']:.2f}`\n"
        message += f"• نسبة النجاح: `{buy_analysis['sell_win_rate']:.1f}%`\n"
        message += f"• متوسط الجودة: `{buy_analysis['sell_avg_quality']:.1f}%`\n"
        message += f"• متوسط الثقة: `{buy_analysis['sell_avg_confidence']:.1f}%`\n\n"
        
        # تحليل فئات البيع
        message += f"🎯 تحليل جودة البيع v7:\n"
        for category in ['standard_sell', 'premium_sell', 'ultra_sell']:
            if category in buy_analysis:
                cat_data = buy_analysis[category]
                emoji = "🟢" if cat_data['avg_pnl'] > 0 else "🔴"
                message += f"• {category.upper().replace('_', ' ')}: {cat_data['trades']} صفقات, نجاح: {cat_data['win_rate']:.1f}%, ربح: ${cat_data['total_pnl']:.2f} {emoji}\n"
        
        message += f"\n📊 مقارنة الأداء v7:\n"
        performance_diff = buy_analysis['sell_win_rate'] - buy_analysis['buy_win_rate']
        pnl_diff = buy_analysis['sell_total_pnl'] - buy_analysis['buy_total_pnl']
        message += f"• فرق النجاح: `{performance_diff:+.1f}%` {'✅' if performance_diff > 0 else '❌'}\n"
        message += f"• فرق الربح: `${pnl_diff:+.2f}` {'✅' if pnl_diff > 0 else '❌'}\n\n"
        
        # توصيات محسنة
        message += f"🎯 توصيات تحسين البيع v7:\n"
        if buy_analysis['total_sell_trades'] == 0:
            message += f"• تم تخفيف شروط البيع بشكل كبير ✅\n"
            message += f"• زيادة حساسية الانزياح البيعي 📊\n"
            message += f"• تحسين إدارة المخاطرة للبيع 🔧\n"
        elif buy_analysis['sell_win_rate'] < 40:
            message += f"• البيع يحتاج مزيداً من التحسين ⚠️\n"
            message += f"• مراجعة أسباب الخسارة المذكورة أعلاه 📉\n"
            message += f"• تعديل إعدادات المخاطرة للبيع 🔧\n"
        elif buy_analysis['sell_win_rate'] < 60:
            message += f"• أداء البيع مقبول ولكن يمكن تحسينه 📈\n"
            message += f"• التركيز على تحسين توقيت الدخول ⏰\n"
            message += f"• تحليل فئات البيع الأكثر ربحية 🎯\n"
        else:
            message += f"• أداء البيع ممتاز - الحفاظ على الإعدادات ✅\n"
            message += f"• يمكن زيادة حجم صفقات البيع 📈\n"
            message += f"• توسيع نطاق فئات البيع 🎯\n"
        
        message += f"\n📈 مستوى الثقة: {'مرتفع' if backtest_result.avg_confidence > 75 else 'متوسط' if backtest_result.avg_confidence > 60 else 'منخفض'} ({backtest_result.avg_confidence:.1f}%) {'✅' if backtest_result.avg_confidence > 70 else '⚠️'}\n"
        
        # معلومات البيانات
        if self.df_global is not None:
            data_period = f"📊 فترة البيانات المحسنة: {len(self.df_global)} شمعة من {self.df_global['timestamp'].min().date()} إلى {self.df_global['timestamp'].max().date()}"
            message += f"\n{data_period}\n"
        else:
            message += f"\n📊 فترة البيانات: معلومات غير متاحة\n"
        
        # تقييم نهائي
        if buy_analysis['total_sell_trades'] > 0 and buy_analysis['sell_win_rate'] > 50 and buy_analysis['sell_total_pnl'] > 0:
            final_msg = "✅ استراتيجية البيع المحسنة تعمل بشكل ممتاز"
        elif buy_analysis['total_sell_trades'] > 0 and buy_analysis['sell_win_rate'] > 40:
            final_msg = "⚠️ استراتيجية البيع المحسنة تعمل ولكن تحتاج تحسينات طفيفة"
        elif buy_analysis['total_sell_trades'] > 0:
            final_msg = "❌ استراتيجية البيع المحسنة تحتاج تحسينات جذرية"
        else:
            final_msg = "❌ استراتيجية البيع المحسنة لم تنتج أي صفقات"
        
        message += f"\n{final_msg}"
        
        return message

# =============================================================================
# الوظيفة الرئيسية المحسنة
# =============================================================================

async def main():
    """الوظيفة الرئيسية المحسنة"""
    
    logger.info("🚀 بدء تشغيل استراتيجية البيع المحسنة v7 - إصدار ذكي وتحليلي")
    
    telegram_notifier = TelegramNotifier(TELEGRAM_BOT_TOKEN, TELEGRAM_CHAT_ID)
    
    # جلب البيانات
    data_fetcher = DataFetcher()
    df = data_fetcher.fetch_historical_data(SYMBOL, TIMEFRAME, DATA_LIMIT)
    
    if df.empty:
        error_msg = "❌ فشل جلب البيانات. تأكد من اتصال الإنترنت وصحة اسم العملة."
        logger.error(error_msg)
        await telegram_notifier.send_message(error_msg)
        return
    
    # تشغيل الاستراتيجية المحسنة
    strategy = EnhancedSellStrategy(telegram_notifier)
    backtest_result = strategy.run_enhanced_backtest(df)
    
    # إرسال التقرير المحسن
    await strategy.send_enhanced_report(backtest_result)
    
    # حفظ النتائج
    if strategy.trade_history:
        # استخدام طريقة آمنة لحفظ البيانات
        safe_trades = []
        for trade in strategy.trade_history:
            safe_trade = {k: v for k, v in trade.items() if v is not None}
            safe_trades.append(safe_trade)
        
        trades_df = pd.DataFrame(safe_trades)
        filename = f'enhanced_sell_trades_v7_{SYMBOL}_{TIMEFRAME}_{datetime.now().strftime("%Y%m%d_%H%M")}.csv'
        trades_df.to_csv(filename, index=False)
        logger.info(f"💾 تم حفظ سجل الصفقات المحسن في {filename}")
    
    logger.info("✅ اكتمل تشغيل استراتيجية البيع المحسنة بنجاح")

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
