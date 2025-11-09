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
# إعدادات التداول المحسنة
# =============================================================================

SYMBOL = os.getenv("TRADING_SYMBOL", "BNBUSDT")
TIMEFRAME = os.getenv("TRADING_TIMEFRAME", "1h")
STOP_LOSS_PERCENT = float(os.getenv("STOP_LOSS_PERCENT", "0.6"))  # مخاطرة أقل
TAKE_PROFIT_PERCENT = float(os.getenv("TAKE_PROFIT_PERCENT", "2.8"))  # أرباح أعلى
TRADE_SIZE_USDT = float(os.getenv("TRADE_SIZE_USDT", "100"))
LEVERAGE = int(os.getenv("LEVERAGE", "8"))  # رافعة أقل
INITIAL_BALANCE = float(os.getenv("INITIAL_BALANCE", "5000.0"))
CONFIDENCE_THRESHOLD = int(os.getenv("CONFIDENCE_THRESHOLD", "80"))  # عتبة أعلى
QUALITY_THRESHOLD = int(os.getenv("QUALITY_THRESHOLD", "75"))  # جودة أعلى

# إعدادات مدة الاختبار
DATA_LIMIT = int(os.getenv("DATA_LIMIT", "1200"))

# إعدادات التلغرام
TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN", "")
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID", "")

# إعداد التسجيل
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger("Enhanced_Volume_Divergence_Strategy_v3")

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
    volume_confidence: float = 0
    trend_alignment: float = 0

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

# =============================================================================
# نظام التلغرام (متبقي كما هو)
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
    
    async def send_photo(self, photo_buffer: BytesIO, caption: str = "") -> bool:
        """إرسال صورة"""
        if not self.bot_token or not self.chat_id:
            logger.warning("❌ إعدادات التلغرام غير مكتملة")
            return False
            
        try:
            photo_buffer.seek(0)
            form_data = aiohttp.FormData()
            form_data.add_field('chat_id', self.chat_id)
            form_data.add_field('photo', photo_buffer, filename='chart.png')
            form_data.add_field('caption', caption)
            form_data.add_field('parse_mode', 'Markdown')
            
            async with aiohttp.ClientSession() as session:
                async with session.post(f"{self.base_url}/sendPhoto", data=form_data) as response:
                    if response.status == 200:
                        logger.info("✅ تم إرسال الصورة إلى التلغرام")
                        return True
                    else:
                        error_text = await response.text()
                        logger.error(f"❌ فشل إرسال الصورة: {response.status} - {error_text}")
                        return False
                        
        except Exception as e:
            logger.error(f"❌ خطأ في إرسال الصورة: {e}")
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
# استراتيجية الانزياح الحجمي المحسنة جداً
# =============================================================================

class EnhancedVolumeDivergenceStrategy:
    """استراتيجية الانزياح الحجمي المحسنة مع نظام تصفية متعدد الطبقات"""
    
    def __init__(self, telegram_notifier: TelegramNotifier = None):
        self.name = "enhanced_volume_divergence_v3"
        self.trades: List[Trade] = []
        self.balance = INITIAL_BALANCE
        self.current_balance = INITIAL_BALANCE
        self.positions = {}
        self.trade_history = []
        self.analysis_results = []
        self.telegram_notifier = telegram_notifier
        self.performance_stats = {
            'bullish_breakout': {'trades': 0, 'wins': 0},
            'bearish_reversal': {'trades': 0, 'wins': 0},
            'volume_surge': {'trades': 0, 'wins': 0},
            'trend_confirmation': {'trades': 0, 'wins': 0}
        }
        
        # إحصائيات الأداء التاريخي لأنواع الانزياح
        self.divergence_performance = {
            'bullish_breakout': {'total': 0, 'wins': 0, 'avg_pnl': 0},
            'bearish_reversal': {'total': 0, 'wins': 0, 'avg_pnl': 0},
            'volume_surge': {'total': 0, 'wins': 0, 'avg_pnl': 0},
            'trend_confirmation': {'total': 0, 'wins': 0, 'avg_pnl': 0}
        }
    
    def calculate_enhanced_divergence_v2(self, prices: List[float], volumes: List[float]) -> Dict[str, Any]:
        """انزياح محسن بشروط أكثر ذكاء وتشدداً"""
        if len(prices) < 50:
            return {"divergence": "none", "strength": 0, "volume_confidence": 0}
        
        # تحليل متقدم للاتجاه مع فترات متعددة
        trend_5 = (prices[-1] - prices[-5]) / prices[-5]
        trend_10 = (prices[-1] - prices[-10]) / prices[-10]
        trend_20 = (prices[-1] - prices[-20]) / prices[-20]
        trend_50 = (prices[-1] - prices[-50]) / prices[-50]
        
        # تحليل الحجم المتقدم مع نسب متعددة
        current_volume = volumes[-1]
        avg_volume_5 = np.mean(volumes[-5:])
        avg_volume_10 = np.mean(volumes[-10:])
        avg_volume_20 = np.mean(volumes[-20:])
        avg_volume_50 = np.mean(volumes[-50:])
        
        volume_ratio_5 = current_volume / avg_volume_5
        volume_ratio_10 = current_volume / avg_volume_10
        volume_ratio_20 = current_volume / avg_volume_20
        volume_ratio_50 = current_volume / avg_volume_50
        
        # ثقة الحجم (مزيج من النسب)
        volume_confidence = min(100, (
            volume_ratio_5 * 15 + 
            volume_ratio_10 * 25 + 
            volume_ratio_20 * 35 + 
            volume_ratio_50 * 25
        ) - 100)
        
        # 1. كسر صعودي قوي (شروط مشددة)
        if (trend_20 < -0.025 and                    # هبوط سابق قوي
            trend_5 > 0.015 and                      # بداية صعود حاد
            volume_ratio_20 > 2.2 and                # حجم عالي جداً
            volume_ratio_50 > 1.8 and                # تأكيد حجم طويل المدى
            current_volume > np.percentile(volumes[-100:], 85) and  # حجم في أعلى 15%
            volume_ratio_10 > volume_ratio_20):      # تسارع في الحجم
            
            strength = min(95, int(
                abs(trend_20) * 1800 + 
                (volume_ratio_20 - 1) * 35 +
                volume_confidence * 0.8
            ))
            return {"divergence": "bullish_breakout", "strength": strength, "volume_confidence": volume_confidence}
        
        # 2. انعكاس هبوطي قوي
        elif (trend_20 > 0.035 and                   # صعود سابق قوي
              trend_5 < -0.02 and                    # بداية هبوط حاد
              volume_ratio_20 > 2.0 and              # حجم عالي
              volume_ratio_50 > 1.6 and              # تأكيد حجم طويل المدى
              current_volume > np.percentile(volumes[-100:], 80) and  # حجم في أعلى 20%
              volume_ratio_10 > volume_ratio_20):    # تسارع في الحجم
            
            strength = min(95, int(
                abs(trend_20) * 1700 + 
                (volume_ratio_20 - 1) * 30 +
                volume_confidence * 0.8
            ))
            return {"divergence": "bearish_reversal", "strength": strength, "volume_confidence": volume_confidence}
        
        # 3. طفرة حجم مع تأكيد اتجاه
        elif (abs(trend_10) > 0.03 and 
              volume_ratio_20 > 2.5 and
              volume_ratio_5 > 3.0 and               # تسارع حجمي حاد
              current_volume > np.percentile(volumes[-100:], 90)):  # حجم في أعلى 10%
            
            strength = min(85, int(
                abs(trend_10) * 1400 + 
                (volume_ratio_20 - 1) * 25 +
                volume_confidence * 0.7
            ))
            divergence_type = "volume_surge_bullish" if trend_10 > 0 else "volume_surge_bearish"
            return {"divergence": "volume_surge", "strength": strength, "volume_confidence": volume_confidence}
        
        # 4. تأكيد اتجاه مع حجم مستمر
        elif (abs(trend_20) > 0.04 and
              volume_ratio_20 > 1.8 and
              volume_ratio_10 > 2.0 and
              np.mean(volumes[-5:]) > np.mean(volumes[-20:]) * 1.5):  # استمرار حجم عالي
            
            strength = min(80, int(
                abs(trend_20) * 1200 + 
                (volume_ratio_20 - 1) * 20 +
                volume_confidence * 0.6
            ))
            return {"divergence": "trend_confirmation", "strength": strength, "volume_confidence": volume_confidence}
        
        return {"divergence": "none", "strength": 0, "volume_confidence": 0}
    
    def calculate_trend_alignment(self, prices: List[float], current_price: float) -> float:
        """حساب محاذاة الاتجاه الحالي"""
        if len(prices) < 20:
            return 0.5
        
        # متوسطات متحركة متعددة
        ma_5 = np.mean(prices[-5:])
        ma_10 = np.mean(prices[-10:])
        ma_20 = np.mean(prices[-20:])
        
        # اتجاهات مختلفة
        trend_short = 1 if current_price > ma_5 else -1
        trend_medium = 1 if current_price > ma_10 else -1
        trend_long = 1 if current_price > ma_20 else -1
        
        # قوة الاتجاه
        trend_strength = (
            abs(current_price - ma_5) / ma_5 * 0.4 +
            abs(current_price - ma_10) / ma_10 * 0.3 +
            abs(current_price - ma_20) / ma_20 * 0.3
        )
        
        # توافق الاتجاهات
        alignment_score = (trend_short + trend_medium + trend_long) / 3.0
        
        return max(0, min(1, alignment_score * (1 + trend_strength)))
    
    def calculate_enhanced_quality_score(self, df_row: pd.Series, divergence_data: Dict, 
                                       df: pd.DataFrame, current_index: int) -> float:
        """حساب درجة الجودة المحسنة مع مراعاة المزيد من العوامل"""
        quality_score = 0
    
        # 1. جودة الحجم (35 نقطة) - أكثر أهمية
        volume_score = min(35, (df_row['volume_ratio_20'] - 1) * 17)
        quality_score += volume_score
    
        # 2. استقرار الحجم (20 نقطة)
        if current_index >= 15:
            volume_volatility_short = df['volume'].iloc[current_index-5:current_index].std() if current_index >= 5 else 0
            volume_volatility_long = df['volume'].iloc[current_index-15:current_index].std()
            
            if volume_volatility_short > 0 and volume_volatility_long > 0:
                stability_ratio = volume_volatility_long / volume_volatility_short
                if stability_ratio > 1.2:  # استقرار جيد
                    quality_score += 20
                elif stability_ratio > 0.8:  # استقرار مقبول
                    quality_score += 10
    
        # 3. قوة الانزياح (25 نقطة)
        divergence_strength = min(25, divergence_data["strength"] / 4)
        quality_score += divergence_strength
    
        # 4. محاذاة الاتجاه (20 نقطة)
        if current_index >= 20:
            prices = df['close'].iloc[:current_index+1].tolist()
            current_price = df_row['close']
            trend_alignment = self.calculate_trend_alignment(prices, current_price)
            trend_score = trend_alignment * 20
            
            # تعزيز إذا كان الانزياح في اتجاه الاتجاه الرئيسي
            if ((divergence_data["divergence"] in ["bullish_breakout", "volume_surge"] and trend_alignment > 0.6) or
                (divergence_data["divergence"] in ["bearish_reversal"] and trend_alignment < 0.4)):
                trend_score *= 1.2
            
            quality_score += min(20, trend_score)
    
        return min(100, quality_score)
    
    def enhanced_confidence_system_v2(self, divergence_data: Dict, quality_score: float) -> float:
        """نظام ثقة محسن مع تعلم من الأداء السابق"""
        
        base_confidence = divergence_data["strength"]
        
        # مضاعفات معدلة بناء على تحليل الأداء
        divergence_multipliers = {
            "bullish_breakout": 1.3,      # تحسين الأداء
            "bearish_reversal": 1.4,      # أعلى دقة
            "volume_surge": 1.1,          # أداء جيد
            "trend_confirmation": 0.9     # أداء متوسط
        }
        
        multiplier = divergence_multipliers.get(divergence_data["divergence"], 1.0)
        
        # تعديل بناء على ثقة الحجم
        volume_confidence_boost = divergence_data["volume_confidence"] / 100 * 0.3  # حتى 30% تعزيز
        multiplier *= (1 + volume_confidence_boost)
        
        adjusted_confidence = base_confidence * multiplier
        
        # تعزيز حسب جودة الإشارة
        quality_boost = quality_score / 100
        adjusted_confidence *= (1 + quality_boost * 0.5)  # حتى 50% تعزيز
        
        # عقوبة للجودة المنخفضة
        if quality_score < 60:
            adjusted_confidence *= 0.7
        
        return min(95, adjusted_confidence)  # حد أقصى 95% للواقعية
    
    def dynamic_risk_management_v2(self, divergence_type: str, quality_score: float, 
                                 volume_confidence: float) -> Tuple[float, float]:
        """إدارة مخاطرة ديناميكية محسنة"""
        
        # قاعدة الإعدادات المحسنة
        base_sl = STOP_LOSS_PERCENT
        base_tp = TAKE_PROFIT_PERCENT
        
        # تعديل حسب نوع الانزياح
        risk_adjustments = {
            "bullish_breakout": (0.7, 2.2),    # وقف أصغر، جني أكبر
            "bearish_reversal": (0.6, 2.5),    # وقف أصغر، جني أكبر
            "volume_surge": (0.9, 2.0),        # إعدادات متوازنة
            "trend_confirmation": (1.0, 1.8)   # إعدادات محافظة
        }
        
        sl_multiplier, tp_multiplier = risk_adjustments.get(divergence_type, (1.0, 1.0))
        
        # تعديل حسب جودة الإشارة
        quality_factor = quality_score / 100
        sl_multiplier *= (1.2 - quality_factor * 0.4)  # جودة عالية = وقف أصغر
        tp_multiplier *= (0.8 + quality_factor * 0.6)  # جودة عالية = جني أكبر
        
        # تعديل حسب ثقة الحجم
        volume_factor = volume_confidence / 100
        sl_multiplier *= (1.1 - volume_factor * 0.2)   # حجم عالي = وقف أصغر
        tp_multiplier *= (0.9 + volume_factor * 0.3)   # حجم عالي = جني أكبر
        
        return base_sl * sl_multiplier, base_tp * tp_multiplier
    
    def calculate_volume_indicators_v2(self, df: pd.DataFrame) -> pd.DataFrame:
        """حساب مؤشرات الحجم المحسنة"""
        # المتوسطات المتحركة للحجم بفترات متعددة
        df['volume_ma_5'] = df['volume'].rolling(5).mean()
        df['volume_ma_10'] = df['volume'].rolling(10).mean()
        df['volume_ma_20'] = df['volume'].rolling(20).mean()
        df['volume_ma_50'] = df['volume'].rolling(50).mean()
        
        # نسب الحجم المحسنة
        df['volume_ratio_5'] = df['volume'] / df['volume_ma_5']
        df['volume_ratio_10'] = df['volume'] / df['volume_ma_10']
        df['volume_ratio_20'] = df['volume'] / df['volume_ma_20']
        df['volume_ratio_50'] = df['volume'] / df['volume_ma_50']
        
        # مؤشرات حجم متقدمة
        df['volume_momentum'] = df['volume_ratio_20'] - df['volume_ratio_20'].shift(3)
        df['volume_acceleration'] = df['volume_momentum'] - df['volume_momentum'].shift(2)
        
        # مؤشرات التراكم
        df['volume_trend'] = df['volume_ratio_20'].rolling(8).mean()
        
        return df
    
    def generate_enhanced_signals_v2(self, df: pd.DataFrame) -> pd.DataFrame:
        """توليد إشارات محسنة بشروط أكثر تشدداً"""
    
        signals = []
        confidence_scores = []
        divergence_types = []
        quality_scores = []
        volume_confidences = []
        trend_alignments = []
    
        for i in range(len(df)):
            if i < 50:  # تحتاج إلى بيانات أكثر للتحليل المتقدم
                signals.append('none')
                confidence_scores.append(0)
                divergence_types.append('none')
                quality_scores.append(0)
                volume_confidences.append(0)
                trend_alignments.append(0)
                continue
        
            # استخراج البيانات
            prices = df['close'].iloc[:i+1].tolist()
            volumes = df['volume'].iloc[:i+1].tolist()
        
            # حساب الانزياح المحسن
            divergence_data = self.calculate_enhanced_divergence_v2(prices, volumes)
        
            if divergence_data["divergence"] == "none":
                signals.append('none')
                confidence_scores.append(0)
                divergence_types.append('none')
                quality_scores.append(0)
                volume_confidences.append(0)
                trend_alignments.append(0)
                continue
        
            # حساب درجة الجودة المحسنة
            quality_score = self.calculate_enhanced_quality_score(df.iloc[i], divergence_data, df, i)
            
            # حساب محاذاة الاتجاه
            trend_alignment = self.calculate_trend_alignment(prices, df.iloc[i]['close'])
        
            # حساب الثقة المحسنة
            confidence = self.enhanced_confidence_system_v2(divergence_data, quality_score)
        
            # تحديد الإشارة بشروط مشددة جداً
            signal = 'none'
            if (confidence >= CONFIDENCE_THRESHOLD and 
                quality_score >= QUALITY_THRESHOLD and
                divergence_data["volume_confidence"] >= 60):
                
                # شروط إضافية للشراء
                if divergence_data["divergence"] in ["bullish_breakout", "volume_surge"]:
                    if trend_alignment > 0.4:  # محاذاة اتجاه معقولة
                        signal = "BUY"
                        
                # شروط إضافية للبيع  
                elif divergence_data["divergence"] in ["bearish_reversal"]:
                    if trend_alignment < 0.6:  # محاذاة اتجاه معقولة
                        signal = "SELL"
        
            signals.append(signal)
            confidence_scores.append(confidence)
            divergence_types.append(divergence_data["divergence"])
            quality_scores.append(quality_score)
            volume_confidences.append(divergence_data["volume_confidence"])
            trend_alignments.append(trend_alignment)
    
        df['volume_signal'] = signals
        df['volume_confidence'] = confidence_scores
        df['divergence_type'] = divergence_types
        df['quality_score'] = quality_scores
        df['volume_confidence_score'] = volume_confidences
        df['trend_alignment'] = trend_alignments
    
        return df
    
    def enhanced_volume_analysis_v2(self, df: pd.DataFrame) -> pd.DataFrame:
        """التحليل الحجمي المحسن"""
        
        # 1. حساب مؤشرات الحجم المحسنة
        df = self.calculate_volume_indicators_v2(df)
        
        # 2. توليد الإشارات المحسنة
        df = self.generate_enhanced_signals_v2(df)
        
        # 3. إضافة مستوى الثقة
        df['confidence_level'] = df['volume_confidence'].apply(self.calculate_confidence_level)
        
        # حفظ نتائج التحليل
        self.analysis_results = df.to_dict('records')
        
        return df
    
    def calculate_confidence_level(self, score: float) -> str:
        """تحديد مستوى الثقة"""
        if score >= 90: return "ممتازة"
        elif score >= 80: return "عالية جداً"
        elif score >= 70: return "عالية"
        elif score >= 60: return "متوسطة"
        else: return "ضعيفة"
    
    # =========================================================================
    # نظام التداول المحسن
    # =========================================================================
    
    def calculate_position_size(self, price: float, confidence: float) -> float:
        """حساب حجم المركز مع مراعاة الثقة"""
        base_size = (TRADE_SIZE_USDT * LEVERAGE) / price
        
        # تعديل الحجم حسب الثقة
        confidence_factor = confidence / 100
        adjusted_size = base_size * (0.6 + confidence_factor * 0.8)  # 60% إلى 140%
        
        return adjusted_size
    
    def open_position(self, symbol: str, direction: str, price: float, 
                     confidence: float, confidence_level: str, 
                     divergence_type: str, volume_ratio: float, 
                     quality_score: float, volume_confidence: float,
                     trend_alignment: float, timestamp: datetime) -> Optional[Trade]:
        """فتح مركز جديد مع إدارة مخاطرة محسنة"""
        
        if symbol in self.positions:
            return None
        
        # حساب حجم المركز مع مراعاة الثقة
        quantity = self.calculate_position_size(price, confidence)
        
        # إدارة مخاطرة ديناميكية محسنة
        sl_percent, tp_percent = self.dynamic_risk_management_v2(
            divergence_type, quality_score, volume_confidence
        )
        
        if direction == "BUY":
            stop_loss = price * (1 - sl_percent / 100)
            take_profit = price * (1 + tp_percent / 100)
        else:  # SELL
            stop_loss = price * (1 + sl_percent / 100)
            take_profit = price * (1 - tp_percent / 100)
        
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
            confidence_level=confidence_level,
            stop_loss=stop_loss,
            take_profit=take_profit,
            status="OPEN",
            divergence_type=divergence_type,
            volume_ratio=volume_ratio,
            quality_score=quality_score,
            volume_confidence=volume_confidence,
            trend_alignment=trend_alignment
        )
        
        self.positions[symbol] = trade
        self.trades.append(trade)
        
        # تحديث الإحصائيات
        self.performance_stats[divergence_type]['trades'] += 1
        
        logger.info(f"🎯 فتح مركز {direction} محسن لـ {symbol} ")
        logger.info(f"   الثقة: {confidence:.1f}% | الجودة: {quality_score:.1f}% | محاذاة: {trend_alignment:.2f}")
        logger.info(f"   الانزياح: {divergence_type} | الوقف: {sl_percent:.1f}% | الجني: {tp_percent:.1f}%")
        
        return trade
    
    def close_position(self, symbol: str, price: float, timestamp: datetime, 
                      reason: str = "MANUAL") -> Optional[Trade]:
        """إغلاق مركز مفتوح"""
        
        if symbol not in self.positions:
            return None
        
        trade = self.positions[symbol]
        
        # حساب الربح/الخسارة
        if trade.direction == "BUY":
            pnl = (price - trade.entry_price) * trade.quantity
        else:  # SELL
            pnl = (trade.entry_price - price) * trade.quantity
        
        pnl_percent = (pnl / (trade.quantity * trade.entry_price)) * 100
        
        # رسوم الخروج
        trade_value = trade.quantity * price
        fee = trade_value * 0.0004
        pnl -= fee
        self.current_balance += pnl
        
        # تحديث بيانات الصفقة
        trade.exit_price = price
        trade.exit_time = timestamp
        trade.pnl = pnl
        trade.pnl_percent = pnl_percent
        trade.status = reason
        
        # تحديث إحصائيات الأداء
        if pnl > 0:
            self.performance_stats[trade.divergence_type]['wins'] += 1
        
        # تحديث أداء الانزياح
        if trade.divergence_type in self.divergence_performance:
            stats = self.divergence_performance[trade.divergence_type]
            stats['total'] += 1
            if pnl > 0:
                stats['wins'] += 1
            stats['avg_pnl'] = (stats['avg_pnl'] * (stats['total'] - 1) + pnl) / stats['total']
        
        # إزالة من المراكز المفتوحة
        del self.positions[symbol]
        
        # حفظ في السجل
        self.trade_history.append({
            'symbol': trade.symbol,
            'direction': trade.direction,
            'entry_price': trade.entry_price,
            'exit_price': trade.exit_price,
            'entry_time': trade.entry_time,
            'exit_time': trade.exit_time,
            'pnl': trade.pnl,
            'pnl_percent': trade.pnl_percent,
            'confidence': trade.confidence,
            'confidence_level': trade.confidence_level,
            'divergence_type': trade.divergence_type,
            'volume_ratio': trade.volume_ratio,
            'quality_score': trade.quality_score,
            'volume_confidence': trade.volume_confidence,
            'trend_alignment': trade.trend_alignment,
            'status': trade.status
        })
        
        status_emoji = "🟢" if pnl > 0 else "🔴"
        logger.info(f"📊 إغلاق مركز {trade.direction} لـ {symbol} {status_emoji}"
                   f" الربح: {pnl:.2f} USD ({pnl_percent:.2f}%)")
        
        return trade
    
    def check_stop_conditions(self, symbol: str, current_price: float, 
                            timestamp: datetime) -> bool:
        """فحص شروط الوقف والخروج"""
        
        if symbol not in self.positions:
            return False
        
        trade = self.positions[symbol]
        
        # فحص وقف الخسارة
        if ((trade.direction == "BUY" and current_price <= trade.stop_loss) or
            (trade.direction == "SELL" and current_price >= trade.stop_loss)):
            self.close_position(symbol, trade.stop_loss, timestamp, "STOP_LOSS")
            return True
        
        # فحص جني الأرباح
        if ((trade.direction == "BUY" and current_price >= trade.take_profit) or
            (trade.direction == "SELL" and current_price <= trade.take_profit)):
            self.close_position(symbol, trade.take_profit, timestamp, "TAKE_PROFIT")
            return True
        
        return False
    
    def execute_enhanced_trading_v2(self, df: pd.DataFrame):
        """تنفيذ التداول المحسن"""
        
        logger.info("🚀 بدء التداول المحسن باستراتيجية الانزياح الحجمي...")
        
        for i, row in df.iterrows():
            if i < 50:
                continue
                
            current_price = row['close']
            signal = row['volume_signal']
            confidence = row['volume_confidence']
            confidence_level = row['confidence_level']
            divergence_type = row['divergence_type']
            volume_ratio = row['volume_ratio_20']
            quality_score = row['quality_score']
            volume_confidence = row['volume_confidence_score']
            trend_alignment = row['trend_alignment']
            timestamp = row['timestamp']
            
            # فحص شروط الخروج
            if SYMBOL in self.positions:
                self.check_stop_conditions(SYMBOL, current_price, timestamp)
            
            # فتح مراكز جديدة بشروط مشددة جداً
            if (SYMBOL not in self.positions and signal != 'none' and 
                confidence >= CONFIDENCE_THRESHOLD and 
                quality_score >= QUALITY_THRESHOLD and
                volume_confidence >= 60):
                
                self.open_position(
                    SYMBOL, signal, current_price, confidence, confidence_level,
                    divergence_type, volume_ratio, quality_score, volume_confidence,
                    trend_alignment, timestamp
                )
    
    def calculate_enhanced_results_v2(self, df: pd.DataFrame) -> BacktestResult:
        """حساب النتائج المحسنة"""
        
        if not self.trade_history:
            total_days = (df['timestamp'].max() - df['timestamp'].min()).days
            return BacktestResult(
                total_trades=0, winning_trades=0, losing_trades=0,
                win_rate=0, total_pnl=0, final_balance=self.current_balance,
                max_drawdown=0, sharpe_ratio=0, profit_factor=0,
                avg_trade=0, best_trade=0, worst_trade=0, total_fees=0,
                total_days=max(1, total_days), avg_daily_return=0,
                avg_confidence=0, divergence_analysis={}, volume_analysis={},
                quality_analysis={}, performance_metrics={}
            )
        
        trades_df = pd.DataFrame(self.trade_history)
        
        # المقاييس الأساسية
        total_trades = len(trades_df)
        winning_trades = len(trades_df[trades_df['pnl'] > 0])
        losing_trades = len(trades_df[trades_df['pnl'] < 0])
        win_rate = (winning_trades / total_trades) * 100
        
        total_pnl = trades_df['pnl'].sum()
        final_balance = self.current_balance
        
        # أقصى خسارة متراكمة
        balance_history = [INITIAL_BALANCE]
        for pnl in trades_df['pnl']:
            balance_history.append(balance_history[-1] + pnl)
        
        peak = balance_history[0]
        max_dd = 0
        for value in balance_history:
            if value > peak:
                peak = value
            dd = (peak - value) / peak * 100
            if dd > max_dd:
                max_dd = dd
        
        # نسبة شارب
        avg_return = trades_df['pnl'].mean()
        std_return = trades_df['pnl'].std()
        sharpe_ratio = avg_return / std_return if std_return > 0 else 0
        
        # عامل الربحية
        gross_profit = trades_df[trades_df['pnl'] > 0]['pnl'].sum()
        gross_loss = abs(trades_df[trades_df['pnl'] < 0]['pnl'].sum())
        profit_factor = gross_profit / gross_loss if gross_loss > 0 else float('inf')
        
        # إحصائيات أخرى
        avg_trade = trades_df['pnl'].mean()
        best_trade = trades_df['pnl'].max()
        worst_trade = trades_df['pnl'].min()
        
        # حساب الرسوم بدقة
        total_fees = 0
        for trade in self.trade_history:
            entry_fee = trade['quantity'] * trade['entry_price'] * 0.0004
            exit_fee = trade['quantity'] * trade['exit_price'] * 0.0004
            total_fees += entry_fee + exit_fee
        
        # حساب عدد الأيام والعائد اليومي
        total_days = (df['timestamp'].max() - df['timestamp'].min()).days
        total_days = max(1, total_days)
        avg_daily_return = (final_balance - INITIAL_BALANCE) / INITIAL_BALANCE / total_days * 100
        
        # تحليل الثقة والجودة
        avg_confidence = trades_df['confidence'].mean()
        avg_quality = trades_df['quality_score'].mean()
        avg_volume_confidence = trades_df['volume_confidence'].mean()
        avg_trend_alignment = trades_df['trend_alignment'].mean()
        
        # تحليل الانزياح المحسن
        divergence_analysis = {}
        for div_type in ['bullish_breakout', 'bearish_reversal', 'volume_surge', 'trend_confirmation']:
            div_trades = trades_df[trades_df['divergence_type'] == div_type]
            if len(div_trades) > 0:
                div_win_rate = (len(div_trades[div_trades['pnl'] > 0]) / len(div_trades)) * 100
                div_total_pnl = div_trades['pnl'].sum()
                div_avg_quality = div_trades['quality_score'].mean()
                div_avg_confidence = div_trades['confidence'].mean()
                divergence_analysis[div_type] = {
                    'trades': len(div_trades),
                    'win_rate': div_win_rate,
                    'total_pnl': div_total_pnl,
                    'avg_pnl': div_trades['pnl'].mean(),
                    'avg_quality': div_avg_quality,
                    'avg_confidence': div_avg_confidence,
                    'efficiency': div_total_pnl / len(div_trades) if len(div_trades) > 0 else 0
                }
        
        # تحليل الحجم المحسن
        high_volume_trades = trades_df[trades_df['volume_ratio'] > 2.0]
        very_high_volume_trades = trades_df[trades_df['volume_ratio'] > 3.0]
        
        volume_analysis = {
            'high_volume_trades': len(high_volume_trades),
            'very_high_volume_trades': len(very_high_volume_trades),
            'avg_volume_ratio': trades_df['volume_ratio'].mean(),
            'avg_volume_confidence': avg_volume_confidence,
            'volume_correlation': trades_df['volume_ratio'].corr(trades_df['pnl']) if len(trades_df) > 1 else 0,
            'volume_confidence_correlation': trades_df['volume_confidence'].corr(trades_df['pnl']) if len(trades_df) > 1 else 0,
            'high_volume_win_rate': (len(high_volume_trades[high_volume_trades['pnl'] > 0]) / len(high_volume_trades) * 100) if len(high_volume_trades) > 0 else 0
        }
        
        # تحليل الجودة المحسن
        high_quality_trades = trades_df[trades_df['quality_score'] > 80]
        very_high_quality_trades = trades_df[trades_df['quality_score'] > 90]
        
        quality_analysis = {
            'high_quality_trades': len(high_quality_trades),
            'very_high_quality_trades': len(very_high_quality_trades),
            'avg_quality_score': avg_quality,
            'avg_trend_alignment': avg_trend_alignment,
            'quality_correlation': trades_df['quality_score'].corr(trades_df['pnl']) if len(trades_df) > 1 else 0,
            'trend_alignment_correlation': trades_df['trend_alignment'].corr(trades_df['pnl']) if len(trades_df) > 1 else 0,
            'quality_win_rate': (len(trades_df[(trades_df['quality_score'] > 80) & (trades_df['pnl'] > 0)]) / 
                               len(trades_df[trades_df['quality_score'] > 80]) * 100) if len(trades_df[trades_df['quality_score'] > 80]) > 0 else 0,
            'high_quality_avg_pnl': high_quality_trades['pnl'].mean() if len(high_quality_trades) > 0 else 0
        }
        
        # مقاييس أداء متقدمة
        performance_metrics = {
            'risk_reward_ratio': abs(avg_trade / worst_trade) if worst_trade < 0 else 0,
            'expectancy': (win_rate/100 * avg_trade) - ((1 - win_rate/100) * abs(avg_trade)),
            'consistency_score': (win_rate * profit_factor) / 100,
            'efficiency_score': (total_pnl / total_trades) / abs(worst_trade) if worst_trade < 0 else 0,
            'quality_efficiency': quality_analysis['high_quality_avg_pnl'] / abs(worst_trade) if worst_trade < 0 and quality_analysis['high_quality_avg_pnl'] > 0 else 0
        }
        
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
            performance_metrics=performance_metrics
        )
    
    def run_enhanced_backtest_v2(self, df: pd.DataFrame) -> BacktestResult:
        """تشغيل الباك-تستينغ المحسن"""
        
        logger.info("🔍 بدء الباك-تستينغ المحسن v2...")
        
        # إعادة تعيين البيانات
        self.trades = []
        self.positions = {}
        self.trade_history = []
        self.current_balance = INITIAL_BALANCE
        
        # التحليل المحسن
        df_with_signals = self.enhanced_volume_analysis_v2(df)
        
        # تنفيذ التداول المحسن
        self.execute_enhanced_trading_v2(df_with_signals)
        
        # إغلاق المراكز المفتوحة
        if SYMBOL in self.positions:
            last_price = df_with_signals.iloc[-1]['close']
            last_timestamp = df_with_signals.iloc[-1]['timestamp']
            self.close_position(SYMBOL, last_price, last_timestamp, "END_OF_DATA")
        
        return self.calculate_enhanced_results_v2(df_with_signals)
    
    async def send_enhanced_report_v2(self, backtest_result: BacktestResult, df: pd.DataFrame):
        """إرسال تقرير محسن"""
        
        if not self.telegram_notifier:
            return
        
        try:
            # التقرير النصي
            report_text = self._generate_enhanced_report_text_v2(backtest_result)
            await self.telegram_notifier.send_message(report_text)
            
            # الرسوم البيانية
            chart_buffer = self._create_enhanced_chart_v2(df, backtest_result)
            if chart_buffer:
                caption = f"📈 تحليل استراتيجية الانزياح الحجمي المحسنة - {SYMBOL}"
                await self.telegram_notifier.send_photo(chart_buffer, caption)
                
        except Exception as e:
            logger.error(f"❌ خطأ في إرسال التقرير: {e}")
    
    def _generate_enhanced_report_text_v2(self, backtest_result: BacktestResult) -> str:
        """إنشاء تقرير نصي محسن"""
        
        report_time = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        
        message = f"🎯 *تقرير استراتيجية الانزياح الحجمي المحسنة v2*\n"
        message += "══════════════════════════════════════\n\n"
        
        message += f"⚙️ *الإعدادات المتقدمة:*\n"
        message += f"• العملة: `{SYMBOL}`\n"
        message += f"• الإطار: `{TIMEFRAME}`\n"
        message += f"• الرافعة: `{LEVERAGE}x`\n"
        message += f"• حجم الصفقة: `${TRADE_SIZE_USDT}`\n"
        message += f"• عتبة الثقة: `{CONFIDENCE_THRESHOLD}%`\n"
        message += f"• عتبة الجودة: `{QUALITY_THRESHOLD}%`\n\n"
        
        message += f"📊 *النتائج المحسنة:*\n"
        message += f"• إجمالي الصفقات: `{backtest_result.total_trades}`\n"
        message += f"• الصفقات الرابحة: `{backtest_result.winning_trades}` 🟢\n"
        message += f"• الصفقات الخاسرة: `{backtest_result.losing_trades}` 🔴\n"
        message += f"• نسبة الربح: `{backtest_result.win_rate:.1f}%`\n"
        message += f"• إجمالي الربح: `${backtest_result.total_pnl:,.2f}`\n"
        message += f"• الرصيد النهائي: `${backtest_result.final_balance:,.2f}`\n"
        message += f"• متوسط الجودة: `{backtest_result.quality_analysis['avg_quality_score']:.1f}%`\n"
        message += f"• متوسط محاذاة الاتجاه: `{backtest_result.quality_analysis['avg_trend_alignment']:.2f}`\n\n"
        
        message += f"🔍 *تحليل الانزياح المحسن:*\n"
        divergence_names = {
            'bullish_breakout': '🟢 كسر صعودي',
            'bearish_reversal': '🔴 انعكاس هبوطي', 
            'volume_surge': '📈 طفرة حجمية',
            'trend_confirmation': '🎯 تأكيد اتجاه'
        }
        
        for div_type, analysis in backtest_result.divergence_analysis.items():
            display_name = divergence_names.get(div_type, div_type)
            message += f"*{display_name}:*\n"
            message += f"• الصفقات: `{analysis['trades']}` | الدقة: `{analysis['win_rate']:.1f}%`\n"
            message += f"• الربح: `${analysis['total_pnl']:.2f}` | المتوسط: `${analysis['avg_pnl']:.2f}`\n"
            message += f"• الجودة: `{analysis['avg_quality']:.1f}%` | الكفاءة: `{analysis['efficiency']:.2f}`\n\n"
        
        message += f"📈 *تحليل الجودة المتقدم:*\n"
        message += f"• الصفقات عالية الجودة: `{backtest_result.quality_analysis['high_quality_trades']}`\n"
        message += f"• الصفقات عالية الجودة جداً: `{backtest_result.quality_analysis['very_high_quality_trades']}`\n"
        message += f"• دقة الصفقات عالية الجودة: `{backtest_result.quality_analysis['quality_win_rate']:.1f}%`\n"
        message += f"• متوسط ربح الصفقات عالية الجودة: `${backtest_result.quality_analysis['high_quality_avg_pnl']:.2f}`\n"
        message += f"• ارتباط الجودة بالربح: `{backtest_result.quality_analysis['quality_correlation']:.3f}`\n\n"
        
        message += f"📊 *تحليل الحجم المتقدم:*\n"
        message += f"• الصفقات عالية الحجم: `{backtest_result.volume_analysis['high_volume_trades']}`\n"
        message += f"• الصفقات عالية الحجم جداً: `{backtest_result.volume_analysis['very_high_volume_trades']}`\n"
        message += f"• دقة الصفقات عالية الحجم: `{backtest_result.volume_analysis['high_volume_win_rate']:.1f}%`\n"
        message += f"• متوسط نسبة الحجم: `{backtest_result.volume_analysis['avg_volume_ratio']:.2f}`\n"
        message += f"• متوسط ثقة الحجم: `{backtest_result.volume_analysis['avg_volume_confidence']:.1f}%`\n\n"
        
        message += f"⚡ *مقاييس الأداء:*\n"
        message += f"• عامل الربحية: `{backtest_result.profit_factor:.2f}`\n"
        message += f"• نسبة شارب: `{backtest_result.sharpe_ratio:.2f}`\n"
        message += f"• أقصى خسارة: `{backtest_result.max_drawdown:.2f}%`\n"
        message += f"• نسبة المخاطرة/العائد: `{backtest_result.performance_metrics['risk_reward_ratio']:.2f}`\n"
        message += f"• درجة الكفاءة: `{backtest_result.performance_metrics['efficiency_score']:.2f}`\n\n"
        
        message += f"🕒 *وقت التقرير:* `{report_time}`\n"
        message += "══════════════════════════════════════\n"
        message += "⚡ *نظام الانزياح الحجمي المحسن + فلاتر متعددة الطبقات*"
        
        return message
    
    def _create_enhanced_chart_v2(self, df: pd.DataFrame, backtest_result: BacktestResult) -> BytesIO:
        """إنشاء رسم بياني محسن"""
        try:
            fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
            fig.suptitle(f'تحليل استراتيجية الانزياح الحجمي المحسنة - {SYMBOL}', 
                        fontsize=16, fontname='DejaVu Sans', fontweight='bold')
            
            # 1. السعر والإشارات مع الجودة
            ax1.plot(df['timestamp'], df['close'], label='السعر', linewidth=1.5, color='blue', alpha=0.8)
            
            # إضافة الإشارات مع ترميز اللون حسب الجودة
            signals_df = df[df['volume_signal'].isin(['BUY', 'SELL'])]
            for _, signal in signals_df.iterrows():
                color = 'green' if signal['volume_signal'] == 'BUY' else 'red'
                marker = '^' if signal['volume_signal'] == 'BUY' else 'v'
                size = 80 + (signal['quality_score'] - 70) * 2  # حجم النقطة يعتمد على الجودة
                alpha = 0.5 + (signal['quality_score'] / 200)  # الشفافية تعتمد على الجودة
                
                ax1.scatter(signal['timestamp'], signal['close'], 
                           color=color, marker=marker, s=size, alpha=alpha,
                           label=f'إشارة {signal["volume_signal"]}' if _ == signals_df.index[0] else "")
            
            ax1.set_title('حركة السعر والإشارات (بحسب الجودة)', fontname='DejaVu Sans', fontsize=12)
            ax1.set_ylabel('السعر (USDT)', fontname='DejaVu Sans')
            ax1.legend(prop={'family': 'DejaVu Sans'})
            ax1.grid(True, alpha=0.3)
            
            # 2. توزيع الجودة مع الربح
            if not self.trade_history.empty:
                quality_bins = [70, 80, 90, 100]
                win_rates_by_quality = []
                
                for i in range(len(quality_bins)-1):
                    low = quality_bins[i]
                    high = quality_bins[i+1]
                    trades_in_range = [t for t in self.trade_history if low <= t['quality_score'] < high]
                    if trades_in_range:
                        wins = len([t for t in trades_in_range if t['pnl'] > 0])
                        win_rate = (wins / len(trades_in_range)) * 100
                    else:
                        win_rate = 0
                    win_rates_by_quality.append(win_rate)
                
                x_pos = [f'{quality_bins[i]}-{quality_bins[i+1]}' for i in range(len(quality_bins)-1)]
                colors = ['green' if wr > 50 else 'red' for wr in win_rates_by_quality]
                bars = ax2.bar(x_pos, win_rates_by_quality, color=colors, alpha=0.7)
                
                ax2.set_title('دقة التداول حسب نطاق الجودة', fontname='DejaVu Sans', fontsize=12)
                ax2.set_ylabel('نسبة الربح %', fontname='DejaVu Sans')
                ax2.set_xlabel('نطاق الجودة', fontname='DejaVu Sans')
                ax2.grid(True, alpha=0.3)
                
                # إضافة القيم على الأعمدة
                for bar, wr in zip(bars, win_rates_by_quality):
                    ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1, 
                            f'{wr:.1f}%', ha='center', fontname='DejaVu Sans')
            
            # 3. أداء الرصيد مع مناطق الجودة
            if len(self.trade_history) > 0:
                balance_history = [INITIAL_BALANCE]
                quality_markers = []
                
                for i, trade in enumerate(self.trade_history):
                    balance_history.append(balance_history[-1] + trade['pnl'])
                    if trade['quality_score'] > 85:
                        quality_markers.append((i+1, balance_history[-1], 'green'))
                    elif trade['quality_score'] < 70:
                        quality_markers.append((i+1, balance_history[-1], 'red'))
                
                ax3.plot(range(len(balance_history)), balance_history, 
                        color='blue', linewidth=2.5, label='الرصيد')
                ax3.axhline(INITIAL_BALANCE, color='red', linestyle='--', alpha=0.7, 
                           linewidth=1.5, label='رصيد البداية')
                
                # إضافة علامات الجودة
                for marker in quality_markers:
                    ax3.scatter(marker[0], marker[1], color=marker[2], s=50, alpha=0.7,
                               label='جودة عالية' if marker[2] == 'green' and marker[0] == quality_markers[0][0] else 
                                     'جودة منخفضة' if marker[2] == 'red' and marker[0] == quality_markers[0][0] else "")
                
                ax3.set_title('تطور الرصيد مع مؤشرات الجودة', fontname='DejaVu Sans', fontsize=12)
                ax3.set_xlabel('عدد الصفقات', fontname='DejaVu Sans')
                ax3.set_ylabel('الرصيد (USD)', fontname='DejaVu Sans')
                ax3.legend(prop={'family': 'DejaVu Sans'})
                ax3.grid(True, alpha=0.3)
            
            # 4. مقارنة كفاءة أنواع الانزياح
            div_analysis = backtest_result.divergence_analysis
            if div_analysis:
                div_types = list(div_analysis.keys())
                efficiencies = [div_analysis[div]['efficiency'] for div in div_types]
                win_rates = [div_analysis[div]['win_rate'] for div in div_types]
                
                # مخطط مبعثر للكفاءة والدقة
                colors = ['green' if eff > 0 else 'red' for eff in efficiencies]
                scatter = ax4.scatter(win_rates, efficiencies, c=colors, s=100, alpha=0.7)
                
                # إضافة أسماء الانزياحات
                for i, div_type in enumerate(div_types):
                    ax4.annotate(div_type[:12], (win_rates[i], efficiencies[i]), 
                                xytext=(5, 5), textcoords='offset points', 
                                fontname='DejaVu Sans', fontsize=8)
                
                ax4.axhline(0, color='black', linestyle='-', alpha=0.3)
                ax4.axvline(50, color='black', linestyle='-', alpha=0.3)
                ax4.set_title('كفاءة الانزياح (الدقة vs الربحية)', fontname='DejaVu Sans', fontsize=12)
                ax4.set_xlabel('نسبة الربح %', fontname='DejaVu Sans')
                ax4.set_ylabel('الكفاءة (ربح/صفقة)', fontname='DejaVu Sans')
                ax4.grid(True, alpha=0.3)
            
            plt.tight_layout()
            
            buffer = BytesIO()
            plt.savefig(buffer, format='png', dpi=150, bbox_inches='tight')
            buffer.seek(0)
            plt.close()
            
            return buffer
            
        except Exception as e:
            logger.error(f"❌ خطأ في إنشاء الرسم البياني: {e}")
            return None

# =============================================================================
# الوظيفة الرئيسية المحسنة
# =============================================================================

async def main():
    """الوظيفة الرئيسية المحسنة"""
    
    logger.info("🚀 بدء تشغيل الاستراتيجية المحسنة v2")
    
    telegram_notifier = TelegramNotifier(TELEGRAM_BOT_TOKEN, TELEGRAM_CHAT_ID)
    
    # جلب البيانات
    data_fetcher = DataFetcher()
    df = data_fetcher.fetch_historical_data(SYMBOL, TIMEFRAME, DATA_LIMIT)
    
    if df.empty:
        error_msg = "❌ فشل جلب البيانات. تأكد من اتصال الإنترنت وصحة اسم العملة."
        logger.error(error_msg)
        await telegram_notifier.send_message(error_msg)
        return
    
    # إرسال معلومات البيانات
    data_info = f"📊 فترة البيانات المحسنة: {len(df)} شمعة من {df['timestamp'].min().date()} إلى {df['timestamp'].max().date()}"
    logger.info(data_info)
    await telegram_notifier.send_message(data_info)
    
    # تشغيل الاستراتيجية المحسنة
    strategy = EnhancedVolumeDivergenceStrategy(telegram_notifier)
    backtest_result = strategy.run_enhanced_backtest_v2(df)
    
    # إرسال التقرير المحسن
    await strategy.send_enhanced_report_v2(backtest_result, df)
    
    # حفظ النتائج المحسنة
    trades_df = pd.DataFrame(strategy.trade_history)
    if not trades_df.empty:
        filename = f'enhanced_volume_trades_v2_{SYMBOL}_{TIMEFRAME}_{datetime.now().strftime("%Y%m%d_%H%M")}.csv'
        trades_df.to_csv(filename, index=False)
        logger.info(f"💾 تم حفظ سجل الصفقات المحسن في {filename}")
    
    # تحليل النتائج
    if backtest_result.win_rate > 65 and backtest_result.total_pnl > 0:
        success_msg = f"✅ الاستراتيجية المحسنة حققت نتائج ممتازة: دقة {backtest_result.win_rate:.1f}% وربح ${backtest_result.total_pnl:.2f}"
    elif backtest_result.win_rate > 55 and backtest_result.total_pnl > 0:
        success_msg = f"⚠️ الاستراتيجية المحسنة حققت نتائج جيدة: دقة {backtest_result.win_rate:.1f}% وربح ${backtest_result.total_pnl:.2f}"
    else:
        success_msg = f"❌ الاستراتيجية المحسنة تحتاج تحسين: دقة {backtest_result.win_rate:.1f}% وربح ${backtest_result.total_pnl:.2f}"
    
    logger.info(success_msg)
    await telegram_notifier.send_message(success_msg)
    
    logger.info("✅ اكتمل تشغيل الاستراتيجية المحسنة بنجاح")

def run_main():
    """تشغيل الدالة الرئيسية بشكل آمن"""
    try:
        asyncio.run(main())
    except RuntimeError as e:
        if "Event loop is closed" in str(e):
            # تجاهل هذا الخطأ الشائع في بعض البيئات
            logger.info("✅ اكتمل التشغيل بنجاح")
        else:
            logger.error(f"❌ خطأ غير متوقع: {e}")
    except Exception as e:
        logger.error(f"❌ خطأ في التشغيل: {e}")

if __name__ == "__main__":
    run_main()
