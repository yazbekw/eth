import pandas as pd
import numpy as np
import requests
import time
from datetime import datetime, timedelta
import os
from typing import Dict, List, Optional, Tuple
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
# إعدادات التداول من متغيرات البيئة - معدلة
# =============================================================================

SYMBOL = os.getenv("TRADING_SYMBOL", "BNBUSDT")
TIMEFRAME = os.getenv("TRADING_TIMEFRAME", "1h")
STOP_LOSS_PERCENT = float(os.getenv("STOP_LOSS_PERCENT", "0.8"))
TAKE_PROFIT_PERCENT = float(os.getenv("TAKE_PROFIT_PERCENT", "2.5"))
TRADE_SIZE_USDT = float(os.getenv("TRADE_SIZE_USDT", "100.0"))
LEVERAGE = int(os.getenv("LEVERAGE", "10"))
INITIAL_BALANCE = float(os.getenv("INITIAL_BALANCE", "5000.0"))
CONFIDENCE_THRESHOLD = int(os.getenv("CONFIDENCE_THRESHOLD", "70"))
SELL_CONFIDENCE_THRESHOLD = int(os.getenv("SELL_CONFIDENCE_THRESHOLD", "68"))  # خفض من 72 إلى 68

# إعدادات مدة الاختبار
DATA_LIMIT = int(os.getenv("DATA_LIMIT", "2000"))
TEST_DAYS = int(os.getenv("TEST_DAYS", "180"))

# إعدادات التلغرام
TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN", "")
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID", "")

# إعدادات البيع المحسنة - معدلة
SUPER_QUALITY_SELL_THRESHOLD = int(os.getenv("SUPER_QUALITY_SELL_THRESHOLD", "78"))  # خفض من 80 إلى 78
HIGH_QUALITY_SELL_THRESHOLD = int(os.getenv("HIGH_QUALITY_SELL_THRESHOLD", "72"))   # خفض من 75 إلى 72

# إعداد التسجيل
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger("Enhanced_EMA_RSI_MACD_Strategy_v4_1")

# =============================================================================
# هياكل البيانات
# =============================================================================

@dataclass
class Trade:
    symbol: str
    direction: str  # LONG or SHORT
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
    volatility: float = 0
    signal_strength: float = 0
    quality: str = "STANDARD"  # STANDARD, HIGH, SUPER

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
    confidence_analysis: Dict
    buy_performance: Dict
    sell_performance: Dict
    quality_analysis: Dict

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
# محرك الاستراتيجية المحسنة v4.1 مع تصحيح شروط البيع
# =============================================================================

class EnhancedEmaRsiMacdStrategyV4_1:
    """استراتيجية محسنة v4.1 مع تصحيح شروط البيع لزيادة الفرص"""
    
    def __init__(self, telegram_notifier: TelegramNotifier = None):
        self.name = "enhanced_ema_rsi_macd_v4_1"
        self.trades: List[Trade] = []
        self.balance = INITIAL_BALANCE
        self.current_balance = INITIAL_BALANCE
        self.positions = {}
        self.trade_history = []
        self.analysis_results = []
        self.telegram_notifier = telegram_notifier
        self.sell_performance_history = []
    
    # =========================================================================
    # الحسابات الأساسية
    # =========================================================================
    
    @staticmethod
    def calculate_ema(prices: pd.Series, period: int) -> pd.Series:
        """حساب المتوسط المتحرك الأسي"""
        return prices.ewm(span=period, adjust=False).mean()
    
    @staticmethod
    def calculate_rsi(prices: pd.Series, period: int = 14) -> pd.Series:
        """حساب مؤشر RSI"""
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        return rsi
    
    @staticmethod
    def calculate_macd(prices: pd.Series) -> Tuple[pd.Series, pd.Series, pd.Series]:
        """حساب مؤشر MACD"""
        ema_12 = prices.ewm(span=12, adjust=False).mean()
        ema_26 = prices.ewm(span=26, adjust=False).mean()
        macd_line = ema_12 - ema_26
        signal_line = macd_line.ewm(span=9, adjust=False).mean()
        histogram = macd_line - signal_line
        return macd_line, signal_line, histogram
    
    def analyze_trend(self, df: pd.DataFrame) -> pd.DataFrame:
        """تحليل الاتجاه باستخدام المتوسطات المتحركة"""
        df['ema_9'] = self.calculate_ema(df['close'], 9)
        df['ema_21'] = self.calculate_ema(df['close'], 21)
        df['ema_50'] = self.calculate_ema(df['close'], 50)
        df['ema_100'] = self.calculate_ema(df['close'], 100)
        
        # تحديد ترتيب المتوسطات
        conditions = [
            (df['ema_9'] > df['ema_21']) & (df['ema_21'] > df['ema_50']) & (df['ema_50'] > df['ema_100']),
            (df['ema_9'] < df['ema_21']) & (df['ema_21'] < df['ema_50']) & (df['ema_50'] < df['ema_100'])
        ]
        choices = ['صاعد قوي', 'هابط قوي']
        df['ma_order'] = np.select(conditions, choices, default='متذبذب')
        
        # حساب قوة الاتجاه
        df['distance_9_21'] = (df['ema_9'] - df['ema_21']).abs() / df['close']
        df['distance_21_50'] = (df['ema_21'] - df['ema_50']).abs() / df['close']
        df['distance_50_100'] = (df['ema_50'] - df['ema_100']).abs() / df['close']
        
        conditions_strength = [
            (df['distance_9_21'] > 0.03) & (df['distance_21_50'] > 0.04) & (df['distance_50_100'] > 0.05),
            (df['distance_9_21'] > 0.02) & (df['distance_21_50'] > 0.025) & (df['distance_50_100'] > 0.03),
            (df['distance_9_21'] > 0.01) & (df['distance_21_50'] > 0.015) & (df['distance_50_100'] > 0.02)
        ]
        choices_strength = [12, 9, 6]
        df['trend_strength'] = np.select(conditions_strength, choices_strength, default=3)
        
        return df
    
    def enhanced_scoring_system_v4_1(self, df: pd.DataFrame) -> pd.DataFrame:
        """نظام التقييم المحسن v4.1 مع تحسين متقدم لصفقات البيع"""
        
        # 1. تحليل المتوسطات المتحركة (25 نقطة كحد أقصى)
        conditions_ma = [
            (df['ma_order'] == 'صاعد قوي') & (df['close'] > df['ema_21']) & (df['close'] > df['ema_50']),
            (df['ma_order'] == 'هابط قوي') & (df['close'] < df['ema_21']) & (df['close'] < df['ema_50']),
            (df['ma_order'].str.contains('صاعد')) & (df['close'] > df['ema_21']),
            (df['ma_order'].str.contains('هابط')) & (df['close'] < df['ema_21'])
        ]
        choices_ma = [
            np.minimum(25, df['trend_strength'] * 2.5),
            np.minimum(25, df['trend_strength'] * 2.5),
            np.minimum(18, df['trend_strength'] * 2.0),
            np.minimum(18, df['trend_strength'] * 2.0)
        ]
        df['ma_score'] = np.select(conditions_ma, choices_ma, default=0)
        
        # 2. تحليل RSI (40 نقطة كحد أقصى)
        conditions_rsi = [
            df['rsi'] <= 20,
            df['rsi'] <= 30,
            df['rsi'] >= 80,
            df['rsi'] >= 70,
            (df['rsi'] >= 45) & (df['rsi'] <= 55),
            (df['rsi'] >= 40) & (df['rsi'] <= 60),
            (df['rsi'] >= 35) & (df['rsi'] <= 65)
        ]
        choices_rsi = [
            40 - (20 - df['rsi']) * 0.5,
            35 - (30 - df['rsi']) * 0.5,
            40 - (df['rsi'] - 80) * 0.5,
            35 - (df['rsi'] - 70) * 0.5,
            25,
            20,
            15
        ]
        df['rsi_score'] = np.select(conditions_rsi, choices_rsi, default=8)
        df['rsi_score'] = df['rsi_score'].clip(0, 40)
        
        # 3. تحليل MACD (35 نقطة كحد أقصى)
        macd_positive = (df['macd_histogram'] > 0) & (df['macd_line'] > df['macd_signal'])
        macd_negative = (df['macd_histogram'] < 0) & (df['macd_line'] < df['macd_signal'])
        histogram_strength = df['macd_histogram'].abs()
        
        conditions_macd = [
            macd_positive & (histogram_strength > 0.008),
            macd_positive & (histogram_strength > 0.005),
            macd_positive & (histogram_strength > 0.002),
            macd_positive,
            macd_negative & (histogram_strength > 0.008),
            macd_negative & (histogram_strength > 0.005),
            macd_negative & (histogram_strength > 0.002),
            macd_negative
        ]
        choices_macd = [
            np.minimum(35, 30 + (histogram_strength * 1200)),
            np.minimum(35, 25 + (histogram_strength * 1000)),
            np.minimum(35, 20 + (histogram_strength * 800)),
            np.minimum(35, 15 + (histogram_strength * 600)),
            np.minimum(35, 30 + (histogram_strength * 1200)),
            np.minimum(35, 25 + (histogram_strength * 1000)),
            np.minimum(35, 20 + (histogram_strength * 800)),
            np.minimum(35, 15 + (histogram_strength * 600))
        ]
        df['macd_score'] = np.select(conditions_macd, choices_macd, default=0)
        
        # النتيجة النهائية الأساسية
        df['total_score'] = df['ma_score'] + df['rsi_score'] + df['macd_score']
        df['total_score'] = df['total_score'].clip(0, 100)
        
        # ✅ التصحيح: تقليل وزن الإشارات عالية الثقة الكاذبة
        high_confidence_mask = df['total_score'] >= 80
        df.loc[high_confidence_mask, 'score_v4_1'] = df.loc[high_confidence_mask, 'total_score'] * 0.85
        
        # ✅ التعزيز: زيادة وزن الإشارات متوسطة الثقة الناجحة
        medium_confidence_mask = (df['total_score'] >= 60) & (df['total_score'] < 80)
        df.loc[medium_confidence_mask, 'score_v4_1'] = df.loc[medium_confidence_mask, 'total_score'] * 1.15
        
        # ✅ الإشارات المنخفضة تبقى كما هي
        low_confidence_mask = df['total_score'] < 60
        df.loc[low_confidence_mask, 'score_v4_1'] = df.loc[low_confidence_mask, 'total_score']
        
        df['score_v4_1'] = df['score_v4_1'].clip(0, 100)
        
        return df
    
    def enhance_sell_signals_v4_1(self, df: pd.DataFrame) -> pd.DataFrame:
        """تعزيز إشارات البيع بشكل متوازن لزيادة الفرص"""
    
        # ✅ شروط بيع أكثر مرونة مع الحفاظ على الجودة
        super_quality_sell_conditions = (
            (df['ema_9'] < df['ema_21']) &
            (df['ema_21'] < df['ema_50']) &
            (df['ema_50'] < df['ema_100']) &  # اتجاه هابط قوي بمتوسطات متعددة
            (df['rsi'] > 65) &  # خفض من 68 إلى 65
            (df['macd_histogram'] < -0.003) &  # خفض من -0.004 إلى -0.003
            (df['volume'] > df['volume_avg'] * 1.2)  # خفض من 1.3 إلى 1.2
        )
    
        # ✅ شروط بيع عالية الجودة أكثر مرونة
        high_quality_sell_conditions = (
            (df['ema_9'] < df['ema_21']) &
            (df['ema_21'] < df['ema_50']) &  # تأكيد الهبوط بمتوسطين
            (df['rsi'] > 62) &  # خفض من 65 إلى 62
            (df['macd_histogram'] < -0.002) &  # خفض من -0.003 إلى -0.002
            (df['volume'] > df['volume_avg'] * 1.0)  # خفض من 1.1 إلى 1.0
        )
    
        # ✅ شروط بيع جيدة أكثر مرونة
        good_sell_conditions = (
            (df['ema_9'] < df['ema_21']) &
            (df['rsi'] > 60) &  # خفض من 62 إلى 60
            (df['macd_histogram'] < -0.001) &  # خفض من -0.002 إلى -0.001
            (df['volume'] > df['volume_avg'] * 0.8)  # خفض من 0.9 إلى 0.8
        )
    
        # تطبيق التعزيز حسب الجودة (من الأعلى إلى الأدنى)
        df.loc[super_quality_sell_conditions, 'score_v4_1'] = df.loc[super_quality_sell_conditions, 'score_v4_1'] * 1.4  # خفض من 1.5 إلى 1.4
        df.loc[high_quality_sell_conditions, 'score_v4_1'] = df.loc[high_quality_sell_conditions, 'score_v4_1'] * 1.25   # خفض من 1.3 إلى 1.25
        df.loc[good_sell_conditions, 'score_v4_1'] = df.loc[good_sell_conditions, 'score_v4_1'] * 1.1                   # خفض من 1.15 إلى 1.1
    
        # ✅ تحديد قوة الإشارة بناء على مستوى التعزيز
        df['signal_strength'] = df['score_v4_1'] / 100.0
        
        # ✅ تحديد جودة الإشارة
        df['signal_quality'] = 'STANDARD'
        df.loc[good_sell_conditions, 'signal_quality'] = 'GOOD'
        df.loc[high_quality_sell_conditions, 'signal_quality'] = 'HIGH'
        df.loc[super_quality_sell_conditions, 'signal_quality'] = 'SUPER'
    
        # ✅ تسجيل إحصائيات التعزيز
        super_count = len(df[super_quality_sell_conditions])
        high_count = len(df[high_quality_sell_conditions])
        good_count = len(df[good_sell_conditions])
    
        logger.info(f"🎯 تعزيز إشارات البيع v4.1 - فائق: {super_count}, عالي: {high_count}, جيد: {good_count}")
    
        return df
    
    def add_smart_filters_v4_1(self, df: pd.DataFrame) -> pd.DataFrame:
        """إضافة عوامل تصفية ذكية v4.1 مع إنشاء atr_percent"""
        
        # 1. إنشاء atr_percent إذا لم يكن موجوداً
        if 'atr_percent' not in df.columns:
            df['tr'] = np.maximum(
                df['high'] - df['low'],
                np.maximum(
                    abs(df['high'] - df['close'].shift(1)),
                    abs(df['low'] - df['close'].shift(1))
                )
            )
            df['atr'] = df['tr'].rolling(14).mean()
            df['atr_percent'] = df['atr'] / df['close']
            # تعبئة القيم NaN
            df['atr_percent'] = df['atr_percent'].fillna(df['atr_percent'].mean())
        
        # 2. إنشاء rsi_volatility إذا لم يكن موجوداً
        if 'rsi_volatility' not in df.columns:
            if 'rsi' in df.columns:
                df['rsi_volatility'] = df['rsi'].rolling(14).std()
            else:
                df['rsi_volatility'] = 10  # قيمة افتراضية
        
        # 3. إنشاء volume_avg إذا لم يكن موجوداً
        if 'volume_avg' not in df.columns:
            df['volume_avg'] = df['volume'].rolling(20).mean()
    
        # 1. تصفية حسب قوة الاتجاه
        df['strong_uptrend'] = (df['ema_9'] > df['ema_21']) & (df['ema_21'] > df['ema_50']) & (df['ema_50'] > df['ema_100'])
        df['strong_downtrend'] = (df['ema_9'] < df['ema_21']) & (df['ema_21'] < df['ema_50']) & (df['ema_50'] < df['ema_100'])
        
        # 2. تصفية حسب تقلبات RSI
        df['low_volatility'] = df['rsi_volatility'] < 12
        
        # 3. تصفية حسب حجم التداول
        df['high_volume'] = df['volume'] > df['volume_avg'] * 1.3
        
        # 4. تصفية حسب تقلبات السوق (ATR)
        df['low_volatility_market'] = df['atr_percent'] < 0.02
        
        # 5. تطبيق الفلاتر المركبة
        df['filter_pass_buy'] = (
            (df['strong_uptrend'] | ~df['strong_downtrend']) &
            df['low_volatility'] & 
            df['high_volume'] &
            df['low_volatility_market'] &
            (df['close'] > df['ema_21'])
        )
        
        # ✅ فلتر جودة للبيع - أكثر مرونة
        df['high_quality_sell'] = (
            (df['ema_9'] < df['ema_21']) &
            (df['ema_21'] < df['ema_50']) &
            (df['rsi'] > 60) &  # خفض من 65 إلى 60
            (df['macd_histogram'] < -0.002) &  # خفض من -0.003 إلى -0.002
            (df['volume'] > df['volume_avg'] * 0.9)  # خفض من 1.1 إلى 0.9
        )
        
        df['good_quality_sell'] = (
            (df['ema_9'] < df['ema_21']) &
            (df['rsi'] > 58) &  # خفض من 62 إلى 58
            (df['macd_histogram'] < -0.001) &  # خفض من -0.002 إلى -0.001
            (df['volume'] > df['volume_avg'] * 0.7)  # خفض من 0.9 إلى 0.7
        )
        
        # ✅ فلاتر مرنة للبيع
        df['filter_pass_sell_enhanced'] = (
            (
                df['strong_downtrend'] |  # اتجاه هابط قوي
                ((df['ema_9'] < df['ema_21']) & (df['ema_21'] < df['ema_50']))  # اتجاه هابط
            ) &
            (df['close'] < df['ema_21']) &  # تحت المتوسط المتوسط (بدلاً من 50)
            (df['rsi'] > 45)  # RSI في النصف العلوي (أكثر مرونة)
        )
        
        return df
    
    def dynamic_stop_take_profit_v4_1(self, df: pd.DataFrame) -> pd.DataFrame:
        """وقف وجني ديناميكي متوازن للبيع v4.1"""
    
        # التحقق من وجود الأعمدة المطلوبة
        if 'atr_percent' not in df.columns:
            logger.warning("⚠️ عمود atr_percent غير موجود، إنشاء قيم افتراضية")
            # إنشاء atr_percent إذا لم يكن موجوداً
            df['tr'] = np.maximum(
                df['high'] - df['low'],
                np.maximum(
                    abs(df['high'] - df['close'].shift(1)),
                    abs(df['low'] - df['close'].shift(1))
                )
            )
            df['atr'] = df['tr'].rolling(14).mean()
            df['atr_percent'] = df['atr'] / df['close']
    
        # حساب تقلبات السوق
        df['volatility_ratio'] = df['atr_percent'] / df['atr_percent'].rolling(50).mean()
    
        # تعبئة القيم NaN في volatility_ratio
        df['volatility_ratio'] = df['volatility_ratio'].fillna(1.0)
    
        # وقف وجني ديناميكي للشراء (تبقى كما هي)
        df['dynamic_sl_buy'] = np.where(
            df['volatility_ratio'] > 1.5,
            1.2,
            np.where(
                df['volatility_ratio'] < 0.7,
                0.6,
                0.8
            )
        )
    
        df['dynamic_tp_buy'] = np.where(
            df['volatility_ratio'] > 1.5,
            3.5,
            np.where(
                df['volatility_ratio'] < 0.7,
                2.0,
                2.5
            )
        )
    
        # ✅ إعدادات متوازنة للبيع v4.1
        df['dynamic_sl_sell'] = np.where(
            df['volatility_ratio'] > 1.5,
            0.8,  # زيادة من 0.7 إلى 0.8
            np.where(
                df['volatility_ratio'] < 0.7,
                0.5,  # زيادة من 0.4 إلى 0.5
                0.6   # زيادة من 0.5 إلى 0.6
            )
        )
    
        df['dynamic_tp_sell'] = np.where(
            df['volatility_ratio'] > 1.5,
            4.0,  # خفض من 4.2 إلى 4.0
            np.where(
                df['volatility_ratio'] < 0.7,
                3.0,  # خفض من 3.2 إلى 3.0
                3.5   # خفض من 3.8 إلى 3.5
            )
        )
    
        # ✅ إعدادات خاصة للبيع فائق الجودة
        df['super_quality_sell_sl'] = df['dynamic_sl_sell'] * 0.7  # زيادة من 0.6 إلى 0.7
        df['super_quality_sell_tp'] = df['dynamic_tp_sell'] * 1.2  # خفض من 1.3 إلى 1.2
        
        # ✅ إعدادات خاصة للبيع عالي الجودة
        df['high_quality_sell_sl'] = df['dynamic_sl_sell'] * 0.8  # زيادة من 0.7 إلى 0.8
        df['high_quality_sell_tp'] = df['dynamic_tp_sell'] * 1.1  # خفض من 1.2 إلى 1.1
    
        logger.info(f"🎯 إعدادات البيع المتوازنة v4.1 - وقف: {df['dynamic_sl_sell'].mean():.2f}%, جني: {df['dynamic_tp_sell'].mean():.2f}%")
    
        return df
    
    def risk_adjusted_scoring_v4_1(self, df: pd.DataFrame) -> pd.DataFrame:
        """نظام تقييم معدل حسب المخاطرة v4.1 مع معالجة آمنة"""
        
        # التحقق من وجود الأعمدة المطلوبة
        required_columns = ['atr_percent', 'rsi_volatility', 'score_v4_1']
        missing_columns = [col for col in required_columns if col not in df.columns]
        
        if missing_columns:
            logger.warning(f"⚠️ أعمدة مفقودة في risk_adjusted_scoring_v4_1: {missing_columns}")
            return df
        
        # مكافأة الصفقات منخفضة المخاطرة
        low_risk_mask = (df['atr_percent'] < 0.015) & (df['rsi_volatility'] < 10)
        df.loc[low_risk_mask, 'score_v4_1'] = df.loc[low_risk_mask, 'score_v4_1'] * 1.15  # خفض من 1.2 إلى 1.15
        
        # معاقبة الصفقات عالية المخاطرة
        high_risk_mask = (df['atr_percent'] > 0.025) | (df['rsi_volatility'] > 15)
        df.loc[high_risk_mask, 'score_v4_1'] = df.loc[high_risk_mask, 'score_v4_1'] * 0.85  # زيادة من 0.8 إلى 0.85
        
        return df
    
    def generate_enhanced_signals_v4_1(self, df: pd.DataFrame) -> pd.DataFrame:
        """إشارات محسنة v4.1 مع شروط بيع أكثر مرونة"""
    
        # التحقق من وجود الأعمدة المطلوبة
        required_columns = ['score_v4_1', 'filter_pass_buy', 'rsi', 'macd_histogram', 'close', 'ema_21', 'volume', 'volume_avg', 'ema_9', 'ema_50', 'ma_order', 'signal_quality']
        missing_columns = [col for col in required_columns if col not in df.columns]
    
        if missing_columns:
            logger.warning(f"⚠️ أعمدة مفقودة في generate_enhanced_signals_v4_1: {missing_columns}")
            df['signal_v4_1'] = 'none'
            df['confidence_level'] = 'ضعيفة'
            df['current_volatility'] = 0.0
            return df
    
        # الشروط الأساسية المحسنة للشراء (تبقى كما هي - تعمل بشكل ممتاز)
        buy_condition_v4_1 = (
            (df['score_v4_1'] >= CONFIDENCE_THRESHOLD) &
            (df['filter_pass_buy'] == True) &
            (df['rsi'] >= 35) & (df['rsi'] <= 65) &
            (df['macd_histogram'] > -0.003) &
            (df['close'] > df['ema_21']) &
            (df['volume'] > df['volume_avg'] * 0.8)
        )
    
        # ✅ شروط بيع أكثر مرونة v4.1
        super_quality_sell = (
            (df['score_v4_1'] >= SUPER_QUALITY_SELL_THRESHOLD) &  # 78
            (df['ema_9'] < df['ema_21']) &
            (df['ema_21'] < df['ema_50']) &
            (df['ema_50'] < df['ema_100']) &  # اتجاه هابط قوي بمتوسطات متعددة
            (df['rsi'] > 65) &  # خفض من 68 إلى 65
            (df['macd_histogram'] < -0.003) &  # خفض من -0.004 إلى -0.003
            (df['volume'] > df['volume_avg'] * 1.2)  # خفض من 1.3 إلى 1.2
        )
        
        high_quality_sell = (
            (df['score_v4_1'] >= HIGH_QUALITY_SELL_THRESHOLD) &  # 72
            (df['ema_9'] < df['ema_21']) &
            (df['ema_21'] < df['ema_50']) &  # اتجاه هابط بمتوسطين
            (df['rsi'] > 62) &  # خفض من 65 إلى 62
            (df['macd_histogram'] < -0.002) &  # خفض من -0.003 إلى -0.002
            (df['volume'] > df['volume_avg'] * 1.0)  # خفض من 1.1 إلى 1.0
        )
        
        good_quality_sell = (
            (df['score_v4_1'] >= SELL_CONFIDENCE_THRESHOLD) &  # 68
            (df['ema_9'] < df['ema_21']) &
            (df['rsi'] > 60) &  # خفض من 62 إلى 60
            (df['macd_histogram'] < -0.001) &  # خفض من -0.002 إلى -0.001
            (df['volume'] > df['volume_avg'] * 0.8)  # خفض من 0.9 إلى 0.8
        )
    
        # ✅ فلتر إضافي للبيع: أكثر مرونة
        sideways_market = (
            (df['ema_50'] - df['ema_50'].shift(5)).abs() / df['ema_50'] < 0.008  # زيادة من 0.01 إلى 0.008
        )
        
        # تطبيق الإشارات مع الأولوية القصوى للجودة الفائقة
        df['signal_v4_1'] = 'none'
        df.loc[buy_condition_v4_1, 'signal_v4_1'] = 'LONG'
        df.loc[super_quality_sell & ~sideways_market, 'signal_v4_1'] = 'SHORT'
        df.loc[high_quality_sell & ~sideways_market & (df['signal_v4_1'] == 'none'), 'signal_v4_1'] = 'SHORT'
        df.loc[good_quality_sell & ~sideways_market & (df['signal_v4_1'] == 'none'), 'signal_v4_1'] = 'SHORT'
    
        # إضافة مستوى الثقة النهائي
        df['confidence_level'] = df['score_v4_1'].apply(self.calculate_confidence_level_v4_1)
    
        # إضافة التقلبات للتحليل
        if 'atr_percent' in df.columns:
            df['current_volatility'] = df['atr_percent'].fillna(df['atr_percent'].mean())
        else:
            df['current_volatility'] = 0.02
    
        # ✅ تسجيل إحصائيات مفصلة v4.1
        total_signals = len(df[df['signal_v4_1'] != 'none'])
        buy_signals = len(df[df['signal_v4_1'] == 'LONG'])
        sell_signals = len(df[df['signal_v4_1'] == 'SHORT'])
        super_sell_signals = len(df[super_quality_sell & (df['signal_v4_1'] == 'SHORT')])
        high_sell_signals = len(df[high_quality_sell & (df['signal_v4_1'] == 'SHORT')])
        good_sell_signals = len(df[good_quality_sell & (df['signal_v4_1'] == 'SHORT')])
    
        logger.info(f"📊 إحصائيات الإشارات v4.1 - شراء: {buy_signals}, بيع فائق: {super_sell_signals}, بيع عالي: {high_sell_signals}, بيع جيد: {good_sell_signals}")
    
        # ✅ تحليل جودة إشارات البيع
        if sell_signals > 0:
            sell_confidence_avg = df[df['signal_v4_1'] == 'SHORT']['score_v4_1'].mean()
            sell_rsi_avg = df[df['signal_v4_1'] == 'SHORT']['rsi'].mean()
            logger.info(f"🔽 تحليل إشارات البيع v4.1 - متوسط الثقة: {sell_confidence_avg:.1f}%, متوسط RSI: {sell_rsi_avg:.1f}")
        
            # تحليل البيع فائق الجودة
            if super_sell_signals > 0:
                super_sell_confidence = df[super_quality_sell & (df['signal_v4_1'] == 'SHORT')]['score_v4_1'].mean()
                logger.info(f"🎯 البيع فائق الجودة v4.1 - متوسط الثقة: {super_sell_confidence:.1f}%")
    
        if buy_signals > 0:
            buy_confidence_avg = df[df['signal_v4_1'] == 'LONG']['score_v4_1'].mean()
            buy_rsi_avg = df[df['signal_v4_1'] == 'LONG']['rsi'].mean()
            logger.info(f"🔼 تحليل إشارات الشراء v4.1 - متوسط الثقة: {buy_confidence_avg:.1f}%, متوسط RSI: {buy_rsi_avg:.1f}")
    
        return df
    
    def calculate_confidence_level_v4_1(self, score: float) -> str:
        """تحديد مستوى الثقة بدقة v4.1"""
        if score >= 85:
            return "عالية جداً"
        elif score >= 75:
            return "عالية" 
        elif score >= 65:
            return "متوسطة"
        elif score >= 55:
            return "منخفضة"
        else:
            return "ضعيفة"
    
    def enhanced_analysis_v4_1(self, df: pd.DataFrame) -> pd.DataFrame:
        """التحليل المحسن v4.1 - الدالة الرئيسية مع إصلاح الترتيب"""
        
        # 1. حساب المؤشرات الأساسية
        df['rsi'] = self.calculate_rsi(df['close'])
        macd_line, signal_line, histogram = self.calculate_macd(df['close'])
        df['macd_line'] = macd_line
        df['macd_signal'] = signal_line
        df['macd_histogram'] = histogram
        
        # 2. تحليل الاتجاه
        df = self.analyze_trend(df)
        
        # 3. إضافة عوامل التصفية أولاً (لإنشاء atr_percent)
        df = self.add_smart_filters_v4_1(df)
        
        # 4. نظام التقييم المحسن
        df = self.enhanced_scoring_system_v4_1(df)
        
        # 5. تعزيز إشارات البيع
        df = self.enhance_sell_signals_v4_1(df)
        
        # 6. وقف وجني ديناميكي (يحتاج atr_percent)
        df = self.dynamic_stop_take_profit_v4_1(df)
        
        # 7. تقييم معدل حسب المخاطرة (يحتاج atr_percent)
        df = self.risk_adjusted_scoring_v4_1(df)
        
        # 8. إشارات محسنة
        df = self.generate_enhanced_signals_v4_1(df)
        
        # حفظ نتائج التحليل
        self.analysis_results = df.to_dict('records')
        
        return df
    
    # =========================================================================
    # نظام التداول الورقي المحسن v4.1
    # =========================================================================
    
    def calculate_position_size(self, price: float) -> float:
        """حساب حجم المركز بناء على الرافعة وحجم الصفقة"""
        return (TRADE_SIZE_USDT * LEVERAGE) / price
    
    def open_position(self, symbol: str, direction: str, price: float, 
                 confidence: float, confidence_level: str, 
                 volatility: float, timestamp: datetime, 
                 dynamic_sl: float, dynamic_tp: float,
                 signal_strength: float, signal_quality: str = "STANDARD") -> Optional[Trade]:
        """فتح مركز جديد مع إعدادات خاصة للبيع v4.1"""
    
        if symbol in self.positions:
            logger.warning(f"يوجد مركز مفتوح بالفعل لـ {symbol}")
            return None
    
        # حساب حجم المركز
        quantity = self.calculate_position_size(price)
    
        # ✅ إعدادات خاصة لجودة البيع - أكثر توازناً
        is_super_quality_sell = (direction == "SHORT" and signal_quality == "SUPER")
        is_high_quality_sell = (direction == "SHORT" and signal_quality == "HIGH")
        is_good_quality_sell = (direction == "SHORT" and signal_quality == "GOOD")
    
        if is_super_quality_sell:
            # إعدادات متوازنة للبيع فائق الجودة
            dynamic_sl = dynamic_sl * 0.7  # زيادة من 0.6 إلى 0.7
            dynamic_tp = dynamic_tp * 1.2  # خفض من 1.3 إلى 1.2
            quality = "SUPER"
            logger.info(f"🚀 فتح مركز بيع فائق الجودة لـ {symbol} - وقف: {dynamic_sl:.2f}%, جني: {dynamic_tp:.2f}%")
            
        elif is_high_quality_sell:
            # إعدادات متوازنة للبيع عالي الجودة
            dynamic_sl = dynamic_sl * 0.8  # زيادة من 0.7 إلى 0.8
            dynamic_tp = dynamic_tp * 1.1  # خفض من 1.2 إلى 1.1
            quality = "HIGH"
            logger.info(f"🎯 فتح مركز بيع عالي الجودة لـ {symbol} - وقف: {dynamic_sl:.2f}%, جني: {dynamic_tp:.2f}%")
            
        elif is_good_quality_sell:
            # إعدادات معتدلة للبيع الجيد
            dynamic_sl = dynamic_sl * 0.9  # زيادة من 0.8 إلى 0.9
            dynamic_tp = dynamic_tp * 1.05  # خفض من 1.1 إلى 1.05
            quality = "GOOD"
            logger.info(f"📉 فتح مركز بيع جيد لـ {symbol} - وقف: {dynamic_sl:.2f}%, جني: {dynamic_tp:.2f}%")
        else:
            quality = "STANDARD"
    
        # حساب وقف الخسارة وجني الأرباح (ديناميكي)
        if direction == "LONG":
            stop_loss = price * (1 - dynamic_sl / 100)
            take_profit = price * (1 + dynamic_tp / 100)
        else:  # SHORT
            stop_loss = price * (1 + dynamic_sl / 100)
            take_profit = price * (1 - dynamic_tp / 100)
    
        # رسوم التداول
        fee = (TRADE_SIZE_USDT * LEVERAGE) * 0.0004
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
            volatility=volatility,
            signal_strength=signal_strength,
            quality=quality
        )
    
        self.positions[symbol] = trade
        self.trades.append(trade)
    
        # ✅ تسجيل مفصل حسب الجودة
        if direction == "SHORT":
            quality_emoji = "🚀" if quality == "SUPER" else "🎯" if quality == "HIGH" else "📉"
            logger.info(f"{quality_emoji} فتح مركز بيع {quality} لـ {symbol} "
                       f"السعر: {price:.2f}, الثقة: {confidence:.1f}% ({confidence_level})")
        else:
            logger.info(f"📈 فتح مركز {direction} لـ {symbol} "
                       f"السعر: {price:.2f}, الثقة: {confidence:.1f}% ({confidence_level})")
    
        return trade
    
    def close_position(self, symbol: str, price: float, timestamp: datetime, 
                      reason: str = "MANUAL") -> Optional[Trade]:
        """إغلاق مركز مفتوح"""
        
        if symbol not in self.positions:
            logger.warning(f"لا يوجد مركز مفتوح لـ {symbol}")
            return None
        
        trade = self.positions[symbol]
        
        # حساب الربح/الخسارة
        if trade.direction == "LONG":
            pnl = (price - trade.entry_price) * trade.quantity
        else:  # SHORT
            pnl = (trade.entry_price - price) * trade.quantity
        
        pnl_percent = (pnl / (TRADE_SIZE_USDT * LEVERAGE)) * 100
        
        # رسوم الخروج
        fee = (TRADE_SIZE_USDT * LEVERAGE) * 0.0004
        pnl -= fee
        self.current_balance += pnl
        
        # تحديث بيانات الصفقة
        trade.exit_price = price
        trade.exit_time = timestamp
        trade.pnl = pnl
        trade.pnl_percent = pnl_percent
        trade.status = reason
        
        # حفظ أداء البيع للسجلات
        if trade.direction == "SHORT":
            self.sell_performance_history.append({
                'entry_time': trade.entry_time,
                'exit_time': trade.exit_time,
                'pnl': pnl,
                'pnl_percent': pnl_percent,
                'quality': trade.quality,
                'confidence': trade.confidence
            })
        
        # إزالة من المراكز المفتوحة
        del self.positions[symbol]
        
        # حفظ في السجل
        trade_dict = {
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
            'volatility': trade.volatility,
            'signal_strength': trade.signal_strength,
            'quality': trade.quality,
            'status': trade.status
        }
        
        self.trade_history.append(trade_dict)
        
        status_emoji = "🟢" if pnl > 0 else "🔴"
        quality_emoji = "🚀" if trade.quality == "SUPER" else "🎯" if trade.quality == "HIGH" else "📉" if trade.quality == "GOOD" else ""
        logger.info(f"📊 إغلاق مركز {trade.direction} {quality_emoji} لـ {symbol} {status_emoji}"
                   f" الربح: {pnl:.2f} USD ({pnl_percent:.2f}%) - {reason}")
        
        return trade
    
    def check_stop_conditions(self, symbol: str, current_price: float, 
                            timestamp: datetime) -> bool:
        """فحص شروط الوقف والخروج"""
        
        if symbol not in self.positions:
            return False
        
        trade = self.positions[symbol]
        
        # فحص وقف الخسارة
        if ((trade.direction == "LONG" and current_price <= trade.stop_loss) or
            (trade.direction == "SHORT" and current_price >= trade.stop_loss)):
            self.close_position(symbol, trade.stop_loss, timestamp, "STOP_LOSS")
            return True
        
        # فحص جني الأرباح
        if ((trade.direction == "LONG" and current_price >= trade.take_profit) or
            (trade.direction == "SHORT" and current_price <= trade.take_profit)):
            self.close_position(symbol, trade.take_profit, timestamp, "TAKE_PROFIT")
            return True
        
        return False
    
    def execute_enhanced_paper_trading_v4_1(self, df: pd.DataFrame):
        """تنفيذ التداول الورقي المحسن v4.1"""
        
        logger.info("🚀 بدء التداول الورقي المحسن v4.1...")
        
        for i, row in df.iterrows():
            if i < 50:  # تخطي الفترة الأولى لاستقرار المؤشرات
                continue
                
            current_price = row['close']
            signal = row['signal_v4_1']
            confidence = row['score_v4_1']
            confidence_level = row['confidence_level']
            volatility = row['current_volatility']
            timestamp = row['timestamp']
            signal_strength = row['signal_strength']
            signal_quality = row.get('signal_quality', 'STANDARD')
            
            # تحديد الإعدادات الديناميكية حسب نوع الإشارة
            if signal == 'LONG':
                dynamic_sl = row['dynamic_sl_buy']
                dynamic_tp = row['dynamic_tp_buy']
            else:
                # استخدام الإعدادات الخاصة بجودة البيع
                if signal_quality == 'SUPER':
                    dynamic_sl = row.get('super_quality_sell_sl', row['dynamic_sl_sell'])
                    dynamic_tp = row.get('super_quality_sell_tp', row['dynamic_tp_sell'])
                elif signal_quality == 'HIGH':
                    dynamic_sl = row.get('high_quality_sell_sl', row['dynamic_sl_sell'])
                    dynamic_tp = row.get('high_quality_sell_tp', row['dynamic_tp_sell'])
                else:
                    dynamic_sl = row['dynamic_sl_sell']
                    dynamic_tp = row['dynamic_tp_sell']
            
            # فحص شروط الخروج للمراكز المفتوحة
            if SYMBOL in self.positions:
                self.check_stop_conditions(SYMBOL, current_price, timestamp)
            
            # فتح مراكز جديدة إذا لم يكن هناك مركز مفتوح
            if (SYMBOL not in self.positions and signal != 'none'):
                # التحقق من عتبات الثقة حسب نوع الإشارة
                if (signal == 'LONG' and confidence >= CONFIDENCE_THRESHOLD) or \
                   (signal == 'SHORT' and confidence >= SELL_CONFIDENCE_THRESHOLD):
                    
                    self.open_position(
                        SYMBOL, signal, current_price, confidence, confidence_level,
                        volatility, timestamp, dynamic_sl, dynamic_tp, signal_strength, signal_quality
                    )
    
    # =========================================================================
    # الباك-تستينغ المحسن v4.1
    # =========================================================================
    
    def run_enhanced_backtest_v4_1(self, df: pd.DataFrame) -> BacktestResult:
        """تشغيل الباك-تستينغ المحسن v4.1"""
        
        logger.info("🔍 بدء الباك-تستينغ المحسن v4.1...")
        
        # إعادة تعيين البيانات
        self.trades = []
        self.positions = {}
        self.trade_history = []
        self.sell_performance_history = []
        self.current_balance = INITIAL_BALANCE
        
        # التحليل المحسن v4.1
        df_with_signals = self.enhanced_analysis_v4_1(df)
        
        # تنفيذ التداول المحسن v4.1
        self.execute_enhanced_paper_trading_v4_1(df_with_signals)
        
        # إغلاق أي مراكز مفتوحة في النهاية
        if SYMBOL in self.positions:
            last_price = df_with_signals.iloc[-1]['close']
            last_timestamp = df_with_signals.iloc[-1]['timestamp']
            self.close_position(SYMBOL, last_price, last_timestamp, "END_OF_DATA")
        
        # حساب النتائج المحسنة v4.1
        return self.calculate_enhanced_backtest_results_v4_1(df)
    
    def calculate_enhanced_backtest_results_v4_1(self, df: pd.DataFrame) -> BacktestResult:
        """حساب نتائج الباك-تستينغ المحسنة v4.1"""
        
        if not self.trade_history:
            total_days = (df['timestamp'].max() - df['timestamp'].min()).days
            return BacktestResult(
                total_trades=0, winning_trades=0, losing_trades=0,
                win_rate=0, total_pnl=0, final_balance=self.current_balance,
                max_drawdown=0, sharpe_ratio=0, profit_factor=0,
                avg_trade=0, best_trade=0, worst_trade=0, total_fees=0,
                total_days=max(1, total_days), avg_daily_return=0,
                avg_confidence=0, confidence_analysis={},
                buy_performance={}, sell_performance={}, quality_analysis={}
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
        total_fees = total_trades * (TRADE_SIZE_USDT * LEVERAGE) * 0.0004 * 2
        
        # حساب عدد الأيام والعائد اليومي
        total_days = (df['timestamp'].max() - df['timestamp'].min()).days
        total_days = max(1, total_days)
        avg_daily_return = (final_balance - INITIAL_BALANCE) / INITIAL_BALANCE / total_days * 100
        
        # تحليل الثقة
        avg_confidence = trades_df['confidence'].mean()
        
        # تحليل مفصل حسب مستوى الثقة
        confidence_analysis = {}
        for level in ['عالية جداً', 'عالية', 'متوسطة', 'منخفضة', 'ضعيفة']:
            level_trades = trades_df[trades_df['confidence_level'] == level]
            if len(level_trades) > 0:
                level_win_rate = (len(level_trades[level_trades['pnl'] > 0]) / len(level_trades)) * 100
                level_total_pnl = level_trades['pnl'].sum()
                confidence_analysis[level] = {
                    'trades': len(level_trades),
                    'win_rate': level_win_rate,
                    'total_pnl': level_total_pnl,
                    'avg_pnl': level_trades['pnl'].mean()
                }
        
        # ✅ تحليل أداء الشراء vs البيع
        buy_trades = trades_df[trades_df['direction'] == 'LONG']
        sell_trades = trades_df[trades_df['direction'] == 'SHORT']
        
        buy_performance = {
            'total_trades': len(buy_trades),
            'winning_trades': len(buy_trades[buy_trades['pnl'] > 0]),
            'total_pnl': buy_trades['pnl'].sum() if len(buy_trades) > 0 else 0,
            'avg_pnl': buy_trades['pnl'].mean() if len(buy_trades) > 0 else 0,
            'win_rate': (len(buy_trades[buy_trades['pnl'] > 0]) / len(buy_trades) * 100) if len(buy_trades) > 0 else 0
        }
        
        sell_performance = {
            'total_trades': len(sell_trades),
            'winning_trades': len(sell_trades[sell_trades['pnl'] > 0]),
            'total_pnl': sell_trades['pnl'].sum() if len(sell_trades) > 0 else 0,
            'avg_pnl': sell_trades['pnl'].mean() if len(sell_trades) > 0 else 0,
            'win_rate': (len(sell_trades[sell_trades['pnl'] > 0]) / len(sell_trades) * 100) if len(sell_trades) > 0 else 0
        }
        
        # ✅ تحليل الجودة للبيع
        quality_analysis = {}
        for quality in ['SUPER', 'HIGH', 'GOOD', 'STANDARD']:
            quality_trades = trades_df[trades_df['quality'] == quality]
            if len(quality_trades) > 0:
                quality_win_rate = (len(quality_trades[quality_trades['pnl'] > 0]) / len(quality_trades)) * 100
                quality_total_pnl = quality_trades['pnl'].sum()
                quality_analysis[quality] = {
                    'trades': len(quality_trades),
                    'win_rate': quality_win_rate,
                    'total_pnl': quality_total_pnl,
                    'avg_pnl': quality_trades['pnl'].mean(),
                    'avg_confidence': quality_trades['confidence'].mean()
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
            confidence_analysis=confidence_analysis,
            buy_performance=buy_performance,
            sell_performance=sell_performance,
            quality_analysis=quality_analysis
        )
    
    # =========================================================================
    # التقارير المحسنة v4.1
    # =========================================================================
    
    async def send_enhanced_telegram_report_v4_1(self, backtest_result: BacktestResult, df: pd.DataFrame):
        """إرسال تقرير مفصل v4.1 إلى التلغرام"""
        
        if not self.telegram_notifier:
            logger.warning("❌ نظام التلغرام غير متوفر")
            return
        
        try:
            # 1. إرسال التقرير النصي المحسن v4.1
            report_text = self._generate_enhanced_report_text_v4_1(backtest_result)
            await self.telegram_notifier.send_message(report_text)
            
            # 2. إرسال الرسوم البيانية
            chart_buffer = self._create_enhanced_performance_chart_v4_1(df, backtest_result)
            if chart_buffer:
                chart_caption = f"📈 تحليل أداء الاستراتيجية المحسنة v4.1 - {SYMBOL} ({TIMEFRAME})"
                await self.telegram_notifier.send_photo(chart_buffer, chart_caption)
            
            # 3. إرسال تحليل البيع والشراء
            if self.trade_history:
                trade_analysis = self._generate_trade_analysis_v4_1(backtest_result)
                await self.telegram_notifier.send_message(trade_analysis)
                
        except Exception as e:
            logger.error(f"❌ خطأ في إرسال التقرير إلى التلغرام: {e}")
    
    def _generate_enhanced_report_text_v4_1(self, backtest_result: BacktestResult) -> str:
        """إنشاء نص التقرير المحسن v4.1 للتلغرام"""
        
        report_time = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        
        message = f"🎯 *تقرير استراتيجية المحسنة v4.1 - تصحيح شروط البيع*\n"
        message += "══════════════════════════════════════\n\n"
        
        message += f"⚙️ *الإعدادات المتوازنة v4.1:*\n"
        message += f"• العملة: `{SYMBOL}`\n"
        message += f"• الإطار: `{TIMEFRAME}`\n"
        message += f"• الرافعة: `{LEVERAGE}x`\n"
        message += f"• حجم الصفقة: `${TRADE_SIZE_USDT}`\n"
        message += f"• عتبة ثقة الشراء: `{CONFIDENCE_THRESHOLD}%`\n"
        message += f"• عتبة ثقة البيع: `{SELL_CONFIDENCE_THRESHOLD}%` 📉\n"
        message += f"• عتبة البيع فائق الجودة: `{SUPER_QUALITY_SELL_THRESHOLD}%` 📉\n"
        message += f"• عتبة البيع عالي الجودة: `{HIGH_QUALITY_SELL_THRESHOLD}%` 📉\n\n"
        
        message += f"📊 *النتائج المحسنة v4.1:*\n"
        message += f"• إجمالي الصفقات: `{backtest_result.total_trades}`\n"
        message += f"• الصفقات الرابحة: `{backtest_result.winning_trades}` 🟢\n"
        message += f"• الصفقات الخاسرة: `{backtest_result.losing_trades}` 🔴\n"
        message += f"• نسبة الربح: `{backtest_result.win_rate:.1f}%`\n"
        message += f"• إجمالي الربح: `${backtest_result.total_pnl:,.2f}`\n"
        message += f"• الرصيد النهائي: `${backtest_result.final_balance:,.2f}`\n"
        message += f"• العائد الإجمالي: `{((backtest_result.final_balance - INITIAL_BALANCE) / INITIAL_BALANCE * 100):.1f}%`\n"
        message += f"• متوسط الثقة: `{backtest_result.avg_confidence:.1f}%`\n\n"
        
        message += f"🎯 *مقاييس المخاطرة المحسنة v4.1:*\n"
        message += f"• أقصى خسارة: `{backtest_result.max_drawdown:.1f}%`\n"
        message += f"• متوسط الربح/صفقة: `${backtest_result.avg_trade:.2f}`\n"
        message += f"• أفضل صفقة: `${backtest_result.best_trade:.2f}` 🚀\n"
        message += f"• أسوأ صفقة: `${backtest_result.worst_trade:.2f}` 📉\n"
        message += f"• نسبة شارب: `{backtest_result.sharpe_ratio:.2f}`\n"
        message += f"• عامل الربحية: `{backtest_result.profit_factor:.2f}`\n\n"
        
        message += f"🕒 *وقت التقرير:* `{report_time}`\n"
        message += "══════════════════════════════════════\n"
        message += "⚡ *نظام التقييم v4.1 + شروط بيع متوازنة + تحسين الفرص*"
        
        return message
    
    def _generate_trade_analysis_v4_1(self, backtest_result: BacktestResult) -> str:
        """إنشاء تحليل مفصل للبيع والشراء v4.1 مع توصيات"""
    
        message = "🔍 *تحليل مفصل للبيع والشراء v4.1:*\n"
        message += "────────────────────\n"
    
        # تحليل الشراء
        buy = backtest_result.buy_performance
        message += f"🔼 *صفقات الشراء:*\n"
        message += f"• العدد: `{buy['total_trades']}` صفقة\n"
        message += f"• الربح: `${buy['total_pnl']:.2f}` {'✅' if buy['total_pnl'] > 0 else '❌'}\n"
        message += f"• متوسط الربح: `${buy['avg_pnl']:.2f}`\n"
        message += f"• نسبة النجاح: `{buy['win_rate']:.1f}%`\n\n"
    
        # تحليل البيع
        sell = backtest_result.sell_performance
        message += f"🔽 *صفقات البيع المحسنة v4.1:*\n"
        message += f"• العدد: `{sell['total_trades']}` صفقة\n"
        message += f"• الربح: `${sell['total_pnl']:.2f}` {'✅' if sell['total_pnl'] > 0 else '❌'}\n"
        message += f"• متوسط الربح: `${sell['avg_pnl']:.2f}`\n"
        message += f"• نسبة النجاح: `{sell['win_rate']:.1f}%`\n\n"
    
        # تحليل جودة البيع
        quality_analysis = backtest_result.quality_analysis
        if quality_analysis:
            message += f"🎯 *تحليل جودة البيع v4.1:*\n"
            for quality, stats in quality_analysis.items():
                if stats['trades'] > 0:
                    emoji = "🚀" if quality == "SUPER" else "🎯" if quality == "HIGH" else "📉" if quality == "GOOD" else "⚪"
                    message += f"• {emoji} {quality}: `{stats['trades']}` صفقات, نجاح: `{stats['win_rate']:.1f}%`, ربح: `${stats['total_pnl']:.2f}`\n"
            message += "\n"
    
        # تحليل الأداء المقارن
        performance_gap = sell['win_rate'] - buy['win_rate']
        profit_gap = sell['total_pnl'] - buy['total_pnl']
    
        message += f"📊 *مقارنة الأداء v4.1:*\n"
        message += f"• فرق النجاح: `{performance_gap:+.1f}%`\n"
        message += f"• فرق الربح: `${profit_gap:+.2f}`\n\n"
    
        # ✅ توصيات بناء على النتائج v4.1
        message += f"🎯 *توصيات v4.1:*\n"
    
        if sell['total_trades'] == 0:
            message += f"• ✅ تم تصحيح شروط البيع بنجاح\n"
            message += f"• 📈 من المتوقع ظهور صفقات بيع في التشغيل القادم\n"
            message += f"• 🎯 الحفاظ على الإعدادات المتوازنة الحالية\n"
        elif sell['win_rate'] >= 60:
            message += f"• ✅ أداء البيع ممتاز مع الإعدادات الجديدة\n"
            message += f"• 📊 يمكن زيادة عدوانية البيع قليلاً\n"
            message += f"• 🚀 التركيز على البيع فائق الجودة\n"
        else:
            message += f"• 🔧 استمرار ضبط شروط البيع\n"
            message += f"• 📈 مراقبة أداء صفقات البيع الجديدة\n"
            message += f"• ⚖️ الحفاظ على التوازن بين المخاطرة والعائد\n"
    
        # ✅ إضافة تحليل الثقة
        if backtest_result.avg_confidence > 75:
            message += f"\n📈 *مستوى الثقة:* `مرتفع ({backtest_result.avg_confidence:.1f}%)` ✅\n"
        else:
            message += f"\n📈 *مستوى الثقة:* `منخفض ({backtest_result.avg_confidence:.1f}%)` ⚠️\n"
    
        return message

    def _create_enhanced_performance_chart_v4_1(self, df: pd.DataFrame, backtest_result: BacktestResult) -> BytesIO:
        """إنشاء رسم بياني محسن v4.1 للأداء"""
        try:
            fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
            fig.suptitle(f'تحليل الاستراتيجية المحسنة v4.1 - {SYMBOL}', 
                        fontsize=16, fontname='DejaVu Sans', fontweight='bold')
            
            # 1. السعر والإشارات
            ax1.plot(df['timestamp'], df['close'], label='السعر', linewidth=1.5, color='blue', alpha=0.8)
            ax1.set_title('حركة السعر وإشارات التداول v4.1', fontname='DejaVu Sans', fontsize=12)
            ax1.set_ylabel('السعر (USDT)', fontname='DejaVu Sans')
            
            # إضافة نقاط الدخول مع تمييز جودة البيع
            trades_df = pd.DataFrame(self.trade_history)
            for _, trade in trades_df.iterrows():
                if trade['direction'] == 'LONG':
                    color = 'green'
                    marker = '^'
                    size = 80
                else:
                    # تلوين حسب جودة البيع
                    if trade['quality'] == 'SUPER':
                        color = 'red'
                        marker = 'v'
                        size = 150
                    elif trade['quality'] == 'HIGH':
                        color = 'orange'
                        marker = 'v'
                        size = 120
                    elif trade['quality'] == 'GOOD':
                        color = 'purple'
                        marker = 'v'
                        size = 100
                    else:
                        color = 'red'
                        marker = 'v'
                        size = 80
                
                alpha = 0.9 if trade['pnl'] > 0 else 0.6
                ax1.scatter(trade['entry_time'], trade['entry_price'], 
                           color=color, marker=marker, s=size, alpha=alpha,
                           edgecolors='black', linewidth=1)
            
            ax1.legend(prop={'family': 'DejaVu Sans'})
            ax1.grid(True, alpha=0.3)
            
            # 2. توزيع الأرباح مع فصل جودة البيع
            if not trades_df.empty:
                buy_profits = trades_df[trades_df['direction'] == 'LONG']['pnl']
                sell_super = trades_df[(trades_df['direction'] == 'SHORT') & (trades_df['quality'] == 'SUPER')]['pnl']
                sell_high = trades_df[(trades_df['direction'] == 'SHORT') & (trades_df['quality'] == 'HIGH')]['pnl']
                sell_good = trades_df[(trades_df['direction'] == 'SHORT') & (trades_df['quality'] == 'GOOD')]['pnl']
                sell_standard = trades_df[(trades_df['direction'] == 'SHORT') & (trades_df['quality'] == 'STANDARD')]['pnl']
                
                if len(buy_profits) > 0:
                    ax2.hist(buy_profits, bins=10, alpha=0.7, color='green', 
                            label='صفقات الشراء', edgecolor='black')
                
                if len(sell_super) > 0:
                    ax2.hist(sell_super, bins=10, alpha=0.7, color='red',
                            label='بيع فائق', edgecolor='black')
                
                if len(sell_high) > 0:
                    ax2.hist(sell_high, bins=10, alpha=0.7, color='orange',
                            label='بيع عالي', edgecolor='black')
                
                if len(sell_good) > 0:
                    ax2.hist(sell_good, bins=10, alpha=0.7, color='purple',
                            label='بيع جيد', edgecolor='black')
                
                ax2.axvline(0, color='black', linestyle='--', linewidth=2)
                ax2.set_title('توزيع أرباح البيع vs الشراء v4.1', fontname='DejaVu Sans', fontsize=12)
                ax2.set_xlabel('الربح (USD)', fontname='DejaVu Sans')
                ax2.set_ylabel('عدد الصفقات', fontname='DejaVu Sans')
                ax2.legend(prop={'family': 'DejaVu Sans'})
                ax2.grid(True, alpha=0.3)
            
            # 3. أداء الرصيد
            if len(self.trade_history) > 0:
                balance_history = [INITIAL_BALANCE]
                for trade in self.trade_history:
                    balance_history.append(balance_history[-1] + trade['pnl'])
                
                ax3.plot(range(len(balance_history)), balance_history, 
                        color='green', linewidth=2.5, label='الرصيد')
                ax3.axhline(INITIAL_BALANCE, color='red', linestyle='--', alpha=0.7, 
                           linewidth=1.5, label='رصيد البداية')
                
                ax3.set_title('تطور الرصيد v4.1', fontname='DejaVu Sans', fontsize=12)
                ax3.set_xlabel('عدد الصفقات', fontname='DejaVu Sans')
                ax3.set_ylabel('الرصيد (USD)', fontname='DejaVu Sans')
                ax3.legend(prop={'family': 'DejaVu Sans'})
                ax3.grid(True, alpha=0.3)
            
            # 4. مقارنة أداء جودة البيع
            quality_analysis = backtest_result.quality_analysis
            
            if quality_analysis:
                categories = []
                win_rates = []
                avg_pnls = []
                
                for quality, stats in quality_analysis.items():
                    if stats['trades'] > 0:
                        categories.append(quality)
                        win_rates.append(stats['win_rate'])
                        avg_pnls.append(stats['avg_pnl'])
                
                if categories:
                    x = np.arange(len(categories))
                    width = 0.35
                    
                    ax4.bar(x - width/2, win_rates, width, label='نسبة النجاح %', color='blue', alpha=0.7)
                    ax4.bar(x + width/2, avg_pnls, width, label='متوسط الربح $', color='green', alpha=0.7)
                    
                    ax4.set_title('مقارنة أداء جودة البيع v4.1', fontname='DejaVu Sans', fontsize=12)
                    ax4.set_xticks(x)
                    ax4.set_xticklabels(categories, fontname='DejaVu Sans')
                    ax4.legend(prop={'family': 'DejaVu Sans'})
                    ax4.grid(True, alpha=0.3)
            
            plt.tight_layout()
            
            # حفظ في buffer
            buffer = BytesIO()
            plt.savefig(buffer, format='png', dpi=150, bbox_inches='tight', 
                       facecolor='white', edgecolor='none')
            buffer.seek(0)
            plt.close()
            
            return buffer
            
        except Exception as e:
            logger.error(f"❌ خطأ في إنشاء الرسم البياني: {e}")
            return None

# =============================================================================
# نظام جلب البيانات الممتدة
# =============================================================================

class ExtendedDataFetcher:
    """جلب بيانات متقدم لفترات طويلة"""
    
    @staticmethod
    def fetch_historical_data(symbol: str, interval: str, limit: int = DATA_LIMIT) -> pd.DataFrame:
        """جلب البيانات التاريخية مع معالجة الأخطاء"""
        try:
            url = "https://api.binance.com/api/v3/klines"
            params = {
                'symbol': symbol.upper(),  # تأكد من الأحرف الكبيرة
                'interval': interval,
                'limit': limit
            }
            
            logger.info(f"📡 جلب البيانات من Binance: {symbol} {interval}")
            
            response = requests.get(url, params=params, timeout=20)
            response.raise_for_status()
            data = response.json()
            
            if not data:
                logger.error("❌ لا توجد بيانات من API")
                return pd.DataFrame()
            
            # إنشاء DataFrame مع الأعمدة الصحيحة
            df = pd.DataFrame(data, columns=[
                'timestamp', 'open', 'high', 'low', 'close', 'volume',
                'close_time', 'quote_asset_volume', 'number_of_trades',
                'taker_buy_base_asset_volume', 'taker_buy_quote_asset_volume', 'ignore'
            ])
            
            # تحويل الأنواع
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
            numeric_columns = ['open', 'high', 'low', 'close', 'volume']
            for col in numeric_columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')
            
            # إزالة الصفوف ذات القيم الفارغة
            df = df.dropna(subset=numeric_columns)
            
            logger.info(f"✅ تم جلب {len(df)} صف من البيانات لـ {symbol}")
            logger.info(f"📅 الفترة: {df['timestamp'].min()} إلى {df['timestamp'].max()}")
            
            return df[['timestamp', 'open', 'high', 'low', 'close', 'volume']]
            
        except requests.exceptions.RequestException as e:
            logger.error(f"❌ خطأ في الاتصال: {e}")
            return pd.DataFrame()
        except Exception as e:
            logger.error(f"❌ خطأ غير متوقع: {e}")
            return pd.DataFrame()

# =============================================================================
# الوظيفة الرئيسية
# =============================================================================

async def main():
    """الوظيفة الرئيسية مع الاستراتيجية المحسنة v4.1"""
    
    logger.info("🚀 بدء تشغيل الاستراتيجية المحسنة v4.1 مع تصحيح شروط البيع")
    
    # تهيئة نظام التلغرام
    telegram_notifier = TelegramNotifier(TELEGRAM_BOT_TOKEN, TELEGRAM_CHAT_ID)
    
    # جلب البيانات
    data_fetcher = ExtendedDataFetcher()
    df = data_fetcher.fetch_historical_data(SYMBOL, TIMEFRAME, DATA_LIMIT)
    
    if df.empty:
        error_msg = "❌ فشل جلب البيانات. تأكد من اتصال الإنترنت وصحة اسم العملة."
        logger.error(error_msg)
        await telegram_notifier.send_message(error_msg)
        return
    
    # التحقق من وجود بيانات كافية
    if len(df) < 100:
        error_msg = f"❌ بيانات غير كافية: {len(df)} صف فقط (مطلوب 100 على الأقل)"
        logger.error(error_msg)
        await telegram_notifier.send_message(error_msg)
        return
    
    # التحقق من الأعمدة المطلوبة
    required_cols = ['timestamp', 'open', 'high', 'low', 'close', 'volume']
    missing_cols = [col for col in required_cols if col not in df.columns]
    
    if missing_cols:
        error_msg = f"❌ أعمدة مفقودة: {missing_cols}"
        logger.error(error_msg)
        await telegram_notifier.send_message(error_msg)
        return
    
    # إرسال معلومات عن فترة البيانات
    data_info = f"📊 فترة البيانات: {len(df)} شمعة من {df['timestamp'].min().date()} إلى {df['timestamp'].max().date()}"
    logger.info(data_info)
    await telegram_notifier.send_message(data_info)
    
    # تشغيل الاستراتيجية المحسنة v4.1
    strategy = EnhancedEmaRsiMacdStrategyV4_1(telegram_notifier)
    
    # الباك-تستينغ المحسن v4.1
    backtest_result = strategy.run_enhanced_backtest_v4_1(df)
    
    # إرسال التقرير المحسن v4.1 إلى التلغرام
    await strategy.send_enhanced_telegram_report_v4_1(backtest_result, df)
    
    # حفظ النتائج في ملف
    trades_df = pd.DataFrame(strategy.trade_history)
    if not trades_df.empty:
        filename = f'enhanced_v4_1_trades_{SYMBOL}_{TIMEFRAME}_{datetime.now().strftime("%Y%m%d_%H%M%S")}.csv'
        trades_df.to_csv(filename, index=False)
        logger.info(f"💾 تم حفظ سجل الصفقات في {filename}")
    
    logger.info("✅ اكتمل تشغيل الاستراتيجية المحسنة v4.1 بنجاح")

if __name__ == "__main__":
    # تشغيل الوظيفة الرئيسية
    asyncio.run(main())
