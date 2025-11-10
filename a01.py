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
# إعدادات التداول من متغيرات البيئة
# =============================================================================

SYMBOL = os.getenv("TRADING_SYMBOL", "BNBUSDT")
TIMEFRAME = os.getenv("TRADING_TIMEFRAME", "1h")
STOP_LOSS_PERCENT = float(os.getenv("STOP_LOSS_PERCENT", "0.8"))
TAKE_PROFIT_PERCENT = float(os.getenv("TAKE_PROFIT_PERCENT", "2.5"))
TRADE_SIZE_USDT = float(os.getenv("TRADE_SIZE_USDT", "100.0"))
LEVERAGE = int(os.getenv("LEVERAGE", "10"))
INITIAL_BALANCE = float(os.getenv("INITIAL_BALANCE", "5000.0"))
CONFIDENCE_THRESHOLD = int(os.getenv("CONFIDENCE_THRESHOLD", "70"))

# تحديث إعدادات البيع الذكية v4.2
SELL_CONFIDENCE_THRESHOLD = int(os.getenv("SELL_CONFIDENCE_THRESHOLD", "65"))
SUPER_QUALITY_SELL_THRESHOLD = int(os.getenv("SUPER_QUALITY_SELL_THRESHOLD", "75"))
HIGH_QUALITY_SELL_THRESHOLD = int(os.getenv("HIGH_QUALITY_SELL_THRESHOLD", "70"))

# إعدادات التحسين الجديدة
VOLUME_BOOST_FACTOR = float(os.getenv("VOLUME_BOOST_FACTOR", "1.2"))
RSI_SELL_OPTIMIZATION = bool(os.getenv("RSI_SELL_OPTIMIZATION", "True"))
ADAPTIVE_CONFIDENCE = bool(os.getenv("ADAPTIVE_CONFIDENCE", "True"))

# إعدادات مدة الاختبار
DATA_LIMIT = int(os.getenv("DATA_LIMIT", "2000"))
TEST_DAYS = int(os.getenv("TEST_DAYS", "180"))

# إعدادات التلغرام
TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN", "")
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID", "")

# إعداد التسجيل
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger("Enhanced_EMA_RSI_MACD_Strategy_v4_2")

# =============================================================================
# هياكل البيانات المحدثة
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
    entry_conditions: Dict = None
    loss_reason: str = ""

@dataclass
class LossAnalysis:
    total_losing_trades: int
    loss_reasons: Dict
    avg_loss_per_trade: float
    common_patterns: List[str]
    improvement_suggestions: List[str]

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
    loss_analysis: LossAnalysis

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
# محرك الاستراتيجية الذكية v4.2 مع تحليل الخسائر
# =============================================================================

class EnhancedEmaRsiMacdStrategyV4:
    """استراتيجية محسنة v4.2 مع تحسينات ذكية لأداء البيع وتحليل الخسائر"""
    
    def __init__(self, telegram_notifier: TelegramNotifier = None):
        self.name = "enhanced_ema_rsi_macd_v4_2"
        self.trades: List[Trade] = []
        self.balance = INITIAL_BALANCE
        self.current_balance = INITIAL_BALANCE
        self.positions = {}
        self.trade_history = []
        self.analysis_results = []
        self.telegram_notifier = telegram_notifier
        self.sell_performance_history = []
        self.market_analysis = {}
    
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
    
    def enhanced_scoring_system_v4(self, df: pd.DataFrame) -> pd.DataFrame:
        """نظام التقييم المحسن v4 مع تحسين متقدم لصفقات البيع"""
        
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
        df.loc[high_confidence_mask, 'score_v4'] = df.loc[high_confidence_mask, 'total_score'] * 0.85
        
        # ✅ التعزيز: زيادة وزن الإشارات متوسطة الثقة الناجحة
        medium_confidence_mask = (df['total_score'] >= 60) & (df['total_score'] < 80)
        df.loc[medium_confidence_mask, 'score_v4'] = df.loc[medium_confidence_mask, 'total_score'] * 1.15
        
        # ✅ الإشارات المنخفضة تبقى كما هي
        low_confidence_mask = df['total_score'] < 60
        df.loc[low_confidence_mask, 'score_v4'] = df.loc[low_confidence_mask, 'total_score']
        
        df['score_v4'] = df['score_v4'].clip(0, 100)
        
        return df
    
    # =========================================================================
    # التحليل الذكي للسوق v4.2
    # =========================================================================
    
    def analyze_market_conditions_v4_2(self, df: pd.DataFrame) -> Dict:
        """تحليل ذكي لظروف السوق لتحسين توقيت البيع"""
        if len(df) < 50:
            return {
                'trend_strength': 0.5,
                'volatility_regime': 'UNKNOWN',
                'market_phase': 'UNKNOWN',
                'volume_profile': {'trend': 'UNKNOWN', 'confidence': 0, 'ratio': 1},
                'support_resistance': {'support': 0, 'resistance': 0, 'distance_to_resistance': 0, 'near_resistance': False},
                'sell_opportunities': {'high_confidence_sells': 0, 'medium_confidence_sells': 0, 'conditions_met': []}
            }
        
        current_data = df.iloc[-1]
        
        market_analysis = {
            'trend_strength': self.calculate_trend_strength(df),
            'volatility_regime': self.detect_volatility_regime(df),
            'market_phase': self.identify_market_phase(df),
            'volume_profile': self.analyze_volume_profile(df),
            'support_resistance': self.detect_support_resistance(df)
        }
        
        # تحليل فرص البيع بناء على ظروف السوق
        sell_opportunities = self.identify_sell_opportunities_v4_2(df, market_analysis)
        market_analysis['sell_opportunities'] = sell_opportunities
        
        self.market_analysis = market_analysis
        return market_analysis
    
    def calculate_trend_strength(self, df: pd.DataFrame) -> float:
        """حساب قوة الاتجاه بشكل أكثر دقة"""
        if len(df) < 50:
            return 0.5
            
        price_change = (df['close'].iloc[-1] - df['close'].iloc[-50]) / df['close'].iloc[-50]
        
        # حساب محاذاة المتوسطات المتحركة
        ema_alignment = 0
        if 'ema_9' in df.columns and 'ema_21' in df.columns:
            ema_alignment = len(df[df['ema_9'] > df['ema_21']]) / len(df)
        
        # حساب التقلبات
        volatility = df['close'].pct_change().std()
        if pd.isna(volatility):
            volatility = 0.02
        
        trend_strength = (abs(price_change) * 0.4 + ema_alignment * 0.4 + (1 - min(volatility, 0.1)) * 0.2)
        return min(trend_strength, 1.0)
    
    def detect_volatility_regime(self, df: pd.DataFrame) -> str:
        """كشف نظام التقلبات الحالي"""
        if 'atr_percent' not in df.columns:
            return "UNKNOWN"
            
        current_atr = df['atr_percent'].iloc[-1]
        avg_atr = df['atr_percent'].mean()
        
        if pd.isna(current_atr) or pd.isna(avg_atr):
            return "NORMAL_VOLATILITY"
            
        if current_atr > avg_atr * 1.5:
            return "HIGH_VOLATILITY"
        elif current_atr < avg_atr * 0.7:
            return "LOW_VOLATILITY"
        else:
            return "NORMAL_VOLATILITY"
    
    def identify_market_phase(self, df: pd.DataFrame) -> str:
        """تحديد مرحلة السوق الحالية"""
        if len(df) < 20:
            return "UNKNOWN"
            
        # حساب المتوسطات المتحركة للاتجاه
        sma_20 = df['close'].rolling(20).mean()
        sma_50 = df['close'].rolling(50).mean()
        
        current_price = df['close'].iloc[-1]
        price_vs_sma20 = current_price / sma_20.iloc[-1] if sma_20.iloc[-1] > 0 else 1
        price_vs_sma50 = current_price / sma_50.iloc[-1] if sma_50.iloc[-1] > 0 else 1
        
        if price_vs_sma20 > 1.02 and price_vs_sma50 > 1.05:
            return "BULLISH"
        elif price_vs_sma20 < 0.98 and price_vs_sma50 < 0.95:
            return "BEARISH"
        else:
            return "SIDEWAYS"
    
    def analyze_volume_profile(self, df: pd.DataFrame) -> Dict:
        """تحليل ملف الحجم لتحسين توقيت البيع"""
        if len(df) < 20:
            return {"trend": "UNKNOWN", "confidence": 0, "ratio": 1}
            
        volume_trend = "NEUTRAL"
        current_volume = df['volume'].iloc[-1]
        avg_volume_20 = df['volume'].rolling(20).mean().iloc[-1]
        
        if pd.isna(current_volume) or pd.isna(avg_volume_20) or avg_volume_20 == 0:
            return {"trend": "UNKNOWN", "confidence": 0, "ratio": 1}
        
        if current_volume > avg_volume_20 * 1.3:
            volume_trend = "HIGH"
        elif current_volume < avg_volume_20 * 0.7:
            volume_trend = "LOW"
            
        volume_confidence = min(abs(current_volume - avg_volume_20) / avg_volume_20, 1.0)
        
        return {
            "trend": volume_trend,
            "confidence": volume_confidence,
            "ratio": current_volume / avg_volume_20
        }
    
    def detect_support_resistance(self, df: pd.DataFrame) -> Dict:
        """كشف مستويات الدعم والمقاومة"""
        if len(df) < 50:
            return {"support": 0, "resistance": 0, "distance_to_resistance": 0, "near_resistance": False}
            
        # استخدام أعلى وأقل 20 فترة للدعم والمقاومة
        resistance = df['high'].rolling(20).max().iloc[-1]
        support = df['low'].rolling(20).min().iloc[-1]
        current_price = df['close'].iloc[-1]
        
        distance_to_resistance = (resistance - current_price) / current_price if current_price > 0 else 0
        
        return {
            "support": support,
            "resistance": resistance,
            "distance_to_resistance": distance_to_resistance,
            "near_resistance": distance_to_resistance < 0.02  # within 2%
        }
    
    def identify_sell_opportunities_v4_2(self, df: pd.DataFrame, market_analysis: Dict) -> Dict:
        """تحديد فرص البيع الذكية بناء على ظروف السوق"""
        opportunities = {
            "high_confidence_sells": 0,
            "medium_confidence_sells": 0,
            "conditions_met": []
        }
        
        if len(df) == 0:
            return opportunities
            
        current_data = df.iloc[-1]
        market_phase = market_analysis['market_phase']
        near_resistance = market_analysis['support_resistance']['near_resistance']
        volume_profile = market_analysis['volume_profile']
        
        # شروط البيع الذكية v4.2
        conditions = []
        
        # 1. شرط المقاومة + حجم مرتفع
        if near_resistance and volume_profile['trend'] == "HIGH":
            conditions.append("RESISTANCE_HIGH_VOLUME")
            opportunities["high_confidence_sells"] += 1
            
        # 2. شرط السوق الهابط + RSI مرتفع
        if market_phase == "BEARISH" and current_data.get('rsi', 0) > 60:
            conditions.append("BEARISH_MARKET_RSI")
            opportunities["high_confidence_sells"] += 1
            
        # 3. شرط التقلبات العالية + اتجاه هابط
        if (market_analysis['volatility_regime'] == "HIGH_VOLATILITY" and 
            current_data.get('ema_9', 0) < current_data.get('ema_21', 1)):
            conditions.append("HIGH_VOL_DOWNTREND")
            opportunities["medium_confidence_sells"] += 1
            
        # 4. شرط الحجم المنخفض في الارتفاع (توزيع)
        if (current_data.get('close', 0) > current_data.get('ema_21', 0) and 
            volume_profile['trend'] == "LOW" and
            current_data.get('rsi', 0) > 65):
            conditions.append("LOW_VOLUME_DISTRIBUTION")
            opportunities["medium_confidence_sells"] += 1
            
        opportunities["conditions_met"] = conditions
        return opportunities

    # =========================================================================
    # نظام التقييم الذكي v4.2
    # =========================================================================
    
    def intelligent_scoring_system_v4_2(self, df: pd.DataFrame) -> pd.DataFrame:
        """نظام التقييم الذكي v4.2 مع تحسينات للبيع"""
        
        # التحليل الأساسي يبقى كما هو
        df = self.enhanced_scoring_system_v4(df)
        
        # ✅ التحديث الذكي v4.2: تعزيز إشارات البيع في ظروف السوق المناسبة
        market_analysis = self.analyze_market_conditions_v4_2(df)
        
        # تعزيز البيع عند وجود فرص عالية الثقة
        high_confidence_opportunities = market_analysis['sell_opportunities']['high_confidence_sells']
        if high_confidence_opportunities > 0:
            sell_conditions = (
                (df['ema_9'] < df['ema_21']) & 
                (df['rsi'] > 58)  # تخفيض عتبة RSI للبيع
            )
            df.loc[sell_conditions, 'score_v4'] = df.loc[sell_conditions, 'score_v4'] * 1.25
            
        # تعزيز البيع عند وجود فرص متوسطة الثقة
        medium_confidence_opportunities = market_analysis['sell_opportunities']['medium_confidence_sells']
        if medium_confidence_opportunities > 0:
            sell_conditions = (
                (df['ema_9'] < df['ema_21']) & 
                (df['rsi'] > 55)  # تخفيض إضافي لعتبة RSI
            )
            df.loc[sell_conditions, 'score_v4'] = df.loc[sell_conditions, 'score_v4'] * 1.15
        
        # ✅ تحسين التعزيز بناء على تحليل الحجم
        volume_boost_conditions = (
            (df['volume'] > df['volume_avg'] * VOLUME_BOOST_FACTOR) &
            (df['ema_9'] < df['ema_21'])
        )
        df.loc[volume_boost_conditions, 'score_v4'] = df.loc[volume_boost_conditions, 'score_v4'] * 1.1
        
        # ✅ ثقة تكيفية بناء على ظروف السوق
        if ADAPTIVE_CONFIDENCE:
            bearish_market = market_analysis['market_phase'] == "BEARISH"
            if bearish_market:
                # زيادة وزن إشارات البيع في السوق الهابط
                sell_signals = (df['ema_9'] < df['ema_21'])
                df.loc[sell_signals, 'score_v4'] = df.loc[sell_signals, 'score_v4'] * 1.2
        
        df['score_v4'] = df['score_v4'].clip(0, 100)
        return df

    # =========================================================================
    # محسن إشارات البيع الذكي v4.2
    # =========================================================================
    
    def intelligent_sell_enhancement_v4_2(self, df: pd.DataFrame) -> pd.DataFrame:
        """تعزيز ذكي لإشارات البيع v4.2"""
        
        market_analysis = self.analyze_market_conditions_v4_2(df)
        
        # ✅ شروط البيع الذكية v4.2 - أكثر مرونة وذكاء
        intelligent_super_sell = (
            (df['score_v4'] >= 75) &  # خفض العتبة من 78 إلى 75
            (df['ema_9'] < df['ema_21']) &
            (df['ema_21'] < df['ema_50']) &
            (df['rsi'] > 62) &  # خفض من 68 إلى 62
            (df['macd_histogram'] < -0.002) &  # خفض من -0.004 إلى -0.002
            (df['volume'] > df['volume_avg'] * 1.1)  # خفض من 1.3 إلى 1.1
        )
        
        intelligent_high_sell = (
            (df['score_v4'] >= 70) &  # خفض العتبة من 72 إلى 70
            (df['ema_9'] < df['ema_21']) &
            (df['ema_21'] < df['ema_50']) &
            (df['rsi'] > 60) &  # خفض من 65 إلى 60
            (df['macd_histogram'] < -0.0015) &  # خفض من -0.003 إلى -0.0015
            (df['volume'] > df['volume_avg'] * 0.9)  # خفض من 1.1 إلى 0.9
        )
        
        intelligent_good_sell = (
            (df['score_v4'] >= 65) &  # خفض العتبة من 68 إلى 65
            (df['ema_9'] < df['ema_21']) &
            (df['rsi'] > 58) &  # خفض من 62 إلى 58
            (df['macd_histogram'] < -0.001) &  # خفض من -0.002 إلى -0.001
            (df['volume'] > df['volume_avg'] * 0.8)  # خفض من 0.9 إلى 0.8
        )
        
        # ✅ تطبيق التعزيز الذكي بناء على ظروف السوق
        market_boost = 1.0
        if market_analysis['market_phase'] == "BEARISH":
            market_boost = 1.3
        elif market_analysis['volatility_regime'] == "HIGH_VOLATILITY":
            market_boost = 1.2
            
        # تطبيق التعزيز
        df.loc[intelligent_super_sell, 'score_v4'] = df.loc[intelligent_super_sell, 'score_v4'] * 1.4 * market_boost
        df.loc[intelligent_high_sell, 'score_v4'] = df.loc[intelligent_high_sell, 'score_v4'] * 1.25 * market_boost
        df.loc[intelligent_good_sell, 'score_v4'] = df.loc[intelligent_good_sell, 'score_v4'] * 1.15 * market_boost
        
        # ✅ تحديث جودة الإشارة
        df['signal_quality'] = 'STANDARD'
        df.loc[intelligent_good_sell, 'signal_quality'] = 'GOOD'
        df.loc[intelligent_high_sell, 'signal_quality'] = 'HIGH'
        df.loc[intelligent_super_sell, 'signal_quality'] = 'SUPER'
        
        # ✅ تسجيل الإحصائيات الذكية
        super_count = len(df[intelligent_super_sell])
        high_count = len(df[intelligent_high_sell])
        good_count = len(df[intelligent_good_sell])
        
        logger.info(f"🧠 التعزيز الذكي v4.2 - فرص بيع: {market_analysis['sell_opportunities']['high_confidence_sells']} عالية, {market_analysis['sell_opportunities']['medium_confidence_sells']} متوسطة")
        logger.info(f"🎯 إشارات البيع الذكية - فائق: {super_count}, عالي: {high_count}, جيد: {good_count}")
        
        return df

    # =========================================================================
    # شروط البيع الذكية v4.2
    # =========================================================================
    
    def generate_intelligent_signals_v4_2(self, df: pd.DataFrame) -> pd.DataFrame:
        """إشارات ذكية v4.2 مع تحسين جذري لشروط البيع"""
        
        # التحليل الذكي للسوق
        market_analysis = self.analyze_market_conditions_v4_2(df)
        
        # الشروط الأساسية للشراء (تبقى قوية)
        buy_condition = (
            (df['score_v4'] >= CONFIDENCE_THRESHOLD) &
            (df['filter_pass_buy'] == True) &
            (df['rsi'] >= 35) & (df['rsi'] <= 65) &
            (df['macd_histogram'] > -0.002) &
            (df['close'] > df['ema_21']) &
            (df['volume'] > df['volume_avg'] * 0.7)  # تخفيف شرط الحجم
        )
        
        # ✅ الشروط الذكية للبيع v4.2 - مرنة ومتوازنة
        intelligent_super_sell = (
            (df['score_v4'] >= 75) &  # عتبة منخفضة
            (df['ema_9'] < df['ema_21']) &
            (df['ema_21'] < df['ema_50']) &
            (df['rsi'] > 60) &  # عتبة RSI معقولة
            (df['macd_histogram'] < -0.002) &
            (df['volume'] > df['volume_avg'] * 0.9)
        )
        
        intelligent_high_sell = (
            (df['score_v4'] >= 70) &
            (df['ema_9'] < df['ema_21']) &
            (df['ema_21'] < df['ema_50']) &
            (df['rsi'] > 58) &
            (df['macd_histogram'] < -0.0015) &
            (df['volume'] > df['volume_avg'] * 0.8)
        )
        
        intelligent_good_sell = (
            (df['score_v4'] >= 65) &
            (df['ema_9'] < df['ema_21']) &
            (df['rsi'] > 56) &
            (df['macd_histogram'] < -0.001) &
            (df['volume'] > df['volume_avg'] * 0.7)
        )
        
        # ✅ فلتر السياق الذكي: تعطيل البيع في ظروف غير مناسبة
        avoid_sell_conditions = (
            (market_analysis['market_phase'] == "BULLISH") &
            (market_analysis['trend_strength'] > 0.7) &
            (df['rsi'] < 50)
        )
        
        # ✅ تطبيق الإشارات الذكية
        df['signal_v4'] = 'none'
        df.loc[buy_condition, 'signal_v4'] = 'LONG'
        
        # تطبيق إشارات البيع مع الفلتر الذكي
        super_sell_mask = intelligent_super_sell & ~avoid_sell_conditions
        high_sell_mask = intelligent_high_sell & ~avoid_sell_conditions & (df['signal_v4'] == 'none')
        good_sell_mask = intelligent_good_sell & ~avoid_sell_conditions & (df['signal_v4'] == 'none')
        
        df.loc[super_sell_mask, 'signal_v4'] = 'SHORT'
        df.loc[high_sell_mask, 'signal_v4'] = 'SHORT'
        df.loc[good_sell_mask, 'signal_v4'] = 'SHORT'
        
        # ✅ تحديث مستوى الثقة
        df['confidence_level'] = df['score_v4'].apply(self.calculate_intelligent_confidence_v4_2)
        
        # ✅ تسجيل التحليل الذكي
        total_signals = len(df[df['signal_v4'] != 'none'])
        buy_signals = len(df[df['signal_v4'] == 'LONG'])
        sell_signals = len(df[df['signal_v4'] == 'SHORT'])
        
        logger.info(f"🧠 الإشارات الذكية v4.2 - إجمالي: {total_signals}, شراء: {buy_signals}, بيع: {sell_signals}")
        logger.info(f"📊 تحليل السوق - المرحلة: {market_analysis['market_phase']}, التقلبات: {market_analysis['volatility_regime']}")
        
        return df
    
    def calculate_intelligent_confidence_v4_2(self, score: float) -> str:
        """تحديد مستوى الثقة الذكي v4.2"""
        if score >= 80:
            return "ممتازة"
        elif score >= 70:
            return "جيدة جداً"
        elif score >= 60:
            return "جيدة"
        elif score >= 50:
            return "متوسطة"
        else:
            return "ضعيفة"

    def enhance_sell_signals_v4(self, df: pd.DataFrame) -> pd.DataFrame:
        """تعزيز إشارات البيع بشكل أكثر ذكاءً وتركيزاً على الجودة"""
    
        # ✅ تعزيز إشارات البيع فائقة الجودة فقط
        super_quality_sell_conditions = (
            (df['ema_9'] < df['ema_21']) &
            (df['ema_21'] < df['ema_50']) &
            (df['ema_50'] < df['ema_100']) &  # اتجاه هابط قوي بمتوسطات متعددة
            (df['rsi'] > 68) &  # زيادة من 65 إلى 68
            (df['macd_histogram'] < -0.004) &  # زيادة من -0.003 إلى -0.004
            (df['volume'] > df['volume_avg'] * 1.3)  # زيادة من 1.2 إلى 1.3
        )
    
        # ✅ تعزيز متوسط لإشارات البيع عالية الجودة
        high_quality_sell_conditions = (
            (df['ema_9'] < df['ema_21']) &
            (df['ema_21'] < df['ema_50']) &  # تأكيد الهبوط بمتوسطين
            (df['rsi'] > 65) &  # زيادة من 62 إلى 65
            (df['macd_histogram'] < -0.003) &  # زيادة من -0.002 إلى -0.003
            (df['volume'] > df['volume_avg'] * 1.1)  # زيادة من 1.0 إلى 1.1
        )
    
        # ✅ تعزيز خفيف للبيع الجيد
        good_sell_conditions = (
            (df['ema_9'] < df['ema_21']) &
            (df['rsi'] > 62) &  # زيادة من 60 إلى 62
            (df['macd_histogram'] < -0.002)  # زيادة من -0.001 إلى -0.002
        )
    
        # تطبيق التعزيز حسب الجودة (من الأعلى إلى الأدنى)
        df.loc[super_quality_sell_conditions, 'score_v4'] = df.loc[super_quality_sell_conditions, 'score_v4'] * 1.5  # تعزيز قوي
        df.loc[high_quality_sell_conditions, 'score_v4'] = df.loc[high_quality_sell_conditions, 'score_v4'] * 1.3   # تعزيز متوسط
        df.loc[good_sell_conditions, 'score_v4'] = df.loc[good_sell_conditions, 'score_v4'] * 1.15                   # تعزيز خفيف
    
        # ✅ تحديد قوة الإشارة بناء على مستوى التعزيز
        df['signal_strength'] = df['score_v4'] / 100.0
        
        # ✅ تحديد جودة الإشارة
        df['signal_quality'] = 'STANDARD'
        df.loc[good_sell_conditions, 'signal_quality'] = 'GOOD'
        df.loc[high_quality_sell_conditions, 'signal_quality'] = 'HIGH'
        df.loc[super_quality_sell_conditions, 'signal_quality'] = 'SUPER'
    
        # ✅ تسجيل إحصائيات التعزيز
        super_count = len(df[super_quality_sell_conditions])
        high_count = len(df[high_quality_sell_conditions])
        good_count = len(df[good_sell_conditions])
    
        logger.info(f"🎯 تعزيز إشارات البيع v4 - فائق: {super_count}, عالي: {high_count}, جيد: {good_count}")
    
        return df
    
    def add_smart_filters_v4(self, df: pd.DataFrame) -> pd.DataFrame:
        """إضافة عوامل تصفية ذكية v4 مع إنشاء atr_percent"""
        
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
        
        # ✅ فلتر جودة للبيع - لتحسين النجاح
        df['high_quality_sell'] = (
            (df['ema_9'] < df['ema_21']) &
            (df['ema_21'] < df['ema_50']) &
            (df['rsi'] > 65) &  # زيادة من 60 إلى 65
            (df['macd_histogram'] < -0.003) &  # زيادة من -0.001 إلى -0.003
            (df['volume'] > df['volume_avg'] * 1.1)  # زيادة من 0.9 إلى 1.1
        )
        
        df['good_quality_sell'] = (
            (df['ema_9'] < df['ema_21']) &
            (df['rsi'] > 62) &  # زيادة من 58 إلى 62
            (df['macd_histogram'] < -0.002) &  # زيادة من -0.0005 إلى -0.002
            (df['volume'] > df['volume_avg'] * 0.9)  # زيادة من 0.7 إلى 0.9
        )
        
        # ✅ فلاتر مرنة للبيع
        df['filter_pass_sell_enhanced'] = (
            (
                df['strong_downtrend'] |  # اتجاه هابط قوي
                ((df['ema_9'] < df['ema_21']) & (df['ema_21'] < df['ema_50']))  # اتجاه هابط
            ) &
            (df['close'] < df['ema_21']) &  # تحت المتوسط المتوسط (بدلاً من 50)
            (df['rsi'] > 50)  # RSI في النصف العلوي (أكثر مرونة)
        )
        
        return df
    
    def dynamic_stop_take_profit_v4(self, df: pd.DataFrame) -> pd.DataFrame:
        """وقف وجني ديناميكي محسن جداً للبيع v4"""
    
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
    
        # ✅ إعدادات محسنة جداً للبيع v4 - أكثر عدوانية وجاذبية
        df['dynamic_sl_sell'] = np.where(
            df['volatility_ratio'] > 1.5,
            0.7,  # وقف صغير في التقلبات العالية
            np.where(
                df['volatility_ratio'] < 0.7,
                0.4,  # وقف صغير جداً في التقلبات المنخفضة
                0.5   # وقف صغير عادي
            )
        )
    
        df['dynamic_tp_sell'] = np.where(
            df['volatility_ratio'] > 1.5,
            4.2,  # جني كبير في التقلبات العالية
            np.where(
                df['volatility_ratio'] < 0.7,
                3.2,  # جني جيد في التقلبات المنخفضة
                3.8   # جني كبير عادي
            )
        )
    
        # ✅ إعدادات خاصة للبيع فائق الجودة
        df['super_quality_sell_sl'] = df['dynamic_sl_sell'] * 0.6  # وقف أصغر
        df['super_quality_sell_tp'] = df['dynamic_tp_sell'] * 1.3  # جني أكبر
        
        # ✅ إعدادات خاصة للبيع عالي الجودة
        df['high_quality_sell_sl'] = df['dynamic_sl_sell'] * 0.7  # وقف أصغر
        df['high_quality_sell_tp'] = df['dynamic_tp_sell'] * 1.2  # جني أكبر
    
        logger.info(f"🎯 إعدادات البيع المحسنة v4 - وقف: {df['dynamic_sl_sell'].mean():.2f}%, جني: {df['dynamic_tp_sell'].mean():.2f}%")
    
        return df
    
    def risk_adjusted_scoring_v4(self, df: pd.DataFrame) -> pd.DataFrame:
        """نظام تقييم معدل حسب المخاطرة v4 مع معالجة آمنة"""
        
        # التحقق من وجود الأعمدة المطلوبة
        required_columns = ['atr_percent', 'rsi_volatility', 'score_v4']
        missing_columns = [col for col in required_columns if col not in df.columns]
        
        if missing_columns:
            logger.warning(f"⚠️ أعمدة مفقودة في risk_adjusted_scoring_v4: {missing_columns}")
            return df
        
        # مكافأة الصفقات منخفضة المخاطرة
        low_risk_mask = (df['atr_percent'] < 0.015) & (df['rsi_volatility'] < 10)
        df.loc[low_risk_mask, 'score_v4'] = df.loc[low_risk_mask, 'score_v4'] * 1.2  # زيادة من 1.15 إلى 1.2
        
        # معاقبة الصفقات عالية المخاطرة
        high_risk_mask = (df['atr_percent'] > 0.025) | (df['rsi_volatility'] > 15)
        df.loc[high_risk_mask, 'score_v4'] = df.loc[high_risk_mask, 'score_v4'] * 0.8  # زيادة من 0.85 إلى 0.8
        
        return df
    
    def generate_enhanced_signals_v4(self, df: pd.DataFrame) -> pd.DataFrame:
        """إشارات محسنة v4 مع إعادة تصميم جذرية لشروط البيع"""
    
        # التحقق من وجود الأعمدة المطلوبة
        required_columns = ['score_v4', 'filter_pass_buy', 'rsi', 'macd_histogram', 'close', 'ema_21', 'volume', 'volume_avg', 'ema_9', 'ema_50', 'ma_order', 'signal_quality']
        missing_columns = [col for col in required_columns if col not in df.columns]
    
        if missing_columns:
            logger.warning(f"⚠️ أعمدة مفقودة في generate_enhanced_signals_v4: {missing_columns}")
            df['signal_v4'] = 'none'
            df['confidence_level'] = 'ضعيفة'
            df['current_volatility'] = 0.0
            return df
    
        # الشروط الأساسية المحسنة للشراء (تبقى كما هي - تعمل بشكل ممتاز)
        buy_condition_v4 = (
            (df['score_v4'] >= CONFIDENCE_THRESHOLD) &
            (df['filter_pass_buy'] == True) &
            (df['rsi'] >= 35) & (df['rsi'] <= 65) &
            (df['macd_histogram'] > -0.003) &
            (df['close'] > df['ema_21']) &
            (df['volume'] > df['volume_avg'] * 0.8)
        )
    
        # ✅ إعادة تصميم جذرية لشروط البيع v4 - التركيز على الجودة
        super_quality_sell = (
            (df['score_v4'] >= SUPER_QUALITY_SELL_THRESHOLD) &  # 80
            (df['ema_9'] < df['ema_21']) &
            (df['ema_21'] < df['ema_50']) &
            (df['ema_50'] < df['ema_100']) &  # اتجاه هابط قوي بمتوسطات متعددة
            (df['rsi'] > 68) &  # زيادة من 65 إلى 68
            (df['macd_histogram'] < -0.004) &  # زيادة من -0.003 إلى -0.004
            (df['volume'] > df['volume_avg'] * 1.3)  # زيادة من 1.2 إلى 1.3
        )
        
        high_quality_sell = (
            (df['score_v4'] >= HIGH_QUALITY_SELL_THRESHOLD) &  # 75
            (df['ema_9'] < df['ema_21']) &
            (df['ema_21'] < df['ema_50']) &  # اتجاه هابط بمتوسطين
            (df['rsi'] > 65) &  # زيادة من 60 إلى 65
            (df['macd_histogram'] < -0.003) &  # زيادة من -0.002 إلى -0.003
            (df['volume'] > df['volume_avg'] * 1.1)  # زيادة من 0.8 إلى 1.1
        )
        
        good_quality_sell = (
            (df['score_v4'] >= SELL_CONFIDENCE_THRESHOLD) &  # 72
            (df['ema_9'] < df['ema_21']) &
            (df['rsi'] > 62) &  # زيادة من 60 إلى 62
            (df['macd_histogram'] < -0.002) &  # زيادة من -0.001 إلى -0.002
            (df['volume'] > df['volume_avg'] * 0.9)  # زيادة من 0.8 إلى 0.9
        )
    
        # ✅ فلتر إضافي للبيع: منع الإشارات في الأسواق الجانبية القوية
        sideways_market = (
            (df['ema_50'] - df['ema_50'].shift(5)).abs() / df['ema_50'] < 0.01  # تقلبات صغيرة
        )
        
        # تطبيق الإشارات مع الأولوية القصوى للجودة الفائقة
        df['signal_v4'] = 'none'
        df.loc[buy_condition_v4, 'signal_v4'] = 'LONG'
        df.loc[super_quality_sell & ~sideways_market, 'signal_v4'] = 'SHORT'
        df.loc[high_quality_sell & ~sideways_market & (df['signal_v4'] == 'none'), 'signal_v4'] = 'SHORT'
        df.loc[good_quality_sell & ~sideways_market & (df['signal_v4'] == 'none'), 'signal_v4'] = 'SHORT'
    
        # إضافة مستوى الثقة النهائي
        df['confidence_level'] = df['score_v4'].apply(self.calculate_confidence_level_v4)
    
        # إضافة التقلبات للتحليل
        if 'atr_percent' in df.columns:
            df['current_volatility'] = df['atr_percent'].fillna(df['atr_percent'].mean())
        else:
            df['current_volatility'] = 0.02
    
        # ✅ تسجيل إحصائيات مفصلة v4
        total_signals = len(df[df['signal_v4'] != 'none'])
        buy_signals = len(df[df['signal_v4'] == 'LONG'])
        sell_signals = len(df[df['signal_v4'] == 'SHORT'])
        super_sell_signals = len(df[super_quality_sell & (df['signal_v4'] == 'SHORT')])
        high_sell_signals = len(df[high_quality_sell & (df['signal_v4'] == 'SHORT')])
        good_sell_signals = len(df[good_quality_sell & (df['signal_v4'] == 'SHORT')])
    
        logger.info(f"📊 إحصائيات الإشارات v4 - شراء: {buy_signals}, بيع فائق: {super_sell_signals}, بيع عالي: {high_sell_signals}, بيع جيد: {good_sell_signals}")
    
        # ✅ تحليل جودة إشارات البيع
        if sell_signals > 0:
            sell_confidence_avg = df[df['signal_v4'] == 'SHORT']['score_v4'].mean()
            sell_rsi_avg = df[df['signal_v4'] == 'SHORT']['rsi'].mean()
            logger.info(f"🔽 تحليل إشارات البيع v4 - متوسط الثقة: {sell_confidence_avg:.1f}%, متوسط RSI: {sell_rsi_avg:.1f}")
        
            # تحليل البيع فائق الجودة
            if super_sell_signals > 0:
                super_sell_confidence = df[super_quality_sell & (df['signal_v4'] == 'SHORT')]['score_v4'].mean()
                logger.info(f"🎯 البيع فائق الجودة v4 - متوسط الثقة: {super_sell_confidence:.1f}%")
    
        if buy_signals > 0:
            buy_confidence_avg = df[df['signal_v4'] == 'LONG']['score_v4'].mean()
            buy_rsi_avg = df[df['signal_v4'] == 'LONG']['rsi'].mean()
            logger.info(f"🔼 تحليل إشارات الشراء v4 - متوسط الثقة: {buy_confidence_avg:.1f}%, متوسط RSI: {buy_rsi_avg:.1f}")
    
        return df
    
    def calculate_confidence_level_v4(self, score: float) -> str:
        """تحديد مستوى الثقة بدقة v4"""
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
    
    def enhanced_analysis_v4(self, df: pd.DataFrame) -> pd.DataFrame:
        """التحليل المحسن v4 - الدالة الرئيسية مع إصلاح الترتيب"""
        
        # 1. حساب المؤشرات الأساسية
        df['rsi'] = self.calculate_rsi(df['close'])
        macd_line, signal_line, histogram = self.calculate_macd(df['close'])
        df['macd_line'] = macd_line
        df['macd_signal'] = signal_line
        df['macd_histogram'] = histogram
        
        # 2. تحليل الاتجاه
        df = self.analyze_trend(df)
        
        # 3. إضافة عوامل التصفية أولاً (لإنشاء atr_percent)
        df = self.add_smart_filters_v4(df)
        
        # 4. نظام التقييم المحسن
        df = self.enhanced_scoring_system_v4(df)
        
        # 5. تعزيز إشارات البيع
        df = self.enhance_sell_signals_v4(df)
        
        # 6. وقف وجني ديناميكي (يحتاج atr_percent)
        df = self.dynamic_stop_take_profit_v4(df)
        
        # 7. تقييم معدل حسب المخاطرة (يحتاج atr_percent)
        df = self.risk_adjusted_scoring_v4(df)
        
        # 8. إشارات محسنة
        df = self.generate_enhanced_signals_v4(df)
        
        # حفظ نتائج التحليل
        self.analysis_results = df.to_dict('records')
        
        return df

    # =========================================================================
    # نظام التداول الذكي v4.2
    # =========================================================================
    
    def calculate_position_size(self, price: float) -> float:
        """حساب حجم المركز بناء على الرافعة وحجم الصفقة"""
        return (TRADE_SIZE_USDT * LEVERAGE) / price
    
    def open_position(self, symbol: str, direction: str, price: float, 
                 confidence: float, confidence_level: str, 
                 volatility: float, timestamp: datetime, 
                 dynamic_sl: float, dynamic_tp: float,
                 signal_strength: float, signal_quality: str = "STANDARD") -> Optional[Trade]:
        """فتح مركز جديد مع إعدادات خاصة للبيع v4"""
    
        if symbol in self.positions:
            logger.warning(f"يوجد مركز مفتوح بالفعل لـ {symbol}")
            return None
    
        # حساب حجم المركز
        quantity = self.calculate_position_size(price)
    
        # ✅ إعدادات خاصة لجودة البيع
        is_super_quality_sell = (direction == "SHORT" and signal_quality == "SUPER")
        is_high_quality_sell = (direction == "SHORT" and signal_quality == "HIGH")
        is_good_quality_sell = (direction == "SHORT" and signal_quality == "GOOD")
    
        if is_super_quality_sell:
            # أفضل إعدادات للبيع فائق الجودة
            dynamic_sl = dynamic_sl * 0.6  # تقليل الوقف بنسبة 40%
            dynamic_tp = dynamic_tp * 1.3  # زيادة الجني بنسبة 30%
            quality = "SUPER"
            logger.info(f"🚀 فتح مركز بيع فائق الجودة لـ {symbol} - وقف: {dynamic_sl:.2f}%, جني: {dynamic_tp:.2f}%")
            
        elif is_high_quality_sell:
            # إعدادات جيدة للبيع عالي الجودة
            dynamic_sl = dynamic_sl * 0.7  # تقليل الوقف بنسبة 30%
            dynamic_tp = dynamic_tp * 1.2  # زيادة الجني بنسبة 20%
            quality = "HIGH"
            logger.info(f"🎯 فتح مركز بيع عالي الجودة لـ {symbol} - وقف: {dynamic_sl:.2f}%, جني: {dynamic_tp:.2f}%")
            
        elif is_good_quality_sell:
            # إعدادات معتدلة للبيع الجيد
            dynamic_sl = dynamic_sl * 0.8  # تقليل الوقف بنسبة 20%
            dynamic_tp = dynamic_tp * 1.1  # زيادة الجني بنسبة 10%
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
    
        # تسجيل شروط الدخول للتحليل
        entry_conditions = {
            'signal_quality': signal_quality,
            'signal_strength': signal_strength,
            'volatility': volatility,
            'dynamic_sl': dynamic_sl,
            'dynamic_tp': dynamic_tp
        }
    
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
            quality=quality,
            entry_conditions=entry_conditions
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
        
        # تحليل سبب الخسارة إذا كانت الصفقة خاسرة
        loss_reason = ""
        if pnl < 0:
            loss_reason = self.analyze_loss_reason(trade, price, reason)
        
        # تحديث بيانات الصفقة
        trade.exit_price = price
        trade.exit_time = timestamp
        trade.pnl = pnl
        trade.pnl_percent = pnl_percent
        trade.status = reason
        trade.loss_reason = loss_reason
        
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
            'status': trade.status,
            'loss_reason': trade.loss_reason
        }
        
        self.trade_history.append(trade_dict)
        
        status_emoji = "🟢" if pnl > 0 else "🔴"
        quality_emoji = "🚀" if trade.quality == "SUPER" else "🎯" if trade.quality == "HIGH" else "📉" if trade.quality == "GOOD" else ""
        
        if pnl < 0:
            logger.info(f"📊 إغلاق مركز {trade.direction} {quality_emoji} لـ {symbol} {status_emoji}"
                       f" الخسارة: {pnl:.2f} USD ({pnl_percent:.2f}%) - {reason}")
            logger.info(f"🔍 سبب الخسارة: {loss_reason}")
        else:
            logger.info(f"📊 إغلاق مركز {trade.direction} {quality_emoji} لـ {symbol} {status_emoji}"
                       f" الربح: {pnl:.2f} USD ({pnl_percent:.2f}%) - {reason}")
        
        return trade

    def analyze_loss_reason(self, trade: Trade, exit_price: float, exit_reason: str) -> str:
        """تحليل سبب الخسارة للصفقات الخاسرة"""
        
        price_change_pct = ((exit_price - trade.entry_price) / trade.entry_price) * 100
        if trade.direction == "SHORT":
            price_change_pct = -price_change_pct
        
        # تحليل بناء على سبب الخروج
        if exit_reason == "STOP_LOSS":
            if trade.direction == "LONG":
                if price_change_pct < -2:
                    return "اتجاه هابط قوي تجاوز وقف الخسارة"
                else:
                    return "تقلبات سريعة أثرت على الوقف"
            else:  # SHORT
                if price_change_pct > 2:
                    return "ارتفاع مفاجئ في السعر تجاوز وقف الخسارة"
                else:
                    return "تقلبات عكسية أثرت على الوقف"
        
        elif exit_reason == "END_OF_DATA":
            return "إغلاق قسري في نهاية البيانات - لم يتحقق الهدف"
        
        # تحليل بناء على ظروف الدخول
        if trade.confidence < 60:
            return "ثقة منخفضة عند الدخول - إشارة ضعيفة"
        
        if trade.volatility > 0.03:
            return "تقلبات عالية أثرت على الصفقة"
        
        if trade.signal_strength < 0.6:
            return "قوة إشارة ضعيفة - عدم تأكيد كافي"
        
        return "خسارة طبيعية - تقلبات السوق"

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

    def execute_intelligent_trading_v4_2(self, df: pd.DataFrame):
        """تنفيذ التداول الذكي v4.2"""
        
        logger.info("🧠 بدء التداول الذكي v4.2...")
        
        for i, row in df.iterrows():
            if i < 50:
                continue
                
            current_price = row['close']
            signal = row['signal_v4']
            confidence = row['score_v4']
            confidence_level = row['confidence_level']
            volatility = row['current_volatility']
            timestamp = row['timestamp']
            signal_strength = row['signal_strength']
            signal_quality = row.get('signal_quality', 'STANDARD')
            
            # ✅ الإعدادات الذكية للوقف والجني
            if signal == 'LONG':
                dynamic_sl = row['dynamic_sl_buy']
                dynamic_tp = row['dynamic_tp_buy']
            else:
                # إعدادات ذكية للبيع بناء على الجودة
                if signal_quality == 'SUPER':
                    dynamic_sl = row.get('super_quality_sell_sl', row['dynamic_sl_sell']) * 0.9  # تخفيض إضافي
                    dynamic_tp = row.get('super_quality_sell_tp', row['dynamic_tp_sell']) * 1.1  # زيادة إضافية
                elif signal_quality == 'HIGH':
                    dynamic_sl = row.get('high_quality_sell_sl', row['dynamic_sl_sell']) * 0.85
                    dynamic_tp = row.get('high_quality_sell_tp', row['dynamic_tp_sell']) * 1.15
                else:
                    dynamic_sl = row['dynamic_sl_sell'] * 0.8
                    dynamic_tp = row['dynamic_tp_sell'] * 1.2
            
            # فحص شروط الخروج الذكية
            if SYMBOL in self.positions:
                self.check_stop_conditions(SYMBOL, current_price, timestamp)
            
            # ✅ فتح المراكز الذكية
            if (SYMBOL not in self.positions and signal != 'none'):
                # عتبات ثقة ذكية ومتوازنة
                min_confidence = CONFIDENCE_THRESHOLD if signal == 'LONG' else 65  # تخفيض عتبة البيع
                
                if confidence >= min_confidence:
                    self.open_position(
                        SYMBOL, signal, current_price, confidence, confidence_level,
                        volatility, timestamp, dynamic_sl, dynamic_tp, signal_strength, signal_quality
                    )

    # =========================================================================
    # تحليل الخسائر المتقدم
    # =========================================================================
    
    def analyze_losing_trades(self) -> LossAnalysis:
        """تحليل متقدم للصفقات الخاسرة"""
        
        if not self.trade_history:
            return LossAnalysis(
                total_losing_trades=0,
                loss_reasons={},
                avg_loss_per_trade=0,
                common_patterns=[],
                improvement_suggestions=[]
            )
        
        trades_df = pd.DataFrame(self.trade_history)
        losing_trades = trades_df[trades_df['pnl'] < 0]
        
        if losing_trades.empty:
            return LossAnalysis(
                total_losing_trades=0,
                loss_reasons={},
                avg_loss_per_trade=0,
                common_patterns=[],
                improvement_suggestions=[]
            )
        
        # تحليل أسباب الخسارة
        loss_reasons = {}
        for reason in losing_trades['loss_reason']:
            if reason:
                loss_reasons[reason] = loss_reasons.get(reason, 0) + 1
        
        # تحليل الأنماط الشائعة
        common_patterns = []
        
        # نمط 1: الخسائر الكبيرة
        big_losses = losing_trades[losing_trades['pnl_percent'] < -3]
        if len(big_losses) > 0:
            common_patterns.append(f"{len(big_losses)} صفقة بخسارة كبيرة (>3%)")
        
        # نمط 2: الخسائر مع ثقة عالية
        high_confidence_losses = losing_trades[losing_trades['confidence'] > 75]
        if len(high_confidence_losses) > 0:
            common_patterns.append(f"{len(high_confidence_losses)} صفقة بخسارة رغم ثقة عالية")
        
        # نمط 3: الخسائر بسبب التقلبات
        high_vol_losses = losing_trades[losing_trades['volatility'] > 0.025]
        if len(high_vol_losses) > 0:
            common_patterns.append(f"{len(high_vol_losses)} صفقة خسارة بسبب تقلبات عالية")
        
        # اقتراحات التحسين
        improvement_suggestions = []
        
        if len(big_losses) > len(losing_trades) * 0.3:  # أكثر من 30% خسائر كبيرة
            improvement_suggestions.append("زيادة وقف الخسارة للخسائر الكبيرة")
        
        if len(high_confidence_losses) > len(losing_trades) * 0.4:  # أكثر من 40% خسائر بثقة عالية
            improvement_suggestions.append("مراجعة نظام التقييم للإشارات عالية الثقة")
        
        if len(high_vol_losses) > len(losing_trades) * 0.5:  # أكثر من 50% خسائر بسبب التقلبات
            improvement_suggestions.append("تجنب التداول في فترات التقلبات العالية")
        
        # اقتراحات عامة
        avg_loss = losing_trades['pnl'].mean()
        if avg_loss < -15:  # متوسط خسارة عالي
            improvement_suggestions.append("تقليل حجم الصفقة أو تحسين إدارة المخاطر")
        
        return LossAnalysis(
            total_losing_trades=len(losing_trades),
            loss_reasons=loss_reasons,
            avg_loss_per_trade=losing_trades['pnl'].mean(),
            common_patterns=common_patterns,
            improvement_suggestions=improvement_suggestions
        )

    # =========================================================================
    # الباك-تستينغ الذكي v4.2
    # =========================================================================
    
    def run_intelligent_backtest_v4_2(self, df: pd.DataFrame) -> BacktestResult:
        """تشغيل الباك-تستينغ الذكي v4.2"""
        
        logger.info("🔍 بدء الباك-تستينغ الذكي v4.2...")
        
        # إعادة تعيين البيانات
        self.trades = []
        self.positions = {}
        self.trade_history = []
        self.sell_performance_history = []
        self.current_balance = INITIAL_BALANCE
        
        # التحليل الذكي v4.2
        df_with_signals = self.enhanced_analysis_v4(df)
        df_with_signals = self.intelligent_scoring_system_v4_2(df_with_signals)
        df_with_signals = self.intelligent_sell_enhancement_v4_2(df_with_signals)
        df_with_signals = self.generate_intelligent_signals_v4_2(df_with_signals)
        
        # تنفيذ التداول الذكي v4.2
        self.execute_intelligent_trading_v4_2(df_with_signals)
        
        # إغلاق المراكز المفتوحة
        if SYMBOL in self.positions:
            last_price = df_with_signals.iloc[-1]['close']
            last_timestamp = df_with_signals.iloc[-1]['timestamp']
            self.close_position(SYMBOL, last_price, last_timestamp, "END_OF_DATA")
        
        # حساب النتائج الذكية
        return self.calculate_intelligent_backtest_results_v4_2(df)

    def calculate_intelligent_backtest_results_v4_2(self, df: pd.DataFrame) -> BacktestResult:
        """حساب نتائج الباك-تستينغ الذكية v4.2"""
        
        if not self.trade_history:
            total_days = (df['timestamp'].max() - df['timestamp'].min()).days
            return BacktestResult(
                total_trades=0, winning_trades=0, losing_trades=0,
                win_rate=0, total_pnl=0, final_balance=self.current_balance,
                max_drawdown=0, sharpe_ratio=0, profit_factor=0,
                avg_trade=0, best_trade=0, worst_trade=0, total_fees=0,
                total_days=max(1, total_days), avg_daily_return=0,
                avg_confidence=0, confidence_analysis={},
                buy_performance={}, sell_performance={}, quality_analysis={},
                loss_analysis=LossAnalysis(0, {}, 0, [], [])
            )
        
        trades_df = pd.DataFrame(self.trade_history)
        
        # المقاييس الأساسية
        total_trades = len(trades_df)
        winning_trades = len(trades_df[trades_df['pnl'] > 0])
        losing_trades = len(trades_df[trades_df['pnl'] < 0])
        win_rate = (winning_trades / total_trades) * 100 if total_trades > 0 else 0
        
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
        for level in ['ممتازة', 'جيدة جداً', 'جيدة', 'متوسطة', 'ضعيفة']:
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
        
        # ✅ تحليل الخسائر المتقدم
        loss_analysis = self.analyze_losing_trades()
        
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
            quality_analysis=quality_analysis,
            loss_analysis=loss_analysis
        )

    # =========================================================================
    # التقارير الذكية v4.2
    # =========================================================================
    
    async def send_intelligent_telegram_report_v4_2(self, backtest_result: BacktestResult, df: pd.DataFrame):
        """إرسال تقرير ذكي v4.2 إلى التلغرام"""
        
        if not self.telegram_notifier:
            return
        
        try:
            # 1. التقرير النصي الذكي
            report_text = self._generate_intelligent_report_text_v4_2(backtest_result)
            await self.telegram_notifier.send_message(report_text)
            
            # 2. الرسوم البيانية الذكية
            chart_buffer = self._create_intelligent_performance_chart_v4_2(df, backtest_result)
            if chart_buffer:
                chart_caption = f"🧠 تحليل الأداء الذكي v4.2 - {SYMBOL} ({TIMEFRAME})"
                await self.telegram_notifier.send_photo(chart_buffer, chart_caption)
            
            # 3. تحليل السوق الذكي
            market_analysis = self._generate_market_analysis_v4_2(df)
            await self.telegram_notifier.send_message(market_analysis)

            # 4. تحليل الخسائر
            loss_analysis = self._generate_loss_analysis_v4_2(backtest_result.loss_analysis)
            await self.telegram_notifier.send_message(loss_analysis)
                
        except Exception as e:
            logger.error(f"❌ خطأ في إرسال التقرير الذكي: {e}")

    def _generate_intelligent_report_text_v4_2(self, backtest_result: BacktestResult) -> str:
        """إنشاء نص التقرير الذكي v4.2"""
        
        report_time = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        
        message = f"🧠 *تقرير الاستراتيجية الذكية v4.2 - تحسينات ذكية للبيع*\n"
        message += "══════════════════════════════════════\n\n"
        
        message += f"⚙️ *الإعدادات الذكية v4.2:*\n"
        message += f"• العملة: `{SYMBOL}`\n"
        message += f"• الإطار: `{TIMEFRAME}`\n"
        message += f"• الرافعة: `{LEVERAGE}x`\n"
        message += f"• حجم الصفقة: `${TRADE_SIZE_USDT}`\n"
        message += f"• عتبة ثقة الشراء: `{CONFIDENCE_THRESHOLD}%`\n"
        message += f"• عتبة ثقة البيع: `{65}%` 📉\n"
        message += f"• عتبة البيع فائق الجودة: `{75}%` 📉\n"
        message += f"• عتبة البيع عالي الجودة: `{70}%` 📉\n"
        message += f"• تعزيز الحجم: `{VOLUME_BOOST_FACTOR}x`\n\n"
        
        message += f"📊 *النتائج الذكية v4.2:*\n"
        message += f"• إجمالي الصفقات: `{backtest_result.total_trades}`\n"
        message += f"• الصفقات الرابحة: `{backtest_result.winning_trades}` 🟢\n"
        message += f"• الصفقات الخاسرة: `{backtest_result.losing_trades}` 🔴\n"
        message += f"• نسبة الربح: `{backtest_result.win_rate:.1f}%`\n"
        message += f"• إجمالي الربح: `${backtest_result.total_pnl:,.2f}`\n"
        message += f"• الرصيد النهائي: `${backtest_result.final_balance:,.2f}`\n"
        message += f"• العائد الإجمالي: `{((backtest_result.final_balance - INITIAL_BALANCE) / INITIAL_BALANCE * 100):.1f}%`\n"
        message += f"• متوسط الثقة: `{backtest_result.avg_confidence:.1f}%`\n\n"
        
        # تحليل البيع والشراء
        buy_perf = backtest_result.buy_performance
        sell_perf = backtest_result.sell_performance
        
        message += f"🔄 *تحليل البيع والشراء الذكي:*\n"
        message += f"• صفقات الشراء: `{buy_perf['total_trades']}` (نجاح: `{buy_perf['win_rate']:.1f}%`)\n"
        message += f"• صفقات البيع: `{sell_perf['total_trades']}` (نجاح: `{sell_perf['win_rate']:.1f}%`)\n"
        message += f"• فرص البيع المستغلة: `{len([t for t in self.trade_history if t['direction'] == 'SHORT'])}`\n\n"
        
        message += f"🎯 *التوصيات الذكية v4.2:*\n"
        
        if sell_perf['total_trades'] == 0:
            message += f"• ✅ تم تفعيل شروط البيع الذكية\n"
            message += f"• 📈 من المتوقع زيادة صفقات البيع\n"
            message += f"• 🧠 النظام يتكيف مع ظروف السوق\n"
        else:
            message += f"• ✅ تم تحقيق توازن بين البيع والشراء\n"
            message += f"• 📊 تحسين مستمر بناء على النتائج\n"
            message += f"• 🎯 الحفاظ على الإعدادات الذكية\n"
        
        message += f"\n🕒 *وقت التقرير:* `{report_time}`\n"
        message += "══════════════════════════════════════\n"
        message += "🧠 *نظام ذكي + تحليل السوق + عتبات مرنة*"
        
        return message

    def _generate_market_analysis_v4_2(self, df: pd.DataFrame) -> str:
        """إنشاء تحليل السوق الذكي v4.2"""
        
        market_analysis = self.analyze_market_conditions_v4_2(df)
        
        message = "📊 *تحليل السوق الذكي v4.2:*\n"
        message += "────────────────────\n"
        
        message += f"• مرحلة السوق: `{market_analysis['market_phase']}`\n"
        message += f"• قوة الاتجاه: `{market_analysis['trend_strength']:.2f}`\n"
        message += f"• نظام التقلبات: `{market_analysis['volatility_regime']}`\n"
        message += f"• اتجاه الحجم: `{market_analysis['volume_profile']['trend']}`\n"
        message += f"• قرب المقاومة: `{'نعم' if market_analysis['support_resistance']['near_resistance'] else 'لا'}`\n\n"
        
        message += f"🎯 *فرص البيع المحددة:*\n"
        opportunities = market_analysis['sell_opportunities']
        message += f"• عالية الثقة: `{opportunities['high_confidence_sells']}`\n"
        message += f"• متوسطة الثقة: `{opportunities['medium_confidence_sells']}`\n"
        
        if opportunities['conditions_met']:
            message += f"• الشروط المتاحة: `{', '.join(opportunities['conditions_met'])}`\n"
        
        return message

    def _generate_loss_analysis_v4_2(self, loss_analysis: LossAnalysis) -> str:
        """إنشاء تحليل الخسائر الذكي v4.2"""
        
        message = "🔍 *تحليل الخسائر المتقدم v4.2:*\n"
        message += "────────────────────\n"
        
        message += f"• إجمالي الصفقات الخاسرة: `{loss_analysis.total_losing_trades}`\n"
        message += f"• متوسط الخسارة لكل صفقة: `${loss_analysis.avg_loss_per_trade:.2f}`\n\n"
        
        if loss_analysis.loss_reasons:
            message += f"📉 *أسباب الخسارة:*\n"
            for reason, count in loss_analysis.loss_reasons.items():
                percentage = (count / loss_analysis.total_losing_trades) * 100
                message += f"• `{reason}`: `{count}` مرات (`{percentage:.1f}%`)\n"
            message += "\n"
        
        if loss_analysis.common_patterns:
            message += f"🎯 *أنماط الخسارة الشائعة:*\n"
            for pattern in loss_analysis.common_patterns:
                message += f"• {pattern}\n"
            message += "\n"
        
        if loss_analysis.improvement_suggestions:
            message += f"💡 *اقتراحات التحسين:*\n"
            for suggestion in loss_analysis.improvement_suggestions:
                message += f"• {suggestion}\n"
        
        return message

    def _create_intelligent_performance_chart_v4_2(self, df: pd.DataFrame, backtest_result: BacktestResult) -> BytesIO:
        """إنشاء رسم بياني ذكي v4.2 للأداء"""
        try:
            fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
            fig.suptitle(f'تحليل الاستراتيجية الذكية v4.2 - {SYMBOL}', 
                        fontsize=16, fontname='DejaVu Sans', fontweight='bold')
            
            # 1. السعر والإشارات
            ax1.plot(df['timestamp'], df['close'], label='السعر', linewidth=1.5, color='blue', alpha=0.8)
            ax1.set_title('حركة السعر وإشارات التداول الذكية v4.2', fontname='DejaVu Sans', fontsize=12)
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
                ax2.set_title('توزيع أرباح البيع vs الشراء الذكي v4.2', fontname='DejaVu Sans', fontsize=12)
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
                
                ax3.set_title('تطور الرصيد الذكي v4.2', fontname='DejaVu Sans', fontsize=12)
                ax3.set_xlabel('عدد الصفقات', fontname='DejaVu Sans')
                ax3.set_ylabel('الرصيد (USD)', fontname='DejaVu Sans')
                ax3.legend(prop={'family': 'DejaVu Sans'})
                ax3.grid(True, alpha=0.3)
            
            # 4. تحليل الخسائر
            loss_analysis = backtest_result.loss_analysis
            if loss_analysis.loss_reasons:
                reasons = list(loss_analysis.loss_reasons.keys())
                counts = list(loss_analysis.loss_reasons.values())
                
                ax4.barh(reasons, counts, color='red', alpha=0.7)
                ax4.set_title('تحليل أسباب الخسائر v4.2', fontname='DejaVu Sans', fontsize=12)
                ax4.set_xlabel('عدد المرات', fontname='DejaVu Sans')
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
# الوظيفة الرئيسية المحدثة
# =============================================================================

async def main():
    """الوظيفة الرئيسية مع الاستراتيجية الذكية v4.2"""
    
    logger.info("🧠 بدء تشغيل الاستراتيجية الذكية v4.2 مع تحسينات ذكية للبيع وتحليل الخسائر")
    
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
    
    # تشغيل الاستراتيجية الذكية v4.2
    strategy = EnhancedEmaRsiMacdStrategyV4(telegram_notifier)
    
    # الباك-تستينغ الذكي v4.2
    backtest_result = strategy.run_intelligent_backtest_v4_2(df)
    
    # إرسال التقرير الذكي v4.2 إلى التلغرام
    await strategy.send_intelligent_telegram_report_v4_2(backtest_result, df)
    
    # حفظ النتائج في ملف
    trades_df = pd.DataFrame(strategy.trade_history)
    if not trades_df.empty:
        filename = f'enhanced_v4_2_trades_{SYMBOL}_{TIMEFRAME}_{datetime.now().strftime("%Y%m%d_%H%M%S")}.csv'
        trades_df.to_csv(filename, index=False)
        logger.info(f"💾 تم حفظ سجل الصفقات في {filename}")
    
    logger.info("✅ اكتمل تشغيل الاستراتيجية الذكية v4.2 بنجاح")

if __name__ == "__main__":
    # تشغيل الوظيفة الرئيسية
    asyncio.run(main())
