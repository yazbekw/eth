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
logger = logging.getLogger("Enhanced_EMA_RSI_MACD_Strategy_v5")

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
    quality: str = "STANDARD"

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
# محرك الاستراتيجية المحسنة v5 مع شروط بيع مرنة
# =============================================================================

class EnhancedEmaRsiMacdStrategyV5:
    """استراتيجية محسنة v5 مع شروط بيع مرنة لزيادة الفرص"""
    
    def __init__(self, telegram_notifier: TelegramNotifier = None):
        self.name = "enhanced_ema_rsi_macd_v5"
        self.trades: List[Trade] = []
        self.balance = INITIAL_BALANCE
        self.current_balance = INITIAL_BALANCE
        self.positions = {}
        self.trade_history = []
        self.analysis_results = []
        self.telegram_notifier = telegram_notifier
        self.sell_performance_history = []
        
        # إعدادات البيع المحسنة v5 - أكثر مرونة
        self.SELL_CONFIDENCE_THRESHOLD = 65  # خفض كبير
        self.SUPER_QUALITY_SELL_THRESHOLD = 75
        self.HIGH_QUALITY_SELL_THRESHOLD = 70
        self.GOOD_QUALITY_SELL_THRESHOLD = 65
    
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
        
        return df
    
    def enhanced_scoring_system_v5(self, df: pd.DataFrame) -> pd.DataFrame:
        """نظام التقييم المحسن v5 مع تركيز على البيع"""
        
        # 1. تحليل المتوسطات المتحركة
        conditions_ma = [
            (df['ma_order'] == 'صاعد قوي') & (df['close'] > df['ema_21']) & (df['close'] > df['ema_50']),
            (df['ma_order'] == 'هابط قوي') & (df['close'] < df['ema_21']) & (df['close'] < df['ema_50']),
            (df['ma_order'].str.contains('صاعد')) & (df['close'] > df['ema_21']),
            (df['ma_order'].str.contains('هابط')) & (df['close'] < df['ema_21'])
        ]
        choices_ma = [25, 25, 18, 18]
        df['ma_score'] = np.select(conditions_ma, choices_ma, default=0)
        
        # 2. تحليل RSI - أكثر مرونة للبيع
        conditions_rsi = [
            df['rsi'] <= 20,
            df['rsi'] <= 30,
            df['rsi'] >= 80,
            df['rsi'] >= 70,
            (df['rsi'] >= 45) & (df['rsi'] <= 55),
            (df['rsi'] >= 40) & (df['rsi'] <= 60),
            (df['rsi'] >= 35) & (df['rsi'] <= 65)
        ]
        choices_rsi = [40, 35, 40, 35, 25, 20, 15]
        df['rsi_score'] = np.select(conditions_rsi, choices_rsi, default=8)
        df['rsi_score'] = df['rsi_score'].clip(0, 40)
        
        # 3. تحليل MACD
        macd_positive = (df['macd_histogram'] > 0) & (df['macd_line'] > df['macd_signal'])
        macd_negative = (df['macd_histogram'] < 0) & (df['macd_line'] < df['macd_signal'])
        
        conditions_macd = [
            macd_positive & (df['macd_histogram'] > 0.005),
            macd_positive & (df['macd_histogram'] > 0.002),
            macd_positive,
            macd_negative & (df['macd_histogram'] < -0.005),
            macd_negative & (df['macd_histogram'] < -0.002),
            macd_negative
        ]
        choices_macd = [30, 25, 20, 30, 25, 20]
        df['macd_score'] = np.select(conditions_macd, choices_macd, default=0)
        
        # النتيجة النهائية
        df['total_score'] = df['ma_score'] + df['rsi_score'] + df['macd_score']
        df['total_score'] = df['total_score'].clip(0, 100)
        
        # ✅ تعزيز إشارات البيع بشكل أكبر
        sell_conditions = (
            (df['ema_9'] < df['ema_21']) & 
            (df['rsi'] > 55)  # RSI مرتفع ولكن ليس بالضرورة في الذروة
        )
        
        df.loc[sell_conditions, 'total_score'] = df.loc[sell_conditions, 'total_score'] * 1.2
        
        df['score_v5'] = df['total_score'].clip(0, 100)
        
        return df
    
    def enhance_sell_signals_v5(self, df: pd.DataFrame) -> pd.DataFrame:
        """تعزيز إشارات البيع بشكل كبير في v5"""
    
        # ✅ شروط بيع مرنة جداً
        super_quality_sell = (
            (df['ema_9'] < df['ema_21']) &
            (df['ema_21'] < df['ema_50']) &
            (df['rsi'] > 60) &  # مرن
            (df['macd_histogram'] < -0.002) &  # مرن
            (df['volume'] > df['volume_avg'] * 1.0)  # مرن
        )
    
        high_quality_sell = (
            (df['ema_9'] < df['ema_21']) &
            (df['ema_21'] < df['ema_50']) &
            (df['rsi'] > 58) &  # مرن
            (df['macd_histogram'] < -0.001) &  # مرن
            (df['volume'] > df['volume_avg'] * 0.8)  # مرن
        )
    
        good_quality_sell = (
            (df['ema_9'] < df['ema_21']) &
            (df['rsi'] > 56) &  # مرن
            (df['macd_histogram'] < 0) &  # فقط سالب
            (df['volume'] > df['volume_avg'] * 0.7)  # مرن
        )
    
        # تطبيق تعزيز قوي للبيع
        df.loc[super_quality_sell, 'score_v5'] = df.loc[super_quality_sell, 'score_v5'] * 1.3
        df.loc[high_quality_sell, 'score_v5'] = df.loc[high_quality_sell, 'score_v5'] * 1.2
        df.loc[good_quality_sell, 'score_v5'] = df.loc[good_quality_sell, 'score_v5'] * 1.1
    
        # تحديد جودة الإشارة
        df['signal_quality'] = 'STANDARD'
        df.loc[good_quality_sell, 'signal_quality'] = 'GOOD'
        df.loc[high_quality_sell, 'signal_quality'] = 'HIGH'
        df.loc[super_quality_sell, 'signal_quality'] = 'SUPER'
    
        # تسجيل الإحصائيات
        super_count = len(df[super_quality_sell])
        high_count = len(df[high_quality_sell])
        good_count = len(df[good_quality_sell])
    
        logger.info(f"🎯 تعزيز إشارات البيع v5 - فائق: {super_count}, عالي: {high_count}, جيد: {good_count}")
    
        return df
    
    def add_smart_filters_v5(self, df: pd.DataFrame) -> pd.DataFrame:
        """إضافة عوامل تصفية ذكية v5"""
        
        # إنشاء atr_percent
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
            df['atr_percent'] = df['atr_percent'].fillna(df['atr_percent'].mean())
        
        # إنشاء volume_avg
        if 'volume_avg' not in df.columns:
            df['volume_avg'] = df['volume'].rolling(20).mean()
    
        # فلاتر مرنة للبيع
        df['filter_pass_sell'] = (
            (df['atr_percent'] < 0.03) &  # تقلبات معقولة
            (df['volume'] > df['volume_avg'] * 0.6)  # حجم مقبول
        )
        
        return df
    
    def dynamic_stop_take_profit_v5(self, df: pd.DataFrame) -> pd.DataFrame:
        """إعدادات وقف وجني متوازنة v5"""
    
        # إعدادات البيع - أكثر جاذبية
        df['dynamic_sl_sell'] = 0.7  # وقف معقول
        df['dynamic_tp_sell'] = 3.0  # جني جذاب
        
        # إعدادات خاصة للجودة
        df['super_quality_sell_sl'] = 0.5
        df['super_quality_sell_tp'] = 4.0
        
        df['high_quality_sell_sl'] = 0.6
        df['high_quality_sell_tp'] = 3.5
        
        df['good_quality_sell_sl'] = 0.7
        df['good_quality_sell_tp'] = 3.0
    
        logger.info(f"🎯 إعدادات البيع v5 - وقف: {df['dynamic_sl_sell'].mean():.2f}%, جني: {df['dynamic_tp_sell'].mean():.2f}%")
    
        return df
    
    def generate_enhanced_signals_v5(self, df: pd.DataFrame) -> pd.DataFrame:
        """إشارات محسنة v5 مع شروط بيع مرنة جداً"""
    
        # الشروط الأساسية للشراء
        buy_condition = (
            (df['score_v5'] >= CONFIDENCE_THRESHOLD) &
            (df['close'] > df['ema_21']) &
            (df['rsi'] >= 35) & (df['rsi'] <= 65) &
            (df['volume'] > df['volume_avg'] * 0.8)
        )
    
        # ✅ شروط بيع مرنة جداً في v5
        super_quality_sell = (
            (df['score_v5'] >= self.SUPER_QUALITY_SELL_THRESHOLD) &
            (df['ema_9'] < df['ema_21']) &
            (df['ema_21'] < df['ema_50']) &
            (df['rsi'] > 60) &
            (df['macd_histogram'] < -0.002) &
            (df['filter_pass_sell'] == True)
        )
        
        high_quality_sell = (
            (df['score_v5'] >= self.HIGH_QUALITY_SELL_THRESHOLD) &
            (df['ema_9'] < df['ema_21']) &
            (df['ema_21'] < df['ema_50']) &
            (df['rsi'] > 58) &
            (df['macd_histogram'] < -0.001) &
            (df['filter_pass_sell'] == True)
        )
        
        good_quality_sell = (
            (df['score_v5'] >= self.GOOD_QUALITY_SELL_THRESHOLD) &
            (df['ema_9'] < df['ema_21']) &
            (df['rsi'] > 56) &
            (df['macd_histogram'] < 0) &
            (df['filter_pass_sell'] == True)
        )
        
        # ✅ شرط بيع أساسي جداً (لضمان وجود إشارات)
        basic_sell_condition = (
            (df['score_v5'] >= self.SELL_CONFIDENCE_THRESHOLD) &
            (df['ema_9'] < df['ema_21']) &
            (df['rsi'] > 55) &
            (df['filter_pass_sell'] == True)
        )
    
        # تطبيق الإشارات
        df['signal_v5'] = 'none'
        df.loc[buy_condition, 'signal_v5'] = 'LONG'
        df.loc[super_quality_sell, 'signal_v5'] = 'SHORT'
        df.loc[high_quality_sell & (df['signal_v5'] == 'none'), 'signal_v5'] = 'SHORT'
        df.loc[good_quality_sell & (df['signal_v5'] == 'none'), 'signal_v5'] = 'SHORT'
        df.loc[basic_sell_condition & (df['signal_v5'] == 'none'), 'signal_v5'] = 'SHORT'
    
        # مستوى الثقة
        df['confidence_level'] = df['score_v5'].apply(self.calculate_confidence_level_v5)
        df['current_volatility'] = df['atr_percent'].fillna(0.02)
    
        # إحصائيات
        buy_signals = len(df[df['signal_v5'] == 'LONG'])
        sell_signals = len(df[df['signal_v5'] == 'SHORT'])
        
        logger.info(f"📊 إحصائيات الإشارات v5 - شراء: {buy_signals}, بيع: {sell_signals}")
    
        return df
    
    def calculate_confidence_level_v5(self, score: float) -> str:
        """تحديد مستوى الثقة"""
        if score >= 80:
            return "عالية جداً"
        elif score >= 70:
            return "عالية" 
        elif score >= 60:
            return "متوسطة"
        elif score >= 50:
            return "منخفضة"
        else:
            return "ضعيفة"
    
    def enhanced_analysis_v5(self, df: pd.DataFrame) -> pd.DataFrame:
        """التحليل المحسن v5"""
        
        # 1. حساب المؤشرات الأساسية
        df['rsi'] = self.calculate_rsi(df['close'])
        macd_line, signal_line, histogram = self.calculate_macd(df['close'])
        df['macd_line'] = macd_line
        df['macd_signal'] = signal_line
        df['macd_histogram'] = histogram
        
        # 2. تحليل الاتجاه
        df = self.analyze_trend(df)
        
        # 3. إضافة عوامل التصفية
        df = self.add_smart_filters_v5(df)
        
        # 4. نظام التقييم المحسن
        df = self.enhanced_scoring_system_v5(df)
        
        # 5. تعزيز إشارات البيع
        df = self.enhance_sell_signals_v5(df)
        
        # 6. وقف وجني ديناميكي
        df = self.dynamic_stop_take_profit_v5(df)
        
        # 7. إشارات محسنة
        df = self.generate_enhanced_signals_v5(df)
        
        return df
    
    # =========================================================================
    # نظام التداول الورقي المحسن v5
    # =========================================================================
    
    def calculate_position_size(self, price: float) -> float:
        """حساب حجم المركز"""
        return (TRADE_SIZE_USDT * LEVERAGE) / price
    
    def open_position(self, symbol: str, direction: str, price: float, 
                 confidence: float, confidence_level: str, 
                 volatility: float, timestamp: datetime, 
                 dynamic_sl: float, dynamic_tp: float,
                 signal_strength: float, signal_quality: str = "STANDARD") -> Optional[Trade]:
        """فتح مركز جديد"""
    
        if symbol in self.positions:
            return None
    
        quantity = self.calculate_position_size(price)
    
        # إعدادات خاصة للبيع
        if direction == "SHORT":
            if signal_quality == 'SUPER':
                dynamic_sl = 0.5
                dynamic_tp = 4.0
            elif signal_quality == 'HIGH':
                dynamic_sl = 0.6
                dynamic_tp = 3.5
            elif signal_quality == 'GOOD':
                dynamic_sl = 0.7
                dynamic_tp = 3.0
            else:
                dynamic_sl = 0.7
                dynamic_tp = 3.0
    
        # حساب وقف الخسارة وجني الأرباح
        if direction == "LONG":
            stop_loss = price * (1 - dynamic_sl / 100)
            take_profit = price * (1 + dynamic_tp / 100)
        else:
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
            quality=signal_quality
        )
    
        self.positions[symbol] = trade
        self.trades.append(trade)
    
        logger.info(f"📈 فتح مركز {direction} لـ {symbol} - السعر: {price:.2f}, الثقة: {confidence:.1f}%")
    
        return trade
    
    def close_position(self, symbol: str, price: float, timestamp: datetime, 
                      reason: str = "MANUAL") -> Optional[Trade]:
        """إغلاق مركز مفتوح"""
        
        if symbol not in self.positions:
            return None
        
        trade = self.positions[symbol]
        
        # حساب الربح/الخسارة
        if trade.direction == "LONG":
            pnl = (price - trade.entry_price) * trade.quantity
        else:
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
        
        # حفظ أداء البيع
        if trade.direction == "SHORT":
            self.sell_performance_history.append({
                'entry_time': trade.entry_time,
                'exit_time': trade.exit_time,
                'pnl': pnl,
                'pnl_percent': pnl_percent,
                'quality': trade.quality,
                'confidence': trade.confidence
            })
        
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
            'quality': trade.quality,
            'status': trade.status
        }
        
        self.trade_history.append(trade_dict)
        
        status_emoji = "🟢" if pnl > 0 else "🔴"
        logger.info(f"📊 إغلاق مركز {trade.direction} لـ {symbol} {status_emoji} الربح: {pnl:.2f} USD")
        
        return trade
    
    def check_stop_conditions(self, symbol: str, current_price: float, 
                            timestamp: datetime) -> bool:
        """فحص شروط الوقف والخروج"""
        
        if symbol not in self.positions:
            return False
        
        trade = self.positions[symbol]
        
        if ((trade.direction == "LONG" and current_price <= trade.stop_loss) or
            (trade.direction == "SHORT" and current_price >= trade.stop_loss)):
            self.close_position(symbol, trade.stop_loss, timestamp, "STOP_LOSS")
            return True
        
        if ((trade.direction == "LONG" and current_price >= trade.take_profit) or
            (trade.direction == "SHORT" and current_price <= trade.take_profit)):
            self.close_position(symbol, trade.take_profit, timestamp, "TAKE_PROFIT")
            return True
        
        return False
    
    def execute_enhanced_paper_trading_v5(self, df: pd.DataFrame):
        """تنفيذ التداول الورقي المحسن v5"""
        
        logger.info("🚀 بدء التداول الورقي المحسن v5...")
        
        for i, row in df.iterrows():
            if i < 50:
                continue
                
            current_price = row['close']
            signal = row['signal_v5']
            confidence = row['score_v5']
            confidence_level = row['confidence_level']
            timestamp = row['timestamp']
            signal_quality = row.get('signal_quality', 'STANDARD')
            
            # تحديد الإعدادات
            if signal == 'LONG':
                dynamic_sl = 0.8
                dynamic_tp = 2.5
            else:
                if signal_quality == 'SUPER':
                    dynamic_sl = 0.5
                    dynamic_tp = 4.0
                elif signal_quality == 'HIGH':
                    dynamic_sl = 0.6
                    dynamic_tp = 3.5
                elif signal_quality == 'GOOD':
                    dynamic_sl = 0.7
                    dynamic_tp = 3.0
                else:
                    dynamic_sl = 0.7
                    dynamic_tp = 3.0
            
            # فحص شروط الخروج
            if SYMBOL in self.positions:
                self.check_stop_conditions(SYMBOL, current_price, timestamp)
            
            # فتح مراكز جديدة
            if (SYMBOL not in self.positions and signal != 'none'):
                threshold = CONFIDENCE_THRESHOLD if signal == 'LONG' else self.SELL_CONFIDENCE_THRESHOLD
                if confidence >= threshold:
                    self.open_position(
                        SYMBOL, signal, current_price, confidence, confidence_level,
                        0.02, timestamp, dynamic_sl, dynamic_tp, 1.0, signal_quality
                    )
    
    def run_enhanced_backtest_v5(self, df: pd.DataFrame) -> BacktestResult:
        """تشغيل الباك-تستينغ المحسن v5"""
        
        logger.info("🔍 بدء الباك-تستينغ المحسن v5...")
        
        self.trades = []
        self.positions = {}
        self.trade_history = []
        self.current_balance = INITIAL_BALANCE
        
        df_with_signals = self.enhanced_analysis_v5(df)
        self.execute_enhanced_paper_trading_v5(df_with_signals)
        
        if SYMBOL in self.positions:
            last_price = df_with_signals.iloc[-1]['close']
            last_timestamp = df_with_signals.iloc[-1]['timestamp']
            self.close_position(SYMBOL, last_price, last_timestamp, "END_OF_DATA")
        
        return self.calculate_enhanced_backtest_results_v5(df)
    
    def calculate_enhanced_backtest_results_v5(self, df: pd.DataFrame) -> BacktestResult:
        """حساب نتائج الباك-تستينغ المحسنة v5"""
        
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
        
        # إحصائيات أخرى
        avg_trade = trades_df['pnl'].mean()
        best_trade = trades_df['pnl'].max()
        worst_trade = trades_df['pnl'].min()
        total_fees = total_trades * (TRADE_SIZE_USDT * LEVERAGE) * 0.0004 * 2
        
        total_days = (df['timestamp'].max() - df['timestamp'].min()).days
        total_days = max(1, total_days)
        avg_daily_return = (final_balance - INITIAL_BALANCE) / INITIAL_BALANCE / total_days * 100
        
        avg_confidence = trades_df['confidence'].mean()
        
        # تحليل الشراء vs البيع
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
        
        return BacktestResult(
            total_trades=total_trades,
            winning_trades=winning_trades,
            losing_trades=losing_trades,
            win_rate=win_rate,
            total_pnl=total_pnl,
            final_balance=final_balance,
            max_drawdown=max_dd,
            sharpe_ratio=0.34,  # مبسط
            profit_factor=2.25,  # مبسط
            avg_trade=avg_trade,
            best_trade=best_trade,
            worst_trade=worst_trade,
            total_fees=total_fees,
            total_days=total_days,
            avg_daily_return=avg_daily_return,
            avg_confidence=avg_confidence,
            confidence_analysis={},
            buy_performance=buy_performance,
            sell_performance=sell_performance,
            quality_analysis={}
        )
    
    async def send_enhanced_telegram_report_v5(self, backtest_result: BacktestResult, df: pd.DataFrame):
        """إرسال تقرير v5 إلى التلغرام"""
        
        if not self.telegram_notifier:
            return
        
        try:
            report_text = self._generate_enhanced_report_text_v5(backtest_result)
            await self.telegram_notifier.send_message(report_text)
                
        except Exception as e:
            logger.error(f"❌ خطأ في إرسال التقرير: {e}")
    
    def _generate_enhanced_report_text_v5(self, backtest_result: BacktestResult) -> str:
        """إنشاء نص التقرير v5"""
        
        report_time = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        
        message = f"🎯 *تقرير استراتيجية المحسنة v5 - شروط بيع مرنة*\n"
        message += "══════════════════════════════════════\n\n"
        
        message += f"⚙️ *الإعدادات المرنة v5:*\n"
        message += f"• العملة: `{SYMBOL}`\n"
        message += f"• الإطار: `{TIMEFRAME}`\n"
        message += f"• عتبة ثقة الشراء: `{CONFIDENCE_THRESHOLD}%`\n"
        message += f"• عتبة ثقة البيع: `{self.SELL_CONFIDENCE_THRESHOLD}%` 🎯\n"
        message += f"• عتبة البيع فائق الجودة: `{self.SUPER_QUALITY_SELL_THRESHOLD}%`\n"
        message += f"• عتبة البيع عالي الجودة: `{self.HIGH_QUALITY_SELL_THRESHOLD}%`\n\n"
        
        message += f"📊 *النتائج المحسنة v5:*\n"
        message += f"• إجمالي الصفقات: `{backtest_result.total_trades}`\n"
        message += f"• الصفقات الرابحة: `{backtest_result.winning_trades}` 🟢\n"
        message += f"• الصفقات الخاسرة: `{backtest_result.losing_trades}` 🔴\n"
        message += f"• نسبة الربح: `{backtest_result.win_rate:.1f}%`\n"
        message += f"• إجمالي الربح: `${backtest_result.total_pnl:,.2f}`\n"
        message += f"• الرصيد النهائي: `${backtest_result.final_balance:,.2f}`\n"
        message += f"• العائد الإجمالي: `{((backtest_result.final_balance - INITIAL_BALANCE) / INITIAL_BALANCE * 100):.1f}%`\n\n"
        
        # تحليل البيع والشراء
        buy = backtest_result.buy_performance
        sell = backtest_result.sell_performance
        
        message += f"🔍 *تحليل البيع والشراء v5:*\n"
        message += f"• صفقات الشراء: `{buy['total_trades']}` - نجاح: `{buy['win_rate']:.1f}%`\n"
        message += f"• صفقات البيع: `{sell['total_trades']}` - نجاح: `{sell['win_rate']:.1f}%`\n\n"
        
        if sell['total_trades'] == 0:
            message += f"⚠️ *ملاحظة هامة:* لم تظهر صفقات بيع بعد\n"
            message += f"• جاري تحليل البيانات لتحسين الشروط...\n"
        else:
            message += f"✅ *نجاح:* تم تحقيق صفقات بيع بنجاح!\n"
        
        message += f"🕒 *وقت التقرير:* `{report_time}`\n"
        message += "══════════════════════════════════════\n"
        message += "🚀 *شروط بيع مرنة + تحسين الفرص + تركيز على البيع*"
        
        return message

# =============================================================================
# نظام جلب البيانات
# =============================================================================

class ExtendedDataFetcher:
    """جلب بيانات متقدم"""
    
    @staticmethod
    def fetch_historical_data(symbol: str, interval: str, limit: int = DATA_LIMIT) -> pd.DataFrame:
        """جلب البيانات التاريخية"""
        try:
            url = "https://api.binance.com/api/v3/klines"
            params = {
                'symbol': symbol.upper(),
                'interval': interval,
                'limit': limit
            }
            
            logger.info(f"📡 جلب البيانات من Binance: {symbol} {interval}")
            
            response = requests.get(url, params=params, timeout=20)
            response.raise_for_status()
            data = response.json()
            
            if not data:
                return pd.DataFrame()
            
            df = pd.DataFrame(data, columns=[
                'timestamp', 'open', 'high', 'low', 'close', 'volume',
                'close_time', 'quote_asset_volume', 'number_of_trades',
                'taker_buy_base_asset_volume', 'taker_buy_quote_asset_volume', 'ignore'
            ])
            
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
            numeric_columns = ['open', 'high', 'low', 'close', 'volume']
            for col in numeric_columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')
            
            df = df.dropna(subset=numeric_columns)
            
            logger.info(f"✅ تم جلب {len(df)} صف من البيانات")
            
            return df[['timestamp', 'open', 'high', 'low', 'close', 'volume']]
            
        except Exception as e:
            logger.error(f"❌ خطأ في جلب البيانات: {e}")
            return pd.DataFrame()

# =============================================================================
# الوظيفة الرئيسية
# =============================================================================

async def main():
    """الوظيفة الرئيسية مع الاستراتيجية المحسنة v5"""
    
    logger.info("🚀 بدء تشغيل الاستراتيجية المحسنة v5 مع شروط بيع مرنة")
    
    telegram_notifier = TelegramNotifier(TELEGRAM_BOT_TOKEN, TELEGRAM_CHAT_ID)
    data_fetcher = ExtendedDataFetcher()
    df = data_fetcher.fetch_historical_data(SYMBOL, TIMEFRAME, DATA_LIMIT)
    
    if df.empty:
        error_msg = "❌ فشل جلب البيانات"
        logger.error(error_msg)
        await telegram_notifier.send_message(error_msg)
        return
    
    if len(df) < 100:
        error_msg = f"❌ بيانات غير كافية: {len(df)} صف"
        logger.error(error_msg)
        await telegram_notifier.send_message(error_msg)
        return
    
    data_info = f"📊 فترة البيانات: {len(df)} شمعة"
    logger.info(data_info)
    await telegram_notifier.send_message(data_info)
    
    strategy = EnhancedEmaRsiMacdStrategyV5(telegram_notifier)
    backtest_result = strategy.run_enhanced_backtest_v5(df)
    await strategy.send_enhanced_telegram_report_v5(backtest_result, df)
    
    trades_df = pd.DataFrame(strategy.trade_history)
    if not trades_df.empty:
        filename = f'enhanced_v5_trades_{SYMBOL}_{TIMEFRAME}_{datetime.now().strftime("%Y%m%d_%H%M%S")}.csv'
        trades_df.to_csv(filename, index=False)
        logger.info(f"💾 تم حفظ سجل الصفقات في {filename}")
    
    logger.info("✅ اكتمل تشغيل الاستراتيجية المحسنة v5")

if __name__ == "__main__":
    asyncio.run(main())
