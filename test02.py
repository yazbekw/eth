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
STOP_LOSS_PERCENT = float(os.getenv("STOP_LOSS_PERCENT", "1.0"))
TAKE_PROFIT_PERCENT = float(os.getenv("TAKE_PROFIT_PERCENT", "2.0"))
TRADE_SIZE_USDT = float(os.getenv("TRADE_SIZE_USDT", "100"))
LEVERAGE = int(os.getenv("LEVERAGE", "10"))
INITIAL_BALANCE = float(os.getenv("INITIAL_BALANCE", "5000.0"))
CONFIDENCE_THRESHOLD = int(os.getenv("CONFIDENCE_THRESHOLD", "60"))

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
logger = logging.getLogger("Volume_Divergence_Strategy")

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
    divergence_type: str = ""
    volume_ratio: float = 0

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
# استراتيجية الانزياح الحجمي (Volume Divergence Strategy)
# =============================================================================

class VolumeDivergenceStrategy:
    """استراتيجية الانزياح بين السعر والحجم - نظام تقييم محسن"""
    
    def __init__(self, telegram_notifier: TelegramNotifier = None):
        self.name = "volume_divergence"
        self.trades: List[Trade] = []
        self.balance = INITIAL_BALANCE
        self.current_balance = INITIAL_BALANCE
        self.positions = {}
        self.trade_history = []
        self.analysis_results = []
        self.telegram_notifier = telegram_notifier
    
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
        
        # كشف الانزياح الإيجابي (هبوط سعر مع ضعف حجم بيع)
        if (price_trend_recent < -0.02 and price_trend_older < -0.02 and
            volume_trend_recent > -0.1 and volume_trend_older < -0.2):
            strength = min(60, int(abs(price_trend_recent) * 1500 + abs(volume_trend_recent) * 100))
            return {"divergence": "positive_bullish", "strength": strength}
        
        # كشف الانزياح السلبي (صعود سعر مع ضعف حجم شراء)
        elif (price_trend_recent > 0.02 and price_trend_older > 0.02 and
              volume_trend_recent < 0.1 and volume_trend_older > 0.2):
            strength = min(60, int(abs(price_trend_recent) * 1500 + abs(volume_trend_recent) * 100))
            return {"divergence": "negative_bearish", "strength": strength}
        
        # كشف التأكيد الحجمي القوي
        elif ((price_trend_recent > 0.03 and volume_trend_recent > 0.4) or
              (price_trend_recent < -0.03 and volume_trend_recent > 0.4)):
            strength = min(70, int(abs(price_trend_recent) * 1200 + volume_trend_recent * 80))
            return {"divergence": "volume_confirmation", "strength": strength}
        
        # كشف الانزياح الخفي
        elif ((abs(price_trend_recent) < 0.01 and volume_trend_recent > 0.3) or
              (abs(price_trend_recent) > 0.02 and abs(volume_trend_recent) < 0.05)):
            strength = min(50, int(abs(price_trend_recent) * 1000 + abs(volume_trend_recent) * 60))
            return {"divergence": "hidden_divergence", "strength": strength}
        
        return {"divergence": "none", "strength": 0}
    
    def enhanced_volume_divergence_scoring(self, divergence_data: Dict, 
                                         price_change: float, volume_change: float,
                                         current_volume: float, avg_volume: float) -> tuple:
        """نظام التقييم المحسن للانزياح الحجمي"""
        base_score = divergence_data["strength"]
        scoring_details = []
        
        if divergence_data["divergence"] == "positive_bullish":
            # تعزيز حسب قوة الانزياح
            price_enhancement = min(25, abs(price_change) * 800)
            volume_enhancement = min(15, volume_change * 30)
            final_score = min(100, base_score + price_enhancement + volume_enhancement)
            
            scoring_details.append(f"انزياح إيجابي قوي: {base_score} نقطة أساسية")
            if price_enhancement > 0:
                scoring_details.append(f"تعزيز سعري: +{price_enhancement:.1f}")
            if volume_enhancement > 0:
                scoring_details.append(f"تعزيز حجمي: +{volume_enhancement:.1f}")
            
        elif divergence_data["divergence"] == "negative_bearish":
            price_enhancement = min(25, abs(price_change) * 800)
            volume_enhancement = min(15, abs(volume_change) * 30)
            final_score = min(100, base_score + price_enhancement + volume_enhancement)
            
            scoring_details.append(f"انزياح سلبي قوي: {base_score} نقطة أساسية")
            if price_enhancement > 0:
                scoring_details.append(f"تعزيز سعري: +{price_enhancement:.1f}")
            if volume_enhancement > 0:
                scoring_details.append(f"تعزيز حجمي: +{volume_enhancement:.1f}")
            
        elif divergence_data["divergence"] == "volume_confirmation":
            enhancement = min(30, abs(price_change) * 600 + volume_change * 25)
            final_score = min(100, base_score + enhancement)
            
            scoring_details.append(f"تأكيد حجمي: {base_score} نقطة أساسية")
            scoring_details.append(f"تعزيز إضافي: +{enhancement:.1f}")
            
        elif divergence_data["divergence"] == "hidden_divergence":
            enhancement = min(20, abs(price_change) * 500 + abs(volume_change) * 20)
            final_score = min(100, base_score + enhancement)
            
            scoring_details.append(f"انزياح خفي: {base_score} نقطة أساسية")
            scoring_details.append(f"تعزيز إضافي: +{enhancement:.1f}")
            
        else:
            final_score = 0
            scoring_details.append("لا يوجد انزياح ملحوظ")
        
        return final_score, scoring_details
    
    def calculate_volume_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """حساب مؤشرات الحجم"""
        # المتوسطات المتحركة للحجم
        df['volume_ma_10'] = df['volume'].rolling(10).mean()
        df['volume_ma_20'] = df['volume'].rolling(20).mean()
        df['volume_ma_50'] = df['volume'].rolling(50).mean()
        
        # نسبة الحجم الحالي إلى المتوسط
        df['volume_ratio_10'] = df['volume'] / df['volume_ma_10']
        df['volume_ratio_20'] = df['volume'] / df['volume_ma_20']
        df['volume_ratio_50'] = df['volume'] / df['volume_ma_50']
        
        # تقلبات الحجم
        df['volume_volatility'] = df['volume'].rolling(20).std()
        
        return df
    
    def detect_volume_patterns(self, df: pd.DataFrame) -> pd.DataFrame:
        """كشف أنماط الحجم"""
        # حجم غير عادي
        df['volume_spike'] = df['volume_ratio_20'] > 2.0
        df['volume_drop'] = df['volume_ratio_20'] < 0.5
        
        # استمرارية الحجم
        df['volume_continuity'] = (df['volume_ratio_20'] > 1.2).rolling(3).sum() >= 2
        
        return df
    
    def generate_volume_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        """توليد إشارات الانزياح الحجمي"""
        
        signals = []
        confidence_scores = []
        divergence_types = []
        scoring_details_list = []
        
        for i in range(len(df)):
            if i < 40:  # تحتاج إلى بيانات كافية
                signals.append('none')
                confidence_scores.append(0)
                divergence_types.append('none')
                scoring_details_list.append([])
                continue
            
            # استخراج البيانات
            prices = df['close'].iloc[:i+1].tolist()
            volumes = df['volume'].iloc[:i+1].tolist()
            
            # حساب الانزياح
            divergence_data = self.calculate_divergence(prices, volumes)
            
            # حساب المؤشرات الإضافية
            current_price = prices[-1]
            price_change_20 = (current_price - prices[-20]) / prices[-20] * 100
            volume_change_20 = (volumes[-1] - np.mean(volumes[-20:])) / np.mean(volumes[-20:]) * 100
            current_volume = volumes[-1]
            avg_volume = np.mean(volumes[-20:])
            
            # نظام التقييم المحسن
            confidence_score, scoring_details = self.enhanced_volume_divergence_scoring(
                divergence_data, price_change_20, volume_change_20, current_volume, avg_volume
            )
            
            confidence_score = round(confidence_score)
            
            # تحديد الإشارة
            signal = 'none'
            if divergence_data["divergence"] == "positive_bullish" and confidence_score >= 40:
                confidence_score = min(95, confidence_score + 5)
                signal = "BUY"
            elif divergence_data["divergence"] == "negative_bearish" and confidence_score >= 40:
                confidence_score = min(95, confidence_score + 5)
                signal = "SELL"
            elif divergence_data["divergence"] == "volume_confirmation" and confidence_score >= 40:
                price_trend = "صاعد" if prices[-1] > prices[-10] else "هابط"
                signal = "BUY" if price_trend == "صاعد" else "SELL"
            elif divergence_data["divergence"] == "hidden_divergence" and confidence_score >= 45:
                if price_change_20 < 0 and volume_change_20 > 0:
                    signal = "BUY"
                elif price_change_20 > 0 and volume_change_20 < 0:
                    signal = "SELL"
            
            signals.append(signal)
            confidence_scores.append(confidence_score)
            divergence_types.append(divergence_data["divergence"])
            scoring_details_list.append(scoring_details)
        
        df['volume_signal'] = signals
        df['volume_confidence'] = confidence_scores
        df['divergence_type'] = divergence_types
        df['scoring_details'] = scoring_details_list
        
        return df
    
    def enhanced_volume_analysis(self, df: pd.DataFrame) -> pd.DataFrame:
        """التحليل الحجمي المحسن - الدالة الرئيسية"""
        
        # 1. حساب مؤشرات الحجم
        df = self.calculate_volume_indicators(df)
        
        # 2. كشف أنماط الحجم
        df = self.detect_volume_patterns(df)
        
        # 3. توليد إشارات الانزياح
        df = self.generate_volume_signals(df)
        
        # 4. إضافة مستوى الثقة
        df['confidence_level'] = df['volume_confidence'].apply(self.calculate_confidence_level)
        
        # حفظ نتائج التحليل
        self.analysis_results = df.to_dict('records')
        
        return df
    
    def calculate_confidence_level(self, score: float) -> str:
        """تحديد مستوى الثقة بدقة"""
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
    
    # =========================================================================
    # نظام التداول الورقي
    # =========================================================================
    
    def calculate_position_size(self, price: float) -> float:
        """حساب حجم المركز بناء على الرافعة وحجم الصفقة"""
        return (TRADE_SIZE_USDT * LEVERAGE) / price
    
    def open_position(self, symbol: str, direction: str, price: float, 
                     confidence: float, confidence_level: str, 
                     divergence_type: str, volume_ratio: float, 
                     timestamp: datetime) -> Optional[Trade]:
        """فتح مركز جديد"""
        
        if symbol in self.positions:
            logger.warning(f"يوجد مركز مفتوح بالفعل لـ {symbol}")
            return None
        
        # حساب حجم المركز
        quantity = self.calculate_position_size(price)
        
        # حساب وقف الخسارة وجني الأرباح
        if direction == "BUY":
            stop_loss = price * (1 - STOP_LOSS_PERCENT / 100)
            take_profit = price * (1 + TAKE_PROFIT_PERCENT / 100)
        else:  # SELL
            stop_loss = price * (1 + STOP_LOSS_PERCENT / 100)
            take_profit = price * (1 - TAKE_PROFIT_PERCENT / 100)
        
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
            divergence_type=divergence_type,
            volume_ratio=volume_ratio
        )
        
        self.positions[symbol] = trade
        self.trades.append(trade)
        
        logger.info(f"📈 فتح مركز {direction} لـ {symbol} "
                   f"السعر: {price:.2f}, الثقة: {confidence:.1f}% ({confidence_level})")
        logger.info(f"   الانزياح: {divergence_type}, نسبة الحجم: {volume_ratio:.2f}x")
        
        return trade
    
    def close_position(self, symbol: str, price: float, timestamp: datetime, 
                      reason: str = "MANUAL") -> Optional[Trade]:
        """إغلاق مركز مفتوح"""
        
        if symbol not in self.positions:
            logger.warning(f"لا يوجد مركز مفتوح لـ {symbol}")
            return None
        
        trade = self.positions[symbol]
        
        # حساب الربح/الخسارة
        if trade.direction == "BUY":
            pnl = (price - trade.entry_price) * trade.quantity
        else:  # SELL
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
            'divergence_type': trade.divergence_type,
            'volume_ratio': trade.volume_ratio,
            'status': trade.status
        }
        
        self.trade_history.append(trade_dict)
        
        status_emoji = "🟢" if pnl > 0 else "🔴"
        logger.info(f"📊 إغلاق مركز {trade.direction} لـ {symbol} {status_emoji}"
                   f" الربح: {pnl:.2f} USD ({pnl_percent:.2f}%) - {reason}")
        
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
    
    def execute_volume_trading(self, df: pd.DataFrame):
        """تنفيذ التداول الورقي بناء على الانزياح الحجمي"""
        
        logger.info("🚀 بدء التداول الورقي باستراتيجية الانزياح الحجمي...")
        
        for i, row in df.iterrows():
            if i < 40:  # تخطي الفترة الأولى لاستقرار المؤشرات
                continue
                
            current_price = row['close']
            signal = row['volume_signal']
            confidence = row['volume_confidence']
            confidence_level = row['confidence_level']
            divergence_type = row['divergence_type']
            volume_ratio = row['volume_ratio_20']
            timestamp = row['timestamp']
            
            # فحص شروط الخروج للمراكز المفتوحة
            if SYMBOL in self.positions:
                self.check_stop_conditions(SYMBOL, current_price, timestamp)
            
            # فتح مراكز جديدة إذا لم يكن هناك مركز مفتوح
            if (SYMBOL not in self.positions and signal != 'none' and 
                confidence >= CONFIDENCE_THRESHOLD):
                
                self.open_position(
                    SYMBOL, signal, current_price, confidence, confidence_level,
                    divergence_type, volume_ratio, timestamp
                )
    
    # =========================================================================
    # الباك-تستينغ
    # =========================================================================
    
    def run_volume_backtest(self, df: pd.DataFrame) -> BacktestResult:
        """تشغيل الباك-تستينغ باستراتيجية الانزياح الحجمي"""
        
        logger.info("🔍 بدء الباك-تستينغ باستراتيجية الانزياح الحجمي...")
        
        # إعادة تعيين البيانات
        self.trades = []
        self.positions = {}
        self.trade_history = []
        self.current_balance = INITIAL_BALANCE
        
        # التحليل الحجمي المحسن
        df_with_signals = self.enhanced_volume_analysis(df)
        
        # تنفيذ التداول
        self.execute_volume_trading(df_with_signals)
        
        # إغلاق أي مراكز مفتوحة في النهاية
        if SYMBOL in self.positions:
            last_price = df_with_signals.iloc[-1]['close']
            last_timestamp = df_with_signals.iloc[-1]['timestamp']
            self.close_position(SYMBOL, last_price, last_timestamp, "END_OF_DATA")
        
        # حساب النتائج
        return self.calculate_volume_backtest_results(df)
    
    def calculate_volume_backtest_results(self, df: pd.DataFrame) -> BacktestResult:
        """حساب نتائج الباك-تستينغ"""
        
        if not self.trade_history:
            total_days = (df['timestamp'].max() - df['timestamp'].min()).days
            return BacktestResult(
                total_trades=0, winning_trades=0, losing_trades=0,
                win_rate=0, total_pnl=0, final_balance=self.current_balance,
                max_drawdown=0, sharpe_ratio=0, profit_factor=0,
                avg_trade=0, best_trade=0, worst_trade=0, total_fees=0,
                total_days=max(1, total_days), avg_daily_return=0,
                avg_confidence=0, divergence_analysis={}, volume_analysis={}
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
        
        # تحليل الانزياح
        divergence_analysis = {}
        for div_type in ['positive_bullish', 'negative_bearish', 'volume_confirmation', 'hidden_divergence']:
            div_trades = trades_df[trades_df['divergence_type'] == div_type]
            if len(div_trades) > 0:
                div_win_rate = (len(div_trades[div_trades['pnl'] > 0]) / len(div_trades)) * 100
                div_total_pnl = div_trades['pnl'].sum()
                divergence_analysis[div_type] = {
                    'trades': len(div_trades),
                    'win_rate': div_win_rate,
                    'total_pnl': div_total_pnl,
                    'avg_pnl': div_trades['pnl'].mean()
                }
        
        # تحليل الحجم
        volume_analysis = {
            'high_volume_trades': len(trades_df[trades_df['volume_ratio'] > 2.0]),
            'avg_volume_ratio': trades_df['volume_ratio'].mean(),
            'volume_correlation': trades_df['volume_ratio'].corr(trades_df['pnl'])
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
            volume_analysis=volume_analysis
        )
    
    # =========================================================================
    # التقارير
    # =========================================================================
    
    async def send_volume_telegram_report(self, backtest_result: BacktestResult, df: pd.DataFrame):
        """إرسال تقرير مفصل إلى التلغرام"""
        
        if not self.telegram_notifier:
            logger.warning("❌ نظام التلغرام غير متوفر")
            return
        
        try:
            # 1. إرسال التقرير النصي
            report_text = self._generate_volume_report_text(backtest_result)
            await self.telegram_notifier.send_message(report_text)
            
            # 2. إرسال الرسوم البيانية
            chart_buffer = self._create_volume_performance_chart(df, backtest_result)
            if chart_buffer:
                chart_caption = f"📊 تحليل أداء استراتيجية الانزياح الحجمي - {SYMBOL} ({TIMEFRAME})"
                await self.telegram_notifier.send_photo(chart_buffer, chart_caption)
            
            # 3. إرسال تحليل الانزياح
            if self.trade_history:
                divergence_analysis = self._generate_divergence_analysis(backtest_result)
                await self.telegram_notifier.send_message(divergence_analysis)
                
        except Exception as e:
            logger.error(f"❌ خطأ في إرسال التقرير إلى التلغرام: {e}")
    
    def _generate_volume_report_text(self, backtest_result: BacktestResult) -> str:
        """إنشاء نص التقرير للتلغرام"""
        
        report_time = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        
        message = f"📊 *تقرير استراتيجية الانزياح الحجمي*\n"
        message += "══════════════════════════════════════\n\n"
        
        message += f"🎯 *الإعدادات:*\n"
        message += f"• العملة: `{SYMBOL}`\n"
        message += f"• الإطار: `{TIMEFRAME}`\n"
        message += f"• الرافعة: `{LEVERAGE}x`\n"
        message += f"• حجم الصفقة: `${TRADE_SIZE_USDT}`\n"
        message += f"• وقف الخسارة: `{STOP_LOSS_PERCENT}%`\n"
        message += f"• جني الأرباح: `{TAKE_PROFIT_PERCENT}%`\n"
        message += f"• عتبة الثقة: `{CONFIDENCE_THRESHOLD}%`\n\n"
        
        message += f"📈 *النتائج الرئيسية:*\n"
        message += f"• إجمالي الصفقات: `{backtest_result.total_trades}`\n"
        message += f"• الصفقات الرابحة: `{backtest_result.winning_trades}` 🟢\n"
        message += f"• الصفقات الخاسرة: `{backtest_result.losing_trades}` 🔴\n"
        message += f"• نسبة الربح: `{backtest_result.win_rate:.1f}%`\n"
        message += f"• إجمالي الربح: `${backtest_result.total_pnl:,.2f}`\n"
        message += f"• الرصيد النهائي: `${backtest_result.final_balance:,.2f}`\n"
        message += f"• العائد الإجمالي: `{((backtest_result.final_balance - INITIAL_BALANCE) / INITIAL_BALANCE * 100):.1f}%`\n"
        message += f"• متوسط الثقة: `{backtest_result.avg_confidence:.1f}%`\n\n"
        
        message += f"🎯 *مقاييس المخاطرة:*\n"
        message += f"• أقصى خسارة: `{backtest_result.max_drawdown:.1f}%`\n"
        message += f"• متوسط الربح/صفقة: `${backtest_result.avg_trade:.2f}`\n"
        message += f"• أفضل صفقة: `${backtest_result.best_trade:.2f}` 🚀\n"
        message += f"• أسوأ صفقة: `${backtest_result.worst_trade:.2f}` 📉\n"
        message += f"• نسبة شارب: `{backtest_result.sharpe_ratio:.2f}`\n"
        message += f"• عامل الربحية: `{backtest_result.profit_factor:.2f}`\n\n"
        
        message += f"⏰ *الفترة الزمنية:*\n"
        message += f"• إجمالي الأيام: `{backtest_result.total_days}`\n"
        message += f"• متوسط العائد اليومي: `{backtest_result.avg_daily_return:.3f}%`\n\n"
        
        message += f"🕒 *وقت التقرير:* `{report_time}`\n"
        message += "══════════════════════════════════════\n"
        message += "⚡ *استراتيجية الانزياح الحجمي - نظام التقييم 0-100*"
        
        return message
    
    def _generate_divergence_analysis(self, backtest_result: BacktestResult) -> str:
        """إنشاء تحليل الانزياح"""
        
        message = "🔍 *تحليل الانزياح الحجمي:*\n"
        message += "────────────────────\n"
        
        divergence_names = {
            'positive_bullish': '🟢 الانزياح الإيجابي',
            'negative_bearish': '🔴 الانزياح السلبي', 
            'volume_confirmation': '📈 التأكيد الحجمي',
            'hidden_divergence': '🎯 الانزياح الخفي'
        }
        
        for div_type, analysis in backtest_result.divergence_analysis.items():
            display_name = divergence_names.get(div_type, div_type)
            message += f"{display_name}:\n"
            message += f"• الصفقات: `{analysis['trades']}`\n"
            message += f"• الدقة: `{analysis['win_rate']:.1f}%`\n"
            message += f"• الربح: `${analysis['total_pnl']:.2f}`\n"
            message += f"• المتوسط: `${analysis['avg_pnl']:.2f}`\n\n"
        
        # تحليل الحجم
        vol_analysis = backtest_result.volume_analysis
        message += f"📊 *تحليل الحجم:*\n"
        message += f"• الصفقات عالية الحجم: `{vol_analysis['high_volume_trades']}`\n"
        message += f"• متوسط نسبة الحجم: `{vol_analysis['avg_volume_ratio']:.2f}x`\n"
        message += f"• ارتباط الحجم بالربح: `{vol_analysis['volume_correlation']:.3f}`\n"
        
        return message

    def _create_volume_performance_chart(self, df: pd.DataFrame, backtest_result: BacktestResult) -> BytesIO:
        """إنشاء رسم بياني للأداء"""
        try:
            fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
            fig.suptitle(f'تحليل استراتيجية الانزياح الحجمي - {SYMBOL}', 
                        fontsize=16, fontname='DejaVu Sans', fontweight='bold')
            
            # 1. السعر والحجم
            ax1.plot(df['timestamp'], df['close'], label='السعر', linewidth=1.5, color='blue', alpha=0.8)
            ax1.set_title('حركة السعر والحجم', fontname='DejaVu Sans', fontsize=12)
            ax1.set_ylabel('السعر (USDT)', fontname='DejaVu Sans', color='blue')
            ax1.tick_params(axis='y', labelcolor='blue')
            
            ax1_vol = ax1.twinx()
            ax1_vol.plot(df['timestamp'], df['volume'], label='الحجم', linewidth=1, color='orange', alpha=0.6)
            ax1_vol.set_ylabel('الحجم', fontname='DejaVu Sans', color='orange')
            ax1_vol.tick_params(axis='y', labelcolor='orange')
            
            # إضافة نقاط الدخول
            trades_df = pd.DataFrame(self.trade_history)
            for _, trade in trades_df.iterrows():
                color = 'green' if trade['direction'] == 'BUY' else 'red'
                marker = '^' if trade['direction'] == 'BUY' else 'v'
                ax1.scatter(trade['entry_time'], trade['entry_price'], 
                           color=color, marker=marker, s=80, alpha=0.8,
                           edgecolors='black', linewidth=0.5)
            
            # 2. توزيع الأرباح حسب نوع الانزياح
            if not trades_df.empty:
                divergence_colors = {
                    'positive_bullish': 'green',
                    'negative_bearish': 'red',
                    'volume_confirmation': 'blue', 
                    'hidden_divergence': 'purple'
                }
                
                for div_type, color in divergence_colors.items():
                    div_trades = trades_df[trades_df['divergence_type'] == div_type]
                    if len(div_trades) > 0:
                        ax2.hist(div_trades['pnl'], bins=10, alpha=0.6, color=color,
                                label=div_type, edgecolor='black')
                
                ax2.axvline(0, color='black', linestyle='--', linewidth=2)
                ax2.set_title('توزيع الأرباح حسب الانزياح', fontname='DejaVu Sans', fontsize=12)
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
                
                ax3.set_title('تطور الرصيد', fontname='DejaVu Sans', fontsize=12)
                ax3.set_xlabel('عدد الصفقات', fontname='DejaVu Sans')
                ax3.set_ylabel('الرصيد (USD)', fontname='DejaVu Sans')
                ax3.legend(prop={'family': 'DejaVu Sans'})
                ax3.grid(True, alpha=0.3)
            
            # 4. مقارنة أداء الانزياح
            div_analysis = backtest_result.divergence_analysis
            if div_analysis:
                div_types = list(div_analysis.keys())
                win_rates = [div_analysis[div]['win_rate'] for div in div_types]
                avg_pnls = [div_analysis[div]['avg_pnl'] for div in div_types]
                
                x = np.arange(len(div_types))
                width = 0.35
                
                ax4.bar(x - width/2, win_rates, width, label='نسبة الربح %', alpha=0.7)
                ax4_twin = ax4.twinx()
                ax4_twin.bar(x + width/2, avg_pnls, width, label='متوسط الربح $', alpha=0.7, color='orange')
                
                ax4.set_title('مقارنة أداء أنواع الانزياح', fontname='DejaVu Sans', fontsize=12)
                ax4.set_xticks(x)
                ax4.set_xticklabels([d[:15] for d in div_types], fontname='DejaVu Sans', rotation=45)
                ax4.legend(prop={'family': 'DejaVu Sans'})
                ax4_twin.legend(prop={'family': 'DejaVu Sans'})
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
# الوظيفة الرئيسية
# =============================================================================

async def main():
    """الوظيفة الرئيسية لتشغيل استراتيجية الانزياح الحجمي"""
    
    logger.info("🚀 بدء تشغيل استراتيجية الانزياح الحجمي")
    
    # تهيئة نظام التلغرام
    telegram_notifier = TelegramNotifier(TELEGRAM_BOT_TOKEN, TELEGRAM_CHAT_ID)
    
    # جلب البيانات
    data_fetcher = DataFetcher()
    df = data_fetcher.fetch_historical_data(SYMBOL, TIMEFRAME, DATA_LIMIT)
    
    if df.empty:
        error_msg = "❌ فشل جلب البيانات. تأكد من اتصال الإنترنت وصحة اسم العملة."
        logger.error(error_msg)
        await telegram_notifier.send_message(error_msg)
        return
    
    # إرسال معلومات عن فترة البيانات
    data_info = f"📊 فترة البيانات: {len(df)} شمعة من {df['timestamp'].min().date()} إلى {df['timestamp'].max().date()}"
    logger.info(data_info)
    await telegram_notifier.send_message(data_info)
    
    # تشغيل استراتيجية الانزياح الحجمي
    strategy = VolumeDivergenceStrategy(telegram_notifier)
    
    # الباك-تستينغ
    backtest_result = strategy.run_volume_backtest(df)
    
    # إرسال التقرير إلى التلغرام
    await strategy.send_volume_telegram_report(backtest_result, df)
    
    # حفظ النتائج في ملف
    trades_df = pd.DataFrame(strategy.trade_history)
    if not trades_df.empty:
        filename = f'volume_divergence_trades_{SYMBOL}_{TIMEFRAME}.csv'
        trades_df.to_csv(filename, index=False)
        logger.info(f"💾 تم حفظ سجل الصفقات في {filename}")
    
    logger.info("✅ اكتمل تشغيل استراتيجية الانزياح الحجمي بنجاح")

if __name__ == "__main__":
    # تشغيل الوظيفة الرئيسية
    asyncio.run(main())
