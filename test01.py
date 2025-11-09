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

SYMBOL = os.getenv("TRADING_SYMBOL", "BTCUSDT")
TIMEFRAME = os.getenv("TRADING_TIMEFRAME", "1h")
STOP_LOSS_PERCENT = float(os.getenv("STOP_LOSS_PERCENT", "1.0"))
TAKE_PROFIT_PERCENT = float(os.getenv("TAKE_PROFIT_PERCENT", "2.0"))
TRADE_SIZE_USDT = float(os.getenv("TRADE_SIZE_USDT", "100"))
LEVERAGE = int(os.getenv("LEVERAGE", "10"))
INITIAL_BALANCE = float(os.getenv("INITIAL_BALANCE", "10000.0"))
CONFIDENCE_THRESHOLD = int(os.getenv("CONFIDENCE_THRESHOLD", "60"))

# إعدادات التلغرام
TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN", "")
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID", "")

# إعداد التسجيل
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger("EMA_RSI_MACD_Strategy")

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
    stop_loss: float = None
    take_profit: float = None
    status: str = "OPEN"  # OPEN, CLOSED, STOP_LOSS, TAKE_PROFIT

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
    
    def _escape_markdown(self, text: str) -> str:
        """هروب الأحرف الخاصة في Markdown"""
        escape_chars = r'_*[]()~`>#+-=|{}.!'
        for char in escape_chars:
            text = text.replace(char, f'\\{char}')
        return text

# =============================================================================
# محرك الاستراتيجية
# =============================================================================

class EmaRsiMacdStrategy:
    """استراتيجية المتوسطات المتحركة + RSI + MACD مع باك-تستينغ وتداول ورقي"""
    
    def __init__(self, telegram_notifier: TelegramNotifier = None):
        self.name = "ema_rsi_macd"
        self.trades: List[Trade] = []
        self.balance = INITIAL_BALANCE
        self.current_balance = INITIAL_BALANCE
        self.positions = {}
        self.trade_history = []
        self.analysis_results = []
        self.telegram_notifier = telegram_notifier
    
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
        
        # تحديد ترتيب المتوسطات
        conditions = [
            (df['ema_9'] > df['ema_21']) & (df['ema_21'] > df['ema_50']),
            (df['ema_9'] < df['ema_21']) & (df['ema_21'] < df['ema_50'])
        ]
        choices = ['صاعد', 'هابط']
        df['ma_order'] = np.select(conditions, choices, default='متذبذب')
        
        # حساب قوة الاتجاه
        df['distance_9_21'] = (df['ema_9'] - df['ema_21']).abs() / df['close']
        df['distance_21_50'] = (df['ema_21'] - df['ema_50']).abs() / df['close']
        
        conditions_strength = [
            (df['distance_9_21'] > 0.03) & (df['distance_21_50'] > 0.04),
            (df['distance_9_21'] > 0.02) & (df['distance_21_50'] > 0.025),
            (df['distance_9_21'] > 0.01) & (df['distance_21_50'] > 0.015)
        ]
        choices_strength = [10, 8, 6]
        df['trend_strength'] = np.select(conditions_strength, choices_strength, default=4)
        
        return df
    
    def enhanced_ema_rsi_macd_scoring(self, df: pd.DataFrame) -> pd.DataFrame:
        """نظام التقييم المحسن 0-100"""
        
        # 1. تحليل المتوسطات المتحركة (25 نقطة كحد أقصى)
        conditions_ma = [
            (df['ma_order'] == 'صاعد') & (df['close'] > df['ema_21']) & (df['close'] > df['ema_50']),
            (df['ma_order'] == 'هابط') & (df['close'] < df['ema_21']) & (df['close'] < df['ema_50']),
            (df['ma_order'] == 'صاعد'),
            (df['ma_order'] == 'هابط')
        ]
        choices_ma = [
            np.minimum(25, df['trend_strength'] * 2.5),
            np.minimum(25, df['trend_strength'] * 2.5),
            np.minimum(15, df['trend_strength'] * 1.5),
            np.minimum(15, df['trend_strength'] * 1.5)
        ]
        df['ma_score'] = np.select(conditions_ma, choices_ma, default=0)
        
        # 2. تحليل RSI (40 نقطة كحد أقصى)
        conditions_rsi = [
            df['rsi'] <= 20,
            df['rsi'] <= 30,
            df['rsi'] >= 80,
            df['rsi'] >= 70,
            (df['rsi'] >= 40) & (df['rsi'] <= 60),
            (df['rsi'] >= 35) & (df['rsi'] <= 65)
        ]
        choices_rsi = [
            38 + (20 - df['rsi']) * 0.4,  # 38-42
            30 + (30 - df['rsi']) * 0.4,  # 30-34
            38 + (df['rsi'] - 80) * 0.4,  # 38-42
            30 + (df['rsi'] - 70) * 0.4,  # 30-34
            25 - (df['rsi'] - 50).abs() * 0.5,  # 20-25
            15 + (15 - (df['rsi'] - 50).abs() * 0.3)  # 12-15
        ]
        df['rsi_score'] = np.select(conditions_rsi, choices_rsi, default=8)
        df['rsi_score'] = df['rsi_score'].clip(0, 40)
        
        # 3. تحليل MACD (35 نقطة كحد أقصى)
        macd_positive = (df['macd_histogram'] > 0) & (df['macd_line'] > df['macd_signal'])
        macd_negative = (df['macd_histogram'] < 0) & (df['macd_line'] < df['macd_signal'])
        histogram_strength = df['macd_histogram'].abs()
        
        conditions_macd = [
            macd_positive & (histogram_strength > 0.005),
            macd_positive & (histogram_strength > 0.002),
            macd_positive,
            macd_negative & (histogram_strength > 0.005),
            macd_negative & (histogram_strength > 0.002),
            macd_negative
        ]
        choices_macd = [
            np.minimum(35, 25 + (histogram_strength * 1500)),
            np.minimum(35, 20 + (histogram_strength * 1200)),
            np.minimum(35, 15 + (histogram_strength * 1000)),
            np.minimum(35, 25 + (histogram_strength * 1500)),
            np.minimum(35, 20 + (histogram_strength * 1200)),
            np.minimum(35, 15 + (histogram_strength * 1000))
        ]
        df['macd_score'] = np.select(conditions_macd, choices_macd, default=0)
        
        # النتيجة النهائية
        df['total_score'] = df['ma_score'] + df['rsi_score'] + df['macd_score']
        df['total_score'] = df['total_score'].clip(0, 100)
        
        return df
    
    def generate_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        """توليد إشارات التداول"""
        # حساب المؤشرات
        df['rsi'] = self.calculate_rsi(df['close'])
        macd_line, signal_line, histogram = self.calculate_macd(df['close'])
        df['macd_line'] = macd_line
        df['macd_signal'] = signal_line
        df['macd_histogram'] = histogram
        
        # تحليل الاتجاه والتقييم
        df = self.analyze_trend(df)
        df = self.enhanced_ema_rsi_macd_scoring(df)
        
        # توليد الإشارات
        buy_condition = (
            (df['total_score'] >= CONFIDENCE_THRESHOLD) &
            ((df['ma_order'] == 'صاعد') | ((df['ma_order'] == 'متذبذب') & (df['close'] > df['ema_50']))) &
            (df['rsi'] <= 65) &
            (df['macd_histogram'] >= -0.001)
        )
        
        sell_condition = (
            (df['total_score'] >= CONFIDENCE_THRESHOLD) &
            ((df['ma_order'] == 'هابط') | ((df['ma_order'] == 'متذبذب') & (df['close'] < df['ema_50']))) &
            (df['rsi'] >= 35) &
            (df['macd_histogram'] <= 0.001)
        )
        
        df['signal'] = 'none'
        df.loc[buy_condition, 'signal'] = 'LONG'
        df.loc[sell_condition, 'signal'] = 'SHORT'
        
        # حفظ نتائج التحليل
        self.analysis_results = df.to_dict('records')
        
        return df
    
    # =========================================================================
    # نظام التداول الورقي (Paper Trading)
    # =========================================================================
    
    def calculate_position_size(self, price: float) -> float:
        """حساب حجم المركز بناء على الرافعة وحجم الصفقة"""
        return (TRADE_SIZE_USDT * LEVERAGE) / price
    
    def open_position(self, symbol: str, direction: str, price: float, 
                     confidence: float, timestamp: datetime) -> Optional[Trade]:
        """فتح مركز جديد"""
        
        if symbol in self.positions:
            logger.warning(f"يوجد مركز مفتوح بالفعل لـ {symbol}")
            return None
        
        # حساب حجم المركز
        quantity = self.calculate_position_size(price)
        
        # حساب وقف الخسارة وجني الأرباح
        if direction == "LONG":
            stop_loss = price * (1 - STOP_LOSS_PERCENT / 100)
            take_profit = price * (1 + TAKE_PROFIT_PERCENT / 100)
        else:  # SHORT
            stop_loss = price * (1 + STOP_LOSS_PERCENT / 100)
            take_profit = price * (1 - TAKE_PROFIT_PERCENT / 100)
        
        # رسوم التداول (افتراضي 0.04% لكل من الدخول والخروج)
        fee = (TRADE_SIZE_USDT * LEVERAGE) * 0.0004
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
            status="OPEN"
        )
        
        self.positions[symbol] = trade
        self.trades.append(trade)
        
        logger.info(f"📈 فتح مركز {direction} لـ {symbol} "
                   f"السعر: {price:.2f}, الكمية: {quantity:.6f}, "
                   f"الثقة: {confidence:.1f}%")
        
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
            'status': trade.status
        })
        
        logger.info(f"📊 إغلاق مركز {trade.direction} لـ {symbol} "
                   f"السعر: {price:.2f}, الربح: {pnl:.2f} USD ({pnl_percent:.2f}%)")
        
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
    
    def execute_paper_trading(self, df: pd.DataFrame):
        """تنفيذ التداول الورقي على البيانات"""
        
        logger.info("🚀 بدء التداول الورقي...")
        
        for i, row in df.iterrows():
            current_price = row['close']
            signal = row['signal']
            confidence = row['total_score']
            timestamp = row['timestamp']
            
            # فحص شروط الخروج للمراكز المفتوحة
            if SYMBOL in self.positions:
                self.check_stop_conditions(SYMBOL, current_price, timestamp)
            
            # فتح مراكز جديدة إذا لم يكن هناك مركز مفتوح
            if (SYMBOL not in self.positions and signal != 'none' and 
                confidence >= CONFIDENCE_THRESHOLD):
                
                self.open_position(SYMBOL, signal, current_price, confidence, timestamp)
    
    # =========================================================================
    # الباك-تستينغ (Backtesting)
    # =========================================================================
    
    def run_backtest(self, df: pd.DataFrame) -> BacktestResult:
        """تشغيل الباك-تستينغ الكامل"""
        
        logger.info("🔍 بدء الباك-تستينغ...")
        
        # إعادة تعيين البيانات
        self.trades = []
        self.positions = {}
        self.trade_history = []
        self.current_balance = INITIAL_BALANCE
        
        # توليد الإشارات
        df_with_signals = self.generate_signals(df)
        
        # تنفيذ التداول
        self.execute_paper_trading(df_with_signals)
        
        # إغلاق أي مراكز مفتوحة في النهاية
        if SYMBOL in self.positions:
            last_price = df_with_signals.iloc[-1]['close']
            last_timestamp = df_with_signals.iloc[-1]['timestamp']
            self.close_position(SYMBOL, last_price, last_timestamp, "END_OF_DATA")
        
        # حساب النتائج
        return self.calculate_backtest_results(df)
    
    def calculate_backtest_results(self, df: pd.DataFrame) -> BacktestResult:
        """حساب نتائج الباك-تستينغ"""
        
        if not self.trade_history:
            total_days = (df['timestamp'].max() - df['timestamp'].min()).days
            return BacktestResult(
                total_trades=0, winning_trades=0, losing_trades=0,
                win_rate=0, total_pnl=0, final_balance=self.current_balance,
                max_drawdown=0, sharpe_ratio=0, profit_factor=0,
                avg_trade=0, best_trade=0, worst_trade=0, total_fees=0,
                total_days=max(1, total_days), avg_daily_return=0
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
        
        # نسبة شارب (مبسطة)
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
            avg_daily_return=avg_daily_return
        )
    
    # =========================================================================
    # التقارير والرسوم البيانية
    # =========================================================================
    
    async def send_telegram_report(self, backtest_result: BacktestResult, df: pd.DataFrame):
        """إرسال تقرير مفصل إلى التلغرام"""
        
        if not self.telegram_notifier:
            logger.warning("❌ نظام التلغرام غير متوفر")
            return
        
        try:
            # 1. إرسال التقرير النصي
            report_text = self._generate_report_text(backtest_result)
            await self.telegram_notifier.send_message(report_text)
            
            # 2. إرسال الرسوم البيانية
            chart_buffer = self._create_performance_chart(df, backtest_result)
            if chart_buffer:
                chart_caption = f"📈 تحليل أداء الاستراتيجية - {SYMBOL} ({TIMEFRAME})"
                await self.telegram_notifier.send_photo(chart_buffer, chart_caption)
            
            # 3. إرسال تحليل الصفقات إذا وجدت
            if self.trade_history:
                trades_analysis = self._generate_trades_analysis()
                await self.telegram_notifier.send_message(trades_analysis)
                
        except Exception as e:
            logger.error(f"❌ خطأ في إرسال التقرير إلى التلغرام: {e}")
    
    def _generate_report_text(self, backtest_result: BacktestResult) -> str:
        """إنشاء نص التقرير للتلغرام"""
        
        report_time = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        
        message = f"📊 *تقرير أداء استراتيجية المتوسطات + RSI + MACD*\n"
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
        message += f"• العائد الإجمالي: `{((backtest_result.final_balance - INITIAL_BALANCE) / INITIAL_BALANCE * 100):.1f}%`\n\n"
        
        message += f"📊 *مقاييس المخاطرة:*\n"
        message += f"• أقصى خسارة: `{backtest_result.max_drawdown:.1f}%`\n"
        message += f"• متوسط الربح/صفقة: `${backtest_result.avg_trade:.2f}`\n"
        message += f"• أفضل صفقة: `${backtest_result.best_trade:.2f}`\n"
        message += f"• أسوأ صفقة: `${backtest_result.worst_trade:.2f}`\n"
        message += f"• نسبة شارب: `{backtest_result.sharpe_ratio:.2f}`\n"
        message += f"• عامل الربحية: `{backtest_result.profit_factor:.2f}`\n\n"
        
        message += f"⏰ *الفترة الزمنية:*\n"
        message += f"• إجمالي الأيام: `{backtest_result.total_days}`\n"
        message += f"• متوسط العائد اليومي: `{backtest_result.avg_daily_return:.2f}%`\n\n"
        
        message += f"🕒 *وقت التقرير:* `{report_time}`\n"
        message += "══════════════════════════════════════\n"
        message += "⚡ *نظام التقييم المحسن 0-100 نقطة*"
        
        return message
    
    def _generate_trades_analysis(self) -> str:
        """إنشاء تحليل مفصل للصفقات"""
        
        trades_df = pd.DataFrame(self.trade_history)
        
        message = "🔍 *تحليل مفصل للصفقات:*\n"
        message += "────────────────────\n"
        
        # تحليل حسب الاتجاه
        long_trades = trades_df[trades_df['direction'] == 'LONG']
        short_trades = trades_df[trades_df['direction'] == 'SHORT']
        
        message += f"🔼 *صفقات الشراء:* `{len(long_trades)}`\n"
        if len(long_trades) > 0:
            message += f"   - متوسط الربح: `${long_trades['pnl'].mean():.2f}`\n"
            message += f"   - إجمالي الربح: `${long_trades['pnl'].sum():.2f}`\n"
        
        message += f"🔽 *صفقات البيع:* `{len(short_trades)}`\n"
        if len(short_trades) > 0:
            message += f"   - متوسط الربح: `${short_trades['pnl'].mean():.2f}`\n"
            message += f"   - إجمالي الربح: `${short_trades['pnl'].sum():.2f}`\n\n"
        
        # تحليل حسب مستوى الثقة
        high_conf = trades_df[trades_df['confidence'] >= 80]
        med_conf = trades_df[(trades_df['confidence'] >= 60) & (trades_df['confidence'] < 80)]
        low_conf = trades_df[trades_df['confidence'] < 60]
        
        message += f"🎯 *التحليل حسب الثقة:*\n"
        if len(high_conf) > 0:
            win_rate_high = (len(high_conf[high_conf['pnl'] > 0]) / len(high_conf)) * 100
            message += f"• عالية (80%+): `{len(high_conf)}` صفقة - ربح: `${high_conf['pnl'].sum():.2f}` - دقة: `{win_rate_high:.1f}%`\n"
        
        if len(med_conf) > 0:
            win_rate_med = (len(med_conf[med_conf['pnl'] > 0]) / len(med_conf)) * 100
            message += f"• متوسطة (60-80%): `{len(med_conf)}` صفقة - ربح: `${med_conf['pnl'].sum():.2f}` - دقة: `{win_rate_med:.1f}%`\n"
        
        if len(low_conf) > 0:
            win_rate_low = (len(low_conf[low_conf['pnl'] > 0]) / len(low_conf)) * 100
            message += f"• منخفضة (<60%): `{len(low_conf)}` صفقة - ربح: `${low_conf['pnl'].sum():.2f}` - دقة: `{win_rate_low:.1f}%`\n"
        
        return message
    
    def _create_performance_chart(self, df: pd.DataFrame, backtest_result: BacktestResult) -> BytesIO:
        """إنشاء رسم بياني للأداء"""
        try:
            fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
            fig.suptitle(f'تحليل استراتيجية المتوسطات + RSI + MACD - {SYMBOL}', fontsize=14, fontname='DejaVu Sans')
            
            # 1. السعر والإشارات
            ax1.plot(df['timestamp'], df['close'], label='السعر', linewidth=1, color='blue')
            ax1.set_title('حركة السعر وإشارات التداول', fontname='DejaVu Sans')
            ax1.set_ylabel('السعر (USDT)', fontname='DejaVu Sans')
            ax1.legend(prop={'family': 'DejaVu Sans'})
            ax1.grid(True, alpha=0.3)
            
            # إضافة نقاط الدخول
            trades_df = pd.DataFrame(self.trade_history)
            for _, trade in trades_df.iterrows():
                color = 'green' if trade['direction'] == 'LONG' else 'red'
                marker = '^' if trade['direction'] == 'LONG' else 'v'
                ax1.scatter(trade['entry_time'], trade['entry_price'], 
                           color=color, marker=marker, s=80, alpha=0.7)
            
            # 2. المؤشرات الفنية
            ax2.plot(df['timestamp'], df['ema_9'], label='EMA 9', alpha=0.8, linewidth=1)
            ax2.plot(df['timestamp'], df['ema_21'], label='EMA 21', alpha=0.8, linewidth=1)
            ax2.plot(df['timestamp'], df['ema_50'], label='EMA 50', alpha=0.8, linewidth=1)
            ax2.set_title('المتوسطات المتحركة', fontname='DejaVu Sans')
            ax2.legend(prop={'family': 'DejaVu Sans'})
            ax2.grid(True, alpha=0.3)
            
            # 3. توزيع الأرباح
            if not trades_df.empty:
                profits = trades_df['pnl']
                ax3.hist(profits, bins=15, alpha=0.7, color='skyblue', edgecolor='black')
                ax3.axvline(0, color='red', linestyle='--', linewidth=2, label='نقطة التعادل')
                ax3.set_title('توزيع أرباح الصفقات', fontname='DejaVu Sans')
                ax3.set_xlabel('الربح (USD)', fontname='DejaVu Sans')
                ax3.set_ylabel('عدد الصفقات', fontname='DejaVu Sans')
                ax3.legend(prop={'family': 'DejaVu Sans'})
                ax3.grid(True, alpha=0.3)
            
            # 4. أداء الرصيد
            if len(self.trade_history) > 0:
                balance_history = [INITIAL_BALANCE]
                for trade in self.trade_history:
                    balance_history.append(balance_history[-1] + trade['pnl'])
                
                ax4.plot(range(len(balance_history)), balance_history, 
                        color='green', linewidth=2, label='الرصيد')
                ax4.axhline(INITIAL_BALANCE, color='red', linestyle='--', alpha=0.7, label='رصيد البداية')
                ax4.set_title('تطور الرصيد', fontname='DejaVu Sans')
                ax4.set_xlabel('عدد الصفقات', fontname='DejaVu Sans')
                ax4.set_ylabel('الرصيد (USD)', fontname='DejaVu Sans')
                ax4.legend(prop={'family': 'DejaVu Sans'})
                ax4.grid(True, alpha=0.3)
            
            plt.tight_layout()
            
            # حفظ في buffer
            buffer = BytesIO()
            plt.savefig(buffer, format='png', dpi=150, bbox_inches='tight')
            buffer.seek(0)
            plt.close()
            
            return buffer
            
        except Exception as e:
            logger.error(f"❌ خطأ في إنشاء الرسم البياني: {e}")
            return None

# =============================================================================
# دعم获取 البيانات
# =============================================================================

class DataFetcher:
    """جلب البيانات من Binance"""
    
    @staticmethod
    def fetch_historical_data(symbol: str, interval: str, limit: int = 1000) -> pd.DataFrame:
        """جلب البيانات التاريخية من Binance"""
        try:
            url = f"https://api.binance.com/api/v3/klines"
            params = {
                'symbol': symbol,
                'interval': interval,
                'limit': limit
            }
            
            response = requests.get(url, params=params, timeout=10)
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
    """الوظيفة الرئيسية لتشغيل الاستراتيجية"""
    
    logger.info("🚀 بدء تشغيل استراتيجية المتوسطات + RSI + MACD")
    
    # تهيئة نظام التلغرام
    telegram_notifier = TelegramNotifier(TELEGRAM_BOT_TOKEN, TELEGRAM_CHAT_ID)
    
    # جلب البيانات
    data_fetcher = DataFetcher()
    df = data_fetcher.fetch_historical_data(SYMBOL, TIMEFRAME, 1000)
    
    if df.empty:
        error_msg = "❌ فشل جلب البيانات. تأكد من اتصال الإنترنت وصحة اسم العملة."
        logger.error(error_msg)
        await telegram_notifier.send_message(error_msg)
        return
    
    # تشغيل الاستراتيجية
    strategy = EmaRsiMacdStrategy(telegram_notifier)
    
    # الباك-تستينغ
    backtest_result = strategy.run_backtest(df)
    
    # إرسال التقرير إلى التلغرام
    await strategy.send_telegram_report(backtest_result, df)
    
    # حفظ النتائج في ملف
    trades_df = pd.DataFrame(strategy.trade_history)
    if not trades_df.empty:
        trades_df.to_csv(f'trades_history_{SYMBOL}_{TIMEFRAME}.csv', index=False)
        logger.info(f"💾 تم حفظ سجل الصفقات في trades_history_{SYMBOL}_{TIMEFRAME}.csv")
    
    logger.info("✅ اكتمل تشغيل الاستراتيجية بنجاح")

if __name__ == "__main__":
    # تشغيل الوظيفة الرئيسية
    asyncio.run(main())
