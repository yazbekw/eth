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
# إعدادات التداول المحسنة - إصدار قراءة السوق
# =============================================================================

SYMBOL = os.getenv("TRADING_SYMBOL", "BNBUSDT")
TIMEFRAME = os.getenv("TRADING_TIMEFRAME", "1h")
STOP_LOSS_PERCENT = float(os.getenv("STOP_LOSS_PERCENT", "0.5"))
TAKE_PROFIT_PERCENT = float(os.getenv("TAKE_PROFIT_PERCENT", "3.5"))
TRADE_SIZE_USDT = float(os.getenv("TRADE_SIZE_USDT", "100"))
LEVERAGE = int(os.getenv("LEVERAGE", "8"))
INITIAL_BALANCE = float(os.getenv("INITIAL_BALANCE", "5000.0"))

# عتبات مرنة تركز على جودة الإشارة
BUY_CONFIDENCE_THRESHOLD = int(os.getenv("BUY_CONFIDENCE_THRESHOLD", "65"))
SELL_CONFIDENCE_THRESHOLD = int(os.getenv("SELL_CONFIDENCE_THRESHOLD", "65"))

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
logger = logging.getLogger("Market_Reader_Strategy")

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
    signal_type: str = ""  # RESISTANCE, SUPPORT, BREAKOUT, etc.
    market_condition: str = ""
    volume_ratio: float = 0
    quality_score: float = 0
    trend_strength: float = 0

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
    market_analysis: Dict
    performance_metrics: Dict
    signal_analysis: Dict

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
# استراتيجية قراءة السوق - بسيطة ومركزة
# =============================================================================

class MarketReaderStrategy:
    """استراتيجية تركز على قراءة السوق بشكل بسيط وفعال"""
    
    def __init__(self, telegram_notifier: TelegramNotifier = None):
        self.name = "market_reader_strategy"
        self.trades: List[Trade] = []
        self.balance = INITIAL_BALANCE
        self.current_balance = INITIAL_BALANCE
        self.positions = {}
        self.trade_history = []
        self.analysis_results = []
        self.telegram_notifier = telegram_notifier
        self.df_global = None
        
        # إحصائيات الإشارات
        self.signal_stats = {
            'resistance_sell': {'trades': 0, 'wins': 0, 'total_pnl': 0},
            'support_buy': {'trades': 0, 'wins': 0, 'total_pnl': 0},
            'breakout_buy': {'trades': 0, 'wins': 0, 'total_pnl': 0},
            'breakdown_sell': {'trades': 0, 'wins': 0, 'total_pnl': 0}
        }
    
    def analyze_market_condition(self, df: pd.DataFrame, lookback_period: int = 100) -> Dict[str, Any]:
        """تحليل حالة السوق بشكل بسيط وواضح"""
        if len(df) < lookback_period:
            lookback_period = len(df)
        
        recent_data = df.tail(lookback_period)
        
        # الاتجاه الأساسي
        price_change = (recent_data['close'].iloc[-1] - recent_data['close'].iloc[0]) / recent_data['close'].iloc[0] * 100
        
        if price_change > 2:
            trend = "UPTREND"
        elif price_change < -2:
            trend = "DOWNTREND"
        else:
            trend = "SIDEWAYS"
        
        # الدعم والمقاومة البسيطة
        resistance_level = recent_data['high'].max()
        support_level = recent_data['low'].min()
        current_price = recent_data['close'].iloc[-1]
        
        # قرب من الدعم أو المقاومة
        near_resistance = current_price >= resistance_level * 0.985  # ضمن 1.5% من المقاومة
        near_support = current_price <= support_level * 1.015       # ضمن 1.5% من الدعم
        
        # قوة الاتجاه
        volatility = recent_data['close'].pct_change().std() * 100
        
        return {
            'trend': trend,
            'trend_strength': abs(price_change),
            'resistance_level': resistance_level,
            'support_level': support_level,
            'near_resistance': near_resistance,
            'near_support': near_support,
            'volatility': 'HIGH' if volatility > 1.5 else 'LOW',
            'current_price': current_price,
            'price_change_percent': price_change
        }
    
    def calculate_support_resistance(self, df: pd.DataFrame, window: int = 20) -> pd.DataFrame:
        """حساب مستويات الدعم والمقاومة البسيطة"""
        df['resistance'] = df['high'].rolling(window=window, min_periods=1).max()
        df['support'] = df['low'].rolling(window=window, min_periods=1).min()
        df['price_position'] = (df['close'] - df['support']) / (df['resistance'] - df['support'])
        return df
    
    def detect_support_buy_signal(self, df: pd.DataFrame, current_index: int, market_condition: Dict) -> Tuple[bool, float]:
        """كشف إشارات الشراء عند الدعم"""
        if current_index < 10:
            return False, 0
        
        current_row = df.iloc[current_index]
        prev_row = df.iloc[current_index-1]
        
        # الشرط الأساسي: قرب الدعم + شمعة صاعدة
        near_support = market_condition['near_support']
        bullish_candle = current_row['close'] > current_row['open']
        volume_surge = current_row['volume'] > df['volume'].rolling(10).mean().iloc[current_index] * 1.2
        
        if near_support and bullish_candle and volume_surge:
            # حساب الثقة
            confidence = 70  # أساسي
            
            # تحسين الثقة حسب الظروف
            if market_condition['trend'] == "UPTREND":
                confidence += 10
            if volume_surge:
                confidence += 10
            if current_row['close'] > prev_row['close']:
                confidence += 5
            
            return True, min(95, confidence)
        
        return False, 0
    
    def detect_resistance_sell_signal(self, df: pd.DataFrame, current_index: int, market_condition: Dict) -> Tuple[bool, float]:
        """كشف إشارات البيع عند المقاومة"""
        if current_index < 10:
            return False, 0
        
        current_row = df.iloc[current_index]
        prev_row = df.iloc[current_index-1]
        
        # الشرط الأساسي: قرب المقاومة + شمعة هابطة
        near_resistance = market_condition['near_resistance']
        bearish_candle = current_row['close'] < current_row['open']
        volume_surge = current_row['volume'] > df['volume'].rolling(10).mean().iloc[current_index] * 1.2
        
        if near_resistance and bearish_candle and volume_surge:
            # حساب الثقة
            confidence = 70  # أساسي
            
            # تحسين الثقة حسب الظروف
            if market_condition['trend'] == "DOWNTREND":
                confidence += 10
            if volume_surge:
                confidence += 10
            if current_row['close'] < prev_row['close']:
                confidence += 5
            
            return True, min(95, confidence)
        
        return False, 0
    
    def detect_breakout_buy_signal(self, df: pd.DataFrame, current_index: int, market_condition: Dict) -> Tuple[bool, float]:
        """كشف إشارات الشراء عند كسر المقاومة"""
        if current_index < 10:
            return False, 0
        
        current_row = df.iloc[current_index]
        resistance_level = market_condition['resistance_level']
        
        # كسر المقاومة مع تأكيد
        breakout = current_row['close'] > resistance_level
        high_volume = current_row['volume'] > df['volume'].rolling(20).mean().iloc[current_index] * 1.5
        strong_move = current_row['close'] > current_row['open'] * 1.01  # صعود 1% على الأقل
        
        if breakout and high_volume and strong_move:
            confidence = 75
            if market_condition['trend'] == "UPTREND":
                confidence += 10
            if high_volume:
                confidence += 10
            
            return True, min(95, confidence)
        
        return False, 0
    
    def detect_breakdown_sell_signal(self, df: pd.DataFrame, current_index: int, market_condition: Dict) -> Tuple[bool, float]:
        """كشف إشارات البيع عند كسر الدعم"""
        if current_index < 10:
            return False, 0
        
        current_row = df.iloc[current_index]
        support_level = market_condition['support_level']
        
        # كسر الدعم مع تأكيد
        breakdown = current_row['close'] < support_level
        high_volume = current_row['volume'] > df['volume'].rolling(20).mean().iloc[current_index] * 1.5
        strong_move = current_row['close'] < current_row['open'] * 0.99  # هبوط 1% على الأقل
        
        if breakdown and high_volume and strong_move:
            confidence = 75
            if market_condition['trend'] == "DOWNTREND":
                confidence += 10
            if high_volume:
                confidence += 10
            
            return True, min(95, confidence)
        
        return False, 0
    
    def calculate_volume_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """حساب مؤشرات الحجم البسيطة"""
        df['volume_ma_10'] = df['volume'].rolling(10, min_periods=1).mean()
        df['volume_ma_20'] = df['volume'].rolling(20, min_periods=1).mean()
        df['volume_ratio_10'] = df['volume'] / df['volume_ma_10'].replace(0, 1)
        df['volume_ratio_20'] = df['volume'] / df['volume_ma_20'].replace(0, 1)
        return df
    
    def generate_market_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        """توليد إشارات بناءً على قراءة السوق"""
        
        buy_signals = []
        sell_signals = []
        buy_confidence_scores = []
        sell_confidence_scores = []
        signal_types = []
        market_conditions = []
        
        for i in range(len(df)):
            try:
                if i < 50:  # تحتاج بيانات كافية للتحليل
                    buy_signals.append('none')
                    sell_signals.append('none')
                    buy_confidence_scores.append(0)
                    sell_confidence_scores.append(0)
                    signal_types.append('none')
                    market_conditions.append('none')
                    continue
                
                # تحليل حالة السوق
                market_data = df.iloc[:i+1]
                market_condition = self.analyze_market_condition(market_data)
                market_conditions.append(market_condition['trend'])
                
                # البحث عن إشارات
                current_signal_type = 'none'
                buy_signal = 'none'
                sell_signal = 'none'
                buy_confidence = 0
                sell_confidence = 0
                
                # 1. إشارات الشراء
                support_buy, support_confidence = self.detect_support_buy_signal(df, i, market_condition)
                breakout_buy, breakout_confidence = self.detect_breakout_buy_signal(df, i, market_condition)
                
                if support_buy and support_confidence >= BUY_CONFIDENCE_THRESHOLD:
                    buy_signal = "BUY"
                    buy_confidence = support_confidence
                    current_signal_type = "SUPPORT_BUY"
                elif breakout_buy and breakout_confidence >= BUY_CONFIDENCE_THRESHOLD:
                    buy_signal = "BUY"
                    buy_confidence = breakout_confidence
                    current_signal_type = "BREAKOUT_BUY"
                
                # 2. إشارات البيع
                resistance_sell, resistance_confidence = self.detect_resistance_sell_signal(df, i, market_condition)
                breakdown_sell, breakdown_confidence = self.detect_breakdown_sell_signal(df, i, market_condition)
                
                if resistance_sell and resistance_confidence >= SELL_CONFIDENCE_THRESHOLD:
                    sell_signal = "SELL"
                    sell_confidence = resistance_confidence
                    current_signal_type = "RESISTANCE_SELL"
                elif breakdown_sell and breakdown_confidence >= SELL_CONFIDENCE_THRESHOLD:
                    sell_signal = "SELL"
                    sell_confidence = breakdown_confidence
                    current_signal_type = "BREAKDOWN_SELL"
                
                buy_signals.append(buy_signal)
                sell_signals.append(sell_signal)
                buy_confidence_scores.append(buy_confidence)
                sell_confidence_scores.append(sell_confidence)
                signal_types.append(current_signal_type)
                
            except Exception as e:
                logger.error(f"❌ خطأ في توليد الإشارات للمؤشر {i}: {e}")
                buy_signals.append('none')
                sell_signals.append('none')
                buy_confidence_scores.append(0)
                sell_confidence_scores.append(0)
                signal_types.append('none')
                market_conditions.append('none')
        
        df['buy_signal'] = buy_signals
        df['sell_signal'] = sell_signals
        df['buy_confidence'] = buy_confidence_scores
        df['sell_confidence'] = sell_confidence_scores
        df['signal_type'] = signal_types
        df['market_condition'] = market_conditions
        
        return df
    
    def calculate_position_size(self, price: float, confidence: float) -> float:
        """حساب حجم المركز بشكل بسيط"""
        base_size = (TRADE_SIZE_USDT * LEVERAGE) / price
        # تعديل بسيط حسب الثقة
        confidence_factor = confidence / 100
        adjusted_size = base_size * (0.7 + confidence_factor * 0.6)
        return adjusted_size
    
    def dynamic_risk_management(self, signal_type: str, confidence: float) -> Tuple[float, float]:
        """إدارة مخاطرة ديناميكية بسيطة"""
        
        base_sl = STOP_LOSS_PERCENT
        base_tp = TAKE_PROFIT_PERCENT
        
        # تعديلات بسيطة حسب نوع الإشارة
        if "BREAKOUT" in signal_type or "BREAKDOWN" in signal_type:
            # إشارات الكسر تحتاج مساحة أكبر
            sl_multiplier = 1.2
            tp_multiplier = 1.3
        elif "SUPPORT" in signal_type or "RESISTANCE" in signal_type:
            # إشارات الدعم/المقاومة أكثر دقة
            sl_multiplier = 0.8
            tp_multiplier = 1.1
        else:
            sl_multiplier = 1.0
            tp_multiplier = 1.0
        
        # تعديل طفيف حسب الثقة
        confidence_factor = confidence / 100
        sl_multiplier *= (1.1 - confidence_factor * 0.2)
        tp_multiplier *= (0.9 + confidence_factor * 0.2)
        
        return base_sl * sl_multiplier, base_tp * tp_multiplier
    
    def open_position(self, symbol: str, direction: str, price: float, 
                     confidence: float, signal_type: str, 
                     market_condition: str, volume_ratio: float,
                     timestamp: datetime) -> Optional[Trade]:
        """فتح مركز جديد"""
        
        if symbol in self.positions:
            return None
        
        # حساب حجم المركز
        quantity = self.calculate_position_size(price, confidence)
        
        # إدارة المخاطرة
        sl_percent, tp_percent = self.dynamic_risk_management(signal_type, confidence)
        
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
            stop_loss=stop_loss,
            take_profit=take_profit,
            status="OPEN",
            signal_type=signal_type,
            market_condition=market_condition,
            volume_ratio=volume_ratio,
            quality_score=confidence,  # استخدام الثقة كمؤشر للجودة
            trend_strength=0
        )
        
        self.positions[symbol] = trade
        self.trades.append(trade)
        
        logger.info(f"🎯 فتح مركز {direction} - {signal_type}")
        logger.info(f"   السعر: {price:.4f} | الثقة: {confidence:.1f}%")
        logger.info(f"   الوقف: {sl_percent:.1f}% | الجني: {tp_percent:.1f}%")
        
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
        else:
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
        
        # تحديث إحصائيات الإشارات
        signal_key = trade.signal_type.lower()
        if signal_key in self.signal_stats:
            stats = self.signal_stats[signal_key]
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
            'signal_type': trade.signal_type,
            'market_condition': trade.market_condition,
            'volume_ratio': trade.volume_ratio,
            'quality_score': trade.quality_score,
            'status': trade.status
        }
        
        if trade.quantity is not None:
            trade_record['quantity'] = trade.quantity
        
        self.trade_history.append(trade_record)
        
        status_emoji = "🟢" if pnl > 0 else "🔴"
        logger.info(f"📊 إغلاق مركز {trade.direction} - {trade.signal_type} {status_emoji}"
                   f" الربح: {pnl:.2f} USD ({pnl_percent:.2f}%)")
        
        return trade
    
    def check_stop_conditions(self, symbol: str, current_price: float, 
                            timestamp: datetime) -> bool:
        """فحص شروط الوقف والخروج"""
        
        if symbol not in self.positions:
            return False
        
        trade = self.positions[symbol]
        
        if ((trade.direction == "BUY" and current_price <= trade.stop_loss) or
            (trade.direction == "SELL" and current_price >= trade.stop_loss)):
            self.close_position(symbol, trade.stop_loss, timestamp, "STOP_LOSS")
            return True
        
        if ((trade.direction == "BUY" and current_price >= trade.take_profit) or
            (trade.direction == "SELL" and current_price <= trade.take_profit)):
            self.close_position(symbol, trade.take_profit, timestamp, "TAKE_PROFIT")
            return True
        
        return False
    
    def execute_market_trading(self, df: pd.DataFrame):
        """تنفيذ التداول بناءً على قراءة السوق"""
        
        logger.info("🚀 بدء التداول باستراتيجية قراءة السوق...")
        
        for i, row in df.iterrows():
            if i < 50:  # تحتاج بيانات كافية
                continue
                
            current_price = row['close']
            buy_signal = row['buy_signal']
            sell_signal = row['sell_signal']
            buy_confidence = row['buy_confidence']
            sell_confidence = row['sell_confidence']
            signal_type = row['signal_type']
            market_condition = row['market_condition']
            volume_ratio = row.get('volume_ratio_10', 1)
            timestamp = row['timestamp']
            
            # فحص شروط الخروج أولاً
            if SYMBOL in self.positions:
                self.check_stop_conditions(SYMBOL, current_price, timestamp)
            
            # فتح مراكز جديدة
            if SYMBOL not in self.positions:
                if sell_signal == "SELL":
                    self.open_position(
                        SYMBOL, "SELL", current_price, sell_confidence, signal_type,
                        market_condition, volume_ratio, timestamp
                    )
                elif buy_signal == "BUY":
                    self.open_position(
                        SYMBOL, "BUY", current_price, buy_confidence, signal_type,
                        market_condition, volume_ratio, timestamp
                    )
    
    def run_market_analysis(self, df: pd.DataFrame) -> pd.DataFrame:
        """تشغيل التحليل الشامل للسوق"""
        df = self.calculate_volume_indicators(df)
        df = self.calculate_support_resistance(df)
        df = self.generate_market_signals(df)
        self.analysis_results = df.to_dict('records')
        return df
    
    def calculate_market_results(self, df: pd.DataFrame) -> BacktestResult:
        """حساب نتائج استراتيجية قراءة السوق"""
        
        if not self.trade_history:
            total_days = (df['timestamp'].max() - df['timestamp'].min()).days
            return BacktestResult(
                total_trades=0, winning_trades=0, losing_trades=0,
                win_rate=0, total_pnl=0, final_balance=self.current_balance,
                max_drawdown=0, sharpe_ratio=0, profit_factor=0,
                avg_trade=0, best_trade=0, worst_trade=0, total_fees=0,
                total_days=max(1, total_days), avg_daily_return=0,
                avg_confidence=0, market_analysis={}, performance_metrics={},
                signal_analysis={}
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
                'signal_type': trade.get('signal_type', ''),
                'market_condition': trade.get('market_condition', ''),
                'volume_ratio': trade.get('volume_ratio', 0),
                'quality_score': trade.get('quality_score', 0)
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
        
        # تحليل الثقة
        avg_confidence = trades_df['confidence'].mean() if not trades_df.empty else 0
        
        # تحليل السوق
        market_analysis = {
            'total_period_days': total_days,
            'avg_daily_return': avg_daily_return,
            'market_trend_distribution': df['market_condition'].value_counts().to_dict(),
            'signal_frequency': df['signal_type'].value_counts().to_dict()
        }
        
        # تحليل الإشارات
        signal_analysis = {}
        for signal_type in self.signal_stats:
            stats = self.signal_stats[signal_type]
            signal_analysis[signal_type] = {
                'trades': stats['trades'],
                'win_rate': (stats['wins'] / stats['trades'] * 100) if stats['trades'] > 0 else 0,
                'total_pnl': stats['total_pnl'],
                'avg_pnl': stats['total_pnl'] / stats['trades'] if stats['trades'] > 0 else 0
            }
        
        # مقاييس الأداء
        performance_metrics = {
            'risk_reward_ratio': abs(avg_trade / worst_trade) if worst_trade < 0 else 0,
            'expectancy': (win_rate/100 * avg_trade) - ((1 - win_rate/100) * abs(avg_trade)),
            'consistency_score': (win_rate * profit_factor) / 100 if profit_factor != float('inf') else 0
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
            market_analysis=market_analysis,
            performance_metrics=performance_metrics,
            signal_analysis=signal_analysis
        )
    
    def run_market_backtest(self, df: pd.DataFrame) -> BacktestResult:
        """تشغيل الباك-تستينغ باستراتيجية قراءة السوق"""
        
        logger.info("🔍 بدء الباك-تستينغ باستراتيجية قراءة السوق...")
        
        # إعادة تعيين البيانات
        self.trades = []
        self.positions = {}
        self.trade_history = []
        self.current_balance = INITIAL_BALANCE
        self.signal_stats = {
            'resistance_sell': {'trades': 0, 'wins': 0, 'total_pnl': 0},
            'support_buy': {'trades': 0, 'wins': 0, 'total_pnl': 0},
            'breakout_buy': {'trades': 0, 'wins': 0, 'total_pnl': 0},
            'breakdown_sell': {'trades': 0, 'wins': 0, 'total_pnl': 0}
        }
        
        # حفظ البيانات العالمية
        self.df_global = df.copy()
        
        # تحليل السوق
        df_with_signals = self.run_market_analysis(df)
        
        # تنفيذ التداول
        self.execute_market_trading(df_with_signals)
        
        # إغلاق المراكز المفتوحة
        if SYMBOL in self.positions:
            last_price = df_with_signals.iloc[-1]['close']
            last_timestamp = df_with_signals.iloc[-1]['timestamp']
            self.close_position(SYMBOL, last_price, last_timestamp, "END_OF_DATA")
        
        return self.calculate_market_results(df_with_signals)
    
    async def send_market_report(self, backtest_result: BacktestResult):
        """إرسال تقرير قراءة السوق"""
        
        if not self.telegram_notifier:
            return
        
        try:
            report_text = self._generate_market_report_text(backtest_result)
            await self.telegram_notifier.send_message(report_text)
                
        except Exception as e:
            logger.error(f"❌ خطأ في إرسال التقرير: {e}")
    
    def _generate_market_report_text(self, backtest_result: BacktestResult) -> str:
        """إنشاء تقرير قراءة السوق"""
        
        report_time = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        
        message = f"📊 تقرير استراتيجية قراءة السوق\n"
        message += "══════════════════════════════════════\n\n"
        
        message += f"⚙️ الإعدادات البسيطة:\n"
        message += f"• العملة: `{SYMBOL}`\n"
        message += f"• الإطار: `{TIMEFRAME}`\n"
        message += f"• الرافعة: `{LEVERAGE}x`\n"
        message += f"• حجم الصفقة: `${TRADE_SIZE_USDT}`\n"
        message += f"• عتبة الثقة: `{BUY_CONFIDENCE_THRESHOLD}%`\n\n"
        
        message += f"📈 النتائج الإجمالية:\n"
        message += f"• إجمالي الصفقات: `{backtest_result.total_trades}`\n"
        message += f"• الصفقات الرابحة: `{backtest_result.winning_trades}` 🟢\n"
        message += f"• الصفقات الخاسرة: `{backtest_result.losing_trades}` 🔴\n"
        message += f"• نسبة الربح: `{backtest_result.win_rate:.1f}%`\n"
        message += f"• إجمالي الربح: `${backtest_result.total_pnl:.2f}`\n"
        message += f"• الرصيد النهائي: `${backtest_result.final_balance:.2f}`\n"
        message += f"• العائد الإجمالي: `{((backtest_result.final_balance - INITIAL_BALANCE) / INITIAL_BALANCE * 100):.1f}%`\n\n"
        
        message += f"🎯 تحليل الإشارات التفصيلي:\n"
        message += "────────────────────\n"
        
        for signal_type, stats in backtest_result.signal_analysis.items():
            if stats['trades'] > 0:
                emoji = "🟢" if stats['avg_pnl'] > 0 else "🔴"
                message += f"• {signal_type.upper().replace('_', ' ')}:\n"
                message += f"  - الصفقات: `{stats['trades']}` | النجاح: `{stats['win_rate']:.1f}%`\n"
                message += f"  - الربح: `${stats['total_pnl']:.2f}` {emoji}\n"
                message += f"  - متوسط الربح: `${stats['avg_pnl']:.2f}`\n\n"
        
        message += f"📊 تحليل حالة السوق:\n"
        market_trends = backtest_result.market_analysis.get('market_trend_distribution', {})
        for trend, count in market_trends.items():
            if count > 0:
                percentage = (count / len(self.df_global)) * 100
                message += f"• {trend}: `{percentage:.1f}%` من الوقت\n"
        
        message += f"\n🕒 وقت التقرير: `{report_time}`\n"
        message += "══════════════════════════════════════\n"
        
        # توصيات بناءً على النتائج
        if backtest_result.win_rate > 60:
            message += "✅ الاستراتيجية تعمل بشكل ممتاز - الحفاظ على الإعدادات\n"
        elif backtest_result.win_rate > 45:
            message += "⚠️  الاستراتيجية جيدة ولكن تحتاج تحسينات طفيفة\n"
        else:
            message += "🔧 الاستراتيجية تحتاج مراجعة الإعدادات\n"
        
        # أفضل إشارة
        best_signal = None
        best_win_rate = 0
        for signal_type, stats in backtest_result.signal_analysis.items():
            if stats['trades'] > 2 and stats['win_rate'] > best_win_rate:
                best_win_rate = stats['win_rate']
                best_signal = signal_type
        
        if best_signal:
            message += f"🎯 أفضل إشارة: {best_signal.replace('_', ' ').upper()} ({best_win_rate:.1f}% نجاح)\n"
        
        return message

# =============================================================================
# الوظيفة الرئيسية
# =============================================================================

async def main():
    """الوظيفة الرئيسية"""
    
    logger.info("🚀 بدء تشغيل استراتيجية قراءة السوق")
    
    telegram_notifier = TelegramNotifier(TELEGRAM_BOT_TOKEN, TELEGRAM_CHAT_ID)
    
    # جلب البيانات
    data_fetcher = DataFetcher()
    df = data_fetcher.fetch_historical_data(SYMBOL, TIMEFRAME, DATA_LIMIT)
    
    if df.empty:
        error_msg = "❌ فشل جلب البيانات. تأكد من اتصال الإنترنت وصحة اسم العملة."
        logger.error(error_msg)
        await telegram_notifier.send_message(error_msg)
        return
    
    # تشغيل الاستراتيجية
    strategy = MarketReaderStrategy(telegram_notifier)
    backtest_result = strategy.run_market_backtest(df)
    
    # إرسال التقرير
    await strategy.send_market_report(backtest_result)
    
    # حفظ النتائج
    if strategy.trade_history:
        safe_trades = []
        for trade in strategy.trade_history:
            safe_trade = {k: v for k, v in trade.items() if v is not None}
            safe_trades.append(safe_trade)
        
        trades_df = pd.DataFrame(safe_trades)
        filename = f'market_reader_trades_{SYMBOL}_{TIMEFRAME}_{datetime.now().strftime("%Y%m%d_%H%M")}.csv'
        trades_df.to_csv(filename, index=False)
        logger.info(f"💾 تم حفظ سجل الصفقات في {filename}")
    
    logger.info("✅ اكتمل تشغيل استراتيجية قراءة السوق بنجاح")

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
