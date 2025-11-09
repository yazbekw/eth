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
# إعدادات التداول من متغيرات البيئة
# =============================================================================

SYMBOL = os.getenv("TRADING_SYMBOL", "BNBUSDT")
TIMEFRAME = os.getenv("TRADING_TIMEFRAME", "1h")
STOP_LOSS_PERCENT = float(os.getenv("STOP_LOSS_PERCENT", "0.8"))
TAKE_PROFIT_PERCENT = float(os.getenv("TAKE_PROFIT_PERCENT", "2.5"))
TRADE_SIZE_USDT = float(os.getenv("TRADE_SIZE_USDT", "100"))
LEVERAGE = int(os.getenv("LEVERAGE", "10"))
INITIAL_BALANCE = float(os.getenv("INITIAL_BALANCE", "5000.0"))
CONFIDENCE_THRESHOLD = int(os.getenv("CONFIDENCE_THRESHOLD", "75"))

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
logger = logging.getLogger("Enhanced_Volume_Divergence_Strategy")

# =============================================================================
# هياكل البيانات
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
# استراتيجية الانزياح الحجمي المحسنة
# =============================================================================

class EnhancedVolumeDivergenceStrategy:
    """استراتيجية الانزياح الحجمي المحسنة مع نظام تصفية ذكي"""
    
    def __init__(self, telegram_notifier: TelegramNotifier = None):
        self.name = "enhanced_volume_divergence"
        self.trades: List[Trade] = []
        self.balance = INITIAL_BALANCE
        self.current_balance = INITIAL_BALANCE
        self.positions = {}
        self.trade_history = []
        self.analysis_results = []
        self.telegram_notifier = telegram_notifier
        self.performance_stats = {
            'positive_bullish': {'trades': 0, 'wins': 0},
            'negative_bearish': {'trades': 0, 'wins': 0},
            'volume_confirmation': {'trades': 0, 'wins': 0},
            'hidden_divergence': {'trades': 0, 'wins': 0}
        }
    
    def calculate_enhanced_divergence(self, prices: List[float], volumes: List[float]) -> Dict[str, Any]:
        """انزياح محسن بشروط أكثر ذكاء"""
        if len(prices) < 40:
            return {"divergence": "none", "strength": 0}
        
        # تحليل متقدم للاتجاه
        short_trend = (prices[-1] - prices[-10]) / prices[-10]
        medium_trend = (prices[-1] - prices[-20]) / prices[-20]
        long_trend = (prices[-1] - prices[-40]) / prices[-40]
        
        # تحليل الحجم المتقدم
        current_volume = volumes[-1]
        avg_volume_20 = np.mean(volumes[-20:])
        avg_volume_40 = np.mean(volumes[-40:])
        volume_ratio_20 = current_volume / avg_volume_20
        volume_ratio_40 = current_volume / avg_volume_40
        
        # 1. الانزياح الإيجابي المحسن (شروط أكثر تشدداً)
        if (medium_trend < -0.03 and                    # هبوط 3% على الأقل
            volume_ratio_20 > 1.8 and                   # حجم عالي جداً
            volume_ratio_40 > 1.5 and                   # حجم أعلى من المتوسط الطويل
            current_volume > np.percentile(volumes[-100:], 70)):  # حجم في أعلى 30%
            
            strength = min(80, int(abs(medium_trend) * 1500 + (volume_ratio_20 - 1) * 40))
            return {"divergence": "positive_bullish", "strength": strength}
        
        # 2. الانزياح السلبي المحسن
        elif (medium_trend > 0.03 and                   # صعود 3% على الأقل
              volume_ratio_20 > 1.6 and                 # حجم عالي
              volume_ratio_40 > 1.3 and                 # حجم أعلى من المتوسط الطويل
              current_volume > np.percentile(volumes[-100:], 60)):  # حجم في أعلى 40%
            
            strength = min(80, int(abs(medium_trend) * 1500 + (volume_ratio_20 - 1) * 40))
            return {"divergence": "negative_bearish", "strength": strength}
        
        # 3. التأكيد الحجمي المحسن
        elif ((abs(short_trend) > 0.02 and volume_ratio_20 > 2.0) or
              (abs(medium_trend) > 0.04 and volume_ratio_20 > 1.5)):
            
            strength = min(70, int(abs(short_trend) * 1200 + (volume_ratio_20 - 1) * 30))
            return {"divergence": "volume_confirmation", "strength": strength}
        
        # 4. الانزياح الخفي المحسن
        elif ((abs(short_trend) < 0.01 and volume_ratio_20 > 2.5) or
              (abs(medium_trend) > 0.02 and volume_ratio_20 < 0.7)):
            
            strength = min(60, int(abs(short_trend) * 1000 + abs(volume_ratio_20 - 1) * 25))
            return {"divergence": "hidden_divergence", "strength": strength}
        
        return {"divergence": "none", "strength": 0}
    
    def calculate_quality_score(self, df_row: pd.Series, divergence_data: Dict) -> float:
        """حساب درجة الجودة للإشارة"""
        quality_score = 0
        
        # 1. جودة الحجم (40 نقطة)
        volume_score = min(40, (df_row['volume_ratio_20'] - 1) * 20)
        quality_score += volume_score
        
        # 2. استقرار الحجم (20 نقطة)
        if df_row['volume_volatility'] < df_row['volume_volatility'] * 0.8:
            quality_score += 20
        
        # 3. قوة الانزياح (20 نقطة)
        divergence_strength = min(20, divergence_data["strength"] / 5)
        quality_score += divergence_strength
        
        # 4. تأكيد الاتجاه (20 نقطة)
        if ((divergence_data["divergence"] in ["positive_bullish", "volume_confirmation"] and 
             df_row['close'] > df_row['close'].shift(5)) or
            (divergence_data["divergence"] in ["negative_bearish"] and 
             df_row['close'] < df_row['close'].shift(5))):
            quality_score += 20
        
        return min(100, quality_score)
    
    def enhanced_confidence_system(self, divergence_data: Dict, quality_score: float) -> float:
        """نظام ثقة محسن مع عقوبات للأداء الضعيف"""
        
        base_confidence = divergence_data["strength"]
        
        # مضاعفات حسب نوع الانزياح (بناء على الإحصائيات السابقة)
        divergence_multipliers = {
            "positive_bullish": 0.7,      # عقوبة 30% - أداء ضعيف سابقاً
            "negative_bearish": 1.3,      # مكافأة 30% - أداء ممتاز
            "volume_confirmation": 0.9,   # عقوبة 10% - أداء متوسط
            "hidden_divergence": 0.6      # عقوبة 40% - أداء ضعيف
        }
        
        multiplier = divergence_multipliers.get(divergence_data["divergence"], 1.0)
        adjusted_confidence = base_confidence * multiplier
        
        # تعزيز حسب جودة الإشارة
        quality_boost = quality_score / 100
        adjusted_confidence *= (1 + quality_boost * 0.5)  # حتى 50% تعزيز
        
        return min(100, adjusted_confidence)
    
    def dynamic_risk_management(self, divergence_type: str, quality_score: float) -> Tuple[float, float]:
        """إدارة مخاطرة ديناميكية"""
        
        # قاعدة الإعدادات
        base_sl = STOP_LOSS_PERCENT
        base_tp = TAKE_PROFIT_PERCENT
        
        # تعديل حسب نوع الانزياح
        risk_adjustments = {
            "positive_bullish": (1.2, 0.8),    # وقف أكبر، جني أصغر
            "negative_bearish": (0.8, 1.5),    # وقف أصغر، جني أكبر
            "volume_confirmation": (1.0, 1.0), # إعدادات عادية
            "hidden_divergence": (1.5, 0.6)    # وقف أكبر بكثير، جني أصغر
        }
        
        sl_multiplier, tp_multiplier = risk_adjustments.get(divergence_type, (1.0, 1.0))
        
        # تعديل حسب جودة الإشارة
        quality_factor = quality_score / 100
        sl_multiplier *= (1.5 - quality_factor * 0.5)  # جودة عالية = وقف أصغر
        tp_multiplier *= (0.5 + quality_factor * 0.5)  # جودة عالية = جني أكبر
        
        return base_sl * sl_multiplier, base_tp * tp_multiplier
    
    def calculate_volume_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """حساب مؤشرات الحجم المتقدمة"""
        # المتوسطات المتحركة للحجم
        df['volume_ma_10'] = df['volume'].rolling(10).mean()
        df['volume_ma_20'] = df['volume'].rolling(20).mean()
        df['volume_ma_50'] = df['volume'].rolling(50).mean()
        
        # نسب الحجم
        df['volume_ratio_10'] = df['volume'] / df['volume_ma_10']
        df['volume_ratio_20'] = df['volume'] / df['volume_ma_20']
        df['volume_ratio_50'] = df['volume'] / df['volume_ma_50']
        
        # تقلبات الحجم
        df['volume_volatility'] = df['volume'].rolling(20).std()
        
        # مؤشرات متقدمة
        df['volume_trend'] = df['volume_ratio_20'].rolling(5).mean()
        df['volume_momentum'] = df['volume'] - df['volume'].shift(5)
        
        return df
    
    def generate_enhanced_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        """توليد إشارات محسنة"""
        
        signals = []
        confidence_scores = []
        divergence_types = []
        quality_scores = []
        
        for i in range(len(df)):
            if i < 50:  # تحتاج إلى بيانات أكثر للتحليل المتقدم
                signals.append('none')
                confidence_scores.append(0)
                divergence_types.append('none')
                quality_scores.append(0)
                continue
            
            # استخراج البيانات
            prices = df['close'].iloc[:i+1].tolist()
            volumes = df['volume'].iloc[:i+1].tolist()
            
            # حساب الانزياح المحسن
            divergence_data = self.calculate_enhanced_divergence(prices, volumes)
            
            if divergence_data["divergence"] == "none":
                signals.append('none')
                confidence_scores.append(0)
                divergence_types.append('none')
                quality_scores.append(0)
                continue
            
            # حساب درجة الجودة
            quality_score = self.calculate_quality_score(df.iloc[i], divergence_data)
            
            # حساب الثقة المحسنة
            confidence = self.enhanced_confidence_system(divergence_data, quality_score)
            
            # تحديد الإشارة مع شروط أكثر تشدداً
            signal = 'none'
            if confidence >= CONFIDENCE_THRESHOLD and quality_score >= 60:
                if divergence_data["divergence"] in ["positive_bullish", "volume_confirmation"]:
                    # تأكيد إضافي للشراء
                    if prices[-1] > np.mean(prices[-20:]):  # فوق المتوسط
                        signal = "BUY"
                elif divergence_data["divergence"] in ["negative_bearish"]:
                    # تأكيد إضافي للبيع
                    if prices[-1] < np.mean(prices[-20:]):  # تحت المتوسط
                        signal = "SELL"
            
            signals.append(signal)
            confidence_scores.append(confidence)
            divergence_types.append(divergence_data["divergence"])
            quality_scores.append(quality_score)
        
        df['volume_signal'] = signals
        df['volume_confidence'] = confidence_scores
        df['divergence_type'] = divergence_types
        df['quality_score'] = quality_scores
        
        return df
    
    def enhanced_volume_analysis(self, df: pd.DataFrame) -> pd.DataFrame:
        """التحليل الحجمي المحسن"""
        
        # 1. حساب مؤشرات الحجم
        df = self.calculate_volume_indicators(df)
        
        # 2. توليد الإشارات المحسنة
        df = self.generate_enhanced_signals(df)
        
        # 3. إضافة مستوى الثقة
        df['confidence_level'] = df['volume_confidence'].apply(self.calculate_confidence_level)
        
        # حفظ نتائج التحليل
        self.analysis_results = df.to_dict('records')
        
        return df
    
    def calculate_confidence_level(self, score: float) -> str:
        """تحديد مستوى الثقة"""
        if score >= 85: return "عالية جداً"
        elif score >= 75: return "عالية"
        elif score >= 65: return "متوسطة"
        elif score >= 55: return "منخفضة"
        else: return "ضعيفة"
    
    # =========================================================================
    # نظام التداول المحسن
    # =========================================================================
    
    def calculate_position_size(self, price: float) -> float:
        """حساب حجم المركز"""
        return (TRADE_SIZE_USDT * LEVERAGE) / price
    
    def open_position(self, symbol: str, direction: str, price: float, 
                     confidence: float, confidence_level: str, 
                     divergence_type: str, volume_ratio: float, 
                     quality_score: float, timestamp: datetime) -> Optional[Trade]:
        """فتح مركز جديد مع إدارة مخاطرة ديناميكية"""
        
        if symbol in self.positions:
            return None
        
        # حساب حجم المركز
        quantity = self.calculate_position_size(price)
        
        # إدارة مخاطرة ديناميكية
        sl_percent, tp_percent = self.dynamic_risk_management(divergence_type, quality_score)
        
        if direction == "BUY":
            stop_loss = price * (1 - sl_percent / 100)
            take_profit = price * (1 + tp_percent / 100)
        else:  # SELL
            stop_loss = price * (1 + sl_percent / 100)
            take_profit = price * (1 - tp_percent / 100)
        
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
            volume_ratio=volume_ratio,
            quality_score=quality_score
        )
        
        self.positions[symbol] = trade
        self.trades.append(trade)
        
        # تحديث الإحصائيات
        self.performance_stats[divergence_type]['trades'] += 1
        
        logger.info(f"📈 فتح مركز {direction} لـ {symbol} "
                   f"الثقة: {confidence:.1f}% | الجودة: {quality_score:.1f}%")
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
        
        # تحديث إحصائيات الأداء
        if pnl > 0:
            self.performance_stats[trade.divergence_type]['wins'] += 1
        
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
    
    def execute_enhanced_trading(self, df: pd.DataFrame):
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
            timestamp = row['timestamp']
            
            # فحص شروط الخروج
            if SYMBOL in self.positions:
                self.check_stop_conditions(SYMBOL, current_price, timestamp)
            
            # فتح مراكز جديدة بشروط مشددة
            if (SYMBOL not in self.positions and signal != 'none' and 
                confidence >= CONFIDENCE_THRESHOLD and quality_score >= 60):
                
                self.open_position(
                    SYMBOL, signal, current_price, confidence, confidence_level,
                    divergence_type, volume_ratio, quality_score, timestamp
                )
    
    def calculate_enhanced_results(self, df: pd.DataFrame) -> BacktestResult:
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
                quality_analysis={}
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
        
        # تحليل الثقة والجودة
        avg_confidence = trades_df['confidence'].mean()
        avg_quality = trades_df['quality_score'].mean()
        
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
            'volume_correlation': trades_df['volume_ratio'].corr(trades_df['pnl']) if len(trades_df) > 1 else 0
        }
        
        # تحليل الجودة
        quality_analysis = {
            'high_quality_trades': len(trades_df[trades_df['quality_score'] > 70]),
            'avg_quality_score': avg_quality,
            'quality_correlation': trades_df['quality_score'].corr(trades_df['pnl']) if len(trades_df) > 1 else 0,
            'quality_win_rate': (len(trades_df[(trades_df['quality_score'] > 70) & (trades_df['pnl'] > 0)]) / 
                               len(trades_df[trades_df['quality_score'] > 70]) * 100) if len(trades_df[trades_df['quality_score'] > 70]) > 0 else 0
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
            quality_analysis=quality_analysis
        )
    
    def run_enhanced_backtest(self, df: pd.DataFrame) -> BacktestResult:
        """تشغيل الباك-تستينغ المحسن"""
        
        logger.info("🔍 بدء الباك-تستينغ المحسن...")
        
        # إعادة تعيين البيانات
        self.trades = []
        self.positions = {}
        self.trade_history = []
        self.current_balance = INITIAL_BALANCE
        
        # التحليل المحسن
        df_with_signals = self.enhanced_volume_analysis(df)
        
        # تنفيذ التداول المحسن
        self.execute_enhanced_trading(df_with_signals)
        
        # إغلاق المراكز المفتوحة
        if SYMBOL in self.positions:
            last_price = df_with_signals.iloc[-1]['close']
            last_timestamp = df_with_signals.iloc[-1]['timestamp']
            self.close_position(SYMBOL, last_price, last_timestamp, "END_OF_DATA")
        
        return self.calculate_enhanced_results(df)
    
    async def send_enhanced_report(self, backtest_result: BacktestResult, df: pd.DataFrame):
        """إرسال تقرير محسن"""
        
        if not self.telegram_notifier:
            return
        
        try:
            # التقرير النصي
            report_text = self._generate_enhanced_report_text(backtest_result)
            await self.telegram_notifier.send_message(report_text)
            
            # الرسوم البيانية
            chart_buffer = self._create_enhanced_chart(df, backtest_result)
            if chart_buffer:
                caption = f"📈 تحليل استراتيجية الانزياح الحجمي المحسنة - {SYMBOL}"
                await self.telegram_notifier.send_photo(chart_buffer, caption)
                
        except Exception as e:
            logger.error(f"❌ خطأ في إرسال التقرير: {e}")
    
    def _generate_enhanced_report_text(self, backtest_result: BacktestResult) -> str:
        """إنشاء تقرير نصي محسن"""
        
        report_time = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        
        message = f"🎯 *تقرير استراتيجية الانزياح الحجمي المحسنة*\n"
        message += "══════════════════════════════════════\n\n"
        
        message += f"⚙️ *الإعدادات المتقدمة:*\n"
        message += f"• العملة: `{SYMBOL}`\n"
        message += f"• الإطار: `{TIMEFRAME}`\n"
        message += f"• الرافعة: `{LEVERAGE}x`\n"
        message += f"• حجم الصفقة: `${TRADE_SIZE_USDT}`\n"
        message += f"• عتبة الثقة: `{CONFIDENCE_THRESHOLD}%`\n"
        message += f"• عتبة الجودة: `60%`\n\n"
        
        message += f"📊 *النتائج المحسنة:*\n"
        message += f"• إجمالي الصفقات: `{backtest_result.total_trades}`\n"
        message += f"• الصفقات الرابحة: `{backtest_result.winning_trades}` 🟢\n"
        message += f"• الصفقات الخاسرة: `{backtest_result.losing_trades}` 🔴\n"
        message += f"• نسبة الربح: `{backtest_result.win_rate:.1f}%`\n"
        message += f"• إجمالي الربح: `${backtest_result.total_pnl:,.2f}`\n"
        message += f"• الرصيد النهائي: `${backtest_result.final_balance:,.2f}`\n"
        message += f"• متوسط الجودة: `{backtest_result.quality_analysis['avg_quality_score']:.1f}%`\n\n"
        
        message += f"🔍 *تحليل الانزياح المحسن:*\n"
        divergence_names = {
            'positive_bullish': '🟢 الانزياح الإيجابي',
            'negative_bearish': '🔴 الانزياح السلبي', 
            'volume_confirmation': '📈 التأكيد الحجمي',
            'hidden_divergence': '🎯 الانزياح الخفي'
        }
        
        for div_type, analysis in backtest_result.divergence_analysis.items():
            display_name = divergence_names.get(div_type, div_type)
            message += f"{display_name}:\n"
            message += f"• الصفقات: `{analysis['trades']}` | الدقة: `{analysis['win_rate']:.1f}%`\n"
            message += f"• الربح: `${analysis['total_pnl']:.2f}` | المتوسط: `${analysis['avg_pnl']:.2f}`\n\n"
        
        message += f"📈 *تحليل الجودة:*\n"
        message += f"• الصفقات عالية الجودة: `{backtest_result.quality_analysis['high_quality_trades']}`\n"
        message += f"• دقة الصفقات عالية الجودة: `{backtest_result.quality_analysis['quality_win_rate']:.1f}%`\n"
        message += f"• ارتباط الجودة بالربح: `{backtest_result.quality_analysis['quality_correlation']:.3f}`\n\n"
        
        message += f"🕒 *وقت التقرير:* `{report_time}`\n"
        message += "══════════════════════════════════════\n"
        message += "⚡ *نظام الانزياح الحجمي المحسن + فلاتر الجودة*"
        
        return message
    
    def _create_enhanced_chart(self, df: pd.DataFrame, backtest_result: BacktestResult) -> BytesIO:
        """إنشاء رسم بياني محسن"""
        try:
            fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
            fig.suptitle(f'تحليل استراتيجية الانزياح الحجمي المحسنة - {SYMBOL}', 
                        fontsize=16, fontname='DejaVu Sans', fontweight='bold')
            
            # 1. السعر والإشارات
            ax1.plot(df['timestamp'], df['close'], label='السعر', linewidth=1.5, color='blue', alpha=0.8)
            ax1.set_title('حركة السعر والإشارات المحسنة', fontname='DejaVu Sans', fontsize=12)
            ax1.set_ylabel('السعر (USDT)', fontname='DejaVu Sans')
            ax1.legend(prop={'family': 'DejaVu Sans'})
            ax1.grid(True, alpha=0.3)
            
            # 2. توزيع الجودة
            if not self.trade_history.empty:
                quality_scores = [t['quality_score'] for t in self.trade_history]
                ax2.hist(quality_scores, bins=15, alpha=0.7, color='green', edgecolor='black')
                ax2.axvline(60, color='red', linestyle='--', label='عتبة الجودة')
                ax2.set_title('توزيع درجات الجودة', fontname='DejaVu Sans', fontsize=12)
                ax2.set_xlabel('درجة الجودة', fontname='DejaVu Sans')
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
                
                colors = ['green' if wr > 50 else 'red' for wr in win_rates]
                bars = ax4.bar(div_types, win_rates, color=colors, alpha=0.7)
                
                ax4.set_title('مقارنة دقة أنواع الانزياح', fontname='DejaVu Sans', fontsize=12)
                ax4.set_ylabel('نسبة الربح %', fontname='DejaVu Sans')
                ax4.set_xticklabels([d[:15] for d in div_types], fontname='DejaVu Sans', rotation=45)
                ax4.grid(True, alpha=0.3)
                
                # إضافة القيم على الأعمدة
                for bar, wr in zip(bars, win_rates):
                    ax4.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1, 
                            f'{wr:.1f}%', ha='center', fontname='DejaVu Sans')
            
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
# الوظيفة الرئيسية
# =============================================================================

async def main():
    """الوظيفة الرئيسية"""
    
    logger.info("🚀 بدء تشغيل الاستراتيجية المحسنة")
    
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
    data_info = f"📊 فترة البيانات: {len(df)} شمعة من {df['timestamp'].min().date()} إلى {df['timestamp'].max().date()}"
    logger.info(data_info)
    await telegram_notifier.send_message(data_info)
    
    # تشغيل الاستراتيجية المحسنة
    strategy = EnhancedVolumeDivergenceStrategy(telegram_notifier)
    backtest_result = strategy.run_enhanced_backtest(df)
    
    # إرسال التقرير
    await strategy.send_enhanced_report(backtest_result, df)
    
    # حفظ النتائج
    trades_df = pd.DataFrame(strategy.trade_history)
    if not trades_df.empty:
        filename = f'enhanced_volume_trades_{SYMBOL}_{TIMEFRAME}.csv'
        trades_df.to_csv(filename, index=False)
        logger.info(f"💾 تم حفظ سجل الصفقات في {filename}")
    
    logger.info("✅ اكتمل تشغيل الاستراتيجية المحسنة بنجاح")

if __name__ == "__main__":
    asyncio.run(main())
