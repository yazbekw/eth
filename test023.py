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
# إعدادات التداول المحسنة للبيع
# =============================================================================

SYMBOL = os.getenv("TRADING_SYMBOL", "BNBUSDT")
TIMEFRAME = os.getenv("TRADING_TIMEFRAME", "1h")
STOP_LOSS_PERCENT = float(os.getenv("STOP_LOSS_PERCENT", "0.5"))  # مخاطرة أقل للبيع
TAKE_PROFIT_PERCENT = float(os.getenv("TAKE_PROFIT_PERCENT", "3.5"))  # أرباح أعلى للبيع
TRADE_SIZE_USDT = float(os.getenv("TRADE_SIZE_USDT", "100"))
LEVERAGE = int(os.getenv("LEVERAGE", "8"))
INITIAL_BALANCE = float(os.getenv("INITIAL_BALANCE", "5000.0"))

# عتبات منفصلة للشراء والبيع
BUY_CONFIDENCE_THRESHOLD = int(os.getenv("BUY_CONFIDENCE_THRESHOLD", "70"))
SELL_CONFIDENCE_THRESHOLD = int(os.getenv("SELL_CONFIDENCE_THRESHOLD", "78"))  # أعلى للبيع
SELL_PREMIUM_THRESHOLD = int(os.getenv("SELL_PREMIUM_THRESHOLD", "82"))  # بيع متميز
SELL_QUALITY_THRESHOLD = int(os.getenv("SELL_QUALITY_THRESHOLD", "80"))  # جودة عالية للبيع

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
logger = logging.getLogger("Enhanced_Sell_Strategy_v5")

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
# استراتيجية الانزياح الحجمي المحسنة للبيع
# =============================================================================

class EnhancedSellStrategy:
    """استراتيجية محسنة تركز على تحسين أداء البيع"""
    
    def __init__(self, telegram_notifier: TelegramNotifier = None):
        self.name = "enhanced_sell_strategy_v5"
        self.trades: List[Trade] = []
        self.balance = INITIAL_BALANCE
        self.current_balance = INITIAL_BALANCE
        self.positions = {}
        self.trade_history = []
        self.analysis_results = []
        self.telegram_notifier = telegram_notifier
        
        # إحصائيات متقدمة للبيع
        self.sell_stats = {
            'standard_sell': {'trades': 0, 'wins': 0, 'total_pnl': 0},
            'premium_sell': {'trades': 0, 'wins': 0, 'total_pnl': 0},
            'ultra_sell': {'trades': 0, 'wins': 0, 'total_pnl': 0}
        }
    
    def calculate_enhanced_sell_divergence(self, prices: List[float], volumes: List[float]) -> Dict[str, Any]:
        """انزياح بيع محسن بشروط أكثر تشدداً"""
        if len(prices) < 60:
            return {"divergence": "none", "strength": 0, "sell_category": "NONE"}
        
        # تحليل متقدم للاتجاه الهبوطي
        trend_5 = (prices[-1] - prices[-5]) / prices[-5]
        trend_10 = (prices[-1] - prices[-10]) / prices[-10]
        trend_20 = (prices[-1] - prices[-20]) / prices[-20]
        trend_50 = (prices[-1] - prices[-50]) / prices[-50]
        
        # تحليل الحجم المتقدم للبيع
        current_volume = volumes[-1]
        avg_volume_10 = np.mean(volumes[-10:])
        avg_volume_20 = np.mean(volumes[-20:])
        avg_volume_50 = np.mean(volumes[-50:])
        
        volume_ratio_10 = current_volume / avg_volume_10
        volume_ratio_20 = current_volume / avg_volume_20
        volume_ratio_50 = current_volume / avg_volume_50
        
        # قوة الاتجاه الهبوطي
        bearish_strength = abs(min(0, trend_10, trend_20, trend_50))
        
        # 1. بيع متميز (شروط مشددة جداً)
        if (trend_20 > 0.08 and                    # صعود قوي سابق
            trend_5 < -0.03 and                    # انعكاس هبوطي حاد
            volume_ratio_20 > 2.5 and              # حجم عالي جداً
            volume_ratio_50 > 2.0 and              # تأكيد حجم طويل المدى
            current_volume > np.percentile(volumes[-100:], 90) and  # حجم في أعلى 10%
            bearish_strength > 0.05):              # قوة هبوط عالية
            
            strength = min(95, int(
                bearish_strength * 2500 + 
                (volume_ratio_20 - 1) * 40 +
                abs(trend_5) * 1500
            ))
            return {"divergence": "bearish_reversal", "strength": strength, "sell_category": "ULTRA"}
        
        # 2. بيع عالي الجودة
        elif (trend_20 > 0.05 and                   # صعود جيد سابق
              trend_10 < -0.02 and                  # بداية هبوط
              volume_ratio_20 > 2.2 and             # حجم عالي
              volume_ratio_10 > 2.5 and             # تسارع حجمي
              current_volume > np.percentile(volumes[-100:], 85) and  # حجم في أعلى 15%
              bearish_strength > 0.03):
            
            strength = min(88, int(
                bearish_strength * 2000 + 
                (volume_ratio_20 - 1) * 35 +
                abs(trend_10) * 1200
            ))
            return {"divergence": "bearish_reversal", "strength": strength, "sell_category": "PREMIUM"}
        
        # 3. بيع قياسي
        elif (trend_20 > 0.03 and                   # صعود معتدل سابق
              trend_5 < -0.015 and                  # انعكاس هبوطي
              volume_ratio_20 > 1.8 and             # حجم جيد
              volume_ratio_10 > 2.0 and             # تسارع حجمي
              current_volume > np.percentile(volumes[-100:], 75) and  # حجم في أعلى 25%
              bearish_strength > 0.02):
            
            strength = min(80, int(
                bearish_strength * 1500 + 
                (volume_ratio_20 - 1) * 30 +
                abs(trend_5) * 1000
            ))
            return {"divergence": "bearish_reversal", "strength": strength, "sell_category": "STANDARD"}
        
        return {"divergence": "none", "strength": 0, "sell_category": "NONE"}
    
    def calculate_buy_divergence(self, prices: List[float], volumes: List[float]) -> Dict[str, Any]:
        """انزياح شراء (محافظ على الأداء الجيد)"""
        if len(prices) < 50:
            return {"divergence": "none", "strength": 0}
        
        trend_20 = (prices[-1] - prices[-20]) / prices[-20]
        current_volume = volumes[-1]
        avg_volume_20 = np.mean(volumes[-20:])
        volume_ratio_20 = current_volume / avg_volume_20
        
        if (trend_20 < -0.03 and
            volume_ratio_20 > 2.0 and
            current_volume > np.percentile(volumes[-100:], 80)):
            
            strength = min(85, int(abs(trend_20) * 1800 + (volume_ratio_20 - 1) * 35))
            return {"divergence": "bullish_reversal", "strength": strength}
        
        return {"divergence": "none", "strength": 0}
    
    def calculate_trend_strength(self, prices: List[float]) -> float:
        """حساب قوة الاتجاه"""
        if len(prices) < 20:
            return 0.5
        
        short_trend = (prices[-1] - prices[-5]) / prices[-5]
        medium_trend = (prices[-1] - prices[-10]) / prices[-10]
        long_trend = (prices[-1] - prices[-20]) / prices[-20]
        
        # متوسط مرجح للاتجاهات
        trend_strength = (abs(short_trend) * 0.4 + abs(medium_trend) * 0.3 + abs(long_trend) * 0.3)
        direction = -1 if (short_trend + medium_trend + long_trend) < 0 else 1
        
        return trend_strength * direction
    
    def calculate_volume_surge(self, volumes: List[float]) -> float:
        """حساب قوة طفرة الحجم"""
        if len(volumes) < 10:
            return 0
        
        current_volume = volumes[-1]
        avg_volume_10 = np.mean(volumes[-10:])
        volume_surge = (current_volume - avg_volume_10) / avg_volume_10
        
        return max(0, volume_surge)
    
    def calculate_sell_quality_score(self, df_row: pd.Series, divergence_data: Dict, 
                                   df: pd.DataFrame, current_index: int) -> float:
        """حساب درجة الجودة للصفقات البيعية"""
        quality_score = 0
        
        # 1. جودة الحجم (30 نقطة)
        volume_score = min(30, (df_row['volume_ratio_20'] - 1) * 15)
        quality_score += volume_score
        
        # 2. قوة الانزياح (25 نقطة)
        divergence_strength = min(25, divergence_data["strength"] / 4)
        quality_score += divergence_strength
        
        # 3. قوة الاتجاه الهبوطي (25 نقطة)
        if current_index >= 20:
            prices = df['close'].iloc[:current_index+1].tolist()
            trend_strength = abs(self.calculate_trend_strength(prices))
            if trend_strength < 0:  # اتجاه هبوطي
                trend_score = min(25, abs(trend_strength) * 500)
                quality_score += trend_score
        
        # 4. طفرة الحجم (20 نقطة)
        if current_index >= 10:
            volumes = df['volume'].iloc[:current_index+1].tolist()
            volume_surge = self.calculate_volume_surge(volumes)
            surge_score = min(20, volume_surge * 100)
            quality_score += surge_score
        
        # مكافأة للبيع المتميز
        if divergence_data["sell_category"] == "ULTRA":
            quality_score += 15
        elif divergence_data["sell_category"] == "PREMIUM":
            quality_score += 10
        
        return min(100, quality_score)
    
    def calculate_buy_quality_score(self, df_row: pd.Series, divergence_data: Dict, 
                                  df: pd.DataFrame, current_index: int) -> float:
        """حساب درجة الجودة للصفقات الشرائية"""
        quality_score = 0
        
        volume_score = min(35, (df_row['volume_ratio_20'] - 1) * 17)
        quality_score += volume_score
        
        divergence_strength = min(25, divergence_data["strength"] / 4)
        quality_score += divergence_strength
        
        if current_index >= 15:
            volume_volatility = df['volume'].iloc[current_index-15:current_index].std()
            current_volatility = df['volume'].iloc[current_index-5:current_index].std() if current_index >= 5 else volume_volatility
            if current_volatility < volume_volatility * 0.8:
                quality_score += 20
        
        return min(100, quality_score)
    
    def enhanced_sell_confidence_system(self, divergence_data: Dict, quality_score: float) -> float:
        """نظام ثقة محسن للبيع"""
        
        base_confidence = divergence_data["strength"]
        
        # مضاعفات حسب فئة البيع
        category_multipliers = {
            "ULTRA": 1.4,
            "PREMIUM": 1.2,
            "STANDARD": 1.0
        }
        
        multiplier = category_multipliers.get(divergence_data["sell_category"], 1.0)
        adjusted_confidence = base_confidence * multiplier
        
        # تعزيز حسب جودة الإشارة
        quality_boost = quality_score / 100
        adjusted_confidence *= (1 + quality_boost * 0.6)
        
        # عقوبة للجودة المنخفضة
        if quality_score < SELL_QUALITY_THRESHOLD:
            adjusted_confidence *= 0.7
        
        return min(95, adjusted_confidence)
    
    def dynamic_sell_risk_management(self, sell_category: str, quality_score: float) -> Tuple[float, float]:
        """إدارة مخاطرة ديناميكية للبيع"""
        
        base_sl = STOP_LOSS_PERCENT
        base_tp = TAKE_PROFIT_PERCENT
        
        # إعدادات أكثر تشدداً للبيع
        risk_adjustments = {
            "ULTRA": (0.5, 4.0),    # وقف صغير جداً، جني كبير
            "PREMIUM": (0.6, 3.5),  # وقف صغير، جني كبير
            "STANDARD": (0.7, 3.0)  # وقف معتدل، جني جيد
        }
        
        sl_multiplier, tp_multiplier = risk_adjustments.get(sell_category, (1.0, 1.0))
        
        # تعديل حسب الجودة
        quality_factor = quality_score / 100
        sl_multiplier *= (1.1 - quality_factor * 0.3)
        tp_multiplier *= (0.9 + quality_factor * 0.4)
        
        return base_sl * sl_multiplier, base_tp * tp_multiplier
    
    def calculate_volume_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """حساب مؤشرات الحجم"""
        df['volume_ma_5'] = df['volume'].rolling(5).mean()
        df['volume_ma_10'] = df['volume'].rolling(10).mean()
        df['volume_ma_20'] = df['volume'].rolling(20).mean()
        df['volume_ma_50'] = df['volume'].rolling(50).mean()
        
        df['volume_ratio_5'] = df['volume'] / df['volume_ma_5']
        df['volume_ratio_10'] = df['volume'] / df['volume_ma_10']
        df['volume_ratio_20'] = df['volume'] / df['volume_ma_20']
        df['volume_ratio_50'] = df['volume'] / df['volume_ma_50']
        
        return df
    
    def generate_enhanced_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        """توليد إشارات محسنة مع تركيز على البيع"""
    
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
            if i < 60:
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
        
            # معالجة إشارات البيع
            sell_signal = 'none'
            sell_confidence = 0
            sell_quality = 0
            
            if sell_divergence["divergence"] != "none":
                sell_quality = self.calculate_sell_quality_score(df.iloc[i], sell_divergence, df, i)
                sell_confidence = self.enhanced_sell_confidence_system(sell_divergence, sell_quality)
                
                # شروط البيع المشددة
                if (sell_confidence >= SELL_CONFIDENCE_THRESHOLD and 
                    sell_quality >= SELL_QUALITY_THRESHOLD and
                    trend_strength < -0.01):  # تأكيد اتجاه هبوطي
                    
                    # شروط إضافية حسب الفئة
                    if sell_divergence["sell_category"] == "ULTRA":
                        sell_signal = "SELL"
                    elif sell_divergence["sell_category"] == "PREMIUM" and sell_confidence >= SELL_PREMIUM_THRESHOLD:
                        sell_signal = "SELL"
                    elif sell_divergence["sell_category"] == "STANDARD" and sell_quality >= 85:
                        sell_signal = "SELL"
            
            # معالجة إشارات الشراء
            buy_signal = 'none'
            buy_confidence = 0
            buy_quality = 0
            
            if buy_divergence["divergence"] != "none":
                buy_quality = self.calculate_buy_quality_score(df.iloc[i], buy_divergence, df, i)
                buy_confidence = buy_divergence["strength"]
                
                if (buy_confidence >= BUY_CONFIDENCE_THRESHOLD and 
                    buy_quality >= 70):
                    buy_signal = "BUY"
        
            buy_signals.append(buy_signal)
            sell_signals.append(sell_signal)
            buy_confidence_scores.append(buy_confidence)
            sell_confidence_scores.append(sell_confidence)
            buy_quality_scores.append(buy_quality)
            sell_quality_scores.append(sell_quality)
            sell_categories.append(sell_divergence["sell_category"])
    
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
        df = self.calculate_volume_indicators(df)
        df = self.generate_enhanced_signals(df)
        self.analysis_results = df.to_dict('records')
        return df
    
    def calculate_position_size(self, price: float, confidence: float, direction: str) -> float:
        """حساب حجم المركز"""
        base_size = (TRADE_SIZE_USDT * LEVERAGE) / price
        
        # تعديل الحجم حسب الثقة والاتجاه
        confidence_factor = confidence / 100
        if direction == "SELL":
            # حجم أصغر قليلاً للبيع لتقليل المخاطرة
            adjusted_size = base_size * (0.7 + confidence_factor * 0.6)
        else:
            adjusted_size = base_size * (0.8 + confidence_factor * 0.4)
        
        return adjusted_size
    
    def open_position(self, symbol: str, direction: str, price: float, 
                     confidence: float, quality_score: float, 
                     sell_category: str, volume_ratio: float, 
                     trend_strength: float, volume_surge: float,
                     timestamp: datetime) -> Optional[Trade]:
        """فتح مركز جديد"""
        
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
            volume_surge=volume_surge
        )
        
        self.positions[symbol] = trade
        self.trades.append(trade)
        
        logger.info(f"🎯 فتح مركز {direction} محسن لـ {symbol}")
        logger.info(f"   الثقة: {confidence:.1f}% | الجودة: {quality_score:.1f}%")
        if direction == "SELL":
            logger.info(f"   فئة البيع: {sell_category} | الوقف: {sl_percent:.1f}% | الجني: {tp_percent:.1f}%")
        
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
            'quality_score': trade.quality_score,
            'sell_category': trade.sell_category,
            'volume_ratio': trade.volume_ratio,
            'trend_strength': trade.trend_strength,
            'volume_surge': trade.volume_surge,
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
        
        logger.info("🚀 بدء التداول المحسن مع تحسين البيع...")
        
        for i, row in df.iterrows():
            if i < 60:
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
            
            # فتح مراكز بيعية جديدة بشروط مشددة
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
                quality_analysis={}, performance_metrics={}, sell_analysis={}
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
        
        # حساب الرسوم
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
        
        # تحليل البيع المتقدم
        sell_trades = trades_df[trades_df['direction'] == 'SELL']
        buy_trades = trades_df[trades_df['direction'] == 'BUY']
        
        sell_analysis = {
            'total_sell_trades': len(sell_trades),
            'sell_win_rate': (len(sell_trades[sell_trades['pnl'] > 0]) / len(sell_trades) * 100) if len(sell_trades) > 0 else 0,
            'sell_total_pnl': sell_trades['pnl'].sum() if len(sell_trades) > 0 else 0,
            'sell_avg_pnl': sell_trades['pnl'].mean() if len(sell_trades) > 0 else 0,
            'sell_avg_confidence': sell_trades['confidence'].mean() if len(sell_trades) > 0 else 0,
            'sell_avg_quality': sell_trades['quality_score'].mean() if len(sell_trades) > 0 else 0,
            'buy_total_trades': len(buy_trades),
            'buy_win_rate': (len(buy_trades[buy_trades['pnl'] > 0]) / len(buy_trades) * 100) if len(buy_trades) > 0 else 0,
            'buy_total_pnl': buy_trades['pnl'].sum() if len(buy_trades) > 0 else 0,
            'buy_avg_pnl': buy_trades['pnl'].mean() if len(buy_trades) > 0 else 0
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
            'avg_volume_ratio': trades_df['volume_ratio'].mean(),
            'volume_correlation': trades_df['volume_ratio'].corr(trades_df['pnl']) if len(trades_df) > 1 else 0
        }
        
        # تحليل الجودة
        quality_analysis = {
            'avg_quality_score': trades_df['quality_score'].mean(),
            'quality_correlation': trades_df['quality_score'].corr(trades_df['pnl']) if len(trades_df) > 1 else 0
        }
        
        # مقاييس الأداء
        performance_metrics = {
            'risk_reward_ratio': abs(avg_trade / worst_trade) if worst_trade < 0 else 0,
            'expectancy': (win_rate/100 * avg_trade) - ((1 - win_rate/100) * abs(avg_trade)),
            'consistency_score': (win_rate * profit_factor) / 100
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
            performance_metrics=performance_metrics,
            sell_analysis=sell_analysis
        )
    
    def run_enhanced_backtest(self, df: pd.DataFrame) -> BacktestResult:
        """تشغيل الباك-تستينغ المحسن"""
        
        logger.info("🔍 بدء الباك-تستينغ المحسن للبيع...")
        
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
    
    async def send_enhanced_report(self, backtest_result: BacktestResult, df: pd.DataFrame):
        """إرسال تقرير محسن"""
        
        if not self.telegram_notifier:
            return
        
        try:
            # التقرير النصي
            report_text = self._generate_enhanced_report_text(backtest_result)
            await self.telegram_notifier.send_message(report_text)
                
        except Exception as e:
            logger.error(f"❌ خطأ في إرسال التقرير: {e}")
    
    def _generate_enhanced_report_text(self, backtest_result: BacktestResult) -> str:
        """إنشاء تقرير نصي محسن"""
        
        report_time = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        
        message = f"🎯 تقرير استراتيجية المحسنة v5 - تحسين متقدم لأداء البيع\n"
        message += "══════════════════════════════════════\n\n"
        
        message += f"⚙️ الإعدادات المتقدمة v5:\n"
        message += f"• العملة: `{SYMBOL}`\n"
        message += f"• الإطار: `{TIMEFRAME}`\n"
        message += f"• الرافعة: `{LEVERAGE}x`\n"
        message += f"• حجم الصفقة: `${TRADE_SIZE_USDT}`\n"
        message += f"• عتبة ثقة الشراء: `{BUY_CONFIDENCE_THRESHOLD}%`\n"
        message += f"• عتبة ثقة البيع: `{SELL_CONFIDENCE_THRESHOLD}%`\n"
        message += f"• عتبة البيع فائق الجودة: `{SELL_PREMIUM_THRESHOLD}%`\n"
        message += f"• عتبة البيع عالي الجودة: `{SELL_QUALITY_THRESHOLD}%`\n\n"
        
        message += f"📊 النتائج المحسنة v5:\n"
        message += f"• إجمالي الصفقات: `{backtest_result.total_trades}`\n"
        message += f"• الصفقات الرابحة: `{backtest_result.winning_trades}` 🟢\n"
        message += f"• الصفقات الخاسرة: `{backtest_result.losing_trades}` 🔴\n"
        message += f"• نسبة الربح: `{backtest_result.win_rate:.1f}%`\n"
        message += f"• إجمالي الربح: `${backtest_result.total_pnl:.2f}`\n"
        message += f"• الرصيد النهائي: `${backtest_result.final_balance:.2f}`\n"
        message += f"• العائد الإجمالي: `{((backtest_result.final_balance - INITIAL_BALANCE) / INITIAL_BALANCE * 100):.1f}%`\n"
        message += f"• متوسط الثقة: `{backtest_result.avg_confidence:.1f}%`\n\n"
        
        message += f"🎯 مقاييس المخاطرة المحسنة v5:\n"
        message += f"• أقصى خسارة: `{backtest_result.max_drawdown:.1f}%`\n"
        message += f"• متوسط الربح/صفقة: `${backtest_result.avg_trade:.2f}`\n"
        message += f"• أفضل صفقة: `${backtest_result.best_trade:.2f}` 🚀\n"
        message += f"• أسوأ صفقة: `${backtest_result.worst_trade:.2f}` 📉\n"
        message += f"• نسبة شارب: `{backtest_result.sharpe_ratio:.2f}`\n"
        message += f"• عامل الربحية: `{backtest_result.profit_factor:.2f}`\n\n"
        
        message += f"🕒 وقت التقرير: `{report_time}`\n"
        message += "══════════════════════════════════════\n"
        message += "⚡ نظام التقييم v5 + تحسين متقدم للبيع + إعدادات متوازنة\n\n"
        
        message += f"🔍 تحليل مفصل للبيع والشراء v5:\n"
        message += "────────────────────\n"
        
        # تحليل الشراء
        buy_analysis = backtest_result.sell_analysis
        message += f"🔼 صفقات الشراء:\n"
        message += f"• العدد: `{buy_analysis['buy_total_trades']} صفقة`\n"
        message += f"• الربح: `${buy_analysis['buy_total_pnl']:.2f}` {'✅' if buy_analysis['buy_total_pnl'] > 0 else '❌'}\n"
        message += f"• متوسط الربح: `${buy_analysis['buy_avg_pnl']:.2f}`\n"
        message += f"• نسبة النجاح: `{buy_analysis['buy_win_rate']:.1f}%`\n\n"
        
        # تحليل البيع
        message += f"🔽 صفقات البيع المحسنة v5:\n"
        message += f"• العدد: `{buy_analysis['total_sell_trades']} صفقة`\n"
        message += f"• الربح: `${buy_analysis['sell_total_pnl']:.2f}` {'✅' if buy_analysis['sell_total_pnl'] > 0 else '❌'}\n"
        message += f"• متوسط الربح: `${buy_analysis['sell_avg_pnl']:.2f}`\n"
        message += f"• نسبة النجاح: `{buy_analysis['sell_win_rate']:.1f}%`\n"
        message += f"• متوسط الجودة: `{buy_analysis['sell_avg_quality']:.1f}%`\n"
        message += f"• متوسط الثقة: `{buy_analysis['sell_avg_confidence']:.1f}%`\n\n"
        
        # تحليل فئات البيع
        message += f"🎯 تحليل جودة البيع v5:\n"
        for category in ['standard_sell', 'premium_sell', 'ultra_sell']:
            if category in buy_analysis:
                cat_data = buy_analysis[category]
                emoji = "🟢" if cat_data['avg_pnl'] > 0 else "🔴"
                message += f"• {category.upper().replace('_', ' ')}: {cat_data['trades']} صفقات, نجاح: {cat_data['win_rate']:.1f}%, ربح: ${cat_data['total_pnl']:.2f} {emoji}\n"
        
        message += f"\n📊 مقارنة الأداء v5:\n"
        performance_diff = buy_analysis['sell_win_rate'] - buy_analysis['buy_win_rate']
        pnl_diff = buy_analysis['sell_total_pnl'] - buy_analysis['buy_total_pnl']
        message += f"• فرق النجاح: `{performance_diff:+.1f}%` {'✅' if performance_diff > 0 else '❌'}\n"
        message += f"• فرق الربح: `${pnl_diff:+.2f}` {'✅' if pnl_diff > 0 else '❌'}\n\n"
        
        # توصيات
        message += f"🎯 توصيات تحسين البيع v5:\n"
        if buy_analysis['total_sell_trades'] == 0:
            message += f"• زيادة حساسية كاشفات البيع 🔍\n"
            message += f"• تخفيض طفيف في عتبات الثقة للبيع 📈\n"
            message += f"• التركيز على البيع فائق الجودة أولاً 🎯\n"
        elif buy_analysis['sell_win_rate'] < 50:
            message += f"• تحسين شروط البيع الحالية 🔧\n"
            message += f"• زيادة عتبات الجودة للبيع 📊\n"
            message += f"• التركيز على فئة ULTRA فقط ⭐\n"
        else:
            message += f"• أداء البيع ممتاز - الحفاظ على الإعدادات ✅\n"
            message += f"• يمكن زيادة حجم صفقات البيع 📈\n"
            message += f"• توسيع نطاق فئات البيع 🎯\n"
        
        message += f"\n📈 مستوى الثقة: {'مرتفع' if backtest_result.avg_confidence > 75 else 'متوسط' if backtest_result.avg_confidence > 60 else 'منخفض'} ({backtest_result.avg_confidence:.1f}%) {'✅' if backtest_result.avg_confidence > 70 else '⚠️'}\n"
        
        # معلومات البيانات
        data_period = f"📊 فترة البيانات المحسنة: {len(df)} شمعة من {df['timestamp'].min().date()} إلى {df['timestamp'].max().date()}"
        message += f"\n{data_period}\n"
        
        # تقييم نهائي
        if buy_analysis['sell_win_rate'] > 60 and buy_analysis['sell_total_pnl'] > 0:
            final_msg = "✅ استراتيجية البيع المحسنة تعمل بشكل ممتاز"
        elif buy_analysis['sell_win_rate'] > 50 and buy_analysis['sell_total_pnl'] > 0:
            final_msg = "⚠️ استراتيجية البيع المحسنة تعمل بشكل جيد ولكن تحتاج تحسينات طفيفة"
        else:
            final_msg = "❌ استراتيجية البيع المحسنة تحتاج تحسينات جذرية"
        
        message += f"\n{final_msg}"
        
        return message

# =============================================================================
# الوظيفة الرئيسية المحسنة
# =============================================================================

async def main():
    """الوظيفة الرئيسية المحسنة"""
    
    logger.info("🚀 بدء تشغيل استراتيجية البيع المحسنة v5")
    
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
    await strategy.send_enhanced_report(backtest_result, df)
    
    # حفظ النتائج
    trades_df = pd.DataFrame(strategy.trade_history)
    if not trades_df.empty:
        filename = f'enhanced_sell_trades_v5_{SYMBOL}_{TIMEFRAME}_{datetime.now().strftime("%Y%m%d_%H%M")}.csv'
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
