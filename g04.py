# ====================== المكتبات ======================
import os
import pandas as pd
import numpy as np
import requests
from datetime import datetime, timedelta
import telebot
import warnings
import logging
from tenacity import retry, stop_after_attempt, wait_exponential
from dotenv import load_dotenv
import matplotlib.pyplot as plt
import io
from collections import Counter

warnings.filterwarnings('ignore')
load_dotenv()

# ====================== إعدادات التفعيل / الإلغاء ======================
ENABLE_TRAILING_STOP = False
ENABLE_DYNAMIC_POSITION_SIZING = True  # مفعل لدعم قوة الإشارة
ENABLE_MARKET_REGIME_FILTER = True
ENABLE_ATR_SL_TP = False
ENABLE_SUPPORT_RESISTANCE_FILTER = True
ENABLE_TIME_FILTER = False
ENABLE_WALK_FORWARD = False
ENABLE_LOGGING = True
ENABLE_DETAILED_REPORT = True

# ====================== دالة مساعدة خارج الكلاس (آمنة) ======================
def interval_to_hours(interval):
    mapping = {
        '1m': 1/60, '3m': 3/60, '5m': 5/60, '15m': 15/60,
        '30m': 30/60, '1h': 1, '2h': 2, '4h': 4, '6h': 6,
        '8h': 8, '12h': 12, '1d': 24, '3d': 72, '1w': 168, '1M': 720
    }
    return mapping.get(interval, 4)

# ====================== الإعدادات الأساسية ======================
TRADE_CONFIG = {
    'symbol': 'BNBUSDT',
    'timeframe': '4h',
    'initial_balance': 200,
    'leverage': 1,
    'base_stop_loss': 0.025,
    'base_take_profit': 0.060,
    'base_position_size': 0.1,
    'max_positions': 4,
    'paper_trading': True,
    'use_trailing_stop': ENABLE_TRAILING_STOP,
    'trailing_stop_percent': 0.015,
    'trailing_activation': 0.02,
    'max_trade_duration': 60,
    'atr_multiplier_sl': 2.0,
    'atr_multiplier_tp': 4.0,
    'atr_period': 14,
    'support_resistance_window': 20,
    'peak_hours': [0, 4, 8, 12, 16, 20],
    'min_volume_ratio': 1.2
}

INDICATOR_CONFIG = {
    'rsi_period': 14,
    'rsi_overbought': 70,
    'rsi_oversold': 30,
    'ema_fast': 9,
    'ema_slow': 21,
    'ema_trend': 50,
    'ema_regime': 200,
    'macd_fast': 12,
    'macd_slow': 26,
    'macd_signal': 9
}

SIGNAL_CONFIG = {
    'min_conditions': 3,
    'use_trend_filter': True,
    'use_volume_filter': True,
    'prevent_conflicts': True,
    'min_signal_strength': 5,    # نطاق مرن لقوة الإشارة
    'max_signal_strength': 10,
    'require_trend_confirmation': True,
    'min_volume_ratio': 1.0
}

TELEGRAM_CONFIG = {
    'bot_token': os.getenv('TELEGRAM_BOT_TOKEN', ''),
    'chat_id': os.getenv('TELEGRAM_CHAT_ID', '')
}

# ====================== إعداد التسجيل ======================
if ENABLE_LOGGING:
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s | %(levelname)s | %(message)s',
        handlers=[logging.StreamHandler()]
    )
logger = logging.getLogger(__name__) if ENABLE_LOGGING else None

# ====================== الكلاس الرئيسي المطور ======================
class AdvancedCryptoBot:
    def __init__(self, trade_config, indicator_config, signal_config):
        self.trade_config = trade_config
        self.indicator_config = indicator_config
        self.signal_config = signal_config
        self.data = None
        self.positions = []
        self.trades = []
        self.current_balance = trade_config['initial_balance']
        self.initial_balance = trade_config['initial_balance']
        self.paper_trading = trade_config.get('paper_trading', True)
        self.analysis_results = {}
        self.signal_strength_results = {}

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, max=10))
    def fetch_binance_data(self, days=60):
        """جلب كل البيانات المطلوبة مع حلقة ذكية - من الكود الأول"""
        try:
            symbol = self.trade_config['symbol']
            interval = self.trade_config['timeframe']
            limit = 1000
            all_data = []
            end_time = int(datetime.now().timestamp() * 1000)
            interval_h = interval_to_hours(interval)
            required_candles = int(days * 24 / interval_h) + 100

            if ENABLE_LOGGING:
                logger.info(f"جلب {required_candles} شمعة من {symbol} ({interval})")

            while len(all_data) < required_candles:
                params = {
                    'symbol': symbol,
                    'interval': interval,
                    'limit': min(limit, required_candles - len(all_data)),
                    'endTime': end_time
                }
                response = requests.get("https://api.binance.com/api/v3/klines", params=params, timeout=15)
                response.raise_for_status()
                data = response.json()
                if not data or len(data) == 0:
                    break
                all_data = data + all_data
                end_time = data[0][0] - 1

            df = pd.DataFrame(all_data, columns=[
                'timestamp', 'open', 'high', 'low', 'close', 'volume',
                'close_time', 'quote_asset_volume', 'number_of_trades',
                'taker_buy_base_asset_volume', 'taker_buy_quote_asset_volume', 'ignore'
            ])

            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
            for col in ['open', 'high', 'low', 'close', 'volume']:
                df[col] = pd.to_numeric(df[col], errors='coerce')

            df = df[['timestamp', 'open', 'high', 'low', 'close', 'volume']]
            df = df.sort_values('timestamp').reset_index(drop=True)
            df = df.drop_duplicates(subset='timestamp')

            self.data = df
            self.calculate_indicators()
            if ENABLE_LOGGING:
                logger.info(f"تم جلب {len(self.data)} شمعة بنجاح")

        except Exception as e:
            if ENABLE_LOGGING:
                logger.error(f"خطأ في جلب البيانات: {e}")
            self.generate_sample_data(days)

    def generate_sample_data(self, days):
        """بيانات عينة آمنة - من الكود الأول"""
        try:
            end_date = datetime.now()
            start_date = end_date - timedelta(days=days)
            interval_h = interval_to_hours(self.trade_config['timeframe'])
            freq_minutes = int(interval_h * 60)
            dates = pd.date_range(start=start_date, end=end_date, freq=f'{freq_minutes}T')

            np.random.seed(42)
            price = 300.0
            prices = []
            for _ in range(len(dates)):
                change = np.random.normal(0, 0.003)
                price *= (1 + change)
                prices.append(price)

            self.data = pd.DataFrame({
                'timestamp': dates,
                'open': prices,
                'high': [p * (1 + abs(np.random.normal(0, 0.002))) for p in prices],
                'low': [p * (1 - abs(np.random.normal(0, 0.002))) for p in prices],
                'close': prices,
                'volume': [abs(np.random.normal(1000, 300)) for _ in prices]
            })
            self.calculate_indicators()
            if ENABLE_LOGGING:
                logger.info(f"تم إنشاء {len(self.data)} شمعة عينة")
        except Exception as e:
            if ENABLE_LOGGING:
                logger.error(f"خطأ في إنشاء بيانات عينة: {e}")

    def calculate_atr(self, period=14):
        """حساب ATR - من الكود الأول"""
        high = self.data['high']
        low = self.data['low']
        close = self.data['close']
        tr0 = abs(high - low)
        tr1 = abs(high - close.shift())
        tr2 = abs(low - close.shift())
        tr = pd.concat([tr0, tr1, tr2], axis=1).max(axis=1)
        return tr.rolling(period).mean()

    def calculate_indicators(self):
        """حساب المؤشرات الفنية مع دمج أفضل الممارسات"""
        df = self.data
        p = self.indicator_config
        t = self.trade_config

        # RSI - بمنطق أكثر قوة
        delta = df['close'].diff()
        gain = (delta.where(delta > 0, 0)).fillna(0)
        loss = (-delta.where(delta < 0, 0)).fillna(0)
        
        avg_gain = gain.rolling(window=p['rsi_period'], min_periods=1).mean()
        avg_loss = loss.rolling(window=p['rsi_period'], min_periods=1).mean()
        
        rs = avg_gain / avg_loss
        rs = rs.replace([np.inf, -np.inf], 0).fillna(0)
        df['rsi'] = 100 - (100 / (1 + rs))

        # المتوسطات المتحركة
        df['ema_fast'] = df['close'].ewm(span=p['ema_fast'], adjust=False, min_periods=1).mean()
        df['ema_slow'] = df['close'].ewm(span=p['ema_slow'], adjust=False, min_periods=1).mean()
        df['ema_trend'] = df['close'].ewm(span=p['ema_trend'], adjust=False, min_periods=1).mean()
        df['ema_regime'] = df['close'].ewm(span=p['ema_regime'], adjust=False, min_periods=1).mean()

        # MACD
        ema_fast = df['close'].ewm(span=p['macd_fast'], adjust=False, min_periods=1).mean()
        ema_slow = df['close'].ewm(span=p['macd_slow'], adjust=False, min_periods=1).mean()
        df['macd'] = ema_fast - ema_slow
        df['macd_signal'] = df['macd'].ewm(span=p['macd_signal'], adjust=False, min_periods=1).mean()
        df['macd_histogram'] = df['macd'] - df['macd_signal']

        # Volume MA
        df['volume_ma'] = df['volume'].rolling(20, min_periods=1).mean()

        # ATR
        if ENABLE_ATR_SL_TP:
            df['atr'] = self.calculate_atr(t['atr_period'])

        # Support & Resistance
        if ENABLE_SUPPORT_RESISTANCE_FILTER:
            window = t['support_resistance_window']
            df['resistance'] = df['high'].rolling(window).max()
            df['support'] = df['low'].rolling(window).min()

        self.data = df
        if ENABLE_LOGGING:
            logger.info("تم حساب جميع المؤشرات")

    def get_market_regime(self, row):
        """تحديد نظام السوق - من الكود الأول"""
        if not ENABLE_MARKET_REGIME_FILTER:
            return "NEUTRAL"
        price = row['close']
        ema200 = row['ema_regime']
        if price > ema200 * 1.05:
            return "BULL"
        elif price < ema200 * 0.95:
            return "BEAR"
        else:
            return "SIDEWAYS"

    def calculate_signal_strength(self, buy_conditions, sell_conditions, row):
        """
        حساب قوة الإشارة المتدرجة من 1-10 
        مبنية على منطق الكود الثاني مع تحسينات الكود الأول
        """
        try:
            # الأساس = الحد الأقصى لعدد الشروط (من الكود الثاني)
            base_conditions = max(buy_conditions, sell_conditions)
            
            if base_conditions == 0:
                return 1
            
            # الحصول على قيم المؤشرات مع معالجة القيم NaN
            rsi = row['rsi'] if not pd.isna(row['rsi']) else 50
            ema_trend_position = 1 if row['close'] > row['ema_trend'] else 0
            macd_strength = abs(row['macd_histogram']) / row['close'] * 1000 if not pd.isna(row['macd_histogram']) else 0
            volume_strength = min(row['volume'] / row['volume_ma'], 3) if not pd.isna(row['volume_ma']) and row['volume_ma'] > 0 else 1
            
            # حساب نقاط القوة الإضافية (منطق الكود الثاني مع تحسينات)
            strength_points = 0
            
            # قوة RSI - تحسين المنطق
            if (buy_conditions > sell_conditions and rsi < 25) or (sell_conditions > buy_conditions and rsi > 75):
                strength_points += 2
                if ENABLE_LOGGING:
                    logger.debug("+2 نقاط لـ RSI متطرف")
            elif (buy_conditions > sell_conditions and rsi < 30) or (sell_conditions > buy_conditions and rsi > 70):
                strength_points += 1
                if ENABLE_LOGGING:
                    logger.debug("+1 نقطة لـ RSI قوي")
            
            # قوة الاتجاه - من الكود الأول مع تحسين
            if (buy_conditions > sell_conditions and ema_trend_position == 1) or \
               (sell_conditions > buy_conditions and ema_trend_position == 0):
                strength_points += 1
                if ENABLE_LOGGING:
                    logger.debug("+1 نقطة لمطابقة الاتجاه")
            
            # قوة MACD - تحسين العتبات
            if macd_strength > 0.8:  # عتبة أعلى لدقة أفضل
                strength_points += 1
                if ENABLE_LOGGING:
                    logger.debug("+1 نقطة لـ MACD قوي")
            elif macd_strength > 0.5:
                strength_points += 0.5
                if ENABLE_LOGGING:
                    logger.debug("+0.5 نقطة لـ MACD متوسط")
            
            # قوة الحجم - من الكود الأول مع عتبات متدرجة
            if volume_strength > 2.0:
                strength_points += 1.5
                if ENABLE_LOGGING:
                    logger.debug("+1.5 نقطة لحجم عالي جداً")
            elif volume_strength > 1.5:
                strength_points += 1
                if ENABLE_LOGGING:
                    logger.debug("+1 نقطة لحجم عالي")
            elif volume_strength > 1.2:
                strength_points += 0.5
                if ENABLE_LOGGING:
                    logger.debug("+0.5 نقطة لحجم جيد")
            
            # إضافة تصفية نظام السوق (من الكود الأول)
            regime = self.get_market_regime(row)
            if regime == "BULL" and buy_conditions > sell_conditions:
                strength_points += 0.5
            elif regime == "BEAR" and sell_conditions > buy_conditions:
                strength_points += 0.5
            
            # حساب القوة النهائية (1-10)
            total_strength = min(base_conditions + strength_points, 10)
            total_strength = max(total_strength, 1)
            
            # تسجيل النتائج للتحليل
            self.signal_strength_results = {
                'base_conditions': base_conditions,
                'strength_points': strength_points,
                'total_strength': total_strength,
                'rsi': rsi,
                'macd_strength': macd_strength,
                'volume_strength': volume_strength,
                'regime': regime
            }
            
            if ENABLE_LOGGING:
                logger.debug(f"قوة الإشارة: {base_conditions} + {strength_points:.1f} = {total_strength:.1f}")
            
            return total_strength
            
        except Exception as e:
            if ENABLE_LOGGING:
                logger.error(f"خطأ في حساب قوة الإشارة: {e}")
            return 1

    def generate_signal(self, row):
        """توليد إشارات تداول مع قوة متدرجة محسنة"""
        try:
            # التحقق من البيانات الناقصة
            required_columns = ['rsi', 'ema_fast', 'ema_slow', 'macd', 'ema_trend', 'volume_ma']
            if any(pd.isna(row[col]) for col in required_columns):
                return 'HOLD', 1, "بيانات ناقصة"

            buy_conditions = 0
            sell_conditions = 0
            condition_details = []

            # 1. شرط RSI
            if row['rsi'] < self.indicator_config['rsi_oversold']:
                buy_conditions += 1
                condition_details.append("RSI منخفض")
            elif row['rsi'] > self.indicator_config['rsi_overbought']:
                sell_conditions += 1
                condition_details.append("RSI مرتفع")

            # 2. شرط EMA
            if row['ema_fast'] > row['ema_slow']:
                buy_conditions += 1
                condition_details.append("EMA صاعد")
            else:
                sell_conditions += 1
                condition_details.append("EMA هابط")

            # 3. شرط MACD
            macd_strength = abs(row['macd_histogram']) > (row['close'] * 0.001)
            if row['macd'] > row['macd_signal'] and macd_strength:
                buy_conditions += 1
                condition_details.append("MACD صاعد")
            elif row['macd'] < row['macd_signal'] and macd_strength:
                sell_conditions += 1
                condition_details.append("MACD هابط")

            # 4. فلتر الاتجاه
            if self.signal_config['use_trend_filter']:
                if row['close'] > row['ema_trend']:
                    buy_conditions += 1
                    condition_details.append("فوق المتوسط 50")
                else:
                    sell_conditions += 1
                    condition_details.append("تحت المتوسط 50")

            # 5. فلتر الحجم
            volume_ratio = row['volume'] / row['volume_ma'] if row['volume_ma'] > 0 else 1
            volume_ok = volume_ratio > self.signal_config.get('min_volume_ratio', 1.0)
            if not volume_ok:
                condition_details.append("الحجم ضعيف")

            # حساب قوة الإشارة المتدرجة (المنطق المطور)
            signal_strength = self.calculate_signal_strength(buy_conditions, sell_conditions, row)

            # تطبيق الفلاتر الإضافية من الكود الأول
            regime = self.get_market_regime(row)
            regime_ok = not ENABLE_MARKET_REGIME_FILTER or \
                       (regime != "BEAR" if buy_conditions > sell_conditions else regime != "BULL")

            # فلتر الوقت
            hour = row['timestamp'].hour
            time_ok = not ENABLE_TIME_FILTER or hour in self.trade_config['peak_hours']

            # فلتر الدعم والمقاومة
            near_level = False
            if ENABLE_SUPPORT_RESISTANCE_FILTER and 'resistance' in row:
                dist_r = abs(row['close'] - row['resistance']) / row['close']
                dist_s = abs(row['close'] - row['support']) / row['close']
                near_level = min(dist_r, dist_s) < 0.003

            # اتخاذ القرار النهائي
            signal = 'HOLD'
            min_conditions = self.signal_config['min_conditions']
            min_strength = self.signal_config.get('min_signal_strength', 5)
            max_strength = self.signal_config.get('max_signal_strength', 10)

            # التحقق من أن قوة الإشارة ضمن النطاق المطلوب
            strength_in_range = min_strength <= signal_strength <= max_strength

            if (buy_conditions >= min_conditions and 
                strength_in_range and
                volume_ok and regime_ok and time_ok and not near_level):
                signal = 'BUY'

            elif (sell_conditions >= min_conditions and 
                  strength_in_range and
                  volume_ok and regime_ok and time_ok and not near_level):
                signal = 'SELL'

            details = " | ".join(condition_details) if condition_details else "لا توجد إشارات قوية"
            
            # إضافة معلومات القوة للتفاصيل
            if signal != 'HOLD':
                details += f" | قوة: {signal_strength:.1f}/10"
            
            return signal, signal_strength, details
            
        except Exception as e:
            if ENABLE_LOGGING:
                logger.error(f"خطأ في توليد الإشارة: {e}")
            return 'HOLD', 1, f"خطأ: {str(e)}"

    def open_position(self, direction, signal_strength, row, details):
        """فتح صفقة جديدة مع حجم ديناميكي بناء على قوة الإشارة"""
        try:
            # حجم المركز الديناميكي بناء على قوة الإشارة
            base_size = self.trade_config['base_position_size']
            if ENABLE_DYNAMIC_POSITION_SIZING:
                # زيادة حجم المركز مع زيادة قوة الإشارة
                size_factor = 0.5 + (signal_strength / 20)  # من 0.5 إلى 1.0
                position_value = self.current_balance * base_size * size_factor * self.trade_config['leverage']
            else:
                position_value = self.current_balance * base_size * self.trade_config['leverage']

            entry = row['close']
            
            # حساب وقف الخسارة وجني الأرباح
            if ENABLE_ATR_SL_TP and 'atr' in row and not pd.isna(row['atr']):
                atr = row['atr']
                if direction == 'BUY':
                    sl = entry - (self.trade_config['atr_multiplier_sl'] * atr)
                    tp = entry + (self.trade_config['atr_multiplier_tp'] * atr)
                else:
                    sl = entry + (self.trade_config['atr_multiplier_sl'] * atr)
                    tp = entry - (self.trade_config['atr_multiplier_tp'] * atr)
            else:
                if direction == 'BUY':
                    sl = entry * (1 - self.trade_config['base_stop_loss'])
                    tp = entry * (1 + self.trade_config['base_take_profit'])
                else:
                    sl = entry * (1 + self.trade_config['base_stop_loss'])
                    tp = entry * (1 - self.trade_config['base_take_profit'])

            position = {
                'id': len(self.trades) + len(self.positions) + 1,
                'direction': direction,
                'entry_price': float(entry),
                'entry_time': row['timestamp'],
                'size': float(position_value),
                'stop_loss': float(sl),
                'take_profit': float(tp),
                'status': 'OPEN',
                'type': 'PAPER' if self.paper_trading else 'REAL',
                'signal_strength': signal_strength,
                'signal_details': details,
                'entry_rsi': float(row['rsi']),
                'entry_macd': float(row['macd']),
                'volume_ratio': float(row['volume'] / row['volume_ma']) if row['volume_ma'] > 0 else 1.0,
                'trailing_stop': float(sl)
            }
            
            self.positions.append(position)
            
            if ENABLE_LOGGING:
                strength_emoji = "💪" * min(int(signal_strength / 2), 5)
                size_percent = (position_value / self.current_balance) * 100
                logger.info(f"فتح {direction} #{position['id']} | قوة: {signal_strength:.1f}/10 {strength_emoji}")
                logger.info(f"الحجم: ${position_value:.2f} ({size_percent:.1f}%) | {details}")
                
        except Exception as e:
            if ENABLE_LOGGING:
                logger.error(f"خطأ فتح الصفقة: {e}")

    # باقي الدوال (execute_backtest, check_exit_conditions, analyze_trades, etc.)
    # تبقى كما هي في الكود الأول مع التعديلات الطفيفة لدعم النظام الجديد
    
    def execute_backtest(self):
        """تنفيذ الباك تستينغ مع النظام المطور"""
        if ENABLE_WALK_FORWARD:
            split = int(len(self.data) * 0.7)
            train_data = self.data.iloc[:split].copy()
            test_data = self.data.iloc[split:].copy()
            datasets = [(train_data, "تدريب"), (test_data, "اختبار")]
        else:
            datasets = [(self.data, "كامل")]

        for data, name in datasets:
            if ENABLE_LOGGING:
                logger.info(f"باك تست: {name} ({len(data)} شمعة) - قوة الإشارة: {self.signal_config['min_signal_strength']}-{self.signal_config['max_signal_strength']}")
            self._run_backtest_on_data(data)

    def _run_backtest_on_data(self, data):
        """تشغيل الباك تست على البيانات مع النظام المطور"""
        min_period = 200
        for i, row in data.iterrows():
            if i < min_period:
                continue
            signal, strength, details = self.generate_signal(row)
            self.check_exit_conditions(row)

            open_pos = len([p for p in self.positions if p['status'] == 'OPEN'])
            if signal in ['BUY', 'SELL'] and open_pos < self.trade_config['max_positions']:
                self.open_position(signal, strength, row, details)

    def check_exit_conditions(self, row):
        """فحص شروط الخروج مع التداول المتعقب المحسن"""
        current_price = row['close']
        for pos in [p for p in self.positions if p['status'] == 'OPEN']:
            pnl = 0.0
            reason = ''
            duration = (row['timestamp'] - pos['entry_time']).total_seconds() / 3600

            # التداول المتعقب المحسن
            if self.trade_config['use_trailing_stop']:
                if pos['direction'] == 'BUY':
                    profit = (current_price - pos['entry_price']) / pos['entry_price']
                    if profit > self.trade_config['trailing_activation']:
                        new_sl = current_price * (1 - self.trade_config['trailing_stop_percent'])
                        pos['trailing_stop'] = max(pos['trailing_stop'], new_sl)
                    if current_price <= pos['trailing_stop']:
                        pnl = profit
                        reason = 'TRAILING_STOP'
                else:
                    profit = (pos['entry_price'] - current_price) / pos['entry_price']
                    if profit > self.trade_config['trailing_activation']:
                        new_sl = current_price * (1 + self.trade_config['trailing_stop_percent'])
                        pos['trailing_stop'] = min(pos['trailing_stop'], new_sl)
                    if current_price >= pos['trailing_stop']:
                        pnl = profit
                        reason = 'TRAILING_STOP'

            if reason:
                pass
            elif duration > self.trade_config['max_trade_duration']:
                pnl = (current_price - pos['entry_price']) / pos['entry_price'] if pos['direction'] == 'BUY' else (pos['entry_price'] - current_price) / pos['entry_price']
                reason = 'TIME_EXIT'
            elif pos['direction'] == 'BUY':
                if current_price <= pos['stop_loss']:
                    pnl = (current_price - pos['entry_price']) / pos['entry_price']
                    reason = 'STOP_LOSS'
                elif current_price >= pos['take_profit']:
                    pnl = (current_price - pos['entry_price']) / pos['entry_price']
                    reason = 'TAKE_PROFIT'
            else:
                if current_price >= pos['stop_loss']:
                    pnl = (pos['entry_price'] - current_price) / pos['entry_price']
                    reason = 'STOP_LOSS'
                elif current_price <= pos['take_profit']:
                    pnl = (pos['entry_price'] - current_price) / pos['entry_price']
                    reason = 'TAKE_PROFIT'

            if reason:
                pos.update({
                    'status': 'CLOSED',
                    'exit_price': current_price,
                    'exit_time': row['timestamp'],
                    'pnl': pnl * self.trade_config['leverage'],
                    'reason': reason,
                    'duration_hours': duration
                })
                self.current_balance += pos['size'] * pos['pnl']
                self.trades.append(pos.copy())
                self.positions.remove(pos)
                if ENABLE_LOGGING:
                    pnl_percent = pos['pnl'] * 100
                    emoji = "🟢" if pnl_percent > 0 else "🔴"
                    logger.info(f"{emoji} إغلاق {pos['direction']} #{pos['id']} | {reason} | {pnl_percent:+.2f}%")

    def analyze_trades(self):
        """تحليل مفصل للصفقات مع تحسينات قوة الإشارة"""
        if not self.trades:
            return {}
        
        df = pd.DataFrame(self.trades)
        
        # تحليل حسب قوة الإشارة (التحليل المحسن)
        signal_performance = df.groupby('signal_strength').agg({
            'pnl': ['mean', 'count', 'sum'],
            'size': 'mean',
            'duration_hours': 'mean'
        }).round(4)
        
        # تحليل إضافي حسب أسباب الإغلاق والوقت
        reason_analysis = df.groupby('reason').agg({
            'pnl': ['mean', 'count', 'sum'],
            'signal_strength': 'mean'
        }).round(4)
        
        # تحليل الأداء حسب قوة الإشارة
        strength_stats = {}
        for strength in range(1, 11):
            strength_trades = df[df['signal_strength'] == strength]
            if not strength_trades.empty:
                win_rate = (strength_trades['pnl'] > 0).mean() * 100
                avg_pnl = strength_trades['pnl'].mean() * 100
                strength_stats[strength] = {
                    'count': len(strength_trades),
                    'win_rate': win_rate,
                    'avg_pnl': avg_pnl
                }
        
        analysis = {
            'signal_performance': signal_performance,
            'reason_analysis': reason_analysis,
            'strength_stats': strength_stats,
            'total_trades': len(df),
            'win_rate': (df['pnl'] > 0).mean() * 100,
            'avg_win': df[df['pnl'] > 0]['pnl'].mean() * 100,
            'avg_loss': df[df['pnl'] <= 0]['pnl'].mean() * 100,
            'best_strength': max(strength_stats.items(), key=lambda x: x[1]['avg_pnl'])[0] if strength_stats else None
        }
        
        self.analysis_results = analysis
        return analysis

    def generate_detailed_report(self):
        """توليد تقرير مفصل مع تحليل قوة الإشارة"""
        if not self.trades:
            return "لا توجد صفقات لتحليلها"
        
        analysis = self.analyze_trades()
        
        report = f"""
📊 **تقرير أداء مفصل - البوت المطور**

**🎯 تحليل قوة الإشارة:**
"""
        
        # تحليل أداء قوة الإشارة
        for strength in sorted(analysis.get('strength_stats', {}).keys()):
            stats = analysis['strength_stats'][strength]
            report += f"• قوة {strength}: {stats['count']} صفقات | ربح {stats['win_rate']:.1f}% | متوسط {stats['avg_pnl']:+.2f}%\n"

        if analysis.get('best_strength'):
            report += f"• أفضل قوة إشارة: {analysis['best_strength']}\n"

        report += f"""
**📈 الإحصائيات الأساسية:**
• إجمالي الصفقات: {analysis['total_trades']}
• معدل الفوز: {analysis['win_rate']:.1f}%
• متوسط الربح: {analysis['avg_win']:+.2f}%
• متوسط الخسارة: {analysis['avg_loss']:.2f}%

**💡 توصيات بناء على قوة الإشارة:**
"""
        
        # توليد توصيات ذكية بناء على تحليل قوة الإشارة
        if analysis.get('strength_stats'):
            best_strength = analysis['best_strength']
            if best_strength and best_strength >= 7:
                report += f"• التركيز على الإشارات بقوة {best_strength}+ للحصول على أفضل النتائج\n"
            elif analysis['win_rate'] < 50:
                report += "• زيادة الحد الأدنى لقوة الإشارة لتحسين الجودة\n"
            else:
                report += "• توزيع قوة الإشارة متوازن - الحفاظ على الإعدادات الحالية\n"

        return report

# ====================== التشغيل ======================
def main():
    if ENABLE_LOGGING:
        logger.info("بدء تشغيل البوت المطور مع نظام قوة الإشارة المحسن")
    
    bot = AdvancedCryptoBot(TRADE_CONFIG, INDICATOR_CONFIG, SIGNAL_CONFIG)
    bot.fetch_binance_data(days=60)
    bot.execute_backtest()
    
    print(bot.generate_report())
    
    if ENABLE_DETAILED_REPORT:
        detailed_report = bot.generate_detailed_report()
        print(detailed_report)
    
    bot.send_detailed_telegram_report()

if __name__ == "__main__":
    main()
