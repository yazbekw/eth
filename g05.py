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
ENABLE_TRAILING_STOP = True
ENABLE_DYNAMIC_POSITION_SIZING = True
ENABLE_MARKET_REGIME_FILTER = False
ENABLE_ATR_SL_TP = False
ENABLE_SUPPORT_RESISTANCE_FILTER = True
ENABLE_TIME_FILTER = False
ENABLE_WALK_FORWARD = False
ENABLE_LOGGING = True
ENABLE_DETAILED_REPORT = True

# ====================== دالة مساعدة ======================
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
    'base_stop_loss': 0.02,
    'base_take_profit': 0.06,
    'base_position_size': 0.2,
    'max_positions': 4,
    'paper_trading': True,
    'use_trailing_stop': ENABLE_TRAILING_STOP,
    'trailing_stop_percent': 0.015,
    'trailing_activation': 0.02,
    'max_trade_duration': 48,
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
    'min_signal_strength': 5,
    'max_signal_strength': 5,
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
        self.balance_history = [trade_config['initial_balance']]

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, max=10))
    def fetch_binance_data(self, days=60):
        """جلب البيانات من Binance"""
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
        """بيانات عينة آمنة"""
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
        """حساب ATR"""
        high = self.data['high']
        low = self.data['low']
        close = self.data['close']
        tr0 = abs(high - low)
        tr1 = abs(high - close.shift())
        tr2 = abs(low - close.shift())
        tr = pd.concat([tr0, tr1, tr2], axis=1).max(axis=1)
        return tr.rolling(period).mean()

    def calculate_indicators(self):
        """حساب المؤشرات الفنية"""
        df = self.data
        p = self.indicator_config
        t = self.trade_config

        # RSI
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
        """تحديد نظام السوق"""
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
        """حساب قوة الإشارة المتدرجة من 1-10"""
        try:
            base_conditions = max(buy_conditions, sell_conditions)
            
            if base_conditions == 0:
                return 1
            
            rsi = row['rsi'] if not pd.isna(row['rsi']) else 50
            ema_trend_position = 1 if row['close'] > row['ema_trend'] else 0
            macd_strength = abs(row['macd_histogram']) / row['close'] * 1000 if not pd.isna(row['macd_histogram']) else 0
            volume_strength = min(row['volume'] / row['volume_ma'], 3) if not pd.isna(row['volume_ma']) and row['volume_ma'] > 0 else 1
            
            strength_points = 0
            
            # قوة RSI
            if (buy_conditions > sell_conditions and rsi < 25) or (sell_conditions > buy_conditions and rsi > 75):
                strength_points += 2
            elif (buy_conditions > sell_conditions and rsi < 30) or (sell_conditions > buy_conditions and rsi > 70):
                strength_points += 1
            
            # قوة الاتجاه
            if (buy_conditions > sell_conditions and ema_trend_position == 1) or \
               (sell_conditions > buy_conditions and ema_trend_position == 0):
                strength_points += 1
            
            # قوة MACD
            if macd_strength > 0.8:
                strength_points += 1
            elif macd_strength > 0.5:
                strength_points += 0.5
            
            # قوة الحجم
            if volume_strength > 2.0:
                strength_points += 1.5
            elif volume_strength > 1.5:
                strength_points += 1
            elif volume_strength > 1.2:
                strength_points += 0.5
            
            # نظام السوق
            regime = self.get_market_regime(row)
            if regime == "BULL" and buy_conditions > sell_conditions:
                strength_points += 0.5
            elif regime == "BEAR" and sell_conditions > buy_conditions:
                strength_points += 0.5
            
            total_strength = min(base_conditions + strength_points, 10)
            total_strength = max(total_strength, 1)
            
            return total_strength
            
        except Exception as e:
            if ENABLE_LOGGING:
                logger.error(f"خطأ في حساب قوة الإشارة: {e}")
            return 1

    def generate_signal(self, row):
        """توليد إشارات تداول"""
        try:
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

            # حساب قوة الإشارة
            signal_strength = self.calculate_signal_strength(buy_conditions, sell_conditions, row)

            # الفلاتر الإضافية
            regime = self.get_market_regime(row)
            regime_ok = not ENABLE_MARKET_REGIME_FILTER or \
                       (regime != "BEAR" if buy_conditions > sell_conditions else regime != "BULL")

            hour = row['timestamp'].hour
            time_ok = not ENABLE_TIME_FILTER or hour in self.trade_config['peak_hours']

            near_level = False
            if ENABLE_SUPPORT_RESISTANCE_FILTER and 'resistance' in row:
                dist_r = abs(row['close'] - row['resistance']) / row['close']
                dist_s = abs(row['close'] - row['support']) / row['close']
                near_level = min(dist_r, dist_s) < 0.003

            # القرار النهائي
            signal = 'HOLD'
            min_conditions = self.signal_config['min_conditions']
            min_strength = self.signal_config.get('min_signal_strength', 5)
            max_strength = self.signal_config.get('max_signal_strength', 10)

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
            
            if signal != 'HOLD':
                details += f" | قوة: {signal_strength:.1f}/10"
            
            return signal, signal_strength, details
            
        except Exception as e:
            if ENABLE_LOGGING:
                logger.error(f"خطأ في توليد الإشارة: {e}")
            return 'HOLD', 1, f"خطأ: {str(e)}"

    def open_position(self, direction, signal_strength, row, details):
        """فتح صفقة جديدة"""
        try:
            # حساب حجم المركز بدقة
            base_size = self.trade_config['base_position_size']
            if ENABLE_DYNAMIC_POSITION_SIZING:
                size_factor = 0.5 + (signal_strength / 20)
                position_value = self.current_balance * base_size * size_factor
            else:
                position_value = self.current_balance * base_size

            entry = row['close']
            
            # حساب وقف الخسارة وجني الأرباح
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
                'size': float(position_value),  # المبلغ المستثمر بالدولار
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

    def execute_backtest(self):
        """تنفيذ الباك تستينغ"""
        if ENABLE_WALK_FORWARD:
            split = int(len(self.data) * 0.7)
            train_data = self.data.iloc[:split].copy()
            test_data = self.data.iloc[split:].copy()
            datasets = [(train_data, "تدريب"), (test_data, "اختبار")]
        else:
            datasets = [(self.data, "كامل")]

        for data, name in datasets:
            if ENABLE_LOGGING:
                logger.info(f"باك تست: {name} ({len(data)} شمعة)")
            self._run_backtest_on_data(data)

    def _run_backtest_on_data(self, data):
        """تشغيل الباك تست على البيانات"""
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
        """فحص شروط الخروج"""
        current_price = row['close']
        for pos in [p for p in self.positions if p['status'] == 'OPEN']:
            pnl_percent = 0.0
            reason = ''
            duration = (row['timestamp'] - pos['entry_time']).total_seconds() / 3600

            # حساب الربح/الخسارة بالنسبة المئوية
            if pos['direction'] == 'BUY':
                pnl_percent = (current_price - pos['entry_price']) / pos['entry_price']
            else:
                pnl_percent = (pos['entry_price'] - current_price) / pos['entry_price']

            # التداول المتعقب
            if self.trade_config['use_trailing_stop']:
                if pos['direction'] == 'BUY':
                    if pnl_percent > self.trade_config['trailing_activation']:
                        new_sl = current_price * (1 - self.trade_config['trailing_stop_percent'])
                        pos['trailing_stop'] = max(pos['trailing_stop'], new_sl)
                    if current_price <= pos['trailing_stop']:
                        reason = 'TRAILING_STOP'
                else:
                    if pnl_percent > self.trade_config['trailing_activation']:
                        new_sl = current_price * (1 + self.trade_config['trailing_stop_percent'])
                        pos['trailing_stop'] = min(pos['trailing_stop'], new_sl)
                    if current_price >= pos['trailing_stop']:
                        reason = 'TRAILING_STOP'

            if reason:
                pass
            elif duration > self.trade_config['max_trade_duration']:
                reason = 'TIME_EXIT'
            elif pos['direction'] == 'BUY':
                if current_price <= pos['stop_loss']:
                    reason = 'STOP_LOSS'
                elif current_price >= pos['take_profit']:
                    reason = 'TAKE_PROFIT'
            else:
                if current_price >= pos['stop_loss']:
                    reason = 'STOP_LOSS'
                elif current_price <= pos['take_profit']:
                    reason = 'TAKE_PROFIT'

            if reason:
                # حساب الربح/الخسارة بالدولار بدقة
                pnl_dollar = pos['size'] * pnl_percent
                
                pos.update({
                    'status': 'CLOSED',
                    'exit_price': current_price,
                    'exit_time': row['timestamp'],
                    'pnl_percent': pnl_percent * 100,  # حفظ النسبة المئوية
                    'pnl_dollar': pnl_dollar,  # حفظ القيمة بالدولار
                    'reason': reason,
                    'duration_hours': duration
                })
                
                # تحديث الرصيد بدقة
                self.current_balance += pnl_dollar
                self.balance_history.append(self.current_balance)
                self.trades.append(pos.copy())
                self.positions.remove(pos)
                
                if ENABLE_LOGGING:
                    emoji = "🟢" if pnl_dollar > 0 else "🔴"
                    logger.info(f"{emoji} إغلاق {pos['direction']} #{pos['id']} | {reason} | ${pnl_dollar:+.2f} ({pnl_percent*100:+.2f}%)")

    def validate_calculations(self):
        """التحقق من صحة الحسابات"""
        if not self.trades:
            return "لا توجد صفقات للتحقق"
        
        issues = []
        total_calculated_pnl = 0
        
        for trade in self.trades:
            # حساب PNL يدوياً للتحقق
            if trade['direction'] == 'BUY':
                calculated_pnl_percent = (trade['exit_price'] - trade['entry_price']) / trade['entry_price']
            else:
                calculated_pnl_percent = (trade['entry_price'] - trade['exit_price']) / trade['entry_price']
            
            calculated_pnl_dollar = trade['size'] * calculated_pnl_percent
            total_calculated_pnl += calculated_pnl_dollar
            
            # التحقق من التطابق
            if abs(calculated_pnl_dollar - trade['pnl_dollar']) > 0.01:
                issues.append(f"الصفقة #{trade['id']}: محسوب ${calculated_pnl_dollar:.2f} vs مسجل ${trade['pnl_dollar']:.2f}")
        
        # التحقق من الرصيد النهائي
        expected_balance = self.initial_balance + total_calculated_pnl
        if abs(expected_balance - self.current_balance) > 0.01:
            issues.append(f"الرصيد: متوقع ${expected_balance:.2f} vs فعلي ${self.current_balance:.2f}")
        
        return "جميع الحسابات صحيحة" if not issues else issues

    def analyze_trades(self):
        """تحليل مفصل للصفقات"""
        if not self.trades:
            return {}
        
        df = pd.DataFrame(self.trades)
        
        # تحليل حسب قوة الإشارة
        strength_stats = {}
        for strength in range(1, 11):
            strength_trades = df[df['signal_strength'] == strength]
            if not strength_trades.empty:
                win_rate = (strength_trades['pnl_dollar'] > 0).mean() * 100
                avg_pnl_percent = strength_trades['pnl_percent'].mean()
                avg_pnl_dollar = strength_trades['pnl_dollar'].mean()
                strength_stats[strength] = {
                    'count': len(strength_trades),
                    'win_rate': win_rate,
                    'avg_pnl_percent': avg_pnl_percent,
                    'avg_pnl_dollar': avg_pnl_dollar
                }
        
        # تحليل أسباب الإغلاق
        reason_stats = df.groupby('reason').agg({
            'pnl_dollar': ['count', 'mean', 'sum'],
            'pnl_percent': 'mean'
        }).round(4)
        
        analysis = {
            'strength_stats': strength_stats,
            'reason_stats': reason_stats,
            'total_trades': len(df),
            'winning_trades': len(df[df['pnl_dollar'] > 0]),
            'losing_trades': len(df[df['pnl_dollar'] < 0]),
            'total_pnl_dollar': df['pnl_dollar'].sum(),
            'total_pnl_percent': (df['pnl_dollar'].sum() / self.initial_balance) * 100,
            'win_rate': (df['pnl_dollar'] > 0).mean() * 100,
            'avg_win_dollar': df[df['pnl_dollar'] > 0]['pnl_dollar'].mean(),
            'avg_loss_dollar': df[df['pnl_dollar'] < 0]['pnl_dollar'].mean(),
            'avg_win_percent': df[df['pnl_percent'] > 0]['pnl_percent'].mean(),
            'avg_loss_percent': df[df['pnl_percent'] < 0]['pnl_percent'].mean(),
        }
        
        # أفضل قوة إشارة
        if strength_stats:
            analysis['best_strength'] = max(strength_stats.items(), 
                                          key=lambda x: x[1]['avg_pnl_dollar'])[0]
        
        self.analysis_results = analysis
        return analysis

    def generate_report(self):
        """توليد تقرير الأداء الأساسي"""
        if not self.trades:
            return "⚠️ لا توجد صفقات"

        analysis = self.analyze_trades()
        validation = self.validate_calculations()

        report = f"""
📈 تقرير الأداء النهائي - النظام المطور
{'='*50}
• الرصيد الابتدائي: ${self.initial_balance:,.2f}
• الرصيد النهائي: ${self.current_balance:,.2f}
• إجمالي الربح/الخسارة: ${analysis['total_pnl_dollar']:+,.2f} ({analysis['total_pnl_percent']:+.2f}%)
{'='*50}
• إجمالي الصفقات: {analysis['total_trades']}
• الصفقات الرابحة: {analysis['winning_trades']}
• الصفقات الخاسرة: {analysis['losing_trades']}
• معدل الفوز: {analysis['win_rate']:.1f}%
{'='*50}
• متوسط الربح: ${analysis['avg_win_dollar']:+.2f} ({analysis['avg_win_percent']:+.2f}%)
• متوسط الخسارة: ${analysis['avg_loss_dollar']:.2f} ({analysis['avg_loss_percent']:.2f}%)
• نسبة الربح/الخسارة: {abs(analysis['avg_win_dollar']/analysis['avg_loss_dollar']):.2f}
{'='*50}
✅ التحقق من الحسابات: {validation}
"""

        return report

    def generate_detailed_report(self):
        """توليد تقرير مفصل"""
        if not self.trades:
            return "لا توجد صفقات لتحليلها"
    
        analysis = self.analyze_trades()
    
        report = f"""
📊 **تقرير أداء مفصل - البوت المطور**
{'='*60}

**🎯 تحليل قوة الإشارة:**
"""
    
        if analysis.get('strength_stats'):
            for strength in sorted(analysis['strength_stats'].keys()):
                stats = analysis['strength_stats'][strength]
                strength_emoji = "💪" * min(strength, 5)
                report += f"• {strength_emoji} قوة {strength}: {stats['count']} صفقات | ربح {stats['win_rate']:.1f}% | متوسط {stats['avg_pnl_percent']:+.2f}%\n"

        if analysis.get('best_strength'):
            report += f"\n**🏆 أفضل أداء:** قوة {analysis['best_strength']}"

        report += f"""
{'='*60}
**📈 الإحصائيات الأساسية:**
• إجمالي الصفقات: {analysis['total_trades']}
• معدل الفوز: {analysis['win_rate']:.1f}%
• إجمالي الربح: ${analysis['total_pnl_dollar']:+,.2f} ({analysis['total_pnl_percent']:+.2f}%)
• متوسط الربح: ${analysis['avg_win_dollar']:+.2f}
• متوسط الخسارة: ${analysis['avg_loss_dollar']:.2f}
{'='*60}
**🔍 تحليل أسباب الإغلاق:**
"""
    
        if not analysis['reason_stats'].empty:
            for reason in analysis['reason_stats'].index:
                count = analysis['reason_stats'].loc[reason, ('pnl_dollar', 'count')]
                avg_dollar = analysis['reason_stats'].loc[reason, ('pnl_dollar', 'mean')]
                avg_percent = analysis['reason_stats'].loc[reason, ('pnl_percent', 'mean')]
                report += f"• {reason}: {count} صفقات | ${avg_dollar:+.2f} ({avg_percent:+.2f}%)\n"

        report += f"""
{'='*60}
**💡 التوصيات:**
"""
        
        if analysis['win_rate'] < 50:
            report += "• ⚠️ زيادة الحد الأدنى لقوة الإشارة لتحسين الجودة\n"
            report += "• 📊 مراجعة شروط الدخول للإشارات منخفضة القوة\n"
        elif analysis['win_rate'] > 60:
            report += "• ✅ الأداء جيد - الحفاظ على الإعدادات الحالية\n"
            report += "• 💰 يمكن زيادة حجم المركز تدريجياً\n"
        else:
            report += "• 🔄 اختبار نطاق قوة أضيق للتحسين\n"

        report += f"""
{'='*60}
**⏰ الفترة:** {self.data['timestamp'].iloc[0].date()} إلى {self.data['timestamp'].iloc[-1].date()}
**🔄 آخر تحديث:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
"""

        return report

    def send_detailed_telegram_report(self):
        """إرسال تقرير عبر التلغرام"""
        if not TELEGRAM_CONFIG['bot_token'] or not TELEGRAM_CONFIG['chat_id']:
            if ENABLE_LOGGING:
                logger.warning("مفاتيح التلغرام غير متوفرة")
            return
        
        try:
            bot = telebot.TeleBot(TELEGRAM_CONFIG['bot_token'])
            
            basic_report = self.generate_report()
            bot.send_message(TELEGRAM_CONFIG['chat_id'], basic_report)
            
            if ENABLE_DETAILED_REPORT and self.trades:
                detailed_report = self.generate_detailed_report()
                
                if len(detailed_report) > 4000:
                    parts = [detailed_report[i:i+4000] for i in range(0, len(detailed_report), 4000)]
                    for part in parts:
                        bot.send_message(TELEGRAM_CONFIG['chat_id'], part)
                        import time
                        time.sleep(1)
                else:
                    bot.send_message(TELEGRAM_CONFIG['chat_id'], detailed_report)
                
                if ENABLE_LOGGING:
                    logger.info("تم إرسال التقرير المفصل إلى التلغرام")
                    
        except Exception as e:
            if ENABLE_LOGGING:
                logger.error(f"خطأ في إرسال التقرير التلغرام: {e}")

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
