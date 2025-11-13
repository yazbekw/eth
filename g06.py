# ====================== المكتبات ======================
import os
import pandas as pd
import numpy as np
import requests
from datetime import datetime, timedelta
import warnings
import logging
from tenacity import retry, stop_after_attempt, wait_exponential
from dotenv import load_dotenv
import schedule
import time
from telegram import Bot
from telegram.error import TelegramError

warnings.filterwarnings('ignore')
load_dotenv()

# ====================== إعدادات التفعيل ======================
ENABLE_TRAILING_STOP = True
ENABLE_DYNAMIC_POSITION_SIZING = False
ENABLE_MARKET_REGIME_FILTER = False
ENABLE_ATR_SL_TP = False
ENABLE_SUPPORT_RESISTANCE_FILTER = True
ENABLE_TIME_FILTER = True
ENABLE_LOGGING = True
ENABLE_DAILY_REPORT = True

# ====================== دالة مساعدة ======================
def interval_to_hours(interval):
    mapping = {
        '1m': 1/60, '3m': 3/60, '5m': 5/60, '15m': 15/60,
        '30m': 30/60, '1h': 1, '2h': 2, '4h': 4, '6h': 6,
        '8h': 8, '12h': 12, '1d': 24, '3d': 72, '1w': 168
    }
    return mapping.get(interval, 4)

# ====================== الإعدادات الأساسية ======================
TRADE_CONFIG = {
    'symbols': ['BTCUSDT', 'BNBUSDT', 'ETHUSDT', 'ADAUSDT', 'DOTUSDT', 'LINKUSDT', 'LTCUSDT', 'XRPUSDT'],
    'timeframe': '4h',
    'initial_balance': 200,
    'leverage': 1,
    'base_stop_loss': 0.025,
    'base_take_profit': 0.065,
    'base_position_size': 0.25,
    'max_positions': 4,
    'paper_trading': False,
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
    'rsi_period': 21,
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
    'max_signal_strength': 10,
    'require_trend_confirmation': True,
    'min_volume_ratio': 1.0
}

BINANCE_CONFIG = {
    'api_key': os.getenv('BINANCE_API_KEY', ''),
    'api_secret': os.getenv('BINANCE_API_SECRET', ''),
    'base_url': 'https://demo.binance.com/api/v3/'  # تغيير إلى Demo
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

# ====================== الكلاس الرئيسي ======================
class AdvancedCryptoBot:
    def __init__(self, trade_config, indicator_config, signal_config, binance_config):
        self.trade_config = trade_config
        self.indicator_config = indicator_config
        self.signal_config = signal_config
        self.binance_config = binance_config
        self.data = {}
        self.positions = []
        self.trades = []
        self.paper_trading = trade_config.get('paper_trading', False)
        
        # تهيئة جلسة requests
        self.session = requests.Session()
        if binance_config['api_key']:
            self.session.headers.update({
                'X-MBX-APIKEY': binance_config['api_key']
            })
        
        # جلب الرصيد
        self.initial_balance = self.get_testnet_balance()
        self.current_balance = self.initial_balance
        
        self.balance_history = [self.initial_balance]
        self.daily_trades = []
        self.last_daily_report = datetime.now().date()
        
        if ENABLE_LOGGING:
            logger.info(f"💰 الرصيد المبدئي: ${self.initial_balance:.2f}")

    def make_request(self, endpoint, params=None, method='GET', signed=False):
        """تنفيذ طلبات HTTP إلى Binance API"""
        url = f"{self.binance_config['base_url']}{endpoint}"
        
        try:
            if signed and self.binance_config['api_secret']:
                if params is None:
                    params = {}
                params['timestamp'] = int(time.time() * 1000)
                
                # إنشاء التوقيع
                query_string = '&'.join([f"{k}={v}" for k, v in params.items()])
                import hmac
                import hashlib
                signature = hmac.new(
                    self.binance_config['api_secret'].encode('utf-8'),
                    query_string.encode('utf-8'),
                    hashlib.sha256
                ).hexdigest()
                params['signature'] = signature
            
            if method == 'GET':
                response = self.session.get(url, params=params, timeout=10)
            else:
                response = self.session.post(url, params=params, timeout=10)
            
            response.raise_for_status()
            return response.json()
            
        except Exception as e:
            if ENABLE_LOGGING:
                logger.error(f"❌ خطأ في الاتصال بـ Binance API: {e}")
            return None

    def get_testnet_balance(self):
        """جلب الرصيد الفعلي من حساب الديمو"""
        try:
            account_info = self.make_request('account', signed=True)
            if not account_info:
                if ENABLE_LOGGING:
                    logger.warning("⚠️ استخدام الرصيد الافتراضي")
                return self.trade_config['initial_balance']
            
            usdt_balance = 0.0
            for balance in account_info.get('balances', []):
                if balance['asset'] == 'USDT':
                    usdt_balance = float(balance['free'])
                    break
            
            if usdt_balance > 0:
                if ENABLE_LOGGING:
                    logger.info(f"💰 الرصيد الفعلي: {usdt_balance} USDT")
                return usdt_balance
            else:
                return self.trade_config['initial_balance']
                
        except Exception as e:
            if ENABLE_LOGGING:
                logger.error(f"❌ خطأ في جلب الرصيد: {e}")
            return self.trade_config['initial_balance']

    def get_current_testnet_balance(self):
        """جلب الرصيد الحالي"""
        try:
            account_info = self.make_request('account', signed=True)
            if not account_info:
                return self.current_balance
            
            for balance in account_info.get('balances', []):
                if balance['asset'] == 'USDT':
                    return float(balance['free'])
            return self.current_balance
        except Exception as e:
            if ENABLE_LOGGING:
                logger.error(f"❌ خطأ في جلب الرصيد الحالي: {e}")
            return self.current_balance

    def update_balance_from_testnet(self):
        """تحديث الرصيد المحلي"""
        new_balance = self.get_current_testnet_balance()
        if new_balance != self.current_balance:
            old_balance = self.current_balance
            self.current_balance = new_balance
            self.balance_history.append(new_balance)
            
            if ENABLE_LOGGING:
                change = new_balance - old_balance
                logger.info(f"📊 تحديث الرصيد: ${old_balance:.2f} → ${new_balance:.2f} ({change:+.2f})")
            
            return True
        return False

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, max=10))
    def fetch_binance_data(self, symbol, days=60):
        """جلب البيانات من Binance Testnet"""
        try:
            interval = self.trade_config['timeframe']
            limit = 1000
            all_data = []
            end_time = int(datetime.now().timestamp() * 1000)
            interval_h = interval_to_hours(interval)
            required_candles = int(days * 24 / interval_h) + 100

            if ENABLE_LOGGING:
                logger.info(f"📊 جلب البيانات لـ {symbol}")

            while len(all_data) < required_candles:
                params = {
                    'symbol': symbol,
                    'interval': interval,
                    'limit': min(limit, required_candles - len(all_data)),
                    'endTime': end_time
                }
                
                data = self.make_request('klines', params=params)
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

            self.data[symbol] = df
            self.calculate_indicators(symbol)
            
            if ENABLE_LOGGING:
                logger.info(f"✅ تم جلب {len(self.data[symbol])} شمعة لـ {symbol}")

        except Exception as e:
            if ENABLE_LOGGING:
                logger.error(f"❌ خطأ في جلب البيانات لـ {symbol}: {e}")
            self.generate_sample_data(symbol, days)

    def generate_sample_data(self, symbol, days):
        """بيانات عينة"""
        try:
            end_date = datetime.now()
            start_date = end_date - timedelta(days=days)
            interval_h = interval_to_hours(self.trade_config['timeframe'])
            freq_minutes = int(interval_h * 60)
            dates = pd.date_range(start=start_date, end=end_date, freq=f'{freq_minutes}T')

            np.random.seed(42)
            price = 30000.0 if 'BTC' in symbol else 2000.0
            prices = []
            for _ in range(len(dates)):
                change = np.random.normal(0, 0.003)
                price *= (1 + change)
                prices.append(price)

            self.data[symbol] = pd.DataFrame({
                'timestamp': dates,
                'open': prices,
                'high': [p * (1 + abs(np.random.normal(0, 0.002))) for p in prices],
                'low': [p * (1 - abs(np.random.normal(0, 0.002))) for p in prices],
                'close': prices,
                'volume': [abs(np.random.normal(1000, 300)) for _ in prices]
            })
            self.calculate_indicators(symbol)
            
            if ENABLE_LOGGING:
                logger.info(f"📈 تم إنشاء بيانات عينة لـ {symbol}")
        except Exception as e:
            if ENABLE_LOGGING:
                logger.error(f"❌ خطأ في إنشاء بيانات عينة لـ {symbol}: {e}")

    def calculate_indicators(self, symbol):
        """حساب المؤشرات الفنية"""
        df = self.data[symbol]
        p = self.indicator_config

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

        # Support & Resistance
        if ENABLE_SUPPORT_RESISTANCE_FILTER:
            window = self.trade_config['support_resistance_window']
            df['resistance'] = df['high'].rolling(window).max()
            df['support'] = df['low'].rolling(window).min()

        self.data[symbol] = df

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
        """حساب قوة الإشارة"""
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
            return max(total_strength, 1)
            
        except Exception as e:
            if ENABLE_LOGGING:
                logger.error(f"❌ خطأ في حساب قوة الإشارة: {e}")
            return 1

    def generate_signal(self, symbol, row):
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
                logger.error(f"❌ خطأ في توليد الإشارة لـ {symbol}: {e}")
            return 'HOLD', 1, f"خطأ: {str(e)}"

    def send_telegram_message(self, message):
        """إرسال رسالة عبر التلغرام"""
        if not TELEGRAM_CONFIG['bot_token'] or not TELEGRAM_CONFIG['chat_id']:
            return
        
        try:
            bot = Bot(token=TELEGRAM_CONFIG['bot_token'])
            bot.send_message(chat_id=TELEGRAM_CONFIG['chat_id'], text=message, parse_mode='Markdown')
        except Exception as e:
            if ENABLE_LOGGING:
                logger.error(f"❌ خطأ في إرسال رسالة التلغرام: {e}")

    def send_trade_notification(self, symbol, position, action, pnl_dollar=0, pnl_percent=0):
        """إرسال إشعار بالصفقة"""
        if action == "OPEN":
            emoji = "🟢" if position['direction'] == 'BUY' else "🔴"
            message = f"""
{emoji} **تم فتح صفقة جديدة** {emoji}

📊 **التفاصيل:**
• الزوج: {symbol}
• الاتجاه: {position['direction']}
• السعر: ${position['entry_price']:.2f}
• الحجم: ${position['size']:.2f}
• قوة الإشارة: {position['signal_strength']:.1f}/10

🛡️ **إدارة المخاطر:**
• وقف الخسارة: ${position['stop_loss']:.2f}
• جني الأرباح: ${position['take_profit']:.2f}

💰 **الرصيد الحالي:** ${self.current_balance:.2f}

⏰ **الوقت:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
"""
        else:
            emoji = "💰" if pnl_dollar > 0 else "💸"
            message = f"""
{emoji} **تم إغلاق الصفقة** {emoji}

📊 **النتيجة:**
• الزوج: {symbol}
• الاتجاه: {position['direction']}
• الربح/الخسارة: ${pnl_dollar:+.2f} ({pnl_percent:+.2f}%)
• المدة: {position['duration_hours']:.1f} ساعة

📈 **التفاصيل:**
• سعر الدخول: ${position['entry_price']:.2f}
• سعر الخروج: ${position['exit_price']:.2f}
• سبب الإغلاق: {position['reason']}

💳 **الرصيد الحالي:** ${self.current_balance:.2f}

⏰ **الوقت:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
"""
        
        self.send_telegram_message(message)

    def open_position(self, symbol, direction, signal_strength, row, details):
        """فتح صفقة جديدة"""
        try:
            self.update_balance_from_testnet()
            
            base_size = self.trade_config['base_position_size']
            if ENABLE_DYNAMIC_POSITION_SIZING:
                size_factor = 0.5 + (signal_strength / 20)
                position_value = self.current_balance * base_size * size_factor
            else:
                position_value = self.current_balance * base_size

            entry = row['close']
            
            if direction == 'BUY':
                sl = entry * (1 - self.trade_config['base_stop_loss'])
                tp = entry * (1 + self.trade_config['base_take_profit'])
            else:
                sl = entry * (1 + self.trade_config['base_stop_loss'])
                tp = entry * (1 - self.trade_config['base_take_profit'])

            position = {
                'id': len(self.trades) + len(self.positions) + 1,
                'symbol': symbol,
                'direction': direction,
                'entry_price': float(entry),
                'entry_time': row['timestamp'],
                'size': float(position_value),
                'stop_loss': float(sl),
                'take_profit': float(tp),
                'status': 'OPEN',
                'signal_strength': signal_strength,
                'signal_details': details,
                'trailing_stop': float(sl)
            }
            
            self.positions.append(position)
            self.daily_trades.append({
                **position,
                'action': 'OPEN',
                'timestamp': datetime.now()
            })
            
            self.send_trade_notification(symbol, position, "OPEN")
            
            if ENABLE_LOGGING:
                logger.info(f"🟢 فتح {direction} على {symbol} | قوة: {signal_strength:.1f}/10")
                logger.info(f"📦 الحجم: ${position_value:.2f} | الرصيد: ${self.current_balance:.2f}")
                
        except Exception as e:
            if ENABLE_LOGGING:
                logger.error(f"❌ خطأ فتح الصفقة على {symbol}: {e}")

    def check_exit_conditions(self, symbol, row):
        """فحص شروط الخروج"""
        current_price = row['close']
        for pos in [p for p in self.positions if p['status'] == 'OPEN' and p['symbol'] == symbol]:
            pnl_percent = 0.0
            reason = ''
            duration = (row['timestamp'] - pos['entry_time']).total_seconds() / 3600

            if pos['direction'] == 'BUY':
                pnl_percent = (current_price - pos['entry_price']) / pos['entry_price']
            else:
                pnl_percent = (pos['entry_price'] - current_price) / pos['entry_price']

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
                pnl_dollar = pos['size'] * pnl_percent
                
                pos.update({
                    'status': 'CLOSED',
                    'exit_price': current_price,
                    'exit_time': row['timestamp'],
                    'pnl_percent': pnl_percent * 100,
                    'pnl_dollar': pnl_dollar,
                    'reason': reason,
                    'duration_hours': duration
                })
                
                self.current_balance += pnl_dollar
                self.balance_history.append(self.current_balance)
                self.trades.append(pos.copy())
                self.daily_trades.append({
                    **pos,
                    'action': 'CLOSE',
                    'timestamp': datetime.now()
                })
                
                self.send_trade_notification(symbol, pos, "CLOSE", pnl_dollar, pnl_percent*100)
                
                self.positions.remove(pos)
                
                if ENABLE_LOGGING:
                    emoji = "💰" if pnl_dollar > 0 else "💸"
                    logger.info(f"{emoji} إغلاق {pos['direction']} على {symbol} | {reason} | ${pnl_dollar:+.2f}")

    def generate_daily_report(self):
        """توليد تقرير يومي"""
        try:
            today = datetime.now().date()
            today_trades = [t for t in self.daily_trades if t['timestamp'].date() == today]
            
            if not today_trades:
                return "📊 **تقرير الأداء اليومي**\n\n⚠️ لم تتم أي صفقات اليوم"
            
            closed_trades = [t for t in today_trades if t.get('status') == 'CLOSED']
            
            total_pnl = sum(t.get('pnl_dollar', 0) for t in closed_trades)
            winning_trades = len([t for t in closed_trades if t.get('pnl_dollar', 0) > 0])
            losing_trades = len([t for t in closed_trades if t.get('pnl_dollar', 0) < 0])
            
            report = f"""
📊 **تقرير الأداء اليومي** 
📅 {today.strftime('%Y-%m-%d')}
{'='*40}

📈 **إحصائيات الصفقات:**
• إجمالي الصفقات: {len(closed_trades)}
• الصفقات الرابحة: {winning_trades} 🟢
• الصفقات الخاسرة: {losing_trades} 🔴
• معدل الفوز: {(winning_trades/len(closed_trades)*100 if closed_trades else 0):.1f}%

💰 **الأداء المالي:**
• إجمالي الربح/الخسارة: ${total_pnl:+.2f}
• الرصيد الحالي: ${self.current_balance:.2f}

🔄 **الصفقات المفتوحة:** {len(self.positions)}
⚡ **الصفقات المغلقة:** {len(closed_trades)}

⏰ **آخر تحديث:** {datetime.now().strftime('%H:%M:%S')}
"""
            
            if closed_trades:
                report += "\n🔍 **آخر الصفقات:**\n"
                for trade in closed_trades[-3:]:
                    emoji = "🟢" if trade.get('pnl_dollar', 0) > 0 else "🔴"
                    report += f"• {trade['symbol']} {trade['direction']} | ${trade.get('pnl_dollar', 0):+.2f} {emoji}\n"
            
            return report
            
        except Exception as e:
            return f"❌ خطأ في توليد التقرير: {e}"

    def send_daily_report_telegram(self):
        """إرسال التقرير اليومي"""
        report = self.generate_daily_report()
        self.send_telegram_message(report)
        
        if ENABLE_LOGGING:
            logger.info("✅ تم إرسال التقرير اليومي")

    def run_live_signal_check(self):
        """فحص الإشارات لجميع الأزواج"""
        try:
            self.update_balance_from_testnet()
            
            open_positions_count = len([p for p in self.positions if p['status'] == 'OPEN'])
            
            for symbol in self.trade_config['symbols']:
                if ENABLE_LOGGING:
                    logger.info(f"🔍 فحص الإشارات لـ {symbol}")
                
                self.fetch_binance_data(symbol, days=7)
                
                if symbol not in self.data or self.data[symbol] is None or len(self.data[symbol]) == 0:
                    continue
                
                last_row = self.data[symbol].iloc[-1]
                signal, strength, details = self.generate_signal(symbol, last_row)
                
                self.check_exit_conditions(symbol, last_row)
                
                if signal in ['BUY', 'SELL'] and open_positions_count < self.trade_config['max_positions']:
                    self.open_position(symbol, signal, strength, last_row, details)
                    open_positions_count += 1
                    
            current_date = datetime.now().date()
            if ENABLE_DAILY_REPORT and current_date != self.last_daily_report:
                if datetime.now().hour == 23:
                    self.send_daily_report_telegram()
                    self.last_daily_report = current_date
                    self.daily_trades = []
                    
        except Exception as e:
            if ENABLE_LOGGING:
                logger.error(f"❌ خطأ في فحص الإشارات: {e}")

# ====================== تشغيل البوت ======================
def run_live_bot():
    """تشغيل البوت الرئيسي"""
    if ENABLE_LOGGING:
        logger.info("🚀 بدء تشغيل البوت")
    
    bot = AdvancedCryptoBot(TRADE_CONFIG, INDICATOR_CONFIG, SIGNAL_CONFIG, BINANCE_CONFIG)
    
    schedule.every(5).minutes.do(bot.run_live_signal_check)
    schedule.every(10).minutes.do(bot.update_balance_from_testnet)
    schedule.every().day.at("23:00").do(bot.send_daily_report_telegram)
    
    bot.run_live_signal_check()
    
    if ENABLE_LOGGING:
        logger.info("✅ تم بدء التشغيل")
    
    while True:
        try:
            schedule.run_pending()
            time.sleep(1)
        except KeyboardInterrupt:
            if ENABLE_LOGGING:
                logger.info("⏹️ إيقاف البوت")
            break
        except Exception as e:
            if ENABLE_LOGGING:
                logger.error(f"❌ خطأ: {e}")
            time.sleep(60)

if __name__ == "__main__":
    run_live_bot()
