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

# ====================== إعدادات استراتيجية الفيبوناتشي ======================
FIBONACCI_CONFIG = {
    'enabled': True,
    'trend_ema_period': 50,  # لتحديد الاتجاه العام
    'fibonacci_levels': [0.236, 0.382, 0.500, 0.618, 0.786],
    'key_levels': [0.382, 0.500, 0.618],  # المستويات الرئيسية للدخول
    'min_trend_strength': 0.02,  # قوة الاتجاه الأدنى (2%)
    'swing_period': 20,  # الفترة لتحديد القمم والقيعان
    'confirmation_candles': ['hammer', 'engulfing', 'doji', 'morning_star'],
    'rsi_oversold': 30,
    'rsi_overbought': 70,
    'volume_threshold': 1.2,  # زيادة الحجم عن المتوسط
}

TRADE_CONFIG = {
    'symbols': ['BTCUSDT', 'ETHUSDT', 'BNBUSDT', 'ADAUSDT', 'XRPUSDT', 'LTCUSDT'],
    'timeframe': '4h',
    'initial_balance': 1000,
    'leverage': 1,
    'position_size': 0.15,  # 15% من الرصيد لكل صفقة
    'max_positions': 3,
    'paper_trading': True,
    'stop_loss_percent': 0.02,  # 2%
    'take_profit_ratios': [1.0, 1.618],  # أهداف فيبوناتشي
    'risk_reward_ratio': 2.0,
}

TELEGRAM_CONFIG = {
    'bot_token': os.getenv('TELEGRAM_BOT_TOKEN', ''),
    'chat_id': os.getenv('TELEGRAM_CHAT_ID', '')
}

BINANCE_CONFIG = {
    'api_key': os.getenv('BINANCE_API_KEY', ''),
    'api_secret': os.getenv('BINANCE_API_SECRET', ''),
    'base_url': 'https://testnet.binance.vision/api/v3/'
}

# ====================== إعداد التسجيل ======================
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)s | %(message)s',
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger(__name__)

# ====================== الكلاس الرئيسي - استراتيجية الفيبوناتشي ======================
class FibonacciTradingBot:
    def __init__(self, fib_config, trade_config, binance_config):
        self.fib_config = fib_config
        self.trade_config = trade_config
        self.binance_config = binance_config
        self.data = {}
        self.positions = []
        self.trades = []
        
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
        
        logger.info(f"💰 الرصيد المبدئي: ${self.initial_balance:.2f}")
        logger.info(f"🎯 بدء تشغيل بوت الفيبوناتشي")

    def make_request(self, endpoint, params=None, method='GET', signed=False):
        """طلبات HTTP لـ Binance Testnet"""
        url = f"{self.binance_config['base_url']}{endpoint}"
        
        try:
            if signed and self.binance_config['api_secret']:
                if params is None:
                    params = {}
                params['timestamp'] = int(time.time() * 1000)
                
                query_string = '&'.join([f"{k}={v}" for k, v in sorted(params.items())])
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
            
            if response.status_code != 200:
                error_data = response.json()
                error_msg = error_data.get('msg', 'Unknown error')
                logger.error(f"❌ خطأ من API: {error_msg}")
                return None
            
            return response.json()
            
        except Exception as e:
            logger.error(f"❌ خطأ في الاتصال: {e}")
            return None

    def get_testnet_balance(self):
        """جلب الرصيد من Testnet"""
        try:
            account_info = self.make_request('account', signed=True)
            if not account_info:
                logger.warning("⚠️ استخدام الرصيد الافتراضي")
                return self.trade_config['initial_balance']
            
            usdt_balance = 0.0
            for balance in account_info.get('balances', []):
                if balance['asset'] == 'USDT':
                    usdt_balance = float(balance['free'])
                    break
            
            if usdt_balance > 0:
                logger.info(f"💰 الرصيد الفعلي: {usdt_balance} USDT")
                return usdt_balance
            else:
                return self.trade_config['initial_balance']
                
        except Exception as e:
            logger.error(f"❌ خطأ في جلب الرصيد: {e}")
            return self.trade_config['initial_balance']

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, max=10))
    def fetch_binance_data(self, symbol, days=30):
        """جلب البيانات من Binance"""
        try:
            interval = self.trade_config['timeframe']
            limit = 500
            all_data = []
            end_time = int(datetime.now().timestamp() * 1000)

            logger.info(f"📊 جلب البيانات لـ {symbol}")

            while len(all_data) < limit:
                params = {
                    'symbol': symbol,
                    'interval': interval,
                    'limit': min(limit, 1000),
                    'endTime': end_time
                }
                
                data = self.make_request('klines', params=params)
                if not data or len(data) == 0:
                    break
                    
                all_data = data + all_data
                if len(data) > 0:
                    end_time = data[0][0] - 1
                else:
                    break

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
            
            self.data[symbol] = df
            self.calculate_technical_indicators(symbol)
            
            logger.info(f"✅ تم جلب {len(df)} شمعة لـ {symbol}")

        except Exception as e:
            logger.error(f"❌ خطأ في جلب البيانات لـ {symbol}: {e}")
            self.generate_sample_data(symbol, days)

    def calculate_technical_indicators(self, symbol):
        """حساب المؤشرات الفنية للفيبوناتشي"""
        df = self.data[symbol]
        
        # المتوسط المتحرك للاتجاه
        df['ema_trend'] = df['close'].ewm(
            span=self.fib_config['trend_ema_period'], 
            adjust=False
        ).mean()
        
        # RSI
        delta = df['close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        df['rsi'] = 100 - (100 / (1 + rs))
        
        # الحجم
        df['volume_ma'] = df['volume'].rolling(20).mean()
        df['volume_ratio'] = df['volume'] / df['volume_ma']
        
        # تحديد القمم والقيعان
        df = self.find_swing_points(df)
        
        # حساب مستويات الفيبوناتشي
        df = self.calculate_fibonacci_levels(df)
        
        self.data[symbol] = df

    def find_swing_points(self, df):
        """تحديد القمم والقيعان للسوينجات"""
        period = self.fib_config['swing_period']
        
        # القمم
        df['swing_high'] = df['high'].rolling(period, center=True).max()
        df['is_peak'] = df['high'] == df['swing_high']
        
        # القيعان
        df['swing_low'] = df['low'].rolling(period, center=True).min()
        df['is_trough'] = df['low'] == df['swing_low']
        
        return df

    def calculate_fibonacci_levels(self, df):
        """حساب مستويات الفيبوناتشي"""
        # البحث عن آخر سوينج واضح
        recent_peaks = df[df['is_peak']].tail(2)
        recent_troughs = df[df['is_trough']].tail(2)
        
        if len(recent_peaks) >= 2 and len(recent_troughs) >= 2:
            # ابحث عن أقوى سوينج
            last_peak = recent_peaks.iloc[-1]
            last_trough = recent_troughs.iloc[-1]
            prev_peak = recent_peaks.iloc[-2]
            prev_trough = recent_troughs.iloc[-2]
            
            # تحديد إذا كان الاتجاه صاعد أو هابط
            if last_peak['high'] > prev_peak['high'] and last_trough['low'] > prev_trough['low']:
                # اتجاه صاعد
                swing_low = prev_trough['low']
                swing_high = last_peak['high']
                trend_direction = 'UP'
            elif last_peak['high'] < prev_peak['high'] and last_trough['low'] < prev_trough['low']:
                # اتجاه هابط
                swing_high = prev_peak['high']
                swing_low = last_trough['low']
                trend_direction = 'DOWN'
            else:
                # اتجاه جانبي
                swing_low = min(prev_trough['low'], last_trough['low'])
                swing_high = max(prev_peak['high'], last_peak['high'])
                trend_direction = 'SIDEWAYS'
            
            # حساب مستويات الفيبوناتشي
            swing_range = swing_high - swing_low
            
            for level in self.fib_config['fibonacci_levels']:
                df[f'fib_{int(level*1000)}'] = swing_high - (swing_range * level)
            
            df['fib_trend'] = trend_direction
            df['fib_swing_low'] = swing_low
            df['fib_swing_high'] = swing_high
            
        return df

    def detect_candlestick_pattern(self, df, index):
        """الكشف عن نماذج الشموع الانعكاسية"""
        if index < 2:
            return None
            
        current = df.iloc[index]
        prev = df.iloc[index-1]
        prev2 = df.iloc[index-2]
        
        # مطرقة (Hammer)
        body = abs(current['close'] - current['open'])
        lower_shadow = current['open'] - current['low'] if current['close'] > current['open'] else current['close'] - current['low']
        upper_shadow = current['high'] - current['close'] if current['close'] > current['open'] else current['high'] - current['open']
        
        if lower_shadow > 2 * body and upper_shadow < body * 0.5:
            return 'hammer'
        
        # شمعة الابتلاع (Engulfing)
        if (current['close'] > current['open'] and prev['close'] < prev['open'] and
            current['open'] < prev['close'] and current['close'] > prev['open']):
            return 'bullish_engulfing'
        
        if (current['close'] < current['open'] and prev['close'] > prev['open'] and
            current['open'] > prev['close'] and current['close'] < prev['open']):
            return 'bearish_engulfing'
        
        # دوجي (Doji)
        if body / (current['high'] - current['low']) < 0.1:
            return 'doji'
        
        return None

    def is_near_fibonacci_level(self, price, df, index):
        """التحقق إذا كان السعر قريب من مستوى فيبوناتشي مهم"""
        if index < 10:
            return False, None
            
        current_data = df.iloc[index]
        
        for level in self.fib_config['key_levels']:
            fib_level = current_data.get(f'fib_{int(level*1000)}', None)
            if fib_level and abs(price - fib_level) / fib_level < 0.005:  # 0.5%
                return True, level
        
        return False, None

    def generate_fibonacci_signal(self, symbol):
        """توليد إشارات التداول بناءً على استراتيجية الفيبوناتشي"""
        try:
            if symbol not in self.data or len(self.data[symbol]) < 30:
                return 'HOLD', "بيانات غير كافية"
            
            df = self.data[symbol]
            current = df.iloc[-1]
            prev = df.iloc[-2]
            
            # 1. تحديد الاتجاه العام
            trend_strength = (current['close'] - current['ema_trend']) / current['ema_trend']
            if abs(trend_strength) < self.fib_config['min_trend_strength']:
                return 'HOLD', "الاتجاه ضعيف"
            
            is_uptrend = current['close'] > current['ema_trend']
            
            # 2. التحقق من قرب السعر من مستوى فيبوناتشي رئيسي
            is_near_fib, fib_level = self.is_near_fibonacci_level(current['close'], df, -1)
            if not is_near_fib:
                return 'HOLD', "ليس عند مستوى فيبوناتشي"
            
            # 3. الكشف عن نماذج الشموع الانعكاسية
            candle_pattern = self.detect_candlestick_pattern(df, -1)
            if not candle_pattern:
                return 'HOLD', "لا توجد شمعة انعكاسية"
            
            # 4. تحقق من مؤشر RSI
            rsi_condition = False
            if is_uptrend and current['rsi'] < self.fib_config['rsi_oversold']:
                rsi_condition = True
            elif not is_uptrend and current['rsi'] > self.fib_config['rsi_overbought']:
                rsi_condition = True
                
            if not rsi_condition:
                return 'HOLD', "RSI غير مناسب"
            
            # 5. تحقق من الحجم
            if current['volume_ratio'] < self.fib_config['volume_threshold']:
                return 'HOLD', "الحجم ضعيف"
            
            # 6. توليد الإشارة النهائية
            if (is_uptrend and candle_pattern in ['hammer', 'bullish_engulfing', 'doji'] and 
                fib_level in [0.382, 0.500, 0.618]):
                signal_strength = self.calculate_signal_strength(current, fib_level, candle_pattern)
                details = f"شراء عند فيبوناتشي {fib_level:.1%} | {candle_pattern} | قوة: {signal_strength:.1f}"
                return 'BUY', details
                
            elif (not is_uptrend and candle_pattern in ['bearish_engulfing', 'doji'] and 
                  fib_level in [0.382, 0.500, 0.618]):
                signal_strength = self.calculate_signal_strength(current, fib_level, candle_pattern)
                details = f"بيع عند فيبوناتشي {fib_level:.1%} | {candle_pattern} | قوة: {signal_strength:.1f}"
                return 'SELL', details
            
            return 'HOLD', "شروط غير متوفرة"
            
        except Exception as e:
            logger.error(f"❌ خطأ في توليد إشارة الفيبوناتشي: {e}")
            return 'HOLD', f"خطأ: {str(e)}"

    def calculate_signal_strength(self, current, fib_level, candle_pattern):
        """حساب قوة الإشارة"""
        strength = 0
        
        # قوة مستوى الفيبوناتشي
        if fib_level == 0.618:
            strength += 3
        elif fib_level == 0.500:
            strength += 2
        elif fib_level == 0.382:
            strength += 1
        
        # قوة نموذج الشمعة
        if candle_pattern in ['bullish_engulfing', 'bearish_engulfing']:
            strength += 2
        elif candle_pattern in ['hammer', 'doji']:
            strength += 1
        
        # قوة RSI
        if (current['rsi'] < 25 or current['rsi'] > 75):
            strength += 1
        
        # قوة الحجم
        if current['volume_ratio'] > 2.0:
            strength += 1
        
        return min(strength, 5)

    def open_fibonacci_position(self, symbol, direction, details):
        """فتح صفقة بناءً على إشارة الفيبوناتشي"""
        try:
            if len([p for p in self.positions if p['status'] == 'OPEN']) >= self.trade_config['max_positions']:
                logger.warning(f"⚠️ وصل الحد الأقصى للصفقات المفتوحة")
                return
            
            df = self.data[symbol]
            current = df.iloc[-1]
            
            position_size = self.current_balance * self.trade_config['position_size']
            entry_price = current['close']
            
            if direction == 'BUY':
                stop_loss = entry_price * (1 - self.trade_config['stop_loss_percent'])
                take_profit = entry_price * (1 + (self.trade_config['stop_loss_percent'] * 
                                                self.trade_config['risk_reward_ratio']))
            else:
                stop_loss = entry_price * (1 + self.trade_config['stop_loss_percent'])
                take_profit = entry_price * (1 - (self.trade_config['stop_loss_percent'] * 
                                                self.trade_config['risk_reward_ratio']))
            
            position = {
                'id': len(self.trades) + 1,
                'symbol': symbol,
                'direction': direction,
                'entry_price': entry_price,
                'entry_time': datetime.now(),
                'size': position_size,
                'stop_loss': stop_loss,
                'take_profit': take_profit,
                'status': 'OPEN',
                'details': details
            }
            
            self.positions.append(position)
            
            logger.info(f"🎯 فتح {direction} على {symbol}")
            logger.info(f"💰 الحجم: ${position_size:.2f} | السعر: ${entry_price:.2f}")
            logger.info(f"🛡️ وقف الخسارة: ${stop_loss:.2f} | جني الأرباح: ${take_profit:.2f}")
            
            self.send_telegram_message(
                f"🎯 **إشارة فيبوناتشي جديدة**\n"
                f"• الزوج: {symbol}\n"
                f"• الاتجاه: {direction}\n"
                f"• السعر: ${entry_price:.2f}\n"
                f"• التفاصيل: {details}\n"
                f"• الرصيد: ${self.current_balance:.2f}"
            )
                
        except Exception as e:
            logger.error(f"❌ خطأ فتح الصفقة: {e}")

    def check_position_management(self):
        """إدارة الصفقات المفتوحة"""
        for position in [p for p in self.positions if p['status'] == 'OPEN']:
            symbol = position['symbol']
            if symbol not in self.data:
                continue
                
            current_price = self.data[symbol].iloc[-1]['close']
            
            if position['direction'] == 'BUY':
                if current_price <= position['stop_loss']:
                    self.close_position(position, 'STOP_LOSS', current_price)
                elif current_price >= position['take_profit']:
                    self.close_position(position, 'TAKE_PROFIT', current_price)
            else:
                if current_price >= position['stop_loss']:
                    self.close_position(position, 'STOP_LOSS', current_price)
                elif current_price <= position['take_profit']:
                    self.close_position(position, 'TAKE_PROFIT', current_price)

    def close_position(self, position, reason, exit_price):
        """إغلاق الصفقة"""
        try:
            pnl_percent = (exit_price - position['entry_price']) / position['entry_price']
            if position['direction'] == 'SELL':
                pnl_percent = -pnl_percent
                
            pnl_dollar = position['size'] * pnl_percent
            
            position.update({
                'status': 'CLOSED',
                'exit_price': exit_price,
                'exit_time': datetime.now(),
                'pnl_percent': pnl_percent * 100,
                'pnl_dollar': pnl_dollar,
                'reason': reason
            })
            
            self.current_balance += pnl_dollar
            self.trades.append(position.copy())
            self.positions.remove(position)
            
            emoji = "💰" if pnl_dollar > 0 else "💸"
            logger.info(f"{emoji} إغلاق {position['direction']} على {position['symbol']} | {reason} | ${pnl_dollar:+.2f}")
            
        except Exception as e:
            logger.error(f"❌ خطأ في إغلاق الصفقة: {e}")

    def send_telegram_message(self, message):
        """إرسال رسالة عبر التلغرام"""
        if not TELEGRAM_CONFIG['bot_token'] or not TELEGRAM_CONFIG['chat_id']:
            return
        
        try:
            bot = Bot(token=TELEGRAM_CONFIG['bot_token'])
            bot.send_message(chat_id=TELEGRAM_CONFIG['chat_id'], text=message, parse_mode='Markdown')
        except Exception as e:
            logger.error(f"❌ خطأ في إرسال رسالة التلغرام: {e}")

    def run_fibonacci_strategy(self):
        """تشغيل استراتيجية الفيبوناتشي"""
        try:
            logger.info("🔍 فحص إشارات الفيبوناتشي...")
            
            for symbol in self.trade_config['symbols']:
                self.fetch_binance_data(symbol, days=30)
                
                signal, details = self.generate_fibonacci_signal(symbol)
                
                if signal in ['BUY', 'SELL']:
                    self.open_fibonacci_position(symbol, signal, details)
                
            self.check_position_management()
            
            logger.info(f"💰 الرصيد الحالي: ${self.current_balance:.2f}")
            logger.info(f"📊 الصفقات المفتوحة: {len([p for p in self.positions if p['status'] == 'OPEN'])}")
            
        except Exception as e:
            logger.error(f"❌ خطأ في تشغيل الاستراتيجية: {e}")

# ====================== تشغيل البوت ======================
def run_fibonacci_bot():
    """تشغيل بوت الفيبوناتشي"""
    logger.info("🚀 بدء تشغيل بوت الفيبوناتشي")
    
    bot = FibonacciTradingBot(FIBONACCI_CONFIG, TRADE_CONFIG, BINANCE_CONFIG)
    
    # إضافة health check server
    try:
        import http.server
        import socketserver
        from threading import Thread
        
        PORT = int(os.getenv('PORT', 10000))
        
        class HealthCheckHandler(http.server.SimpleHTTPRequestHandler):
            def do_GET(self):
                if self.path == '/health':
                    self.send_response(200)
                    self.send_header('Content-type', 'text/plain')
                    self.end_headers()
                    self.wfile.write(b'Fibonacci Bot is running')
                else:
                    self.send_response(404)
                    self.end_headers()
        
        def run_health_server():
            with socketserver.TCPServer(("", PORT), HealthCheckHandler) as httpd:
                logger.info(f"🌐 Health server running on port {PORT}")
                httpd.serve_forever()
        
        server_thread = Thread(target=run_health_server, daemon=True)
        server_thread.start()
        
    except Exception as e:
        logger.warning(f"⚠️ Could not start health server: {e}")

    # المهام المجدولة
    schedule.every(10).minutes.do(bot.run_fibonacci_strategy)
    schedule.every(1).hour.do(lambda: bot.send_telegram_message(
        f"📊 **تقرير بوت الفيبوناتشي**\n"
        f"• الرصيد: ${bot.current_balance:.2f}\n"
        f"• الصفقات المفتوحة: {len([p for p in bot.positions if p['status'] == 'OPEN'])}\n"
        f"• إجمالي الصفقات: {len(bot.trades)}"
    ))
    
    # التشغيل الفوري
    bot.run_fibonacci_strategy()
    
    logger.info("✅ بدء المراقبة المستمرة...")
    
    while True:
        try:
            schedule.run_pending()
            time.sleep(1)
        except KeyboardInterrupt:
            logger.info("⏹️ إيقاف البوت")
            break
        except Exception as e:
            logger.error(f"❌ خطأ: {e}")
            time.sleep(60)

if __name__ == "__main__":
    run_fibonacci_bot()
