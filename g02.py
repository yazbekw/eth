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
from dotenv import load_dotenv  # اختياري لكن موصى به

warnings.filterwarnings('ignore')
load_dotenv()  # تحميل المتغيرات من .env

# ====================== إعدادات التفعيل / الإلغاء ======================
ENABLE_TRAILING_STOP = True
ENABLE_DYNAMIC_POSITION_SIZING = True
ENABLE_MARKET_REGIME_FILTER = True
ENABLE_ATR_SL_TP = True
ENABLE_SUPPORT_RESISTANCE_FILTER = True
ENABLE_TIME_FILTER = True
ENABLE_WALK_FORWARD = True
ENABLE_LOGGING = True

# ====================== دالة مساعدة خارج الكلاس (آمنة) ======================
def interval_to_hours(interval):
    mapping = {
        '1m': 1/60, '3m': 3/60, '5m': 5/60, '15m': 15/60,
        '30m': 30/60, '1h': 1, '2h': 2, '4h': 4, '6h': 6,
        '8h': 8, '12h': 12, '1d': 24, '3d': 72, '1w': 168, '1M': 720
    }
    return mapping.get(interval, 4)  # افتراضي 4h

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
    'min_signal_strength': 5,
    'max_signal_strength': 5
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

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, max=10))
    def fetch_binance_data(self, days=60):
        """جلب كل البيانات المطلوبة مع حلقة ذكية"""
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
        high = self.data['high']
        low = self.data['low']
        close = self.data['close']
        tr0 = abs(high - low)
        tr1 = abs(high - close.shift())
        tr2 = abs(low - close.shift())
        tr = pd.concat([tr0, tr1, tr2], axis=1).max(axis=1)
        return tr.rolling(period).mean()

    def calculate_indicators(self):
        df = self.data
        p = self.indicator_config

        # RSI
        delta = df['close'].diff()
        gain = delta.where(delta > 0, 0)
        loss = -delta.where(delta < 0, 0)
        avg_gain = gain.rolling(p['rsi_period'], min_periods=1).mean()
        avg_loss = loss.rolling(p['rsi_period'], min_periods=1).mean()
        rs = avg_gain / avg_loss.replace(0, np.nan)
        df['rsi'] = 100 - (100 / (1 + rs))

        # EMAs
        df['ema_fast'] = df['close'].ewm(span=p['ema_fast'], adjust=False).mean()
        df['ema_slow'] = df['close'].ewm(span=p['ema_slow'], adjust=False).mean()
        df['ema_trend'] = df['close'].ewm(span=p['ema_trend'], adjust=False).mean()
        df['ema_regime'] = df['close'].ewm(span=p['ema_regime'], adjust=False).mean()

        # MACD
        ema_fast = df['close'].ewm(span=p['macd_fast'], adjust=False).mean()
        ema_slow = df['close'].ewm(span=p['macd_slow'], adjust=False).mean()
        df['macd'] = ema_fast - ema_slow
        df['macd_signal'] = df['macd'].ewm(span=p['macd_signal'], adjust=False).mean()
        df['macd_histogram'] = df['macd'] - df['macd_signal']

        # Volume MA
        df['volume_ma'] = df['volume'].rolling(20, min_periods=1).mean()

        # ATR
        if ENABLE_ATR_SL_TP:
            df['atr'] = self.calculate_atr(p['atr_period'])

        # Support & Resistance
        if ENABLE_SUPPORT_RESISTANCE_FILTER:
            window = self.trade_config['support_resistance_window']
            df['resistance'] = df['high'].rolling(window).max()
            df['support'] = df['low'].rolling(window).min()

        self.data = df
        if ENABLE_LOGGING:
            logger.info("تم حساب جميع المؤشرات")

    def get_market_regime(self, row):
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
        base = max(buy_conditions, sell_conditions)
        if base == 0:
            return 1

        rsi = row['rsi']
        trend_ok = (buy_conditions > sell_conditions and row['close'] > row['ema_trend']) or \
                   (sell_conditions > buy_conditions and row['close'] < row['ema_trend'])
        macd_strength = abs(row['macd_histogram']) / row['close'] * 1000
        volume_ratio = row['volume'] / row['volume_ma'] if row['volume_ma'] > 0 else 1

        points = base
        if (rsi < 25 and buy_conditions > sell_conditions) or (rsi > 75 and sell_conditions > buy_conditions):
            points += 2
        if trend_ok:
            points += 1
        if macd_strength > 0.5:
            points += 1
        if volume_ratio > 1.5:
            points += 1

        return min(max(points, 1), 10)

    def generate_signal(self, row):
        required = ['rsi', 'ema_fast', 'ema_slow', 'macd', 'ema_trend', 'volume_ma']
        if any(pd.isna(row[col]) for col in required):
            return 'HOLD', 1, "بيانات ناقصة"

        buy_conditions = sell_conditions = 0
        details = []

        # RSI
        if row['rsi'] < self.indicator_config['rsi_oversold']:
            buy_conditions += 1
            details.append("RSI منخفض")
        elif row['rsi'] > self.indicator_config['rsi_overbought']:
            sell_conditions += 1
            details.append("RSI مرتفع")

        # EMA Crossover
        if row['ema_fast'] > row['ema_slow']:
            buy_conditions += 1
            details.append("EMA صاعد")
        else:
            sell_conditions += 1
            details.append("EMA هابط")

        # MACD
        if row['macd'] > row['macd_signal'] and abs(row['macd_histogram']) > row['close'] * 0.001:
            buy_conditions += 1
            details.append("MACD صاعد")
        elif row['macd'] < row['macd_signal'] and abs(row['macd_histogram']) > row['close'] * 0.001:
            sell_conditions += 1
            details.append("MACD هابط")

        # Trend Filter
        if row['close'] > row['ema_trend']:
            buy_conditions += 1
            details.append("فوق EMA50")
        else:
            sell_conditions += 1
            details.append("تحت EMA50")

        # Volume Filter
        volume_ratio = row['volume'] / row['volume_ma'] if row['volume_ma'] > 0 else 1
        volume_ok = volume_ratio > self.trade_config['min_volume_ratio']

        # Time Filter
        hour = row['timestamp'].hour
        time_ok = not ENABLE_TIME_FILTER or hour in self.trade_config['peak_hours']

        # Market Regime
        regime = self.get_market_regime(row)
        regime_ok = regime != "BEAR" if buy_conditions > sell_conditions else regime != "BULL"

        # Support/Resistance
        near_level = False
        if ENABLE_SUPPORT_RESISTANCE_FILTER and 'resistance' in row:
            dist_r = abs(row['close'] - row['resistance']) / row['close']
            dist_s = abs(row['close'] - row['support']) / row['close']
            near_level = min(dist_r, dist_s) < 0.003

        # Final Decision
        strength = self.calculate_signal_strength(buy_conditions, sell_conditions, row)
        min_strength = self.signal_config['min_signal_strength']
        max_strength = self.signal_config['max_signal_strength']

        if (buy_conditions >= self.signal_config['min_conditions'] and
            strength >= min_strength and strength <= max_strength and
            volume_ok and time_ok and regime_ok and not near_level):
            return 'BUY', strength, " | ".join(details)

        elif (sell_conditions >= self.signal_config['min_conditions'] and
              strength >= min_strength and strength <= max_strength and
              volume_ok and time_ok and regime_ok and not near_level):
            return 'SELL', strength, " | ".join(details)

        return 'HOLD', 1, "لا تلبي الشروط"

    def execute_backtest(self):
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
        min_period = 200
        for i, row in data.iterrows():
            if i < min_period:
                continue
            signal, strength, details = self.generate_signal(row)
            self.check_exit_conditions(row)

            open_pos = len([p for p in self.positions if p['status'] == 'OPEN'])
            if signal in ['BUY', 'SELL'] and open_pos < self.trade_config['max_positions']:
                self.open_position(signal, strength, row, details)

    def open_position(self, direction, strength, row, details):
        try:
            base_size = self.trade_config['base_position_size']
            size_factor = strength / 10 if ENABLE_DYNAMIC_POSITION_SIZING else 1.0
            position_value = self.current_balance * base_size * size_factor * self.trade_config['leverage']

            entry = row['close']
            atr = row['atr'] if ENABLE_ATR_SL_TP and 'atr' in row and not pd.isna(row['atr']) else entry * 0.01

            if ENABLE_ATR_SL_TP:
                sl = entry - (self.trade_config['atr_multiplier_sl'] * atr) if direction == 'BUY' else entry + (self.trade_config['atr_multiplier_sl'] * atr)
                tp = entry + (self.trade_config['atr_multiplier_tp'] * atr) if direction == 'BUY' else entry - (self.trade_config['atr_multiplier_tp'] * atr)
            else:
                sl = entry * (1 - self.trade_config['base_stop_loss']) if direction == 'BUY' else entry * (1 + self.trade_config['base_stop_loss'])
                tp = entry * (1 + self.trade_config['base_take_profit']) if direction == 'BUY' else entry * (1 - self.trade_config['base_take_profit'])

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
                'signal_strength': strength,
                'signal_details': details,
                'trailing_stop': float(sl)
            }
            self.positions.append(position)
            if ENABLE_LOGGING:
                logger.info(f"فتح {direction} #{position['id']} | قوة: {strength} | حجم: ${position_value:.2f}")
        except Exception as e:
            if ENABLE_LOGGING:
                logger.error(f"خطأ فتح الصفقة: {e}")

    def check_exit_conditions(self, row):
        current_price = row['close']
        for pos in [p for p in self.positions if p['status'] == 'OPEN']:
            pnl = 0.0
            reason = ''
            duration = (row['timestamp'] - pos['entry_time']).total_seconds() / 3600

            # Trailing Stop
            if self.trade_config['use_trailing_stop'] and pos['direction'] == 'BUY':
                profit = (current_price - pos['entry_price']) / pos['entry_price']
                if profit > self.trade_config['trailing_activation']:
                    new_sl = current_price * (1 - self.trade_config['trailing_stop_percent'])
                    pos['trailing_stop'] = max(pos['trailing_stop'], new_sl)
                if current_price <= pos['trailing_stop']:
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
                    'reason': reason
                })
                self.current_balance += pos['size'] * pos['pnl']
                self.trades.append(pos.copy())
                self.positions.remove(pos)
                if ENABLE_LOGGING:
                    logger.info(f"إغلاق {pos['direction']} #{pos['id']} | {reason} | {pos['pnl']*100:+.2f}%")

    def generate_report(self):
        if not self.trades:
            return "لا توجد صفقات"

        df = pd.DataFrame(self.trades)
        balance_history = [self.initial_balance]
        balance = self.initial_balance
        for t in self.trades:
            balance += t['size'] * t['pnl']
            balance_history.append(balance)

        total_pnl = self.current_balance - self.initial_balance
        total_pnl_pct = total_pnl / self.initial_balance * 100
        win_rate = len(df[df['pnl'] > 0]) / len(df) * 100 if len(df) > 0 else 0
        returns = df['pnl']
        sharpe = (returns.mean() / returns.std() * np.sqrt(252)) if returns.std() > 0 else 0
        drawdowns = pd.Series(balance_history) / pd.Series(balance_history).cummax() - 1
        max_dd = drawdowns.min() * 100

        report = f"""
تقرير الأداء النهائي
الفترة: {self.data['timestamp'].iloc[0].date()} إلى {self.data['timestamp'].iloc[-1].date()}
الرصيد: ${self.initial_balance:,.2f} → ${self.current_balance:,.2f} ({total_pnl_pct:+.2f}%)
إجمالي الصفقات: {len(self.trades)} | الفوز: {win_rate:.1f}%
متوسط الربح: {df[df['pnl']>0]['pnl'].mean()*100:+.2f}% | الخسارة: {df[df['pnl']<=0]['pnl'].mean()*100:+.2f}%
Sharpe Ratio: {sharpe:.2f} | أقصى انخفاض: {max_dd:.2f}%
        """
        return report.strip()

    def send_telegram_report(self):
        if not TELEGRAM_CONFIG['bot_token'] or not TELEGRAM_CONFIG['chat_id']:
            if ENABLE_LOGGING:
                logger.warning("مفاتيح التلغرام غير متوفرة")
            return
        try:
            bot = telebot.TeleBot(TELEGRAM_CONFIG['bot_token'])
            report = self.generate_report()
            if len(report) > 4000:
                for i in range(0, len(report), 4000):
                    bot.send_message(TELEGRAM_CONFIG['chat_id'], report[i:i+4000])
            else:
                bot.send_message(TELEGRAM_CONFIG['chat_id'], report)
            if ENABLE_LOGGING:
                logger.info("تم إرسال التقرير إلى التلغرام")
        except Exception as e:
            if ENABLE_LOGGING:
                logger.error(f"خطأ الإرسال: {e}")

# ====================== التشغيل ======================
def main():
    if ENABLE_LOGGING:
        logger.info("بدء تشغيل البوت الاحترافي")
    bot = AdvancedCryptoBot(TRADE_CONFIG, INDICATOR_CONFIG, SIGNAL_CONFIG)
    bot.fetch_binance_data(days=60)
    bot.execute_backtest()
    print(bot.generate_report())
    bot.send_telegram_report()

if __name__ == "__main__":
    main()
