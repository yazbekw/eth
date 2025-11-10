import os
import pandas as pd
import numpy as np
import requests
from datetime import datetime, timedelta
import telebot
import warnings
warnings.filterwarnings('ignore')

# ==================== الإعدادات الأساسية ====================

TRADE_CONFIG = {
    'symbol': 'BNBUSDT',
    'timeframe': '1h',
    'initial_balance': 1000,
    'leverage': 3,
    'stop_loss': 0.04,
    'take_profit': 0.055,
    'position_size': 0.07,
    'max_positions': 3,
    'paper_trading': True
}

INDICATOR_CONFIG = {
    'rsi_period': 21,
    'rsi_overbought': 70,
    'rsi_oversold': 30,
    'ema_fast': 9,
    'ema_slow': 21,
    'ema_trend': 50,
    'macd_fast': 12,
    'macd_slow': 26,
    'macd_signal': 9
}

SIGNAL_CONFIG = {
    'min_conditions': 2,
    'use_trend_filter': True,
    'use_volume_filter': True,
    'min_volume_ratio': 0.8
}

# إعدادات التلغرام - تأكد من تعبئتها في Render
TELEGRAM_CONFIG = {
    'bot_token': os.getenv('TELEGRAM_BOT_TOKEN', ''),
    'chat_id': os.getenv('TELEGRAM_CHAT_ID', '')
}

class SimpleCryptoBot:
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
        
    # ... جميع الدوال السابقة تبقى كما هي ...
    def fetch_binance_data(self, days=180):
        """جلب البيانات من Binance لمدة 6 أشهر"""
        try:
            symbol = self.trade_config['symbol']
            interval = self.trade_config['timeframe']
            
            end_date = datetime.now()
            start_date = end_date - timedelta(days=days)
            
            print(f"📅 جلب بيانات {days} يوم من {start_date.date()} إلى {end_date.date()}")
            
            url = "https://api.binance.com/api/v3/klines"
            params = {
                'symbol': symbol,
                'interval': interval,
                'limit': days * 24
            }
            
            response = requests.get(url, params=params, timeout=15)
            data = response.json()
            
            df = pd.DataFrame(data, columns=[
                'timestamp', 'open', 'high', 'low', 'close', 'volume',
                'close_time', 'quote_asset_volume', 'number_of_trades',
                'taker_buy_base_asset_volume', 'taker_buy_quote_asset_volume', 'ignore'
            ])
            
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
            for col in ['open', 'high', 'low', 'close', 'volume']:
                df[col] = pd.to_numeric(df[col], errors='coerce')
            
            df = df[['timestamp', 'open', 'high', 'low', 'close', 'volume']]
            df = df.sort_values('timestamp').reset_index(drop=True)
            
            self.data = df
            self.calculate_indicators()
            print(f"✅ تم جلب {len(self.data)} شمعة للعملة {symbol} ({days} يوم)")
            
        except Exception as e:
            print(f"❌ خطأ في جلب البيانات: {e}")
            self.generate_sample_data(days)
    
    def generate_sample_data(self, days):
        """بيانات عينة إذا فشل الاتصال"""
        print(f"📊 استخدام بيانات عينة لـ {days} يوم...")
        
        end_date = datetime.now()
        start_date = end_date - timedelta(days=days)
        dates = pd.date_range(start=start_date, end=end_date, freq='1H')
        
        np.random.seed(42)
        price = 300.0
        prices = []
        
        for i in range(len(dates)):
            volatility = 0.005 if i % 24 == 0 else 0.002
            change = np.random.normal(0, volatility)
            price = price * (1 + change)
            prices.append(price)
        
        self.data = pd.DataFrame({
            'timestamp': dates,
            'open': prices,
            'high': [p * (1 + abs(np.random.normal(0, 0.002))) for p in prices],
            'low': [p * (1 - abs(np.random.normal(0, 0.002))) for p in prices],
            'close': prices,
            'volume': [abs(np.random.normal(1000, 200)) for _ in prices]
        })
        
        self.calculate_indicators()
    
    def calculate_indicators(self):
        """حساب المؤشرات الفنية"""
        # RSI
        delta = self.data['close'].diff()
        gain = delta.where(delta > 0, 0.0)
        loss = -delta.where(delta < 0, 0.0)
        
        avg_gain = gain.rolling(window=self.indicator_config['rsi_period']).mean()
        avg_loss = loss.rolling(window=self.indicator_config['rsi_period']).mean()
        rs = avg_gain / avg_loss
        self.data['rsi'] = 100 - (100 / (1 + rs))
        
        # المتوسطات المتحركة
        self.data['ema_fast'] = self.data['close'].ewm(span=self.indicator_config['ema_fast'], adjust=False).mean()
        self.data['ema_slow'] = self.data['close'].ewm(span=self.indicator_config['ema_slow'], adjust=False).mean()
        self.data['ema_trend'] = self.data['close'].ewm(span=self.indicator_config['ema_trend'], adjust=False).mean()
        
        # MACD
        ema_fast = self.data['close'].ewm(span=self.indicator_config['macd_fast'], adjust=False).mean()
        ema_slow = self.data['close'].ewm(span=self.indicator_config['macd_slow'], adjust=False).mean()
        self.data['macd'] = ema_fast - ema_slow
        self.data['macd_signal'] = self.data['macd'].ewm(span=self.indicator_config['macd_signal'], adjust=False).mean()
        
        # حجم التداول المتوسط
        self.data['volume_ma'] = self.data['volume'].rolling(window=20).mean()
        
        print("✅ تم حساب المؤشرات الفنية")
    
    def generate_signal(self, row):
        """توليد إشارات التداول"""
        if any(pd.isna(row[key]) for key in ['rsi', 'ema_slow', 'macd', 'ema_trend', 'volume_ma']):
            return 'HOLD', 0
        
        buy_conditions = 0
        sell_conditions = 0
        
        if row['rsi'] < self.indicator_config['rsi_oversold']:
            buy_conditions += 1
        elif row['rsi'] > self.indicator_config['rsi_overbought']:
            sell_conditions += 1
            
        if row['ema_fast'] > row['ema_slow']:
            buy_conditions += 1
        else:
            sell_conditions += 1
            
        macd_strength = abs(row['macd'] - row['macd_signal']) > (row['close'] * 0.001)
        if row['macd'] > row['macd_signal'] and macd_strength:
            buy_conditions += 1
        elif row['macd'] < row['macd_signal'] and macd_strength:
            sell_conditions += 1
        
        if self.signal_config['use_trend_filter']:
            if row['close'] > row['ema_trend']:
                buy_conditions += 0.5
            else:
                sell_conditions += 0.5
        
        if self.signal_config['use_volume_filter']:
            volume_confirm = row['volume'] > row['volume_ma'] * self.signal_config['min_volume_ratio']
            if volume_confirm:
                buy_conditions += 0.5
                sell_conditions += 0.5
        
        signal = 'HOLD'
        strength = 0
        
        min_conditions = self.signal_config['min_conditions']
        
        if buy_conditions >= min_conditions:
            signal = 'BUY'
            strength = buy_conditions
        elif sell_conditions >= min_conditions:
            signal = 'SELL' 
            strength = sell_conditions
        
        return signal, strength
    
    def execute_backtest(self):
        """تنفيذ الباك تستينغ"""
        print("🔄 بدء الباك تستينغ...")
        if self.paper_trading:
            print("📝 وضع التداول الورقي مفعل")
        
        min_period = max(
            self.indicator_config['ema_slow'],
            self.indicator_config['rsi_period'], 
            self.indicator_config['ema_trend'],
            20
        )
        
        for i, row in self.data.iterrows():
            if i < min_period:
                continue
            
            signal, strength = self.generate_signal(row)
            self.check_exit_conditions(row)
            
            open_positions = len([p for p in self.positions if p['status'] == 'OPEN'])
            if (signal in ['BUY', 'SELL'] and 
                open_positions < self.trade_config['max_positions']):
                self.open_position(signal, row)
        
        print(f"✅ تم الانتهاء - {len(self.trades)} صفقة")
    
    def open_position(self, direction, row):
        """فتح صفقة جديدة"""
        position_size = self.current_balance * self.trade_config['position_size'] * self.trade_config['leverage']
        
        if direction == 'BUY':
            stop_loss = row['close'] * (1 - self.trade_config['stop_loss'])
            take_profit = row['close'] * (1 + self.trade_config['take_profit'])
        else:
            stop_loss = row['close'] * (1 + self.trade_config['stop_loss'])
            take_profit = row['close'] * (1 - self.trade_config['take_profit'])
        
        position = {
            'id': len(self.positions) + 1,
            'direction': direction,
            'entry_price': float(row['close']),
            'entry_time': row['timestamp'],
            'size': float(position_size),
            'stop_loss': float(stop_loss),
            'take_profit': float(take_profit),
            'status': 'OPEN',
            'type': 'PAPER' if self.paper_trading else 'REAL'
        }
        
        self.positions.append(position)
        trade_type = "ورقي" if self.paper_trading else "حقيقي"
        print(f"📈 فتح صفقة {trade_type} {direction} #{position['id']} بسعر {row['close']:.2f}")
    
    def check_exit_conditions(self, row):
        """فحص شروط الخروج"""
        current_price = float(row['close'])
        
        for position in self.positions:
            if position['status'] == 'OPEN':
                pnl = 0.0
                reason = ''
                
                if position['direction'] == 'BUY':
                    if current_price <= position['stop_loss']:
                        pnl = (current_price - position['entry_price']) / position['entry_price']
                        reason = 'STOP_LOSS'
                    elif current_price >= position['take_profit']:
                        pnl = (current_price - position['entry_price']) / position['entry_price']
                        reason = 'TAKE_PROFIT'
                else:
                    if current_price >= position['stop_loss']:
                        pnl = (position['entry_price'] - current_price) / position['entry_price']
                        reason = 'STOP_LOSS'
                    elif current_price <= position['take_profit']:
                        pnl = (position['entry_price'] - current_price) / position['entry_price']
                        reason = 'TAKE_PROFIT'
                
                if reason:
                    loss_reason = self.analyze_loss_reason(position, row) if pnl < 0 else ""
                    
                    position.update({
                        'status': 'CLOSED',
                        'exit_price': current_price,
                        'exit_time': row['timestamp'],
                        'pnl': float(pnl * self.trade_config['leverage']),
                        'reason': reason,
                        'loss_reason': loss_reason
                    })
                    
                    self.current_balance += position['size'] * position['pnl']
                    self.trades.append(position.copy())
                    
                    pnl_percent = position['pnl'] * 100
                    emoji = "🟢" if pnl_percent > 0 else "🔴"
                    trade_type = "ورقي" if self.paper_trading else "حقيقي"
                    reason_text = f"{reason} {loss_reason}" if loss_reason else reason
                    print(f"{emoji} إغلاق صفقة {trade_type} {position['direction']} #{position['id']} - {reason_text} - {pnl_percent:+.2f}%")
    
    def analyze_loss_reason(self, position, row):
        """تحليل أسباب الخسارة"""
        reasons = []
        
        if position['direction'] == 'BUY':
            if row['rsi'] > 60:
                reasons.append("RSI مرتفع")
            if row['ema_fast'] < row['ema_slow']:
                reasons.append("الاتجاه هابط")
            if row['macd'] < row['macd_signal']:
                reasons.append("MACD هابط")
            if row['close'] < row['ema_trend']:
                reasons.append("تحت المتوسط 50")
            if row['volume'] < row['volume_ma'] * 0.8:
                reasons.append("حجم تداول منخفض")
        else:
            if row['rsi'] < 40:
                reasons.append("RSI منخفض")
            if row['ema_fast'] > row['ema_slow']:
                reasons.append("الاتجاه صاعد")
            if row['macd'] > row['macd_signal']:
                reasons.append("MACD صاعد")
            if row['close'] > row['ema_trend']:
                reasons.append("فوق المتوسط 50")
            if row['volume'] < row['volume_ma'] * 0.8:
                reasons.append("حجم تداول منخفض")
        
        return ", ".join(reasons) if reasons else "لا يوجد سبب واضح"
    
    def generate_report(self):
        """توليد تقرير الأداء"""
        if not self.trades:
            return "⚠️ لا توجد صفقات"
        
        total_trades = len(self.trades)
        winning_trades = [t for t in self.trades if t['pnl'] > 0]
        losing_trades = [t for t in self.trades if t['pnl'] <= 0]
        
        win_rate = len(winning_trades) / total_trades * 100
        
        total_balance_change = self.current_balance - self.initial_balance
        total_pnl_percent = (total_balance_change / self.initial_balance) * 100
        
        avg_win = np.mean([t['pnl'] for t in winning_trades]) if winning_trades else 0
        avg_loss = np.mean([t['pnl'] for t in losing_trades]) if losing_trades else 0
        
        profit_loss_ratio = "N/A"
        if avg_loss != 0:
            profit_loss_ratio = f"{abs(avg_win/avg_loss):.2f}"
        
        loss_analysis = self.analyze_loss_patterns(losing_trades)
        
        report = f"""
تقرير البوت - {self.trade_config['symbol']}
{'📝 (تداول ورقي)' if self.paper_trading else '💰 (تداول حقيقي)'}

الأداء الحقيقي:
- الرصيد الابتدائي: {self.initial_balance:,.2f}$
- الرصيد النهائي: {self.current_balance:,.2f}$
- إجمالي الربح/الخسارة: {total_balance_change:+,.2f}$ ({total_pnl_percent:+.2f}%)

إحصائيات التداول:
- إجمالي الصفقات: {total_trades}
- الصفقات الرابحة: {len(winning_trades)} ({win_rate:.1f}%)
- الصفقات الخاسرة: {len(losing_trades)}
- متوسط الربح: {avg_win*100:+.2f}%
- متوسط الخسارة: {avg_loss*100:.2f}%
- نسبة الربح/الخسارة: {profit_loss_ratio}

تحليل أسباب الخسارة:
{loss_analysis}

الإعدادات المستخدمة:
- وقف الخسارة: {self.trade_config['stop_loss']*100}%
- جني الأرباح: {self.trade_config['take_profit']*100}%
- حجم المركز: {self.trade_config['position_size']*100}%
- الرافعة: {self.trade_config['leverage']}x
- فلتر الحجم: {'مفعل' if self.signal_config['use_volume_filter'] else 'معطل'}
- التداول الورقي: {'مفعل' if self.paper_trading else 'معطل'}
        """
        
        return report
    
    def analyze_loss_patterns(self, losing_trades):
        """تحليل أنماط الخسارة"""
        if not losing_trades:
            return "✅ لا توجد صفقات خاسنة"
        
        stop_loss_count = len([t for t in losing_trades if t['reason'] == 'STOP_LOSS'])
        
        common_reasons = {}
        for trade in losing_trades:
            if 'loss_reason' in trade and trade['loss_reason']:
                reasons = trade['loss_reason'].split(", ")
                for reason in reasons:
                    common_reasons[reason] = common_reasons.get(reason, 0) + 1
        
        analysis = f"""
- الصفقات الخاسنة: {len(losing_trades)}
- بسبب وقف الخسارة: {stop_loss_count}
- متوسط مدة الصفقات الخاسنة: {self.calculate_avg_trade_duration(losing_trades)}

الأسباب الشائعة للخسارة:
"""
        
        for reason, count in sorted(common_reasons.items(), key=lambda x: x[1], reverse=True):
            percentage = (count / len(losing_trades)) * 100
            analysis += f"- {reason}: {count} مرة ({percentage:.1f}%)\n"
        
        if stop_loss_count / len(losing_trades) > 0.7:
            analysis += "\nتوصية: معظم الخسائر بسبب وقف الخسارة -可以考虑 زيادة وقف الخسارة قليلاً\n"
        
        return analysis
    
    def calculate_avg_trade_duration(self, trades):
        """حساب متوسط مدة الصفقات"""
        if not trades:
            return "0"
        
        durations = []
        for trade in trades:
            if 'exit_time' in trade and 'entry_time' in trade:
                duration = (trade['exit_time'] - trade['entry_time']).total_seconds() / 3600
                durations.append(duration)
        
        return f"{float(np.mean(durations)):.1f} ساعة" if durations else "غير متوفر"
    
    def send_telegram_report(self):
        """إرسال التقرير عبر التلغرام - الإصدار المعدل"""
        try:
            bot_token = TELEGRAM_CONFIG['bot_token']
            chat_id = TELEGRAM_CONFIG['chat_id']
            
            if not bot_token or not chat_id:
                print("❌ مفاتيح التلغرام غير متوفرة")
                print("يرجى تعيين TELEGRAM_BOT_TOKEN و TELEGRAM_CHAT_ID في متغيرات البيئة")
                return
            
            print(f"🔍 التحقق من مفاتيح التلغرام...")
            print(f"البوت Token: {'***' + bot_token[-4:] if bot_token else 'غير موجود'}")
            print(f"الدردشة ID: {chat_id if chat_id else 'غير موجود'}")
            
            bot = telebot.TeleBot(bot_token)
            report = self.generate_report()
            
            # تقسيم التقرير إذا كان طويلاً
            if len(report) > 4000:
                parts = [report[i:i+4000] for i in range(0, len(report), 4000)]
                for i, part in enumerate(parts):
                    bot.send_message(chat_id, f"الجزء {i+1}:\n{part}")
                    print(f"✅ تم إرسال الجزء {i+1} إلى التلغرام")
            else:
                bot.send_message(chat_id, report)
                print("✅ تم إرسال التقرير إلى التلغرام بنجاح")
                
        except Exception as e:
            print(f"❌ خطأ في إرسال التقرير إلى التلغرام: {e}")
            print("تفاصيل الخطأ:", str(e))

def main():
    print("🚀 بدء تشغيل البوت لـ 180 يوم...")
    
    # التحقق من وجود مفاتيح التلغرام
    if not TELEGRAM_CONFIG['bot_token'] or not TELEGRAM_CONFIG['chat_id']:
        print("⚠️  تنبيه: مفاتيح التلغرام غير متوفرة")
        print("يرجى تعيين القيم التالية في Render:")
        print("TELEGRAM_BOT_TOKEN=رقم_توكن_البوت")
        print("TELEGRAM_CHAT_ID=رقم_الدردشة")
    
    bot = SimpleCryptoBot(TRADE_CONFIG, INDICATOR_CONFIG, SIGNAL_CONFIG)
    bot.fetch_binance_data(days=180)
    bot.execute_backtest()
    
    report = bot.generate_report()
    print(report)
    
    # إرسال التقرير إلى التلغرام
    bot.send_telegram_report()
    
    print("✅ انتهى التشغيل")

if __name__ == "__main__":
    main()