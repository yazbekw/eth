import os
import pandas as pd
import numpy as np
import requests
from datetime import datetime, timedelta
import telebot
import warnings
warnings.filterwarnings('ignore')

# ==================== الإعدادات الأساسية - يمكن تعديلها بسهولة ====================

# إعدادات التداول
TRADE_CONFIG = {
    'symbol': 'BTCUSDT',
    'timeframe': '1h',
    'initial_balance': 1000,
    'leverage': 3,
    'stop_loss': 0.025,    # 2.5% - زيادة من 2%
    'take_profit': 0.035,  # 3.5% - تقليل من 4% لتحسين نسبة R:R
    'position_size': 0.07,  # 7% - تقليل من 10%
    'max_positions': 3
}

# إعدادات المؤشرات
INDICATOR_CONFIG = {
    'rsi_period': 21,      # زيادة من 14 إلى 21
    'rsi_overbought': 70,
    'rsi_oversold': 30,
    'ema_fast': 9,
    'ema_slow': 21,
    'ema_trend': 50,       # إضافة متوسط الاتجاه
    'macd_fast': 12,
    'macd_slow': 26,
    'macd_signal': 9
}

# إعدادات الإشارات
SIGNAL_CONFIG = {
    'min_conditions': 3,    # زيادة من 2 إلى 3 شروط
    'use_trend_filter': True,
    'use_volume_filter': False  # يمكن تفعيله لاحقاً
}

# إعدادات التلغرام
TELEGRAM_CONFIG = {
    'bot_token': os.getenv('TELEGRAM_BOT_TOKEN', ''),
    'chat_id': os.getenv('TELEGRAM_CHAT_ID', '')
}

# ==================== كود البوت الأساسي ====================

class SimpleCryptoBot:
    def __init__(self, trade_config, indicator_config, signal_config):
        self.trade_config = trade_config
        self.indicator_config = indicator_config
        self.signal_config = signal_config
        self.data = None
        self.positions = []
        self.trades = []
        self.current_balance = trade_config['initial_balance']
        
    def fetch_binance_data(self, days=90):
        """جلب البيانات من Binance"""
        try:
            symbol = self.trade_config['symbol']
            interval = self.trade_config['timeframe']
            limit = days * 24
            
            url = "https://api.binance.com/api/v3/klines"
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
            
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
            for col in ['open', 'high', 'low', 'close', 'volume']:
                df[col] = pd.to_numeric(df[col], errors='coerce')
            
            df = df[['timestamp', 'open', 'high', 'low', 'close', 'volume']]
            df = df.sort_values('timestamp').reset_index(drop=True)
            
            self.data = df
            self.calculate_indicators()
            print(f"✅ تم جلب {len(self.data)} شمعة للعملة {symbol}")
            
        except Exception as e:
            print(f"❌ خطأ في جلب البيانات: {e}")
            self.generate_sample_data(days)
    
    def generate_sample_data(self, days):
        """بيانات عينة إذا فشل الاتصال"""
        print("📊 استخدام بيانات عينة...")
        
        end_date = datetime.now()
        start_date = end_date - timedelta(days=days)
        dates = pd.date_range(start=start_date, end=end_date, freq='1H')
        
        np.random.seed(42)
        price = 30000.0
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
        self.data['ema_fast'] = self.data['close'].ewm(
            span=self.indicator_config['ema_fast'], adjust=False).mean()
        self.data['ema_slow'] = self.data['close'].ewm(
            span=self.indicator_config['ema_slow'], adjust=False).mean()
        self.data['ema_trend'] = self.data['close'].ewm(
            span=self.indicator_config['ema_trend'], adjust=False).mean()
        
        # MACD
        ema_fast = self.data['close'].ewm(span=self.indicator_config['macd_fast'], adjust=False).mean()
        ema_slow = self.data['close'].ewm(span=self.indicator_config['macd_slow'], adjust=False).mean()
        self.data['macd'] = ema_fast - ema_slow
        self.data['macd_signal'] = self.data['macd'].ewm(
            span=self.indicator_config['macd_signal'], adjust=False).mean()
        
        print("✅ تم حساب المؤشرات الفنية")
    
    def generate_signal(self, row):
        """توليد إشارات التداول مع التحسينات"""
        if any(pd.isna(row[key]) for key in ['rsi', 'ema_slow', 'macd', 'ema_trend']):
            return 'HOLD', 0
        
        buy_conditions = 0
        sell_conditions = 0
        
        # الشروط الأساسية
        if row['rsi'] < self.indicator_config['rsi_oversold']:
            buy_conditions += 1
        elif row['rsi'] > self.indicator_config['rsi_overbought']:
            sell_conditions += 1
            
        if row['ema_fast'] > row['ema_slow']:
            buy_conditions += 1
        else:
            sell_conditions += 1
            
        if row['macd'] > row['macd_signal']:
            buy_conditions += 1
        else:
            sell_conditions += 1
        
        # تصفية الاتجاه (التحسين الجديد)
        if self.signal_config['use_trend_filter']:
            if row['close'] > row['ema_trend']:
                buy_conditions += 0.5  # تعزيز إشارات الشراء
            else:
                sell_conditions += 0.5  # تعزيز إشارات البيع
        
        signal = 'HOLD'
        strength = 0
        
        # الدخول فقط عند تحقيق العدد المطلوب من الشروط
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
        
        min_period = max(
            self.indicator_config['ema_slow'],
            self.indicator_config['rsi_period'], 
            self.indicator_config['ema_trend']
        )
        
        for i, row in self.data.iterrows():
            if i < min_period:
                continue
            
            signal, strength = self.generate_signal(row)
            self.check_exit_conditions(row)
            
            # فتح صفقات جديدة
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
            'status': 'OPEN'
        }
        
        self.positions.append(position)
        print(f"📈 فتح {direction} #{position['id']} بسعر {row['close']:.2f}")
    
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
                    position.update({
                        'status': 'CLOSED',
                        'exit_price': current_price,
                        'exit_time': row['timestamp'],
                        'pnl': float(pnl * self.trade_config['leverage']),
                        'reason': reason
                    })
                    
                    self.current_balance += position['size'] * position['pnl']
                    self.trades.append(position.copy())
                    
                    pnl_percent = position['pnl'] * 100
                    emoji = "🟢" if pnl_percent > 0 else "🔴"
                    print(f"{emoji} إغلاق {position['direction']} #{position['id']} - {reason} - {pnl_percent:+.2f}%")
    
    def generate_report(self):
        """توليد تقرير الأداء"""
        if not self.trades:
            return "⚠️ لا توجد صفقات"
        
        total_trades = len(self.trades)
        winning_trades = [t for t in self.trades if t['pnl'] > 0]
        losing_trades = [t for t in self.trades if t['pnl'] <= 0]
        
        win_rate = len(winning_trades) / total_trades * 100
        total_pnl = sum(t['pnl'] for t in self.trades)
        total_pnl_percent = (total_pnl / self.trade_config['initial_balance']) * 100
        
        avg_win = np.mean([t['pnl'] for t in winning_trades]) if winning_trades else 0
        avg_loss = np.mean([t['pnl'] for t in losing_trades]) if losing_trades else 0
        
        report = f"""
📊 **تقرير البوت - {self.trade_config['symbol']}**

**الأداء:**
• الصفقات: {total_trades} (ربح: {len(winning_trades)}, خسارة: {len(losing_trades)})
• نسبة الفوز: {win_rate:.1f}%
• إجمالي الربح: {total_pnl:+.2f}$ ({total_pnl_percent:+.2f}%)
• متوسط الربح: {avg_win*100:+.2f}%
• متوسط الخسارة: {avg_loss*100:.2f}%
• الرصيد النهائي: {self.current_balance:.2f}$

**التحسينات المطبقة:**
• وقف الخسارة: {self.trade_config['stop_loss']*100}% (↑ من 2%)
• جني الأرباح: {self.trade_config['take_profit']*100}% (↓ من 4%) 
• حجم المركز: {self.trade_config['position_size']*100}% (↓ من 10%)
• RSI: {self.indicator_config['rsi_period']} (↑ من 14)
• شروط الدخول: {self.signal_config['min_conditions']}/3 (↑ من 2/3)
• فلتر الاتجاه: {self.signal_config['use_trend_filter']}
        """
        
        return report
    
    def send_telegram_report(self):
        """إرسال التقرير عبر التلغرام"""
        try:
            if not TELEGRAM_CONFIG['bot_token'] or not TELEGRAM_CONFIG['chat_id']:
                return
            
            bot = telebot.TeleBot(TELEGRAM_CONFIG['bot_token'])
            report = self.generate_report()
            bot.send_message(TELEGRAM_CONFIG['chat_id'], report, parse_mode='Markdown')
            print("✅ تم إرسال التقرير")
            
        except Exception as e:
            print(f"❌ خطأ في الإرسال: {e}")

# ==================== التشغيل الرئيسي ====================

def main():
    print("🚀 بدء تشغيل البوت مع التحسينات...")
    
    bot = SimpleCryptoBot(TRADE_CONFIG, INDICATOR_CONFIG, SIGNAL_CONFIG)
    bot.fetch_binance_data(days=90)
    bot.execute_backtest()
    
    report = bot.generate_report()
    print(report)
    
    bot.send_telegram_report()
    print("✅ انتهى التشغيل")

if __name__ == "__main__":
    main()
