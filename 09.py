import os
import pandas as pd
import numpy as np
import requests
from datetime import datetime, timedelta
import telebot
import warnings
warnings.filterwarnings('ignore')

# ==================== الإعدادات النهائية المحسنة ====================

TRADE_CONFIG = {
    'symbol': 'BNBUSDT',
    'timeframe': '1h',
    'initial_balance': 200,
    'leverage': 1,
    'stop_loss': 0.015,        # تقليل إضافي إلى 1.5%
    'take_profit': 0.045,      # تقليل إلى 4.5% لنسبة أفضل
    'position_size': 0.06,     # تقليل حجم المركز
    'max_positions': 1,        # صفقة واحدة فقط في الوقت
    'paper_trading': True,
    'use_trailing_stop': True,
    'max_trade_duration': 48   # تقليل المدة القصوى
}

INDICATOR_CONFIG = {
    'rsi_period': 14,
    'rsi_overbought': 70,      # إرجاع القيم الطبيعية
    'rsi_oversold': 30,
    'ema_fast': 9,
    'ema_slow': 21,
    'ema_trend': 50,
    'macd_fast': 12,
    'macd_slow': 26,
    'macd_signal': 9
}

SIGNAL_CONFIG = {
    'min_conditions': 3,       # زيادة إلى 4 شروط
    'use_trend_filter': True,
    'use_volume_filter': True,
    'min_volume_ratio': 1,   # زيادة عتبة الحجم
    'require_trend_confirmation': False,
    'prevent_conflicts': True,
    'min_signal_strength': 3   # حد أدنى لقوة الإشارة
}

TELEGRAM_CONFIG = {
    'bot_token': os.getenv('TELEGRAM_BOT_TOKEN', ''),
    'chat_id': os.getenv('TELEGRAM_CHAT_ID', '')
}

class FinalCryptoBot:
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
        
    def fetch_binance_data(self, days=30):
        """جلب البيانات من Binance"""
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
            print(f"✅ تم جلب {len(self.data)} شمعة للعملة {symbol}")
            
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
        self.data['macd_histogram'] = self.data['macd'] - self.data['macd_signal']
        
        # حجم التداول المتوسط
        self.data['volume_ma'] = self.data['volume'].rolling(window=20).mean()
        
        # مؤشرات إضافية
        self.data['price_change'] = self.data['close'].pct_change()
        self.data['volatility'] = self.data['price_change'].rolling(window=20).std()
        
        print("✅ تم حساب المؤشرات الفنية")
    
    def generate_signal(self, row):
        """توليد إشارات تداول محكمة"""
        if any(pd.isna(row[key]) for key in ['rsi', 'ema_slow', 'macd', 'ema_trend', 'volume_ma', 'volatility']):
            return 'HOLD', 0, "بيانات ناقصة"
        
        buy_conditions = 0
        sell_conditions = 0
        condition_details = []
        
        # 1. شرط RSI مع نطاق أضيق
        if row['rsi'] < 28:  # أكثر تشدداً للشراء
            buy_conditions += 1
            condition_details.append("RSI شديد الانخفاض")
        elif row['rsi'] > 72:  # أكثر تشدداً للبيع
            sell_conditions += 1
            condition_details.append("RSI شديد الارتفاع")
        elif 30 <= row['rsi'] <= 70:
            condition_details.append("RSI في النطاق المحايد")
        
        # 2. شرط EMA مع تأكيد الاتجاه
        ema_bullish = row['ema_fast'] > row['ema_slow'] and row['close'] > row['ema_slow']
        ema_bearish = row['ema_fast'] < row['ema_slow'] and row['close'] < row['ema_slow']
        
        if ema_bullish:
            buy_conditions += 1
            condition_details.append("EMA صاعد")
        elif ema_bearish:
            sell_conditions += 1
            condition_details.append("EMA هابط")
        
        # 3. شرط MACD مع قوة الإشارة
        macd_strength = abs(row['macd_histogram']) > (row['close'] * 0.002)
        if row['macd'] > row['macd_signal'] and macd_strength and row['macd_histogram'] > 0:
            buy_conditions += 1
            condition_details.append("MACD صاعد قوي")
        elif row['macd'] < row['macd_signal'] and macd_strength and row['macd_histogram'] < 0:
            sell_conditions += 1
            condition_details.append("MACD هابط قوي")
        
        # 4. فلتر الاتجاه الرئيسي
        if self.signal_config['use_trend_filter']:
            trend_bullish = row['close'] > row['ema_trend'] and row['ema_fast'] > row['ema_trend']
            trend_bearish = row['close'] < row['ema_trend'] and row['ema_fast'] < row['ema_trend']
            
            if trend_bullish:
                buy_conditions += 1
                condition_details.append("الاتجاه صاعد قوي")
            elif trend_bearish:
                sell_conditions += 1
                condition_details.append("الاتجاه هابط قوي")
        
        # 5. فلتر الحجم المعزز
        volume_ok = row['volume'] > row['volume_ma'] * self.signal_config['min_volume_ratio']
        if not volume_ok:
            condition_details.append("الحجم غير كافٍ")
            return 'HOLD', 0, " | ".join(condition_details)
        else:
            condition_details.append("الحجم جيد")
        
        # 6. فلتر التقلبات
        low_volatility = row['volatility'] < row['close'] * 0.01
        if low_volatility:
            condition_details.append("تقلبات منخفضة")
        
        # 🔥 فحص التعارض الصارم
        has_conflict = False
        conflict_reason = ""
        
        if buy_conditions >= self.signal_config['min_conditions']:
            # فحص تعارض الشراء
            if (row['close'] < row['ema_trend'] or 
                row['macd_histogram'] < 0 or
                row['rsi'] > 60):
                has_conflict = True
                conflict_reason = "تعارض في إشارة الشراء"
        
        elif sell_conditions >= self.signal_config['min_conditions']:
            # فحص تعارض البيع
            if (row['close'] > row['ema_trend'] or 
                row['macd_histogram'] > 0 or
                row['rsi'] < 40):
                has_conflict = True
                conflict_reason = "تعارض في إشارة البيع"
        
        # اتخاذ القرار النهائي
        signal = 'HOLD'
        strength = 0
        min_conditions = self.signal_config['min_conditions']
        
        if (buy_conditions >= min_conditions and 
            not has_conflict and 
            volume_ok and
            buy_conditions >= self.signal_config.get('min_signal_strength', 3)):
            signal = 'BUY'
            strength = buy_conditions
        elif (sell_conditions >= min_conditions and 
              not has_conflict and 
              volume_ok and
              sell_conditions >= self.signal_config.get('min_signal_strength', 3)):
            signal = 'SELL'
            strength = sell_conditions
        
        details = " | ".join(condition_details)
        if has_conflict:
            details += f" | {conflict_reason}"
        
        return signal, strength, details
    
    def execute_backtest(self):
        """تنفيذ الباك تستينغ المحكم"""
        print("🔄 بدء الباك تستينغ المحكم...")
        print(f"🎯 الإعدادات: {self.signal_config['min_conditions']} شروط بحد أدنى قوة {self.signal_config.get('min_signal_strength', 4)}")
        
        min_period = max(
            self.indicator_config['ema_slow'],
            self.indicator_config['rsi_period'], 
            self.indicator_config['ema_trend'],
            20
        )
        
        for i, row in self.data.iterrows():
            if i < min_period:
                continue
            
            signal, strength, details = self.generate_signal(row)
            self.check_exit_conditions(row)
            
            open_positions = len([p for p in self.positions if p['status'] == 'OPEN'])
            
            if (signal in ['BUY', 'SELL'] and 
                open_positions < self.trade_config['max_positions'] and
                strength >= self.signal_config.get('min_signal_strength', 4)):
                
                self.open_position(signal, strength, row, details)
        
        print(f"✅ تم الانتهاء - {len(self.trades)} صفقة")
    
    def open_position(self, direction, strength, row, details):
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
            'type': 'PAPER' if self.paper_trading else 'REAL',
            'signal_strength': strength,
            'signal_details': details
        }
        
        self.positions.append(position)
        trade_type = "ورقي" if self.paper_trading else "حقيقي"
        print(f"📈 فتح صفقة {trade_type} {direction} #{position['id']}")
        print(f"   💪 قوة: {strength}/4 | 📊 التفاصيل: {details}")
    
    def check_exit_conditions(self, row):
        """فحص شروط الخروج"""
        current_price = float(row['close'])
        
        for position in self.positions:
            if position['status'] == 'OPEN':
                pnl = 0.0
                reason = ''
                
                # فحص المدة الزمنية القصوى
                duration_hours = (row['timestamp'] - position['entry_time']).total_seconds() / 3600
                if duration_hours > self.trade_config['max_trade_duration']:
                    if position['direction'] == 'BUY':
                        pnl = (current_price - position['entry_price']) / position['entry_price']
                    else:
                        pnl = (position['entry_price'] - current_price) / position['entry_price']
                    reason = 'TIME_EXIT'
                
                # فحص وقف الخسارة وجني الأرباح
                elif position['direction'] == 'BUY':
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
                
                # تطبيق الوقف المتتبع
                if (reason in ['', 'TIME_EXIT'] and 
                    self.trade_config.get('use_trailing_stop', False)):
                    self.trailing_stop_loss(position, current_price)
                
                if reason:
                    loss_reason = self.analyze_loss_reason(position, row) if pnl < 0 else ""
                    
                    position.update({
                        'status': 'CLOSED',
                        'exit_price': current_price,
                        'exit_time': row['timestamp'],
                        'pnl': float(pnl * self.trade_config['leverage']),
                        'reason': reason,
                        'loss_reason': loss_reason,
                        'duration_hours': duration_hours
                    })
                    
                    self.current_balance += position['size'] * position['pnl']
                    self.trades.append(position.copy())
                    
                    pnl_percent = position['pnl'] * 100
                    emoji = "🟢" if pnl_percent > 0 else "🔴"
                    trade_type = "ورقي" if self.paper_trading else "حقيقي"
                    print(f"{emoji} إغلاق صفقة {trade_type} {position['direction']} #{position['id']} - {reason} - {pnl_percent:+.2f}%")
    
    def trailing_stop_loss(self, position, current_price):
        """وقف خسارة متتبع"""
        if position['direction'] == 'BUY':
            unrealized_pnl = (current_price - position['entry_price']) / position['entry_price']
            if unrealized_pnl > 0.015:  # عندما يصل الربح إلى 1.5%
                new_stop_loss = current_price * (1 - self.trade_config['stop_loss'] * 0.5)
                if new_stop_loss > position['stop_loss']:
                    position['stop_loss'] = new_stop_loss
        else:
            unrealized_pnl = (position['entry_price'] - current_price) / position['entry_price']
            if unrealized_pnl > 0.015:
                new_stop_loss = current_price * (1 + self.trade_config['stop_loss'] * 0.5)
                if new_stop_loss < position['stop_loss']:
                    position['stop_loss'] = new_stop_loss
    
    def analyze_loss_reason(self, position, row):
        """تحليل أسباب الخسارة"""
        reasons = []
        
        if position['direction'] == 'BUY':
            if row['rsi'] > 65: reasons.append("RSI مرتفع")
            if row['ema_fast'] < row['ema_slow']: reasons.append("الاتجاه هابط")
            if row['macd_histogram'] < 0: reasons.append("MACD هابط")
            if row['close'] < row['ema_trend']: reasons.append("تحت المتوسط 50")
        else:
            if row['rsi'] < 35: reasons.append("RSI منخفض")
            if row['ema_fast'] > row['ema_slow']: reasons.append("الاتجاه صاعد")
            if row['macd_histogram'] > 0: reasons.append("MACD صاعد")
            if row['close'] > row['ema_trend']: reasons.append("فوق المتوسط 50")
        
        return ", ".join(reasons) if reasons else "لا أسباب واضحة"
    
    def generate_report(self):
        """توليد تقرير الأداء"""
        if not self.trades:
            return "⚠️ لا توجد صفقات"
        
        total_trades = len(self.trades)
        winning_trades = [t for t in self.trades if t['pnl'] > 0]
        losing_trades = [t for t in self.trades if t['pnl'] <= 0]
        
        win_rate = len(winning_trades) / total_trades * 100 if total_trades > 0 else 0
        
        total_balance_change = self.current_balance - self.initial_balance
        total_pnl_percent = (total_balance_change / self.initial_balance) * 100
        
        avg_win = np.mean([t['pnl'] for t in winning_trades]) * 100 if winning_trades else 0
        avg_loss = np.mean([t['pnl'] for t in losing_trades]) * 100 if losing_trades else 0
        
        # تحليل الصفقات القوية
        strong_trades = [t for t in self.trades if t.get('signal_strength', 0) >= 4]
        strong_win_rate = len([t for t in strong_trades if t['pnl'] > 0]) / len(strong_trades) * 100 if strong_trades else 0
        
        report = f"""
📊 التقرير النهائي - {self.trade_config['symbol']}
{'📝 (تداول ورقي)' if self.paper_trading else '💰 (تداول حقيقي)'}

الأداء المالي:
- الرصيد الابتدائي: {self.initial_balance:,.2f}$
- الرصيد النهائي: {self.current_balance:,.2f}$
- إجمالي الربح/الخسارة: {total_balance_change:+,.2f}$ ({total_pnl_percent:+.2f}%)

إحصائيات التداول:
- إجمالي الصفقات: {total_trades}
- الصفقات الرابحة: {len(winning_trades)} ({win_rate:.1f}%)
- الصفقات الخاسرة: {len(losing_trades)}
- متوسط الربح: {avg_win:+.2f}%
- متوسط الخسارة: {avg_loss:.2f}%
- نسبة الربح/الخسارة: {abs(avg_win/avg_loss):.2f if avg_loss != 0 else 'N/A'}

جودة الإشارات:
- الصفقات قوية الإشارة: {len(strong_trades)}
- نسبة ربح الصفقات القوية: {strong_win_rate:.1f}%

الإعدادات المستخدمة:
- وقف الخسارة: {self.trade_config['stop_loss']*100}%
- جني الأرباح: {self.trade_config['take_profit']*100}%
- حجم المركز: {self.trade_config['position_size']*100}%
- الرافعة: {self.trade_config['leverage']}x
- الشروط المطلوبة: {self.signal_config['min_conditions']}
- حد قوة الإشارة: {self.signal_config.get('min_signal_strength', 4)}
        """
        
        return report

def main():
    print("🚀 بدء تشغيل البوت النهائي...")
    
    bot = FinalCryptoBot(TRADE_CONFIG, INDICATOR_CONFIG, SIGNAL_CONFIG)
    bot.fetch_binance_data(days=30)
    bot.execute_backtest()
    
    report = bot.generate_report()
    print(report)
    
    print("✅ انتهى التشغيل بنجاح")

if __name__ == "__main__":
    main()
