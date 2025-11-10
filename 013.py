import os
import pandas as pd
import numpy as np
import requests
from datetime import datetime, timedelta
import telebot
import warnings
warnings.filterwarnings('ignore')

# ==================== الإعدادات المحسنة ====================

TRADE_CONFIG = {
    'symbol': 'BNBUSDT',
    'timeframe': '1h',
    'initial_balance': 200,
    'leverage': 1,
    'stop_loss': 0.025,        # تقليل من 3.5% إلى 2.5%
    'take_profit': 0.06,       # زيادة من 5% إلى 6%
    'position_size': 0.08,     # تقليل حجم المركز
    'max_positions': 2,        # تقليل المراكز المتزامنة
    'paper_trading': True,
    'use_trailing_stop': True, # إضافة وقف متتبع
    'max_trade_duration': 72   # أقصى مدة للصفقة (ساعة)
}

INDICATOR_CONFIG = {
    'rsi_period': 14,          # تقليل الفترة
    'rsi_overbought': 65,      # تقليل العتبات
    'rsi_oversold': 35,
    'ema_fast': 8,
    'ema_slow': 21,
    'ema_trend': 50,
    'macd_fast': 12,
    'macd_slow': 26,
    'macd_signal': 9
}

SIGNAL_CONFIG = {
    'min_conditions': 3,
    'use_trend_filter': True,
    'use_volume_filter': True,
    'min_volume_ratio': 1.0,   # زيادة عتبة الحجم
    'require_volume_confirmation': True,
    'prevent_indicator_conflicts': True
}

TELEGRAM_CONFIG = {
    'bot_token': os.getenv('TELEGRAM_BOT_TOKEN', ''),
    'chat_id': os.getenv('TELEGRAM_CHAT_ID', '')
}

class ImprovedCryptoBot:
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
        """حساب المؤشرات الفنية المحسنة"""
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
        
        print("✅ تم حساب المؤشرات الفنية المحسنة")
    
    def generate_signal(self, row):
        """توليد إشارات تداول محسنة بدون تعارض"""
        if any(pd.isna(row[key]) for key in ['rsi', 'ema_slow', 'macd', 'ema_trend', 'volume_ma']):
            return 'HOLD', 0
        
        buy_conditions = 0
        sell_conditions = 0
        
        # 1. شرط RSI (نقطة كاملة)
        if row['rsi'] < self.indicator_config['rsi_oversold']:
            buy_conditions += 1
        elif row['rsi'] > self.indicator_config['rsi_overbought']:
            sell_conditions += 1
            
        # 2. شرط EMA (نقطة كاملة)
        if row['ema_fast'] > row['ema_slow']:
            buy_conditions += 1
        else:
            sell_conditions += 1
            
        # 3. شرط MACD (نقطة كاملة)
        macd_strength = abs(row['macd'] - row['macd_signal']) > (row['close'] * 0.001)
        if row['macd'] > row['macd_signal'] and macd_strength:
            buy_conditions += 1
        elif row['macd'] < row['macd_signal'] and macd_strength:
            sell_conditions += 1
        
        # 4. فلتر الاتجاه (نقطة كاملة)
        if self.signal_config['use_trend_filter']:
            if row['close'] > row['ema_trend']:
                buy_conditions += 1
            else:
                sell_conditions += 1
        
        # 5. فلتر الحجم (شرط إلزامي)
        volume_ok = True
        if self.signal_config['use_volume_filter']:
            volume_ok = row['volume'] > row['volume_ma'] * self.signal_config['min_volume_ratio']
        
        # 🔥 منع التعارض بين المؤشرات
        has_conflict = False
        if self.signal_config['prevent_indicator_conflicts']:
            if buy_conditions >= self.signal_config['min_conditions']:
                # إذا كانت إشارة شراء، تأكد أن المؤشرات لا تعارضها بشدة
                if (row['macd'] < row['macd_signal'] and 
                    row['close'] < row['ema_trend'] and 
                    buy_conditions == self.signal_config['min_conditions']):
                    has_conflict = True
            elif sell_conditions >= self.signal_config['min_conditions']:
                # إذا كانت إشارة بيع، تأكد أن المؤشرات لا تعارضها بشدة
                if (row['macd'] > row['macd_signal'] and 
                    row['close'] > row['ema_trend'] and 
                    sell_conditions == self.signal_config['min_conditions']):
                    has_conflict = True
        
        # اتخاذ القرار النهائي
        signal = 'HOLD'
        strength = 0
        min_conditions = self.signal_config['min_conditions']
        
        if (buy_conditions >= min_conditions and 
            not has_conflict and 
            volume_ok):
            signal = 'BUY'
            strength = buy_conditions
        elif (sell_conditions >= min_conditions and 
              not has_conflict and 
              volume_ok):
            signal = 'SELL'
            strength = sell_conditions
        
        return signal, strength
    
    def dynamic_position_sizing(self, signal_strength):
        """حجم مركز ديناميكي بناءً على قوة الإشارة"""
        base_size = self.trade_config['position_size']
        
        if signal_strength >= 4:
            size_multiplier = 1.2
        elif signal_strength >= 3:
            size_multiplier = 1.0
        else:
            size_multiplier = 0.5
        
        return base_size * size_multiplier
    
    def execute_backtest(self):
        """تنفيذ الباك تستينغ المحسن"""
        print("🔄 بدء الباك تستينغ المحسن...")
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
                self.open_position(signal, strength, row)
        
        print(f"✅ تم الانتهاء - {len(self.trades)} صفقة")
    
    def open_position(self, direction, strength, row):
        """فتح صفقة جديدة بحجم ديناميكي"""
        position_size_percent = self.dynamic_position_sizing(strength)
        position_size = self.current_balance * position_size_percent * self.trade_config['leverage']
        
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
            'signal_strength': strength
        }
        
        self.positions.append(position)
        trade_type = "ورقي" if self.paper_trading else "حقيقي"
        size_percent = position_size_percent * 100
        print(f"📈 فتح صفقة {trade_type} {direction} #{position['id']} قوة: {strength}/4 حجم: {size_percent:.1f}%")
    
    def check_exit_conditions(self, row):
        """فحص شروط الخروج المحسنة"""
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
                    reason_text = f"{reason} {loss_reason}" if loss_reason else reason
                    print(f"{emoji} إغلاق صفقة {trade_type} {position['direction']} #{position['id']} - {reason_text} - {pnl_percent:+.2f}%")
    
    def trailing_stop_loss(self, position, current_price):
        """وقف خسارة متتبع للأرباح"""
        if position['direction'] == 'BUY':
            unrealized_pnl = (current_price - position['entry_price']) / position['entry_price']
            
            # عندما يصل الربح إلى 2%， ابدأ التتبع
            if unrealized_pnl > 0.02:
                new_stop_loss = current_price * (1 - self.trade_config['stop_loss'] * 0.7)
                if new_stop_loss > position['stop_loss']:
                    position['stop_loss'] = new_stop_loss
        else:
            unrealized_pnl = (position['entry_price'] - current_price) / position['entry_price']
            
            if unrealized_pnl > 0.02:
                new_stop_loss = current_price * (1 + self.trade_config['stop_loss'] * 0.7)
                if new_stop_loss < position['stop_loss']:
                    position['stop_loss'] = new_stop_loss
    
    def analyze_loss_reason(self, position, row):
        """تحليل أسباب الخسارة المحسن"""
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
        
        return ", ".join(reasons) if reasons else "تحليل: لا يوجد سبب واضح"
    
    def generate_report(self):
        """توليد تقرير الأداء المحسن"""
        if not self.trades:
            return "⚠️ لا توجد صفقات"
        
        total_trades = len(self.trades)
        winning_trades = [t for t in self.trades if t['pnl'] > 0]
        losing_trades = [t for t in self.trades if t['pnl'] <= 0]
        
        win_rate = len(winning_trades) / total_trades * 100
        
        total_balance_change = self.current_balance - self.initial_balance
        total_pnl_percent = (total_balance_change / self.initial_balance) * 100
        
        avg_win = np.mean([t['pnl'] for t in winning_trades]) * 100 if winning_trades else 0
        avg_loss = np.mean([t['pnl'] for t in losing_trades]) * 100 if losing_trades else 0
        
        profit_loss_ratio = "N/A"
        if avg_loss != 0:
            profit_loss_ratio = f"{abs(avg_win/avg_loss):.2f}"
        
        # تحليل الصفقات حسب قوة الإشارة
        strong_trades = [t for t in self.trades if t.get('signal_strength', 0) >= 4]
        strong_win_rate = len([t for t in strong_trades if t['pnl'] > 0]) / len(strong_trades) * 100 if strong_trades else 0
        
        loss_analysis = self.analyze_loss_patterns(losing_trades)
        
        report = f"""
📊 تقرير البوت المحسن - {self.trade_config['symbol']}
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
- نسبة الربح/الخسارة: {profit_loss_ratio}

تحليل الجودة:
- الصفقات قوية الإشارة: {len(strong_trades)}
- نسبة ربح الصفقات القوية: {strong_win_rate:.1f}%

{loss_analysis}

الإعدادات المستخدمة:
- وقف الخسارة: {self.trade_config['stop_loss']*100}%
- جني الأرباح: {self.trade_config['take_profit']*100}%
- حجم المركز: {self.trade_config['position_size']*100}%
- الرافعة: {self.trade_config['leverage']}x
- الوقف المتتبع: {'مفعل' if self.trade_config.get('use_trailing_stop', False) else 'معطل'}
- أقصى مدة صفقة: {self.trade_config.get('max_trade_duration', 'N/A')} ساعة
        """
        
        return report
    
    def analyze_loss_patterns(self, losing_trades):
        """تحليل أنماط الخسارة المحسن"""
        if not losing_trades:
            return "✅ لا توجد صفقات خاسنة"
        
        stop_loss_count = len([t for t in losing_trades if t['reason'] == 'STOP_LOSS'])
        time_exit_count = len([t for t in losing_trades if t['reason'] == 'TIME_EXIT'])
        
        common_reasons = {}
        for trade in losing_trades:
            if 'loss_reason' in trade and trade['loss_reason']:
                reasons = trade['loss_reason'].split(", ")
                for reason in reasons:
                    common_reasons[reason] = common_reasons.get(reason, 0) + 1
        
        # تحليل قوة الإشارة في الصفقات الخاسنة
        weak_signals = len([t for t in losing_trades if t.get('signal_strength', 0) <= 3])
        
        analysis = f"""
تحليل الصفقات الخاسنة ({len(losing_trades)} صفقة):
- بسبب وقف الخسارة: {stop_loss_count}
- بسبب انتهاء الوقت: {time_exit_count}
- متوسط المدة: {self.calculate_avg_trade_duration(losing_trades)}
- صفقات بإشارة ضعيفة: {weak_signals} ({weak_signals/len(losing_trades)*100:.1f}%)

الأسباب الشائعة:
"""
        
        for reason, count in sorted(common_reasons.items(), key=lambda x: x[1], reverse=True):
            percentage = (count / len(losing_trades)) * 100
            analysis += f"- {reason}: {count} مرة ({percentage:.1f}%)\n"
        
        # توصيات بناءً على التحليل
        if stop_loss_count / len(losing_trades) > 0.6:
            analysis += "\n💡 التوصية: معظم الخسائر بسبب وقف الخسارة - يمكن زيادة وقف الخسارة قليلاً أو تحسين توقيت الدخول\n"
        
        if weak_signals / len(losing_trades) > 0.5:
            analysis += "💡 التوصية: معظم الخسائر كانت بإشارات ضعيفة - رفع عتبة قوة الإشارة المطلوبة\n"
        
        return analysis
    
    def calculate_avg_trade_duration(self, trades):
        """حساب متوسط مدة الصفقات"""
        if not trades:
            return "0"
        
        durations = []
        for trade in trades:
            if 'duration_hours' in trade:
                durations.append(trade['duration_hours'])
        
        return f"{float(np.mean(durations)):.1f} ساعة" if durations else "غير متوفر"
    
    def send_telegram_report(self):
        """إرسال التقرير عبر التلغرام"""
        try:
            bot_token = TELEGRAM_CONFIG['bot_token']
            chat_id = TELEGRAM_CONFIG['chat_id']
            
            if not bot_token or not chat_id:
                print("❌ مفاتيح التلغرام غير متوفرة")
                return
            
            bot = telebot.TeleBot(bot_token)
            report = self.generate_report()
            
            if len(report) > 4000:
                parts = [report[i:i+4000] for i in range(0, len(report), 4000)]
                for i, part in enumerate(parts):
                    bot.send_message(chat_id, f"الجزء {i+1}:\n{part}")
            else:
                bot.send_message(chat_id, report)
                print("✅ تم إرسال التقرير إلى التلغرام بنجاح")
                
        except Exception as e:
            print(f"❌ خطأ في إرسال التقرير إلى التلغرام: {e}")

def main():
    print("🚀 بدء تشغيل البوت المحسن...")
    
    # التحقق من وجود مفاتيح التلغرام
    if not TELEGRAM_CONFIG['bot_token'] or not TELEGRAM_CONFIG['chat_id']:
        print("⚠️  تنبيه: مفاتيح التلغرام غير متوفرة")
    
    bot = ImprovedCryptoBot(TRADE_CONFIG, INDICATOR_CONFIG, SIGNAL_CONFIG)
    bot.fetch_binance_data(days=30)
    bot.execute_backtest()
    
    report = bot.generate_report()
    print(report)
    
    # إرسال التقرير إلى التلغرام
    bot.send_telegram_report()
    
    print("✅ انتهى التشغيل بنجاح")

if __name__ == "__main__":
    main()
