import os
import pandas as pd
import numpy as np
import requests
from datetime import datetime, timedelta
import telebot
import warnings
warnings.filterwarnings('ignore')

# ==================== الإعدادات المخففة مع قوة الإشارة المتدرجة ====================

TRADE_CONFIG = {
    'symbol': 'BNBUSDT',
    'timeframe': '4h',
    'initial_balance': 200,
    'leverage': 1,
    'stop_loss': 0.025,
    'take_profit': 0.060,
    'position_size': 0.1,
    'max_positions': 4,
    'paper_trading': True,
    'use_trailing_stop': True,
    'max_trade_duration': 60
}

INDICATOR_CONFIG = {
    'rsi_period': 14,
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
    'min_conditions': 3,
    'use_trend_filter': True,
    'use_volume_filter': True,
    'min_volume_ratio': 1.0,
    'require_trend_confirmation': True,
    'prevent_conflicts': True,
    'min_signal_strength': 4,
    'max_signal_strength': 8
}

TELEGRAM_CONFIG = {
    'bot_token': os.getenv('TELEGRAM_BOT_TOKEN', ''),
    'chat_id': os.getenv('TELEGRAM_CHAT_ID', '')
}

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
        self.signal_strength_results = {}
        
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
                'limit': 1000
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
        gain = (delta.where(delta > 0, 0)).fillna(0)
        loss = (-delta.where(delta < 0, 0)).fillna(0)
        
        avg_gain = gain.rolling(window=self.indicator_config['rsi_period'], min_periods=1).mean()
        avg_loss = loss.rolling(window=self.indicator_config['rsi_period'], min_periods=1).mean()
        
        rs = avg_gain / avg_loss
        rs = rs.replace([np.inf, -np.inf], 0).fillna(0)
        
        self.data['rsi'] = 100 - (100 / (1 + rs))
        
        # المتوسطات المتحركة
        self.data['ema_fast'] = self.data['close'].ewm(
            span=self.indicator_config['ema_fast'], adjust=False, min_periods=1
        ).mean()
        self.data['ema_slow'] = self.data['close'].ewm(
            span=self.indicator_config['ema_slow'], adjust=False, min_periods=1
        ).mean()
        self.data['ema_trend'] = self.data['close'].ewm(
            span=self.indicator_config['ema_trend'], adjust=False, min_periods=1
        ).mean()
        
        # MACD
        ema_fast = self.data['close'].ewm(
            span=self.indicator_config['macd_fast'], adjust=False, min_periods=1
        ).mean()
        ema_slow = self.data['close'].ewm(
            span=self.indicator_config['macd_slow'], adjust=False, min_periods=1
        ).mean()
        
        self.data['macd'] = ema_fast - ema_slow
        self.data['macd_signal'] = self.data['macd'].ewm(
            span=self.indicator_config['macd_signal'], adjust=False, min_periods=1
        ).mean()
        self.data['macd_histogram'] = self.data['macd'] - self.data['macd_signal']
        
        # حجم التداول المتوسط
        self.data['volume_ma'] = self.data['volume'].rolling(window=20, min_periods=1).mean()
        
        print("✅ تم حساب المؤشرات الفنية")
    
    def calculate_signal_strength(self, buy_conditions, sell_conditions, row):
        """حساب قوة الإشارة المتدرجة من 1-10"""
        try:
            # استخدام الحد الأقصى لعدد الشروط
            base_conditions = max(buy_conditions, sell_conditions)
            
            if base_conditions == 0:
                return 1
            
            # الحصول على قيم المؤشرات
            rsi = row['rsi'] if not pd.isna(row['rsi']) else 50
            ema_trend_position = 1 if row['close'] > row['ema_trend'] else 0
            macd_strength = abs(row['macd_histogram']) / row['close'] * 1000 if not pd.isna(row['macd_histogram']) else 0
            volume_strength = min(row['volume'] / row['volume_ma'], 3) if not pd.isna(row['volume_ma']) and row['volume_ma'] > 0 else 1
            
            # حساب نقاط القوة الإضافية
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
            if macd_strength > 0.5:
                strength_points += 1
            
            # قوة الحجم
            if volume_strength > 1.5:
                strength_points += 1
            
            # حساب القوة النهائية (1-10)
            total_strength = min(base_conditions + strength_points, 10)
            total_strength = max(total_strength, 1)
            
            return total_strength
            
        except Exception as e:
            print(f"❌ خطأ في حساب قوة الإشارة: {e}")
            return 1
    
    def generate_signal(self, row):
        """توليد إشارات تداول مع قوة متدرجة"""
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
            volume_ok = volume_ratio > self.signal_config['min_volume_ratio']
            if not volume_ok:
                condition_details.append("الحجم ضعيف")
            
            # حساب قوة الإشارة المتدرجة
            signal_strength = self.calculate_signal_strength(buy_conditions, sell_conditions, row)
            
            # اتخاذ القرار النهائي
            signal = 'HOLD'
            min_conditions = self.signal_config['min_conditions']
            min_strength = self.signal_config.get('min_signal_strength', 3)
            max_strength = self.signal_config.get('max_signal_strength', 10)
            
            # التحقق من أن قوة الإشارة ضمن النطاق المطلوب
            strength_in_range = min_strength <= signal_strength <= max_strength
            
            if (buy_conditions >= min_conditions and 
                strength_in_range and
                volume_ok):
                signal = 'BUY'
            elif (sell_conditions >= min_conditions and 
                  strength_in_range and
                  volume_ok):
                signal = 'SELL'
            
            details = " | ".join(condition_details) if condition_details else "لا توجد إشارات قوية"
            return signal, signal_strength, details
            
        except Exception as e:
            print(f"❌ خطأ في توليد الإشارة: {e}")
            return 'HOLD', 1, f"خطأ: {str(e)}"
    
    def execute_backtest(self):
        """تنفيذ الباك تستينغ مع قوة الإشارة"""
        min_strength = self.signal_config.get('min_signal_strength', 3)
        max_strength = self.signal_config.get('max_signal_strength', 10)
        
        print(f"🔄 بدء الباك تستينغ - قوة الإشارة: {min_strength} إلى {max_strength}")
        
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
            
            # التحقق من نطاق قوة الإشارة قبل فتح الصفقة
            if (signal in ['BUY', 'SELL'] and 
                open_positions < self.trade_config['max_positions'] and
                min_strength <= strength <= max_strength):
                
                self.open_position(signal, strength, row, details)
        
        print(f"✅ تم الانتهاء - {len(self.trades)} صفقة")
    
    def open_position(self, direction, signal_strength, row, details):
        """فتح صفقة جديدة"""
        try:
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
                'signal_strength': signal_strength,
                'signal_details': details
            }
            
            self.positions.append(position)
            trade_type = "ورقي" if self.paper_trading else "حقيقي"
            strength_emoji = "💪" * min(signal_strength, 5)
            print(f"📈 فتح صفقة {trade_type} {direction} #{position['id']}")
            print(f"   {strength_emoji} قوة: {signal_strength}/10 | 📊 {details}")
            
        except Exception as e:
            print(f"❌ خطأ في فتح الصفقة: {e}")

    def check_exit_conditions(self, row):
        """فحص شروط الخروج"""
        current_price = float(row['close'])
        
        for position in [p for p in self.positions if p['status'] == 'OPEN']:
            try:
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
                
                if reason:
                    position.update({
                        'status': 'CLOSED',
                        'exit_price': current_price,
                        'exit_time': row['timestamp'],
                        'pnl': float(pnl * self.trade_config['leverage']),
                        'reason': reason,
                        'duration_hours': duration_hours
                    })
                    
                    self.current_balance += position['size'] * position['pnl']
                    self.trades.append(position.copy())
                    
                    pnl_percent = position['pnl'] * 100
                    emoji = "🟢" if pnl_percent > 0 else "🔴"
                    trade_type = "ورقي" if self.paper_trading else "حقيقي"
                    print(f"{emoji} إغلاق {trade_type} {position['direction']} #{position['id']} - {reason} - {pnl_percent:+.2f}%")
                    
            except Exception as e:
                print(f"❌ خطأ في فحص شروط الخروج: {e}")

    def generate_report(self):
        """توليد تقرير الأداء مع تحليل قوة الإشارة"""
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
        
        # تحليل الصفقات حسب قوة الإشارة
        strength_analysis = {}
        min_strength = self.signal_config.get('min_signal_strength', 3)
        max_strength = self.signal_config.get('max_signal_strength', 10)
        
        for strength in range(min_strength, max_strength + 1):
            strength_trades = [t for t in self.trades if t.get('signal_strength', 0) == strength]
            if strength_trades:
                strength_win_rate = len([t for t in strength_trades if t['pnl'] > 0]) / len(strength_trades) * 100
                strength_avg_pnl = np.mean([t['pnl'] for t in strength_trades]) * 100
                strength_analysis[strength] = {
                    'count': len(strength_trades),
                    'win_rate': strength_win_rate,
                    'avg_pnl': strength_avg_pnl
                }
        
        # بناء التقرير
        report = f"""
📊 التقرير النهائي - {self.trade_config['symbol']}
{'📝 (تداول ورقي)' if self.paper_trading else '💰 (تداول حقيقي)'}

الأداء المالي:
- الرصيد الابتدائي: ${self.initial_balance:,.2f}
- الرصيد النهائي: ${self.current_balance:,.2f}
- إجمالي الربح/الخسارة: ${total_balance_change:+,.2f} ({total_pnl_percent:+.2f}%)

إحصائيات التداول:
- إجمالي الصفقات: {total_trades}
- الصفقات الرابحة: {len(winning_trades)} ({win_rate:.1f}%)
- الصفقات الخاسرة: {len(losing_trades)}
- متوسط الربح: {avg_win:+.2f}%
- متوسط الخسارة: {avg_loss:.2f}%

تحليل قوة الإشارة (النطاق {min_strength}-{max_strength}):"""
        
        for strength, analysis in strength_analysis.items():
            report += f"\n- قوة {strength}: {analysis['count']} صفقات | ربح {analysis['win_rate']:.1f}% | متوسط {analysis['avg_pnl']:+.2f}%"
        
        report += f"""

الإعدادات المستخدمة:
- وقف الخسارة: {self.trade_config['stop_loss']*100:.1f}%
- جني الأرباح: {self.trade_config['take_profit']*100:.1f}%
- حجم المركز: {self.trade_config['position_size']*100:.1f}%
- الشروط المطلوبة: {self.signal_config['min_conditions']}
- نطاق قوة الإشارة: {min_strength}-{max_strength}
        """
        
        return report

    def send_telegram_report(self):
        """إرسال التقرير عبر التلغرام"""
        try:
            bot_token = TELEGRAM_CONFIG['bot_token']
            chat_id = TELEGRAM_CONFIG['chat_id']
            
            if not bot_token or not chat_id:
                print("❌ مفاتيح التلغرام غير متوفرة")
                print("يرجى تعيين TELEGRAM_BOT_TOKEN و TELEGRAM_CHAT_ID في متغيرات البيئة")
                return
            
            print("🔍 محاولة إرسال التقرير إلى التلغرام...")
            
            bot = telebot.TeleBot(bot_token)
            report = self.generate_report()
            
            # تقسيم التقرير إذا كان طويلاً
            if len(report) > 4000:
                parts = [report[i:i+4000] for i in range(0, len(report), 4000)]
                for i, part in enumerate(parts):
                    try:
                        bot.send_message(chat_id, f"الجزء {i+1}:\n{part}")
                        print(f"✅ تم إرسال الجزء {i+1} إلى التلغرام")
                    except Exception as e:
                        print(f"❌ خطأ في إرسال الجزء {i+1}: {e}")
            else:
                try:
                    bot.send_message(chat_id, report)
                    print("✅ تم إرسال التقرير إلى التلغرام بنجاح")
                except Exception as e:
                    print(f"❌ خطأ في إرسال التقرير: {e}")
                    
        except Exception as e:
            print(f"❌ خطأ في إرسال التقرير إلى التلغرام: {e}")
            print("تفاصيل الخطأ:", str(e))

def main():
    print("🚀 بدء تشغيل البوت المتقدم مع نطاق قوة الإشارة...")
    print("=" * 60)
    
    # التحقق من وجود مفاتيح التلغرام
    if not TELEGRAM_CONFIG['bot_token'] or not TELEGRAM_CONFIG['chat_id']:
        print("⚠️  تنبيه: مفاتيح التلغرام غير متوفرة")
        print("يرجى تعيين القيم التالية في Render:")
        print("TELEGRAM_BOT_TOKEN=رقم_توكن_البوت")
        print("TELEGRAM_CHAT_ID=رقم_الدردشة")
    else:
        print("✅ تم العثور على مفاتيح التلغرام")
    
    bot = AdvancedCryptoBot(TRADE_CONFIG, INDICATOR_CONFIG, SIGNAL_CONFIG)
    bot.fetch_binance_data(days=30)
    
    print("\n📈 تشغيل الباك تستينغ...")
    bot.execute_backtest()
    
    report = bot.generate_report()
    print(report)
    
    # إرسال التقرير إلى التلغرام
    bot.send_telegram_report()
    
    print("✅ انتهى التشغيل بنجاح")

if __name__ == "__main__":
    main()
