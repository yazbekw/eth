import os
import pandas as pd
import numpy as np
import requests
from datetime import datetime, timedelta
import telebot
import warnings
warnings.filterwarnings('ignore')

# ==================== إعدادات الاستراتيجية الأساسية ====================
STRATEGY_CONFIG = {
    'symbol': 'BNBUSDT',
    'timeframe': '1h',
    'initial_balance': 1000,
    'leverage': 3,
    'stop_loss': 0.02,    # 2%
    'take_profit': 0.04,  # 4%
    'position_size': 0.1, # 10%
    
    # إعدادات المؤشرات
    'rsi_period': 14,
    'rsi_overbought': 70,
    'rsi_oversold': 30,
    'ema_fast': 9,
    'ema_slow': 21,
    'macd_fast': 12,
    'macd_slow': 26,
    'macd_signal': 9
}

class CryptoTradingBot:
    def __init__(self, config):
        self.config = config
        self.data = None
        self.positions = []
        self.trades = []
        self.current_balance = config['initial_balance']
        self.portfolio_value = []
        
        # إعدادات التلغرام من متغيرات البيئة
        self.telegram_bot_token = os.getenv('TELEGRAM_BOT_TOKEN', '')
        self.telegram_chat_id = os.getenv('TELEGRAM_CHAT_ID', '')
        
    def fetch_binance_data(self, days=90):
        """جلب البيانات الحقيقية من Binance"""
        try:
            symbol = self.config['symbol']
            interval = self.config['timeframe']
            limit = days * 24  # ساعات في 3 أشهر
            
            url = "https://api.binance.com/api/v3/klines"
            params = {
                'symbol': symbol,
                'interval': interval,
                'limit': limit
            }
            
            response = requests.get(url, params=params, timeout=10)
            data = response.json()
            
            # تحويل البيانات إلى DataFrame
            df = pd.DataFrame(data, columns=[
                'timestamp', 'open', 'high', 'low', 'close', 'volume',
                'close_time', 'quote_asset_volume', 'number_of_trades',
                'taker_buy_base_asset_volume', 'taker_buy_quote_asset_volume', 'ignore'
            ])
            
            # تحويل الأنواع
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
            for col in ['open', 'high', 'low', 'close', 'volume']:
                df[col] = pd.to_numeric(df[col], errors='coerce')
            
            df = df[['timestamp', 'open', 'high', 'low', 'close', 'volume']]
            df = df.sort_values('timestamp').reset_index(drop=True)
            
            self.data = df
            self.calculate_indicators()
            
            print(f"تم جلب {len(self.data)} شمعة من Binance للعملة {symbol}")
            
        except Exception as e:
            print(f"خطأ في جلب البيانات من Binance: {e}")
            # استخدام بيانات وهمية كبديل
            self.generate_sample_data(days)
    
    def generate_sample_data(self, days):
        """إنشاء بيانات عينة في حالة فشل جلب البيانات"""
        print("جاري إنشاء بيانات عينة...")
        
        end_date = datetime.now()
        start_date = end_date - timedelta(days=days)
        
        dates = pd.date_range(start=start_date, end=end_date, freq='1H')
        np.random.seed(42)
        
        price = 30000.0
        prices = []
        for i in range(len(dates)):
            # محاكاة تحركات واقعية أكثر
            if i % 24 == 0:  # تقلب أعلى كل يوم
                volatility = 0.005
            else:
                volatility = 0.002
                
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
        try:
            # RSI
            delta = self.data['close'].diff()
            gain = delta.where(delta > 0, 0.0)
            loss = -delta.where(delta < 0, 0.0)
            
            avg_gain = gain.rolling(window=self.config['rsi_period']).mean()
            avg_loss = loss.rolling(window=self.config['rsi_period']).mean()
            
            rs = avg_gain / avg_loss
            self.data['rsi'] = 100 - (100 / (1 + rs))
            
            # المتوسطات المتحركة
            self.data['ema_fast'] = self.data['close'].ewm(
                span=self.config['ema_fast'], 
                adjust=False
            ).mean()
            self.data['ema_slow'] = self.data['close'].ewm(
                span=self.config['ema_slow'], 
                adjust=False
            ).mean()
            
            # MACD
            ema_fast = self.data['close'].ewm(
                span=self.config['macd_fast'], 
                adjust=False
            ).mean()
            ema_slow = self.data['close'].ewm(
                span=self.config['macd_slow'], 
                adjust=False
            ).mean()
            
            self.data['macd'] = ema_fast - ema_slow
            self.data['macd_signal'] = self.data['macd'].ewm(
                span=self.config['macd_signal'], 
                adjust=False
            ).mean()
            self.data['macd_histogram'] = self.data['macd'] - self.data['macd_signal']
            
            print("تم حساب المؤشرات الفنية بنجاح")
            
        except Exception as e:
            print(f"خطأ في حساب المؤشرات: {e}")
    
    def generate_signal(self, row):
        """توليد إشارات التداول"""
        try:
            if (pd.isna(row['rsi']) or pd.isna(row['ema_slow']) or 
                pd.isna(row['macd']) or pd.isna(row['ema_fast'])):
                return 'HOLD', 0
            
            signal = 'HOLD'
            strength = 0
            
            # شروط الشراء
            buy_conditions = 0
            if row['rsi'] < self.config['rsi_oversold']:
                buy_conditions += 1
            if row['ema_fast'] > row['ema_slow']:
                buy_conditions += 1
            if row['macd'] > row['macd_signal']:
                buy_conditions += 1
            
            # شروط البيع
            sell_conditions = 0
            if row['rsi'] > self.config['rsi_overbought']:
                sell_conditions += 1
            if row['ema_fast'] < row['ema_slow']:
                sell_conditions += 1
            if row['macd'] < row['macd_signal']:
                sell_conditions += 1
            
            if buy_conditions >= 2:
                signal = 'BUY'
                strength = buy_conditions
            elif sell_conditions >= 2:
                signal = 'SELL'
                strength = sell_conditions
            
            return signal, strength
            
        except Exception as e:
            print(f"خطأ في توليد الإشارة: {e}")
            return 'HOLD', 0
    
    def execute_backtest(self):
        """تنفيذ الباك تستينغ"""
        print("بدء الباك تستينغ لمدة 3 أشهر...")
        
        min_period = max(
            self.config['ema_slow'], 
            self.config['rsi_period'], 
            self.config['macd_slow']
        )
        
        for i, row in self.data.iterrows():
            if i < min_period:
                continue
            
            signal, strength = self.generate_signal(row)
            
            # إغلاق الصفقات بناء على وقف الخسارة وجني الأرباح
            self.check_exit_conditions(row)
            
            # فتح صفقات جديدة (الحد الأقصى 3 صفقات في وقت واحد)
            open_positions = len([p for p in self.positions if p['status'] == 'OPEN'])
            if signal == 'BUY' and strength >= 2 and open_positions < 3:
                self.open_position('BUY', row)
            elif signal == 'SELL' and strength >= 2 and open_positions < 3:
                self.open_position('SELL', row)
            
            # تحديث قيمة المحفظة
            self.update_portfolio_value(row)
        
        print(f"تم الانتهاء من الباك تستينغ - {len(self.trades)} صفقة تم تنفيذها")
    
    def open_position(self, direction, row):
        """فتح صفقة جديدة"""
        try:
            position_size = self.current_balance * self.config['position_size'] * self.config['leverage']
            
            if direction == 'BUY':
                stop_loss_price = row['close'] * (1 - self.config['stop_loss'])
                take_profit_price = row['close'] * (1 + self.config['take_profit'])
            else:  # SELL
                stop_loss_price = row['close'] * (1 + self.config['stop_loss'])
                take_profit_price = row['close'] * (1 - self.config['take_profit'])
            
            position = {
                'id': len(self.positions) + 1,
                'direction': direction,
                'entry_price': float(row['close']),
                'entry_time': row['timestamp'],
                'size': float(position_size),
                'stop_loss': float(stop_loss_price),
                'take_profit': float(take_profit_price),
                'status': 'OPEN'
            }
            
            self.positions.append(position)
            print(f"📈 فتح صفقة {direction} #{position['id']} بسعر {row['close']:.2f}")
            
        except Exception as e:
            print(f"خطأ في فتح الصفقة: {e}")
    
    def check_exit_conditions(self, row):
        """فحص شروط الخروج من الصفقات"""
        current_price = float(row['close'])
        
        for position in self.positions:
            if position['status'] == 'OPEN':
                try:
                    pnl = 0.0
                    reason = ''
                    
                    if position['direction'] == 'BUY':
                        if current_price <= position['stop_loss']:
                            pnl = (current_price - position['entry_price']) / position['entry_price']
                            reason = 'STOP_LOSS'
                        elif current_price >= position['take_profit']:
                            pnl = (current_price - position['entry_price']) / position['entry_price']
                            reason = 'TAKE_PROFIT'
                    else:  # SELL
                        if current_price >= position['stop_loss']:
                            pnl = (position['entry_price'] - current_price) / position['entry_price']
                            reason = 'STOP_LOSS'
                        elif current_price <= position['take_profit']:
                            pnl = (position['entry_price'] - current_price) / position['entry_price']
                            reason = 'TAKE_PROFIT'
                    
                    if reason:
                        position['status'] = 'CLOSED'
                        position['exit_price'] = current_price
                        position['exit_time'] = row['timestamp']
                        position['pnl'] = float(pnl * self.config['leverage'])
                        position['reason'] = reason
                        
                        # تحديث الرصيد
                        self.current_balance += position['size'] * position['pnl']
                        
                        self.trades.append(position.copy())
                        
                        pnl_percent = position['pnl'] * 100
                        emoji = "🟢" if pnl_percent > 0 else "🔴"
                        print(f"{emoji} إغلاق صفقة {position['direction']} #{position['id']} - السبب: {reason} - الربح: {pnl_percent:+.2f}%")
                        
                except Exception as e:
                    print(f"خطأ في فحص شروط الخروج: {e}")
    
    def update_portfolio_value(self, row):
        """تحديث قيمة المحفظة"""
        try:
            open_positions_value = 0.0
            current_price = float(row['close'])
            
            for pos in self.positions:
                if pos['status'] == 'OPEN':
                    if pos['direction'] == 'BUY':
                        pnl_ratio = (current_price - pos['entry_price']) / pos['entry_price']
                    else:  # SELL
                        pnl_ratio = (pos['entry_price'] - current_price) / pos['entry_price']
                    
                    open_positions_value += pos['size'] * pnl_ratio * self.config['leverage']
            
            portfolio_value = self.current_balance + open_positions_value
            self.portfolio_value.append({
                'timestamp': row['timestamp'],
                'value': float(portfolio_value)
            })
            
        except Exception as e:
            print(f"خطأ في تحديث قيمة المحفظة: {e}")
    
    def generate_report(self):
        """توليد تقرير مفصل"""
        if not self.trades:
            return "⚠️ لا توجد صفقات تم تنفيذها خلال الفترة"
        
        try:
            total_trades = len(self.trades)
            winning_trades = [t for t in self.trades if t['pnl'] > 0]
            losing_trades = [t for t in self.trades if t['pnl'] <= 0]
            
            win_rate = len(winning_trades) / total_trades * 100 if total_trades > 0 else 0
            total_pnl = sum(t['pnl'] for t in self.trades)
            total_pnl_percent = (total_pnl / self.config['initial_balance']) * 100
            
            avg_win = float(np.mean([t['pnl'] for t in winning_trades])) if winning_trades else 0.0
            avg_loss = float(np.mean([t['pnl'] for t in losing_trades])) if losing_trades else 0.0
            
            # تحليل الصفقات الخاسنة
            losing_analysis = self.analyze_losing_trades(losing_trades)
            
            report = f"""
📊 **تقرير أداء البوت التداولي - {self.config['symbol']}**

**الإحصائيات العامة:**
• إجمالي الصفقات: {total_trades}
• الصفقات الرابحة: {len(winning_trades)} ({win_rate:.1f}%)
• الصفقات الخاسرة: {len(losing_trades)}
• إجمالي الربح/الخسارة: {total_pnl:+.2f}$ ({total_pnl_percent:+.2f}%)
• متوسط الربح: {avg_win*100:+.2f}%
• متوسط الخسارة: {avg_loss*100:.2f}%
• الرصيد النهائي: {self.current_balance:.2f}$

**تحليل الصفقات الخاسنة:**
{losing_analysis}

**اقتراحات التحسين:**
{self.generate_improvement_suggestions(win_rate, avg_win, avg_loss, total_trades)}
            """
            
            return report
            
        except Exception as e:
            return f"خطأ في توليد التقرير: {e}"
    
    def analyze_losing_trades(self, losing_trades):
        """تحليل الصفقات الخاسنة"""
        if not losing_trades:
            return "✅ لا توجد صفقات خاسنة"
        
        try:
            stop_loss_count = len([t for t in losing_trades if t['reason'] == 'STOP_LOSS'])
            early_exit_count = len([t for t in losing_trades if t['pnl'] > -0.02])
            
            analysis = f"""
• إجمالي الصفقات الخاسنة: {len(losing_trades)}
• الصفقات التي أغلقت بوقف الخسارة: {stop_loss_count}
• الصفقات التي أغلقت بخسارة طفيفة: {early_exit_count}
• متوسط مدة الصفقات الخاسنة: {self.calculate_avg_trade_duration(losing_trades)}
            
**الأنماط الملاحظة:**
"""
            
            if stop_loss_count / len(losing_trades) > 0.7:
                analysis += "• معظم الخسائر بسبب وقف الخسارة - قد تحتاج إلى تعديل إعدادات الوقف\n"
            
            if early_exit_count > len(losing_trades) * 0.5:
                analysis += "• العديد من الصفقات أغلقت بخسارة طفيفة - قد تحتاج إلى زيادة وقف الخسارة قليلاً\n"
            
            return analysis
            
        except Exception as e:
            return f"خطأ في تحليل الصفقات الخاسنة: {e}"
    
    def calculate_avg_trade_duration(self, trades):
        """حساب متوسط مدة الصفقات"""
        if not trades:
            return "0"
        
        try:
            durations = []
            for trade in trades:
                if 'exit_time' in trade and 'entry_time' in trade:
                    duration = (trade['exit_time'] - trade['entry_time']).total_seconds() / 3600
                    durations.append(duration)
            
            return f"{float(np.mean(durations)):.1f} ساعة" if durations else "غير متوفر"
            
        except Exception as e:
            return f"خطأ: {e}"
    
    def generate_improvement_suggestions(self, win_rate, avg_win, avg_loss, total_trades):
        """توليد اقتراحات للتحسين"""
        suggestions = []
        
        if win_rate < 50 and total_trades > 10:
            suggestions.append("• زيادة فترة RSI إلى 21 للتقليل من الإشارات الكاذبة")
            suggestions.append("• إضافة شرط تقاطع المتوسطات مع حجم التداول")
        
        if avg_win < abs(avg_loss) and avg_loss != 0 and total_trades > 10:
            suggestions.append("• زيادة نسبة جني الأرباح إلى 5-6% لتحسين نسبة الربح/الخسارة")
            suggestions.append("• استخدام وقف خسارة متحرك بعد تحقيق جزء من الأرباح")
        
        if total_trades > 50:
            suggestions.append("• تقليل حجم المركز إلى 5-7% لإدارة أفضل للمخاطر")
            suggestions.append("• إضافة تصفية بالاتجاه العام باستخدام المتوسط 50")
        
        if not suggestions:
            suggestions.append("• الاستراتيجية تعمل بشكل جيد، الحفاظ على الإعدادات الحالية")
        
        return "\n".join(suggestions)
    
    def send_telegram_report(self):
        """إرسال التقرير عبر التلغرام"""
        try:
            if not self.telegram_bot_token or not self.telegram_chat_id:
                print("⚠️ مفاتيح التلغرام غير متوفرة - تخطي الإرسال")
                return
            
            bot = telebot.TeleBot(self.telegram_bot_token)
            report = self.generate_report()
            
            # إضافة معلومات الإعدادات
            settings_info = f"""

**إعدادات الاستراتيجية:**
• العملة: {self.config['symbol']}
• الرافعة: {self.config['leverage']}x
• وقف الخسارة: {self.config['stop_loss']*100}%
• جني الأرباح: {self.config['take_profit']*100}%
• حجم المركز: {self.config['position_size']*100}%
• الفترة: 3 أشهر
            """
            
            full_message = report + settings_info
            bot.send_message(self.telegram_chat_id, full_message, parse_mode='Markdown')
            print("✅ تم إرسال التقرير إلى التلغرام")
            
        except Exception as e:
            print(f"❌ خطأ في إرسال التقرير: {e}")

# ==================== التنفيذ الرئيسي ====================

def main():
    print("🚀 بدء تشغيل بوت التداول...")
    
    # إنشاء البوت
    bot = CryptoTradingBot(STRATEGY_CONFIG)
    
    # جلب البيانات الحقيقية من Binance
    bot.fetch_binance_data(days=90)
    
    # تشغيل الباك تستينغ
    bot.execute_backtest()
    
    # توليد وعرض التقرير
    report = bot.generate_report()
    print(report)
    
    # إرسال التقرير عبر التلغرام
    bot.send_telegram_report()
    
    print("✅ تم الانتهاء من التشغيل بنجاح")

if __name__ == "__main__":
    main()
