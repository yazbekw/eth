import os
import pandas as pd
import numpy as np
import requests
from datetime import datetime, timedelta
import telebot
import warnings
warnings.filterwarnings('ignore')

# ==================== إعدادات العقود الآجلة ====================

TRADE_CONFIG = {
    'symbol': 'BNBUSDT',
    'timeframe': '1h',
    'initial_balance': 1000,
    'leverage': 10,           # زيادة الرافعة للعقود الآجلة
    'stop_loss': 0.02,        # 2% - تقليل وقف الخسارة
    'take_profit': 0.04,      # 4% - تقليل جني الأرباح
    'position_size': 0.05,    # 5% - تقليل حجم المركز
    'max_positions': 3,
    'trading_type': 'futures', # تداول عقود آجلة
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

TELEGRAM_CONFIG = {
    'bot_token': os.getenv('TELEGRAM_BOT_TOKEN', ''),
    'chat_id': os.getenv('TELEGRAM_CHAT_ID', '')
}

class FuturesTradingBot:
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
        self.leverage = trade_config['leverage']
        
    def fetch_binance_futures_data(self, days=90):
        """جلب بيانات العقود الآجلة من Binance"""
        try:
            symbol = self.trade_config['symbol']
            interval = self.trade_config['timeframe']
            
            print(f"📊 جلب بيانات العقود الآجلة لـ {symbol} لمدة {days} يوم...")
            
            url = "https://fapi.binance.com/fapi/v1/klines"
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
            print(f"✅ تم جلب {len(self.data)} شمعة للعقود الآجلة {symbol}")
            
        except Exception as e:
            print(f"❌ خطأ في جلب البيانات: {e}")
            self.generate_sample_futures_data(days)
    
    def generate_sample_futures_data(self, days):
        """بيانات عينة للعقود الآجلة"""
        print("📊 استخدام بيانات عينة للعقود الآجلة...")
        
        end_date = datetime.now()
        start_date = end_date - timedelta(days=days)
        dates = pd.date_range(start=start_date, end=end_date, freq='1H')
        
        np.random.seed(42)
        price = 300.0
        prices = []
        
        for i in range(len(dates)):
            volatility = 0.008 if i % 24 == 0 else 0.004  # تقلب أعلى للعقود
            change = np.random.normal(0, volatility)
            price = price * (1 + change)
            prices.append(price)
        
        self.data = pd.DataFrame({
            'timestamp': dates,
            'open': prices,
            'high': [p * (1 + abs(np.random.normal(0, 0.003))) for p in prices],
            'low': [p * (1 - abs(np.random.normal(0, 0.003))) for p in prices],
            'close': prices,
            'volume': [abs(np.random.normal(5000, 1000)) for _ in prices]  # حجم أعلى
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
        
        print("✅ تم حساب المؤشرات الفنية للعقود الآجلة")
    
    def calculate_liquidation_price(self, position):
        """حساب سعر التصفية للعقود الآجلة"""
        entry_price = position['entry_price']
        leverage = self.leverage
        
        if position['direction'] == 'LONG':
            # للشراء: سعر التصفية = سعر الدخول * (1 - 1/الرافعة)
            liquidation_price = entry_price * (1 - 1/leverage)
        else:  # SHORT
            # للبيع: سعر التصفية = سعر الدخول * (1 + 1/الرافعة)
            liquidation_price = entry_price * (1 + 1/leverage)
        
        return liquidation_price
    
    def generate_signal(self, row):
        """توليد إشارات التداول للعقود الآجلة"""
        if any(pd.isna(row[key]) for key in ['rsi', 'ema_slow', 'macd', 'ema_trend', 'volume_ma']):
            return 'HOLD', 0
        
        long_conditions = 0
        short_conditions = 0
        
        # شروط الشراء (LONG)
        if row['rsi'] < self.indicator_config['rsi_oversold']:
            long_conditions += 1
        if row['ema_fast'] > row['ema_slow']:
            long_conditions += 1
        if row['macd'] > row['macd_signal']:
            long_conditions += 1
        
        # شروط البيع (SHORT)
        if row['rsi'] > self.indicator_config['rsi_overbought']:
            short_conditions += 1
        if row['ema_fast'] < row['ema_slow']:
            short_conditions += 1
        if row['macd'] < row['macd_signal']:
            short_conditions += 1
        
        # تصفية الاتجاه
        if self.signal_config['use_trend_filter']:
            if row['close'] > row['ema_trend']:
                long_conditions += 0.5
            else:
                short_conditions += 0.5
        
        # تصفية الحجم
        if self.signal_config['use_volume_filter']:
            volume_confirm = row['volume'] > row['volume_ma'] * self.signal_config['min_volume_ratio']
            if volume_confirm:
                long_conditions += 0.5
                short_conditions += 0.5
        
        signal = 'HOLD'
        strength = 0
        
        min_conditions = self.signal_config['min_conditions']
        
        if long_conditions >= min_conditions:
            signal = 'LONG'
            strength = long_conditions
        elif short_conditions >= min_conditions:
            signal = 'SHORT'
            strength = short_conditions
        
        return signal, strength
    
    def execute_backtest(self):
        """تنفيذ الباك تستينغ للعقود الآجلة"""
        print("🔄 بدء الباك تستينغ للعقود الآجلة...")
        print(f"💰 الرافعة: {self.leverage}x")
        
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
            
            # فتح صفقات جديدة
            open_positions = len([p for p in self.positions if p['status'] == 'OPEN'])
            if (signal in ['LONG', 'SHORT'] and 
                open_positions < self.trade_config['max_positions']):
                self.open_position(signal, row)
        
        print(f"✅ تم الانتهاء - {len(self.trades)} صفقة آجلة")
    
    def open_position(self, direction, row):
        """فتح صفقة آجلة جديدة"""
        position_size = self.current_balance * self.trade_config['position_size'] * self.leverage
        
        if direction == 'LONG':
            stop_loss = row['close'] * (1 - self.trade_config['stop_loss'])
            take_profit = row['close'] * (1 + self.trade_config['take_profit'])
        else:  # SHORT
            stop_loss = row['close'] * (1 + self.trade_config['stop_loss'])
            take_profit = row['close'] * (1 - self.trade_config['take_profit'])
        
        liquidation_price = self.calculate_liquidation_price({
            'direction': direction,
            'entry_price': row['close']
        })
        
        position = {
            'id': len(self.positions) + 1,
            'direction': direction,
            'entry_price': float(row['close']),
            'entry_time': row['timestamp'],
            'size': float(position_size),
            'stop_loss': float(stop_loss),
            'take_profit': float(take_profit),
            'liquidation_price': float(liquidation_price),
            'leverage': self.leverage,
            'status': 'OPEN',
            'type': 'FUTURES'
        }
        
        self.positions.append(position)
        trade_type = "ورقي" if self.paper_trading else "حقيقي"
        print(f"📈 فتح صفقة {direction} {trade_type} #{position['id']}")
        print(f"   🎯 السعر: {row['close']:.2f} | الرافعة: {self.leverage}x")
        print(f"   ⚠️  التصفية: {liquidation_price:.2f}")
    
    def check_exit_conditions(self, row):
        """فحص شروط الخروج للعقود الآجلة"""
        current_price = float(row['close'])
        
        for position in self.positions:
            if position['status'] == 'OPEN':
                pnl = 0.0
                reason = ''
                
                # التحقق من التصفية أولاً
                if (position['direction'] == 'LONG' and current_price <= position['liquidation_price']) or \
                   (position['direction'] == 'SHORT' and current_price >= position['liquidation_price']):
                    pnl = -1.0  # خسارة كاملة
                    reason = 'LIQUIDATION'
                
                # ثم التحقق من وقف الخسارة وجني الأرباح
                elif position['direction'] == 'LONG':
                    if current_price <= position['stop_loss']:
                        pnl = (current_price - position['entry_price']) / position['entry_price']
                        reason = 'STOP_LOSS'
                    elif current_price >= position['take_profit']:
                        pnl = (current_price - position['entry_price']) / position['entry_price']
                        reason = 'TAKE_PROFIT'
                else:  # SHORT
                    if current_price >= position['stop_loss']:
                        pnl = (position['entry_price'] - current_price) / position['entry_price']
                        reason = 'STOP_LOSS'
                    elif current_price <= position['take_profit']:
                        pnl = (position['entry_price'] - current_price) / position['entry_price']
                        reason = 'TAKE_PROFIT'
                
                if reason:
                    loss_reason = self.analyze_loss_reason(position, row) if pnl < 0 else ""
                    
                    # حساب الربح/الخسارة مع الرافعة
                    final_pnl = pnl * self.leverage
                    
                    position.update({
                        'status': 'CLOSED',
                        'exit_price': current_price,
                        'exit_time': row['timestamp'],
                        'pnl': float(final_pnl),
                        'reason': reason,
                        'loss_reason': loss_reason
                    })
                    
                    self.current_balance += position['size'] * final_pnl
                    self.trades.append(position.copy())
                    
                    pnl_percent = final_pnl * 100
                    emoji = "🟢" if pnl_percent > 0 else "🔴"
                    trade_type = "ورقي" if self.paper_trading else "حقيقي"
                    reason_text = f"{reason} {loss_reason}" if loss_reason else reason
                    
                    print(f"{emoji} إغلاق صفقة {position['direction']} #{position['id']}")
                    print(f"   📊 السبب: {reason_text} | الربح: {pnl_percent:+.2f}%")
    
    def analyze_loss_reason(self, position, row):
        """تحليل أسباب الخسارة للعقود الآجلة"""
        reasons = []
        
        if position['direction'] == 'LONG':
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
        else:  # SHORT
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
        """توليد تقرير مفصل للعقود الآجلة"""
        if not self.trades:
            return "⚠️ لا توجد صفقات آجلة"
        
        total_trades = len(self.trades)
        winning_trades = [t for t in self.trades if t['pnl'] > 0]
        losing_trades = [t for t in self.trades if t['pnl'] <= 0]
        
        long_trades = [t for t in self.trades if t['direction'] == 'LONG']
        short_trades = [t for t in self.trades if t['direction'] == 'SHORT']
        
        win_rate = len(winning_trades) / total_trades * 100
        
        total_balance_change = self.current_balance - self.initial_balance
        total_pnl_percent = (total_balance_change / self.initial_balance) * 100
        
        avg_win = np.mean([t['pnl'] for t in winning_trades]) if winning_trades else 0
        avg_loss = np.mean([t['pnl'] for t in losing_trades]) if losing_trades else 0
        
        profit_loss_ratio = "N/A"
        if avg_loss != 0:
            profit_loss_ratio = f"{abs(avg_win/avg_loss):.2f}"
        
        # تحليل الصفقات المنصفة
        liquidated_trades = [t for t in self.trades if t['reason'] == 'LIQUIDATION']
        
        # تحليل الأداء حسب نوع الصفقة
        long_win_rate = len([t for t in long_trades if t['pnl'] > 0]) / len(long_trades) * 100 if long_trades else 0
        short_win_rate = len([t for t in short_trades if t['pnl'] > 0]) / len(short_trades) * 100 if short_trades else 0
        
        loss_analysis = self.analyze_futures_loss_patterns(losing_trades)
        
        report = f"""
📊 تقرير العقود الآجلة - {self.trade_config['symbol']}
🎯 الرافعة: {self.leverage}x | 📝 تداول ورقي

الأداء الحقيقي:
- الرصيد الابتدائي: {self.initial_balance:,.2f}$
- الرصيد النهائي: {self.current_balance:,.2f}$
- إجمالي الربح/الخسارة: {total_balance_change:+,.2f}$ ({total_pnl_percent:+.2f}%)

إحصائيات التداول:
- إجمالي الصفقات: {total_trades}
- صفقات شراء (LONG): {len(long_trades)} (ربح: {long_win_rate:.1f}%)
- صفقات بيع (SHORT): {len(short_trades)} (ربح: {short_win_rate:.1f}%)
- الصفقات الرابحة: {len(winning_trades)} ({win_rate:.1f}%)
- الصفقات الخاسرة: {len(losing_trades)}
- الصفقات المُصفاة: {len(liquidated_trades)}
- متوسط الربح: {avg_win*100:+.2f}%
- متوسط الخسارة: {avg_loss*100:.2f}%
- نسبة الربح/الخسارة: {profit_loss_ratio}

تحليل مفصل للخسائر:
{loss_analysis}

الإعدادات المستخدمة:
- الرافعة: {self.leverage}x
- وقف الخسارة: {self.trade_config['stop_loss']*100}%
- جني الأرباح: {self.trade_config['take_profit']*100}%
- حجم المركز: {self.trade_config['position_size']*100}%
- نوع التداول: عقود آجلة
        """
        
        return report
    
    def analyze_futures_loss_patterns(self, losing_trades):
        """تحليل أنماط الخسارة للعقود الآجلة"""
        if not losing_trades:
            return "✅ لا توجد صفقات خاسنة"
        
        stop_loss_count = len([t for t in losing_trades if t['reason'] == 'STOP_LOSS'])
        liquidation_count = len([t for t in losing_trades if t['reason'] == 'LIQUIDATION'])
        take_profit_count = len([t for t in losing_trades if t['reason'] == 'TAKE_PROFIT'])
        
        long_losses = [t for t in losing_trades if t['direction'] == 'LONG']
        short_losses = [t for t in losing_trades if t['direction'] == 'SHORT']
        
        common_reasons = {}
        for trade in losing_trades:
            if 'loss_reason' in trade and trade['loss_reason']:
                reasons = trade['loss_reason'].split(", ")
                for reason in reasons:
                    common_reasons[reason] = common_reasons.get(reason, 0) + 1
        
        analysis = f"""
- الصفقات الخاسنة: {len(losing_trades)}
- خسائر شراء (LONG): {len(long_losses)}
- خسائر بيع (SHORT): {len(short_losses)}
- بسبب وقف الخسارة: {stop_loss_count}
- بسبب التصفية: {liquidation_count}
- متوسط مدة الصفقات الخاسنة: {self.calculate_avg_trade_duration(losing_trades)}

الأسباب الشائعة للخسارة:
"""
        
        for reason, count in sorted(common_reasons.items(), key=lambda x: x[1], reverse=True):
            percentage = (count / len(losing_trades)) * 100
            analysis += f"- {reason}: {count} مرة ({percentage:.1f}%)\n"
        
        # توصيات خاصة بالعقود الآجلة
        if liquidation_count > 0:
            analysis += f"\n⚠️  تحذير: {liquidation_count} صفقة تم تصفيتها!"
            analysis += "\nتوصية: تقليل الرافعة أو زيادة وقف الخسارة\n"
        
        if stop_loss_count / len(losing_trades) > 0.6:
            analysis += "\nتوصية: معظم الخسائر بسبب وقف الخسارة -考虑 تعديل الإعدادات\n"
        
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
        """إرسال التقرير عبر التلغرام"""
        try:
            bot_token = TELEGRAM_CONFIG['bot_token']
            chat_id = TELEGRAM_CONFIG['chat_id']
            
            if not bot_token or not chat_id:
                print("❌ مفاتيح التلغرام غير متوفرة")
                return
            
            bot = telebot.TeleBot(bot_token)
            report = self.generate_report()
            
            # تقسيم التقرير إذا كان طويلاً
            if len(report) > 4000:
                parts = [report[i:i+4000] for i in range(0, len(report), 4000)]
                for i, part in enumerate(parts):
                    bot.send_message(chat_id, f"الجزء {i+1}:\n{part}")
            else:
                bot.send_message(chat_id, report)
            
            print("✅ تم إرسال تقرير العقود الآجلة إلى التلغرام")
                
        except Exception as e:
            print(f"❌ خطأ في إرسال التقرير: {e}")

def main():
    print("🚀 بدء تشغيل بوت العقود الآجلة...")
    
    bot = FuturesTradingBot(TRADE_CONFIG, INDICATOR_CONFIG, SIGNAL_CONFIG)
    bot.fetch_binance_futures_data(days=90)
    bot.execute_backtest()
    
    report = bot.generate_report()
    print(report)
    
    bot.send_telegram_report()
    print("✅ انتهى تشغيل بوت العقود الآجلة")

if __name__ == "__main__":
    main()
