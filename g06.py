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
from dotenv import load_dotenv
import matplotlib.pyplot as plt
import io
from collections import Counter
from binance.client import Client
from binance.enums import *
import schedule
import time
import threading

warnings.filterwarnings('ignore')
load_dotenv()

# ====================== إعدادات التفعيل / الإلغاء ======================
ENABLE_TRAILING_STOP = True
ENABLE_DYNAMIC_POSITION_SIZING = False
ENABLE_MARKET_REGIME_FILTER = False
ENABLE_ATR_SL_TP = False
ENABLE_SUPPORT_RESISTANCE_FILTER = True
ENABLE_TIME_FILTER = True
ENABLE_LOGGING = True
ENABLE_DETAILED_REPORT = True
ENABLE_DAILY_REPORT = True

# ====================== دالة مساعدة ======================
def interval_to_hours(interval):
    mapping = {
        '1m': 1/60, '3m': 3/60, '5m': 5/60, '15m': 15/60,
        '30m': 30/60, '1h': 1, '2h': 2, '4h': 4, '6h': 6,
        '8h': 8, '12h': 12, '1d': 24, '3d': 72, '1w': 168, '1M': 720
    }
    return mapping.get(interval, 4)

# ====================== الإعدادات الأساسية ======================
TRADE_CONFIG = {
    'symbol': 'BNBUSDT',
    'timeframe': '4h',
    'initial_balance': 200,
    'leverage': 1,
    'base_stop_loss': 0.025,
    'base_take_profit': 0.065,
    'base_position_size': 0.25,
    'max_positions': 4,
    'paper_trading': True,
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
    'max_signal_strength': 6,
    'require_trend_confirmation': True,
    'min_volume_ratio': 1.0
}

# إعدادات بينانس
BINANCE_CONFIG = {
    'api_key': os.getenv('BINANCE_API_KEY', ''),
    'api_secret': os.getenv('BINANCE_API_SECRET', ''),
    'testnet': True  # استخدام testnet
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

# ====================== الكلاس الرئيسي المطور مع بينانس ======================
class AdvancedCryptoBot:
    def __init__(self, trade_config, indicator_config, signal_config, binance_config):
        self.trade_config = trade_config
        self.indicator_config = indicator_config
        self.signal_config = signal_config
        self.binance_config = binance_config
        self.data = None
        self.positions = []
        self.trades = []
        self.current_balance = trade_config['initial_balance']
        self.initial_balance = trade_config['initial_balance']
        self.paper_trading = trade_config.get('paper_trading', True)
        self.analysis_results = {}
        self.signal_strength_results = {}
        self.balance_history = [trade_config['initial_balance']]
        self.daily_trades = []
        self.last_daily_report = datetime.now().date()
        
        # تهيئة عميل بينانس
        self.binance_client = None
        if not self.paper_trading and binance_config['api_key'] and binance_config['api_secret']:
            try:
                self.binance_client = Client(
                    api_key=binance_config['api_key'],
                    api_secret=binance_config['api_secret'],
                    testnet=binance_config['testnet']
                )
                if ENABLE_LOGGING:
                    logger.info("✅ تم الاتصال بمنصة بينانس بنجاح (Testnet)")
            except Exception as e:
                if ENABLE_LOGGING:
                    logger.error(f"❌ فشل الاتصال ببينانس: {e}")

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, max=10))
    def fetch_binance_data(self, days=60):
        """جلب البيانات من Binance"""
        try:
            symbol = self.trade_config['symbol']
            interval = self.trade_config['timeframe']
            limit = 1000
            all_data = []
            end_time = int(datetime.now().timestamp() * 1000)
            interval_h = interval_to_hours(interval)
            required_candles = int(days * 24 / interval_h) + 100

            if ENABLE_LOGGING:
                logger.info(f"📊 جلب {required_candles} شمعة من {symbol} ({interval})")

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
                logger.info(f"✅ تم جلب {len(self.data)} شمعة بنجاح")

        except Exception as e:
            if ENABLE_LOGGING:
                logger.error(f"❌ خطأ في جلب البيانات: {e}")
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
                logger.info(f"📈 تم إنشاء {len(self.data)} شمعة عينة")
        except Exception as e:
            if ENABLE_LOGGING:
                logger.error(f"❌ خطأ في إنشاء بيانات عينة: {e}")

    def calculate_indicators(self):
        """حساب المؤشرات الفنية"""
        # ... (نفس الدوال السابقة)
        pass

    def get_market_regime(self, row):
        """تحديد نظام السوق"""
        # ... (نفس الدوال السابقة)
        pass

    def calculate_signal_strength(self, buy_conditions, sell_conditions, row):
        """حساب قوة الإشارة المتدرجة من 1-10"""
        # ... (نفس الدوال السابقة)
        pass

    def generate_signal(self, row):
        """توليد إشارات تداول"""
        # ... (نفس الدوال السابقة)
        pass

    def send_trade_notification(self, position, action, pnl_dollar=0, pnl_percent=0):
        """إرسال إشعار تفصيلي بالصفقة عبر التلغرام"""
        if not TELEGRAM_CONFIG['bot_token'] or not TELEGRAM_CONFIG['chat_id']:
            return
        
        try:
            bot = telebot.TeleBot(TELEGRAM_CONFIG['bot_token'])
            
            if action == "OPEN":
                emoji = "🟢" if position['direction'] == 'BUY' else "🔴"
                message = f"""
{emoji} **تم فتح صفقة جديدة** {emoji}

📊 **التفاصيل:**
• الزوج: {self.trade_config['symbol']}
• الاتجاه: {position['direction']}
• السعر: ${position['entry_price']:.2f}
• الحجم: ${position['size']:.2f}
• قوة الإشارة: {position['signal_strength']:.1f}/10

🛡️ **إدارة المخاطر:**
• وقف الخسارة: ${position['stop_loss']:.2f}
• جني الأرباح: ${position['take_profit']:.2f}

📈 **المؤشرات:**
• RSI: {position['entry_rsi']:.1f}
• نسبة الحجم: {position['volume_ratio']:.2f}

⏰ **الوقت:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
"""
            else:  # CLOSE
                emoji = "💰" if pnl_dollar > 0 else "💸"
                message = f"""
{emoji} **تم إغلاق الصفقة** {emoji}

📊 **النتيجة:**
• الزوج: {self.trade_config['symbol']}
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
            
            bot.send_message(TELEGRAM_CONFIG['chat_id'], message, parse_mode='Markdown')
            
        except Exception as e:
            if ENABLE_LOGGING:
                logger.error(f"❌ خطأ في إرسال إشعار الصفقة: {e}")

    def open_position(self, direction, signal_strength, row, details):
        """فتح صفقة جديدة"""
        try:
            # حساب حجم المركز بدقة
            base_size = self.trade_config['base_position_size']
            if ENABLE_DYNAMIC_POSITION_SIZING:
                size_factor = 0.5 + (signal_strength / 20)
                position_value = self.current_balance * base_size * size_factor
            else:
                position_value = self.current_balance * base_size

            entry = row['close']
            
            # حساب وقف الخسارة وجني الأرباح
            if direction == 'BUY':
                sl = entry * (1 - self.trade_config['base_stop_loss'])
                tp = entry * (1 + self.trade_config['base_take_profit'])
            else:
                sl = entry * (1 + self.trade_config['base_stop_loss'])
                tp = entry * (1 - self.trade_config['base_take_profit'])

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
                'signal_strength': signal_strength,
                'signal_details': details,
                'entry_rsi': float(row['rsi']),
                'entry_macd': float(row['macd']),
                'volume_ratio': float(row['volume'] / row['volume_ma']) if row['volume_ma'] > 0 else 1.0,
                'trailing_stop': float(sl)
            }
            
            self.positions.append(position)
            self.daily_trades.append({
                **position,
                'action': 'OPEN',
                'timestamp': datetime.now()
            })
            
            # إرسال إشعار فتح الصفقة
            self.send_trade_notification(position, "OPEN")
            
            if ENABLE_LOGGING:
                strength_emoji = "💪" * min(int(signal_strength / 2), 5)
                size_percent = (position_value / self.current_balance) * 100
                logger.info(f"🟢 فتح {direction} #{position['id']} | قوة: {signal_strength:.1f}/10 {strength_emoji}")
                logger.info(f"📦 الحجم: ${position_value:.2f} ({size_percent:.1f}%) | {details}")
                
        except Exception as e:
            if ENABLE_LOGGING:
                logger.error(f"❌ خطأ فتح الصفقة: {e}")

    def check_exit_conditions(self, row):
        """فحص شروط الخروج"""
        current_price = row['close']
        for pos in [p for p in self.positions if p['status'] == 'OPEN']:
            pnl_percent = 0.0
            reason = ''
            duration = (row['timestamp'] - pos['entry_time']).total_seconds() / 3600

            # حساب الربح/الخسارة بالنسبة المئوية
            if pos['direction'] == 'BUY':
                pnl_percent = (current_price - pos['entry_price']) / pos['entry_price']
            else:
                pnl_percent = (pos['entry_price'] - current_price) / pos['entry_price']

            # التداول المتعقب
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
                # حساب الربح/الخسارة بالدولار بدقة
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
                
                # تحديث الرصيد بدقة
                self.current_balance += pnl_dollar
                self.balance_history.append(self.current_balance)
                self.trades.append(pos.copy())
                self.daily_trades.append({
                    **pos,
                    'action': 'CLOSE',
                    'timestamp': datetime.now()
                })
                
                # إرسال إشعار إغلاق الصفقة
                self.send_trade_notification(pos, "CLOSE", pnl_dollar, pnl_percent*100)
                
                self.positions.remove(pos)
                
                if ENABLE_LOGGING:
                    emoji = "💰" if pnl_dollar > 0 else "💸"
                    logger.info(f"{emoji} إغلاق {pos['direction']} #{pos['id']} | {reason} | ${pnl_dollar:+.2f} ({pnl_percent*100:+.2f}%)")

    def generate_daily_report(self):
        """توليد تقرير يومي مفصل"""
        try:
            today = datetime.now().date()
            today_trades = [t for t in self.daily_trades if t['timestamp'].date() == today]
            
            if not today_trades:
                return "📊 **تقرير الأداء اليومي**\n\n⚠️ لم تتم أي صفقات اليوم"
            
            open_trades = [t for t in today_trades if t.get('status') == 'OPEN']
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
• التغير اليومي: {(total_pnl/self.initial_balance)*100:+.2f}%

🔄 **الصفقات المفتوحة:** {len(open_trades)}
⚡ **الصفقات المغلقة:** {len(closed_trades)}

⏰ **آخر تحديث:** {datetime.now().strftime('%H:%M:%S')}
"""
            
            # إضافة تفاصيل الصفقات المغلقة
            if closed_trades:
                report += "\n🔍 **تفاصيل الصفقات المغلقة:**\n"
                for trade in closed_trades[-5:]:  # آخر 5 صفقات
                    emoji = "🟢" if trade.get('pnl_dollar', 0) > 0 else "🔴"
                    report += f"• {trade['direction']} | ${trade.get('pnl_dollar', 0):+.2f} {emoji}\n"
            
            return report
            
        except Exception as e:
            if ENABLE_LOGGING:
                logger.error(f"❌ خطأ في توليد التقرير اليومي: {e}")
            return f"❌ خطأ في توليد التقرير اليومي: {e}"

    def send_daily_report_telegram(self):
        """إرسال التقرير اليومي عبر التلغرام"""
        if not TELEGRAM_CONFIG['bot_token'] or not TELEGRAM_CONFIG['chat_id']:
            return
        
        try:
            report = self.generate_daily_report()
            bot = telebot.TeleBot(TELEGRAM_CONFIG['bot_token'])
            bot.send_message(TELEGRAM_CONFIG['chat_id'], report, parse_mode='Markdown')
            
            if ENABLE_LOGGING:
                logger.info("✅ تم إرسال التقرير اليومي إلى التلغرام")
                
        except Exception as e:
            if ENABLE_LOGGING:
                logger.error(f"❌ خطأ في إرسال التقرير اليومي: {e}")

    def run_live_signal_check(self):
        """تشغيل فحص الإشارات في الوقت الحقيقي"""
        try:
            self.fetch_binance_data(days=7)  # جلب بيانات 7 أيام فقط للسرعة
            
            if self.data is None or len(self.data) == 0:
                if ENABLE_LOGGING:
                    logger.error("❌ لا توجد بيانات للتحليل")
                return
            
            # الحصول على آخر شمعة
            last_row = self.data.iloc[-1]
            signal, strength, details = self.generate_signal(last_row)
            
            # فحص شروط الخروج للصفقات المفتوحة
            self.check_exit_conditions(last_row)
            
            # فتح صفقات جديدة إذا كانت هناك إشارة
            open_positions = len([p for p in self.positions if p['status'] == 'OPEN'])
            if signal in ['BUY', 'SELL'] and open_positions < self.trade_config['max_positions']:
                self.open_position(signal, strength, last_row, details)
                if ENABLE_LOGGING:
                    logger.info(f"🎯 إشارة {signal} جديدة | قوة: {strength:.1f}/10")
            
            # إرسال التقرير اليومي في نهاية اليوم
            current_date = datetime.now().date()
            if ENABLE_DAILY_REPORT and current_date != self.last_daily_report:
                if datetime.now().hour == 23:  # الساعة 11 مساءً
                    self.send_daily_report_telegram()
                    self.last_daily_report = current_date
                    self.daily_trades = []  # تفريغ الصفقات اليومية
                    
        except Exception as e:
            if ENABLE_LOGGING:
                logger.error(f"❌ خطأ في فحص الإشارات: {e}")

# ====================== تشغيل البوت في الوقت الحقيقي ======================
def run_live_bot():
    """تشغيل البوت في الوضع الحي"""
    if ENABLE_LOGGING:
        logger.info("🚀 بدء تشغيل البوت المطور في الوضع الحي")
    
    bot = AdvancedCryptoBot(TRADE_CONFIG, INDICATOR_CONFIG, SIGNAL_CONFIG, BINANCE_CONFIG)
    
    # جدولة المهام
    schedule.every(5).minutes.do(bot.run_live_signal_check)  # فحص كل 5 دقائق
    schedule.every().day.at("23:00").do(bot.send_daily_report_telegram)  # تقرير يومي الساعة 11 مساءً
    
    # التشغيل الفوري الأول
    bot.run_live_signal_check()
    
    if ENABLE_LOGGING:
        logger.info("✅ تم بدء التشغيل وجدولة المهام")
    
    # حلقة التشغيل الرئيسية
    while True:
        try:
            schedule.run_pending()
            time.sleep(1)
        except KeyboardInterrupt:
            if ENABLE_LOGGING:
                logger.info("⏹️ إيقاف البوت بواسطة المستخدم")
            break
        except Exception as e:
            if ENABLE_LOGGING:
                logger.error(f"❌ خطأ في حلقة التشغيل: {e}")
            time.sleep(60)  # انتظار دقيقة قبل إعادة المحاولة

if __name__ == "__main__":
    run_live_bot()
