
import os
import pandas as pd
import numpy as np
from binance.client import Client
from binance.enums import *
import time
from datetime import datetime, timedelta
import requests
import logging
import warnings
warnings.filterwarnings('ignore')
from dotenv import load_dotenv
from flask import Flask
import threading
import json

# تحميل متغيرات البيئة
load_dotenv()

# إنشاء تطبيق Flask
app = Flask(__name__)

@app.route('/')
def health_check():
    return {'status': 'healthy', 'service': 'crypto-trading-bot', 'timestamp': datetime.now().isoformat()}

@app.route('/status')
def status():
    return {'status': 'running', 'bot': 'Crypto Trading Bot', 'time': datetime.now().isoformat()}

@app.route('/recent_trades')
def recent_trades():
    try:
        bot = Crypto_Trading_Bot()
        report = bot.generate_12h_trading_report()
        return report
    except Exception as e:
        return {'error': str(e)}

@app.route('/daily_report')
def daily_report():
    try:
        bot = Crypto_Trading_Bot()
        report = bot.generate_daily_performance_report()
        return report
    except Exception as e:
        return {'error': str(e)}

def run_flask_app():
    port = int(os.environ.get('PORT', 10000))
    app.run(host='0.0.0.0', port=port, debug=False)

# إعداد logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('bot_activity.log', encoding='utf-8'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class TelegramNotifier:
    def __init__(self, token, chat_id):
        self.token = token
        self.chat_id = chat_id
        self.base_url = f"https://api.telegram.org/bot{token}"
    
    def send_message(self, message):
        try:
            logger.info(f"محاولة إرسال رسالة إلى Telegram: {message}")
            url = f"{self.base_url}/sendMessage"
            payload = {
                'chat_id': self.chat_id,
                'text': message,
                'parse_mode': 'HTML'
            }
            response = requests.post(url, data=payload, timeout=10)
            if response.status_code != 200:
                error_msg = f"فشل إرسال رسالة Telegram: {response.text}"
                logger.error(error_msg)
                return False
            else:
                logger.info("تم إرسال الرسالة إلى Telegram بنجاح")
                return True
        except Exception as e:
            error_msg = f"خطأ في إرسال رسالة Telegram: {e}"
            logger.error(error_msg)
            return False

class PerformanceAnalyzer:
    def __init__(self):
        self.daily_trades = []
        self.daily_start_balance = 0
        self.daily_start_time = datetime.now()
        self.daily_max_loss_limit = 0.02  # 2% حد الخسارة اليومية
        self.trading_enabled = True
        
    def add_trade(self, trade_data):
        self.daily_trades.append(trade_data)
        
    def calculate_daily_performance(self, current_balance):
        total_trades = len(self.daily_trades)
        winning_trades = len([t for t in self.daily_trades if t.get('profit_loss', 0) > 0])
        losing_trades = total_trades - winning_trades
        
        win_rate = (winning_trades / total_trades * 100) if total_trades > 0 else 0
        
        total_profit = sum(t.get('profit_loss', 0) for t in self.daily_trades if t.get('profit_loss', 0) > 0)
        total_loss = abs(sum(t.get('profit_loss', 0) for t in self.daily_trades if t.get('profit_loss', 0) < 0))
        
        profit_factor = (total_profit / total_loss) if total_loss > 0 else float('inf')
        
        daily_pnl = current_balance - self.daily_start_balance
        daily_return = (daily_pnl / self.daily_start_balance * 100) if self.daily_start_balance > 0 else 0
        
        return {
            'daily_start_balance': self.daily_start_balance,
            'daily_end_balance': current_balance,
            'daily_pnl': daily_pnl,
            'daily_return': daily_return,
            'total_trades': total_trades,
            'winning_trades': winning_trades,
            'losing_trades': losing_trades,
            'win_rate': win_rate,
            'total_profit': total_profit,
            'total_loss': total_loss,
            'profit_factor': profit_factor,
            'avg_profit_per_trade': (total_profit / winning_trades) if winning_trades > 0 else 0,
            'avg_loss_per_trade': (total_loss / losing_trades) if losing_trades > 0 else 0,
            'trading_enabled': self.trading_enabled
        }
    
    def check_daily_loss_limit(self, current_balance):
        """التحقق من الحد اليومي للخسارة"""
        daily_pnl = current_balance - self.daily_start_balance
        daily_loss_pct = abs(daily_pnl / self.daily_start_balance) if daily_pnl < 0 else 0
        
        if daily_loss_pct >= self.daily_max_loss_limit and self.trading_enabled:
            self.trading_enabled = False
            return False, daily_loss_pct
        return self.trading_enabled, daily_loss_pct
    
    def reset_daily_stats(self, new_start_balance):
        self.daily_trades = []
        self.daily_start_balance = new_start_balance
        self.daily_start_time = datetime.now()
        self.trading_enabled = True

class Crypto_Trading_Bot:
    def __init__(self, api_key=None, api_secret=None, telegram_token=None, telegram_chat_id=None):
        self.notifier = None
        self.trade_history = []
        self.performance_analyzer = PerformanceAnalyzer()
        self.load_trade_history()
        self.last_buy_prices = {} 
        
        # إعدادات العتبات الجديدة
        self.BASELINE_BUY_THRESHOLD = 45 # رفع من 25 إلى 35
        self.STRICT_BUY_THRESHOLD = 55  # رفع من 20 إلى 45 (للأوامر الممتلئة)
        self.SELL_THRESHOLD = 35     # عتبة البيع تبقى كما هي

        self.active_trailing_stops = {}  # لتتبع التريلينغ ستوب


        self.last_buy_contributions = {}
        self.last_sell_contributions = {}
        self.active_trailing_stops = {}  # لتتبع التريلينغ ستوب
        self.symbols = ["BNBUSDT", "ETHUSDT"]  #

        self.MIN_TRADE_SIZE = 5  # ← أضف هذا
        self.MAX_TRADE_SIZE = 50  # ← أضف هذا
        self.MAX_ALGO_ORDERS = 10  # ← أضف هذا
        self.fee_rate = 0.0005  # ← أضف هذا
        self.slippage = 0.00015  # ← أضف هذا
        self.STOP_LOSS = 0.02  # ← أضف هذا
        self.MAX_POSITION_SIZE = 0.5
        
        self.api_key = api_key or os.environ.get('BINANCE_API_KEY')
        self.api_secret = api_secret or os.environ.get('BINANCE_API_SECRET')
        telegram_token = telegram_token or os.environ.get('TELEGRAM_BOT_TOKEN')
        telegram_chat_id = telegram_chat_id or os.environ.get('TELEGRAM_CHAT_ID')
        
        if not self.api_key or not self.api_secret:
            error_msg = "❌ مفاتيح Binance غير موجودة"
            logger.error(error_msg)
            raise ValueError(error_msg)
            
        try:
            self.client = Client(self.api_key, self.api_secret)
            logger.info("✅ تم الاتصال بمنصة Binance الفعلية")
            self.test_connection()
                
        except Exception as e:
            error_msg = f"❌ فشل الاتصال بـ Binance: {e}"
            logger.error(error_msg)
            raise ConnectionError(error_msg)
            
        self.fee_rate = 0.0005
        self.slippage = 0.00015
        self.trades = []
        self.symbols = ["BNBUSDT", "ETHUSDT"]  # إضافة ETH إلى العملات المتداولة
        self.STOP_LOSS = 0.02
        self.MAX_POSITION_SIZE = 0.5
        
        # إعدادات إدارة الأوامر
        self.MAX_ALGO_ORDERS = 10
        self.ORDERS_TO_CANCEL = 2
        
        # إعدادات حجم الصفقة بالدولار حسب قوة الإشارة
        self.MIN_TRADE_SIZE = 5
        self.MAX_TRADE_SIZE = 50
        
        if telegram_token and telegram_chat_id:
            self.notifier = TelegramNotifier(telegram_token, telegram_chat_id)
            logger.info("تم تهيئة إشعارات Telegram")
        else:
            logger.warning("مفاتيح Telegram غير موجودة، سيتم تعطيل الإشعارات")
        
        try:
            self.initial_balance = self.get_real_balance()
            self.performance_analyzer.daily_start_balance = self.initial_balance
        
            # الحصول على الرصيد التفصيلي
            detailed_balance = self.get_detailed_balance()
            balance_details = "\n".join(detailed_balance)
        
            success_msg = f"✅ تم تهيئة البوت بنجاح - الرصيد الابتدائي: ${self.initial_balance:.2f}"
            logger.info(success_msg)
        
            if self.notifier:
                self.notifier.send_message(
                    f"🤖 <b>بدء تشغيل بوت تداول العملات المشفرة المحسن</b>\n\n"
                    f"{success_msg}\n"
                    f"📊 <b>الرصيد التفصيلي:</b>\n{balance_details}\n\n"
                    f"🪙 العملات المتداولة: BNB, ETH\n"
                    f"📦 نطاق حجم الصفقة: ${self.MIN_TRADE_SIZE}-${self.MAX_TRADE_SIZE}\n"
                    f"🔢 الحد الأقصى للأوامر: {self.MAX_ALGO_ORDERS}\n"
                    f"🟢 عتبة الشراء الأساسية: {self.BASELINE_BUY_THRESHOLD}%\n"
                    f"🟡 عتبة الشراء المشددة: {self.STRICT_BUY_THRESHOLD}%\n"
                    f"🔴 عتبة البيع: {self.SELL_THRESHOLD}%\n"
                    f"⛔ حد الخسارة اليومي: 2%\n"
                    f"🗳️ نظام التصويت: مفعل مع مؤشر ADX\n"
                    f"📉 التريلينغ ستوب: مفعل"
                )
        except Exception as e:
            logger.error(f"خطأ في جلب الرصيد الابتدائي: {e}")
            self.initial_balance = 0

    def start_trading(self, cycle_interval=300):
        """بدء التداول التلقائي"""
        logger.info(f"🚀 بدء التداول التلقائي - دورة كل {cycle_interval} ثانية")
    
        if self.notifier:
            self.notifier.send_message(
                f"🚀 <b>بدء التداول التلقائي</b>\n\n"
                f"⏰ الفاصل الزمني: {cycle_interval} ثانية\n"
                f"📊 الرصيد الابتدائي: ${self.initial_balance:.2f}\n"
                f"🪙 العملات: {', '.join(self.symbols)}"
            )
    
        while True:
            try:
                self.run_trading_cycle()
                logger.info(f"⏳ انتظار {cycle_interval} ثانية للدورة القادمة...")
                time.sleep(cycle_interval)
            except Exception as e:
                logger.error(f"❌ خطأ في التداول التلقائي: {e}")
                if self.notifier:
                    self.notifier.send_message(f"❌ <b>خطأ في التداول التلقائي:</b>\n{str(e)}")
                time.sleep(60)  # انتظار دقيقة قبل إعادة المحاولة
	
    def get_detailed_balance(self):
        """الحصول على الرصيد التفصيلي لكل عملة"""
        try:
            account = self.client.get_account()
            detailed_balance = []
        
            for asset in account['balances']:
                free = float(asset['free'])
                locked = float(asset['locked'])
                total = free + locked
            
                if total > 0:  # عرض فقط العملات التي لها رصيد
                    detailed_balance.append(f"{asset['asset']}: {total:.8f} (free: {free:.8f}, locked: {locked:.8f})")
        
            return detailed_balance
        
        except Exception as e:
            logger.error(f"خطأ في جلب الرصيد التفصيلي: {e}")
            return ["غير متوفر"]

    def update_trailing_stops(self, symbol, current_price):
        if symbol not in self.active_trailing_stops:
            return False
    
        # تحديث أعلى سعر وأخذ الربح
        if current_price > self.active_trailing_stops[symbol]['highest_price']:
            self.active_trailing_stops[symbol]['highest_price'] = current_price
            self.active_trailing_stops[symbol]['stop_price'] = current_price * (1 - self.STOP_LOSS)
    
        # التحقق من أخذ الربح (+1.5%)
        if current_price >= self.active_trailing_stops[symbol]['highest_price'] * 1.015:
            # تنفيذ البيع بأخذ الربح
            success, message = self.execute_sell_order(symbol, -100, "take_profit")
            if success:
                logger.info(f"تم أخذ الربح لـ {symbol} بالسعر {current_price}")
            return True
    
        # التحقق من وقف الخسارة
        if current_price <= self.active_trailing_stops[symbol]['stop_price']:
            # تنفيذ البيع بوقف الخسارة
            success, message = self.execute_sell_order(symbol, -100, "stop_loss")
            if success:
                logger.info(f"تم وقف الخسارة لـ {symbol} بالسعر {current_price}")
            return True
    
        return False

    def load_trade_history(self):
        """تحميل تاريخ الصفقات من ملف"""
        try:
            if os.path.exists('trade_history.json'):
                with open('trade_history.json', 'r', encoding='utf-8') as f:
                    self.trade_history = json.load(f)
        except Exception as e:
            logger.error(f"خطأ في تحميل تاريخ الصفقات: {e}")
            self.trade_history = []

    def save_trade_history(self):
        """حفظ تاريخ الصفقات إلى ملف"""
        try:
            with open('trade_history.json', 'w', encoding='utf-8') as f:
                json.dump(self.trade_history, f, ensure_ascii=False, indent=2)
        except Exception as e:
            logger.error(f"خطأ في حفظ تاريخ الصفقات: {e}")

    def add_trade_record(self, symbol, trade_type, quantity, price, trade_size, signal_strength, order_id=None, status="executed", profit_loss=0, exit_type=None):
        """إضافة سجل صفقة جديدة"""
        trade_record = {
            'symbol': symbol,
            'timestamp': datetime.now().isoformat(),
            'type': trade_type,
            'quantity': quantity,
            'price': price,
            'trade_size': trade_size,
            'signal_strength': signal_strength,
            'order_id': order_id,
            'status': status,
            'profit_loss': profit_loss,
            'exit_type': exit_type  # stop_loss, take_profit, trailing_stop, manual
        }
        self.trade_history.append(trade_record)
        self.performance_analyzer.add_trade(trade_record)
        self.save_trade_history()

    def generate_12h_trading_report(self):
        """إنشاء تقرير التداول لآخر 12 ساعة"""
        try:
            twelve_hours_ago = datetime.now() - timedelta(hours=12)
            recent_trades = [
                trade for trade in self.trade_history 
                if datetime.fromisoformat(trade['timestamp']) >= twelve_hours_ago
            ]
            
            if not recent_trades:
                return {"message": "لا توجد صفقات في آخر 12 ساعة"}
            
            # حساب الإحصائيات لكل عملة
            report = {"period": "آخر 12 ساعة", "symbols": {}}
            
            for symbol in self.symbols:
                symbol_trades = [t for t in recent_trades if t['symbol'] == symbol]
                
                if not symbol_trades:
                    continue
                    
                buy_trades = [t for t in symbol_trades if t['type'] == 'buy']
                sell_trades = [t for t in symbol_trades if t['type'] == 'sell']
                
                total_buy_size = sum(t['trade_size'] for t in buy_trades)
                total_sell_size = sum(t['trade_size'] for t in sell_trades)
                avg_buy_strength = np.mean([t['signal_strength'] for t in buy_trades]) if buy_trades else 0
                avg_sell_strength = np.mean([t['signal_strength'] for t in sell_trades]) if sell_trades else 0
                
                profitable_trades = [t for t in symbol_trades if t.get('profit_loss', 0) > 0]
                losing_trades = [t for t in symbol_trades if t.get('profit_loss', 0) < 0]
                
                # تحليل أنواع الخروج
                exit_types = {}
                for trade in symbol_trades:
                    if trade.get('exit_type'):
                        exit_types[trade['exit_type']] = exit_types.get(trade['exit_type'], 0) + 1
                
                report["symbols"][symbol] = {
                    "total_trades": len(symbol_trades),
                    "buy_trades": len(buy_trades),
                    "sell_trades": len(sell_trades),
                    "profitable_trades": len(profitable_trades),
                    "losing_trades": len(losing_trades),
                    "win_rate": (len(profitable_trades) / len(symbol_trades) * 100) if symbol_trades else 0,
                    "total_buy_size": round(total_buy_size, 2),
                    "total_sell_size": round(total_sell_size, 2),
                    "total_profit": sum(t.get('profit_loss', 0) for t in profitable_trades),
                    "total_loss": abs(sum(t.get('profit_loss', 0) for t in losing_trades)),
                    "avg_buy_signal_strength": round(avg_buy_strength, 1),
                    "avg_sell_signal_strength": round(avg_sell_strength, 1),
                    "exit_types": exit_types,
                    "recent_trades": symbol_trades[-5:]  # آخر 5 صفقات لكل عملة
                }
            
            # إحصائيات عامة
            all_profitable = [t for t in recent_trades if t.get('profit_loss', 0) > 0]
            all_losing = [t for t in recent_trades if t.get('profit_loss', 0) < 0]
            
            report["overall"] = {
                "total_trades": len(recent_trades),
                "profitable_trades": len(all_profitable),
                "losing_trades": len(all_losing),
                "win_rate": (len(all_profitable) / len(recent_trades) * 100) if recent_trades else 0,
                "total_profit": sum(t.get('profit_loss', 0) for t in all_profitable),
                "total_loss": abs(sum(t.get('profit_loss', 0) for t in all_losing))
            }
            
            return report
        except Exception as e:
            logger.error(f"خطأ في إنشاء تقرير التداول: {e}")
            return {"error": str(e)}

    def check_key_levels(self, symbol, current_price, data):
        # التحقق من القمم والقيعان
        resistance = data['bb_upper'].iloc[-1]
        support = data['bb_lower'].iloc[-1]
    
        # إذا near resistance - avoid buying
        if current_price > resistance * 0.993:
            return "near_resistance"
        # إذا near support - avoid selling
        elif current_price < support * 1.07:
            return "near_support"
        return "neutral"

    def generate_daily_performance_report(self):
        """إنشاء تقرير أداء يومي شامل"""
        try:
            current_balance = self.get_real_balance()
            performance = self.performance_analyzer.calculate_daily_performance(current_balance)
            
            # تحليل جودة الإشارات
            strong_signals = [t for t in self.performance_analyzer.daily_trades if abs(t['signal_strength']) >= 80]
            medium_signals = [t for t in self.performance_analyzer.daily_trades if 50 <= abs(t['signal_strength']) < 80]
            weak_signals = [t for t in self.performance_analyzer.daily_trades if abs(t['signal_strength']) < 50]
            
            strong_win_rate = (len([t for t in strong_signals if t.get('profit_loss', 0) > 0]) / len(strong_signals) * 100) if strong_signals else 0
            medium_win_rate = (len([t for t in medium_signals if t.get('profit_loss', 0) > 0]) / len(medium_signals) * 100) if medium_signals else 0
            weak_win_rate = (len([t for t in weak_signals if t.get('profit_loss', 0) > 0]) / len(weak_signals) * 100) if weak_signals else 0
            
            # تحليل الأداء حسب العملة
            symbol_performance = {}
            for symbol in self.symbols:
                symbol_trades = [t for t in self.performance_analyzer.daily_trades if t.get('symbol') == symbol]
                if symbol_trades:
                    symbol_profitable = [t for t in symbol_trades if t.get('profit_loss', 0) > 0]
                    symbol_win_rate = (len(symbol_profitable) / len(symbol_trades) * 100) if symbol_trades else 0
                    symbol_total_profit = sum(t.get('profit_loss', 0) for t in symbol_profitable)
                    
                    symbol_performance[symbol] = {
                        'total_trades': len(symbol_trades),
                        'win_rate': symbol_win_rate,
                        'total_profit': symbol_total_profit
                    }
            
            report = {
                "date": datetime.now().strftime("%Y-%m-%d"),
                "performance": performance,
                "signal_analysis": {
                    "strong_signals": len(strong_signals),
                    "strong_win_rate": round(strong_win_rate, 1),
                    "medium_signals": len(medium_signals),
                    "medium_win_rate": round(medium_win_rate, 1),
                    "weak_signals": len(weak_signals),
                    "weak_win_rate": round(weak_win_rate, 1)
                },
                "symbol_performance": symbol_performance,
                "recommendations": self.generate_recommendations(performance)
            }
            
            return report
        except Exception as e:
            logger.error(f"خطأ في إنشاء التقرير اليومي: {e}")
            return {"error": str(e)}

    def generate_recommendations(self, performance):
        """توليد توصيات بناء على الأداء"""
        recommendations = []
        
        if performance['win_rate'] < 50:
            recommendations.append("⚡ فكر في تعديل استراتيجية الشراء/البيع")
        
        if performance['profit_factor'] < 1.5:
            recommendations.append("📉 ضعيف - تحتاج إلى تحسين نسبة الربح/الخسارة")
        elif performance['profit_factor'] < 2.0:
            recommendations.append("📊 متوسط - أداء مقبول ولكن يمكن التحسين")
        else:
            recommendations.append("📈 ممتاز - استمر في الاستراتيجية الحالية")
        
        if performance['total_trades'] > 15:
            recommendations.append("⚠️ عدد الصفقات مرتفع - فكر في تقليل التردد")
        elif performance['total_trades'] < 5:
            recommendations.append("ℹ️ عدد الصفقات منخفض - قد تحتاج إلى زيادة حساسية الإشارات")
        
        return recommendations

    def test_connection(self):
        try:
            server_time = self.client.get_server_time()
            logger.info(f"✅ الاتصال ناجح - وقت الخادم: {server_time['serverTime']}")
            
            account_info = self.client.get_account()
            logger.info("✅ جلب معلومات الحساب ناجح")
            
            public_ip = self.get_public_ip()
            logger.info(f"🌐 IP الخادم: {public_ip}")
            
            print("="*50)
            print("✅ اختبار الاتصال ناجح!")
            print("وضع التشغيل: فعلي")
            print(f"IP الخادم: {public_ip}")
            print(f"العملات المتداولة: {', '.join(self.symbols)}")
            print(f"حجم الصفقة: ${self.MIN_TRADE_SIZE}-${self.MAX_TRADE_SIZE}")
            print(f"الحد الأقصى للأوامر: {self.MAX_ALGO_ORDERS}")
            print(f"عتبة الشراء الأساسية: {self.BASELINE_BUY_THRESHOLD}%")
            print(f"عتبة الشراء المشددة: {self.STRICT_BUY_THRESHOLD}%")
            print(f"عتبة البيع: {self.SELL_THRESHOLD}%")
            print(f"حد الخسارة اليومي: 2%")
            print("="*50)
            
            return True
        except Exception as e:
            logger.error(f"❌ فشل الاتصال: {e}")
            return False

    def get_public_ip(self):
        try:
            response = requests.get('https://api.ipify.org?format=json', timeout=10)
            return response.json()['ip']
        except:
            return "غير معروف"
    
    def get_real_balance(self):
        try:
            account = self.client.get_account()
            balances = {asset['asset']: float(asset['free']) + float(asset['locked']) for asset in account['balances']}
            
            prices = self.client.get_all_tickers()
            price_dict = {item['symbol']: float(item['price']) for item in prices}
            
            total_balance = 0
            for asset, balance in balances.items():
                if balance > 0:
                    if asset == 'USDT':
                        total_balance += balance
                    else:
                        symbol = asset + 'USDT'
                        if symbol in price_dict:
                            total_balance += balance * price_dict[symbol]
            
            return total_balance
        except Exception as e:
            error_msg = f"❌ خطأ في جلب الرصيد من المنصة: {e}"
            logger.error(error_msg)
            if self.notifier:
                self.notifier.send_message(error_msg)
            raise
    
    def get_account_balance_details(self):
        try:
            account = self.client.get_account()
            balances = {asset['asset']: {
                'free': float(asset['free']),
                'locked': float(asset['locked']),
                'total': float(asset['free']) + float(asset['locked'])
            } for asset in account['balances'] if float(asset['free']) > 0 or float(asset['locked']) > 0}
            
            # الحصول على أسعار جميع العملات
            tickers = self.client.get_all_tickers()
            prices = {}
            for symbol in self.symbols:
                for ticker in tickers:
                    if ticker['symbol'] == symbol:
                        prices[symbol] = float(ticker['price'])
                        break
            
            total_balance = self.get_real_balance()
            
            return total_balance, balances, prices
        except Exception as e:
            error_msg = f"❌ خطأ في الحصول على رصيد الحساب: {e}"
            logger.error(error_msg)
            return None, None, None
    
    def send_notification(self, message):
        logger.info(message)
        if self.notifier:
            success = self.notifier.send_message(message)
            if not success:
                logger.error("فشل إرسال الإشعار إلى Telegram")
            return success
        return False
    
    def format_price(self, price, symbol):
        """تقريب السعر حسب متطلبات Binance"""
        try:
            info = self.client.get_symbol_info(symbol)
            price_filter = [f for f in info['filters'] if f['filterType'] == 'PRICE_FILTER'][0]
            tick_size = float(price_filter['tickSize'])
            
            formatted_price = round(price / tick_size) * tick_size
            return round(formatted_price, 8)
        except Exception as e:
            logger.error(f"خطأ في تقريب السعر: {e}")
            return round(price, 4)

    def get_algo_orders_count(self, symbol):
        """الحصول على عدد جميع الأوامر المعلقة"""
        try:
            open_orders = self.client.get_open_orders(symbol=symbol)
            return len(open_orders)
        except Exception as e:
            logger.error(f"خطأ في جلب عدد الأوامر النشطة: {e}")
            return 0
    
    def get_total_orders_count(self):
        """الحصول على إجمالي عدد الأوامر لجميع العملات"""
        total_orders = 0
        for symbol in self.symbols:
            total_orders += self.get_algo_orders_count(symbol)
        return total_orders
    
    def get_order_space_status(self, symbol):
        """الحصول على حالة مساحة الأوامر (الجديدة)"""
        try:
            current_orders = self.get_algo_orders_count(symbol)
            
            if current_orders >= self.MAX_ALGO_ORDERS:
                return "FULL"  # الأوامر ممتلئة
            elif current_orders >= (self.MAX_ALGO_ORDERS - 2):
                return "NEAR_FULL"  # الأوامر قريبة من الامتلاء
            else:
                return "AVAILABLE"  # المساحة متاحة
                
        except Exception as e:
            logger.error(f"خطأ في التحقق من حالة الأوامر: {e}")
            return "FULL"  # في حالة الخطأ، نفترض أن الأوامر ممتلئة للسلامة
    
    def cancel_oldest_orders(self, symbol, num_to_cancel=2):
        """إلغاء أقدم الأوامر"""
        try:
            open_orders = self.client.get_open_orders(symbol=symbol)
        
            all_orders = []
            for order in open_orders:
                order_time = datetime.fromtimestamp(order['time'] / 1000)
                all_orders.append({
                    'orderId': order['orderId'],
                    'time': order_time,
                    'type': order['type'],
                    'side': order['side'],
                    'price': order.get('price', 'N/A'),
                    'quantity': order.get('origQty', 'N/A')
                })
        
            all_orders.sort(key=lambda x: x['time'])
        
            cancelled_count = 0
            cancelled_info = []
        
            for i in range(min(num_to_cancel, len(all_orders))):
                try:
                    self.client.cancel_order(
                        symbol=symbol,
                        orderId=all_orders[i]['orderId']
                    )
                    cancelled_count += 1
                    cancelled_info.append(f"{all_orders[i]['type']} - {all_orders[i]['side']} - {all_orders[i]['price']}")
                    logger.info(f"تم إلغاء الأمر القديم: {all_orders[i]['orderId']}")
                
                    self.add_trade_record(
                        symbol=symbol,
                        trade_type="cancel",
                        quantity=float(all_orders[i]['quantity']),
                        price=float(all_orders[i]['price']),
                        trade_size=0,
                        signal_strength=0,
                        order_id=all_orders[i]['orderId'],
                        status="cancelled"
                    )
                
                except Exception as e:
                    logger.error(f"خطأ في إلغاء الأمر {all_orders[i]['orderId']}: {e}")
        
            return cancelled_count, cancelled_info
        
        except Exception as e:
            logger.error(f"خطأ في إلغاء الأوامر القديمة: {e}")
            return 0, []
    
    def manage_order_space(self, symbol):
        """إدارة مساحة الأوامر (محدثة)"""
        try:
            order_status = self.get_order_space_status(symbol)
            
            if order_status == "FULL":
                self.send_notification(f"⛔ الأوامر ممتلئة لـ {symbol} - تم إلغاء الصفقة الجديدة لحماية الصفقات الحالية")
                return False
            
            return True
            
        except Exception as e:
            logger.error(f"خطأ في إدارة المساحة: {e}")
            return False
    
    def calculate_adx(self, data, period=14):
        """حساب مؤشر ADX (Average Directional Index)"""
        try:
            high = data['high']
            low = data['low']
            close = data['close']
            
            # حساب +DM و -DM
            up_move = high.diff()
            down_move = low.diff().abs() * -1
            
            plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0)
            minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0)
            
            # حساب True Range
            tr1 = high - low
            tr2 = (high - close.shift()).abs()
            tr3 = (low - close.shift()).abs()
            tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
            
            # حساب المتوسطات
            atr = tr.rolling(period).mean()
            plus_di = 100 * (pd.Series(plus_dm).rolling(period).mean() / atr)
            minus_di = 100 * (pd.Series(minus_dm).rolling(period).mean() / atr)
            
            # حساب ADX
            dx = 100 * ((plus_di - minus_di).abs() / (plus_di + minus_di))
            adx = dx.rolling(period).mean()
            
            return adx, plus_di, minus_di
        except Exception as e:
            logger.error(f"خطأ في حساب ADX: {e}")
            return None, None, None

    def calculate_signal_strength(self, data, signal_type='buy'):
        """تقييم قوة الإشارة من -100 إلى +100% مع نظام التصويت المعدل"""
        latest = data.iloc[-1]

        # منع الشراء في ذروة الشراء
        if signal_type == 'buy' and latest['rsi'] > 65:
            return -100  # لا تشتري أبداً
    
        # منع البيع في ذروة البيع  
        if signal_type == 'sell' and latest['rsi'] < 35:
            return -100  # لا تبيع أبداً
    
        # نظام التصويت المعدل (مجموع الأوزان = 100%)
        votes = {
            'market_trend': 0,      # 20%
            'moving_averages': 0,   # 15%
            'macd': 0,              # 15%
            'rsi': 0,               # 12%
            'bollinger_bands': 0,   # 15%
            'volume': 0,            # 15%
            'adx': 0                # 8%
        }
    
        # عوامل التطبيع للأوزان
        normalization_factors = {
            'market_trend': 0.20,      # 20%
            'moving_averages': 0.15,   # 15%
            'macd': 0.15,              # 15%
            'rsi': 0.12,               # 12%
            'bollinger_bands': 0.15,   # 15%
            'volume': 0.15,            # 15%
            'adx': 0.08                # 8%
        }
    
        # حساب مساهمة كل مؤشر مع التطبيع
        indicator_contributions = {}
        indicator_contributions['market_trend'] = self.calculate_market_trend_score(data, signal_type) * normalization_factors['market_trend']
        indicator_contributions['moving_averages'] = self.calculate_ema_score(data, signal_type) * normalization_factors['moving_averages']
        indicator_contributions['macd'] = self.calculate_macd_score(data, signal_type) * normalization_factors['macd']
        indicator_contributions['rsi'] = self.calculate_rsi_score(data, signal_type) * normalization_factors['rsi']
        indicator_contributions['bollinger_bands'] = self.calculate_bollinger_bands_score(data, signal_type) * normalization_factors['bollinger_bands']
        indicator_contributions['volume'] = self.calculate_volume_score(data, signal_type) * normalization_factors['volume']
        indicator_contributions['adx'] = self.calculate_adx_score(data, signal_type) * normalization_factors['adx']

        # نظام التصويت (كل مؤشر يصوت بنعم/لا/محايد)
        votes['market_trend'] = 1 if indicator_contributions['market_trend'] > 2 else (-1 if indicator_contributions['market_trend'] < -2 else 0)
        votes['moving_averages'] = 1 if indicator_contributions['moving_averages'] > 1.5 else (-1 if indicator_contributions['moving_averages'] < -1.5 else 0)
        votes['macd'] = 1 if indicator_contributions['macd'] > 1.5 else (-1 if indicator_contributions['macd'] < -1.5 else 0)
        votes['rsi'] = 1 if indicator_contributions['rsi'] > 1.2 else (-1 if indicator_contributions['rsi'] < -1.2 else 0)
        votes['bollinger_bands'] = 1 if indicator_contributions['bollinger_bands'] > 1.5 else (-1 if indicator_contributions['bollinger_bands'] < -1.5 else 0)
        votes['volume'] = 1 if indicator_contributions['volume'] > 1.5 else (-1 if indicator_contributions['volume'] < -1.5 else 0)
        votes['adx'] = 1 if indicator_contributions['adx'] > 0.8 else (-1 if indicator_contributions['adx'] < -0.8 else 0)

        # حساب النتيجة الإجمالية بناء على التصويت
        total_votes = sum(votes.values())
        max_possible_votes = len(votes)
    
        # تحويل التصويت إلى نسبة مئوية (-100% إلى +100%)
        vote_percentage = (total_votes / max_possible_votes) * 100
    
        # الجمع بين النظام القديم والجديد (70% للنظام القديم، 30% للتصويت)
        old_score = sum(indicator_contributions.values())
        combined_score = (old_score * 0.7) + (vote_percentage * 0.3)
    
        # تخزين مساهمات المؤشرات حسب نوع الإشارة
        if signal_type == 'buy':
            self.last_buy_contributions = indicator_contributions
        else:
            self.last_sell_contributions = indicator_contributions

        # تطبيق الحدود (-100 إلى +100)
        final_score = max(min(combined_score, 100), -100)
    
        # تسجيل التفاصيل للتحليل
        logger.debug(f"إشارة {signal_type} - النتيجة القديمة: {old_score:.1f}, التصويت: {vote_percentage:.1f}%, النهائية: {final_score:.1f}%")
    
        return final_score

    def calculate_adx_score(self, data, signal_type):
        """حساب درجة ADX"""
        try:
            adx, plus_di, minus_di = self.calculate_adx(data)
            if adx is None:
                return 0
                
            current_adx = adx.iloc[-1]
            current_plus_di = plus_di.iloc[-1]
            current_minus_di = minus_di.iloc[-1]
            
            if signal_type == 'buy':
                if current_adx > 25 and current_plus_di > current_minus_di:
                    return 15  # اتجاه صعودي قوي
                elif current_adx > 20 and current_plus_di > current_minus_di:
                    return 10  # اتجاه صعودي
                elif current_adx < 15:
                    return -10  # سوق جانبي
                else:
                    return 0
            else:  # sell
                if current_adx > 25 and current_minus_di > current_plus_di:
                    return 15  # اتجاه هبوطي قوي
                elif current_adx > 20 and current_minus_di > current_plus_di:
                    return 10  # اتجاه هبوطي
                elif current_adx < 15:
                    return -10  # سوق جانبي
                else:
                    return 0
        except Exception as e:
            logger.error(f"خطأ في حساب درجة ADX: {e}")
            return 0

    def calculate_ema_score(self, data, signal_type):
        """حساب درجة المتوسطات المتحركة بتدرج منطقي"""
        latest = data.iloc[-1]
    
        # حساب قوة الاتجاه بالنسبة لـ EMA 34
        price_vs_ema = ((latest['close'] - latest['ema34']) / latest['ema34']) * 100
    
        if signal_type == 'buy':
            if price_vs_ema > 5.0:  # فوق EMA 34 بأكثر من 5%
                return 20.0  # 100%
            elif price_vs_ema > 2.5:  # فوق EMA 34 بأكثر من 2.5%
                return 15.0  # 75%
            elif price_vs_ema > 0:  # فوق EMA 34
                return 10.0  # 50%
            elif price_vs_ema > -2.5:  # تحت EMA 34 بأقل من 2.5%
                return 5.0  # 25%
            else:  # تحت EMA 34 بأكثر من 2.5%
                return 0.0  # 0%
        else:  # sell
            if price_vs_ema < -5.0:  # تحت EMA 34 بأكثر من 5%
                return 20.0  # 100%
            elif price_vs_ema < -2.5:  # تحت EMA 34 بأكثر من 2.5%
                return 15.0  # 75%
            elif price_vs_ema < 0:  # تحت EMA 34
                return 10.0  # 50%
            elif price_vs_ema < 2.5:  # فوق EMA 34 بأقل من 2.5%
                return 5.0  # 25%
            else:  # فوق EMA 34 بأكثر من 2.5%
                return 0.0  # 0%

    def calculate_macd_score(self, data, signal_type):
        """حساب درجة MACD"""
        latest = data.iloc[-1]
        prev = data.iloc[-2]
    
        macd_diff = latest['macd'] - latest['macd_signal']
        prev_macd_diff = prev['macd'] - prev['macd_signal']
    
        if signal_type == 'buy':
            if macd_diff > 0 and prev_macd_diff <= 0:  # تقاطع صعودي جديد
                return 20.0  # 100%
            elif macd_diff > 0 and macd_diff > prev_macd_diff:  # اتجاه صعودي متسارع
                return 15.0  # 75%
            elif macd_diff > 0:  # إيجابي ولكن ثابت
                return 10.0  # 50%
            elif macd_diff < 0 and macd_diff > prev_macd_diff:  # تحسن لكن لا يزال سلبي
                return 5.0  # 25%
            else:  # سلبي ومتراجع
                return 0.0  # 0%
        else:  # sell
            if macd_diff < 0 and prev_macd_diff >= 0:  # تقاطع هبوطي جديد
                return 20.0  # 100%
            elif macd_diff < 0 and macd_diff < prev_macd_diff:  # اتجاه هبوطي متسارع
                return 15.0  # 75%
            elif macd_diff < 0:  # سلبي ولكن ثابت
                return 10.0  # 50%
            elif macd_diff > 0 and macd_diff < prev_macd_diff:  # تدهور لكن لا يزال إيجابي
                return 5.0  # 25%
            else:  # إيجابي ومتزايد
                return 0.0  # 0%

    def calculate_rsi_score(self, data, signal_type):
        """حساب درجة RSI"""
        latest = data.iloc[-1]
    
        if signal_type == 'buy':
            if latest['rsi'] < 30:  # ذروة بيع قوية
                return 15.0  # 100%
            elif latest['rsi'] < 35:  # ذروة بيع
                return 12.0  # 80%
            elif latest['rsi'] < 40:  # قريب من ذروة البيع
                return 9.0  # 60%
            elif latest['rsi'] < 45:  # محايد مائل للبيع
                return 6.0  # 40%
            elif latest['rsi'] < 50:  # محايد
                return 3.0  # 20%
            else:  # في منطقة الشراء
                return 0.0  # 0%
        else:  # sell
            if latest['rsi'] > 70:  # ذروة شراء قوية
                return 15.0  # 100%
            elif latest['rsi'] > 65:  # ذروة شراء
                return 12.0  # 80%
            elif latest['rsi'] > 60:  # قريب من ذروة الشراء
                return 9.0  # 60%
            elif latest['rsi'] > 55:  # محايد مائل للشراء
                return 6.0  # 40%
            elif latest['rsi'] > 50:  # محايد
                return 3.0  # 20%
            else:  # في منطقة البيع
                return 0.0  # 0%

    def calculate_bollinger_bands_score(self, data, signal_type):
        """حساب درجة بولينجر باند"""
        latest = data.iloc[-1]
    
        # حساب موقع السعر بالنسبة للباندات
        bb_position = (latest['close'] - latest['bb_lower']) / (latest['bb_upper'] - latest['bb_lower'])
    
        if signal_type == 'buy':
            if bb_position < 0.1:  # تحت الباند السفلي مباشرة
                return 20.0  # 100%
            elif bb_position < 0.2:  # قريب من الباند السفلي
                return 15.0  # 75%
            elif bb_position < 0.3:  # في الثلث السفلي
                return 10.0  # 50%
            elif bb_position < 0.4:  # تحت الوسط
                return 5.0  # 25%
            else:  # في النصف العلوي
                return 0.0  # 0%
        else:  # sell
            if bb_position > 0.9:  # فوق الباند العلوي مباشرة
                return 20.0  # 100%
            elif bb_position > 0.8:  # قريب من الباند العلوي
                return 15.0  # 75%
            elif bb_position > 0.7:  # في الثلث العلوي
                return 10.0  # 50%
            elif bb_position > 0.6:  # فوق الوسط
                return 5.0  # 25%
            else:  # في النصف السفلي
                return 0.0  # 0%

    def calculate_volume_score(self, data, signal_type):
        """حساب درجة الحجم"""
        latest = data.iloc[-1]
        prev = data.iloc[-2]
    
        # حساب نسبة الحجم إلى المتوسط
        volume_ratio = latest['volume'] / latest['volume_ma']
    
        if signal_type == 'buy':
            if volume_ratio > 2.0:  # حجم كبير جداً
                return 20.0  # 100%
            elif volume_ratio > 1.5:  # حجم كبير
                return 15.0  # 75%
            elif volume_ratio > 1.2:  # حجم فوق المتوسط
                return 10.0  # 50%
            elif volume_ratio > 1.0:  # حجم طبيعي
                return 5.0  # 25%
            else:  # حجم ضعيف
                return 0.0  # 0%
        else:  # sell
            if volume_ratio > 2.0:  # حجم كبير جداً (للبيع أيضاً)
                return 20.0  # 100%
            elif volume_ratio > 1.5:  # حجم كبير
                return 15.0  # 75%
            elif volume_ratio > 1.2:  # حجم فوق المتوسط
                return 10.0  # 50%
            elif volume_ratio > 1.0:  # حجم طبيعي
                return 5.0  # 25%
            else:  # حجم ضعيف
                return 0.0  # 0%

    def calculate_market_trend_score(self, data, signal_type):
        """حساب درجة اتجاه السوق"""
        latest = data.iloc[-1]
    
        # اتجاه EMA 8 بالنسبة لـ EMA 21
        ema_trend = ((latest['ema8'] - latest['ema21']) / latest['ema21']) * 100
    
        if signal_type == 'buy':
            if ema_trend > 2.0:  # اتجاه صعودي قوي
                return 25.0  # 100%
            elif ema_trend > 1.0:  # اتجاه صعودي
                return 18.75  # 75%
            elif ema_trend > 0:  # اتجاه صعودي خفيف
                return 12.5  # 50%
            elif ema_trend > -1.0:  # محايد مائل للهبوط
                return 6.25  # 25%
            else:  # اتجاه هبوطي
                return 0.0  # 0%
        else:  # sell
            if ema_trend < -2.0:  # اتجاه هبوطي قوي
                return 25.0  # 100%
            elif ema_trend < -1.0:  # اتجاه هبوطي
                return 18.75  # 75%
            elif ema_trend < 0:  # اتجاه هبوطي خفيف
                return 12.5  # 50%
            elif ema_trend < 1.0:  # محايد مائل للصعود
                return 6.25  # 25%
            else:  # اتجاه صعودي
                return 0.0  # 0%

    def get_historical_data(self, symbol, interval='15m', limit=100):
        """جلب البيانات التاريخية"""
        try:
            klines = self.client.get_klines(symbol=symbol, interval=interval, limit=limit)
            data = []
            for k in klines:
                data.append({
                    'timestamp': k[0],
                    'open': float(k[1]),
                    'high': float(k[2]),
                    'low': float(k[3]),
                    'close': float(k[4]),
                    'volume': float(k[5])
                })
            df = pd.DataFrame(data)
            return df
        except Exception as e:
            logger.error(f"❌ خطأ في جلب البيانات التاريخية لـ {symbol}: {e}")
            return None

    def calculate_technical_indicators(self, data):
        """حساب المؤشرات الفنية"""
        try:
            df = data.copy()
            
            # المتوسطات المتحركة
            df['ema8'] = df['close'].ewm(span=8).mean()
            df['ema21'] = df['close'].ewm(span=21).mean()
            df['ema34'] = df['close'].ewm(span=34).mean()
            df['ema55'] = df['close'].ewm(span=55).mean()
            
            # MACD
            exp12 = df['close'].ewm(span=12).mean()
            exp26 = df['close'].ewm(span=26).mean()
            df['macd'] = exp12 - exp26
            df['macd_signal'] = df['macd'].ewm(span=9).mean()
            
            # RSI
            delta = df['close'].diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
            rs = gain / loss
            df['rsi'] = 100 - (100 / (1 + rs))
            
            # بولينجر باند
            df['bb_middle'] = df['close'].rolling(window=20).mean()
            bb_std = df['close'].rolling(window=20).std()
            df['bb_upper'] = df['bb_middle'] + (bb_std * 2)
            df['bb_lower'] = df['bb_middle'] - (bb_std * 2)
            
            # متوسط الحجم
            df['volume_ma'] = df['volume'].rolling(window=20).mean()
            
            return df
        except Exception as e:
            logger.error(f"❌ خطأ في حساب المؤشرات الفنية: {e}")
            return data

    
    def determine_trade_size(self, signal_strength, symbol):
        # التحقق من عدم تجاوز الحد الأقصى
        current_balance = self.get_real_balance()
        asset = symbol.replace('USDT', '')
    
        # الحصول على القيمة الحالية للعملة
        try:
            balance = self.client.get_asset_balance(asset=asset)
            if balance and float(balance['free']) > 0:
                ticker = self.client.get_symbol_ticker(symbol=symbol)
                current_price = float(ticker['price'])
                current_value = float(balance['free']) * current_price
            
                # إذا تجاوز الحد المسموح
                if current_value > current_balance * self.MAX_POSITION_SIZE:
                    return 0, 0  # لا تشتري أكثر
        except:
            pass
        
        try:
            # تحويل قوة الإشارة إلى نسبة مئوية مطلقة
            strength_percentage = abs(signal_strength) / 100.0
            
            # حساب حجم الصفقة بناء على قوة الإشارة
            trade_size = self.MIN_TRADE_SIZE + (self.MAX_TRADE_SIZE - self.MIN_TRADE_SIZE) * strength_percentage
            
            # الحصول على سعر العملة الحالي
            ticker = self.client.get_symbol_ticker(symbol=symbol)
            current_price = float(ticker['price'])
            
            # حساب الكمية بناء على حجم الصفقة والسعر
            quantity = trade_size / current_price
            
            # الحصول على معلومات الرمز لتقريب الكمية
            info = self.client.get_symbol_info(symbol)
            step_size = None
            for f in info['filters']:
                if f['filterType'] == 'LOT_SIZE':
                    step_size = float(f['stepSize'])
                    break
            
            if step_size:
                # تقريب الكمية إلى أقرب خطوة صحيحة
                quantity = round(quantity / step_size) * step_size
            
            # التأكد من أن الكمية لا تقل عن الحد الأدنى
            min_qty = None
            for f in info['filters']:
                if f['filterType'] == 'LOT_SIZE':
                    min_qty = float(f['minQty'])
                    break
            
            if min_qty and quantity < min_qty:
                quantity = min_qty
            
            return quantity, trade_size
            
        except Exception as e:
            logger.error(f"❌ خطأ في تحديد حجم الصفقة: {e}")
            return 0, 0

    def execute_buy_order(self, symbol, signal_strength):
        """تنفيذ أمر شراء"""
        try:
            # التحقق من مساحة الأوامر أولاً
            if not self.manage_order_space(symbol):
                return False, "لا توجد مساحة للأوامر الجديدة"
            
            # الحصول على السعر الحالي
            ticker = self.client.get_symbol_ticker(symbol=symbol)
            current_price = float(ticker['price'])
            
            # تحديد حجم الصفقة
            quantity, trade_size = self.determine_trade_size(signal_strength, symbol)
            
            if quantity <= 0:
                return False, "حجم الصفقة غير صالح"
            
            # تنفيذ أمر السوق
            order = self.client.order_market_buy(
                symbol=symbol,
                quantity=quantity
            )
            
            # حساب السعر الفعلي مع الانزلاق
            executed_price = float(order['fills'][0]['price']) if order['fills'] else current_price

            # إعداد التريلينغ ستوب بعد الشراء
            self.active_trailing_stops[symbol] = {
                'highest_price': executed_price,
                'stop_price': executed_price * (1 - self.STOP_LOSS),
                'buy_price': executed_price
            }

            self.last_buy_prices[symbol] = executed_price
            
            # إضافة سجل الصفقة
            self.add_trade_record(
                symbol=symbol,
                trade_type="buy",
                quantity=quantity,
                price=executed_price,
                trade_size=trade_size,
                signal_strength=signal_strength,
                order_id=order['orderId']
            )
            
            # إرسال إشعار
            message = (
                f"✅ <b>تم تنفيذ أمر شراء</b>\n\n"
                f"العملة: {symbol}\n"
                f"الكمية: {quantity:.6f}\n"
                f"السعر: ${executed_price:.4f}\n"
                f"حجم الصفقة: ${trade_size:.2f}\n"
                f"قوة الإشارة: {signal_strength:.1f}%\n"
                f"وقت التنفيذ: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
            )
            
            self.send_notification(message)
            
            return True, "تم تنفيذ أمر الشراء بنجاح"
            
        except Exception as e:
            error_msg = f"❌ خطأ في تنفيذ أمر الشراء لـ {symbol}: {e}"
            logger.error(error_msg)
            return False, error_msg

    def execute_sell_order(self, symbol, signal_strength, exit_type=None):
        """تنفيذ أمر بيع"""
        try:
            # التحقق من مساحة الأوامر أولاً
            if not self.manage_order_space(symbol):
                return False, "لا توجد مساحة للأوامر الجديدة"
    
            # الحصول على رصيد العملة
            asset = symbol.replace('USDT', '')
            balance = self.client.get_asset_balance(asset=asset)
            if not balance or float(balance['free']) <= 0:
                return False, "لا يوجد رصيد كافٍ للبيع"
    
            quantity = float(balance['free'])
    
            # الحصول على السعر الحالي
            ticker = self.client.get_symbol_ticker(symbol=symbol)
            current_price = float(ticker['price'])
    
            # تنفيذ أمر السوق
            order = self.client.order_market_sell(
                symbol=symbol,
                quantity=quantity
            )
    
            # حساب السعر الفعلي مع الانزلاق
            executed_price = float(order['fills'][0]['price']) if order['fills'] else current_price
    
            # حساب حجم الصفقة
            trade_size = quantity * executed_price
    
            # حساب الربح/الخسارة إذا كان هناك سعر شراء سابق
            profit_loss = 0
            if symbol in self.last_buy_prices:
                buy_price = self.last_buy_prices[symbol]
                profit_loss = (executed_price - buy_price) * quantity
                del self.last_buy_prices[symbol]  # إزالة السعر بعد البيع
        
            # إضافة سجل الصفقة
            self.add_trade_record(
                symbol=symbol,
                trade_type="sell",
                quantity=quantity,
                price=executed_price,
                trade_size=trade_size,
                signal_strength=signal_strength,
                order_id=order['orderId'],
                profit_loss=profit_loss,
                exit_type=exit_type
            )
     
            # إزالة التريلينغ ستوب إذا كان موجوداً
            if symbol in self.active_trailing_stops:
                del self.active_trailing_stops[symbol]
         
            # إنشاء الرسالة بناءً على نوع البيع
            if exit_type == "trailing_stop":
                message = (
                    f"🔄 <b>بيع بالتريلينغ ستوب</b>\n\n"
                    f"العملة: {symbol}\n"
                    f"الكمية: {quantity:.6f}\n"
                    f"السعر: ${executed_price:.4f}\n"
                    f"حجم الصفقة: ${trade_size:.2f}\n"
                    f"الربح/الخسارة: ${profit_loss:.2f}\n"
                    f"السبب: وقف خسارة أو أخذ ربح تلقائي"
                )
            else:
                message = (
                    f"✅ <b>تم تنفيذ أمر بيع</b>\n\n"
                    f"العملة: {symbol}\n"
                    f"الكمية: {quantity:.6f}\n"
                    f"السعر: ${executed_price:.4f}\n"
                    f"حجم الصفقة: ${trade_size:.2f}\n"
                    f"قوة الإشارة: {signal_strength:.1f}%\n"
                    f"الربح/الخسارة: ${profit_loss:.2f}\n"
                    f"وقت التنفيذ: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
                )
    
            self.send_notification(message)
    
            return True, "تم تنفيذ أمر البيع بنجاح"
    
        except Exception as e:
            error_msg = f"❌ خطأ في تنفيذ أمر البيع لـ {symbol}: {e}"
            logger.error(error_msg)
            return False, error_msg
    
    def run_trading_cycle(self):
        """تشغيل دورة تداول كاملة"""
        try:
            logger.info("=" * 50)
            logger.info(f"بدء دورة التداول - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        
            # التحقق من حد الخسارة اليومي
            current_balance = self.get_real_balance()
            trading_enabled, daily_loss_pct = self.performance_analyzer.check_daily_loss_limit(current_balance)
        
            if not trading_enabled:
                message = (
                    f"⏸️ <b>تم إيقاف التداول اليومي</b>\n\n"
                    f"الخسارة اليومية: {daily_loss_pct * 100:.2f}%\n"
                    f"تجاوز حد الخسارة المسموح به (2%)\n"
                    f"سيستأنف التداول تلقائياً غداً"
                )
                self.send_notification(message)
                logger.warning("تم إيقاف التداول بسبب تجاوز حد الخسارة اليومي")
                return

            # تحديث وقف الخسارة المتابع
            for symbol in self.symbols:
                try:
                    ticker = self.client.get_symbol_ticker(symbol=symbol)
                    current_price = float(ticker['price'])
                    if self.update_trailing_stops(symbol, current_price):
                        self.execute_sell_order(symbol, 100)
                except Exception as e:
                    logger.error(f"خطأ في التريلينغ ستوب لـ {symbol}: {e}")
        
            # تحليل كل عملة وجمع النتائج
            analysis_results = []
            trade_actions = []
        
            for symbol in self.symbols:
                try:
                    logger.info(f"تحليل {symbol}...")
                
                    # جلب البيانات التاريخية
                    data = self.get_historical_data(symbol)
                    if data is None or len(data) < 50:
                        logger.warning(f"بيانات غير كافية لـ {symbol}")
                        analysis_results.append(f"❌ {symbol}: بيانات غير كافية")
                        continue
                
                    # حساب المؤشرات الفنية
                    data = self.calculate_technical_indicators(data)
                
                    # حساب قوة الإشارة
                    buy_signal = self.calculate_signal_strength(data, 'buy')
                    sell_signal = self.calculate_signal_strength(data, 'sell')

                    current_price = data['close'].iloc[-1]
                    key_level = self.check_key_levels(symbol, current_price, data)

                    logger.info(f"{symbol} - إشارة الشراء: {buy_signal:.1f}%, إشارة البيع: {sell_signal:.1f}%")
                
                    # جمع نتائج التحليل
                    signal_status = ""
                    action_taken = ""
                
                    if buy_signal >= self.BASELINE_BUY_THRESHOLD:
                        signal_status = "🟢 شراء"
                    
                        if key_level == "near_resistance":
                            # حساب المسافة من المقاومة
                            resistance_price = data['bb_upper'].iloc[-1]
                            distance_pct = ((resistance_price - current_price) / resistance_price) * 100
                        
                            skip_message = f"⏭️ تخطي الشراء - قريب من المقاومة ({distance_pct:.2f}%)"
                            logger.info(skip_message)
                            action_taken = f"❌ تخطي شراء: قريب من المقاومة ({distance_pct:.2f}% تحت)"
                    
                        else:
                            # شرط إضافي للأوامر الممتلئة
                            order_status = self.get_order_space_status(symbol)
                            if order_status == "NEAR_FULL" and buy_signal < self.STRICT_BUY_THRESHOLD:
                                skip_message = f"⏭️ تخطي الشراء - إشارة غير قوية كفاية"
                                logger.info(skip_message)
                                action_taken = "❌ تخطي شراء: إشارة ضعيفة للأوامر الممتلئة"
                        
                            else:
                                success, message = self.execute_buy_order(symbol, buy_signal)
                                logger.info(f"نتيجة أمر الشراء: {message}")
                                action_taken = f"✅ تم الشراء: {buy_signal:.1f}%"
                
                    elif sell_signal >= self.SELL_THRESHOLD:
                        signal_status = "🔴 بيع"
                    
                        if key_level == "near_support":
                            # حساب المسافة من الدعم
                            support_price = data['bb_lower'].iloc[-1]
                            distance_pct = ((current_price - support_price) / support_price) * 100
                        
                            skip_message = f"⏭️ تخطي البيع - قريب من الدعم ({distance_pct:.2f}%)"
                            logger.info(skip_message)
                            action_taken = f"❌ تخطي بيع: قريب من الدعم ({distance_pct:.2f}% فوق)"
                    
                        else:
                            # شرط إضافي للأوامر الممتلئة
                            order_status = self.get_order_space_status(symbol)
                            if order_status == "NEAR_FULL" and sell_signal < (self.SELL_THRESHOLD + 10):
                                skip_message = f"⏭️ تخطي البيع - إشارة غير قوية كفاية"
                                logger.info(skip_message)
                                action_taken = "❌ تخطي بيع: إشارة ضعيفة للأوامر الممتلئة"
                        
                            else:
                                success, message = self.execute_sell_order(symbol, sell_signal)
                                logger.info(f"نتيجة أمر البيع: {message}")
                                action_taken = f"✅ تم البيع: {sell_signal:.1f}%"
                
                    else:
                        signal_status = "🟡 لا شيء"
                        action_taken = "➡️ لا إجراء: إشارات ضعيفة"
                
                    # إضافة النتائج للتحليل
                    level_info = ""
                    if key_level == "near_resistance":
                        resistance_price = data['bb_upper'].iloc[-1]
                        distance_pct = ((resistance_price - current_price) / resistance_price) * 100
                        level_info = f" | 📈 {distance_pct:.2f}% تحت المقاومة"
                    elif key_level == "near_support":
                        support_price = data['bb_lower'].iloc[-1]
                        distance_pct = ((current_price - support_price) / support_price) * 100
                        level_info = f" | 📉 {distance_pct:.2f}% فوق الدعم"
                
                    analysis_results.append(
                        f"• {symbol}: الشراء {buy_signal:.1f}% | البيع {sell_signal:.1f}% | {signal_status}{level_info}"
                    )
                
                    # إضافة الإجراءات المتخذة
                    if action_taken:
                        trade_actions.append(f"• {symbol}: {action_taken}")
                    
                except Exception as e:
                    error_msg = f"❌ خطأ في معالجة {symbol}: {e}"
                    logger.error(error_msg)
                    analysis_results.append(f"❌ {symbol}: خطأ في المعالجة")
                    trade_actions.append(f"• {symbol}: ❌ خطأ: {str(e)}")
                    continue
            
                # تأجيل بين العملات
                time.sleep(1)
        
            # إرسال رسالة واحدة بنتائج جميع العملات
            if self.notifier and analysis_results:
                results_text = "\n".join(analysis_results)
                actions_text = "\n".join(trade_actions) if trade_actions else "• لا توجد إجراءات"
            
                summary_msg = (
                    f"📊 <b>ملخص دورة التداول</b>\n\n"
                    f"<b>التحليل الفني:</b>\n{results_text}\n\n"
                    f"<b>الإجراءات المتخذة:</b>\n{actions_text}\n\n"
                    f"⏰ الوقت: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n"
                    f"💰 الرصيد: ${current_balance:.2f}\n"
                    f"🔢 الأوامر النشطة: {self.get_total_orders_count()}"
				)
                self.notifier.send_message(summary_msg)
        
            logger.info(f"انتهت دورة التداول - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
            logger.info("=" * 50)
        
        except Exception as e:
            logger.error(f"❌ خطأ في دورة التداول: {e}")
            if self.notifier:
                self.notifier.send_message(f"❌ <b>خطأ في دورة التداول:</b>\n{str(e)}")
            
def main():
    """الدالة الرئيسية لتشغيل البوت"""
    try:
        # بدء خادم Flask في خيط منفصل
        flask_thread = threading.Thread(target=run_flask_app, daemon=True)
        flask_thread.start()
        logger.info("تم بدء خادم Flask للرصد الصحي")
        
        # تهيئة وتشغيل بوت التداول
        bot = Crypto_Trading_Bot()
        bot.start_trading(cycle_interval=300)  # دورة كل 5 دقائق
        
    except Exception as e:
        logger.error(f"❌ خطأ في الدالة الرئيسية: {e}")
        if 'bot' in locals() and hasattr(bot, 'notifier') and bot.notifier:
            bot.notifier.send_message(f"❌ <b>فشل تشغيل البوت:</b>\n{str(e)}")


if __name__ == "__main__":
    main()
