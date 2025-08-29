import ccxt
import pandas as pd
import numpy as np
import time
import requests
import logging
import os
from datetime import datetime
import pytz 
import sys
import threading
from flask import Flask

DAMASCUS_TZ = pytz.timezone('Asia/Damascus')

# ------------------- Configuration from Environment Variables -------------------
COINEX_ACCESS_ID = os.environ.get('COINEX_ACCESS_ID')
COINEX_SECRET_KEY = os.environ.get('COINEX_SECRET_KEY')
TELEGRAM_BOT_TOKEN = os.environ.get('TELEGRAM_BOT_TOKEN')
TELEGRAM_CHAT_ID = os.environ.get('TELEGRAM_CHAT_ID')

# التحكم في التداول - تغيير إلى false لايقاف التداول
TRADING_ENABLED = os.environ.get('TRADING_ENABLED', 'true').lower() == 'true'

# إعدادات التداول لرأس مال 100 دولار
SYMBOL = 'ETH/USDT'
TOTAL_CAPITAL = 100  # رأس المال الإجمالي بالدولار
ORDER_SIZE = 10      # حجم كل صفقة (10 دولار)
MAX_EXPOSURE = 30    # أقصى تعرض صافي (30% من رأس المال)

# إعدادات المؤشرات
TIMEFRAME = '5m'
ATR_PERIOD = 14
STD_DEV_PERIOD = 14
ATR_MULTIPLIER = 1.8
STD_DEV_MULTIPLIER = 3

# إعدادات التحكم بالمخاطر
MAX_RETRIES = 3
RETRY_DELAY = 5

# ------------------- Initialize -------------------
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('eth_market_maker.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger()

# Initialize Coinex exchange
coinex = ccxt.coinex({
    'apiKey': COINEX_ACCESS_ID,
    'secret': COINEX_SECRET_KEY,
    'enableRateLimit': True,
})

# ------------------- Telegram Functions -------------------
def send_telegram_message(message):
    """Send message to Telegram"""
    try:
        url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"
        payload = {
            'chat_id': TELEGRAM_CHAT_ID,
            'text': message,
            'parse_mode': 'HTML'
        }
        response = requests.post(url, data=payload, timeout=10)
        return response.json()
    except Exception as e:
        logger.error(f"Failed to send Telegram message: {e}")

# ------------------- Enhanced API Functions -------------------
def robust_api_call(func, *args, **kwargs):
    """استدعاء API مع إعادة المحاولة عند الفشل"""
    for attempt in range(MAX_RETRIES):
        try:
            return func(*args, **kwargs)
        except (ccxt.NetworkError, ccxt.ExchangeError) as e:
            logger.warning(f"API call attempt {attempt+1} failed: {e}")
            if attempt < MAX_RETRIES - 1:
                time.sleep(RETRY_DELAY)
            else:
                raise e
        except Exception as e:
            logger.error(f"Unexpected error in API call: {e}")
            raise e

# ------------------- Trading Functions -------------------
def get_historical_data(limit=100):
    """Get historical OHLCV data"""
    try:
        ohlcv = robust_api_call(coinex.fetch_ohlcv, SYMBOL, TIMEFRAME, limit=limit)
        df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
        return df
    except Exception as e:
        logger.error(f"Error fetching historical data: {e}")
        send_telegram_message(f"❌ Error fetching data: {e}")
        return None

def calculate_indicators(df):
    """Calculate ATR and Standard Deviation"""
    try:
        if df is None or len(df) < ATR_PERIOD:
            return None
            
        # Calculate True Range
        df['prev_close'] = df['close'].shift(1)
        df['tr1'] = df['high'] - df['low']
        df['tr2'] = abs(df['high'] - df['prev_close'])
        df['tr3'] = abs(df['low'] - df['prev_close'])
        df['true_range'] = df[['tr1', 'tr2', 'tr3']].max(axis=1)
        
        # Calculate ATR
        df['atr'] = df['true_range'].rolling(window=ATR_PERIOD).mean()
        
        # Calculate price changes and Std Dev
        df['price_change_pct'] = df['close'].pct_change()
        df['std_dev'] = df['price_change_pct'].rolling(window=STD_DEV_PERIOD).std() * 100
        
        return df
    except Exception as e:
        logger.error(f"Error calculating indicators: {e}")
        return None

def get_current_price():
    """Get current market price"""
    try:
        ticker = robust_api_call(coinex.fetch_ticker, SYMBOL)
        return ticker['last']
    except Exception as e:
        logger.error(f"Error getting current price: {e}")
        send_telegram_message(f"❌ Error getting price: {e}")
        return None

def get_balance():
    """Get account balances"""
    try:
        balance = robust_api_call(coinex.fetch_balance)
        eth_balance = balance['ETH']['free'] if 'ETH' in balance else 0
        usdt_balance = balance['USDT']['free'] if 'USDT' in balance else 0
        return eth_balance, usdt_balance
    except Exception as e:
        logger.error(f"Error getting balance: {e}")
        send_telegram_message(f"❌ Error getting balance: {e}")
        return 0, 0

def cancel_all_orders():
    """Cancel all open orders for our symbol (الوظيفة الأصلية)"""
    try:
        orders = robust_api_call(coinex.fetch_open_orders, SYMBOL)
        for order in orders:
            try:
                robust_api_call(coinex.cancel_order, order['id'], SYMBOL)
                logger.info(f"Cancelled order {order['id']}")
                time.sleep(0.1)
            except Exception as e:
                logger.error(f"Error cancelling order {order['id']}: {e}")
    except Exception as e:
        logger.error(f"Error fetching orders: {e}")

def monitor_orders():
    """مراقبة الأوامر وإلغاء تلك التي لم تنجز بعد فترة زمنية (وظيفة جديدة)"""
    try:
        orders = robust_api_call(coinex.fetch_open_orders, SYMBOL)
        for order in orders:
            order_id = order['id']
            order_time = datetime.fromtimestamp(order['timestamp']/1000, DAMASCUS_TZ)
            time_diff = (datetime.now(DAMASCUS_TZ) - order_time).total_seconds() / 60
            
            # إلغاء الأوامر القديمة (أكثر من 30 دقيقة)
            if time_diff > 30:
                robust_api_call(coinex.cancel_order, order_id, SYMBOL)
                logger.info(f"Cancelled stale order {order_id} after {time_diff:.1f} minutes")
                send_telegram_message(f"⏹️ Cancelled stale order {order_id} after {time_diff:.1f} minutes")
                
    except Exception as e:
        logger.error(f"Error monitoring orders: {e}")

def calculate_order_distance(current_price, atr, std_dev):
    """Calculate dynamic order distance"""
    try:
        atr_distance = ATR_MULTIPLIER * atr
        std_dev_distance = current_price * (STD_DEV_MULTIPLIER * std_dev / 100)
        # إضافة حد أدنى وأقصى للمسافة
        min_distance = current_price * 0.002  # 0.2%
        max_distance = current_price * 0.01   # 1%
        return max(min(max(atr_distance, std_dev_distance), max_distance), min_distance)
    except:
        return current_price * 0.005  # 0.5% fallback

def calculate_position_sizing(current_price, eth_balance, usdt_balance):
    """Calculate optimal position sizing with auto-balancing"""
    current_exposure_usd = eth_balance * current_price
    total_balance = current_exposure_usd + usdt_balance
    
    # Calculate target ETH exposure (50% of total balance)
    target_eth_exposure = total_balance * 0.5
    current_eth_exposure = eth_balance * current_price
    
    # Determine if we need to buy or sell to rebalance
    if current_eth_exposure < target_eth_exposure - ORDER_SIZE:
        # Need to buy ETH
        buy_amount_usd = min(ORDER_SIZE, target_eth_exposure - current_eth_exposure)
        return 'buy', buy_amount_usd / current_price
    elif current_eth_exposure > target_eth_exposure + ORDER_SIZE:
        # Need to sell ETH
        sell_amount_eth = min(eth_balance, (current_eth_exposure - target_eth_exposure) / current_price)
        return 'sell', sell_amount_eth
    else:
        # Portfolio is balanced
        return 'balanced', 0

# ------------------- Performance Tracking -------------------
trade_history = []

def track_performance(order_type, amount, price, fee=0.001):
    """تتبع أداء الصفقات"""
    trade = {
        'timestamp': datetime.now(DAMASCUS_TZ),
        'type': order_type,
        'amount': amount,
        'price': price,
        'fee': fee,
        'total': amount * price
    }
    trade_history.append(trade)
    
    # حفظ الأداء في ملف كل 10 صفقات
    if len(trade_history) % 10 == 0:
        save_performance_report()

def save_performance_report():
    """حفظ تقرير الأداء في ملف"""
    try:
        df = pd.DataFrame(trade_history)
        df.to_csv('trading_performance.csv', index=False)
        logger.info("Performance report saved")
    except Exception as e:
        logger.error(f"Error saving performance report: {e}")

def place_orders():
    """Main function to place market making orders"""
    try:
        # التحقق إذا كان التداول موقفاً
        if not TRADING_ENABLED:
            # مراقبة فقط بدون تداول
            if datetime.now(DAMASCUS_TZ).minute % 30 == 0:  # إشعار كل 30 دقيقة
                current_price = get_current_price()
                eth_balance, usdt_balance = get_balance()
                if current_price:
                    exposure = eth_balance * current_price
                    send_telegram_message(
                        f"⏸️ Trading PAUSED | Price: {current_price} | "
                        f"Exposure: ${exposure:.1f} | USDT: {usdt_balance:.1f}"
                    )
            return
            
        # مراقبة وإلغاء الأوامر القديمة (مكملة وليست بديلة)
        monitor_orders()
            
        # Get market data
        df = get_historical_data(limit=100)
        if df is None or len(df) < ATR_PERIOD:
            logger.warning("Not enough data for indicators")
            return
        
        df = calculate_indicators(df)
        if df is None:
            return
        
        current_price = get_current_price()
        if current_price is None:
            return
        
        latest = df.iloc[-1]
        atr = latest['atr']
        std_dev = latest['std_dev']
        
        if pd.isna(atr) or pd.isna(std_dev):
            logger.warning("Indicator values are not available yet")
            return
        
        # Get balances and calculate exposure
        eth_balance, usdt_balance = get_balance()
        current_exposure_usd = eth_balance * current_price
        total_balance = current_exposure_usd + usdt_balance
        
        # إشعار الاختبار كل 15 دقيقة
        current_minute = datetime.now(DAMASCUS_TZ).minute
        if current_minute % 15 == 0 and current_second < 30:  # إشعار اختبار كل 15 دقيقة
            current_price = get_current_price()
            eth_balance, usdt_balance = get_balance()
    
            if current_price:
                exposure = eth_balance * current_price
                total_balance = exposure + usdt_balance
                test_msg = f"""
        🤖 <b>Bot Status - {datetime.now(DAMASCUS_TZ).strftime('%H:%M:%S')}</b>
        ━━━━━━━━━━━━━━━━━━━━
        📈 <b>Price:</b> {current_price} USDT
        💰 <b>ETH Balance:</b> {eth_balance:.4f} (${exposure:.1f})
        💵 <b>USDT Balance:</b> {usdt_balance:.1f}
        🏦 <b>Total Balance:</b> ${total_balance:.1f}
        🔄 <i>System operational - Next check in 15min</i>
                """
                send_telegram_message(test_msg)
                time.sleep(1)  # منع إرسال متعدد
        
        # الإشعار على رأس الساعة
        current_second = datetime.now(DAMASCUS_TZ).second
        if current_minute == 0 and current_second < 30:
            message = f"""
📊 <b>Hourly Report - {datetime.now(DAMASCUS_TZ).strftime('%H:%M')}</b>
━━━━━━━━━━━━━━━━━━━━
📈 <b>Price:</b> {current_price} USDT
💰 <b>ETH Balance:</b> {eth_balance:.4f} (${current_exposure_usd:.1f})
💵 <b>USDT Balance:</b> {usdt_balance:.1f}
🏦 <b>Total:</b> ${total_balance:.1f}
🕐 <i>Next update: {(datetime.now(DAMASCUS_TZ).hour + 1) % 24}:00</i>
            """
            send_telegram_message(message)
        
        # Check if within maximum exposure limits
        if current_exposure_usd > MAX_EXPOSURE:
            send_telegram_message("⚠️ Maximum exposure reached. No new buy orders.")
            return
        
        if usdt_balance < ORDER_SIZE and eth_balance < 0.001:
            send_telegram_message("⚠️ Low balance. Please deposit more funds.")
            return
        
        # Calculate order distance and prices
        order_distance = calculate_order_distance(current_price, atr, std_dev)
        buy_price = round(current_price - order_distance, 2)
        sell_price = round(current_price + order_distance, 2)
        
        # Cancel existing orders (الوظيفة الأصلية - تلغي الكل)
        cancel_all_orders()
        
        # Auto-balancing logic
        action, amount = calculate_position_sizing(current_price, eth_balance, usdt_balance)
        
        if action == 'buy' and usdt_balance >= ORDER_SIZE:
            # Place buy order
            buy_amount = amount
            if buy_amount * buy_price >= 5:  # Minimum order value check
                try:
                    order = robust_api_call(coinex.create_limit_buy_order, SYMBOL, buy_amount, buy_price)
                    send_telegram_message(f"✅ BUY: {buy_amount:.4f} ETH @ {buy_price} USDT")
                    track_performance('buy', buy_amount, buy_price)
                except Exception as e:
                    logger.error(f"Buy order error: {e}")
                    send_telegram_message(f"❌ Buy order failed: {e}")
        
        elif action == 'sell' and eth_balance >= 0.001:
            # Place sell order
            sell_amount = amount
            if sell_amount >= 0.001:  # Minimum order size check
                try:
                    order = robust_api_call(coinex.create_limit_sell_order, SYMBOL, sell_amount, sell_price)
                    send_telegram_message(f"✅ SELL: {sell_amount:.4f} ETH @ {sell_price} USDT")
                    track_performance('sell', sell_amount, sell_price)
                except Exception as e:
                    logger.error(f"Sell order error: {e}")
                    send_telegram_message(f"❌ Sell order failed: {e}")
        
    except Exception as e:
        logger.error(f"Error in place_orders: {e}")
        send_telegram_message(f"❌ Error in trading logic: {e}")

# ------------------- Web Service Setup -------------------
app = Flask(__name__)

@app.route('/')
def health_check():
    return "ETH Market Maker is running", 200

@app.route('/status')
def status():
    current_price = get_current_price()
    eth_balance, usdt_balance = get_balance()
    
    if current_price:
        exposure = eth_balance * current_price
        total = exposure + usdt_balance
    else:
        exposure = total = 0
        
    return {
        "status": "active",
        "trading_enabled": TRADING_ENABLED,
        "symbol": SYMBOL,
        "eth_balance": eth_balance,
        "usdt_balance": usdt_balance,
        "exposure_usd": exposure,
        "total_balance": total,
        "last_update": datetime.now(DAMASCUS_TZ).isoformat()
    }, 200

@app.route('/performance')
def performance():
    try:
        df = pd.DataFrame(trade_history)
        return df.to_json(orient='records'), 200, {'Content-Type': 'application/json'}
    except:
        return {"message": "No performance data available"}, 404

def run_flask_app():
    """تشغيل Flask app للـ health checks"""
    port = int(os.environ.get('PORT', 10000))
    app.run(host='0.0.0.0', port=port, debug=False, use_reloader=False)

def main():
    """الدالة الرئيسية المعدلة للعمل كـ Web Service"""
    try:
        # بدء Flask في thread منفصل
        flask_thread = threading.Thread(target=run_flask_app, daemon=True)
        flask_thread.start()
        
        # التحقق من المتغيرات البيئية
        required_env_vars = ['COINEX_ACCESS_ID', 'COINEX_SECRET_KEY', 'TELEGRAM_BOT_TOKEN', 'TELEGRAM_CHAT_ID']
        missing_vars = [var for var in required_env_vars if not os.environ.get(var)]
        
        if missing_vars:
            error_msg = f"❌ Missing environment variables: {', '.join(missing_vars)}"
            print(error_msg)
            sys.exit(1)
        
        # إرسال رسالة البدء
        initial_status = "ENABLED" if TRADING_ENABLED else "PAUSED"
        start_message = f"""
🚀 <b>ETH Market Maker Started as Web Service!</b>
━━━━━━━━━━━━━━━━━━━━
📊 <b>Status:</b> {initial_status}
💼 <b>Capital:</b> ${TOTAL_CAPITAL}
📦 <b>Order Size:</b> ${ORDER_SIZE}
🌐 <b>Health:</b> http://localhost:{os.environ.get('PORT', 10000)}/
🕐 <i>Start Time: {datetime.now(DAMASCUS_TZ).strftime('%Y-%m-%d %H:%M:%S')}</i>
        """
        
        send_telegram_message(start_message)
        logger.info(f"Bot started as Web Service. Trading enabled: {TRADING_ENABLED}")
        
        # الحلقة الرئيسية
        while True:
            try:
                place_orders()
                time.sleep(300)  # 5 دقائق
                
            except KeyboardInterrupt:
                logger.info("Bot stopped by user")
                send_telegram_message("⏹️ Bot stopped manually")
                break
            except Exception as e:
                logger.error(f"Error in main loop: {e}")
                time.sleep(60)
                
    except Exception as e:
        error_msg = f"❌ FATAL ERROR: {str(e)}"
        logger.error(error_msg)
        send_telegram_message(error_msg[:4000])
        sys.exit(1)

if __name__ == "__main__":
    main()

