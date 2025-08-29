import ccxt
import pandas as pd
import numpy as np
import time
import requests
import logging
import os
from datetime import datetime, timedelta
import pytz 
import sys
import threading
from flask import Flask, jsonify
import ta
from typing import Dict, List, Optional, Tuple, Union

app = Flask(__name__)

from dotenv import load_dotenv

# تحميل متغيرات البيئة
load_dotenv()

# ------------------- Configuration -------------------
TREND_FILTER_PERIOD = 50
TREND_THRESHOLD = 0.0015
DAMASCUS_TZ = pytz.timezone('Asia/Damascus')

# Environment Variables
COINEX_ACCESS_ID = os.environ.get('COINEX_ACCESS_ID')
COINEX_SECRET_KEY = os.environ.get('COINEX_SECRET_KEY')
TELEGRAM_BOT_TOKEN = os.environ.get('TELEGRAM_BOT_TOKEN')
TELEGRAM_CHAT_ID = os.environ.get('TELEGRAM_CHAT_ID')
TRADING_ENABLED = os.environ.get('TRADING_ENABLED', 'true').lower() == 'true'
PAPER_TRADING = os.environ.get('PAPER_TRADING', 'false').lower() == 'true'

# Trading Parameters
SYMBOL = 'ETH/USDT'
BASE_CURRENCY = 'ETH'
QUOTE_CURRENCY = 'USDT'
TOTAL_CAPITAL = 100
ORDER_SIZE_PERCENT = 0.1  # 10% of capital per trade
MAX_EXPOSURE_PERCENT = 0.3  # 30% maximum exposure
STOP_LOSS_PERCENT = 0.02  # 2% stop loss
TAKE_PROFIT_PERCENT = 0.03  # 3% take profit
TRAILING_STOP_PERCENT = 0.01  # 1% trailing stop

# Indicator Settings
TIMEFRAME = '5m'
ATR_PERIOD = 14
RSI_PERIOD = 14
MACD_FAST = 12
MACD_SLOW = 26
MACD_SIGNAL = 9
ADX_PERIOD = 14
ADX_THRESHOLD = 25
BOLLINGER_PERIOD = 20
BOLLINGER_STD = 2
ICHIMOKU_PERIOD = 9, 26, 52  # Tenkan, Kijun, Senkou

# Risk Management
MAX_RETRIES = 3
RETRY_DELAY = 2
MAX_DAILY_LOSS = 0.05  # 5% maximum daily loss
MAX_POSITIONS = 3  # Maximum simultaneous positions

# ------------------- Initialize -------------------
class UnicodeEscapeHandler(logging.StreamHandler):
    def emit(self, record):
        try:
            msg = self.format(record)
            stream = self.stream
            msg = msg.encode('utf-8', 'replace').decode('utf-8')
            stream.write(msg + self.terminator)
            self.flush()
        except Exception:
            self.handleError(record)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('eth_market_maker_v3.log', encoding='utf-8'),
        UnicodeEscapeHandler()
    ]
)
logger = logging.getLogger()

# Initialize exchange
coinex = ccxt.coinex({
    'apiKey': COINEX_ACCESS_ID,
    'secret': COINEX_SECRET_KEY,
    'enableRateLimit': True,
    'options': {
        'defaultType': 'spot',
        'adjustForTimeDifference': True,
    },
    'timeout': 30000,
    'verbose': False,
})

# ------------------- Global Variables -------------------
trade_history = []
daily_pnl = 0
daily_starting_balance = 0
open_positions = {}
position_counter = 0
PAPER_BALANCE = {'ETH': 0.5, 'USDT': 50.0}
last_trade_time = None
trade_cooldown = 300  # 5 minutes cooldown between trades

# ------------------- Technical Indicators -------------------
def calculate_indicators(df: pd.DataFrame) -> Optional[pd.DataFrame]:
    """Calculate all technical indicators"""
    try:
        if df is None or len(df) < 100:
            return None
            
        # Use TA-Lib for more accurate calculations
        df = df.copy()
        
        # ATR
        df['atr'] = ta.ATR(df['high'], df['low'], df['close'], timeperiod=ATR_PERIOD)
        
        # RSI
        df['rsi'] = ta.RSI(df['close'], timeperiod=RSI_PERIOD)
        
        # MACD
        macd, macd_signal, macd_hist = ta.MACD(
            df['close'], fastperiod=MACD_FAST, 
            slowperiod=MACD_SLOW, signalperiod=MACD_SIGNAL
        )
        df['macd'] = macd
        df['macd_signal'] = macd_signal
        df['macd_hist'] = macd_hist
        
        # ADX
        df['adx'] = ta.ADX(df['high'], df['low'], df['close'], timeperiod=ADX_PERIOD)
        
        # Bollinger Bands
        df['bb_upper'], df['bb_middle'], df['bb_lower'] = ta.BBANDS(
            df['close'], timeperiod=BOLLINGER_PERIOD, 
            nbdevup=BOLLINGER_STD, nbdevdn=BOLLINGER_STD
        )
        
        # Ichimoku Cloud
        tenkan_sen = (df['high'].rolling(9).max() + df['low'].rolling(9).min()) / 2
        kijun_sen = (df['high'].rolling(26).max() + df['low'].rolling(26).min()) / 2
        senkou_span_a = ((tenkan_sen + kijun_sen) / 2).shift(26)
        senkou_span_b = ((df['high'].rolling(52).max() + df['low'].rolling(52).min()) / 2).shift(26)
        
        df['ichimoku_tenkan'] = tenkan_sen
        df['ichimoku_kijun'] = kijun_sen
        df['ichimoku_senkou_a'] = senkou_span_a
        df['ichimoku_senkou_b'] = senkou_span_b
        
        # Momentum
        df['momentum'] = df['close'].pct_change(5)
        df['volume_ma'] = df['volume'].rolling(20).mean()
        df['volume_ratio'] = df['volume'] / df['volume_ma']
        
        # Price action signals
        df['candle_size'] = (df['high'] - df['low']) / df['low']
        df['body_ratio'] = abs(df['close'] - df['open']) / (df['high'] - df['low']).replace(0, 0.001)
        
        return df.dropna()
        
    except Exception as e:
        logger.error(f"Error calculating indicators: {e}")
        return None

# ------------------- Enhanced API Functions -------------------
def robust_api_call(func, *args, **kwargs):
    """API call with retry logic and error handling"""
    for attempt in range(MAX_RETRIES):
        try:
            result = func(*args, **kwargs)
            time.sleep(0.1)  # Rate limiting
            return result
        except (ccxt.NetworkError, ccxt.ExchangeError, requests.exceptions.RequestException) as e:
            logger.warning(f"API call attempt {attempt+1} failed: {e}")
            if attempt < MAX_RETRIES - 1:
                time.sleep(RETRY_DELAY * (attempt + 1))
            else:
                raise e
        except Exception as e:
            logger.error(f"Unexpected error in API call: {e}")
            raise e

# ------------------- Data Functions -------------------
def get_historical_data(limit: int = 200, timeframe: str = None) -> Optional[pd.DataFrame]:
    """Get historical OHLCV data with caching"""
    try:
        tf = timeframe or TIMEFRAME
        ohlcv = robust_api_call(coinex.fetch_ohlcv, SYMBOL, tf, limit=limit)
        
        df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
        df.set_index('timestamp', inplace=True)
        
        return df
    except Exception as e:
        logger.error(f"Error fetching historical data: {e}")
        return None

def get_current_price() -> Optional[float]:
    """Get current market price with fallback"""
    try:
        ticker = robust_api_call(coinex.fetch_ticker, SYMBOL)
        return float(ticker['last'])
    except Exception as e:
        logger.error(f"Error getting current price: {e}")
        return None

def get_balance() -> Tuple[float, float]:
    """Get account balances with error handling"""
    try:
        if PAPER_TRADING:
            return PAPER_BALANCE['ETH'], PAPER_BALANCE['USDT']
            
        balance = robust_api_call(coinex.fetch_balance)
        eth_balance = balance.get(BASE_CURRENCY, {}).get('free', 0) or 0
        usdt_balance = balance.get(QUOTE_CURRENCY, {}).get('free', 0) or 0
        return float(eth_balance), float(usdt_balance)
    except Exception as e:
        logger.error(f"Error getting balance: {e}")
        return 0.0, 0.0

# ------------------- Telegram Functions -------------------
def send_telegram_message(message: str) -> bool:
    """Send message to Telegram with formatting"""
    try:
        if not TELEGRAM_BOT_TOKEN or not TELEGRAM_CHAT_ID:
            return False
            
        url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"
        payload = {
            'chat_id': TELEGRAM_CHAT_ID,
            'text': message,
            'parse_mode': 'HTML',
            'disable_web_page_preview': True
        }
        
        response = requests.post(url, json=payload, timeout=10)
        return response.status_code == 200
    except Exception as e:
        logger.error(f"Failed to send Telegram message: {e}")
        return False

# ------------------- Order Management -------------------
def cancel_all_orders():
    """Cancel all open orders safely"""
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
    """Monitor and cancel stale orders"""
    try:
        orders = robust_api_call(coinex.fetch_open_orders, SYMBOL)
        for order in orders:
            order_time = datetime.fromtimestamp(order['timestamp']/1000, DAMASCUS_TZ)
            time_diff = (datetime.now(DAMASCUS_TZ) - order_time).total_seconds() / 60
            
            if time_diff > 10:  # Cancel orders older than 10 minutes
                robust_api_call(coinex.cancel_order, order['id'], SYMBOL)
                logger.info(f"Cancelled stale order {order['id']}")
                
    except Exception as e:
        logger.error(f"Error monitoring orders: {e}")

# ------------------- Risk Management -------------------
def check_daily_loss_limit() -> bool:
    """Check if daily loss limit is exceeded"""
    global daily_pnl
    if daily_pnl <= -MAX_DAILY_LOSS * daily_starting_balance:
        send_telegram_message(f"🚫 Daily loss limit exceeded! PnL: ${daily_pnl:.2f}")
        return True
    return False

def calculate_position_size(current_price: float, total_balance: float, 
                          volatility_factor: float = 1.0) -> float:
    """Calculate dynamic position size based on multiple factors"""
    try:
        # Get volatility data
        df = get_historical_data(limit=50)
        if df is not None and len(df) > 20:
            returns = df['close'].pct_change().dropna()
            volatility = returns.std() * np.sqrt(365)
            
            # Adjust size based on volatility
            if volatility > 0.8:  # High volatility
                size_multiplier = 0.5
            elif volatility < 0.2:  # Low volatility
                size_multiplier = 1.2
            else:
                size_multiplier = 1.0
        else:
            size_multiplier = 1.0
            
        # Base size calculation
        base_size = total_balance * ORDER_SIZE_PERCENT * size_multiplier * volatility_factor
        
        # Apply maximum exposure limit
        max_size = total_balance * MAX_EXPOSURE_PERCENT
        
        # Minimum order size check
        min_order_size = 5.0  # $5 minimum
        calculated_size = min(base_size, max_size)
        
        return max(calculated_size, min_order_size)
        
    except Exception as e:
        logger.error(f"Error calculating position size: {e}")
        return total_balance * ORDER_SIZE_PERCENT

# ------------------- Trading Signals -------------------
def generate_trading_signals(df: pd.DataFrame) -> Optional[Dict]:
    """Generate comprehensive trading signals"""
    if df is None or len(df) < 50:
        return None
    
    latest = df.iloc[-1]
    prev = df.iloc[-2]
    
    # Calculate signal strengths
    signals = {
        # Trend signals
        'trend_strength': latest.get('adx', 0),
        'trend_direction': 'bullish' if latest['close'] > latest['close'].shift(20).mean() else 'bearish',
        
        # MACD signals
        'macd_bullish': latest.get('macd', 0) > latest.get('macd_signal', 0) and prev.get('macd', 0) <= prev.get('macd_signal', 0),
        'macd_bearish': latest.get('macd', 0) < latest.get('macd_signal', 0) and prev.get('macd', 0) >= prev.get('macd_signal', 0),
        'macd_hist_bullish': latest.get('macd_hist', 0) > 0 and prev.get('macd_hist', 0) <= 0,
        'macd_hist_bearish': latest.get('macd_hist', 0) < 0 and prev.get('macd_hist', 0) >= 0,
        
        # RSI signals
        'rsi': latest.get('rsi', 50),
        'rsi_overbought': latest.get('rsi', 50) > 70,
        'rsi_oversold': latest.get('rsi', 50) < 30,
        'rsi_divergence': check_rsi_divergence(df),
        
        # Bollinger Bands signals
        'price_above_bb': latest['close'] > latest.get('bb_upper', latest['close']),
        'price_below_bb': latest['close'] < latest.get('bb_lower', latest['close']),
        'price_middle_bb': latest.get('bb_middle', 0) is not None and 
                          abs(latest['close'] - latest['bb_middle']) / latest['bb_middle'] < 0.01,
        
        # Ichimoku signals
        'ichimoku_bullish': latest['close'] > latest.get('ichimoku_tenkan', 0) and 
                           latest['close'] > latest.get('ichimoku_kijun', 0),
        'ichimoku_bearish': latest['close'] < latest.get('ichimoku_tenkan', 0) and 
                           latest['close'] < latest.get('ichimoku_kijun', 0),
        'cloud_bullish': latest.get('ichimoku_senkou_a', 0) > latest.get('ichimoku_senkou_b', 0),
        'cloud_bearish': latest.get('ichimoku_senkou_a', 0) < latest.get('ichimoku_senkou_b', 0),
        
        # Volume signals
        'volume_spike': latest.get('volume_ratio', 1) > 1.5,
        'volume_low': latest.get('volume_ratio', 1) < 0.5,
        
        # Price action
        'candle_size': latest.get('candle_size', 0),
        'body_ratio': latest.get('body_ratio', 0),
        'momentum': latest.get('momentum', 0),
    }
    
    return signals

def check_rsi_divergence(df: pd.DataFrame, period: int = 14) -> str:
    """Check for RSI divergence"""
    if len(df) < period * 2:
        return "none"
        
    # Simple divergence check
    price_high = df['close'].rolling(period).max()
    price_low = df['close'].rolling(period).min()
    rsi_high = df['rsi'].rolling(period).max()
    rsi_low = df['rsi'].rolling(period).min()
    
    # Bullish divergence: price makes lower low, RSI makes higher low
    if (df['close'].iloc[-1] < price_low.iloc[-2] and 
        df['rsi'].iloc[-1] > rsi_low.iloc[-2]):
        return "bullish"
        
    # Bearish divergence: price makes higher high, RSI makes lower high
    if (df['close'].iloc[-1] > price_high.iloc[-2] and 
        df['rsi'].iloc[-1] < rsi_high.iloc[-2]):
        return "bearish"
        
    return "none"

def should_enter_trade(signals: Dict, current_price: float, df: pd.DataFrame) -> Optional[str]:
    """Determine if we should enter a trade with improved logic"""
    if not signals:
        return None
        
    # Avoid trading in strong trends
    if signals['trend_strength'] > ADX_THRESHOLD:
        return None
        
    # Check trade cooldown
    global last_trade_time
    if last_trade_time and (datetime.now() - last_trade_time).total_seconds() < trade_cooldown:
        return None
        
    # Check maximum positions
    if len(open_positions) >= MAX_POSITIONS:
        return None
        
    # Calculate buy and sell scores with weighted system
    buy_score = 0
    sell_score = 0
    
    # Buy signals with weights
    if signals.get('macd_bullish'): buy_score += 1.5
    if signals.get('macd_hist_bullish'): buy_score += 1.0
    if signals.get('rsi_oversold'): 
        buy_score += 2.0 if signals['rsi'] < 25 else 1.0
    if signals.get('price_below_bb'): buy_score += 1.0
    if signals.get('ichimoku_bullish'): buy_score += 1.0
    if signals.get('cloud_bullish'): buy_score += 0.5
    if signals.get('rsi_divergence') == 'bullish': buy_score += 1.5
    if signals.get('volume_spike') and signals['momentum'] > 0: buy_score += 0.5
    
    # Sell signals with weights
    if signals.get('macd_bearish'): sell_score += 1.5
    if signals.get('macd_hist_bearish'): sell_score += 1.0
    if signals.get('rsi_overbought'): 
        sell_score += 2.0 if signals['rsi'] > 75 else 1.0
    if signals.get('price_above_bb'): sell_score += 1.0
    if signals.get('ichimoku_bearish'): sell_score += 1.0
    if signals.get('cloud_bearish'): sell_score += 0.5
    if signals.get('rsi_divergence') == 'bearish': sell_score += 1.5
    if signals.get('volume_spike') and signals['momentum'] < 0: sell_score += 0.5
    
    # Filter out small candle signals
    if signals.get('candle_size', 0) < 0.002:
        return None
        
    # Check market trend alignment
    market_trend = check_market_trend(df)
    
    # Require minimum score and clear signal strength
    min_score = 2.5
    score_diff = 0.8
    
    if buy_score >= min_score and buy_score - sell_score >= score_diff:
        if market_trend in ["bullish", "neutral"]:
            return 'buy'
            
    elif sell_score >= min_score and sell_score - buy_score >= score_diff:
        if market_trend in ["bearish", "neutral"]:
            return 'sell'
            
    return None

def check_market_trend(df: pd.DataFrame, short_period: int = 20, long_period: int = 50) -> str:
    """Check overall market trend with multiple timeframes"""
    if len(df) < long_period:
        return "neutral"
        
    # Multiple timeframe analysis
    sma_short = df['close'].rolling(short_period).mean().iloc[-1]
    sma_long = df['close'].rolling(long_period).mean().iloc[-1]
    price_vs_sma = (df['close'].iloc[-1] - sma_long) / sma_long
    
    # Ichimoku trend
    tenkan = df['ichimoku_tenkan'].iloc[-1] if 'ichimoku_tenkan' in df.columns else sma_short
    kijun = df['ichimoku_kijun'].iloc[-1] if 'ichimoku_kijun' in df.columns else sma_long
    
    if (df['close'].iloc[-1] > tenkan > kijun and price_vs_sma > 0.01):
        return "bullish"
    elif (df['close'].iloc[-1] < tenkan < kijun and price_vs_sma < -0.01):
        return "bearish"
    else:
        return "neutral"

# ------------------- Trade Execution -------------------
def execute_trade(signal: str, current_price: float, amount_usd: float, df: pd.DataFrame) -> bool:
    """Execute trade with improved risk management"""
    try:
        amount = amount_usd / current_price
        
        # Minimum order size check
        min_order_size = 0.001
        if amount < min_order_size:
            logger.warning(f"Order size too small: {amount:.6f} ETH")
            return False
        
        # Smart price calculation
        spread = 0.001  # 0.1% spread
        
        if signal == 'buy':
            entry_price = current_price * (1 - spread)
            stop_loss_price = entry_price * (1 - STOP_LOSS_PERCENT)
            take_profit_price = entry_price * (1 + TAKE_PROFIT_PERCENT)
            
            if TRADING_ENABLED and not PAPER_TRADING:
                order_id = place_limit_order('buy', amount, entry_price)
                if not order_id:
                    return False
            elif PAPER_TRADING:
                paper_trade('buy', amount, entry_price)
                
        else:  # sell
            entry_price = current_price * (1 + spread)
            stop_loss_price = entry_price * (1 + STOP_LOSS_PERCENT)
            take_profit_price = entry_price * (1 - TAKE_PROFIT_PERCENT)
            
            if TRADING_ENABLED and not PAPER_TRADING:
                order_id = place_limit_order('sell', amount, entry_price)
                if not order_id:
                    return False
            elif PAPER_TRADING:
                paper_trade('sell', amount, entry_price)
        
        # Store position information
        global position_counter
        position_id = f"{signal}_{position_counter}"
        position_counter += 1
        
        open_positions[position_id] = {
            'type': signal,
            'entry_price': entry_price,
            'amount': amount,
            'stop_loss': stop_loss_price,
            'take_profit': take_profit_price,
            'trailing_stop': stop_loss_price,
            'timestamp': datetime.now(DAMASCUS_TZ),
            'status': 'open'
        }
        
        # Update last trade time
        global last_trade_time
        last_trade_time = datetime.now()
        
        # Send notification
        message = f"✅ {signal.upper()}: {amount:.4f} ETH @ {entry_price:.2f} USDT\n"
        message += f"SL: {stop_loss_price:.2f} | TP: {take_profit_price:.2f}\n"
        message += f"Size: ${amount_usd:.2f} | Risk: {STOP_LOSS_PERCENT*100:.1f}%"
        
        send_telegram_message(message)
        track_performance(signal, amount, entry_price)
        
        return True
        
    except Exception as e:
        logger.error(f"Trade execution error: {e}")
        send_telegram_message(f"❌ Trade failed: {e}")
        return False

def place_limit_order(order_type: str, amount: float, price: float) -> Optional[str]:
    """Place a limit order with proper error handling"""
    try:
        if order_type == 'buy':
            order = robust_api_call(coinex.create_limit_buy_order, SYMBOL, amount, price)
        else:
            order = robust_api_call(coinex.create_limit_sell_order, SYMBOL, amount, price)
        
        return order['id']
    except Exception as e:
        logger.error(f"Error placing limit order: {e}")
        return None

# ------------------- Paper Trading -------------------
def paper_trade(action: str, amount: float, price: float):
    """Execute paper trade without real risk"""
    try:
        fee = amount * price * 0.001  # 0.1% fee
        
        if action == 'buy':
            cost = amount * price + fee
            if PAPER_BALANCE['USDT'] >= cost:
                PAPER_BALANCE['USDT'] -= cost
                PAPER_BALANCE['ETH'] += amount
                logger.info(f"📘 PAPER BUY: {amount:.4f} ETH @ {price:.2f} = ${cost:.2f}")
            else:
                logger.warning("📘 PAPER BUY: Insufficient USDT balance")
                
        elif action == 'sell':
            if PAPER_BALANCE['ETH'] >= amount:
                revenue = amount * price - fee
                PAPER_BALANCE['ETH'] -= amount
                PAPER_BALANCE['USDT'] += revenue
                logger.info(f"📕 PAPER SELL: {amount:.4f} ETH @ {price:.2f} = ${revenue:.2f}")
            else:
                logger.warning("📕 PAPER SELL: Insufficient ETH balance")
                
    except Exception as e:
        logger.error(f"Paper trade error: {e}")

# ------------------- Position Management -------------------
def monitor_positions(current_price: float):
    """Monitor and manage open positions with trailing stops"""
    global daily_pnl, open_positions
    
    positions_to_close = []
    
    for position_id, position in list(open_positions.items()):
        if position['status'] != 'open':
            continue
            
        pnl_percent = 0
        current_pnl = 0
        
        if position['type'] == 'buy':
            pnl_percent = (current_price - position['entry_price']) / position['entry_price']
            current_pnl = position['amount'] * (current_price - position['entry_price'])
            
            # Update trailing stop
            if current_price > position['entry_price'] * (1 + 0.005):  # 0.5% move
                new_stop = current_price * (1 - TRAILING_STOP_PERCENT)
                if new_stop > position['trailing_stop']:
                    position['trailing_stop'] = new_stop
            
            # Check exit conditions
            if (current_price <= position['trailing_stop'] or 
                current_price >= position['take_profit'] or
                current_price <= position['stop_loss']):
                positions_to_close.append((position_id, position, pnl_percent, current_pnl))
                
        else:  # sell position
            pnl_percent = (position['entry_price'] - current_price) / position['entry_price']
            current_pnl = position['amount'] * (position['entry_price'] - current_price)
            
            # Update trailing stop
            if current_price < position['entry_price'] * (1 - 0.005):  # 0.5% move
                new_stop = current_price * (1 + TRAILING_STOP_PERCENT)
                if new_stop < position['trailing_stop']:
                    position['trailing_stop'] = new_stop
            
            if (current_price >= position['trailing_stop'] or 
                current_price <= position['take_profit'] or
                current_price >= position['stop_loss']):
                positions_to_close.append((position_id, position, pnl_percent, current_pnl))
    
    # Close positions
    for position_id, position, pnl_percent, current_pnl in positions_to_close:
        close_position(position_id, position, current_price, pnl_percent, current_pnl)

def close_position(position_id: str, position: Dict, current_price: float, 
                  pnl_percent: float, current_pnl: float):
    """Close a position and update tracking"""
    try:
        if position['type'] == 'buy':
            if TRADING_ENABLED and not PAPER_TRADING:
                robust_api_call(coinex.create_market_sell_order, SYMBOL, position['amount'])
            elif PAPER_TRADING:
                paper_trade('sell', position['amount'], current_price)
        else:
            if TRADING_ENABLED and not PAPER_TRADING:
                robust_api_call(coinex.create_market_buy_order, SYMBOL, position['amount'])
            elif PAPER_TRADING:
                paper_trade('buy', position['amount'], current_price)
        
        # Update daily PnL
        global daily_pnl
        daily_pnl += current_pnl
        
        # Update position status
        position['status'] = 'closed'
        position['exit_price'] = current_price
        position['exit_time'] = datetime.now(DAMASCUS_TZ)
        position['pnl'] = current_pnl
        position['pnl_percent'] = pnl_percent
        
        # Send notification
        emoji = "✅" if current_pnl > 0 else "❌"
        message = f"{emoji} Position closed: {position['type'].upper()}\n"
        message += f"P&L: {pnl_percent*100:+.2f}% (${current_pnl:+.2f})\n"
        message += f"Entry: {position['entry_price']:.2f} | Exit: {current_price:.2f}\n"
        message += f"Duration: {(datetime.now(DAMASCUS_TZ) - position['timestamp']).total_seconds()/60:.1f} min"
        
        send_telegram_message(message)
        
    except Exception as e:
        logger.error(f"Error closing position {position_id}: {e}")

# ------------------- Performance Tracking -------------------
def track_performance(order_type: str, amount: float, price: float, fee: float = 0.001):
    """Track trade performance"""
    trade = {
        'timestamp': datetime.now(DAMASCUS_TZ),
        'type': order_type,
        'amount': amount,
        'price': price,
        'fee': fee,
        'total': amount * price
    }
    trade_history.append(trade)
    
    # Save performance periodically
    if len(trade_history) % 10 == 0:
        save_performance_report()

def save_performance_report():
    """Save performance report to file"""
    try:
        if trade_history:
            df = pd.DataFrame(trade_history)
            df.to_csv('trading_performance_v3.csv', index=False)
            logger.info("Performance report saved")
    except Exception as e:
        logger.error(f"Error saving performance report: {e}")

# ------------------- Market Analysis -------------------
def check_market_conditions() -> bool:
    """Check overall market conditions before trading"""
    try:
        # Multi-timeframe analysis
        timeframes = ['15m', '1h', '4h']
        bearish_count = 0
        
        for tf in timeframes:
            df = get_historical_data(limit=100, timeframe=tf)
            if df is not None:
                trend = check_market_trend(df)
                if trend == "bearish":
                    bearish_count += 1
        
        # Avoid trading if multiple timeframes show bearish trend
        if bearish_count >= 2:
            logger.info("Market conditions not favorable - multiple bearish timeframes")
            return False
            
        return True
        
    except Exception as e:
        logger.error(f"Error checking market conditions: {e}")
        return True

def calculate_volatility(df: pd.DataFrame, period: int = 20) -> float:
    """Calculate market volatility based on ATR"""
    if len(df) < period or 'atr' not in df.columns:
        return 0
    
    atr = df['atr'].iloc[-1]
    price = df['close'].iloc[-1]
    
    if price > 0:
        return atr / price
    return 0

# ------------------- Main Trading Logic -------------------
def trading_cycle():
    """Main trading cycle with improved logic"""
    global daily_pnl, daily_starting_balance
    
    try:
        # Initialize daily tracking at midnight
        now = datetime.now(DAMASCUS_TZ)
        if now.hour == 0 and now.minute < 5:
            eth_balance, usdt_balance = get_balance()
            current_price = get_current_price()
            if current_price:
                daily_starting_balance = (eth_balance * current_price) + usdt_balance
                daily_pnl = 0
                send_telegram_message(f"📊 New trading day started. Balance: ${daily_starting_balance:.2f}")
        
        # Check daily loss limit
        if check_daily_loss_limit():
            logger.warning("Daily loss limit exceeded, pausing trading")
            time.sleep(600)
            return
        
        # Check market conditions
        if not check_market_conditions():
            logger.info("Market conditions not favorable for trading")
            time.sleep(300)
            return
        
        # Get market data
        df = get_historical_data(limit=200)
        if df is None:
            logger.warning("Failed to get historical data")
            time.sleep(60)
            return
            
        df = calculate_indicators(df)
        if df is None:
            logger.warning("Failed to calculate indicators")
            time.sleep(60)
            return
            
        current_price = get_current_price()
        if current_price is None:
            logger.warning("Failed to get current price")
            time.sleep(60)
            return
            
        # Monitor and manage existing positions
        monitor_positions(current_price)
        
        # Generate trading signals
        signals = generate_trading_signals(df)
        if signals is None:
            logger.warning("Failed to generate signals")
            time.sleep(60)
            return
            
        # Get current balance
        eth_balance, usdt_balance = get_balance()
        total_balance = (eth_balance * current_price) + usdt_balance
        
        # Calculate position size with volatility adjustment
        volatility = calculate_volatility(df)
        volatility_factor = 1.0 if volatility < 0.5 else 0.7  # Reduce size in high volatility
        position_size = calculate_position_size(current_price, total_balance, volatility_factor)
        
        # Check if we should enter a trade
        signal = should_enter_trade(signals, current_price, df)
        
        # Execute trade if conditions are met
        if signal and position_size >= 5:
            if volatility < 0.7:  # Avoid trading in extremely high volatility
                execute_trade(signal, current_price, position_size, df)
            else:
                logger.info(f"High volatility detected ({volatility:.2f}), skipping trade")
        
        # Send status update every hour
        if now.minute == 0:
            send_status_update(current_price, eth_balance, usdt_balance, signals)
            
        # Monitor and cancel stale orders
        monitor_orders()
            
    except Exception as e:
        logger.error(f"Error in trading cycle: {e}")
        send_telegram_message(f"❌ Trading cycle error: {e}")
        time.sleep(60)

def send_status_update(current_price: float, eth_balance: float, 
                      usdt_balance: float, signals: Dict):
    """Send status update to Telegram"""
    exposure = eth_balance * current_price
    total_balance = exposure + usdt_balance
    
    message = f"""
📊 <b>Hourly Status - {datetime.now(DAMASCUS_TZ).strftime('%H:%M')}</b>
━━━━━━━━━━━━━━━━━━━━
💰 <b>Price:</b> {current_price:.2f} USDT
🏦 <b>Total Balance:</b> ${total_balance:.2f}
📈 <b>Exposure:</b> ${exposure:.2f}
📉 <b>Daily P&L:</b> ${daily_pnl:.2f}
🔍 <b>Signals:</b> ADX: {signals.get('trend_strength', 0):.1f}, RSI: {signals.get('rsi', 0):.1f}
🔄 <b>Open Positions:</b> {len(open_positions)}
⚡ <b>Trend:</b> {signals.get('trend_direction', 'neutral')}
    """
    
    send_telegram_message(message)

# ------------------- Backtesting -------------------
def backtest_strategy():
    """Backtest strategy with improved accuracy"""
    logger.info("Starting backtest...")
    send_telegram_message("Starting historical backtest...")

    try:
        # Get sufficient historical data
        df = get_historical_data(limit=1000)
        if df is None or len(df) < 100:
            raise ValueError("Insufficient data for backtesting")
            
        df = calculate_indicators(df)
        if df is None:
            raise ValueError("Failed to calculate indicators")

        # Backtest settings
        initial_eth = 0.5
        initial_usdt = 50.0
        fee_rate = 0.001
        
        # Track balances
        eth_balance = initial_eth
        usdt_balance = initial_usdt
        trades = []
        equity_curve = []
        
        # Run backtest
        for i in range(100, len(df)):
            current_data = df.iloc[:i+1]
            current_price = df['close'].iloc[i]
            
            # Generate signals
            signals = generate_trading_signals(current_data)
            if not signals:
                continue
                
            # Check for trade signal
            signal = should_enter_trade(signals, current_price, current_data)
            
            # Calculate position size
            current_equity = eth_balance * current_price + usdt_balance
            position_size = calculate_position_size(current_price, current_equity)
            
            # Execute trade if signal exists
            if signal and position_size >= 5:
                amount = position_size / current_price
                fee = position_size * fee_rate
                
                if signal == 'buy' and usdt_balance >= position_size + fee:
                    usdt_balance -= (position_size + fee)
                    eth_balance += amount
                    
                    trades.append({
                        'timestamp': df.index[i],
                        'type': 'buy',
                        'price': current_price,
                        'amount': amount,
                        'value': position_size,
                        'fee': fee
                    })
                    
                elif signal == 'sell' and eth_balance >= amount:
                    revenue = amount * current_price
                    fee = revenue * fee_rate
                    eth_balance -= amount
                    usdt_balance += (revenue - fee)
                    
                    trades.append({
                        'timestamp': df.index[i],
                        'type': 'sell',
                        'price': current_price,
                        'amount': amount,
                        'value': revenue,
                        'fee': fee
                    })
            
            # Track equity curve
            equity_curve.append(eth_balance * current_price + usdt_balance)
        
        # Calculate final results
        final_price = df['close'].iloc[-1]
        final_equity = eth_balance * final_price + usdt_balance
        initial_equity = initial_eth * df['close'].iloc[100] + initial_usdt
        total_profit = final_equity - initial_equity
        roi = (total_profit / initial_equity) * 100
        
        # Calculate performance metrics
        equity_series = pd.Series(equity_curve)
        returns = equity_series.pct_change().dropna()
        
        sharpe_ratio = returns.mean() / returns.std() * np.sqrt(365) if len(returns) > 1 else 0
        max_drawdown = (equity_series / equity_series.cummax() - 1).min()
        
        # Analyze trades
        winning_trades = [t for t in trades if 
                         (t['type'] == 'buy' and t['price'] < final_price) or 
                         (t['type'] == 'sell' and t['price'] > final_price)]
        win_rate = len(winning_trades) / len(trades) * 100 if trades else 0
        
        # Send report
        report = f"""
📊 <b>Backtest Results</b>
━━━━━━━━━━━━━━━━━━━━
📅 <b>Period:</b> {df.index[100].strftime('%Y-%m-%d')} to {df.index[-1].strftime('%Y-%m-%d')}
💰 <b>Initial:</b> ${initial_equity:.2f} | <b>Final:</b> ${final_equity:.2f}
📈 <b>Profit/Loss:</b> ${total_profit:+.2f} (<b>{roi:+.2f}%</b>)
🔄 <b>Trades:</b> {len(trades)}
✅ <b>Win Rate:</b> {win_rate:.1f}%
📉 <b>Max Drawdown:</b> {max_drawdown*100:.1f}%
⚡ <b>Sharpe Ratio:</b> {sharpe_ratio:.2f}
💸 <b>Total Fees:</b> ${sum(t['fee'] for t in trades):.2f}
        """
        
        send_telegram_message(report)
        logger.info(f"Backtest completed. ROI: {roi:+.2f}% | Trades: {len(trades)}")
        
    except Exception as e:
        error_msg = f"❌ Backtest failed: {str(e)}"
        logger.error(error_msg)
        send_telegram_message(error_msg)

# ------------------- Web Service -------------------
@app.route('/')
def health_check():
    return jsonify({"status": "active", "version": "v3.0"})

@app.route('/status')
def status():
    current_price = get_current_price()
    eth_balance, usdt_balance = get_balance()
    
    exposure = eth_balance * current_price if current_price else 0
    total = exposure + usdt_balance
    
    return jsonify({
        "status": "active",
        "trading_enabled": TRADING_ENABLED,
        "paper_trading": PAPER_TRADING,
        "current_price": current_price,
        "eth_balance": eth_balance,
        "usdt_balance": usdt_balance,
        "exposure_usd": exposure,
        "total_balance": total,
        "daily_pnl": daily_pnl,
        "open_positions": len(open_positions),
        "last_update": datetime.now(DAMASCUS_TZ).isoformat()
    })

@app.route('/control/<action>')
def control(action):
    global TRADING_ENABLED
    if action == 'start':
        TRADING_ENABLED = True
        send_telegram_message("🚀 Trading ENABLED by web command")
        return jsonify({"status": "trading_enabled"})
    elif action == 'stop':
        TRADING_ENABLED = False
        cancel_all_orders()
        send_telegram_message("⏹️ Trading DISABLED by web command")
        return jsonify({"status": "trading_disabled"})
    else:
        return jsonify({"error": "invalid_action"}), 400

@app.route('/reset_daily')
def reset_daily():
    global daily_pnl, daily_starting_balance
    eth_balance, usdt_balance = get_balance()
    current_price = get_current_price()
    if current_price:
        daily_starting_balance = (eth_balance * current_price) + usdt_balance
        daily_pnl = 0
        return jsonify({"status": "daily_reset", "new_balance": daily_starting_balance})
    return jsonify({"error": "could_not_reset"}), 400

def run_flask_app():
    port = int(os.environ.get('PORT', 10000))
    app.run(host='0.0.0.0', port=port, debug=False, use_reloader=False)

# ------------------- Main Function -------------------
def main():
    """Main function with improved initialization"""
    try:
        # Reset daily tracking
        global daily_pnl, daily_starting_balance
        eth_balance, usdt_balance = get_balance()
        current_price = get_current_price()
        if current_price:
            daily_starting_balance = (eth_balance * current_price) + usdt_balance
            daily_pnl = 0
        
        # Start Flask in separate thread
        flask_thread = threading.Thread(target=run_flask_app, daemon=True)
        flask_thread.start()

        # Check required environment variables
        required_vars = ['COINEX_ACCESS_ID', 'COINEX_SECRET_KEY', 
                        'TELEGRAM_BOT_TOKEN', 'TELEGRAM_CHAT_ID']
        missing_vars = [var for var in required_vars if not os.environ.get(var)]
        
        if missing_vars:
            error_msg = f"❌ Missing environment variables: {', '.join(missing_vars)}"
            print(error_msg)
            sys.exit(1)

        # Determine operation mode
        if PAPER_TRADING:
            mode = "PAPER TRADING"
        elif TRADING_ENABLED:
            mode = "LIVE TRADING"
        else:
            mode = "MONITORING"
        
        # Send startup message
        startup_msg = f"""
🚀 <b>ETH Market Maker V3 Started!</b>
━━━━━━━━━━━━━━━━━━━━
🧪 <b>Mode:</b> {mode}
📊 <b>Status:</b> {'ENABLED' if TRADING_ENABLED else 'PAUSED'}
💼 <b>Capital:</b> ${TOTAL_CAPITAL}
📦 <b>Order Size:</b> ${TOTAL_CAPITAL * ORDER_SIZE_PERCENT:.2f}
🌐 <b>Health:</b> http://localhost:{os.environ.get('PORT', 10000)}/
🕐 <i>Start Time: {datetime.now(DAMASCUS_TZ).strftime('%Y-%m-%d %H:%M:%S')}</i>
        """
        
        send_telegram_message(startup_msg)
        logger.info(f"Bot started in {mode} mode")

        # Run initial backtest
        if not PAPER_TRADING:
            threading.Thread(target=backtest_strategy, daemon=True).start()

        # Main trading loop
        while True:
            try:
                if not PAPER_TRADING:
                    trading_cycle()
                else:
                    # Paper trading simulation
                    df = get_historical_data(limit=100)
                    if df is not None:
                        df = calculate_indicators(df)
                        current_price = get_current_price()
                        if current_price:
                            signals = generate_trading_signals(df)
                            if signals:
                                signal = should_enter_trade(signals, current_price, df)
                                if signal:
                                    eth_balance, usdt_balance = get_balance()
                                    total_balance = (eth_balance * current_price) + usdt_balance
                                    position_size = calculate_position_size(current_price, total_balance)
                                    if position_size >= 5:
                                        paper_trade(signal, position_size / current_price, current_price)
                
                time.sleep(300)  # 5 minutes between cycles
                
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


