from fastapi import FastAPI, HTTPException
import httpx
import asyncio
import os
import time
import pandas as pd
import numpy as np
from datetime import datetime
import logging
from typing import Dict, Any, List, Tuple
import json

# =============================================================================
# إعدادات البوت المتقدم
# =============================================================================

TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN", "")
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID", "")
EXECUTOR_BOT_URL = os.getenv("EXECUTOR_BOT_URL", "")
EXECUTOR_BOT_API_KEY = os.getenv("EXECUTOR_BOT_API_KEY", "")
EXECUTE_TRADES = os.getenv("EXECUTE_TRADES", "false").lower() == "true"

# إعدادات التداول
SCAN_INTERVAL = 300  # 5 دقائق بين كل فحص
CONFIDENCE_THRESHOLD_SINGLE = 60  # عتبة الإشارة الواحدة
CONFIDENCE_THRESHOLD_MULTIPLE = 55  # عتبة الإشارات المتعددة
MIN_STRATEGY_CONFIDENCE = 40  # أقل ثقة للاستراتيجيات المحتسبة

# العملات المدعومة
SUPPORTED_COINS = {
    'eth': {'name': 'Ethereum', 'binance_symbol': 'ETHUSDT', 'symbol': 'ETH'},
    'bnb': {'name': 'Binance Coin', 'binance_symbol': 'BNBUSDT', 'symbol': 'BNB'},
    'btc': {'name': 'Bitcoin', 'binance_symbol': 'BTCUSDT', 'symbol': 'BTC'},
}

TIMEFRAME = '1h'  # إطار زمني موحد لجميع الاستراتيجيات

# إعداد التسجيل
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger("advanced_signal_generator")

app = FastAPI(title="Advanced Crypto Signal Generator")

# إحصائيات النظام
system_stats = {
    "start_time": time.time(),
    "total_scans": 0,
    "signals_generated": 0,
    "signals_sent": 0,
    "last_heartbeat": None,
    "strategies_performance": {
        "ema_rsi_macd": {"calls": 0, "signals": 0},
        "volume_divergence": {"calls": 0, "signals": 0},
        "smart_money": {"calls": 0, "signals": 0}
    }
}

# =============================================================================
# الاستراتيجية 1: EMA + RSI + MACD (الأساسية)
# =============================================================================

class EmaRsiMacdStrategy:
    """الاستراتيجية الأساسية: المتوسطات + RSI + MACD"""
    
    def __init__(self):
        self.name = "ema_rsi_macd"
    
    @staticmethod
    def calculate_ema(prices: List[float], period: int) -> float:
        """حساب المتوسط المتحرك الأسي"""
        if len(prices) < period:
            return prices[-1] if prices else 0
        series = pd.Series(prices)
        ema = series.ewm(span=period, adjust=False).mean()
        return round(ema.iloc[-1], 4)
    
    @staticmethod
    def calculate_rsi(prices: List[float], period: int = 14) -> float:
        """حساب مؤشر RSI"""
        if len(prices) < period + 1:
            return 50.0
        
        deltas = np.diff(prices)
        gains = np.where(deltas > 0, deltas, 0)
        losses = np.where(deltas < 0, -deltas, 0)
        
        avg_gains = pd.Series(gains).rolling(period).mean().dropna().values
        avg_losses = pd.Series(losses).rolling(period).mean().dropna().values
        
        if len(avg_gains) == 0 or len(avg_losses) == 0:
            return 50.0
        
        if avg_losses[-1] == 0:
            return 100.0
        
        rs = avg_gains[-1] / avg_losses[-1]
        rsi = 100 - (100 / (1 + rs))
        return round(min(max(rsi, 0), 100), 2)
    
    @staticmethod
    def calculate_macd(prices: List[float]) -> Dict[str, float]:
        """حساب مؤشر MACD"""
        if len(prices) < 26:
            return {'macd': 0, 'signal': 0, 'histogram': 0}
        
        ema_12 = pd.Series(prices).ewm(span=12, adjust=False).mean().values
        ema_26 = pd.Series(prices).ewm(span=26, adjust=False).mean().values
        
        macd_line = ema_12[-1] - ema_26[-1]
        signal_line = pd.Series([ema_12[i] - ema_26[i] for i in range(len(prices))]).ewm(span=9, adjust=False).mean().values[-1]
        histogram = macd_line - signal_line
        
        return {
            'macd': round(macd_line, 4),
            'signal': round(signal_line, 4),
            'histogram': round(histogram, 4)
        }
    
    def analyze_trend(self, prices: List[float], current_price: float) -> Dict[str, Any]:
        """تحليل الاتجاه باستخدام المتوسطات المتحركة"""
        ema_9 = self.calculate_ema(prices, 9)
        ema_21 = self.calculate_ema(prices, 21)
        ema_50 = self.calculate_ema(prices, 50)
        
        ma_order = "صاعد" if ema_9 > ema_21 > ema_50 else "هابط" if ema_9 < ema_21 < ema_50 else "متذبذب"
        
        trend_strength = 0
        if ma_order == "صاعد":
            distance_9_21 = abs(ema_9 - ema_21) / current_price
            distance_21_50 = abs(ema_21 - ema_50) / current_price
            if distance_9_21 > 0.02 and distance_21_50 > 0.03:
                trend_strength = 10
            elif distance_9_21 > 0.01 and distance_21_50 > 0.015:
                trend_strength = 7
            else:
                trend_strength = 4
        elif ma_order == "هابط":
            distance_9_21 = abs(ema_9 - ema_21) / current_price
            distance_21_50 = abs(ema_21 - ema_50) / current_price
            if distance_9_21 > 0.02 and distance_21_50 > 0.03:
                trend_strength = 10
            elif distance_9_21 > 0.01 and distance_21_50 > 0.015:
                trend_strength = 7
            else:
                trend_strength = 4
        
        return {
            'ema_9': ema_9,
            'ema_21': ema_21,
            'ema_50': ema_50,
            'order': ma_order,
            'strength': trend_strength,
            'price_above_21': current_price > ema_21,
            'price_above_50': current_price > ema_50
        }
    
    def generate_signal(self, prices: List[float], volumes: List[float]) -> Dict[str, Any]:
        """توليد إشارة تداول"""
        system_stats["strategies_performance"][self.name]["calls"] += 1
        
        if len(prices) < 50:
            return {"signal": "none", "confidence": 0, "reasons": ["بيانات غير كافية"]}
        
        current_price = prices[-1]
        rsi = self.calculate_rsi(prices)
        macd = self.calculate_macd(prices)
        trend = self.analyze_trend(prices, current_price)
        
        confidence_score = 0
        reasons = []
        
        # تحليل المتوسطات (40 نقطة كحد أقصى)
        ma_score = 0
        if trend['order'] == "صاعد" and trend['price_above_21'] and trend['price_above_50']:
            ma_score = trend['strength'] * 4
            reasons.append(f"المتوسطات صاعدة (قوة: {trend['strength']}/10)")
        elif trend['order'] == "هابط" and not trend['price_above_21'] and not trend['price_above_50']:
            ma_score = trend['strength'] * 4
            reasons.append(f"المتوسطات هابطة (قوة: {trend['strength']}/10)")
        
        # تحليل RSI (30 نقطة كحد أقصى)
        rsi_score = 0
        if 40 <= rsi <= 65:
            distance_from_50 = abs(rsi - 50)
            rsi_score = max(0, 30 - (distance_from_50 * 1.5))
            reasons.append(f"RSI في منطقة مناسبة: {rsi}")
        elif 35 <= rsi <= 60:
            distance_from_50 = abs(rsi - 50)
            rsi_score = max(0, 30 - (distance_from_50 * 1.5))
            reasons.append(f"RSI في منطقة مناسبة: {rsi}")
        
        # تحليل MACD (30 نقطة كحد أقصى)
        macd_score = 0
        if macd['histogram'] > 0 and macd['macd'] > macd['signal']:
            macd_score = min(30, abs(macd['histogram']) * 1000)
            reasons.append(f"MACD إيجابي: {macd['histogram']:.4f}")
        elif macd['histogram'] < 0 and macd['macd'] < macd['signal']:
            macd_score = min(30, abs(macd['histogram']) * 1000)
            reasons.append(f"MACD سلبي: {macd['histogram']:.4f}")
        
        confidence_score = ma_score + rsi_score + macd_score
        
        signal_type = "none"
        if confidence_score >= 40:  # عتبة دنوى للاستراتيجية
            if trend['order'] == "صاعد" and 40 <= rsi <= 65 and macd['histogram'] > 0:
                signal_type = "BUY"
            elif trend['order'] == "هابط" and 35 <= rsi <= 60 and macd['histogram'] < 0:
                signal_type = "SELL"
        
        if signal_type != "none":
            system_stats["strategies_performance"][self.name]["signals"] += 1
        
        return {
            "signal": signal_type,
            "confidence": round(confidence_score),
            "price": current_price,
            "reasons": reasons,
            "timestamp": time.time()
        }

# =============================================================================
# الاستراتيجية 2: الانزياح الحجمي السعري
# =============================================================================

class VolumeDivergenceStrategy:
    """استراتيجية الانزياح بين السعر والحجم"""
    
    def __init__(self):
        self.name = "volume_divergence"
    
    @staticmethod
    def calculate_divergence(prices: List[float], volumes: List[float], 
                           lookback_period: int = 20) -> Dict[str, Any]:
        """حساب الانزياح بين حركة السعر والحجم"""
        if len(prices) < lookback_period * 2:
            return {"divergence": "none", "strength": 0}
        
        # تحليل الاتجاه السعري
        recent_prices = prices[-lookback_period:]
        older_prices = prices[-lookback_period*2:-lookback_period]
        
        price_trend_recent = (recent_prices[-1] - recent_prices[0]) / recent_prices[0]
        price_trend_older = (older_prices[-1] - older_prices[0]) / older_prices[0]
        
        # تحليل اتجاه الحجم
        recent_volumes = volumes[-lookback_period:]
        older_volumes = volumes[-lookback_period*2:-lookback_period]
        
        volume_trend_recent = (recent_volumes[-1] - np.mean(recent_volumes)) / np.mean(recent_volumes)
        volume_trend_older = (older_volumes[-1] - np.mean(older_volumes)) / np.mean(older_volumes)
        
        # كشف الانزياح الإيجابي (الأسعار تنخفض لكن الحجم يضعف)
        if (price_trend_recent < -0.03 and price_trend_older < -0.03 and
            volume_trend_recent > -0.2 and volume_trend_older < -0.3):
            strength = min(80, int(abs(price_trend_recent) * 1000 + abs(volume_trend_recent) * 100))
            return {"divergence": "positive_bullish", "strength": strength}
        
        # كشف الانزياح السلبي (الأسعار ترتفع لكن الحجم يضعف)
        elif (price_trend_recent > 0.03 and price_trend_older > 0.03 and
              volume_trend_recent < 0.2 and volume_trend_older > 0.3):
            strength = min(80, int(abs(price_trend_recent) * 1000 + abs(volume_trend_recent) * 100))
            return {"divergence": "negative_bearish", "strength": strength}
        
        # كشف التأكيد الحجمي
        elif ((price_trend_recent > 0.02 and volume_trend_recent > 0.3) or
              (price_trend_recent < -0.02 and volume_trend_recent > 0.3)):
            strength = min(70, int(abs(price_trend_recent) * 800 + volume_trend_recent * 50))
            return {"divergence": "volume_confirmation", "strength": strength}
        
        return {"divergence": "none", "strength": 0}
    
    def generate_signal(self, prices: List[float], volumes: List[float]) -> Dict[str, Any]:
        """توليد إشارة بناء على الانزياح الحجمي السعري"""
        system_stats["strategies_performance"][self.name]["calls"] += 1
        
        if len(prices) < 50:
            return {"signal": "none", "confidence": 0, "reasons": ["بيانات غير كافية"]}
        
        current_price = prices[-1]
        divergence_data = self.calculate_divergence(prices, volumes)
        
        confidence_score = 0
        signal_type = "none"
        reasons = []
        
        # الانزياح الإيجابي (إشارة شراء)
        if divergence_data["divergence"] == "positive_bullish":
            confidence_score = divergence_data["strength"]
            signal_type = "BUY"
            reasons = [
                "انزياح إيجابي: هبوط الأسعار مع ضعف حجم البيع",
                "تشير إلى استنفاد البائعين واستعداد للارتداد",
                f"قوة الانزياح: {divergence_data['strength']}%"
            ]
        
        # الانزياح السلبي (إشارة بيع)
        elif divergence_data["divergence"] == "negative_bearish":
            confidence_score = divergence_data["strength"]
            signal_type = "SELL"
            reasons = [
                "انزياح سلبي: صعود الأسعار مع ضعف حجم الشراء",
                "تشير إلى استنفاد المشترين واستعداد للهبوط",
                f"قوة الانزياح: {divergence_data['strength']}%"
            ]
        
        # تأكيد حجمي قوي
        elif divergence_data["divergence"] == "volume_confirmation":
            price_trend = "صاعد" if prices[-1] > prices[-10] else "هابط"
            
            if price_trend == "صاعد":
                confidence_score = divergence_data["strength"]
                signal_type = "BUY"
                reasons = [
                    "تأكيد حجمي قوي للصعود",
                    "حجم الشراء يدعم استمرار الاتجاه الصاعد",
                    f"قوة التأكيد: {divergence_data['strength']}%"
                ]
            else:
                confidence_score = divergence_data["strength"]
                signal_type = "SELL"
                reasons = [
                    "تأكيد حجمي قوي للهبوط", 
                    "حجم البيع يدعم استمرار الاتجاه الهابط",
                    f"قوة التأكيد: {divergence_data['strength']}%"
                ]
        
        if signal_type != "none" and confidence_score >= 40:
            system_stats["strategies_performance"][self.name]["signals"] += 1
        
        return {
            "signal": signal_type,
            "confidence": confidence_score,
            "price": current_price,
            "reasons": reasons,
            "timestamp": time.time()
        }

# =============================================================================
# الاستراتيجية 3: التراكم والتوزيع الذكي
# =============================================================================

class SmartMoneyStrategy:
    """استراتيجية تراكم وتوزيع ذكية تعتمد على تحليل تدفق الأموال"""
    
    def __init__(self):
        self.name = "smart_money"
    
    @staticmethod
    def detect_smart_money_patterns(prices: List[float], volumes: List[float], 
                                  window: int = 10) -> Dict[str, Any]:
        """كشف أنماط الأموال الذكية"""
        if len(prices) < window * 2:
            return {"pattern": "unknown", "confidence": 0}
        
        # تحليل العلاقة بين السعر والحجم
        price_change = (prices[-1] - prices[-window]) / prices[-window]
        volume_change = (volumes[-1] - np.mean(volumes[-window*2:-window])) / np.mean(volumes[-window*2:-window])
        
        # كشف أنماط التراكم
        if price_change < -0.02 and volume_change > 0.5:
            return {"pattern": "accumulation", "confidence": min(80, int(volume_change * 30))}
        
        # كشف أنماط التوزيع
        elif price_change > 0.02 and volume_change > 0.5:
            return {"pattern": "distribution", "confidence": min(80, int(volume_change * 30))}
        
        # كشف امتصاص البيع/الشراء
        elif abs(price_change) < 0.01 and volume_change > 1.0:
            return {"pattern": "absorption", "confidence": min(70, int(volume_change * 25))}
        
        return {"pattern": "no_pattern", "confidence": 0}
    
    @staticmethod
    def calculate_volume_clusters(prices: List[float], volumes: List[float]) -> Dict[str, Any]:
        """تحليل تجمعات الحجوم عند مستويات سعرية"""
        if len(prices) < 20:
            return {"high_volume_areas": {}, "strongest_level": None}
        
        # تقسيم البيانات إلى نطاقات سعرية
        recent_prices = prices[-20:]
        recent_volumes = volumes[-20:]
        
        price_min, price_max = min(recent_prices), max(recent_prices)
        if price_max - price_min == 0:
            return {"high_volume_areas": {}, "strongest_level": None}
        
        bins = 5
        bin_size = (price_max - price_min) / bins
        
        volume_clusters = {}
        for i in range(len(recent_prices)):
            bin_index = min(bins-1, int((recent_prices[i] - price_min) / bin_size))
            bin_key = f"{price_min + bin_index * bin_size:.2f}"
            volume_clusters[bin_key] = volume_clusters.get(bin_key, 0) + recent_volumes[i]
        
        # العثور على مناطق الحجم العالي
        max_volume = max(volume_clusters.values()) if volume_clusters else 0
        high_volume_areas = {k: v for k, v in volume_clusters.items() 
                           if v > max_volume * 0.7}
        
        return {
            "high_volume_areas": high_volume_areas,
            "strongest_level": max(volume_clusters, key=volume_clusters.get) if volume_clusters else None
        }
    
    def generate_signal(self, prices: List[float], volumes: List[float]) -> Dict[str, Any]:
        """توليد إشارة بناء على تحليل التراكم والتوزيع"""
        system_stats["strategies_performance"][self.name]["calls"] += 1
        
        if len(prices) < 50:
            return {"signal": "none", "confidence": 0, "reasons": ["بيانات غير كافية"]}
        
        current_price = prices[-1]
        
        # كشف أنماط الأموال الذكية
        smart_pattern = self.detect_smart_money_patterns(prices, volumes)
        
        # تحليل تجمعات الحجوم
        volume_clusters = self.calculate_volume_clusters(prices, volumes)
        
        # حساب متوسط الحجم المتحرك
        volume_ma_20 = np.mean(volumes[-20:])
        current_volume = volumes[-1]
        volume_ratio = current_volume / volume_ma_20 if volume_ma_20 > 0 else 1
        
        confidence_score = 0
        signal_type = "none"
        reasons = []
        
        # إشارة شراء: تراكم + حجم عالي
        if (smart_pattern["pattern"] == "accumulation" and 
            volume_ratio > 1.5 and
            smart_pattern["confidence"] > 40):
            
            confidence_score = smart_pattern["confidence"]
            signal_type = "BUY"
            reasons = [
                f"نمط تراكم الأموال الذكية (ثقة: {smart_pattern['confidence']}%)",
                f"حجم التداول: {volume_ratio:.1f}x المتوسط",
                "الشراء عند مناطق تراكم المؤسسات"
            ]
        
        # إشارة بيع: توزيع + حجم عالي
        elif (smart_pattern["pattern"] == "distribution" and 
              volume_ratio > 1.5 and
              smart_pattern["confidence"] > 40):
            
            confidence_score = smart_pattern["confidence"]
            signal_type = "SELL"
            reasons = [
                f"نمط توزيع الأموال الذكية (ثقة: {smart_pattern['confidence']}%)",
                f"حجم التداول: {volume_ratio:.1f}x المتوسط", 
                "البيع عند مناطق توزيع المؤسسات"
            ]
        
        # إشارة امتصاص
        elif (smart_pattern["pattern"] == "absorption" and 
              volume_ratio > 2.0 and
              smart_pattern["confidence"] > 40):
            
            price_trend = "صاعد" if prices[-1] > prices[-20] else "هابط"
            
            if price_trend == "صاعد":
                confidence_score = smart_pattern["confidence"]
                signal_type = "BUY"
                reasons = [
                    "امتصاص بيع قوي من قبل المشترين الأقوياء",
                    f"حجم امتصاص: {volume_ratio:.1f}x المتوسط",
                    "استعداد لصعود قوي"
                ]
            else:
                confidence_score = smart_pattern["confidence"]
                signal_type = "SELL" 
                reasons = [
                    "امتصاص شراء قوي من قبل البائعين الأقوياء",
                    f"حجم امتصاص: {volume_ratio:.1f}x المتوسط",
                    "استعداد لهبوط قوي"
                ]
        
        if signal_type != "none" and confidence_score >= 40:
            system_stats["strategies_performance"][self.name]["signals"] += 1
        
        return {
            "signal": signal_type,
            "confidence": confidence_score,
            "price": current_price,
            "reasons": reasons,
            "timestamp": time.time()
        }

# =============================================================================
# محرك الإشارات المتقدم
# =============================================================================

class AdvancedSignalEngine:
    """محرك الإشارات المتقدم الذي يدير الاستراتيجيات الثلاث"""
    
    def __init__(self):
        self.strategies = {
            "ema_rsi_macd": EmaRsiMacdStrategy(),
            "volume_divergence": VolumeDivergenceStrategy(),
            "smart_money": SmartMoneyStrategy()
        }
    
    def process_strategy_signals(self, strategy_signals: Dict[str, Dict]) -> Dict[str, Any]:
        """معالجة إشارات الاستراتيجيات وتطبيق قواعد الدمج"""
        
        # استخراج الإشارات فوق العتبة الدنيا فقط
        valid_signals = {}
        for strategy_name, signal in strategy_signals.items():
            if signal["signal"] != "none" and signal["confidence"] >= MIN_STRATEGY_CONFIDENCE:
                valid_signals[strategy_name] = signal
        
        if not valid_signals:
            return {"signal": "none", "confidence": 0, "reasons": ["لا توجد إشارات قوية"]}
        
        # كشف التضارب
        signals_list = list(valid_signals.values())
        buy_signals = [s for s in signals_list if s["signal"] == "BUY"]
        sell_signals = [s for s in signals_list if s["signal"] == "SELL"]
        
        if buy_signals and sell_signals:
            return {"signal": "none", "confidence": 0, "reasons": ["تضارب في الإشارات - تم الإلغاء"]}
        
        # تحديد الإشارة النهائية
        final_signal = "BUY" if buy_signals else "SELL" if sell_signals else "none"
        
        if final_signal == "none":
            return {"signal": "none", "confidence": 0, "reasons": ["لا توجد إشارات واضحة"]}
        
        # حساب الثقة النهائية بناء على القواعد
        active_signals = buy_signals if final_signal == "BUY" else sell_signals
        confidences = [s["confidence"] for s in active_signals]
        
        if len(active_signals) == 1:
            # إشارة واحدة - العتبة 60%
            if confidences[0] >= CONFIDENCE_THRESHOLD_SINGLE:
                final_confidence = confidences[0]
            else:
                return {"signal": "none", "confidence": 0, "reasons": [f"إشارة واحدة ضعيفة ({confidences[0]}%)"]}
        else:
            # إشارات متعددة - المتوسط يجب أن يكون ≥55%
            avg_confidence = sum(confidences) / len(confidences)
            if avg_confidence >= CONFIDENCE_THRESHOLD_MULTIPLE:
                final_confidence = avg_confidence
            else:
                return {"signal": "none", "confidence": 0, "reasons": [f"متوسط الثقة ضعيف ({avg_confidence:.1f}%)"]}
        
        # جمع الأسباب
        all_reasons = []
        for strategy_name, signal in valid_signals.items():
            if signal["signal"] == final_signal:
                all_reasons.extend(signal["reasons"])
        
        return {
            "signal": final_signal,
            "confidence": round(final_confidence, 1),
            "price": active_signals[0]["price"],
            "strategies_analysis": strategy_signals,
            "winning_strategies": len(active_signals),
            "total_strategies": len(self.strategies),
            "reasons": all_reasons,
            "timestamp": time.time()
        }
    
    async def analyze_coin(self, coin_symbol: str, prices: List[float], volumes: List[float]) -> Dict[str, Any]:
        """تحليل عملة باستخدام جميع الاستراتيجيات"""
        strategy_signals = {}
        
        for strategy_name, strategy in self.strategies.items():
            try:
                signal = strategy.generate_signal(prices, volumes)
                strategy_signals[strategy_name] = signal
                logger.info(f"📊 {strategy_name} لـ {coin_symbol}: {signal['signal']} ({signal['confidence']}%)")
            except Exception as e:
                logger.error(f"خطأ في استراتيجية {strategy_name} لـ {coin_symbol}: {e}")
                strategy_signals[strategy_name] = {"signal": "none", "confidence": 0, "reasons": [f"خطأ: {str(e)}"]}
        
        # معالجة الإشارات
        final_signal = self.process_strategy_signals(strategy_signals)
        final_signal["coin_symbol"] = coin_symbol
        
        return final_signal

# =============================================================================
# باقي المكونات (محدثة)
# =============================================================================

class BinanceDataFetcher:
    """جلب البيانات من Binance"""
    
    def __init__(self):
        self.client = httpx.AsyncClient(timeout=30.0)
        self.signal_engine = AdvancedSignalEngine()
    
    async def get_coin_data(self, coin_symbol: str, timeframe: str) -> Dict[str, Any]:
        """جلب بيانات العملة وتحليلها"""
        try:
            url = f"https://api.binance.com/api/v3/klines?symbol={coin_symbol}&interval={timeframe}&limit=100"
            logger.info(f"🔍 جلب بيانات {coin_symbol} من Binance...")
            response = await self.client.get(url)
            
            if response.status_code == 200:
                data = response.json()
                prices = [float(item[4]) for item in data]  # أسعار الإغلاق
                volumes = [float(item[5]) for item in data]  # أحجام التداول
                
                logger.info(f"✅ تم جلب {len(prices)} سعر و {len(volumes)} حجم لـ {coin_symbol}")
                
                # تحليل باستخدام محرك الإشارات المتقدم
                analysis_result = await self.signal_engine.analyze_coin(coin_symbol, prices, volumes)
                analysis_result['prices'] = prices
                analysis_result['volumes'] = volumes
                analysis_result['timeframe'] = timeframe
                
                logger.info(f"🎯 نتيجة تحليل {coin_symbol}: {analysis_result['signal']} ({analysis_result['confidence']}%)")
                
                return analysis_result
            else:
                logger.error(f"❌ فشل جلب البيانات لـ {coin_symbol}: {response.status_code}")
                return {"signal": "none", "confidence": 0, "reasons": ["فشل جلب البيانات"]}
                
        except Exception as e:
            logger.error(f"❌ خطأ في جلب بيانات {coin_symbol}: {e}")
            return {"signal": "none", "confidence": 0, "reasons": [f"خطأ: {str(e)}"]}

class TelegramNotifier:
    """إشعارات التليجرام المحدثة"""
    
    def __init__(self, token: str, chat_id: str):
        self.token = token
        self.chat_id = chat_id
        self.base_url = f"https://api.telegram.org/bot{token}"
    
    async def send_advanced_signal_alert(self, coin: str, signal_data: Dict[str, Any]) -> bool:
        """إرسال إشعار إشارة متقدم"""
        if signal_data["signal"] == "none":
            return False
        
        try:
            message = self._build_advanced_signal_message(coin, signal_data)
            
            payload = {
                'chat_id': self.chat_id,
                'text': message,
                'parse_mode': 'Markdown'
            }
            
            async with httpx.AsyncClient() as client:
                response = await client.post(f"{self.base_url}/sendMessage", 
                                           json=payload, timeout=10.0)
            
            if response.status_code == 200:
                logger.info(f"📨 تم إرسال إشعار إشارة متقدم لـ {coin}")
                return True
            else:
                logger.error(f"❌ فشل إرسال الإشعار: {response.status_code}")
                return False
                
        except Exception as e:
            logger.error(f"❌ خطأ في إرسال الإشعار: {e}")
            return False
    
    def _build_advanced_signal_message(self, coin: str, signal_data: Dict[str, Any]) -> str:
        """بناء رسالة إشارة متقدمة"""
        signal_type = signal_data["signal"]
        confidence = signal_data["confidence"]
        price = signal_data["price"]
        winning_strategies = signal_data["winning_strategies"]
        total_strategies = signal_data["total_strategies"]
        strategies_analysis = signal_data["strategies_analysis"]
        
        if signal_type == "BUY":
            emoji = "🟢"
            action = "شراء"
        else:  # SELL
            emoji = "🔴" 
            action = "بيع"
        
        message = f"{emoji} **إشارة {action} - {coin.upper()}**\n"
        message += "─" * 35 + "\n"
        message += f"💰 **السعر:** `${price:,.2f}`\n"
        message += f"🎯 **الثقة النهائية:** `{confidence}%`\n"
        message += f"📊 **الاستراتيجيات:** `{winning_strategies}/{total_strategies}`\n"
        message += f"⏰ **الإطار:** `{TIMEFRAME}`\n\n"
        
        # تفاصيل الاستراتيجيات
        message += "**تحليل الاستراتيجيات:**\n"
        for strategy_name, analysis in strategies_analysis.items():
            status_emoji = "✅" if analysis["signal"] == signal_type else "➖" if analysis["signal"] == "none" else "❌"
            display_name = strategy_name.replace('_', ' ').title()
            message += f"{status_emoji} **{display_name}:** `{analysis['confidence']}%`"
            if analysis["signal"] != "none" and analysis["signal"] != signal_type:
                message += f" (⚠️ {analysis['signal']})"
            message += "\n"
        
        message += "\n**الأسباب:**\n"
        for i, reason in enumerate(signal_data["reasons"][:5], 1):  # أول 5 أسباب فقط
            message += f"• {reason}\n"
        
        message += "─" * 35 + "\n"
        message += f"🕒 **الوقت:** `{datetime.now().strftime('%H:%M %d/%m')}`\n"
        message += "⚡ **المحرك المتقدم للإشارات**"
        
        return message
    
    async def send_heartbeat(self, executor_connected: bool, signals_count: int = 0, 
                        recent_analysis: Dict[str, Any] = None) -> bool:
        """إرسال نبضة اتصال مع تحليل قوة الإشارات"""
        try:
            current_time = datetime.now().strftime('%H:%M %d/%m/%Y')
            uptime_seconds = time.time() - system_stats["start_time"]
            uptime_str = self._format_uptime(uptime_seconds)
        
            status_emoji = "✅" if executor_connected else "❌"
            status_text = "متصل" if executor_connected else "غير متصل"
        
            # إحصائيات الاستراتيجيات
            strategies_stats = system_stats["strategies_performance"]
        
            message = f"💓 **نبضة النظام المتقدم**\n"
            message += "─" * 35 + "\n"
            message += f"⏰ **الوقت:** `{current_time}`\n"
            message += f"⏱️ **مدة التشغيل:** `{uptime_str}`\n"
            message += f"🔗 **الاتصال بالمنفذ:** {status_emoji} `{status_text}`\n"
            message += f"📊 **الإشارات المرسلة:** `{signals_count}`\n"
            message += f"🔍 **المسحات الكلية:** `{system_stats['total_scans']}`\n\n"
        
            message += "**أداء الاستراتيجيات:**\n"
            for strategy_name, stats in strategies_stats.items():
                success_rate = (stats["signals"] / stats["calls"] * 100) if stats["calls"] > 0 else 0
                display_name = strategy_name.replace('_', ' ').title()
                message += f"• **{display_name}:** `{stats['signals']}/{stats['calls']}` ({success_rate:.1f}%)\n"
        
            # قسم تحليل قوة الإشارات (مُحسّن)
            if recent_analysis:
                message += "\n**📈 تحليل قوة الإشارات الأخيرة:**\n"
            
                for coin, analysis in recent_analysis.items():
                    if analysis and analysis.get('strategies_analysis'):
                        strategies_data = analysis['strategies_analysis']
                    
                        # حساب قوة الإشارة الإجمالية مع تفاصيل البيع والشراء
                        buy_signals = []
                        sell_signals = []
                        
                        for strategy_name, strat_data in strategies_data.items():
                            if strat_data['signal'] == 'BUY' and strat_data['confidence'] > 0:
                                buy_signals.append(strat_data['confidence'])
                            elif strat_data['signal'] == 'SELL' and strat_data['confidence'] > 0:
                                sell_signals.append(strat_data['confidence'])
                        
                        if buy_signals and sell_signals:
                            # تضارب
                            buy_avg = sum(buy_signals) / len(buy_signals)
                            sell_avg = sum(sell_signals) / len(sell_signals)
                            message += f"⚡ **{coin.upper()}:** تضارب (🟢 شراء: {buy_avg:.1f}% | 🔴 بيع: {sell_avg:.1f}%)\n"
                        
                        elif buy_signals:
                            # اتجاه شراء
                            avg_confidence = sum(buy_signals) / len(buy_signals)
                            max_confidence = max(buy_signals)
                            emoji = "🟢" if avg_confidence >= 40 else "🟡"
                            message += f"{emoji} **{coin.upper()}:** اتجاه شراء ({len(buy_signals)}/3) - قوة: {avg_confidence:.1f}% (أعلى: {max_confidence}%)\n"
                        
                        elif sell_signals:
                            # اتجاه بيع
                            avg_confidence = sum(sell_signals) / len(sell_signals)
                            max_confidence = max(sell_signals)
                            emoji = "🔴" if avg_confidence >= 40 else "🟠"
                            message += f"{emoji} **{coin.upper()}:** اتجاه بيع ({len(sell_signals)}/3) - قوة: {avg_confidence:.1f}% (أعلى: {max_confidence}%)\n"
                        
                        else:
                            # لا توجد إشارات نشطة
                            all_confidences = [strat_data['confidence'] for strat_data in strategies_data.values() if strat_data['confidence'] > 0]
                            if all_confidences:
                                max_confidence = max(all_confidences)
                                if max_confidence > 30:
                                    message += f"⚪ **{coin.upper()}:** إشارات ضعيفة (أعلى: {max_confidence}%)\n"
                                else:
                                    message += f"⚫ **{coin.upper()}:** لا توجد إشارات قوية (أعلى: {max_confidence}%)\n"
                            else:
                                message += f"⚫ **{coin.upper()}:** لا توجد إشارات\n"
                    else:
                        message += f"⚫ **{coin.upper()}:** بيانات غير متوفرة\n"
        
            message += "─" * 35 + "\n"
            message += "✅ **جميع الأنظمة تعمل بشكل طبيعي**"
        
            payload = {
                'chat_id': self.chat_id,
                'text': message,
                'parse_mode': 'Markdown'
            }
        
            async with httpx.AsyncClient() as client:
                response = await client.post(f"{self.base_url}/sendMessage", 
                                       json=payload, timeout=10.0)
        
            if response.status_code == 200:
                logger.info("💓 تم إرسال نبضة النظام بنجاح")
                system_stats["last_heartbeat"] = current_time
                return True
            else:
                logger.error(f"❌ فشل إرسال النبضة: {response.status_code}")
                return False
            
        except Exception as e:
            logger.error(f"❌ خطأ في إرسال النبضة: {e}")
            return False
    
    def _format_uptime(self, seconds: float) -> str:
        """تنسيق مدة التشغيل"""
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        
        if hours > 0:
            return f"{hours} ساعة, {minutes} دقيقة"
        else:
            return f"{minutes} دقيقة"

class ExecutorBotClient:
    """عميل للتواصل مع بوت التنفيذ"""
    
    def __init__(self, base_url: str, api_key: str):
        self.base_url = base_url
        self.api_key = api_key
        self.client = httpx.AsyncClient(timeout=30.0)
    
    async def send_trade_signal(self, signal_data: Dict[str, Any]) -> bool:
        """إرسال إشارة تداول إلى البوت المنفذ"""
        if not EXECUTE_TRADES:
            logger.info("تنفيذ الصفقات معطل في الإعدادات")
            return False
            
        try:
            headers = {
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json"
            }
            
            payload = {
                "signal": signal_data,
                "timestamp": time.time(),
                "source": "advanced_signal_generator",
                "version": "2.0.0"
            }
            
            response = await self.client.post(
                f"{self.base_url}/api/trade/signal",
                json=payload,
                headers=headers
            )
            
            if response.status_code == 200:
                logger.info(f"✅ تم إرسال إشارة للتنفيذ: {signal_data['coin']} - {signal_data['action']}")
                system_stats["signals_sent"] += 1
                return True
            else:
                logger.error(f"❌ فشل إرسال الإشارة: {response.status_code}")
                return False
                
        except Exception as e:
            logger.error(f"❌ خطأ في التواصل مع البوت المنفذ: {e}")
            return False

    async def health_check(self) -> bool:
        """فحص حالة البوت المنفذ"""
        try:
            response = await self.client.get(f"{self.base_url}/health", timeout=10.0)
            return response.status_code == 200
        except Exception as e:
            logger.error(f"❌ فحص صحة البوت المنفذ فشل: {e}")
            return False

# =============================================================================
# التهيئة
# =============================================================================

data_fetcher = BinanceDataFetcher()
notifier = TelegramNotifier(TELEGRAM_BOT_TOKEN, TELEGRAM_CHAT_ID)
executor_client = ExecutorBotClient(EXECUTOR_BOT_URL, EXECUTOR_BOT_API_KEY)

# =============================================================================
# المهام الأساسية
# =============================================================================

# إضافة متغير عالمي لتخزين آخر تحليل
recent_analysis = {}

async def advanced_market_scanner_task():
    """المهمة الرئيسية للمسح الضوئي المتقدم"""
    global recent_analysis
    logger.info("🚀 بدء مهمة مسح السوق المتقدم كل 5 دقائق")
    
    while True:
        try:
            signals_found = 0
            scan_results = []
            current_analysis = {}  # تحليل هذه الدورة
            
            logger.info(f"🔍 بدء مسح {len(SUPPORTED_COINS)} عملة...")
            
            for coin_key, coin_data in SUPPORTED_COINS.items():
                try:
                    logger.info(f"📊 معالجة {coin_key}...")
                    # جلب البيانات وتحليلها
                    analysis_result = await data_fetcher.get_coin_data(coin_data['binance_symbol'], TIMEFRAME)
                    
                    # حفظ التحليل الحالي
                    current_analysis[coin_key] = analysis_result
                    
                    # إذا كانت هناك إشارة قوية
                    if (analysis_result["signal"] != "none" and 
                        analysis_result["confidence"] >= CONFIDENCE_THRESHOLD_SINGLE):
                        
                        logger.info(f"🎯 إشارة {analysis_result['signal']} لـ {coin_key} - ثقة: {analysis_result['confidence']}% - استراتيجيات: {analysis_result['winning_strategies']}/{analysis_result['total_strategies']}")
                        
                        scan_results.append({
                            'coin': coin_key,
                            'coin_data': coin_data,
                            'analysis': analysis_result
                        })
                        signals_found += 1
                    else:
                        logger.info(f"➖ لا توجد إشارة قوية لـ {coin_key} (ثقة: {analysis_result.get('confidence', 0)}%)")
                        
                except Exception as e:
                    logger.error(f"❌ خطأ في معالجة {coin_key}: {e}")
                    continue
            
            # تحديث التحليل الأخير
            recent_analysis = current_analysis
            logger.info(f"💾 تم حفظ تحليل {len(current_analysis)} عملة")
            
            # إرسال إشعار موحد بجميع الإشارات
            if signals_found > 0:
                await send_unified_alert(scan_results)
                
                # إرسال إشارات التنفيذ
                for result in scan_results:
                    trade_signal = {
                        "coin": result['coin'],
                        "symbol": result['coin_data']['binance_symbol'],
                        "action": result['analysis']["signal"],
                        "timeframe": TIMEFRAME,
                        "price": result['analysis']["price"],
                        "confidence": result['analysis']["confidence"],
                        "winning_strategies": result['analysis']["winning_strategies"],
                        "total_strategies": result['analysis']["total_strategies"],
                        "reasons": result['analysis']["reasons"],
                        "strategies_analysis": result['analysis']["strategies_analysis"]
                    }
                    
                    await executor_client.send_trade_signal(trade_signal)
                    await asyncio.sleep(1)
            
            system_stats["total_scans"] += 1
            system_stats["signals_generated"] += signals_found
            
            if signals_found > 0:
                logger.info(f"✅ اكتملت دورة المسح - تم العثور على {signals_found} إشارة")
            else:
                logger.info("✅ اكتملت دورة المسح - لا توجد إشارات قوية")
            
            await asyncio.sleep(SCAN_INTERVAL)
            
        except Exception as e:
            logger.error(f"❌ خطأ في المهمة الرئيسية: {e}")
            await asyncio.sleep(60)

async def heartbeat_task():
    """مهمة إرسال النبضات الدورية مع تحليل الإشارات"""
    global recent_analysis
    logger.info("💓 بدء مهمة النبضات الدورية كل ساعتين")
    
    await asyncio.sleep(300)
    
    while True:
        try:
            executor_health = await executor_client.health_check()
            
            success = await notifier.send_heartbeat(
                executor_connected=executor_health,
                signals_count=system_stats["signals_sent"],
                recent_analysis=recent_analysis  # إضافة التحليل الأخير
            )
            
            if success:
                logger.info("✅ تم إرسال النبضة الدورية بنجاح")
            else:
                logger.error("❌ فشل إرسال النبضة الدورية")
                
            await asyncio.sleep(7200)
                
        except Exception as e:
            logger.error(f"❌ خطأ في مهمة النبضات: {e}")
            await asyncio.sleep(300)

async def send_unified_alert(scan_results: List[Dict]):
    """إرسال إشعار موحد بجميع الإشارات"""
    if not scan_results:
        return
    
    try:
        message = "📊 **تقرير المسح المتقدم**\n"
        message += "─" * 35 + "\n"
        message += f"⏰ **الوقت:** `{datetime.now().strftime('%H:%M %d/%m')}`\n"
        message += f"🔍 **العملات المفحوصة:** `{len(SUPPORTED_COINS)}`\n"
        message += f"🎯 **الإشارات المكتشفة:** `{len(scan_results)}`\n\n"
        
        for i, result in enumerate(scan_results, 1):
            signal_type = result['analysis']["signal"]
            confidence = result['analysis']["confidence"]
            winning_strategies = result['analysis']["winning_strategies"]
            
            emoji = "🟢" if signal_type == "BUY" else "🔴"
            message += f"{emoji} **{result['coin'].upper()}:** {signal_type} ({confidence}%) - {winning_strategies}/3 استراتيجيات\n"
        
        message += "─" * 35 + "\n"
        message += "⚡ **المحرك المتقدم للإشارات**"
        
        payload = {
            'chat_id': TELEGRAM_CHAT_ID,
            'text': message,
            'parse_mode': 'Markdown'
        }
        
        async with httpx.AsyncClient() as client:
            await client.post(f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage", 
                            json=payload, timeout=10.0)
            
        logger.info(f"📨 تم إرسال التقرير الموحد ({len(scan_results)} إشارة)")
        
    except Exception as e:
        logger.error(f"❌ خطأ في إرسال التقرير الموحد: {e}")


# =============================================================================
# واجهات API
# =============================================================================

@app.get("/")
async def root():
    return {
        "message": "Advanced Crypto Signal Generator",
        "status": "running",
        "version": "2.0.0",
        "strategies": list(data_fetcher.signal_engine.strategies.keys()),
        "confidence_threshold_single": CONFIDENCE_THRESHOLD_SINGLE,
        "confidence_threshold_multiple": CONFIDENCE_THRESHOLD_MULTIPLE,
        "supported_coins": list(SUPPORTED_COINS.keys()),
        "timeframe": TIMEFRAME
    }

@app.get("/scan/{coin}")
async def scan_coin(coin: str):
    if coin not in SUPPORTED_COINS:
        raise HTTPException(404, "العملة غير مدعومة")
    
    coin_data = SUPPORTED_COINS[coin]
    analysis_result = await data_fetcher.get_coin_data(coin_data['binance_symbol'], TIMEFRAME)
    
    return {
        "coin": coin,
        "timeframe": TIMEFRAME,
        "analysis": analysis_result
    }

@app.get("/system-stats")
async def get_system_stats():
    uptime = time.time() - system_stats["start_time"]
    hours = int(uptime // 3600)
    minutes = int((uptime % 3600) // 60)
    
    return {
        "uptime": f"{hours} ساعة, {minutes} دقيقة",
        "uptime_seconds": uptime,
        "total_scans": system_stats["total_scans"],
        "signals_generated": system_stats["signals_generated"],
        "signals_sent": system_stats["signals_sent"],
        "last_heartbeat": system_stats["last_heartbeat"],
        "strategies_performance": system_stats["strategies_performance"],
        "confidence_thresholds": {
            "single_signal": CONFIDENCE_THRESHOLD_SINGLE,
            "multiple_signals": CONFIDENCE_THRESHOLD_MULTIPLE,
            "min_strategy_confidence": MIN_STRATEGY_CONFIDENCE
        },
        "supported_coins_count": len(SUPPORTED_COINS),
        "timeframe": TIMEFRAME
    }

@app.get("/test-signal/{coin}")
async def test_signal(coin: str):
    """اختبار توليد إشارة لعملة معينة"""
    if coin not in SUPPORTED_COINS:
        raise HTTPException(404, "العملة غير مدعومة")
    
    coin_data = SUPPORTED_COINS[coin]
    analysis_result = await data_fetcher.get_coin_data(coin_data['binance_symbol'], TIMEFRAME)
    
    # إرسال إشعار تجريبي
    if analysis_result["signal"] != "none":
        await notifier.send_advanced_signal_alert(coin, analysis_result)
    
    return {
        "coin": coin,
        "timeframe": TIMEFRAME,
        "analysis": analysis_result,
        "test_alert_sent": analysis_result["signal"] != "none"
    }

@app.get("/test-heartbeat")
async def test_heartbeat():
    """اختبار إرسال نبضة يدوية"""
    global recent_analysis
    try:
        executor_health = await executor_client.health_check()
        success = await notifier.send_heartbeat(
            executor_connected=executor_health,
            signals_count=system_stats["signals_sent"],
            recent_analysis=recent_analysis
        )
        
        return {
            "status": "success" if success else "error",
            "executor_connected": executor_health,
            "message": "تم إرسال النبضة بنجاح" if success else "فشل إرسال النبضة"
        }
    except Exception as e:
        return {"status": "error", "message": str(e)}

@app.get("/health")
async def health_check():
    """فحص صحة النظام"""
    try:
        executor_health = await executor_client.health_check()
        
        return {
            "status": "healthy",
            "timestamp": datetime.now().isoformat(),
            "executor_connected": executor_health,
            "system_stats": {
                "uptime_seconds": time.time() - system_stats["start_time"],
                "total_scans": system_stats["total_scans"],
                "signals_sent": system_stats["signals_sent"]
            }
        }
    except Exception as e:
        return {"status": "unhealthy", "error": str(e)}

# =============================================================================
# تشغيل التطبيق
# =============================================================================

@app.on_event("startup")
async def startup_event():
    logger.info("🚀 بدء تشغيل مولد الإشارات المتقدم")
    logger.info(f"🎯 الاستراتيجيات: {list(data_fetcher.signal_engine.strategies.keys())}")
    logger.info(f"💰 العملات المدعومة: {list(SUPPORTED_COINS.keys())}")
    logger.info(f"⏰ الإطار الزمني: {TIMEFRAME}")
    logger.info(f"📊 عتبة الإشارة الواحدة: {CONFIDENCE_THRESHOLD_SINGLE}%")
    logger.info(f"📈 عتبة الإشارات المتعددة: {CONFIDENCE_THRESHOLD_MULTIPLE}%")
    logger.info(f"🔍 فاصل المسح: {SCAN_INTERVAL} ثانية")
    
    # إرسال نبضة بدء التشغيل
    try:
        executor_health = await executor_client.health_check()
        await notifier.send_heartbeat(
            executor_connected=executor_health, 
            signals_count=system_stats["signals_sent"]
        )
    except Exception as e:
        logger.error(f"❌ خطأ في إرسال نبضة البدء: {e}")
    
    # بدء المهام
    asyncio.create_task(advanced_market_scanner_task())
    asyncio.create_task(heartbeat_task())
    
    logger.info("✅ تم بدء جميع المهام بنجاح")

@app.on_event("shutdown")
async def shutdown_event():
    logger.info("🛑 إيقاف مولد الإشارات المتقدم")

if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)
