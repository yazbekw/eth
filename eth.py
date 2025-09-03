import os
import requests
import pandas as pd
import time
from datetime import datetime, timedelta
import telegram
from telegram import Update
from telegram.ext import Application, CommandHandler, ContextTypes
import logging
import asyncio
from typing import List, Dict, Any
import schedule
from bs4 import BeautifulSoup
import re
from flask import Flask, jsonify
import threading
import aiohttp

# إعداد التسجيل
logging.basicConfig(
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    level=logging.INFO
)
logger = logging.getLogger(__name__)

# تطبيق Flask لمراقبة الصحة
app = Flask(__name__)

@app.route('/')
def health_check():
    return jsonify({
        'status': 'healthy',
        'timestamp': datetime.now().isoformat(),
        'service': 'Crypto News Tracker Bot'
    })

@app.route('/health')
def health():
    return jsonify({'status': 'ok', 'time': datetime.now().isoformat()})

def run_flask():
    """تشغيل خادم Flask في الخلفية"""
    app.run(host='0.0.0.0', port=5000, debug=False, use_reloader=False)

class CryptoNewsTracker:
    def __init__(self, telegram_bot_token: str = None, telegram_chat_id: str = None):
        self.crypto_list = ['bitcoin', 'binance-coin']
        self.keywords = {
            'positive': ['adoption', 'partnership', 'launch', 'growth', 'positive', 'bullish', 
                        'approval', 'investment', 'institutional', 'greenlight', 'surge', 'rally',
                        'integrate', 'support', 'list', 'burn', 'upgrade', 'breakthrough'],
            'negative': ['hack', 'ban', 'regulation', 'lawsuit', 'negative', 'bearish', 'reject',
                        'crash', 'selloff', 'warning', 'fraud', 'scam', 'attack', 'exploit',
                        'delay', 'problem', 'issue', 'down', 'outage', 'investigation', 'sec'],
            'volatility': ['volatility', 'swing', 'flash crash', 'flash rally', 'whale movement', 
                          'large transfer', 'market move', 'liquidate', 'volatile', 'swing']
        }
        
        # إعداد بوت التلغرام إذا وجد التوكن
        self.telegram_bot = None
        self.telegram_chat_id = telegram_chat_id
        if telegram_bot_token:
            try:
                self.telegram_bot = telegram.Bot(token=telegram_bot_token)
                logger.info("تم تهيئة بوت التلغرام بنجاح")
            except Exception as e:
                logger.error(f"فشل في تهيئة بوت التلغرام: {e}")
        
        # آخر وقت لتحديث
        self.last_update = datetime.now()
        self.news_sources = {
            'newsapi': True,
            'binance': True,
            'coingecko': True,
            'coindesk': True,
            'cointelegraph': True
        }
    
    async def send_telegram_message(self, message: str, parse_mode: str = 'HTML'):
        """إرسال رسالة عبر التلغرام"""
        if self.telegram_bot and self.telegram_chat_id:
            try:
                # تقسيم الرسالة إذا كانت طويلة جداً (حد التلغرام هو 4096 حرف)
                if len(message) > 4000:
                    parts = [message[i:i+4000] for i in range(0, len(message), 4000)]
                    for part in parts:
                        await self.telegram_bot.send_message(
                            chat_id=self.telegram_chat_id,
                            text=part,
                            parse_mode=parse_mode
                        )
                        await asyncio.sleep(1)  # تجنب rate limiting
                else:
                    await self.telegram_bot.send_message(
                        chat_id=self.telegram_chat_id,
                        text=message,
                        parse_mode=parse_mode
                    )
                return True
            except Exception as e:
                logger.error(f"فشل في إرسال الرسالة عبر التلغرام: {e}")
        return False
    
    def calculate_news_quality(self, news_item: Dict[str, Any]) -> float:
        """حساب جودة الخبر بناء على عدة عوامل"""
        quality_score = 0.0
        content = (news_item['title'] + ' ' + news_item.get('content', '')).lower()
        
        # 1. مصدر الخبر (وزن 30%)
        source_rank = {
            'coindesk': 0.95, 'cointelegraph': 0.95, 'decrypt': 0.85,
            'the block': 0.9, 'bloomberg': 0.98, 'reuters': 0.98,
            'financial times': 0.95, 'wall street journal': 0.96,
            'binance': 0.9, 'newsapi': 0.8, 'coingecko': 0.85,
            'unknown': 0.6
        }
        
        source = news_item.get('source', 'unknown').lower()
        for known_source, score in source_rank.items():
            if known_source in source:
                quality_score += score * 0.3
                break
        else:
            quality_score += 0.6 * 0.3  # مصدر غير معروف
        
        # 2. حداثة الخبر (وزن 20%)
        published_at = news_item.get('published_at', '')
        if published_at:
            try:
                # تحويل وقت النشر إلى timestamp
                if 'T' in published_at:
                    pub_time = datetime.fromisoformat(published_at.replace('Z', '+00:00'))
                else:
                    pub_time = datetime.strptime(published_at, '%Y-%m-%d %H:%M:%S')
                
                time_diff = (datetime.now() - pub_time).total_seconds() / 3600  # الفارق بالساعات
                freshness = max(0, 1 - (time_diff / 48))  # تناقص الجودة بعد 48 ساعة
                quality_score += freshness * 0.2
            except:
                quality_score += 0.5 * 0.2  # وقت غير معروف
        else:
            quality_score += 0.3 * 0.2  # لا يوجد وقت نشر
        
        # 3. طول المحتوى (وزن 15%)
        content_length = len(content)
        length_score = min(1.0, content_length / 300)  # 300 حرف يعتبر محتوى كافي
        quality_score += length_score * 0.15
        
        # 4. وجود كلمات مفتاحية مهمة (وزن 25%)
        keyword_score = 0
        for category, words in self.keywords.items():
            for word in words:
                if word in content:
                    keyword_score += 0.05
                    break
        
        keyword_score = min(1.0, keyword_score)  # لا تتجاوز 1
        quality_score += keyword_score * 0.25
        
        # 5. التفاعل (إعجابات، مشاركات) - وزن 10%
        engagement = news_item.get('votes', {}).get('positive', 0) + news_item.get('votes', {}).get('negative', 0)
        engagement_score = min(1.0, engagement / 20)  # 20 تفاعل يعتبر جيد
        quality_score += engagement_score * 0.1
        
        return round(quality_score * 100, 2)  # تحويل إلى نسبة مئوية
    
    def get_newsapi_news(self):
        """جلب الأخبار من NewsAPI باستخدام المفتاح الخاص"""
        news_items = []
        try:
            newsapi_key = os.getenv('NEWSAPI_KEY')
            if not newsapi_key:
                logger.warning("مفتاح NewsAPI غير متوفر")
                return news_items
            
            # البحث عن أخبار البيتكوين والعملات المشفرة
            keywords = ['bitcoin', 'cryptocurrency', 'blockchain', 'binance', 'crypto']
            for keyword in keywords:
                url = f"https://newsapi.org/v2/everything?q={keyword}&sortBy=publishedAt&language=en&apiKey={newsapi_key}"
                response = requests.get(url, timeout=15)
                data = response.json()
                
                for article in data.get('articles', [])[:5]:  # أول 5 نتائج لكل كلمة
                    # تجنب التكرار
                    if not any(item['title'] == article['title'] for item in news_items):
                        news_items.append({
                            'title': article['title'],
                            'source': article['source']['name'],
                            'url': article['url'],
                            'published_at': article['publishedAt'],
                            'content': article['description'] or article['title'],
                            'api_source': 'newsapi'
                        })
                
                time.sleep(1)  # تجنب rate limiting
            
            logger.info(f"تم جلب {len(news_items)} خبر من NewsAPI")
        except Exception as e:
            logger.error(f"خطأ في جلب أخبار NewsAPI: {e}")
        
        return news_items
    
    def get_binance_news(self):
        """جلب أخبار Binance"""
        news_items = []
        try:
            url = "https://www.binance.com/en/support/announcement/c-48"
            headers = {
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
            }
            response = requests.get(url, headers=headers, timeout=10)
            soup = BeautifulSoup(response.text, 'lxml')
            
            # البحث عن عناصر الأخبار في Binance
            news_elements = soup.select('.css-1ej4hfo a')
            for element in news_elements[:10]:
                title = element.get_text(strip=True)
                if title and any(keyword in title.lower() for keyword in ['binance', 'bnb', 'bitcoin', 'btc', 'crypto']):
                    href = element.get('href', '')
                    if href and not href.startswith('http'):
                        href = f"https://www.binance.com{href}"
                    
                    news_items.append({
                        'title': title,
                        'source': 'Binance Announcements',
                        'url': href,
                        'published_at': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                        'content': title,
                        'api_source': 'binance'
                    })
            
            logger.info(f"تم جلب {len(news_items)} خبر من Binance")
        except Exception as e:
            logger.error(f"خطأ في جلب أخبار Binance: {e}")
        
        return news_items
    
    def get_coingecko_news(self):
        """جلب أخبار من CoinGecko"""
        news_items = []
        try:
            url = "https://www.coingecko.com/en/news"
            headers = {
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
            }
            response = requests.get(url, headers=headers, timeout=10)
            soup = BeautifulSoup(response.text, 'lxml')
            
            # البحث عن عناصر الأخبار
            news_elements = soup.select('[data-target="news-content.title"]')
            for element in news_elements[:10]:
                title = element.get_text(strip=True)
                if title and any(keyword in title.lower() for keyword in ['bitcoin', 'btc', 'binance', 'bnb', 'crypto']):
                    # الحصول على رابط الخبر
                    parent_link = element.find_parent('a')
                    href = parent_link.get('href', '') if parent_link else ''
                    if href and not href.startswith('http'):
                        href = f"https://www.coingecko.com{href}"
                    
                    news_items.append({
                        'title': title,
                        'source': 'CoinGecko',
                        'url': href,
                        'published_at': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                        'content': title,
                        'api_source': 'coingecko'
                    })
            
            logger.info(f"تم جلب {len(news_items)} خبر من CoinGecko")
        except Exception as e:
            logger.error(f"خطأ في جلب أخبار CoinGecko: {e}")
        
        return news_items
    
    def get_coindesk_news(self):
        """جلب أخبار من CoinDesk"""
        news_items = []
        try:
            url = "https://www.coindesk.com/"
            headers = {
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
            }
            response = requests.get(url, headers=headers, timeout=10)
            soup = BeautifulSoup(response.text, 'lxml')
            
            # البحث عن عناصر الأخبار الرئيسية
            news_elements = soup.select('h1, h2, h3, h4, h5, h6')
            for element in news_elements[:15]:
                title = element.get_text(strip=True)
                if title and len(title) > 20 and any(keyword in title.lower() for keyword in ['bitcoin', 'btc', 'binance', 'bnb', 'crypto', 'ether']):
                    # الحصول على رابط الخبر
                    parent_link = element.find_parent('a')
                    href = parent_link.get('href', '') if parent_link else ''
                    if href and not href.startswith('http'):
                        href = f"https://www.coindesk.com{href}"
                    
                    news_items.append({
                        'title': title,
                        'source': 'CoinDesk',
                        'url': href,
                        'published_at': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                        'content': title,
                        'api_source': 'coindesk'
                    })
            
            logger.info(f"تم جلب {len(news_items)} خبر من CoinDesk")
        except Exception as e:
            logger.error(f"خطأ في جلب أخبار CoinDesk: {e}")
        
        return news_items
    
    def get_cointelegraph_news(self):
        """جلب أخبار من CoinTelegraph"""
        news_items = []
        try:
            url = "https://cointelegraph.com/"
            headers = {
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
            }
            response = requests.get(url, headers=headers, timeout=10)
            soup = BeautifulSoup(response.text, 'lxml')
            
            # البحث عن عناصر الأخبار
            news_elements = soup.select('h1, h2, h3, h4, h5, h6')
            for element in news_elements[:15]:
                title = element.get_text(strip=True)
                if title and len(title) > 20 and any(keyword in title.lower() for keyword in ['bitcoin', 'btc', 'binance', 'bnb', 'crypto', 'ether']):
                    # الحصول على رابط الخبر
                    parent_link = element.find_parent('a')
                    href = parent_link.get('href', '') if parent_link else ''
                    if href and not href.startswith('http'):
                        href = f"https://cointelegraph.com{href}"
                    
                    news_items.append({
                        'title': title,
                        'source': 'CoinTelegraph',
                        'url': href,
                        'published_at': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                        'content': title,
                        'api_source': 'cointelegraph'
                    })
            
            logger.info(f"تم جلب {len(news_items)} خبر من CoinTelegraph")
        except Exception as e:
            logger.error(f"خطأ في جلب أخبار CoinTelegraph: {e}")
        
        return news_items
    
    def get_crypto_news(self) -> List[Dict[str, Any]]:
        """جلب الأخبار من جميع المصادر المتاحة"""
        news_items = []
        
        # جمع الأخبار من جميع المصادر المفعلة
        if self.news_sources.get('newsapi', True):
            news_items.extend(self.get_newsapi_news())
        
        if self.news_sources.get('binance', True):
            news_items.extend(self.get_binance_news())
        
        if self.news_sources.get('coingecko', True):
            news_items.extend(self.get_coingecko_news())
        
        if self.news_sources.get('coindesk', True):
            news_items.extend(self.get_coindesk_news())
        
        if self.news_sources.get('cointelegraph', True):
            news_items.extend(self.get_cointelegraph_news())
        
        # إزالة التكرارات بناء على العنوان
        unique_news = []
        seen_titles = set()
        
        for news in news_items:
            if news['title'] and news['title'] not in seen_titles:
                seen_titles.add(news['title'])
                unique_news.append(news)
        
        logger.info(f"تم جمع {len(unique_news)} خبر فريد من جميع المصادر")
        return unique_news
    
    def analyze_news_sentiment(self, news_items: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """تحليل مشاعر الأخبار"""
        analyzed_news = []
        
        for news in news_items:
            sentiment = 'neutral'
            sentiment_score = 0
            content = (news['title'] + ' ' + news.get('content', '')).lower()
            
            # حساب درجة المشاعر
            positive_count = sum(1 for word in self.keywords['positive'] if word in content)
            negative_count = sum(1 for word in self.keywords['negative'] if word in content)
            volatility_count = sum(1 for word in self.keywords['volatility'] if word in content)
            
            # تحديد المشاعر الأساسية
            if positive_count > negative_count:
                sentiment = 'positive'
                sentiment_score = positive_count - negative_count
            elif negative_count > positive_count:
                sentiment = 'negative'
                sentiment_score = negative_count - positive_count
            
            # حساب جودة الخبر
            quality_score = self.calculate_news_quality(news)
            
            # إضافة التحليل إلى الخبر
            news['sentiment'] = sentiment
            news['sentiment_score'] = sentiment_score
            news['volatility_indicators'] = volatility_count
            news['quality_score'] = quality_score
            analyzed_news.append(news)
            
        return analyzed_news
    
    def get_price_data(self) -> Dict[str, Any]:
        """جلب بيانات الأسعار من Binance API"""
        prices = {}
        try:
            # سعر البيتكوين
            btc_response = requests.get("https://api.binance.com/api/v3/ticker/24hr?symbol=BTCUSDT", timeout=10)
            btc_data = btc_response.json()
            prices['bitcoin'] = {
                'price': float(btc_data['lastPrice']),
                'change': float(btc_data['priceChangePercent']),
                'high': float(btc_data['highPrice']),
                'low': float(btc_data['lowPrice']),
                'volume': float(btc_data['volume'])
            }
            
            # سعر بينانس كوين
            bnb_response = requests.get("https://api.binance.com/api/v3/ticker/24hr?symbol=BNBUSDT", timeout=10)
            bnb_data = bnb_response.json()
            prices['binance-coin'] = {
                'price': float(bnb_data['lastPrice']),
                'change': float(bnb_data['priceChangePercent']),
                'high': float(bnb_data['highPrice']),
                'low': float(bnb_data['lowPrice']),
                'volume': float(bnb_data['volume'])
            }
        except Exception as e:
            logger.error(f"فشل في جلب بيانات الأسعار: {e}")
            
        return prices
    
    def generate_report(self) -> Dict[str, Any]:
        """إنشاء تقرير شامل"""
        logger.info("جاري جمع البيانات...")
        
        # جلب الأخبار وتحليلها
        news = self.get_crypto_news()
        analyzed_news = self.analyze_news_sentiment(news)
        
        # جلب بيانات الأسعار
        prices = self.get_price_data()
        
        # إنشاء التقرير
        report = {
            'generated_at': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            'prices': prices,
            'news': analyzed_news,
            'summary': {
                'bitcoin_news_count': sum(1 for news in analyzed_news if 'bitcoin' in news['title'].lower() or 'btc' in news['title'].lower()),
                'binance_news_count': sum(1 for news in analyzed_news if 'binance' in news['title'].lower() or 'bnb' in news['title'].lower()),
                'positive_news_count': sum(1 for news in analyzed_news if news['sentiment'] == 'positive'),
                'negative_news_count': sum(1 for news in analyzed_news if news['sentiment'] == 'negative'),
                'avg_quality_score': round(sum(news['quality_score'] for news in analyzed_news) / len(analyzed_news), 2) if analyzed_news else 0,
                'total_news': len(analyzed_news)
            }
        }
        
        self.last_update = datetime.now()
        return report
    
    async def send_alerts(self, report: Dict[str, Any]):
        """إرسال تنبيهات عند وجود أخبار مهمة"""
        important_news = [news for news in report['news'] 
                         if (news['sentiment'] in ['positive', 'negative'] and news['sentiment_score'] >= 2)
                         or news['volatility_indicators'] >= 2
                         or news['quality_score'] >= 80]
        
        if important_news:
            alert_message = "🚨 <b>تنبيهات أخبار العملات المشفرة</b> 🚨\n\n"
            
            for i, news in enumerate(important_news[:5]):  # إرسال最多5 تنبيهات
                sentiment_emoji = "📈" if news['sentiment'] == 'positive' else "📉" if news['sentiment'] == 'negative' else "➡️"
                alert_message += f"{i+1}. {sentiment_emoji} <b>{news['title']}</b>\n"
                alert_message += f"   📊 الجودة: {news['quality_score']}% | المصدر: {news['source']}\n"
                alert_message += f"   🔗 <a href='{news['url']}'>قراءة المزيد</a>\n\n"
            
            alert_message += f"⏰ التقرير generated at: {report['generated_at']}"
            
            await self.send_telegram_message(alert_message)
    
    async def send_daily_report(self):
        """إرسال تقرير يومي"""
        report = self.generate_report()
        
        # إنشاء رسالة التقرير
        message = "📊 <b>تقرير يومي لأخبار العملات المشفرة</b> 📊\n\n"
        
        # أسعار العملات
        message += "<b>أسعار العملات:</b>\n"
        for crypto, data in report['prices'].items():
            change_emoji = "🟢" if data['change'] >= 0 else "🔴"
            message += f"• {crypto}: ${data['price']:,.2f} {change_emoji} ({data['change']:+.2f}%)\n"
        
        # ملخص الأخبار
        message += f"\n<b>ملخص الأخبار:</b>\n"
        message += f"• 📰 إجمالي الأخبار: {report['summary']['total_news']}\n"
        message += f"• 📈 أخبار البيتكوين: {report['summary']['bitcoin_news_count']}\n"
        message += f"• 💰 أخبار بينانس: {report['summary']['binance_news_count']}\n"
        message += f"• ✅ أخبار إيجابية: {report['summary']['positive_news_count']}\n"
        message += f"• ❌ أخبار سلبية: {report['summary']['negative_news_count']}\n"
        message += f"• 🎯 جودة متوسطة: {report['summary']['avg_quality_score']}%\n"
        
        # أهم 3 أخبار
        top_news = sorted(report['news'], key=lambda x: (x['quality_score'], x['sentiment_score']), reverse=True)[:3]
        if top_news:
            message += f"\n<b>أهم الأخبار:</b>\n"
            for i, news in enumerate(top_news):
                sentiment_emoji = "📈" if news['sentiment'] == 'positive' else "📉" if news['sentiment'] == 'negative' else "➡️"
                message += f"{i+1}. {sentiment_emoji} {news['title']} (جودة: {news['quality_score']}%)\n"
        
        message += f"\n⏰ التقرير generated at: {report['generated_at']}"
        
        await self.send_telegram_message(message)
        await self.send_alerts(report)
    
    def run_scheduler(self):
        """تشغيل المجدول"""
        # تحديث كل ساعة
        schedule.every().hour.do(lambda: asyncio.run(self.send_alerts(self.generate_report())))
        
        # تقرير يومي الساعة 10 صباحاً
        schedule.every().day.at("10:00").do(lambda: asyncio.run(self.send_daily_report()))
        
        # تقرير مسائي الساعة 8 مساءً
        schedule.every().day.at("20:00").do(lambda: asyncio.run(self.send_daily_report()))
        
        logger.info("بدأ المجدول")
        while True:
            schedule.run_pending()
            time.sleep(60)

# معالجات أوامر التلغرام
async def start_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """معالج أمر /start"""
    await update.message.reply_text(
        "مرحباً! أنا بوت متتبع أخبار العملات المشفرة.\n"
        "سأقوم بإرسال تحديثات دورية عن أخبار البيتكوين وبينانس كوين.\n\n"
        "الأوامر المتاحة:\n"
        "/start - عرض هذه الرسالة\n"
        "/report - الحصول على تقرير فوري\n"
        "/price - عرض الأسعار الحالية\n"
        "/sources - عرض المصادر المستخدمة"
    )

async def report_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """معالج أمر /report"""
    tracker = context.bot_data.get('tracker')
    if tracker:
        await update.message.reply_text("جاري إعداد التقرير...")
        await tracker.send_daily_report()
    else:
        await update.message.reply_text("البوت غير مهيء بشكل صحيح.")

async def price_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """معالج أمر /price"""
    tracker = context.bot_data.get('tracker')
    if tracker:
        prices = tracker.get_price_data()
        message = "💰 <b>الأسعار الحالية:</b>\n\n"
        
        for crypto, data in prices.items():
            change_emoji = "🟢" if data['change'] >= 0 else "🔴"
            message += f"<b>{crypto}:</b> ${data['price']:,.2f} {change_emoji} ({data['change']:+.2f}%)\n"
            message += f"   📊 High: ${data['high']:,.2f} | Low: ${data['low']:,.2f}\n\n"
        
        await update.message.reply_text(message, parse_mode='HTML')
    else:
        await update.message.reply_text("البوت غير مهيء بشكل صحيح.")

async def sources_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """معالج أمر /sources"""
    tracker = context.bot_data.get('tracker')
    if tracker:
        message = "📰 <b>المصادر المستخدمة:</b>\n\n"
        for source, enabled in tracker.news_sources.items():
            status = "✅" if enabled else "❌"
            message += f"{status} {source.capitalize()}\n"
        
        message += f"\n⏰ آخر تحديث: {tracker.last_update.strftime('%Y-%m-%d %H:%M:%S')}"
        await update.message.reply_text(message, parse_mode='HTML')
    else:
        await update.message.reply_text("البوت غير مهيء بشكل صحيح.")

async def main():
    """الدالة الرئيسية"""
    # الحصول على المتغيرات البيئية
    telegram_token = os.getenv('TELEGRAM_BOT_TOKEN')
    telegram_chat_id = os.getenv('TELEGRAM_CHAT_ID')
    
    if not telegram_token or not telegram_chat_id:
        logger.error("لم يتم تعيين TELEGRAM_BOT_TOKEN أو TELEGRAM_CHAT_ID")
        return
    
    # بدء خادم Flask في خيط منفصل
    flask_thread = threading.Thread(target=run_flask, daemon=True)
    flask_thread.start()
    logger.info("بدأ خادم Flask في الخلفية")
    
    # إنشاء متتبع الأخبار
    tracker = CryptoNewsTracker(telegram_token, telegram_chat_id)
    
    # إعداد بوت التلغرام
    application = Application.builder().token(telegram_token).build()
    
    # إضافة المعالجات
    application.add_handler(CommandHandler("start", start_command))
    application.add_handler(CommandHandler("report", report_command))
    application.add_handler(CommandHandler("price", price_command))
    application.add_handler(CommandHandler("sources", sources_command))
    
    # حفظ المتتبع في بيانات البوت
    application.bot_data['tracker'] = tracker
    
    # بدء المجدول في خلفية
    asyncio.create_task(asyncio.to_thread(tracker.run_scheduler))
    
    # إرسال رسالة بدء التشغيل
    await tracker.send_telegram_message("🤖 بدأ البوت في التشغيل بنجاح!")
    
    # بدء البوت
    logger.info("بدأ البوت في الاست polling")
    await application.run_polling()

if __name__ == "__main__":
    # التشغيل على Render (سيعمل كخدمة ويب)
    if os.getenv('RENDER', None):
        # على Render، تشغيل البوت مباشرة
        telegram_token = os.getenv('TELEGRAM_BOT_TOKEN')
        telegram_chat_id = os.getenv('TELEGRAM_CHAT_ID')
        
        if telegram_token and telegram_chat_id:
            # بدء خادم Flask
            flask_thread = threading.Thread(target=run_flask, daemon=True)
            flask_thread.start()
            
            tracker = CryptoNewsTracker(telegram_token, telegram_chat_id)
            asyncio.run(tracker.send_telegram_message("🤖 بدأ البوت في التشغيل على Render"))
            tracker.run_scheduler()
        else:
            print("لم يتم تعيين TELEGRAM_BOT_TOKEN أو TELEGRAM_CHAT_ID")
    else:
        # التشغيل المحلي
        asyncio.run(main())
