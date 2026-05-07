import feedparser
from textblob import TextBlob
import time
import os
import json

# Path Configuration
DATA_DIR = "data"
SENTIMENT_FILE = os.path.join(DATA_DIR, "market_sentiment.json")

# RSS Feeds for Financial News (Institutional Sources)
FEEDS = [
    "https://www.reutersagency.com/feed/?best-topics=wealth&post_type=best",
    "https://www.cnbc.com/id/10000664/device/rss/rss.html",
    "https://www.cnbc.com/id/100003114/device/rss/rss.html",
    "https://www.cnbc.com/id/15839072/device/rss/rss.html",
    "https://www.investing.com/rss/news.rss",
    "https://www.investing.com/rss/market_overview.rss"
]

# Keywords that institutions care about (Advanced Weights)
INSTITUTIONAL_KEYWORDS = {
    "hawkish": 0.5, "dovish": -0.5, "inflation": -0.3, "recession": -0.6,
    "growth": 0.3, "yield": 0.2, "fed": 0.1, "rate hike": -0.4, "rate cut": 0.4,
    "surplus": 0.3, "deficit": -0.3, "stable": 0.2, "volatile": -0.2,
    "unemployment": -0.3, "gdp": 0.2, "pmi": 0.2, "nfp": 0.3
}

def analyze_sentiment_llm_free(text):
    """
    Langkah 3: Integrasi LLM Gratis (Local NLP Optimization).
    Karena user menginginkan opsi gratis, kita menggunakan 'vaderSentiment' 
    yang dioptimalkan khusus untuk teks finansial/sosial media.
    """
    from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
    analyzer = SentimentIntensityAnalyzer()
    
    # VADER is much better than TextBlob for financial context
    vs = analyzer.polarity_scores(text)
    score = vs['compound']
    
    # Custom Context Reinforcement (Langkah 3 Advanced)
    for kw, weight in INSTITUTIONAL_KEYWORDS.items():
        if kw in text.lower():
            score += weight
            
    return max(-1.0, min(1.0, score))

def fetch_news_sentiment():
    """
    Analisis sentimen institusi menggunakan VADER (LLM-Free/Gratis) 
    dan Institutional Keyword Weighting.
    """
    all_headlines = []
    
    print("--- Fetching Institutional Market Data (LLM-Free Mode) ---")
    for url in FEEDS:
        try:
            feed = feedparser.parse(url)
            for entry in feed.entries:
                all_headlines.append(entry.title)
        except Exception as e:
            print(f"Error fetching feed {url}: {e}")

    if not all_headlines: return 0.0

    total_score = 0
    for headline in all_headlines:
        score = analyze_sentiment_llm_free(headline)
        total_score += score
        
    avg_sentiment = total_score / len(all_headlines)
    avg_sentiment = max(-1.0, min(1.0, avg_sentiment))
    
    sentiment_data = {
        "timestamp": time.time(),
        "score": round(avg_sentiment, 4),
        "news_count": len(all_headlines),
        "method": "VADER + Institutional Weights (Free LLM Alternative)",
        "status": "Neutral" if abs(avg_sentiment) < 0.1 else ("Strong Bullish" if avg_sentiment > 0.3 else ("Bullish" if avg_sentiment > 0 else ("Strong Bearish" if avg_sentiment < -0.3 else "Bearish")))
    }
    
    if not os.path.exists(DATA_DIR):
        os.makedirs(DATA_DIR)
        
    with open(SENTIMENT_FILE, "w") as f:
        json.dump(sentiment_data, f, indent=4)
        
    print(f"Sentiment Analysis Complete: {sentiment_data['status']} ({sentiment_data['score']})")
    return avg_sentiment

def get_current_sentiment():
    """Mengambil skor sentimen terakhir yang tersimpan."""
    if os.path.exists(SENTIMENT_FILE):
        with open(SENTIMENT_FILE, "r") as f:
            data = json.load(f)
            # Check if data is older than 1 hour
            if time.time() - data['timestamp'] < 3600:
                return data['score']
    return fetch_news_sentiment()

if __name__ == "__main__":
    fetch_news_sentiment()
