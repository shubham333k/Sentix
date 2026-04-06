"""
NLP Utilities for Sentiment Analysis and Topic Modeling
Contains all core NLP functions for the AI Review Analyzer
"""

import re
import os
import io
import warnings
from typing import List, Dict, Tuple, Any, Optional
from collections import Counter
from datetime import datetime

import numpy as np
import pandas as pd
from tqdm import tqdm

# Suppress warnings
warnings.filterwarnings('ignore')

# NLP & ML Libraries
from transformers import pipeline, AutoTokenizer, AutoModelForSequenceClassification
from sentence_transformers import SentenceTransformer
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from textblob import TextBlob

# Topic Modeling
try:
    from bertopic import BERTopic
    from bertopic.vectorizers import ClassTfidfTransformer
    BERTOPIC_AVAILABLE = True
except ImportError:
    BERTOPIC_AVAILABLE = False

# Visualization
from wordcloud import WordCloud
import matplotlib.pyplot as plt
import seaborn as sns

# LangChain & Groq
from langchain_groq import ChatGroq
try:
    from langchain.chains import LLMChain
    from langchain.prompts import PromptTemplate
except ImportError:
    from langchain_core.runnables import RunnableSequence as LLMChain
    from langchain_core.prompts import PromptTemplate

# Download NLTK data
import nltk
nltk.download('punkt', quiet=True)
nltk.download('stopwords', quiet=True)
nltk.download('averaged_perceptron_tagger', quiet=True)

# =============================================================================
# CONFIGURATION
# =============================================================================

ASPECT_KEYWORDS = {
    'price': ['price', 'cost', 'expensive', 'cheap', 'value', 'money', 'worth', 'affordable', 'budget', 'deal', 'discount'],
    'quality': ['quality', 'durable', 'sturdy', 'well-made', 'premium', 'excellent', 'superior', 'flimsy', 'cheaply made'],
    'delivery': ['delivery', 'shipping', 'arrived', 'package', 'box', 'damaged', 'fast', 'slow', 'quick', 'late', 'prompt'],
    'service': ['service', 'customer', 'support', 'help', 'response', 'refund', 'return', 'warranty', 'helpful', 'rude'],
    'appearance': ['look', 'design', 'color', 'beautiful', 'aesthetic', 'stylish', 'elegant', 'ugly', 'attractive'],
    'functionality': ['work', 'function', 'feature', 'easy', 'difficult', 'broken', 'defective', 'useful', 'convenient']
}

STOPWORDS = set(nltk.corpus.stopwords.words('english')) if hasattr(nltk.corpus, 'stopwords') else {
    'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by', 'is', 'was', 'are', 'this', 'that', 'it', 'its'
}

# =============================================================================
# DATA LOADING & PREPROCESSING
# =============================================================================

def clean_text(text: str) -> str:
    """Clean and normalize text data."""
    if pd.isna(text):
        return ""
    text = str(text).lower()
    text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
    text = re.sub(r'<.*?>', '', text)
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def extract_sentiment_basic(text: str) -> Dict[str, float]:
    """Extract sentiment using TextBlob as fallback."""
    blob = TextBlob(text)
    polarity = blob.sentiment.polarity
    subjectivity = blob.sentiment.subjectivity
    
    if polarity > 0.1:
        label = "POSITIVE"
        score = 0.5 + (polarity * 0.5)
    elif polarity < -0.1:
        label = "NEGATIVE"
        score = 0.5 + (abs(polarity) * 0.5)
    else:
        label = "NEUTRAL"
        score = 0.5
    
    return {
        'label': label,
        'score': score,
        'polarity': polarity,
        'subjectivity': subjectivity
    }

class SentimentAnalyzer:
    """Advanced sentiment analysis using Hugging Face transformers with fallback."""
    
    def __init__(self, model_name: str = "distilbert-base-uncased-finetuned-sst-2-english"):
        self.model_name = model_name
        self.classifier = None
        self.use_transformer = False
        self._load_model()
    
    def _load_model(self):
        """Load the transformer model with fallback to TextBlob."""
        try:
            self.classifier = pipeline(
                "sentiment-analysis",
                model=self.model_name,
                tokenizer=self.model_name,
                device=-1  # CPU
            )
            self.use_transformer = True
        except Exception as e:
            print(f"Transformer model loading failed: {e}")
            print("Falling back to TextBlob for sentiment analysis")
            self.use_transformer = False
    
    def analyze(self, texts: List[str], batch_size: int = 32) -> List[Dict]:
        """Analyze sentiment for a list of texts."""
        results = []
        
        if self.use_transformer and self.classifier:
            # Use transformer model
            for i in tqdm(range(0, len(texts), batch_size), desc="Sentiment Analysis"):
                batch = texts[i:i+batch_size]
                batch = [t[:512] for t in batch if t]  # Truncate to 512 tokens
                
                try:
                    batch_results = self.classifier(batch)
                    for res in batch_results:
                        results.append({
                            'label': res['label'],
                            'score': res['score']
                        })
                except Exception as e:
                    # Fallback for failed batches
                    for text in batch:
                        results.append(extract_sentiment_basic(text))
        else:
            # Use TextBlob fallback
            for text in tqdm(texts, desc="Sentiment Analysis (TextBlob)"):
                results.append(extract_sentiment_basic(text))
        
        return results
    
    def analyze_single(self, text: str) -> Dict:
        """Analyze sentiment for a single text."""
        if self.use_transformer and self.classifier:
            try:
                result = self.classifier(text[:512])[0]
                return {'label': result['label'], 'score': result['score']}
            except:
                return extract_sentiment_basic(text)
        return extract_sentiment_basic(text)

# =============================================================================
# TOPIC MODELING
# =============================================================================

class TopicModeler:
    """BERTopic-based topic modeling with fallback to LDA."""
    
    def __init__(self, embedding_model: str = "all-MiniLM-L6-v2", min_topic_size: int = 10):
        self.embedding_model_name = embedding_model
        self.min_topic_size = min_topic_size
        self.model = None
        self.embeddings = None
        self.embedding_model = None
        self.use_bertopic = BERTOPIC_AVAILABLE
        self.topics = None
        self.topic_info = None
    
    def fit_transform(self, texts: List[str], progress_bar=None) -> List[int]:
        """Fit topic model and return topic assignments."""
        if not texts:
            return []
        
        # Clean texts
        cleaned_texts = [clean_text(t) for t in texts if clean_text(t)]
        
        if self.use_bertopic:
            return self._fit_bertopic(cleaned_texts, progress_bar)
        else:
            return self._fit_lda(cleaned_texts, progress_bar)
    
    def _fit_bertopic(self, texts: List[str], progress_bar=None) -> List[int]:
        """Fit BERTopic model."""
        try:
            # Initialize embedding model
            self.embedding_model = SentenceTransformer(self.embedding_model_name)
            
            if progress_bar:
                progress_bar.progress(0.2, text="Generating embeddings...")
            
            self.embeddings = self.embedding_model.encode(texts, show_progress_bar=True)
            
            if progress_bar:
                progress_bar.progress(0.6, text="Building topic model...")
            
            # Initialize BERTopic
            self.model = BERTopic(
                embedding_model=self.embedding_model,
                min_topic_size=self.min_topic_size,
                verbose=True
            )
            
            if progress_bar:
                progress_bar.progress(0.8, text="Fitting model...")
            
            self.topics, self.probs = self.model.fit_transform(texts, self.embeddings)
            self.topic_info = self.model.get_topic_info()
            
            if progress_bar:
                progress_bar.progress(1.0, text="Topic modeling complete!")
            
            return self.topics
            
        except Exception as e:
            print(f"BERTopic failed: {e}, falling back to LDA")
            self.use_bertopic = False
            return self._fit_lda(texts, progress_bar)
    
    def _fit_lda(self, texts: List[str], progress_bar=None) -> List[int]:
        """Fit LDA as fallback."""
        try:
            from sklearn.decomposition import LatentDirichletAllocation
            
            if progress_bar:
                progress_bar.progress(0.3, text="Preparing LDA...")
            
            # Create TF-IDF matrix
            vectorizer = TfidfVectorizer(max_features=1000, stop_words='english', ngram_range=(1, 2))
            tfidf = vectorizer.fit_transform(texts)
            
            if progress_bar:
                progress_bar.progress(0.6, text="Fitting LDA model...")
            
            # Fit LDA
            lda = LatentDirichletAllocation(n_components=10, random_state=42, max_iter=10)
            lda.fit(tfidf)
            
            # Get topic assignments
            doc_topics = lda.transform(tfidf)
            self.topics = np.argmax(doc_topics, axis=1)
            
            # Create topic info DataFrame
            feature_names = vectorizer.get_feature_names_out()
            topic_data = []
            
            for idx, topic in enumerate(lda.components_):
                top_indices = topic.argsort()[-10:][::-1]
                keywords = [feature_names[i] for i in top_indices]
                kw_str = ', '.join(keywords[:3])
                topic_data.append({
                    'Topic': idx,
                    'Count': np.sum(self.topics == idx),
                    'Name': 'Topic {}: {}'.format(idx, kw_str)
                })
            
            self.topic_info = pd.DataFrame(topic_data)
            
            if progress_bar:
                progress_bar.progress(1.0, text="LDA complete!")
            
            return self.topics.tolist()
            
        except Exception as e:
            print(f"LDA also failed: {e}")
            return [0] * len(texts)
    
    def get_topic_keywords(self, topic_id: int, n_words: int = 10) -> List[Tuple[str, float]]:
        """Get top keywords for a topic."""
        if self.use_bertopic and self.model and topic_id != -1:
            try:
                return self.model.get_topic(topic_id)[:n_words]
            except:
                return []
        elif self.topic_info is not None and topic_id < len(self.topic_info):
            # For LDA, parse from topic name
            name = self.topic_info.iloc[topic_id]['Name']
            keywords = name.split(': ')[1].split(', ') if ': ' in name else []
            return [(kw, 1.0) for kw in keywords[:n_words]]
        return []
    
    def visualize_topics(self):
        """Generate topic visualization."""
        if self.use_bertopic and self.model:
            try:
                return self.model.visualize_topics()
            except:
                return None
        return None

# =============================================================================
# ASPECT EXTRACTION
# =============================================================================

def extract_aspects(text: str) -> Dict[str, float]:
    """Extract mentioned aspects with sentiment scores."""
    text_lower = text.lower()
    aspects = {}
    
    for aspect, keywords in ASPECT_KEYWORDS.items():
        mention_count = sum(1 for kw in keywords if kw in text_lower)
        if mention_count > 0:
            # Get local sentiment around aspect mentions
            blob = TextBlob(text)
            aspects[aspect] = {
                'mentions': mention_count,
                'sentiment': blob.sentiment.polarity
            }
    
    return aspects

def extract_key_phrases(texts: List[str], n_phrases: int = 20) -> List[Tuple[str, int]]:
    """Extract most common key phrases using TF-IDF."""
    if not texts:
        return []
    
    cleaned = [clean_text(t) for t in texts if clean_text(t)]
    
    # Extract n-grams
    vectorizer = TfidfVectorizer(
        max_features=1000,
        stop_words='english',
        ngram_range=(2, 3),
        min_df=2
    )
    
    try:
        tfidf_matrix = vectorizer.fit_transform(cleaned)
        feature_names = vectorizer.get_feature_names_out()
        
        # Get mean TF-IDF scores
        mean_scores = np.array(tfidf_matrix.mean(axis=0)).flatten()
        
        # Get top phrases
        top_indices = mean_scores.argsort()[-n_phrases:][::-1]
        phrases = [(feature_names[i], int(mean_scores[i] * 1000)) for i in top_indices]
        
        return phrases
    except:
        return []

def analyze_aspect_sentiment(df: pd.DataFrame) -> pd.DataFrame:
    """Analyze sentiment by aspect across the dataset."""
    aspect_sentiments = {aspect: [] for aspect in ASPECT_KEYWORDS.keys()}
    
    for _, row in df.iterrows():
        text = str(row.get('cleaned_text', ''))
        overall_sentiment = row.get('sentiment_score', 0)
        
        aspects = extract_aspects(text)
        for aspect, data in aspects.items():
            aspect_sentiments[aspect].append({
                'sentiment': data['sentiment'],
                'mentions': data['mentions'],
                'overall_sentiment': overall_sentiment
            })
    
    # Summarize
    summary = []
    for aspect, data_list in aspect_sentiments.items():
        if data_list:
            avg_sentiment = np.mean([d['sentiment'] for d in data_list])
            total_mentions = sum(d['mentions'] for d in data_list)
            summary.append({
                'Aspect': aspect.capitalize(),
                'Avg_Sentiment': avg_sentiment,
                'Total_Mentions': total_mentions,
                'Mention_Rate': len(data_list) / len(df) * 100
            })
    
    return pd.DataFrame(summary).sort_values('Total_Mentions', ascending=False)

# =============================================================================
# WORD CLOUD GENERATION
# =============================================================================

def generate_wordcloud(texts: List[str], title: str = "Word Cloud", 
                       colormap: str = 'viridis', width: int = 800, height: int = 400) -> plt.Figure:
    """Generate a word cloud from texts."""
    text = ' '.join([clean_text(t) for t in texts if clean_text(t)])
    
    wordcloud = WordCloud(
        width=width,
        height=height,
        background_color='white',
        colormap=colormap,
        max_words=100,
        stopwords=STOPWORDS,
        contour_width=1,
        contour_color='steelblue'
    ).generate(text)
    
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.imshow(wordcloud, interpolation='bilinear')
    ax.axis('off')
    ax.set_title(title, fontsize=16, fontweight='bold', pad=20)
    
    return fig

def generate_sentiment_wordclouds(df: pd.DataFrame) -> Dict[str, plt.Figure]:
    """Generate separate word clouds for each sentiment."""
    clouds = {}
    
    for sentiment in ['POSITIVE', 'NEGATIVE', 'NEUTRAL']:
        texts = df[df['sentiment_label'] == sentiment]['Text'].tolist()
        if texts:
            colormap = 'RdYlGn' if sentiment == 'POSITIVE' else 'Reds' if sentiment == 'NEGATIVE' else 'Blues'
            clouds[sentiment] = generate_wordcloud(texts, f"{sentiment} Reviews Word Cloud", colormap)
    
    return clouds

# =============================================================================
# GROQ/LANGCHAIN INTEGRATION
# =============================================================================

def initialize_groq_llm(api_key: str, model: str = "llama3-70b-8192") -> Optional[ChatGroq]:
    """Initialize Groq LLM."""
    try:
        llm = ChatGroq(
            api_key=api_key,
            model_name=model,
            temperature=0.3,
            max_tokens=2048
        )
        return llm
    except Exception as e:
        print(f"Error initializing Groq: {e}")
        return None

def create_insights_chain(llm: ChatGroq):
    """Create LangChain for generating insights."""
    template = """
    You are an expert Product Insights Analyst. Analyze the following product review data and provide actionable insights.
    
    Context:
    - Total Reviews: {total_reviews}
    - Average Rating: {avg_rating}/5
    - Sentiment Distribution: {sentiment_dist}
    - Top Topics: {top_topics}
    - Aspect Sentiments: {aspect_sentiments}
    
    User Question: {question}
    
    Sample Reviews (Representative):
    {sample_reviews}
    
    Provide a detailed, professional response that:
    1. Directly answers the user's question
    2. Includes specific data points and metrics
    3. Offers actionable recommendations
    4. Identifies trends and patterns
    
    Response:
    """
    
    prompt = PromptTemplate(
        input_variables=["total_reviews", "avg_rating", "sentiment_dist", 
                        "top_topics", "aspect_sentiments", "question", "sample_reviews"],
        template=template
    )
    
    return LLMChain(llm=llm, prompt=prompt)

def generate_executive_summary(df: pd.DataFrame, topic_modeler: TopicModeler, 
                               llm: Optional[ChatGroq] = None) -> str:
    """Generate executive summary using LLM or rule-based fallback."""
    
    # Prepare summary statistics
    total_reviews = len(df)
    avg_rating = df['Score'].mean() if 'Score' in df.columns else 0
    sentiment_counts = df['sentiment_label'].value_counts().to_dict()
    
    if llm:
        try:
            # Get topic keywords
            topic_summary = []
            if topic_modeler.topic_info is not None:
                for _, row in topic_modeler.topic_info.head(5).iterrows():
                    topic_id = row['Topic']
                    if topic_id != -1:
                        keywords = topic_modeler.get_topic_keywords(topic_id, 5)
                        topic_summary.append(f"Topic {topic_id}: {', '.join([k[0] for k in keywords])}")
            
            # Get aspect analysis
            aspect_df = analyze_aspect_sentiment(df)
            aspect_summary = aspect_df.head(3).to_string() if not aspect_df.empty else "N/A"
            
            # Sample reviews
            sample_reviews = []
            for sentiment in ['NEGATIVE', 'POSITIVE']:
                samples = df[df['sentiment_label'] == sentiment].head(2)['Text'].tolist()
                sample_reviews.extend([f"[{sentiment}] {s[:200]}..." for s in samples])
            
            chain = create_insights_chain(llm)
            
            result = chain.run({
                "total_reviews": total_reviews,
                "avg_rating": f"{avg_rating:.2f}",
                "sentiment_dist": str(sentiment_counts),
                "top_topics": "\n".join(topic_summary),
                "aspect_sentiments": aspect_summary,
                "question": "Generate a comprehensive executive summary with key findings and recommendations for product improvement.",
                "sample_reviews": "\n".join(sample_reviews[:4])
            })
            
            return result
            
        except Exception as e:
            print(f"LLM summary generation failed: {e}")
    
    # Fallback rule-based summary
    summary = f"""
    # Executive Summary - Product Review Analysis
    
    ## Overview
    - **Total Reviews Analyzed**: {total_reviews:,}
    - **Average Rating**: {avg_rating:.2f}/5.0
    - **Data Quality**: High confidence with comprehensive sentiment and topic analysis
    
    ## Sentiment Distribution
    - Positive Reviews: {sentiment_counts.get('POSITIVE', 0):,} ({sentiment_counts.get('POSITIVE', 0)/total_reviews*100:.1f}%)
    - Negative Reviews: {sentiment_counts.get('NEGATIVE', 0):,} ({sentiment_counts.get('NEGATIVE', 0)/total_reviews*100:.1f}%)
    - Neutral Reviews: {sentiment_counts.get('NEUTRAL', 0):,} ({sentiment_counts.get('NEUTRAL', 0)/total_reviews*100:.1f}%)
    
    ## Key Findings
    """
    
    # Add aspect insights
    aspect_df = analyze_aspect_sentiment(df)
    if not aspect_df.empty:
        summary += "\n### Aspect Analysis\n"
        for _, row in aspect_df.head(5).iterrows():
            sentiment_indicator = "✓" if row['Avg_Sentiment'] > 0 else "⚠" if row['Avg_Sentiment'] < 0 else "○"
            summary += f"- {sentiment_indicator} **{row['Aspect']}**: {row['Mention_Rate']:.1f}% mention rate, avg sentiment {row['Avg_Sentiment']:+.2f}\n"
    
    # Add topic insights
    if topic_modeler.topic_info is not None:
        summary += "\n### Top Discussion Topics\n"
        for _, row in topic_modeler.topic_info.head(5).iterrows():
            if row['Topic'] != -1:
                keywords = topic_modeler.get_topic_keywords(row['Topic'], 3)
                kw_str = ", ".join([k[0] for k in keywords])
                summary += f"- **Topic {row['Topic']}** ({row['Count']} reviews): {kw_str}\n"
    
    summary += f"""
    
    ## Recommendations
    1. **Focus Areas**: Address concerns in aspects with negative sentiment
    2. **Leverage Strengths**: Amplify marketing around positively-rated features
    3. **Monitor Trends**: Track emerging topics for proactive response
    
    ---
    *Generated on {datetime.now().strftime('%Y-%m-%d %H:%M')}*
    """
    
    return summary

# =============================================================================
# PDF REPORT GENERATION
# =============================================================================

def generate_pdf_report(df: pd.DataFrame, topic_modeler: TopicModeler, 
                        summary: str, output_path: str = "executive_report.pdf"):
    """Generate PDF executive report."""
    from fpdf import FPDF
    
    class PDF(FPDF):
        def header(self):
            self.set_font('Arial', 'B', 12)
            self.cell(0, 10, 'AI Product Review Analysis - Executive Report', 0, 1, 'C')
            self.ln(2)
        
        def footer(self):
            self.set_y(-15)
            self.set_font('Arial', 'I', 8)
            self.cell(0, 10, f'Page {self.page_no()} | {datetime.now().strftime("%Y-%m-%d")}', 0, 0, 'C')
    
    pdf = PDF()
    pdf.add_page()
    pdf.set_auto_page_break(auto=True, margin=15)
    
    # Title
    pdf.set_font('Arial', 'B', 20)
    pdf.cell(0, 15, 'Product Review Sentiment & Topic Analysis', 0, 1, 'C')
    pdf.ln(5)
    
    # Key Metrics
    pdf.set_font('Arial', 'B', 14)
    pdf.cell(0, 10, 'Key Metrics', 0, 1, 'L')
    pdf.set_font('Arial', '', 11)
    
    total = len(df)
    avg_rating = df['Score'].mean() if 'Score' in df.columns else 0
    sentiment_counts = df['sentiment_label'].value_counts()
    
    pdf.cell(0, 8, f'Total Reviews: {total:,}', 0, 1, 'L')
    pdf.cell(0, 8, f'Average Rating: {avg_rating:.2f}/5.0', 0, 1, 'L')
    pdf.cell(0, 8, f'Positive: {sentiment_counts.get("POSITIVE", 0):,} ({sentiment_counts.get("POSITIVE", 0)/total*100:.1f}%)', 0, 1, 'L')
    pdf.cell(0, 8, f'Negative: {sentiment_counts.get("NEGATIVE", 0):,} ({sentiment_counts.get("NEGATIVE", 0)/total*100:.1f}%)', 0, 1, 'L')
    pdf.ln(5)
    
    # Executive Summary
    pdf.set_font('Arial', 'B', 14)
    pdf.cell(0, 10, 'Executive Summary', 0, 1, 'L')
    pdf.set_font('Arial', '', 10)
    
    # Clean summary for PDF (replace unicode)
    clean_summary = summary.encode('latin-1', 'replace').decode('latin-1')
    pdf.multi_cell(0, 6, clean_summary)
    
    pdf.ln(5)
    
    # Topic Summary
    if topic_modeler.topic_info is not None:
        pdf.add_page()
        pdf.set_font('Arial', 'B', 14)
        pdf.cell(0, 10, 'Topic Analysis', 0, 1, 'L')
        pdf.set_font('Arial', '', 10)
        
        for _, row in topic_modeler.topic_info.head(10).iterrows():
            if row['Topic'] != -1:
                keywords = topic_modeler.get_topic_keywords(row['Topic'], 5)
                kw_str = ", ".join([k[0] for k in keywords])
                pdf.set_font('Arial', 'B', 10)
                pdf.cell(0, 6, f"Topic {row['Topic']}: {row['Count']} reviews", 0, 1, 'L')
                pdf.set_font('Arial', '', 10)
                pdf.cell(0, 6, f"Keywords: {kw_str}", 0, 1, 'L')
                pdf.ln(2)
    
    pdf.output(output_path)
    return output_path

# =============================================================================
# DOWNLOAD SAMPLE DATA
# =============================================================================

def download_sample_data(data_dir: str = "data") -> str:
    """Download Amazon Fine Food Reviews dataset."""
    import requests
    
    os.makedirs(data_dir, exist_ok=True)
    output_path = os.path.join(data_dir, "amazon_reviews.csv")
    
    if os.path.exists(output_path):
        return output_path
    
    # Use a reliable public dataset URL (Kaggle dataset via direct link)
    # Fallback to generating sample data if download fails
    urls = [
        "https://raw.githubusercontent.com/amazon-science/amazon-product-review-abuse-detection/main/data/sample.csv",
    ]
    
    for url in urls:
        try:
            response = requests.get(url, timeout=30)
            if response.status_code == 200:
                with open(output_path, 'wb') as f:
                    f.write(response.content)
                return output_path
        except:
            continue
    
    # Generate synthetic sample data if download fails
    return generate_sample_data(output_path)

def generate_sample_data(output_path: str) -> str:
    """Generate synthetic sample review data."""
    np.random.seed(42)
    n_samples = 5000
    
    products = ['Coffee Maker', 'Wireless Headphones', 'Fitness Tracker', 'Bluetooth Speaker', 
                'Laptop Stand', 'Phone Case', 'USB Cable', 'Power Bank', 'Smart Watch', 'Tablet']
    
    positive_templates = [
        "Great {product}! Really happy with the quality and {aspect}.",
        "Love this {product}. Excellent {aspect} and fast delivery.",
        "Best {product} I've bought. Amazing {aspect} for the price.",
        "Highly recommend! The {aspect} exceeded my expectations.",
        "Perfect {product}. Great {aspect} and customer service was helpful."
    ]
    
    negative_templates = [
        "Disappointed with the {product}. Poor {aspect} and overpriced.",
        "Terrible {aspect}. The {product} broke after a week.",
        "Waste of money. Bad {aspect} and delivery was late.",
        "Not worth it. The {product} has awful {aspect}.",
        "Avoid this {product}. Terrible {aspect} and no support."
    ]
    
    neutral_templates = [
        "Average {product}. The {aspect} is okay but nothing special.",
        "Decent {product} for the price. {aspect} could be better.",
        "It's fine. {aspect} works but not impressive.",
        "Okay {product}. Average {aspect}, met expectations.",
        "Standard {product}. Nothing great about the {aspect}."
    ]
    
    aspects = ['quality', 'price', 'design', 'performance', 'durability', 'battery life']
    
    reviews = []
    for i in range(n_samples):
        product = np.random.choice(products)
        aspect = np.random.choice(aspects)
        
        sentiment_choice = np.random.choice(['pos', 'neg', 'neu'], p=[0.5, 0.3, 0.2])
        
        if sentiment_choice == 'pos':
            template = np.random.choice(positive_templates)
            score = np.random.choice([4, 5], p=[0.3, 0.7])
        elif sentiment_choice == 'neg':
            template = np.random.choice(negative_templates)
            score = np.random.choice([1, 2], p=[0.6, 0.4])
        else:
            template = np.random.choice(neutral_templates)
            score = 3
        
        text = template.format(product=product, aspect=aspect)
        
        reviews.append({
            'Id': i + 1,
            'ProductId': f'P{np.random.randint(1000, 9999)}',
            'UserId': f'U{np.random.randint(10000, 99999)}',
            'ProfileName': f'User_{i}',
            'HelpfulnessNumerator': np.random.randint(0, 50),
            'HelpfulnessDenominator': np.random.randint(10, 100),
            'Score': score,
            'Time': int(pd.Timestamp('2020-01-01').timestamp()) + np.random.randint(0, 94608000),
            'Summary': text[:50] + '...',
            'Text': text,
            'Product': product
        })
    
    df = pd.DataFrame(reviews)
    df.to_csv(output_path, index=False)
    return output_path
