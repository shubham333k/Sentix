"""
AI Product Review Sentiment & Topic Analyzer
Main Streamlit Application
"""

import os
import io
import base64
from datetime import datetime
from typing import Optional, List, Dict

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Set page configuration - MUST be first Streamlit command
st.set_page_config(
    page_title="AI Product Review Sentiment & Topic Analyzer",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Load environment variables
from dotenv import load_dotenv
load_dotenv()

# Custom CSS for modern UI
CUSTOM_CSS = """
<style>
    /* Modern color scheme */
    :root {
        --primary: #6366f1;
        --secondary: #8b5cf6;
        --success: #22c55e;
        --warning: #f59e0b;
        --danger: #ef4444;
        --info: #3b82f6;
        --bg-dark: #0f172a;
        --bg-light: #f8fafc;
    }
    
    /* Main container styling */
    .main {
        padding: 2rem;
    }
    
    /* Metric cards */
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        border-radius: 15px;
        padding: 1.5rem;
        color: white;
        box-shadow: 0 10px 25px rgba(0,0,0,0.1);
        transition: transform 0.3s ease;
    }
    .metric-card:hover {
        transform: translateY(-5px);
    }
    
    /* Section headers */
    .section-header {
        font-size: 1.8rem;
        font-weight: 700;
        background: linear-gradient(90deg, #6366f1, #8b5cf6);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 1.5rem;
    }
    
    /* Tab styling */
    .stTabs [data-baseweb="tab-list"] {
        gap: 10px;
    }
    .stTabs [data-baseweb="tab"] {
        border-radius: 8px 8px 0 0;
        padding: 15px 25px;
        font-weight: 600;
    }
    .stTabs [aria-selected="true"] {
        background: linear-gradient(135deg, #6366f1, #8b5cf6) !important;
        color: white !important;
    }
    
    /* Chat styling */
    .chat-message {
        padding: 1rem;
        border-radius: 10px;
        margin: 0.5rem 0;
    }
    .chat-user {
        background: linear-gradient(135deg, #667eea, #764ba2);
        color: white;
        margin-left: 20%;
    }
    .chat-assistant {
        background: #f1f5f9;
        border-left: 4px solid #6366f1;
        margin-right: 20%;
    }
    
    /* Button styling */
    .stButton>button {
        border-radius: 10px;
        font-weight: 600;
        padding: 0.75rem 1.5rem;
        transition: all 0.3s ease;
    }
    .stButton>button:hover {
        transform: scale(1.02);
        box-shadow: 0 5px 15px rgba(99, 102, 241, 0.3);
    }
    
    /* Dark mode adjustments */
    [data-testid="stAppViewContainer"][data-theme="dark"] .chat-assistant {
        background: #1e293b;
        color: #f8fafc;
    }
    
    /* Spinner styling */
    .stSpinner > div {
        border-color: #6366f1 !important;
    }
    
    /* Dataframe styling */
    .stDataFrame {
        border-radius: 10px;
        overflow: hidden;
    }
    
    /* Download buttons */
    .download-btn {
        display: inline-flex;
        align-items: center;
        gap: 8px;
        padding: 10px 20px;
        background: linear-gradient(135deg, #22c55e, #16a34a);
        color: white;
        border-radius: 8px;
        text-decoration: none;
        font-weight: 600;
        transition: all 0.3s ease;
    }
    .download-btn:hover {
        transform: translateY(-2px);
        box-shadow: 0 5px 15px rgba(34, 197, 94, 0.3);
    }
</style>
"""

st.markdown(CUSTOM_CSS, unsafe_allow_html=True)

# Import utilities
from utils import (
    SentimentAnalyzer, TopicModeler, clean_text, generate_wordcloud,
    generate_sentiment_wordclouds, analyze_aspect_sentiment, extract_key_phrases,
    initialize_groq_llm, create_insights_chain, generate_executive_summary,
    generate_pdf_report, download_sample_data, ASPECT_KEYWORDS
)

# =============================================================================
# SESSION STATE INITIALIZATION
# =============================================================================

def init_session_state():
    """Initialize session state variables."""
    defaults = {
        'df': None,
        'sentiment_analyzer': None,
        'topic_modeler': None,
        'processed': False,
        'chat_history': [],
        'summary': '',
        'llm': None,
        'api_key_set': False,
        'wordclouds': {},
        'aspect_df': None,
        'key_phrases': []
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value

init_session_state()

# =============================================================================
# SIDEBAR
# =============================================================================

def render_sidebar():
    """Render the sidebar with controls."""
    with st.sidebar:
        st.markdown("""
        <div style="text-align: center; padding: 1rem 0;">
            <h1 style="font-size: 1.5rem; margin: 0;">🧠 AI Review Analyzer</h1>
            <p style="font-size: 0.9rem; color: #64748b; margin-top: 0.5rem;">
                NLP-Powered Product Insights
            </p>
        </div>
        """, unsafe_allow_html=True)
        
        st.divider()
        
        # API Key Section
        st.subheader("🔑 Configuration")
        
        groq_api_key = st.text_input(
            "Groq API Key (for GenAI chat)",
            type="password",
            value=os.getenv("GROQ_API_KEY", ""),
            help="Get free key at console.groq.com"
        )
        
        if groq_api_key and groq_api_key != "your_groq_api_key_here":
            os.environ["GROQ_API_KEY"] = groq_api_key
            st.session_state.llm = initialize_groq_llm(groq_api_key)
            st.session_state.api_key_set = st.session_state.llm is not None
            
            if st.session_state.api_key_set:
                st.success("✅ Groq API connected!")
            else:
                st.error("❌ Failed to connect")
        
        st.divider()
        
        # Data Source Section
        st.subheader("📊 Data Source")
        
        data_source = st.radio(
            "Choose data source:",
            ["📥 Use Sample Dataset", "📤 Upload Your CSV"],
            index=0
        )
        
        df = None
        
        if data_source == "📤 Upload Your CSV":
            uploaded_file = st.file_uploader(
                "Upload reviews CSV",
                type=["csv"],
                help="CSV should have 'Text' and 'Score' columns (optional: 'Product', 'Time')"
            )
            if uploaded_file:
                with st.spinner("📂 Loading uploaded data..."):
                    df = pd.read_csv(uploaded_file)
                st.success(f"✅ Loaded {len(df):,} reviews")
        else:
            if st.button("📥 Download Sample Data", use_container_width=True):
                with st.spinner("⬇️ Downloading sample dataset..."):
                    data_path = download_sample_data("data")
                    df = pd.read_csv(data_path)
                st.success(f"✅ Loaded {len(df):,} sample reviews")
        
        st.divider()
        
        # Theme Toggle
        st.subheader("🎨 Appearance")
        
        # Note: Streamlit's native theme is handled by the user in settings
        # We add custom dark mode indicator
        st.info("💡 Use ⋮ (top right) → Settings → Theme to toggle dark/light mode")
        
        st.divider()
        
        # About
        with st.expander("ℹ️ About"):
            st.markdown("""
            **AI Product Review Sentiment & Topic Analyzer**
            
            Built with:
            - 🤗 Hugging Face Transformers
            - 🧩 BERTopic / LDA
            - ⚡ Groq LLM
            - 📊 Plotly & Streamlit
            
            v1.0.0 | 2026
            """)
        
        return df

# =============================================================================
# MAIN DASHBOARD
# =============================================================================

def render_overview_tab(df: pd.DataFrame):
    """Render Overview & EDA tab."""
    st.markdown('<p class="section-header">📊 Overview & Exploratory Data Analysis</p>', unsafe_allow_html=True)
    
    # Key Metrics Row
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("📝 Total Reviews", f"{len(df):,}", delta=None)
    
    with col2:
        avg_rating = df['Score'].mean() if 'Score' in df.columns else 0
        st.metric("⭐ Average Rating", f"{avg_rating:.2f}/5", 
                 delta=f"{avg_rating - 3:.2f} from neutral")
    
    with col3:
        avg_length = df['Text'].str.len().mean() if 'Text' in df.columns else 0
        st.metric("📏 Avg Review Length", f"{avg_length:.0f} chars", delta=None)
    
    with col4:
        products = df['Product'].nunique() if 'Product' in df.columns else df['ProductId'].nunique() if 'ProductId' in df.columns else 1
        st.metric("📦 Products Analyzed", f"{products:,}", delta=None)
    
    st.divider()
    
    # Charts Row 1
    col_left, col_right = st.columns(2)
    
    with col_left:
        st.subheader("⭐ Rating Distribution")
        if 'Score' in df.columns:
            rating_counts = df['Score'].value_counts().sort_index()
            colors = ['#ef4444', '#f97316', '#f59e0b', '#22c55e', '#16a34a']
            
            fig = px.bar(
                x=rating_counts.index,
                y=rating_counts.values,
                labels={'x': 'Rating', 'y': 'Count'},
                color=rating_counts.index,
                color_continuous_scale=colors,
                text=rating_counts.values
            )
            fig.update_traces(texttemplate='%{text:,}', textposition='outside')
            fig.update_layout(
                showlegend=False,
                height=350,
                margin=dict(l=20, r=20, t=30, b=20),
                coloraxis_showscale=False
            )
            st.plotly_chart(fig, use_container_width=True)
        
    with col_right:
        st.subheader("💬 Review Length Distribution")
        if 'Text' in df.columns:
            df['review_length'] = df['Text'].str.len()
            
            fig = px.histogram(
                df,
                x='review_length',
                nbins=50,
                labels={'review_length': 'Review Length (characters)', 'count': 'Frequency'},
                color_discrete_sequence=['#6366f1']
            )
            fig.update_layout(height=350, margin=dict(l=20, r=20, t=30, b=20))
            fig.add_vline(x=df['review_length'].mean(), line_dash="dash", line_color="red",
                         annotation_text=f"Mean: {df['review_length'].mean():.0f}")
            st.plotly_chart(fig, use_container_width=True)
    
    st.divider()
    
    # Charts Row 2 - Monthly Trends and Products
    col_left2, col_right2 = st.columns(2)
    
    with col_left2:
        st.subheader("📅 Review Volume Over Time")
        if 'Time' in df.columns:
            try:
                df['datetime'] = pd.to_datetime(df['Time'], unit='s')
                monthly_counts = df.groupby(df['datetime'].dt.to_period('M')).size().reset_index()
                monthly_counts['datetime'] = monthly_counts['datetime'].astype(str)
                
                fig = px.line(
                    monthly_counts,
                    x='datetime',
                    y=0,
                    labels={'datetime': 'Month', '0': 'Review Count'},
                    markers=True,
                    line_shape='spline'
                )
                fig.update_traces(line_color='#8b5cf6', marker_color='#6366f1', marker_size=8)
                fig.update_layout(height=350, margin=dict(l=20, r=20, t=30, b=20))
                st.plotly_chart(fig, use_container_width=True)
            except:
                st.info("Time data format not recognized for trend analysis")
        elif 'Product' in df.columns or 'ProductId' in df.columns:
            # Show product-wise distribution instead
            prod_col = 'Product' if 'Product' in df.columns else 'ProductId'
            top_products = df[prod_col].value_counts().head(10).reset_index()
            
            fig = px.bar(
                top_products,
                x='count',
                y=prod_col,
                orientation='h',
                labels={'count': 'Review Count', prod_col: 'Product'},
                color='count',
                color_continuous_scale='Viridis'
            )
            fig.update_layout(height=350, margin=dict(l=20, r=20, t=30, b=20))
            st.plotly_chart(fig, use_container_width=True)
    
    with col_right2:
        st.subheader("☁️ Word Cloud - All Reviews")
        if st.session_state.wordclouds.get('all'):
            st.pyplot(st.session_state.wordclouds['all'])
        else:
            with st.spinner("Generating word cloud..."):
                wc = generate_wordcloud(df['Text'].tolist(), "All Reviews Word Cloud")
                st.session_state.wordclouds['all'] = wc
                st.pyplot(wc)
    
    st.divider()
    
    # Data Preview
    st.subheader("🔍 Data Preview")
    display_cols = ['Product', 'ProductId', 'Score', 'Summary', 'Text', 'sentiment_label', 'sentiment_score']
    available_cols = [c for c in display_cols if c in df.columns]
    st.dataframe(df[available_cols].head(50), use_container_width=True)

def render_sentiment_tab(df: pd.DataFrame):
    """Render Sentiment Analysis tab."""
    st.markdown('<p class="section-header">💭 Sentiment Analysis</p>', unsafe_allow_html=True)
    
    # Overall Sentiment
    col1, col2, col3 = st.columns([1, 1, 2])
    
    sentiment_counts = df['sentiment_label'].value_counts()
    total = len(df)
    
    with col1:
        pos_pct = sentiment_counts.get('POSITIVE', 0) / total * 100
        st.metric("😊 Positive", f"{sentiment_counts.get('POSITIVE', 0):,}", 
                 delta=f"{pos_pct:.1f}%", delta_color="normal")
    
    with col2:
        neg_pct = sentiment_counts.get('NEGATIVE', 0) / total * 100
        st.metric("😞 Negative", f"{sentiment_counts.get('NEGATIVE', 0):,}", 
                 delta=f"{neg_pct:.1f}%", delta_color="inverse")
    
    with col3:
        # Sentiment distribution pie chart
        fig = px.pie(
            values=sentiment_counts.values,
            names=sentiment_counts.index,
            color=sentiment_counts.index,
            color_discrete_map={
                'POSITIVE': '#22c55e',
                'NEGATIVE': '#ef4444',
                'NEUTRAL': '#64748b'
            },
            hole=0.5
        )
        fig.update_traces(textinfo='percent+label', pull=[0.05 if s == 'POSITIVE' else 0 for s in sentiment_counts.index])
        fig.update_layout(height=250, showlegend=True, margin=dict(l=20, r=20, t=30, b=20))
        st.plotly_chart(fig, use_container_width=True)
    
    st.divider()
    
    # Sentiment by Product
    st.subheader("📦 Sentiment by Product")
    
    prod_col = 'Product' if 'Product' in df.columns else 'ProductId' if 'ProductId' in df.columns else None
    
    if prod_col:
        # Calculate sentiment by product
        product_sentiment = df.groupby([prod_col, 'sentiment_label']).size().unstack(fill_value=0)
        product_sentiment['Total'] = product_sentiment.sum(axis=1)
        product_sentiment = product_sentiment.sort_values('Total', ascending=False).head(15)
        
        # Calculate percentages
        for col in ['POSITIVE', 'NEGATIVE', 'NEUTRAL']:
            if col in product_sentiment.columns:
                product_sentiment[f'{col}_pct'] = product_sentiment[col] / product_sentiment['Total'] * 100
        
        # Stacked bar chart
        fig = go.Figure()
        
        if 'POSITIVE' in product_sentiment.columns:
            fig.add_trace(go.Bar(
                name='Positive',
                x=product_sentiment.index,
                y=product_sentiment['POSITIVE'],
                marker_color='#22c55e',
                text=product_sentiment['POSITIVE_pct'].round(1).astype(str) + '%',
                textposition='inside'
            ))
        
        if 'NEUTRAL' in product_sentiment.columns:
            fig.add_trace(go.Bar(
                name='Neutral',
                x=product_sentiment.index,
                y=product_sentiment['NEUTRAL'],
                marker_color='#64748b',
                text=product_sentiment['NEUTRAL_pct'].round(1).astype(str) + '%',
                textposition='inside'
            ))
        
        if 'NEGATIVE' in product_sentiment.columns:
            fig.add_trace(go.Bar(
                name='Negative',
                x=product_sentiment.index,
                y=product_sentiment['NEGATIVE'],
                marker_color='#ef4444',
                text=product_sentiment['NEGATIVE_pct'].round(1).astype(str) + '%',
                textposition='inside'
            ))
        
        fig.update_layout(
            barmode='stack',
            height=450,
            xaxis_title='Product',
            yaxis_title='Number of Reviews',
            legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='center', x=0.5),
            margin=dict(l=50, r=50, t=80, b=100),
            xaxis_tickangle=-45
        )
        st.plotly_chart(fig, use_container_width=True)
    
    st.divider()
    
    # Aspect-Based Sentiment
    st.subheader("🔍 Aspect-Based Sentiment Analysis")
    
    if st.session_state.aspect_df is not None and not st.session_state.aspect_df.empty:
        aspect_df = st.session_state.aspect_df
        
        col_left, col_right = st.columns(2)
        
        with col_left:
            # Aspect mention rates
            fig = px.bar(
                aspect_df,
                x='Aspect',
                y='Mention_Rate',
                color='Avg_Sentiment',
                color_continuous_scale=['#ef4444', '#f59e0b', '#22c55e'],
                labels={'Mention_Rate': 'Mention Rate (%)', 'Avg_Sentiment': 'Avg Sentiment'},
                title='Aspect Mention Rates'
            )
            fig.update_layout(height=350, margin=dict(l=20, r=20, t=40, b=20))
            st.plotly_chart(fig, use_container_width=True)
        
        with col_right:
            # Aspect sentiment scores
            fig = px.scatter(
                aspect_df,
                x='Mention_Rate',
                y='Avg_Sentiment',
                size='Total_Mentions',
                color='Aspect',
                labels={'Mention_Rate': 'Mention Rate (%)', 'Avg_Sentiment': 'Average Sentiment'},
                title='Aspect Sentiment vs Mention Rate'
            )
            fig.update_layout(height=350, margin=dict(l=20, r=20, t=40, b=20))
            fig.add_hline(y=0, line_dash="dash", line_color="gray")
            st.plotly_chart(fig, use_container_width=True)
        
        # Aspect details table
        st.dataframe(
            aspect_df.style.background_gradient(subset=['Avg_Sentiment'], cmap='RdYlGn', vmin=-1, vmax=1)
            .format({'Avg_Sentiment': '{:+.2f}', 'Mention_Rate': '{:.1f}%'}),
            use_container_width=True
        )
    else:
        st.info("Aspect analysis not available. Process data to generate insights.")
    
    st.divider()
    
    # Sentiment Word Clouds
    st.subheader("☁️ Sentiment-Based Word Clouds")
    
    wc_cols = st.columns(3)
    
    for idx, (sentiment, title_suffix) in enumerate([
        ('POSITIVE', 'Positive'), 
        ('NEGATIVE', 'Negative'), 
        ('NEUTRAL', 'Neutral')
    ]):
        with wc_cols[idx]:
            st.write(f"**{title_suffix} Reviews**")
            if sentiment in st.session_state.wordclouds:
                st.pyplot(st.session_state.wordclouds[sentiment])
            else:
                texts = df[df['sentiment_label'] == sentiment]['Text'].tolist()
                if texts:
                    colormap = 'Greens' if sentiment == 'POSITIVE' else 'Reds' if sentiment == 'NEGATIVE' else 'Blues'
                    wc = generate_wordcloud(texts, f"{title_suffix} Reviews", colormap)
                    st.session_state.wordclouds[sentiment] = wc
                    st.pyplot(wc)

def render_topics_tab(df: pd.DataFrame):
    """Render Topic Modeling tab."""
    st.markdown('<p class="section-header">📚 Topic Modeling & Emerging Issues</p>', unsafe_allow_html=True)
    
    topic_modeler = st.session_state.topic_modeler
    
    if topic_modeler is None or topic_modeler.topic_info is None:
        st.warning("⚠️ Topic modeling not completed. Please run the analysis first.")
        return
    
    # Topic Overview
    st.subheader("📊 Topic Overview")
    
    topic_info = topic_modeler.topic_info
    # Filter out -1 (outliers)
    topic_info = topic_info[topic_info['Topic'] != -1]
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        # Topic distribution
        fig = px.bar(
            topic_info.head(10),
            x='Topic',
            y='Count',
            color='Count',
            color_continuous_scale='Viridis',
            labels={'Count': 'Number of Reviews', 'Topic': 'Topic ID'},
            text='Count'
        )
        fig.update_traces(texttemplate='%{text:,}', textposition='outside')
        fig.update_layout(height=400, margin=dict(l=20, r=20, t=30, b=20))
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        # Topic summary
        st.metric("🔢 Total Topics", len(topic_info))
        st.metric("📄 Reviews with Topics", topic_info['Count'].sum())
        st.metric("📈 Avg Reviews per Topic", f"{topic_info['Count'].mean():.0f}")
        
        # Outlier percentage
        if -1 in st.session_state.topic_modeler.topic_info['Topic'].values:
            outlier_count = st.session_state.topic_modeler.topic_info[st.session_state.topic_modeler.topic_info['Topic'] == -1]['Count'].iloc[0]
            outlier_pct = outlier_count / len(df) * 100
            st.metric("⚠️ Outlier Reviews", f"{outlier_pct:.1f}%")
    
    st.divider()
    
    # Top Topics Details
    st.subheader("🔍 Top 10 Topics - Detailed View")
    
    for idx, row in topic_info.head(10).iterrows():
        topic_id = row['Topic']
        keywords = topic_modeler.get_topic_keywords(topic_id, 10)
        
        with st.expander(f"📌 Topic {topic_id}: {row['Count']:,} reviews | Top words: {', '.join([k[0] for k in keywords[:5]])}"):
            col_kw, col_samples = st.columns([1, 2])
            
            with col_kw:
                st.write("**Keywords & Weights:**")
                for word, weight in keywords:
                    st.progress(weight, text=f"{word} ({weight:.3f})")
            
            with col_samples:
                st.write("**Sample Reviews:**")
                # Get sample reviews for this topic
                topic_mask = st.session_state.topic_modeler.topics == topic_id
                samples = df[topic_mask]['Text'].head(3).tolist() if 'Text' in df.columns else []
                
                for i, sample in enumerate(samples, 1):
                    sentiment = df[topic_mask]['sentiment_label'].iloc[i-1] if 'sentiment_label' in df.columns else 'UNKNOWN'
                    emoji = "😊" if sentiment == "POSITIVE" else "😞" if sentiment == "NEGATIVE" else "😐"
                    st.markdown(f"{emoji} *{sample[:200]}...*")
    
    st.divider()
    
    # Key Phrases
    st.subheader("🔑 Most Important Key Phrases")
    
    if st.session_state.key_phrases:
        phrases_df = pd.DataFrame(st.session_state.key_phrases, columns=['Phrase', 'Importance'])
        
        col_left, col_right = st.columns([2, 1])
        
        with col_left:
            fig = px.bar(
                phrases_df.head(15),
                x='Importance',
                y='Phrase',
                orientation='h',
                color='Importance',
                color_continuous_scale='Plasma',
                labels={'Importance': 'TF-IDF Score (×1000)', 'Phrase': 'Key Phrase'}
            )
            fig.update_layout(height=500, margin=dict(l=20, r=20, t=30, b=20), yaxis=dict(autorange='reversed'))
            st.plotly_chart(fig, use_container_width=True)
        
        with col_right:
            st.dataframe(phrases_df.head(20), use_container_width=True, hide_index=True)

def render_chat_tab(df: pd.DataFrame):
    """Render Ask Anything (GenAI Chat) tab."""
    st.markdown('<p class="section-header">🤖 Ask Anything - AI Product Insights Assistant</p>', unsafe_allow_html=True)
    
    # Check API key
    if not st.session_state.api_key_set:
        st.warning("⚠️ Please add your Groq API key in the sidebar to use the AI chat feature.")
        st.markdown("""
        **To get a free Groq API key:**
        1. Visit [console.groq.com](https://console.groq.com/keys)
        2. Sign up with your email
        3. Create a new API key (free tier: 1M tokens/min, 20M tokens/day)
        """)
        return
    
    # Chat interface
    st.markdown("""
    <div style="background: linear-gradient(135deg, #667eea20, #764ba220); padding: 1rem; border-radius: 10px; margin-bottom: 1rem;">
        <p style="margin: 0; color: #4b5563;">
            💡 <strong>Example questions:</strong> "What are customers complaining about most?" | 
            "Summarize negative reviews for Product X" | "What new features should we add?" |
            "Compare sentiment between top 2 products"
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    # Chat history display
    chat_container = st.container()
    with chat_container:
        for message in st.session_state.chat_history:
            if message['role'] == 'user':
                st.markdown(f'''
                <div class="chat-message chat-user">
                    <strong>You:</strong> {message['content']}
                </div>
                ''', unsafe_allow_html=True)
            else:
                st.markdown(f'''
                <div class="chat-message chat-assistant">
                    <strong>🤖 AI Assistant:</strong> {message['content']}
                </div>
                ''', unsafe_allow_html=True)
    
    # Input
    user_question = st.chat_input("Ask about your product reviews...", key="chat_input")
    
    if user_question:
        # Add user message
        st.session_state.chat_history.append({'role': 'user', 'content': user_question})
        
        # Generate context for the LLM
        with st.spinner("🤖 Analyzing data and generating response..."):
            try:
                # Prepare context
                total_reviews = len(df)
                avg_rating = df['Score'].mean() if 'Score' in df.columns else 0
                sentiment_counts = df['sentiment_label'].value_counts().to_dict()
                
                # Top topics
                topic_summary = []
                if st.session_state.topic_modeler and st.session_state.topic_modeler.topic_info is not None:
                    for _, row in st.session_state.topic_modeler.topic_info.head(5).iterrows():
                        topic_id = row['Topic']
                        if topic_id != -1:
                            keywords = st.session_state.topic_modeler.get_topic_keywords(topic_id, 5)
                            topic_summary.append(f"Topic {topic_id} ({row['Count']} reviews): {', '.join([k[0] for k in keywords])}")
                
                # Aspect insights
                aspect_summary = ""
                if st.session_state.aspect_df is not None:
                    aspect_summary = st.session_state.aspect_df.head(5).to_string(index=False)
                
                # Sample reviews by sentiment
                samples = []
                for sentiment in ['NEGATIVE', 'POSITIVE', 'NEUTRAL']:
                    sentiment_df = df[df['sentiment_label'] == sentiment]
                    if len(sentiment_df) > 0:
                        sample = sentiment_df.sample(min(2, len(sentiment_df)), random_state=42)
                        for _, row in sample.iterrows():
                            prod = row.get('Product', row.get('ProductId', 'Unknown'))
                            samples.append(f"[{sentiment}] Product: {prod}, Review: {row['Text'][:150]}...")
                
                # Create prompt
                context = f"""
                Product Review Dataset Analysis:
                
                DATASET SUMMARY:
                - Total Reviews: {total_reviews:,}
                - Average Rating: {avg_rating:.2f}/5.0
                - Sentiment Distribution: {sentiment_counts}
                
                TOP DISCUSSION TOPICS:
                {chr(10).join(topic_summary) if topic_summary else "Topic analysis not available"}
                
                ASPECT ANALYSIS:
                {aspect_summary if aspect_summary else "Aspect analysis not available"}
                
                SAMPLE REVIEWS (Representative):
                {chr(10).join(samples[:6])}
                
                USER QUESTION: {user_question}
                
                Provide a detailed, data-driven response. Include specific numbers and actionable recommendations.
                """
                
                # Call LLM
                response = st.session_state.llm.invoke(context)
                
                # Add assistant response
                st.session_state.chat_history.append({'role': 'assistant', 'content': response.content})
                
            except Exception as e:
                error_msg = f"Sorry, I encountered an error: {str(e)}. Please try a different question."
                st.session_state.chat_history.append({'role': 'assistant', 'content': error_msg})
        
        # Rerun to display new messages
        st.rerun()
    
    # Clear chat button
    if st.session_state.chat_history and st.button("🗑️ Clear Chat History", type="secondary"):
        st.session_state.chat_history = []
        st.rerun()

def render_summary_tab(df: pd.DataFrame):
    """Render Executive Summary & Actionable Insights tab."""
    st.markdown('<p class="section-header">📋 Executive Summary & Actionable Insights</p>', unsafe_allow_html=True)
    
    # Generate summary if not exists
    if not st.session_state.summary:
        with st.spinner("📊 Generating executive summary..."):
            st.session_state.summary = generate_executive_summary(
                df, 
                st.session_state.topic_modeler,
                st.session_state.llm
            )
    
    # Display summary
    st.markdown(st.session_state.summary)
    
    st.divider()
    
    # Download options
    st.subheader("📥 Download Reports & Data")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        # Download full analysis CSV
        csv_buffer = io.StringIO()
        df.to_csv(csv_buffer, index=False)
        csv_data = csv_buffer.getvalue()
        
        st.download_button(
            label="📄 Download Full Analysis (CSV)",
            data=csv_data,
            file_name=f"review_analysis_{datetime.now().strftime('%Y%m%d_%H%M')}.csv",
            mime="text/csv",
            use_container_width=True
        )
    
    with col2:
        # Download PDF report
        if st.button("📄 Generate PDF Report", use_container_width=True):
            with st.spinner("Generating PDF..."):
                pdf_path = f"executive_report_{datetime.now().strftime('%Y%m%d_%H%M')}.pdf"
                generate_pdf_report(df, st.session_state.topic_modeler, st.session_state.summary, pdf_path)
                
                with open(pdf_path, "rb") as f:
                    pdf_data = f.read()
                
                st.download_button(
                    label="⬇️ Download PDF Report",
                    data=pdf_data,
                    file_name=pdf_path,
                    mime="application/pdf",
                    use_container_width=True
                )
    
    with col3:
        # Download summary text
        st.download_button(
            label="📝 Download Summary (TXT)",
            data=st.session_state.summary,
            file_name=f"executive_summary_{datetime.now().strftime('%Y%m%d_%H%M')}.txt",
            mime="text/plain",
            use_container_width=True
        )
    
    st.divider()
    
    # Action items
    st.subheader("✅ Recommended Actions")
    
    # Generate action items based on data
    actions = []
    
    # Check sentiment distribution
    sentiment_counts = df['sentiment_label'].value_counts()
    neg_pct = sentiment_counts.get('NEGATIVE', 0) / len(df) * 100
    
    if neg_pct > 20:
        actions.append({
            'priority': 'High 🔴',
            'action': 'Address negative sentiment',
            'detail': f'{neg_pct:.1f}% negative reviews detected. Investigate common complaints.',
            'impact': 'Customer retention'
        })
    
    # Check aspect analysis
    if st.session_state.aspect_df is not None:
        negative_aspects = st.session_state.aspect_df[st.session_state.aspect_df['Avg_Sentiment'] < -0.2]
        for _, aspect in negative_aspects.head(3).iterrows():
            actions.append({
                'priority': 'High 🔴',
                'action': f'Improve {aspect["Aspect"]}',
                'detail': f'Low sentiment ({aspect["Avg_Sentiment"]:+.2f}) in {aspect["Aspect"]} aspect. {aspect["Mention_Rate"]:.1f}% of reviews mention this.',
                'impact': 'Product quality'
            })
        
        # High mention, neutral sentiment aspects
        neutral_aspects = st.session_state.aspect_df[
            (st.session_state.aspect_df['Mention_Rate'] > 30) & 
            (abs(st.session_state.aspect_df['Avg_Sentiment']) < 0.1)
        ]
        for _, aspect in neutral_aspects.head(2).iterrows():
            actions.append({
                'priority': 'Medium 🟡',
                'action': f'Enhance {aspect["Aspect"]} messaging',
                'detail': f'High mention rate ({aspect["Mention_Rate"]:.1f}%) but neutral sentiment. Opportunity for differentiation.',
                'impact': 'Marketing'
            })
    
    # Topic-based actions
    if st.session_state.topic_modeler and st.session_state.topic_modeler.topic_info is not None:
        # Find complaint-related topics
        for _, row in st.session_state.topic_modeler.topic_info.head(10).iterrows():
            topic_id = row['Topic']
            if topic_id == -1:
                continue
            keywords = st.session_state.topic_modeler.get_topic_keywords(topic_id, 5)
            kw_list = [k[0] for k in keywords]
            
            # Check for complaint keywords
            complaint_indicators = ['bad', 'terrible', 'worst', 'broke', 'problem', 'issue', 'defective', 'cheap']
            if any(kw in complaint_indicators for kw in kw_list):
                actions.append({
                    'priority': 'High 🔴',
                    'action': f'Investigate Topic {topic_id}',
                    'detail': f'Emerging issue: {", ".join(kw_list[:3])} mentioned in {row["Count"]} reviews.',
                    'impact': 'Quality control'
                })
    
    # Add default actions if few generated
    if len(actions) < 3:
        actions.extend([
            {
                'priority': 'Medium 🟡',
                'action': 'Monitor trending topics',
                'detail': 'Set up weekly topic modeling to catch emerging issues early.',
                'impact': 'Proactive management'
            },
            {
                'priority': 'Low 🟢',
                'action': 'Amplify positive feedback',
                'detail': f'Leverage the {sentiment_counts.get("POSITIVE", 0)/len(df)*100:.1f}% positive reviews in marketing.',
                'impact': 'Brand reputation'
            }
        ])
    
    # Display actions
    actions_df = pd.DataFrame(actions)
    st.dataframe(
        actions_df,
        column_config={
            'priority': st.column_config.TextColumn('Priority'),
            'action': st.column_config.TextColumn('Action Item'),
            'detail': st.column_config.TextColumn('Details'),
            'impact': st.column_config.TextColumn('Expected Impact')
        },
        use_container_width=True,
        hide_index=True
    )

# =============================================================================
# DATA PROCESSING
# =============================================================================

def process_data(df: pd.DataFrame, progress_bar) -> pd.DataFrame:
    """Process data through NLP pipeline."""
    df = df.copy()
    
    # Ensure required columns exist
    if 'Text' not in df.columns:
        # Try to find text column
        text_cols = [c for c in df.columns if any(word in c.lower() for word in ['text', 'review', 'content', 'body'])]
        if text_cols:
            df['Text'] = df[text_cols[0]]
        else:
            st.error("No text column found. Expected column: 'Text', 'Review', 'Content', or 'Body'")
            return df
    
    if 'Score' not in df.columns and 'Rating' in df.columns:
        df['Score'] = df['Rating']
    
    if 'Score' not in df.columns:
        st.warning("No 'Score' or 'Rating' column found. Rating analysis will be limited.")
        df['Score'] = 3  # Default neutral score
    
    # Clean text
    progress_bar.progress(0.1, text="🧹 Cleaning text data...")
    df['cleaned_text'] = df['Text'].apply(clean_text)
    
    # Initialize sentiment analyzer
    progress_bar.progress(0.15, text="🤗 Loading sentiment model...")
    if st.session_state.sentiment_analyzer is None:
        st.session_state.sentiment_analyzer = SentimentAnalyzer()
    
    # Sentiment analysis
    progress_bar.progress(0.25, text="💭 Analyzing sentiment...")
    texts = df['Text'].tolist()
    sentiment_results = st.session_state.sentiment_analyzer.analyze(texts)
    
    df['sentiment_label'] = [r['label'] for r in sentiment_results]
    df['sentiment_score'] = [r['score'] for r in sentiment_results]
    
    # Add polarity from textblob
    df['sentiment_polarity'] = [r.get('polarity', 0) for r in sentiment_results]
    
    # Topic modeling
    progress_bar.progress(0.6, text="📚 Building topic model...")
    if st.session_state.topic_modeler is None:
        st.session_state.topic_modeler = TopicModeler(min_topic_size=min(20, len(df) // 50))
    
    topics = st.session_state.topic_modeler.fit_transform(df['Text'].tolist(), progress_bar)
    df['topic'] = topics
    
    # Aspect analysis
    progress_bar.progress(0.9, text="🔍 Analyzing aspects...")
    st.session_state.aspect_df = analyze_aspect_sentiment(df)
    
    # Key phrases
    progress_bar.progress(0.95, text="🔑 Extracting key phrases...")
    st.session_state.key_phrases = extract_key_phrases(df['Text'].tolist(), 30)
    
    progress_bar.progress(1.0, text="✅ Analysis complete!")
    
    return df

# =============================================================================
# MAIN APP
# =============================================================================

def main():
    """Main application entry point."""
    # Header
    st.markdown("""
    <div style="text-align: center; padding: 2rem 0;">
        <h1 style="font-size: 3rem; font-weight: 800; margin-bottom: 0.5rem;">
            🧠 AI Product Review
            <span style="background: linear-gradient(90deg, #6366f1, #8b5cf6); -webkit-background-clip: text; -webkit-text-fill-color: transparent;">
                Sentiment & Topic Analyzer
            </span>
        </h1>
        <p style="font-size: 1.2rem; color: #64748b; max-width: 800px; margin: 0 auto;">
            Transform customer feedback into actionable insights with 
            <strong>Hugging Face Transformers</strong>, <strong>BERTopic</strong> & <strong>Groq LLM</strong>
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    # Sidebar
    new_df = render_sidebar()
    
    # Update dataframe if new data loaded
    if new_df is not None:
        st.session_state.df = new_df
        st.session_state.processed = False
        st.session_state.sentiment_analyzer = None
        st.session_state.topic_modeler = None
        st.session_state.wordclouds = {}
        st.session_state.summary = ''
        st.rerun()
    
    # Check if data is loaded
    if st.session_state.df is None:
        st.info("👈 Get started by selecting a data source in the sidebar!")
        
        # Feature showcase
        st.markdown("""
        <div style="display: grid; grid-template-columns: repeat(3, 1fr); gap: 1.5rem; margin-top: 2rem;">
            <div style="background: #f8fafc; padding: 1.5rem; border-radius: 15px; border-left: 4px solid #6366f1;">
                <h3 style="margin: 0 0 0.5rem 0;">💭 Sentiment Analysis</h3>
                <p style="color: #64748b; margin: 0;">DistilBERT-powered sentiment classification with aspect-based analysis</p>
            </div>
            <div style="background: #f8fafc; padding: 1.5rem; border-radius: 15px; border-left: 4px solid #8b5cf6;">
                <h3 style="margin: 0 0 0.5rem 0;">📚 Topic Modeling</h3>
                <p style="color: #64748b; margin: 0;">BERTopic/LDA for automatic topic discovery and emerging issue detection</p>
            </div>
            <div style="background: #f8fafc; padding: 1.5rem; border-radius: 15px; border-left: 4px solid #22c55e;">
                <h3 style="margin: 0 0 0.5rem 0;">🤖 AI Assistant</h3>
                <p style="color: #64748b; margin: 0;">Groq LLM-powered chat for natural language insights and recommendations</p>
            </div>
        </div>
        """, unsafe_allow_html=True)
        return
    
    df = st.session_state.df
    
    # Process button
    if not st.session_state.processed:
        st.divider()
        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            if st.button("🚀 Run Complete NLP Analysis", type="primary", use_container_width=True):
                progress_bar = st.progress(0, text="Starting analysis...")
                
                try:
                    processed_df = process_data(df, progress_bar)
                    st.session_state.df = processed_df
                    st.session_state.processed = True
                    st.success("✅ Analysis complete! Explore the tabs below.")
                    st.balloons()
                except Exception as e:
                    st.error(f"❌ Error during analysis: {str(e)}")
                    st.exception(e)
        return
    
    # Main tabs
    tabs = st.tabs([
        "📊 Overview & EDA",
        "💭 Sentiment Analysis", 
        "📚 Topic Modeling",
        "🤖 Ask Anything",
        "📋 Executive Summary"
    ])
    
    df = st.session_state.df
    
    with tabs[0]:
        render_overview_tab(df)
    
    with tabs[1]:
        render_sentiment_tab(df)
    
    with tabs[2]:
        render_topics_tab(df)
    
    with tabs[3]:
        render_chat_tab(df)
    
    with tabs[4]:
        render_summary_tab(df)

if __name__ == "__main__":
    main()
