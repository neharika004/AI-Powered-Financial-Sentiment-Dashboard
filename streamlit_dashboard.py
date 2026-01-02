import streamlit as st
import requests
import numpy as np
import time
import matplotlib.pyplot as plt
import pandas as pd

# Function to get live financial news from the Flask API
@st.cache_data
def get_live_news():
    try:
        url = 'http://localhost:5000/financial-news'  # Assuming Flask runs on localhost:5000
        response = requests.get(url)
        response.raise_for_status()  # Raise an error for bad responses
        return response.json()
    except Exception as e:
        st.error(f"Error fetching news: {e}")
        return None

# Function to interpret sentiment
def interpret_sentiment(sentiment_value):
    if sentiment_value == 0.0:
        return 'Positive'
    elif sentiment_value == 1.0:
        return 'Neutral'
    elif sentiment_value == 2.0:
        return 'Negative'
    else:
        return 'Unknown'

# Streamlit App Layout
st.title("Real-Time Financial News Sentiment Dashboard")
st.subheader("Live Financial News with Sentiment Analysis")

# Auto-refresh every 10 seconds or on pagination change
refresh_interval = 10

# Add a selectbox for filtering news by sentiment
sentiment_filter = st.selectbox('Filter by Sentiment', ('All', 'Positive', 'Negative', 'Neutral'))

# Placeholder for the news and sentiment
placeholder = st.empty()

# Get the live news data
news_data = get_live_news()

if news_data:
    news_sentiments = []
    
    # Process sentiment for each news item
    for article in news_data:
        title = article['title']
        sentiment_value = article['sentiment']
        sentiment = interpret_sentiment(sentiment_value)
        news_sentiments.append({'title': title, 'sentiment': sentiment})
    
    # Filter news by sentiment
    if sentiment_filter != 'All':
        news_sentiments = [news for news in news_sentiments if news['sentiment'] == sentiment_filter]
    
    # Pagination setup
    page_size = 10  # Show 10 news at a time
    total_news = len(news_sentiments)
    total_pages = (total_news + page_size - 1) // page_size  # Total pages based on news count

    # Check if there are any pages to display
    if total_pages > 0:
        page = st.number_input('Page', 1, total_pages, 1, key='page_number_input') - 1  # Select page number (1-indexed)

        # Display news for the current page
        start_index = page * page_size
        end_index = start_index + page_size
        current_page_news = news_sentiments[start_index:end_index]

        # Display the news in the placeholder container
        with placeholder.container():
            for news in current_page_news:
                st.write(f"**Title**: {news['title']}")
                st.write(f"**Sentiment**: {news['sentiment']}")
                st.write("---")
        
        # Improved sentiment distribution chart
        st.subheader("Sentiment Distribution")

        # Prepare data for chart
        sentiment_df = pd.DataFrame(news_sentiments)
        sentiment_counts = sentiment_df['sentiment'].value_counts()

        # Create a pie chart using matplotlib
        fig, ax = plt.subplots()
        ax.pie(sentiment_counts, labels=sentiment_counts.index, autopct='%1.1f%%', startangle=90, colors=['#66b3ff', '#99ff99', '#ff9999'])
        ax.axis('equal')  # Equal aspect ratio ensures that pie chart is drawn as a circle.

        # Display the pie chart
        st.pyplot(fig)
    else:
        st.write("No news available for the selected sentiment filter.")
    
    # Sleep for auto-refresh
    time.sleep(refresh_interval)
else:
    st.write("No news available at the moment.")
