import os
import random

import pandas as pd
import numpy as np
from nltk import word_tokenize 
from nltk.util import ngrams
from nltk.corpus import stopwords
from textblob import TextBlob
import streamlit as st
import matplotlib.pyplot as plt
import plotly.express as px
from transformers import pipeline
import tokenizers
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import NMF
from sklearn.pipeline import make_pipeline
    
# CSS to inject contained in a string
hide_dataframe_row_index = """
            <style>
            .row_heading.level0 {display:none}
            .blank {display:none}
            </style>
            """

# Inject CSS with Markdown
st.markdown(hide_dataframe_row_index, unsafe_allow_html=True)

if 'sentiments_run' not in st.session_state:
    st.session_state['sentiments_run'] = False
    st.session_state['sentiments'] = ''


st.title('App Sentiment Analysis')

 

st.header("Choose a CSV file ")
with st.expander("What kind of file do i need to upload ?"):
            st.write("""
                A CSV file that contains data with column names app, year, comments.
            """)  
uploaded_file = st.file_uploader('File', type = 'csv')

if uploaded_file is not None:
    try:
        app_df = pd.read_csv(uploaded_file,index_col=False)
        ref_columns = pd.DataFrame(app_df.columns,columns = ['headers'])
    except:
        st.error(' Error loading file. Is it a CSV file with the mandantory columns :app,year,comments ?')
        st.stop()
    if not uploaded_file.name.endswith('.csv') or not set(['app','year','comments']).issubset(list(ref_columns['headers'])):
        st.error(' Error loading file. Please use a CSV file with the mandantory columns : app,year,comments ?')
        st.stop()
    
    with st.spinner('Finding Sentiments using transformer models...'):

        if st.session_state['sentiments_run'] == False :
            sentiments = pipeline("sentiment-analysis")(list(app_df['comments']))
            sentiment_list = []
            for sentiment in sentiments:
                sentiment_list.append(sentiment['label'])
            st.session_state['sentiments_run'] =  True
            st.session_state['sentiments'] = sentiment_list

    st.subheader('Base Records')
    col1, col2, col3 = st.columns([10,10,10])
    col1.metric("No of records", len(app_df))
    col2.metric("Play store", len(app_df[app_df['app']=='play_store']))
    col3.metric("App store", len(app_df[app_df['app']=='app_store']))

    st.success('We are done ! Lets analyze this further...')
    # For testing
    #st.session_state['sentiments'] = [random.choice(['POSITIVE', 'NEGATIVE']) 
    #                                 for i in range(len(app_df))]
    app_df['sentiment'] = st.session_state['sentiments']
    
    app_df = app_df[['date', 'ratings', 'comments', 'app', 'year','sentiment']]

    #selected_options = st.multiselect('', app_df['year'].unique())
    years = app_df['year'].unique()
    #st.header("Which year you want to analyze sentiments ?")
    selected_yr_options = st.sidebar.multiselect(
    "Select year ",
    years
    )
    all_years = st.sidebar.checkbox("Select all years")

    selected_app_options = st.sidebar.multiselect(
    "Play or App store ",
    app_df['app'].unique()
    )
    all_data = st.sidebar.checkbox("Select all data")

    selected_analysis_type = st.sidebar.multiselect(
    "Word Freq or Topic Modeling ",
    ['Word Freq', 'Topic Modeling']
    )
    all_analysis = st.sidebar.checkbox("Select all analysis options")
    
    if selected_yr_options == [] or all_years :
        filter_df = app_df[app_df['year'].isin(app_df['year'].unique())]
    else :
        filter_df = app_df[app_df['year'].isin(selected_yr_options)]
    
    if selected_app_options == [] or all_data :
        filter_df = filter_df[app_df['app'].isin(app_df['app'].unique())]
    else:
        filter_df = filter_df[app_df['app'].isin(selected_app_options)]

    st.subheader('Filtered Records')
    col1, col2, col3 = st.columns([10,10,10])
    col1.metric("No of records", len(filter_df))
    col2.metric("Play store", len(filter_df[filter_df['app']=='play_store']))
    col3.metric("App store", len(filter_df[filter_df['app']=='app_store']))

    st.dataframe(filter_df)
    fig = px.pie(filter_df, values='year', names='sentiment', title='Sentiment breakup')
    st.plotly_chart(fig)

    
    neg_comments_df = filter_df[filter_df['sentiment'] == 'NEGATIVE']
    pos_comments_df = filter_df[filter_df['sentiment'] == 'POSITIVE']

    stoplist = stopwords.words('english')

    def display_word_frequency(df_text_column):

        

        c_vec = CountVectorizer(stop_words=stoplist, ngram_range=(2,3))

        # matrix of ngrams
        ngrams = c_vec.fit_transform(df_text_column)
        # count frequency of ngrams
        count_values = ngrams.toarray().sum(axis=0)
        # list of ngrams
        vocab = c_vec.vocabulary_
        freq_df = pd.DataFrame(sorted([(count_values[i],k) for k,i in vocab.items()], reverse=True)
                    ).rename(columns={0: 'frequency', 1:'bigram/trigram'})
        

        return freq_df
    if 'Word Freq' in selected_analysis_type or all_analysis: #['Word Freq', 'Topic Modeling']
        with st.expander("About Word Frequency"):
                st.write("""
                    Here we are generating most frequent words in a sentiment after 
                    cleaning up for common *stop* words.
                """)

        col1, col2 = st.columns([10,10])

        with col1:
            st.subheader('Positive Comments Word Frequency')
            st.dataframe(display_word_frequency(pos_comments_df['comments']))

        with col2:
            st.subheader("Negative Comments Word Frequency")
            st.dataframe(display_word_frequency(neg_comments_df['comments']))
        
    
    def print_top_words(df_text_column,n_top_words):

        tfidf_vectorizer = TfidfVectorizer(stop_words=stoplist, ngram_range=(2,3))
        
        nmf = NMF(n_components=7,max_iter=1000)
        pipe = make_pipeline(tfidf_vectorizer, nmf)
        pipe.fit(df_text_column) # change here
        topic_df = pd.DataFrame(columns = ['Topics'])
        feature_names = tfidf_vectorizer.get_feature_names()

        for topic_idx, topic in enumerate(nmf.components_):
            message = ''
            message = "Topic #%d: " % topic_idx
            message += " ~ ".join([feature_names[i]
                                for i in topic.argsort()[:-n_top_words - 1:-1]])
            #message = message + '\n'
            print(message.strip())
            topic_df.loc[len(topic_df)] = [message.lstrip()]
        return topic_df

    if 'Topic Modeling' in selected_analysis_type or all_analysis: #['Word Freq', 'Topic Modeling']
        with st.expander("About Topic Modeling"):
            st.write("""
                Here we are generating Topic Models using non negative 
                matrix factorization (NMF).
            """)    

        col1, col2 = st.columns([10,10])

        with col1:
            st.subheader("Positive Comments Topic Modeling")
            st.dataframe(print_top_words(pos_comments_df['comments'],n_top_words=3))

        with col2:
            st.subheader("Negative Comments Topic Modeling")
            st.dataframe(print_top_words(neg_comments_df['comments'],n_top_words=5))
