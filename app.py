# app.py - Streamlit app cho mô hình DistilBERT vs VADER

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# Load mô hình từ Hugging Face Hub
@st.cache_resource
def load_model():
    model_name = "nlptown/bert-base-multilingual-uncased-sentiment"
    model = AutoModelForSequenceClassification.from_pretrained(model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    return model.to(device), tokenizer

# Thiết bị
device = "cuda" if torch.cuda.is_available() else "cpu"
model, tokenizer = load_model()
model.eval()

# Hàm làm sạch văn bản
def clean_text(text):
    import re
    return re.sub(r'[^a-zA-Z0-9\s]', '', text.lower())

# Dự đoán với DistilBERT
@st.cache_data

def predict_distilbert(text):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding="max_length", max_length=128).to(device)
    with torch.no_grad():
        outputs = model(**inputs)
        probs = torch.nn.functional.softmax(outputs.logits, dim=-1)
        return torch.argmax(probs).item() + 1

def convert_star_to_label(star):
    if star <= 2:
        return 'negative'
    elif star == 3:
        return 'neutral'
    else:
        return 'positive'

# Dự đoán với VADER
vader = SentimentIntensityAnalyzer()
def vader_sentiment(text, threshold=0.25):
    score = vader.polarity_scores(text)['compound']
    if score >= threshold:
        return 'positive'
    elif score <= 0:
        return 'negative'
    else:
        return 'neutral'

# 1. Tiêu đề app
st.set_page_config(page_title="Sentiment Analysis: DistilBERT vs VADER", layout="wide")
st.title("🔍 So sánh mô hình cảm xúc: DistilBERT vs VADER")

# 2. Tải dữ liệu
uploaded_file = st.file_uploader("📂 Tải lên file dulieu2.csv", type=["csv"])
if uploaded_file:
    data = pd.read_csv(uploaded_file)
    data['cleaned_text'] = data['review'].astype(str).apply(clean_text)
    data['true_label'] = data['rating'].apply(lambda r: convert_star_to_label(r))

    # Dự đoán cảm xúc
    st.spinner("Đang phân tích cảm xúc với 2 mô hình...")
    data['vader_label'] = data['cleaned_text'].apply(lambda x: vader_sentiment(x))
    data['bert_star'] = data['cleaned_text'].apply(predict_distilbert)
    data['bert_label'] = data['bert_star'].apply(convert_star_to_label)

    st.success("✅ Dự đoán hoàn tất!")

    # Tabs giao diện
    tab1, tab2, tab3 = st.tabs(["📊 Tổng quan", "📈 So sánh mô hình", "📝 Dự đoán từng văn bản"])

    with tab1:
        st.subheader("1. Thống kê cơ bản")
        st.write("**5 dòng đầu:**")
        st.dataframe(data.head())
        st.write("**Tổng số bản ghi:**", len(data))

        st.write("### Phân bố rating")
        fig1, ax1 = plt.subplots()
        sns.countplot(x='rating', data=data, ax=ax1)
        st.pyplot(fig1)

    with tab2:
        st.subheader("2. So sánh kết quả dự đoán")
        acc_vader = accuracy_score(data['true_label'], data['vader_label'])
        acc_bert = accuracy_score(data['true_label'], data['bert_label'])

        st.metric("🎯 Accuracy VADER", f"{acc_vader:.2%}")
        st.metric("🤖 Accuracy DistilBERT", f"{acc_bert:.2%}")

        st.write("### Ma trận nhầm lẫn")
        fig2, ax2 = plt.subplots(1, 2, figsize=(12, 5))
        sns.heatmap(confusion_matrix(data['true_label'], data['vader_label'], labels=['negative','neutral','positive']),
                    annot=True, cmap='Blues', fmt='d', ax=ax2[0])
        ax2[0].set_title("VADER")

        sns.heatmap(confusion_matrix(data['true_label'], data['bert_label'], labels=['negative','neutral','positive']),
                    annot=True, cmap='YlOrBr', fmt='d', ax=ax2[1])
        ax2[1].set_title("DistilBERT")
        st.pyplot(fig2)

    with tab3:
        st.subheader("3. Dự đoán văn bản mới")
        text_input = st.text_area("Nhập văn bản đánh giá:", "Sản phẩm này rất tốt và đáng mua!")
        true_label = st.selectbox("Nhãn thật (nếu biết):", ["positive", "neutral", "negative"], index=0)

        if st.button("Phân tích"):
            clean = clean_text(text_input)
            vader_result = vader_sentiment(clean)
            bert_star = predict_distilbert(clean)
            bert_result = convert_star_to_label(bert_star)

            st.write(f"**VADER:** {vader_result} {'✅' if vader_result == true_label else '❌'}")
            st.write(f"**DistilBERT:** {bert_result} ({bert_star} sao) {'✅' if bert_result == true_label else '❌'}")
