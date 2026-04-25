import streamlit as st
import pickle
import re
import string

# Load models
with open('tfidf_vectorizer.pkl', 'rb') as f:
    tfidf = pickle.load(f)

with open('logistic_model.pkl', 'rb') as f:
    model = pickle.load(f)

# Clean text function
def clean_text(text):
    text = text.lower()
    text = re.sub(r'\[.*?\]', '', text)
    text = re.sub(r'https?://\S+|www\.\S+', '', text)
    text = re.sub(r'<.*?>+', '', text)
    text = re.sub(r'[%s]' % re.escape(string.punctuation), '', text)
    text = re.sub(r'\n', '', text)
    text = re.sub(r'\w*\d\w*', '', text)
    return text

# Prediction function
def predict_news(text):
    cleaned = clean_text(text)
    vec = tfidf.transform([cleaned])
    conf = model.predict_proba(vec)[0]
    fake_conf = conf[0] * 100
    real_conf = conf[1] * 100

    if fake_conf >= 65:
        verdict = "FAKE"
    elif real_conf >= 45:
        verdict = "REAL"
    else:
        verdict = "UNVERIFIED"

    return verdict, fake_conf, real_conf

# Page config
st.set_page_config(
    page_title="Fake News Detector",
    page_icon="📰",
    layout="centered"
)

# Header
st.title("📰 Fake News Detection System")
st.subheader("Powered by Machine Learning + NLP")
st.info("ℹ️ Best results on political and general news articles.")
st.markdown("---")

# Input
news_input = st.text_area(
    "📝 Paste news article here:",
    height=250,
    placeholder="Paste any news article here..."
)

col1, col2 = st.columns([1, 1])
with col1:
    detect_btn = st.button("🔍 Detect", use_container_width=True)
with col2:
    clear_btn = st.button("🗑️ Clear", use_container_width=True)

if detect_btn:
    if news_input.strip() == "":
        st.warning("⚠️ Please paste a news article first!")
    else:
        with st.spinner("Analyzing article..."):
            verdict, fake_conf, real_conf = predict_news(news_input)

        st.markdown("---")
        st.subheader("📊 Result")

        if verdict == "FAKE":
            st.error("🚨 FAKE NEWS DETECTED!")
        elif verdict == "REAL":
            st.success("✅ REAL NEWS!")
        else:
            st.warning("⚠️ UNVERIFIED — Cannot determine clearly")

        # Metrics
        col3, col4, col5 = st.columns(3)
        with col3:
            st.metric("🔴 Fake", f"{fake_conf:.2f}%")
        with col4:
            st.metric("🟢 Real", f"{real_conf:.2f}%")
        with col5:
            st.metric("📋 Verdict", verdict)

        # Progress bar
        st.markdown("#### Confidence")
        st.progress(float(real_conf/100))

# Sample news
st.markdown("---")
st.markdown("#### 💡 Try These Samples")

col6, col7 = st.columns(2)
with col6:
    st.markdown("**✅ Real News:**")
    st.code("The Federal Reserve raised interest rates by 0.25% following inflation concerns reported by Reuters.")

with col7:
    st.markdown("**🚨 Fake News:**")
    st.code("SHOCKING: Bill Gates microchips found in COVID vaccines, whistleblower reveals secret agenda.")

# Footer
st.markdown("---")
st.markdown("Built with ❤️ | Accuracy: 94.64% | Dataset: 71,537 articles")