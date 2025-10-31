import streamlit as st
import pickle

# Inject powerful dark-mode CSS and professional styling
st.markdown("""
<style>
html, body, .stApp { 
    background: linear-gradient(135deg, #0f2027 0%, #2c5364 100%);
    color: #F3F6F9;
    font-family: 'Montserrat', Arial, sans-serif;
}
.main-header {
    font-size: 2.8rem;
    font-weight: 700;
    color: #F8A623;
    letter-spacing: 1.5px;
    text-align: center;
    margin: 35px 0 15px 0;
    text-shadow: 1px 1px 14px #111;
}
.sub-header {
    text-align: center;
    color: #7FDBFF;
    margin-bottom: 38px;
    font-size: 1.28rem;
    font-weight: 500;
    letter-spacing: .8px;
}
.stTextArea textarea {
    background: #222d32;
    color: #F3F6F9;
    font-size: 1.15rem;
    border: 2px solid #F8A623;
    border-radius: 10px;
}
.stButton>button {
    background-color: #F8A623;
    color: #212D38;
    font-size: 1.1rem;
    font-weight: bold;
    border-radius: 12px;
    border: none;
    box-shadow: 0 2px 8px rgba(230,170,70,0.14);
    transition: background 0.2s;
}
.stButton>button:hover {
    background-color: #FFB84D;
    color: #141E30;
}
.result-card {
    margin-top: 36px;
    padding: 30px 36px;
    background: #212D38;
    border-radius: 20px;
    box-shadow: 0 8px 36px rgba(44,62,80, 0.39);
    text-align: center;
    border-left: 6px solid #F8A623;
}
.strong-pos { color: #26D962; font-size: 2.5rem; }
.strong-neg { color: #D92646; font-size: 2.5rem; }
.strong-neutral { color: #F8A623; font-size: 2.5rem; }
.confidence-score {
    font-size: 1.08rem; 
    font-weight: 600; 
    color: #7FDBFF; 
    margin-top: 16px;
}
</style>
<link href="https://fonts.googleapis.com/css?family=Montserrat:700,600&display=swap" rel="stylesheet">
""", unsafe_allow_html=True)

# ====== Load Models on Startup ======
@st.cache_resource
def load_models():
    with open("sentiment_analysis_model.pkl", "rb") as f:
        classifier = pickle.load(f)
    with open("word_features.pkl", "rb") as f:
        w_features = pickle.load(f)
    return classifier, w_features

classifier, w_features = load_models()

def extract_features(document, w_features):
    document_words = set(document)
    features = {}
    for word in w_features:
        features[f'contains({word})'] = (word in document_words)
    return features

def classify_tweet(text):
    words_cleaned = [
        word for word in text.lower().split()
        if len(word) >= 3 and
        'http' not in word and
        not word.startswith('@') and
        not word.startswith('#') and
        word != 'rt'
    ]
    features = extract_features(words_cleaned, w_features)
    sentiment = classifier.classify(features)
    try:
        prob_dist = classifier.prob_classify(features)
        confidence = prob_dist.prob(sentiment)
    except Exception:
        confidence = None
    return sentiment, confidence

# ======= HEADER =======
st.markdown("<div class='main-header'>🚀 Twitter Sentiment Analyzer</div>", unsafe_allow_html=True)
st.markdown("<div class='sub-header'>Get bold, actionable sentiment insights from your tweets</div>", unsafe_allow_html=True)

tweet = st.text_area("Paste tweet text here:", "", height=150)

analyze_btn = st.button("Analyze Sentiment")

if analyze_btn:
    if tweet.strip():
        with st.spinner("🚥 Crunching your tweet..."):
            sentiment, confidence = classify_tweet(tweet)
        # Strong color and icon for sentiment
        if sentiment == "Positive":
            cls = "strong-pos"
            icon = "🚀👍"
        elif sentiment == "Negative":
            cls = "strong-neg"
            icon = "⛔️😡"
        else:
            cls = "strong-neutral"
            icon = "⚡️😐"
        st.markdown(
            f"""
            <div class='result-card'>
                <div class='{cls}'>{icon}<br>{sentiment}</div>
                <div class='confidence-score'>Confidence: {confidence:.2f}</div>
                <div style='margin-top:16px;font-size:1.05rem;'><b>Your input:</b></div>
                <div style='background:#293250;border-radius:9px;padding:12px;margin-top:7px;'>{tweet}</div>
            </div>
            """,
            unsafe_allow_html=True
        )
    else:
        st.warning("Please enter tweet text to analyze.")

st.markdown("<p style='text-align:center;font-size:1rem;color:#8D99AE;margin-top:18px;'>© 2025 Twitter Sentiment Analyzer · Strong UI Mode</p>", unsafe_allow_html=True)
