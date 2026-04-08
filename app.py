import streamlit as st
import pandas as pd
import numpy as np
import string
import time
import re
import pickle
import os
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import nltk
from nltk.corpus import stopwords
import warnings
warnings.filterwarnings("ignore")

# ─── Page Config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="VibeCheck AI 🧠",
    page_icon="🎭",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ─── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>

@import url('https://fonts.googleapis.com/css2?family=Poppins:wght@300;400;600;700&display=swap');

* { font-family: 'Poppins', sans-serif; }

/* Main background */
.stApp{
background: linear-gradient(135deg,#000000,#1c1c1c);
color:white;
}

/* Animated gradient title */
    .hero-title {
        font-size: 3.2rem;
        font-weight: 700;
        text-align: center;
        background: linear-gradient(90deg, #6c63ff, #e91e8c, #ff6b35, #6c63ff);
        background-size: 300%;
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        animation: gradientShift 4s ease infinite;
        margin-bottom: 0.2rem;
    }
    @keyframes gradientShift {
        0%  { background-position: 0% 50%; }
        50% { background-position: 100% 50%; }
        100%{ background-position: 0% 50%; }
    }

/* Subtitle */
.hero-sub{
text-align:center;
color:#cfcfcf;
margin-bottom:2rem;
}

/* Cards */
.glass-card{
background:rgba(255,255,255,0.05);
border-radius:15px;
padding:1.5rem;
border:1px solid rgba(255,255,255,0.15);
backdrop-filter:blur(10px);
}

/* Sidebar */
section[data-testid="stSidebar"]{
background:#000000 !important;
}

section[data-testid="stSidebar"] *{
color:white !important;
}

/* Buttons */
.stButton>button{
background:linear-gradient(90deg,#8E2DE2,#E754B8) !important;
color:white !important;
border:none !important;
border-radius:30px !important;
padding:0.6rem 2rem !important;
font-weight:600 !important;
}

.stButton>button:hover{
box-shadow:0 0 20px rgba(142,45,226,0.6);
transform:scale(1.04);
}

/* Text area */
.stTextArea textarea{
background:rgba(255,255,255,0.07) !important;
border:1px solid #8E2DE2 !important;
color:white !important;
border-radius:10px !important;
}

/* Metrics */
.metric-val{
font-size:2rem;
font-weight:700;
color:#E7D889;
}

.metric-lbl{
color:#aaa;
}

/* Progress */
.stProgress > div > div{
background:linear-gradient(90deg,#8E2DE2,#E754B8);
}

</style>
""", unsafe_allow_html=True)

# ─── Emotion Config ────────────────────────────────────────────────────────────
EMOTION_CONFIG = {
    "joy":      {"emoji": "😄", "color": "#FFD700", "bg": "rgba(255,215,0,0.15)"},
    "sadness":  {"emoji": "😢", "color": "#4FC3F7", "bg": "rgba(79,195,247,0.15)"},
    "anger":    {"emoji": "😠", "color": "#EF5350", "bg": "rgba(239,83,80,0.15)"},
    "fear":     {"emoji": "😨", "color": "#AB47BC", "bg": "rgba(171,71,188,0.15)"},
    "love":     {"emoji": "❤️",  "color": "#EC407A", "bg": "rgba(236,64,122,0.15)"},
    "surprise": {"emoji": "😲", "color": "#26C6DA", "bg": "rgba(38,198,218,0.15)"},
}

# ─── NLTK Setup ────────────────────────────────────────────────────────────────
@st.cache_resource(show_spinner=False)
def load_nltk():
    try:
        nltk.download("stopwords", quiet=True)
        return set(stopwords.words("english"))
    except:
        return set()

stop_words = load_nltk()

# ─── Preprocessing ─────────────────────────────────────────────────────────────
def preprocess(txt):
    txt = txt.lower()
    txt = txt.translate(str.maketrans("", "", string.punctuation))
    txt = "".join(c for c in txt if not c.isdigit())
    txt = "".join(c for c in txt if c.isascii())
    words = txt.split()
    txt = " ".join(w for w in words if w not in stop_words)
    return txt

# ─── Model Training ────────────────────────────────────────────────────────────
@st.cache_resource(show_spinner=False)
def train_models(data_path="train.txt"):
    # Try loading from uploaded file or use demo data
    try:
        df = pd.read_csv(data_path, sep=";", header=None, names=["text", "emotion"])
    except FileNotFoundError:
        # Demo fallback with sample data for testing
        demo_texts = [
            "i feel so happy today amazing wonderful day",
            "i am so angry frustrated cannot believe this",
            "feeling so sad lonely and depressed today",
            "i love you so much you make me feel great",
            "i am scared terrified of what might happen",
            "wow what a surprise i never expected that",
            "this is the best day of my life so joyful",
            "i hate everything going wrong disaster",
            "missing you so much tears eyes crying",
            "darling you are my everything i adore you",
            "oh my god cannot believe it shocked amazed",
            "frightened nervous anxious about the result",
        ] * 50
        demo_labels = ["joy","anger","sadness","love","fear","surprise"] * 100
        df = pd.DataFrame({"text": demo_texts, "emotion": demo_labels})

    unique_emotions = df["emotion"].unique()
    emotion_map = {emo: i for i, emo in enumerate(unique_emotions)}
    inv_map = {v: k for k, v in emotion_map.items()}
    df["emotion"] = df["emotion"].map(emotion_map)
    df["text"] = df["text"].apply(preprocess)
    df["text"] = df["text"].astype(str)

    x_train, x_test, y_train, y_test = train_test_split(
        df["text"], df["emotion"], test_size=0.2, random_state=42
    )

    bow = CountVectorizer()
    x_train_bow = bow.fit_transform(x_train)
    x_test_bow  = bow.transform(x_test)

    tfidf = TfidfTransformer()
    x_train_tfidf = tfidf.fit_transform(x_train_bow)
    x_test_tfidf  = tfidf.transform(x_test_bow)

    # Naive Bayes (BOW)
    nb_bow = MultinomialNB()
    nb_bow.fit(x_train_bow, y_train)
    acc_nb_bow = accuracy_score(y_test, nb_bow.predict(x_test_bow))

    # Naive Bayes (TF-IDF)
    nb_tfidf = MultinomialNB()
    nb_tfidf.fit(x_train_tfidf, y_train)
    acc_nb_tfidf = accuracy_score(y_test, nb_tfidf.predict(x_test_tfidf))

    # Logistic Regression (TF-IDF)
    lr = LogisticRegression(max_iter=1000)
    lr.fit(x_train_tfidf, y_train)
    acc_lr = accuracy_score(y_test, lr.predict(x_test_tfidf))

    return {
        "bow": bow, "tfidf": tfidf,
        "nb_bow": nb_bow, "nb_tfidf": nb_tfidf, "lr": lr,
        "acc_nb_bow": acc_nb_bow, "acc_nb_tfidf": acc_nb_tfidf, "acc_lr": acc_lr,
        "inv_map": inv_map, "emotion_map": emotion_map,
        "x_test_tfidf": x_test_tfidf, "y_test": y_test,
        "df": df
    }

# ─── Sidebar ──────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🎭 VibeCheck AI")
    st.markdown("---")

    page = st.radio("📌 Navigate", ["🏠 Home & Predict", "📊 Model Analytics", "🔬 Pipeline Explorer", "ℹ️ About"])

    st.markdown("---")
    st.markdown("### ⚙️ Settings")
    model_choice = st.selectbox(
        "Choose Model",
        ["🏆 Logistic Regression (Best)", "🤖 Naive Bayes + TF-IDF", "🌱 Naive Bayes + BOW"]
    )
    st.markdown("---")

    # Upload custom data
    uploaded = st.file_uploader("📂 Upload train.txt", type=["txt"])
    if uploaded:
        with open("train.txt", "wb") as f:
            f.write(uploaded.read())
        st.success("✅ Data uploaded!")
        st.cache_resource.clear()

    st.markdown("---")
    st.markdown("<small style='color:#888'>Built with ❤️ using Scikit-learn + Streamlit</small>", unsafe_allow_html=True)

# ─── Load Models ──────────────────────────────────────────────────────────────
with st.spinner("🔄 Training models... please wait"):
    M = train_models()

# ─── Helper: Predict ──────────────────────────────────────────────────────────
def predict_emotion(text, model_key="lr"):
    cleaned = preprocess(text)
    bow_vec = M["bow"].transform([cleaned])
    tfidf_vec = M["tfidf"].transform(bow_vec)

    if model_key == "lr":
        model = M["lr"]
        proba = model.predict_proba(tfidf_vec)[0]
        pred  = model.predict(tfidf_vec)[0]
    elif model_key == "nb_tfidf":
        model = M["nb_tfidf"]
        proba = model.predict_proba(tfidf_vec)[0]
        pred  = model.predict(tfidf_vec)[0]
    else:
        model = M["nb_bow"]
        proba = model.predict_proba(bow_vec)[0]
        pred  = model.predict(bow_vec)[0]

    emotion_label = M["inv_map"].get(pred, "unknown")
    classes = model.classes_
    proba_dict = {M["inv_map"].get(c, str(c)): float(p) for c, p in zip(classes, proba)}
    return emotion_label, proba_dict

# ─── Page: Home & Predict ─────────────────────────────────────────────────────
if "🏠 Home & Predict" in page:
    st.markdown('<div class="hero-title">🎭 VibeCheck AI </div>', unsafe_allow_html=True)
    st.markdown('<div class="hero-sub">Real-time Emotion Detection from Text using NLP & Machine Learning</div>', unsafe_allow_html=True)

    # Stats row
    col1, col2, col3, col4 = st.columns(4)
    stats = [
        ("🎯", f"{M['acc_lr']*100:.1f}%", "Best Accuracy"),
        ("🧠", "3", "ML Models"),
        ("📝", "6", "Emotions Detected"),
        ("⚡", "TF-IDF", "Feature Method"),
    ]
    for col, (icon, val, lbl) in zip([col1,col2,col3,col4], stats):
        with col:
            st.markdown(f"""
            <div class="metric-box">
                <div style="font-size:1.8rem">{icon}</div>
                <div class="metric-val">{val}</div>
                <div class="metric-lbl">{lbl}</div>
            </div>""", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # Main predict area
    left, right = st.columns([1.2, 1])

    with left:
        st.markdown('<div class="glass-card">', unsafe_allow_html=True)
        st.markdown("### ✍️ Enter Your Text")
        user_input = st.text_area(
            "",
            placeholder="Type something like: 'I'm so excited about my new job!' 🚀",
            height=150,
            label_visibility="collapsed"
        )

        examples = ["I feel amazing today! Everything is going perfect! 🌟",
                    "I'm so angry, nothing is working out for me.",
                    "I miss you so much, this loneliness is unbearable 😢",
                    "I love you with all my heart ❤️",
                    "I'm terrified of what's going to happen next",
                    "Oh my god, I can't believe this just happened! 😲"]

        st.markdown("**💡 Try an example:**")
        ex_cols = st.columns(2)
        for i, ex in enumerate(examples[:4]):
            with ex_cols[i % 2]:
                if st.button(ex[:35]+"…", key=f"ex_{i}"):
                    user_input = ex
                    st.session_state["last_input"] = ex

        if st.button("🔍 Detect Emotion", key="predict_btn"):
            if user_input.strip():
                model_key = "lr" if "Logistic" in model_choice else ("nb_tfidf" if "TF-IDF" in model_choice else "nb_bow")
                with st.spinner("🧠 Analyzing emotion..."):
                    time.sleep(0.6)
                    emotion, proba = predict_emotion(user_input, model_key)
                    st.session_state["result"] = (emotion, proba, user_input)
            else:
                st.warning("⚠️ Please enter some text first!")

        st.markdown('</div>', unsafe_allow_html=True)

    with right:
        if "result" in st.session_state:
            emotion, proba, _ = st.session_state["result"]
            cfg = EMOTION_CONFIG.get(emotion, {"emoji":"🤔","color":"#aaa","bg":"rgba(200,200,200,0.1)"})

            st.markdown("### 🎯 Prediction Result")
            st.markdown(f"""
            <div class="glass-card" style="border-color:{cfg['color']}40; text-align:center">
                <div style="font-size:4rem; animation: popIn 0.5s ease">{cfg['emoji']}</div>
                <div class="emotion-badge" style="background:{cfg['bg']}; color:{cfg['color']}; border:2px solid {cfg['color']}40; margin-top:0.5rem">
                    {emotion.upper()}
                </div>
                <div style="color:#aaa; font-size:0.85rem; margin-top:0.8rem">
                    Confidence: <b style="color:{cfg['color']}">{proba.get(emotion,0)*100:.1f}%</b>
                </div>
            </div>""", unsafe_allow_html=True)

            # Probability bars
            st.markdown("**📊 Emotion Probabilities:**")
            sorted_proba = sorted(proba.items(), key=lambda x: x[1], reverse=True)
            for emo, prob in sorted_proba:
                c = EMOTION_CONFIG.get(emo, {}).get("color", "#aaa")
                emoji = EMOTION_CONFIG.get(emo, {}).get("emoji", "")
                st.markdown(f"""
                <div style="margin:0.3rem 0; display:flex; align-items:center; gap:0.5rem">
                    <span style="width:80px; font-size:0.8rem">{emoji} {emo}</span>
                    <div style="flex:1; background:rgba(255,255,255,0.08); border-radius:50px; height:10px; overflow:hidden">
                        <div style="width:{prob*100:.1f}%; height:100%; background:{c}; border-radius:50px; transition:width 1s ease"></div>
                    </div>
                    <span style="font-size:0.8rem; color:{c}; width:45px">{prob*100:.1f}%</span>
                </div>""", unsafe_allow_html=True)
        else:
            st.markdown("""
            <div class="glass-card" style="text-align:center; padding: 3rem 1rem">
                <div style="font-size:4rem">🎭</div>
                <div style="color:#aaa; margin-top:1rem">Your prediction will appear here</div>
                <div style="color:#666; font-size:0.85rem; margin-top:0.5rem">Type something and click Detect!</div>
            </div>""", unsafe_allow_html=True)

# ─── Page: Model Analytics ────────────────────────────────────────────────────
elif "📊 Model Analytics" in page:
    st.markdown('<div class="hero-title" style="font-size:2.2rem">📊 Model Analytics</div>', unsafe_allow_html=True)
    st.markdown('<div class="hero-sub">Compare models, visualize metrics & understand performance</div>', unsafe_allow_html=True)

    # Accuracy comparison
    st.markdown("### 🏆 Model Accuracy Comparison")
    models = ["Naive Bayes\n(BOW)", "Naive Bayes\n(TF-IDF)", "Logistic Reg\n(TF-IDF)"]
    accs   = [M["acc_nb_bow"], M["acc_nb_tfidf"], M["acc_lr"]]
    colors = ["#7b2ff7","#f953c6","#00c6ff"]

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    fig.patch.set_facecolor("#1a1a2e")

    # Bar chart
    ax = axes[0]
    ax.set_facecolor("#16213e")
    bars = ax.bar(models, [a*100 for a in accs], color=colors, width=0.5, edgecolor="none", zorder=3)
    for bar, acc in zip(bars, accs):
        ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.5, f"{acc*100:.1f}%",
                ha="center", va="bottom", color="white", fontweight="bold", fontsize=11)
    ax.set_ylim(0, 105)
    ax.set_ylabel("Accuracy (%)", color="white")
    ax.set_title("Model Accuracy", color="white", fontsize=13, fontweight="bold")
    ax.tick_params(colors="white")
    ax.spines[:].set_visible(False)
    ax.yaxis.grid(True, color="#333", linewidth=0.5)
    ax.set_axisbelow(True)

    # Horizontal comparison
    ax2 = axes[1]
    ax2.set_facecolor("#16213e")
    y_pos = range(len(models))
    hbars = ax2.barh([m.replace("\n"," ") for m in models], [a*100 for a in accs], color=colors, edgecolor="none")
    for bar, acc in zip(hbars, accs):
        ax2.text(bar.get_width()+0.3, bar.get_y()+bar.get_height()/2, f"{acc*100:.1f}%",
                va="center", color="white", fontsize=10, fontweight="bold")
    ax2.set_xlim(0, 110)
    ax2.set_xlabel("Accuracy (%)", color="white")
    ax2.set_title("Comparative View", color="white", fontsize=13, fontweight="bold")
    ax2.tick_params(colors="white")
    ax2.spines[:].set_visible(False)
    ax2.xaxis.grid(True, color="#333", linewidth=0.5)

    plt.tight_layout()
    st.pyplot(fig)

    # Confusion matrix
    st.markdown("### 🔷 Confusion Matrix (Logistic Regression)")
    y_pred_lr = M["lr"].predict(M["x_test_tfidf"])
    cm = confusion_matrix(M["y_test"], y_pred_lr)
    labels = [M["inv_map"].get(i, str(i)) for i in sorted(M["inv_map"].keys())]

    fig2, ax3 = plt.subplots(figsize=(8, 5))
    fig2.patch.set_facecolor("#1a1a2e")
    ax3.set_facecolor("#16213e")
    sns.heatmap(cm, annot=True, fmt="d", cmap="Purples",
                xticklabels=labels, yticklabels=labels,
                ax=ax3, linewidths=0.5, linecolor="#333",
                cbar_kws={"shrink": 0.8})
    ax3.set_xlabel("Predicted", color="white", fontsize=11)
    ax3.set_ylabel("Actual", color="white", fontsize=11)
    ax3.set_title("Confusion Matrix", color="white", fontsize=13, fontweight="bold")
    ax3.tick_params(colors="white")
    plt.tight_layout()
    st.pyplot(fig2)

    # Emotion distribution
    st.markdown("### 🌈 Emotion Distribution in Dataset")
    label_counts = M["df"]["emotion"].map(M["inv_map"]).value_counts()
    fig3, ax4 = plt.subplots(figsize=(8, 4))
    fig3.patch.set_facecolor("#1a1a2e")
    ax4.set_facecolor("#16213e")
    palette = [EMOTION_CONFIG.get(e, {}).get("color","#aaa") for e in label_counts.index]
    bars2 = ax4.bar(label_counts.index, label_counts.values, color=palette, edgecolor="none")
    for bar, val in zip(bars2, label_counts.values):
        ax4.text(bar.get_x()+bar.get_width()/2, bar.get_height()+20, str(val),
                ha="center", color="white", fontsize=10, fontweight="bold")
    ax4.set_ylabel("Count", color="white")
    ax4.set_title("Emotion Class Distribution", color="white", fontsize=13, fontweight="bold")
    ax4.tick_params(colors="white")
    ax4.spines[:].set_visible(False)
    ax4.yaxis.grid(True, color="#333", linewidth=0.5)
    plt.tight_layout()
    st.pyplot(fig3)

# ─── Page: Pipeline Explorer ──────────────────────────────────────────────────
elif "🔬 Pipeline Explorer" in page:
    st.markdown('<div class="hero-title" style="font-size:2.2rem">🔬 NLP Pipeline Explorer</div>', unsafe_allow_html=True)
    st.markdown('<div class="hero-sub">See how raw text transforms step-by-step through the pipeline</div>', unsafe_allow_html=True)

    demo_text = st.text_input("🔤 Enter text to trace through pipeline:",
                              value="I am feeling SO happy today!!! 😊 #blessed 123")

    if demo_text:
        steps = []

        # Step 1
        steps.append(("1️⃣  Raw Input", demo_text, "Original unprocessed text"))

        # Step 2
        lower = demo_text.lower()
        steps.append(("2️⃣  Lowercase", lower, "Convert all characters to lowercase"))

        # Step 3
        no_punc = lower.translate(str.maketrans("","",string.punctuation))
        steps.append(("3️⃣  Remove Punctuation", no_punc, "Strip all punctuation marks"))

        # Step 4
        no_nums = "".join(c for c in no_punc if not c.isdigit())
        steps.append(("4️⃣  Remove Numbers", no_nums, "Remove all numeric characters"))

        # Step 5
        ascii_only = "".join(c for c in no_nums if c.isascii())
        steps.append(("5️⃣  Remove Emojis/Non-ASCII", ascii_only, "Keep only ASCII characters"))

        # Step 6
        words = ascii_only.split()
        no_stop = " ".join(w for w in words if w not in stop_words)
        removed = [w for w in words if w in stop_words]
        steps.append(("6️⃣  Remove Stopwords", no_stop,
                       f"Removed: {removed if removed else 'none'} | Stop words filtered using NLTK"))

        # Step 7: BOW
        bow_vec = M["bow"].transform([no_stop])
        nnz = bow_vec.nnz
        steps.append(("7️⃣  Bag of Words (BOW)", f"Sparse matrix with {nnz} non-zero features",
                       "CountVectorizer converts text to word frequency vector"))

        # Step 8: TF-IDF
        tfidf_vec = M["tfidf"].transform(bow_vec)
        steps.append(("8️⃣  TF-IDF Transformation", f"TF-IDF vector (shape: {tfidf_vec.shape})",
                       "Weights words by importance across all documents"))

        for step, value, desc in steps:
            st.markdown(f"""
            <div class="glass-card">
                <div style="display:flex; align-items:flex-start; gap:1rem">
                    <div>
                        <div style="font-size:1rem; font-weight:600; color:#00e5ff">{step}</div>
                        <div style="color:#aaa; font-size:0.8rem; margin-top:0.2rem">💬 {desc}</div>
                        <div style="background:rgba(0,0,0,0.3); border-radius:8px; padding:0.5rem 1rem; margin-top:0.6rem; font-family:monospace; color:#f0f0f0; font-size:0.95rem; border-left: 3px solid #7b2ff7">
                            {value}
                        </div>
                    </div>
                </div>
            </div>""", unsafe_allow_html=True)

# ─── Page: About ──────────────────────────────────────────────────────────────
elif "ℹ️ About" in page:
    st.markdown('<div class="hero-title" style="font-size:2.2rem">ℹ️ About This Project</div>', unsafe_allow_html=True)

    st.markdown("""
    <div class="glass-card">
        <h3>🎯 Project Overview</h3>
        <p>EmotiSense AI is an <b>NLP-based Emotion Detection system</b> that classifies text into 6 human emotions.
        It demonstrates a complete ML pipeline — from raw text preprocessing to model training and real-time inference.</p>
    </div>

    <div class="glass-card">
        <h3>🔧 Tech Stack</h3>
        <div style="display:flex; flex-wrap:wrap; gap:0.6rem; margin-top:0.5rem">
            <span style="background:rgba(123,47,247,0.3); padding:0.3rem 0.8rem; border-radius:50px; font-size:0.85rem">🐍 Python</span>
            <span style="background:rgba(0,198,255,0.2); padding:0.3rem 0.8rem; border-radius:50px; font-size:0.85rem">📦 Scikit-learn</span>
            <span style="background:rgba(249,83,198,0.2); padding:0.3rem 0.8rem; border-radius:50px; font-size:0.85rem">🌿 NLTK</span>
            <span style="background:rgba(255,215,0,0.2); padding:0.3rem 0.8rem; border-radius:50px; font-size:0.85rem">🐼 Pandas</span>
            <span style="background:rgba(255,100,100,0.2); padding:0.3rem 0.8rem; border-radius:50px; font-size:0.85rem">📊 Matplotlib / Seaborn</span>
            <span style="background:rgba(100,200,100,0.2); padding:0.3rem 0.8rem; border-radius:50px; font-size:0.85rem">⚡ Streamlit</span>
        </div>
    </div>

    <div class="glass-card">
        <h3>🧪 Models Used</h3>
        <table style="width:100%; border-collapse:collapse; font-size:0.9rem">
            <tr style="background:rgba(255,255,255,0.08)">
                <th style="padding:0.5rem; text-align:left">Model</th>
                <th style="padding:0.5rem; text-align:left">Vectorizer</th>
                <th style="padding:0.5rem; text-align:left">Best For</th>
            </tr>
            <tr>
                <td style="padding:0.5rem">Multinomial Naive Bayes</td>
                <td style="padding:0.5rem">Bag of Words (BOW)</td>
                <td style="padding:0.5rem">Baseline / Speed</td>
            </tr>
            <tr style="background:rgba(255,255,255,0.04)">
                <td style="padding:0.5rem">Multinomial Naive Bayes</td>
                <td style="padding:0.5rem">TF-IDF</td>
                <td style="padding:0.5rem">Improved baseline</td>
            </tr>
            <tr>
                <td style="padding:0.5rem">Logistic Regression ⭐</td>
                <td style="padding:0.5rem">TF-IDF</td>
                <td style="padding:0.5rem">Best accuracy</td>
            </tr>
        </table>
    </div>

    <div class="glass-card">
        <h3>📁 Project Files</h3>
        <div style="font-family:monospace; font-size:0.9rem; color:#aaa">
            📂 project/<br>
            &nbsp;&nbsp;├── 📓 index.ipynb &nbsp;&nbsp;&nbsp;&nbsp;# Training notebook<br>
            &nbsp;&nbsp;├── 🐍 app.py &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;# This Streamlit app<br>
            &nbsp;&nbsp;├── 📄 train.txt &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;# Dataset<br>
            &nbsp;&nbsp;├── 💾 model.pkl &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;# Saved LR model<br>
            &nbsp;&nbsp;└── 💾 vectorizer.pkl # Saved vectorizer
        </div>
    </div>
    """, unsafe_allow_html=True)