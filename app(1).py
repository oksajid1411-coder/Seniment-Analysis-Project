import streamlit as st
import joblib
import pandas as pd
from pathlib import Path
import time

# ============================================================
# PAGE CONFIG
# ============================================================
st.set_page_config(
    page_title="Sentiment Analyzer",
    page_icon="💭",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ============================================================
# CUSTOM CSS — MODERN ANIMATED UI
# ============================================================
st.markdown(
    """
    <style>
    /* ---------- Global ---------- */
    .stApp {
        background:
            radial-gradient(circle at 10% 10%, rgba(99,102,241,.12), transparent 28%),
            radial-gradient(circle at 90% 20%, rgba(168,85,247,.12), transparent 28%),
            linear-gradient(135deg, #f8fafc 0%, #eef2ff 50%, #faf5ff 100%);
    }

    [data-testid="stHeader"] {
        background: rgba(255,255,255,0);
    }

    /* ---------- Animated background blobs ---------- */
    .blob {
        position: fixed;
        width: 260px;
        height: 260px;
        border-radius: 50%;
        filter: blur(70px);
        opacity: .25;
        z-index: -1;
        animation: floatBlob 10s ease-in-out infinite alternate;
    }

    .blob.one {
        top: 8%;
        left: -70px;
        background: #6366f1;
    }

    .blob.two {
        right: -80px;
        top: 45%;
        background: #a855f7;
        animation-delay: 2s;
    }

    .blob.three {
        left: 35%;
        bottom: -120px;
        background: #06b6d4;
        animation-delay: 4s;
    }

    @keyframes floatBlob {
        from { transform: translate(0, 0) scale(1); }
        to { transform: translate(35px, -25px) scale(1.12); }
    }

    /* ---------- Hero ---------- */
    .hero {
        padding: 35px 20px 25px;
        text-align: center;
        animation: heroIn .9s ease-out both;
    }

    .hero-icon {
        display: inline-flex;
        width: 76px;
        height: 76px;
        align-items: center;
        justify-content: center;
        border-radius: 24px;
        font-size: 38px;
        background: linear-gradient(135deg, #6366f1, #a855f7);
        box-shadow: 0 18px 45px rgba(99,102,241,.30);
        animation: iconFloat 3s ease-in-out infinite;
    }

    .hero-title {
        margin: 18px 0 5px;
        font-size: clamp(2.3rem, 5vw, 4rem);
        font-weight: 850;
        letter-spacing: -2px;
        background: linear-gradient(90deg, #4f46e5, #7c3aed, #9333ea);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        animation: titleGlow 3s ease-in-out infinite alternate;
    }

    .hero-subtitle {
        color: #64748b;
        font-size: 1.08rem;
        margin: 0 auto;
        max-width: 650px;
    }

    @keyframes heroIn {
        from { opacity: 0; transform: translateY(-25px); }
        to { opacity: 1; transform: translateY(0); }
    }

    @keyframes iconFloat {
        0%,100% { transform: translateY(0) rotate(0deg); }
        50% { transform: translateY(-8px) rotate(2deg); }
    }

    @keyframes titleGlow {
        from { filter: drop-shadow(0 0 0 rgba(99,102,241,0)); }
        to { filter: drop-shadow(0 8px 22px rgba(99,102,241,.18)); }
    }

    /* ---------- Glass cards ---------- */
    .glass-card {
        background: rgba(255,255,255,.72);
        border: 1px solid rgba(255,255,255,.85);
        border-radius: 22px;
        padding: 24px;
        box-shadow: 0 18px 50px rgba(15,23,42,.08);
        backdrop-filter: blur(16px);
        -webkit-backdrop-filter: blur(16px);
        animation: cardIn .7s ease-out both;
    }

    .glass-card:hover {
        transform: translateY(-3px);
        box-shadow: 0 24px 60px rgba(15,23,42,.12);
        transition: .25s ease;
    }

    @keyframes cardIn {
        from { opacity: 0; transform: translateY(18px); }
        to { opacity: 1; transform: translateY(0); }
    }

    /* ---------- Result card ---------- */
    .result-card {
        position: relative;
        overflow: hidden;
        border-radius: 22px;
        padding: 25px;
        text-align: center;
        color: white;
        background: linear-gradient(135deg, #4f46e5, #7c3aed, #9333ea);
        box-shadow: 0 20px 45px rgba(79,70,229,.25);
        animation: resultIn .65s cubic-bezier(.2,.8,.2,1) both;
    }

    .result-card::before {
        content: "";
        position: absolute;
        width: 180px;
        height: 180px;
        border-radius: 50%;
        background: rgba(255,255,255,.13);
        top: -100px;
        right: -60px;
        animation: shine 5s linear infinite;
    }

    .result-emoji {
        font-size: 3rem;
        animation: pop .65s cubic-bezier(.2,1.4,.4,1) both;
    }

    .result-label {
        font-size: 1.8rem;
        font-weight: 800;
        margin: 7px 0;
    }

    .confidence {
        font-size: .95rem;
        opacity: .9;
    }

    @keyframes resultIn {
        from { opacity: 0; transform: scale(.94) translateY(15px); }
        to { opacity: 1; transform: scale(1) translateY(0); }
    }

    @keyframes pop {
        from { opacity: 0; transform: scale(.4); }
        to { opacity: 1; transform: scale(1); }
    }

    @keyframes shine {
        from { transform: rotate(0deg); }
        to { transform: rotate(360deg); }
    }

    /* ---------- Metric cards ---------- */
    .metric-card {
        background: rgba(255,255,255,.78);
        border: 1px solid rgba(226,232,240,.8);
        border-radius: 18px;
        padding: 20px;
        text-align: center;
        box-shadow: 0 10px 30px rgba(15,23,42,.06);
        animation: metricIn .55s ease-out both;
    }

    .metric-icon {
        font-size: 1.7rem;
    }

    .metric-value {
        font-size: 1.9rem;
        font-weight: 800;
        color: #312e81;
        margin-top: 4px;
    }

    .metric-label {
        color: #64748b;
        font-size: .9rem;
    }

    @keyframes metricIn {
        from { opacity: 0; transform: translateY(12px); }
        to { opacity: 1; transform: translateY(0); }
    }

    /* ---------- Probability cards ---------- */
    .prob-card {
        background: rgba(255,255,255,.80);
        border: 1px solid rgba(226,232,240,.8);
        border-radius: 18px;
        padding: 18px;
        text-align: center;
        box-shadow: 0 10px 25px rgba(15,23,42,.05);
        animation: probIn .5s ease-out both;
    }

    .prob-emoji {
        font-size: 1.8rem;
    }

    .prob-name {
        font-weight: 750;
        margin: 7px 0 10px;
        color: #1e293b;
    }

    .bar-bg {
        width: 100%;
        height: 9px;
        border-radius: 20px;
        overflow: hidden;
        background: #e2e8f0;
    }

    .bar-fill {
        height: 100%;
        border-radius: 20px;
        background: linear-gradient(90deg, #6366f1, #a855f7);
        transform-origin: left;
        animation: growBar 1s cubic-bezier(.2,.8,.2,1) both;
    }

    .prob-value {
        margin-top: 8px;
        font-size: 1.1rem;
        font-weight: 800;
        color: #4f46e5;
    }

    @keyframes probIn {
        from { opacity: 0; transform: translateY(12px); }
        to { opacity: 1; transform: translateY(0); }
    }

    @keyframes growBar {
        from { transform: scaleX(0); }
        to { transform: scaleX(1); }
    }

    /* ---------- Buttons ---------- */
    .stButton > button {
        border: 0;
        border-radius: 14px;
        min-height: 50px;
        font-weight: 750;
        background: linear-gradient(135deg, #4f46e5, #7c3aed);
        color: white;
        box-shadow: 0 10px 25px rgba(79,70,229,.22);
        transition: .2s ease;
    }

    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 15px 32px rgba(79,70,229,.30);
    }

    /* ---------- Text area ---------- */
    textarea {
        border-radius: 16px !important;
    }

    /* ---------- Sidebar ---------- */
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #f8fafc 0%, #eef2ff 100%);
    }

    .sidebar-title {
        font-weight: 800;
        color: #312e81;
        font-size: 1.15rem;
    }

    .stat-row {
        display: flex;
        justify-content: space-between;
        padding: 9px 0;
        border-bottom: 1px solid rgba(148,163,184,.2);
        color: #475569;
    }

    /* ---------- Footer ---------- */
    .footer {
        text-align: center;
        color: #94a3b8;
        padding: 25px 0 10px;
        font-size: .85rem;
    }

    /* ---------- Reduce motion accessibility ---------- */
    @media (prefers-reduced-motion: reduce) {
        *, *::before, *::after {
            animation-duration: .01ms !important;
            animation-iteration-count: 1 !important;
            transition-duration: .01ms !important;
        }
    }
    </style>

    <div class="blob one"></div>
    <div class="blob two"></div>
    <div class="blob three"></div>
    """,
    unsafe_allow_html=True,
)

# ============================================================
# MODEL PATHS
# ============================================================
BASE_DIR = Path(__file__).resolve().parent
VECTORIZER_PATH = BASE_DIR / "count_vec.joblib"
MODEL_PATH = BASE_DIR / "model_lr.joblib"

# ============================================================
# LOAD MODEL
# ============================================================
@st.cache_resource
def load_models():
    if not VECTORIZER_PATH.exists():
        raise FileNotFoundError(
            f"Vectorizer not found: {VECTORIZER_PATH}"
        )

    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Model not found: {MODEL_PATH}"
        )

    vectorizer = joblib.load(VECTORIZER_PATH)
    model = joblib.load(MODEL_PATH)

    return vectorizer, model


try:
    vectorizer, model = load_models()
except Exception as e:
    st.error("❌ Model files could not be loaded.")
    st.code(str(e))
    st.info(
        "Make sure app.py, count_vec.joblib and model_lr.joblib "
        "are in the same project folder."
    )
    st.stop()

# ============================================================
# IMPORTANT:
# Build label mapping from the MODEL itself.
# This avoids the previous class-order bug.
# ============================================================
DISPLAY_INFO = {
    "Negative": {"emoji": "😢"},
    "Positive": {"emoji": "😊"},
    "Neutral": {"emoji": "😐"},
    "Irrelevant": {"emoji": "🤔"},
}

# The model stores the actual class order.
MODEL_CLASSES = list(model.classes_)

# Convert class values to display names.
# Works with string classes and integer encoded classes.
DEFAULT_INTEGER_MAPPING = {
    0: "Irrelevant",
    1: "Negative",
    2: "Neutral",
    3: "Positive",
}

def get_display_name(class_value):
    if isinstance(class_value, str):
        return class_value

    try:
        return DEFAULT_INTEGER_MAPPING[int(class_value)]
    except (ValueError, TypeError, KeyError):
        return str(class_value)



# ============================================================
# HERO
# ============================================================
st.markdown(
    """
    <div class="hero">
        <div class="hero-icon">💭</div>
        <div class="hero-title">Sentiment Analyzer</div>
        <p class="hero-subtitle">
            Analyze text using your trained Machine Learning model
            with real-time probability insights.
        </p>
    </div>
    """,
    unsafe_allow_html=True,
)

# ============================================================
# INPUT
# ============================================================
st.markdown(
    """
    <div class="glass-card">
        <h3 style="margin-top:0;">📝 Enter Your Text</h3>
        <p style="color:#64748b;">
            Write or paste a sentence below.
        </p>
    </div>
    """,
    unsafe_allow_html=True,
)

user_input = st.text_area(
    "Text input",
    placeholder="Example: I absolutely love this product!",
    height=150,
    label_visibility="collapsed",
)

analyze_button = st.button(
    "🔍 Analyze Sentiment",
    use_container_width=True,
    type="primary",
)

# ============================================================
# ANALYSIS
# ============================================================
if analyze_button:
    if not user_input.strip():
        st.warning("⚠️ Please enter some text first.")
        st.stop()

    with st.spinner("✨ Analyzing your text..."):
        time.sleep(0.35)

        try:
            text_vectorized = vectorizer.transform([user_input])

            # Actual prediction
            prediction = model.predict(text_vectorized)[0]

    # Probability
            probabilities = model.predict_proba(text_vectorized)[0]

        except Exception as e:
            st.error("❌ Prediction failed.")
            st.code(str(e))
            st.stop()

    # Actual model class -> display label
    predicted_label = get_display_name(prediction)
    prediction_index = MODEL_CLASSES.index(prediction)
    confidence_score = float(probabilities[prediction_index]) * 100
    emoji = DISPLAY_INFO.get(
        predicted_label,
        {"emoji": "🤖"}
    )["emoji"]

    st.markdown("## 🎯 Analysis Result")

    # --------------------------------------------------------
    # Main result
    # --------------------------------------------------------
    st.markdown(
        f"""
        <div class="result-card">
            <div class="result-emoji">{emoji}</div>
            <div class="result-label">{predicted_label}</div>
            <div class="confidence">
                Confidence: <strong>{confidence_score:.1f}%</strong>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.write("")

    # --------------------------------------------------------
    # Metrics
    # --------------------------------------------------------
    metric1, metric2, metric3 = st.columns(3)

    with metric1:
        st.markdown(
            f"""
            <div class="metric-card">
                <div class="metric-icon">🔤</div>
                <div class="metric-value">{len(user_input):,}</div>
                <div class="metric-label">Characters</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with metric2:
        st.markdown(
            f"""
            <div class="metric-card">
                <div class="metric-icon">📝</div>
                <div class="metric-value">{len(user_input.split()):,}</div>
                <div class="metric-label">Words</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with metric3:
        st.markdown(
            f"""
            <div class="metric-card">
                <div class="metric-icon">🎯</div>
                <div class="metric-value">{confidence_score:.1f}%</div>
                <div class="metric-label">Confidence</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    # --------------------------------------------------------
    # Probability breakdown
    # --------------------------------------------------------
    st.markdown("## 📊 Sentiment Probabilities")

    prob_columns = st.columns(len(probabilities))

    probability_rows = []

    for idx, class_value in enumerate(MODEL_CLASSES):
        label = get_display_name(class_value)
        probability = float(probabilities[idx])
        class_emoji = DISPLAY_INFO.get(
            label,
            {"emoji": "🤖"}
        )["emoji"]

        probability_rows.append(
            {
                "Sentiment": label,
                "Probability": probability,
                "Percentage": f"{probability * 100:.2f}%",
            }
        )

        with prob_columns[idx]:
            st.markdown(
                f"""
                <div class="prob-card">
                    <div class="prob-emoji">{class_emoji}</div>
                    <div class="prob-name">{label}</div>
                    <div class="bar-bg">
                        <div class="bar-fill"
                             style="width:{probability * 100:.2f}%;">
                        </div>
                    </div>
                    <div class="prob-value">
                        {probability * 100:.1f}%
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

    # --------------------------------------------------------
    # Detailed table
    # --------------------------------------------------------
    st.markdown("### 📋 Detailed Breakdown")

    prob_df = pd.DataFrame(probability_rows)
    prob_df = prob_df.sort_values(
        "Probability",
        ascending=False
    ).reset_index(drop=True)

    st.dataframe(
        prob_df,
        use_container_width=True,
        hide_index=True,
    )

    # --------------------------------------------------------
    # Input preview
    # --------------------------------------------------------
    st.markdown("### 📄 Your Input")

    st.markdown(
        f"""
        <div class="glass-card">
            <div style="
                color:#334155;
                font-size:1rem;
                line-height:1.7;
                word-wrap:break-word;
            ">
                {user_input}
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

# ============================================================
# FOOTER
# ============================================================
st.markdown(
    """
    <div class="footer">
        Built with ❤️ using Streamlit & Machine Learning
        <br>
        Sentiment Analysis Application
    </div>
    """,
    unsafe_allow_html=True,
)

