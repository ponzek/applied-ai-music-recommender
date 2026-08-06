"""Streamlit Web UI -- Music Recommender with dreamy pastel aesthetic.

Run with: streamlit run src/app.py
"""

import streamlit as st
from src.recommender import load_songs, recommend_songs, STRATEGIES
from src.rag import explain_with_context
from src.bias_detector import generate_bias_report
from src.evaluation import evaluate_recommendations
from src.confidence import score_all_confidence

# Page config
st.set_page_config(page_title="Music Recommender AI", page_icon="🎵", layout="wide")

# Custom CSS
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;500;600;700&display=swap');

    /* Base */
    .stApp {
        background: linear-gradient(160deg, #faf5f8 0%, #f0e8ee 30%, #e8ede8 60%, #f5eff5 100%);
    }

    /* Apply the page font to normal text and controls only.
       Do not apply it to every span because Streamlit icons use spans. */
    .stApp,
    .stApp p,
    .stApp li,
    .stApp h1,
    .stApp h2,
    .stApp h3,
    .stApp label,
    .stApp button,
    .stApp input,
    .stApp textarea,
    .stApp select {
        font-family: 'Outfit', sans-serif !important;
    }

    /* Normal body text */
    .stMarkdown p,
    .stMarkdown li,
    .stText,
    div[data-testid="stText"],
    .stCaption p {
        color: #2b3d2b !important;
        font-size: 18px !important;
        line-height: 1.6 !important;
    }

    /* Keep Streamlit's built-in icons as icons instead of showing
       words such as double_arrow_right or keyboard_arrow_right. */
    span[data-testid="stIconMaterial"],
    span[data-testid="stExpanderToggleIcon"],
    .material-symbols-rounded,
    .material-symbols-outlined {
        font-family: "Material Symbols Rounded", "Material Symbols Outlined" !important;
        font-size: 1.25rem !important;
        line-height: 1 !important;
        letter-spacing: normal !important;
        text-transform: none !important;
        white-space: nowrap !important;
        word-wrap: normal !important;
        direction: ltr !important;
        font-feature-settings: "liga" !important;
        -webkit-font-feature-settings: "liga" !important;
        -webkit-font-smoothing: antialiased !important;
    }

    /* Sidebar open/close buttons */
    [data-testid="collapsedControl"] span[data-testid="stIconMaterial"],
    [data-testid="stSidebarCollapseButton"] span[data-testid="stIconMaterial"],
    button[data-testid="stBaseButton-headerNoPadding"] span[data-testid="stIconMaterial"] {
        color: #6b4a63 !important;
        font-size: 1.4rem !important;
        visibility: visible !important;
        width: auto !important;
        overflow: visible !important;
    }

    /* Headers */
    h1 {
        color: #6b4a63 !important;
        font-weight: 700 !important;
        font-size: 2rem !important;
        letter-spacing: -0.5px;
    }
    h2, h3 {
        color: #5a4a5a !important;
        font-weight: 600 !important;
        font-size: 1.2rem !important;
    }

    /* Sidebar */
    section[data-testid="stSidebar"] {
        background: linear-gradient(180deg, #f5eef3 0%, #eae5e0 100%) !important;
        border-right: 2px solid #d9c4d4;
    }
    section[data-testid="stSidebar"] h2 {
        color: #6b4a63 !important;
        font-size: 1.1rem !important;
    }

    /* Metric cards */
    div[data-testid="stMetric"] {
        background: rgba(255, 255, 255, 0.7);
        border: 1px solid #e0d0dc;
        border-radius: 14px;
        padding: 14px 18px;
        box-shadow: 0 2px 12px rgba(180, 150, 170, 0.12);
    }
    div[data-testid="stMetric"] label {
        color: #8b6a83 !important;
        font-size: 12px !important;
        font-weight: 500 !important;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }
    div[data-testid="stMetric"] div[data-testid="stMetricValue"] {
        color: #4a3545 !important;
        font-size: 1.6rem !important;
        font-weight: 700 !important;
    }

    /* Primary button */
    .stButton > button[kind="primary"],
    .stButton > button {
        background: linear-gradient(135deg, #c9a0bc, #b8919e) !important;
        color: #fff !important;
        border: none !important;
        border-radius: 24px !important;
        padding: 10px 28px !important;
        font-weight: 600 !important;
        font-size: 14px !important;
        letter-spacing: 0.3px;
        box-shadow: 0 3px 14px rgba(200, 160, 188, 0.35) !important;
        transition: all 0.25s ease !important;
    }
    .stButton > button:hover {
        transform: translateY(-1px) !important;
        box-shadow: 0 5px 20px rgba(200, 160, 188, 0.5) !important;
    }

    /* Song card rows */
    div[data-testid="stHorizontalBlock"] {
        border-radius: 12px;
    }

    /* Expanders */
    details {
        background: rgba(255, 255, 255, 0.5) !important;
        border: 1px solid #e0d0dc !important;
        border-radius: 12px !important;
    }
    details summary {
        color: #6b4a63 !important;
        font-weight: 500 !important;
    }

    details summary span:not([data-testid="stIconMaterial"]):not([data-testid="stExpanderToggleIcon"]) {
        color: #6b4a63 !important;
        font-family: 'Outfit', sans-serif !important;
        font-weight: 500 !important;
    }

    /* Alert boxes */
    div[data-testid="stAlert"] {
        border-radius: 12px;
        font-size: 14px !important;
    }

    /* Selectbox / slider labels */
    .stSelectbox label, .stSlider label, .stCheckbox label {
        color: #4a3a4a !important;
        font-weight: 500 !important;
        font-size: 13px !important;
    }

    /* Dividers */
    hr {
        border-color: #e0d0dc !important;
        opacity: 0.5;
    }
</style>
""", unsafe_allow_html=True)


# Header
st.markdown("# 🎵 Music Recommender ✨")
st.caption("🌸 RAG + Agentic AI + Bias Detection + Confidence Scoring 🌸")

# Load songs
@st.cache_data
def get_songs():
    return load_songs("data/songs.csv")

songs = get_songs()
st.sidebar.success(f"🎶 {len(songs)} songs loaded")

# Sidebar
st.sidebar.markdown("## 🌷 Your Profile")

genres = sorted(set(s["genre"] for s in songs))
moods = sorted(set(s["mood"] for s in songs))

genre = st.sidebar.selectbox("Favorite Genre", genres, index=genres.index("alt-rock") if "alt-rock" in genres else 0)
mood = st.sidebar.selectbox("Favorite Mood", moods, index=moods.index("melancholic") if "melancholic" in moods else 0)
energy = st.sidebar.slider("Target Energy", 0.0, 1.0, 0.65, 0.05)
acoustic = st.sidebar.checkbox("Prefer Acoustic", value=True)

strategy_key = st.sidebar.selectbox("Scoring Strategy", list(STRATEGIES.keys()), index=0)
k = st.sidebar.slider("Number of Recommendations", 3, 15, 10)

user_prefs = {
    "genre": genre,
    "mood": mood,
    "energy": energy,
    "likes_acoustic": acoustic,
}

# Main
if st.sidebar.button("✨ Get Recommendations", type="primary"):
    strategy = STRATEGIES[strategy_key]

    with st.spinner("🎵 Finding your perfect songs..."):
        recommendations = recommend_songs(user_prefs, songs, k=k, strategy=strategy, diversity=True)

    if not recommendations:
        st.warning("No recommendations found. Try different preferences.")
    else:
        st.markdown(f"### 🎧 Top {k} Picks — *{strategy_key}*")
        for i, (song, score, explanation) in enumerate(recommendations, 1):
            with st.container():
                col1, col2, col3 = st.columns([3, 1, 1])
                col1.markdown(f"**{i}. {song['title']}** by {song['artist']}")
                col2.metric("Score", f"{score:.2f}")
                col3.metric("Energy", f"{song['energy']:.2f}")
                st.caption(f"🎵 {song['genre']}  ·  {song['mood']}  ·  {explanation}")
                st.divider()

        # Evaluation + Bias
        eval_col, bias_col = st.columns(2)

        with eval_col:
            st.markdown("### 📊 Evaluation")
            evaluation = evaluate_recommendations(user_prefs, recommendations, songs)
            cols = st.columns(2)
            for i, name in enumerate(["relevance", "diversity", "coverage", "novelty"]):
                result = evaluation[name]
                cols[i % 2].metric(name.title(), f"{result['score']:.2f}")
            overall = evaluation["overall"]
            st.info(f"Grade: **{overall['grade']}** ({overall['score']:.2f})")

        with bias_col:
            st.markdown("### 🔍 Bias Check")
            report = generate_bias_report(recommendations, songs, user_prefs)
            for name, label in {"genre_bias": "Genre", "popularity_bias": "Popularity", "language_bias": "Language", "artist_concentration": "Artist repetition"}.items():
                result = report[name]
                icon = "⚠️" if result["bias_detected"] else "✅"
                st.markdown(f"{icon} **{label}:** {result['details']}")
            summary = report["summary"]
            if summary["biases_detected"]:
                st.warning(summary["verdict"])
            else:
                st.success(summary["verdict"])

        # Expandable
        with st.expander("💫 Confidence Scores"):
            conf_scores = score_all_confidence(user_prefs, recommendations)
            for (song, score, _), conf in zip(recommendations, conf_scores):
                level_icon = {"high": "🌟", "medium": "✨", "low": "💫"}[conf["level"]]
                st.markdown(f"{level_icon} **{song['title'][:35]}**: {conf['confidence']:.2f} ({conf['level']})")
            avg_conf = sum(c["confidence"] for c in conf_scores) / len(conf_scores)
            st.metric("Average Confidence", f"{avg_conf:.2f}")

        with st.expander("🌸 RAG Explanation — Top Pick"):
            with st.spinner("Asking LLM..."):
                try:
                    top_song = recommendations[0][0]
                    explanation = explain_with_context(user_prefs, top_song)
                    st.markdown(explanation)
                except Exception as e:
                    st.error(f"LLM unavailable: {e}")

else:
    st.markdown("---")
    st.markdown("*Set your preferences in the sidebar and click* **✨ Get Recommendations** *to begin* 🎵")

    st.markdown("### 🌷 Catalog Overview")
    from collections import Counter
    genre_counts = Counter(s["genre"] for s in songs)
    mood_counts = Counter(s["mood"] for s in songs)

    col1, col2 = st.columns(2)
    with col1:
        st.markdown("**🎵 Genres**")
        for g, c in genre_counts.most_common():
            st.markdown(f"&nbsp;&nbsp;{g}: {c} songs")
    with col2:
        st.markdown("**💜 Moods**")
        for m, c in mood_counts.most_common():
            st.markdown(f"&nbsp;&nbsp;{m}: {c} songs")
