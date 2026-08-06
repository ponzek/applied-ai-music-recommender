"""Tests for the Music Recommender — Applied AI System.

Covers: scoring logic, RAG retrieval, bias detection, confidence scoring,
evaluation metrics, and agent pipeline integration.
"""

from src.recommender import Song, UserProfile, Recommender, recommend_songs, load_songs, STRATEGIES
from src.rag import retrieve, get_mood_similarity_score, get_similar_moods, load_knowledge
from src.bias_detector import (
    detect_genre_bias, detect_popularity_bias,
    detect_language_bias, detect_artist_concentration,
    generate_bias_report,
)
from src.confidence import score_confidence, score_all_confidence, should_refine
from src.evaluation import evaluate_recommendations, score_relevance, score_diversity


# ---------------------------------------------------------------------------
# Original Tests (backward compatibility)
# ---------------------------------------------------------------------------

def make_small_recommender() -> Recommender:
    songs = [
        Song(
            id=1, title="Test Pop Track", artist="Test Artist",
            genre="pop", mood="happy", energy=0.8,
            tempo_bpm=120, valence=0.9, danceability=0.8, acousticness=0.2,
        ),
        Song(
            id=2, title="Chill Lofi Loop", artist="Test Artist",
            genre="lofi", mood="chill", energy=0.4,
            tempo_bpm=80, valence=0.6, danceability=0.5, acousticness=0.9,
        ),
    ]
    return Recommender(songs)


def test_recommend_returns_songs_sorted_by_score():
    user = UserProfile(
        favorite_genre="pop", favorite_mood="happy",
        target_energy=0.8, likes_acoustic=False,
    )
    rec = make_small_recommender()
    results = rec.recommend(user, k=2)

    assert len(results) == 2
    assert results[0].genre == "pop"
    assert results[0].mood == "happy"


def test_explain_recommendation_returns_non_empty_string():
    user = UserProfile(
        favorite_genre="pop", favorite_mood="happy",
        target_energy=0.8, likes_acoustic=False,
    )
    rec = make_small_recommender()
    song = rec.songs[0]

    explanation = rec.explain_recommendation(user, song)
    assert isinstance(explanation, str)
    assert explanation.strip() != ""


def test_acoustic_melancholic_user_prefers_matching_songs():
    songs = [
        Song(
            id=1, title="Cure My Tragedy", artist="Cold",
            genre="alt-rock", mood="melancholic", energy=0.6,
            tempo_bpm=100, valence=0.28, danceability=0.42, acousticness=0.30,
        ),
        Song(
            id=2, title="Gym Hero", artist="Max Pulse",
            genre="pop", mood="intense", energy=0.93,
            tempo_bpm=132, valence=0.77, danceability=0.88, acousticness=0.05,
        ),
    ]
    user = UserProfile(
        favorite_genre="alt-rock", favorite_mood="melancholic",
        target_energy=0.65, likes_acoustic=True,
    )
    rec = Recommender(songs)
    results = rec.recommend(user, k=2)

    assert results[0].title == "Cure My Tragedy"
    assert results[0].genre == "alt-rock"


# ---------------------------------------------------------------------------
# RAG Tests
# ---------------------------------------------------------------------------

def test_rag_retrieves_genre_knowledge():
    """RAG should find relevant knowledge for a known genre."""
    results = retrieve({"genre": "alt-rock", "mood": "melancholic"})
    assert len(results) > 0
    categories = [r["category"] for r in results]
    assert "genre" in categories


def test_rag_retrieves_mood_knowledge():
    """RAG should find relevant knowledge for a known mood."""
    results = retrieve({"genre": "", "mood": "heartbreak"})
    assert len(results) > 0
    categories = [r["category"] for r in results]
    assert "mood" in categories


def test_mood_similarity_exact_match():
    """Exact mood match should return 1.0."""
    assert get_mood_similarity_score("happy", "happy") == 1.0


def test_mood_similarity_related_moods():
    """Similar moods should return partial credit (0.7)."""
    score = get_mood_similarity_score("melancholic", "heartbreak")
    assert score > 0.0, "Melancholic and heartbreak should be similar"
    assert score < 1.0, "They shouldn't be an exact match"


def test_mood_similarity_unrelated_moods():
    """Unrelated moods should return 0.0."""
    score = get_mood_similarity_score("happy", "dark")
    assert score == 0.0


# ---------------------------------------------------------------------------
# Bias Detection Tests
# ---------------------------------------------------------------------------

def _make_biased_recs():
    """Create recommendations that are biased toward one genre."""
    songs = [
        ({"genre": "alt-rock", "artist": "Cold", "mood": "dark", "popularity": 55, "language": "en"}, 5.0, "reason"),
        ({"genre": "alt-rock", "artist": "Cold", "mood": "melancholic", "popularity": 50, "language": "en"}, 4.5, "reason"),
        ({"genre": "alt-rock", "artist": "AFI", "mood": "melancholic", "popularity": 42, "language": "en"}, 4.0, "reason"),
        ({"genre": "alt-rock", "artist": "Cold", "mood": "dark", "popularity": 48, "language": "en"}, 3.5, "reason"),
        ({"genre": "r&b", "artist": "Lauryn Hill", "mood": "heartbreak", "popularity": 88, "language": "en"}, 3.0, "reason"),
    ]
    return songs


def test_detect_genre_bias_finds_concentration():
    """Should detect genre bias when 4/5 songs are alt-rock but user asked for pop."""
    recs = _make_biased_recs()
    # User asked for pop, but got mostly alt-rock — that's genuine bias
    result = detect_genre_bias(recs, user_prefs={"genre": "pop"})
    assert result["bias_detected"] is True
    assert result["score"] >= 0.6


def test_detect_artist_concentration():
    """Should detect when one artist appears multiple times."""
    recs = _make_biased_recs()
    result = detect_artist_concentration(recs)
    assert result["bias_detected"] is True
    assert "Cold" in result["details"]


def test_full_bias_report_structure():
    """Bias report should have all expected keys."""
    recs = _make_biased_recs()
    report = generate_bias_report(recs)
    assert "genre_bias" in report
    assert "popularity_bias" in report
    assert "language_bias" in report
    assert "artist_concentration" in report
    assert "summary" in report
    assert "biases_detected" in report["summary"]


# ---------------------------------------------------------------------------
# Confidence Tests
# ---------------------------------------------------------------------------

def test_high_match_gets_high_confidence():
    """A song matching all user preferences should have high confidence."""
    prefs = {"genre": "pop", "mood": "happy", "energy": 0.8, "likes_acoustic": False}
    song = {"genre": "pop", "mood": "happy", "energy": 0.82, "acousticness": 0.2}
    result = score_confidence(prefs, song, 7.5, "mood match; genre match; energy; acoustic")
    assert result["confidence"] >= 0.6
    assert result["level"] in ("high", "medium")


def test_low_match_gets_low_confidence():
    """A song matching nothing should have low confidence."""
    prefs = {"genre": "pop", "mood": "happy", "energy": 0.8, "likes_acoustic": False}
    song = {"genre": "jazz", "mood": "relaxed", "energy": 0.3, "acousticness": 0.9}
    result = score_confidence(prefs, song, 1.0, "energy closeness")
    assert result["confidence"] < 0.5
    assert result["level"] in ("low", "medium")


def test_should_refine_with_low_confidence():
    """Should trigger refinement when confidence is low."""
    low_scores = [{"confidence": 0.2}, {"confidence": 0.3}, {"confidence": 0.1}]
    assert should_refine(low_scores, threshold=0.4) is True


def test_should_not_refine_with_high_confidence():
    """Should not trigger refinement when confidence is high."""
    high_scores = [{"confidence": 0.8}, {"confidence": 0.9}, {"confidence": 0.7}]
    assert should_refine(high_scores, threshold=0.4) is False


# ---------------------------------------------------------------------------
# Evaluation Tests
# ---------------------------------------------------------------------------

def _make_diverse_recs():
    """Create a set of diverse recommendations."""
    songs = [
        ({"genre": "pop", "artist": "A", "mood": "happy", "energy": 0.8, "popularity": 70, "release_decade": "2020s"}, 5.0, "r"),
        ({"genre": "lofi", "artist": "B", "mood": "chill", "energy": 0.4, "popularity": 45, "release_decade": "2020s"}, 4.0, "r"),
        ({"genre": "rock", "artist": "C", "mood": "intense", "energy": 0.9, "popularity": 60, "release_decade": "2010s"}, 3.5, "r"),
    ]
    return songs


def test_diversity_score_with_diverse_recs():
    """Diverse recommendations should score high on diversity."""
    recs = _make_diverse_recs()
    result = score_diversity(recs)
    assert result["score"] >= 0.8
    assert result["unique_genres"] == 3


def test_relevance_with_matching_recs():
    """Recommendations matching user prefs should score high on relevance."""
    prefs = {"genre": "pop", "mood": "happy", "energy": 0.8}
    recs = [({"genre": "pop", "mood": "happy", "energy": 0.82}, 5.0, "r")]
    result = score_relevance(prefs, recs)
    assert result["score"] >= 0.8
