"""Integration tests — run the real pipeline end-to-end with actual CSV data.

These catch issues that unit tests miss: shape mismatches between components,
key errors when real data flows through the pipeline, and regressions when
scoring weights change.
"""

import pytest

from src.recommender import recommend_songs, STRATEGIES
from src.evaluation import evaluate_recommendations
from src.bias_detector import generate_bias_report
from src.confidence import score_all_confidence, should_refine, self_critique


# -- CSV → Recommend pipeline ------------------------------------------------

def test_loaded_songs_have_required_keys(full_catalog):
    # The scorer assumes these keys exist — if the CSV schema drifts
    # and a column gets renamed, this catches it immediately.
    required = {"id", "title", "artist", "genre", "mood",
                "energy", "tempo_bpm", "valence", "danceability", "acousticness"}
    for song in full_catalog[:20]:
        missing = required - song.keys()
        assert not missing, f"Song {song.get('id')} missing: {missing}"


def test_recommend_returns_correct_count(small_catalog, sample_user_prefs):
    results = recommend_songs(sample_user_prefs, small_catalog, k=5)
    assert len(results) == 5
    # Each result should be (song_dict, score, explanation)
    for song, score, explanation in results:
        assert isinstance(song, dict)
        assert isinstance(score, (int, float))
        assert isinstance(explanation, str)


def test_results_come_back_sorted(small_catalog, sample_user_prefs):
    results = recommend_songs(sample_user_prefs, small_catalog, k=5)
    scores = [score for _, score, _ in results]
    assert scores == sorted(scores, reverse=True)


# -- Recommend → Evaluate pipeline -------------------------------------------

def test_evaluation_structure_and_ranges(small_catalog, sample_user_prefs):
    # This is the key integration test the feedback asked for:
    # run recommend_songs, then feed that directly into evaluate_recommendations
    recs = recommend_songs(sample_user_prefs, small_catalog, k=5)
    result = evaluate_recommendations(sample_user_prefs, recs, small_catalog)

    # Check all expected sections exist
    for key in ("relevance", "diversity", "coverage", "novelty", "overall"):
        assert key in result

    # All metric scores should be 0-1
    for metric in ("relevance", "diversity", "coverage", "novelty"):
        assert 0.0 <= result[metric]["score"] <= 1.0

    assert result["overall"]["grade"] in {"A+", "A", "B", "C", "D", "F"}


# -- Recommend → Bias Report pipeline ----------------------------------------

def test_bias_report_from_real_recommendations(small_catalog, sample_user_prefs):
    # The feedback specifically called out: "a test that runs the recommender
    # on a known profile and then checks the bias report would catch regressions
    # in the scoring-to-bias pipeline that unit tests miss"
    recs = recommend_songs(sample_user_prefs, small_catalog, k=5)
    report = generate_bias_report(recs, full_catalog=small_catalog,
                                  user_prefs=sample_user_prefs)

    for section in ("genre_bias", "popularity_bias", "language_bias", "artist_concentration"):
        assert isinstance(report[section]["bias_detected"], bool)
        assert isinstance(report[section]["score"], (int, float))

    # The count in the summary should match the actual flags
    sections = ("genre_bias", "popularity_bias", "language_bias", "artist_concentration")
    flagged = sum(1 for s in sections if report[s]["bias_detected"])
    assert report["summary"]["biases_detected"] == flagged


def test_bias_report_with_mismatched_profile(small_catalog, mismatched_user_prefs):
    # Edge case: user wants a rare genre — make sure nothing crashes
    recs = recommend_songs(mismatched_user_prefs, small_catalog, k=5)
    report = generate_bias_report(recs, full_catalog=small_catalog,
                                  user_prefs=mismatched_user_prefs)
    assert "summary" in report


# -- Recommend → Confidence pipeline ------------------------------------------

def test_confidence_on_real_recommendations(small_catalog, sample_user_prefs):
    recs = recommend_songs(sample_user_prefs, small_catalog, k=5)
    scores = score_all_confidence(sample_user_prefs, recs)

    assert len(scores) == len(recs)
    for cs in scores:
        assert 0.0 <= cs["confidence"] <= 1.0
        assert cs["level"] in ("high", "medium", "low")

    # should_refine just needs to return a bool without crashing
    assert isinstance(should_refine(scores), bool)


def test_self_critique_fallback(small_catalog, sample_user_prefs):
    # Without Ollama running, self_critique should fall back to the
    # rule-based critique — not crash
    recs = recommend_songs(sample_user_prefs, small_catalog, k=5)
    scores = score_all_confidence(sample_user_prefs, recs)
    critique = self_critique(sample_user_prefs, recs, scores)

    assert isinstance(critique["issues"], list)
    assert isinstance(critique["suggestions"], list)
    assert "overall_assessment" in critique


# -- Full end-to-end ----------------------------------------------------------

def test_full_pipeline(small_catalog, sample_user_prefs):
    # Walk through every stage of the pipeline in order,
    # same as the agent does at runtime
    recs = recommend_songs(sample_user_prefs, small_catalog, k=5)
    assert len(recs) > 0

    evaluation = evaluate_recommendations(sample_user_prefs, recs, small_catalog)
    assert 0.0 <= evaluation["overall"]["score"] <= 1.0

    bias = generate_bias_report(recs, full_catalog=small_catalog,
                                user_prefs=sample_user_prefs)
    assert isinstance(bias["summary"]["biases_detected"], int)

    conf = score_all_confidence(sample_user_prefs, recs)
    assert len(conf) == len(recs)

    critique = self_critique(sample_user_prefs, recs, conf)
    assert "overall_assessment" in critique


# -- All strategies on real data ----------------------------------------------

@pytest.mark.parametrize("strategy_name", list(STRATEGIES.keys()))
def test_each_strategy_produces_valid_output(small_catalog, sample_user_prefs, strategy_name):
    # Make sure every strategy works on real data and feeds into evaluation
    strategy = STRATEGIES[strategy_name]
    recs = recommend_songs(sample_user_prefs, small_catalog, k=5, strategy=strategy)

    assert len(recs) == 5
    for song, score, explanation in recs:
        assert isinstance(song, dict)
        assert len(explanation) > 0

    # Each strategy's output should also evaluate cleanly
    result = evaluate_recommendations(sample_user_prefs, recs, small_catalog)
    assert 0.0 <= result["overall"]["score"] <= 1.0
