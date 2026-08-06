"""Shared test fixtures — loads real data once so integration tests stay fast."""

import pytest
from collections import Counter
from src.recommender import load_songs


@pytest.fixture(scope="session")
def full_catalog():
    """Load the full song catalog once for all tests."""
    songs = load_songs("data/songs.csv")
    assert len(songs) > 0
    return songs


@pytest.fixture(scope="session")
def small_catalog(full_catalog):
    return full_catalog[:100]


@pytest.fixture(scope="session")
def catalog_genres(full_catalog):
    return sorted({song["genre"] for song in full_catalog})


@pytest.fixture(scope="session")
def catalog_moods(full_catalog):
    return sorted({song["mood"] for song in full_catalog})


@pytest.fixture(scope="session")
def sample_user_prefs(full_catalog):
    # Instead of hardcoding "pop" or "happy", we pull the most common
    # genre and mood from the actual dataset so the tests adapt
    # automatically if the CSV changes.
    genre_counts = Counter(song["genre"] for song in full_catalog)
    mood_counts = Counter(song["mood"] for song in full_catalog)

    most_common_genre = genre_counts.most_common(1)[0][0]
    most_common_mood = mood_counts.most_common(1)[0][0]

    # Use median energy as a realistic target
    energies = sorted(song["energy"] for song in full_catalog)
    median_energy = energies[len(energies) // 2]

    return {
        "genre": most_common_genre,
        "mood": most_common_mood,
        "energy": median_energy,
        "likes_acoustic": False,
    }


@pytest.fixture(scope="session")
def mismatched_user_prefs(catalog_genres, catalog_moods):
    # Pick the least common genre/mood to test the "nothing matches well" path
    return {
        "genre": catalog_genres[-1],
        "mood": catalog_moods[-1],
        "energy": 0.2,
        "likes_acoustic": True,
    }
