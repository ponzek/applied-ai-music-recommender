"""Simple evaluation helpers for the music recommender."""


def evaluate_recommendations(user_prefs, recommendations, full_catalog):
    relevance = score_relevance(user_prefs, recommendations)
    diversity = score_diversity(recommendations)
    coverage = score_coverage(recommendations, full_catalog)
    novelty = score_novelty(recommendations)

    # Coverage is useful to display, but it should not lower the grade for
    # one personalized list. A five-song list cannot cover a large catalog.
    overall_score = (
        relevance["score"] * 0.70
        + diversity["score"] * 0.20
        + novelty["score"] * 0.10
    )

    return {
        "relevance": relevance,
        "diversity": diversity,
        "coverage": coverage,
        "novelty": novelty,
        "overall": {
            "score": round(overall_score, 2),
            "grade": _score_to_grade(overall_score),
        },
    }


def score_relevance(user_prefs, recommendations):
    if not recommendations:
        return {"score": 0.0, "details": "No recommendations"}

    total = len(recommendations)
    wanted_genre = user_prefs.get("genre")
    wanted_mood = user_prefs.get("mood")
    target_energy = user_prefs.get("energy", 0.5)

    genre_matches = sum(
        song.get("genre") == wanted_genre
        for song, _, _ in recommendations
    )
    mood_matches = sum(
        song.get("mood") == wanted_mood
        for song, _, _ in recommendations
    )

    energy_differences = [
        abs(song.get("energy", 0.5) - target_energy)
        for song, _, _ in recommendations
    ]
    energy_score = max(0.0, 1.0 - sum(energy_differences) / total)

    genre_rate = genre_matches / total
    mood_rate = mood_matches / total
    score = genre_rate * 0.30 + mood_rate * 0.40 + energy_score * 0.30

    return {
        "score": round(score, 2),
        "genre_match_rate": f"{genre_matches}/{total}",
        "mood_match_rate": f"{mood_matches}/{total}",
        "avg_energy_closeness": round(energy_score, 2),
        "details": (
            f"Genre: {genre_rate:.0%}, Mood: {mood_rate:.0%}, "
            f"Energy closeness: {energy_score:.2f}"
        ),
    }


def score_diversity(recommendations):
    """Check variety without punishing a user for choosing a genre or mood."""
    if not recommendations:
        return {"score": 0.0, "details": "No recommendations"}

    songs = [song for song, _, _ in recommendations]
    total = len(songs)

    unique_artists = len({song.get("artist", "") for song in songs})
    artist_score = unique_artists / total

    decades = {song.get("release_decade", "") for song in songs}
    decades.discard("")
    decade_target = min(total, 3)
    decade_score = min(len(decades) / decade_target, 1.0) if decade_target else 0.0

    popularity_groups = set()
    for song in songs:
        popularity = song.get("popularity", 50)
        if popularity < 40:
            popularity_groups.add("low")
        elif popularity < 70:
            popularity_groups.add("medium")
        else:
            popularity_groups.add("high")

    popularity_score = len(popularity_groups) / 3
    score = artist_score * 0.60 + decade_score * 0.20 + popularity_score * 0.20

    return {
        "score": round(score, 2),
        "unique_genres": len({song.get("genre", "") for song in songs}),
        "unique_artists": unique_artists,
        "unique_moods": len({song.get("mood", "") for song in songs}),
        "unique_decades": len(decades),
        "details": (
            f"{unique_artists} artists, {len(decades)} decades, "
            f"{len(popularity_groups)} popularity ranges"
        ),
    }


def score_coverage(recommendations, full_catalog):
    """Show catalog reach. This metric is informational only."""
    if not recommendations or not full_catalog:
        return {"score": 0.0, "details": "No data"}

    catalog_genres = {song.get("genre", "") for song in full_catalog}
    catalog_moods = {song.get("mood", "") for song in full_catalog}
    rec_genres = {song.get("genre", "") for song, _, _ in recommendations}
    rec_moods = {song.get("mood", "") for song, _, _ in recommendations}

    genre_coverage = len(rec_genres) / len(catalog_genres) if catalog_genres else 0
    mood_coverage = len(rec_moods) / len(catalog_moods) if catalog_moods else 0
    score = (genre_coverage + mood_coverage) / 2

    return {
        "score": round(score, 2),
        "genre_coverage": f"{len(rec_genres)}/{len(catalog_genres)}",
        "mood_coverage": f"{len(rec_moods)}/{len(catalog_moods)}",
        "details": (
            f"Genres: {genre_coverage:.0%}, Moods: {mood_coverage:.0%} "
            "(not included in grade)"
        ),
    }


def score_novelty(recommendations):
    if not recommendations:
        return {"score": 0.0, "details": "No recommendations"}

    popularity = [
        song.get("popularity", 50)
        for song, _, _ in recommendations
    ]
    average = sum(popularity) / len(popularity)
    score = max(0.0, 1.0 - average / 100)

    return {
        "score": round(score, 2),
        "avg_popularity": round(average, 1),
        "details": f"Average popularity: {average:.0f}/100",
    }


def _score_to_grade(score):
    if score >= 0.90:
        return "A+"
    if score >= 0.80:
        return "A"
    if score >= 0.70:
        return "B"
    if score >= 0.60:
        return "C"
    if score >= 0.50:
        return "D"
    return "F"


def format_evaluation(results):
    lines = ["=" * 55, "EVALUATION", "=" * 55]

    for name in ("relevance", "diversity", "coverage", "novelty"):
        item = results[name]
        lines.append(f"{name.title():<12} {item['score']:.2f}  {item['details']}")

    overall = results["overall"]
    lines.append("-" * 55)
    lines.append(f"Overall: {overall['score']:.2f} ({overall['grade']})")
    return "\n".join(lines)


def format_evaluation_table(results):
    lines = [
        "| Metric | Score | Details |",
        "|---|---:|---|",
    ]

    for name in ("relevance", "diversity", "coverage", "novelty"):
        item = results[name]
        lines.append(f"| {name.title()} | {item['score']:.2f} | {item['details']} |")

    overall = results["overall"]
    lines.append(
        f"| **Overall** | **{overall['score']:.2f} ({overall['grade']})** | "
        "Coverage is informational |"
    )
    return "\n".join(lines)
