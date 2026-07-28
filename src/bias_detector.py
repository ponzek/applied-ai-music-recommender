"""Basic checks for patterns that may make recommendations less balanced."""

from collections import Counter


def _songs_from(recommendations):
    return [song for song, _, _ in recommendations]


def _comparison_pool(full_catalog, user_prefs):
    if not full_catalog:
        return []

    wanted_genre = (user_prefs or {}).get("genre")
    if not wanted_genre:
        return full_catalog

    matching = [
        song for song in full_catalog
        if song.get("genre") == wanted_genre
    ]
    return matching or full_catalog


def detect_genre_bias(recommendations, user_prefs=None):
    songs = _songs_from(recommendations)
    if not songs:
        return {"bias_detected": False, "score": 0.0, "details": "No recommendations"}

    counts = Counter(song.get("genre", "unknown") for song in songs)
    top_genre, top_count = counts.most_common(1)[0]
    ratio = top_count / len(songs)
    wanted_genre = (user_prefs or {}).get("genre")

    # A personalized list is expected to contain the genre the user selected.
    if wanted_genre and top_genre == wanted_genre:
        return {
            "bias_detected": False,
            "score": 0.0,
            "details": f"{top_count}/{len(songs)} songs match the requested genre ({wanted_genre})",
            "distribution": dict(counts),
        }

    detected = ratio > 0.60
    return {
        "bias_detected": detected,
        "score": round(ratio if detected else 0.0, 2),
        "details": f"{top_count}/{len(songs)} songs are {top_genre} ({ratio:.0%})",
        "distribution": dict(counts),
    }


def detect_popularity_bias(recommendations, full_catalog=None, user_prefs=None):
    songs = _songs_from(recommendations)
    if not songs:
        return {"bias_detected": False, "score": 0.0, "details": "No recommendations"}

    rec_average = sum(song.get("popularity", 0) for song in songs) / len(songs)
    pool = _comparison_pool(full_catalog, user_prefs)

    if pool:
        pool_average = sum(song.get("popularity", 0) for song in pool) / len(pool)
        detected = rec_average > 75 and rec_average > pool_average + 15
        details = (
            f"Recommended average: {rec_average:.0f}/100; "
            f"similar catalog average: {pool_average:.0f}/100"
        )
        difference = max(0.0, rec_average - pool_average)
        score = min(difference / 50, 1.0) if detected else 0.0
    else:
        detected = rec_average > 80
        details = f"Average popularity: {rec_average:.0f}/100"
        score = max(0.0, (rec_average - 80) / 20) if detected else 0.0

    return {
        "bias_detected": detected,
        "score": round(score, 2),
        "details": details,
        "values": [song.get("popularity", 0) for song in songs],
    }


def detect_language_bias(recommendations, full_catalog=None, user_prefs=None):
    songs = _songs_from(recommendations)
    if not songs:
        return {"bias_detected": False, "score": 0.0, "details": "No recommendations"}

    rec_counts = Counter(song.get("language", "en") for song in songs)
    rec_non_english = sum(count for lang, count in rec_counts.items() if lang != "en")
    rec_ratio = rec_non_english / len(songs)

    pool = _comparison_pool(full_catalog, user_prefs)
    if not pool:
        detected = rec_ratio == 0 and len(songs) >= 5
        return {
            "bias_detected": detected,
            "score": 1.0 if detected else 0.0,
            "details": f"{rec_non_english}/{len(songs)} songs are non-English",
            "distribution": dict(rec_counts),
        }

    pool_counts = Counter(song.get("language", "en") for song in pool)
    pool_non_english = sum(count for lang, count in pool_counts.items() if lang != "en")
    pool_ratio = pool_non_english / len(pool)

    if pool_non_english == 0:
        return {
            "bias_detected": False,
            "score": 0.0,
            "details": "No non-English songs are available in the matching catalog group",
            "distribution": dict(rec_counts),
        }

    gap = pool_ratio - rec_ratio
    detected = pool_ratio >= 0.20 and gap >= 0.20

    return {
        "bias_detected": detected,
        "score": round(max(gap, 0.0), 2) if detected else 0.0,
        "details": (
            f"Non-English results: {rec_ratio:.0%}; "
            f"available in matching catalog: {pool_ratio:.0%}"
        ),
        "distribution": dict(rec_counts),
    }


def detect_artist_concentration(recommendations):
    songs = _songs_from(recommendations)
    if not songs:
        return {"bias_detected": False, "score": 0.0, "details": "No recommendations"}

    counts = Counter(song.get("artist", "unknown") for song in songs)
    artist, count = counts.most_common(1)[0]
    ratio = count / len(songs)
    detected = count >= 3 or ratio > 0.50

    return {
        "bias_detected": detected,
        "score": round(ratio if detected else 0.0, 2),
        "details": f"{artist} appears {count}/{len(songs)} times ({ratio:.0%})",
        "distribution": dict(counts),
    }


def generate_bias_report(recommendations, full_catalog=None, user_prefs=None):
    report = {
        "genre_bias": detect_genre_bias(recommendations, user_prefs),
        "popularity_bias": detect_popularity_bias(
            recommendations, full_catalog, user_prefs
        ),
        "language_bias": detect_language_bias(
            recommendations, full_catalog, user_prefs
        ),
        "artist_concentration": detect_artist_concentration(recommendations),
    }

    found = [
        name for name, result in report.items()
        if result["bias_detected"]
    ]

    report["summary"] = {
        "biases_detected": len(found),
        "bias_types": found,
        "overall_score": round(
            sum(item["score"] for item in report.values()) / 4,
            2,
        ),
        "verdict": _get_verdict(len(found)),
    }
    return report


def _get_verdict(number_found):
    if number_found == 0:
        return "No obvious bias patterns were found in this list."
    if number_found == 1:
        return "One possible bias pattern was found. Review the detail above."
    if number_found == 2:
        return "A couple of bias patterns were found and may need adjustment."
    return "Several bias patterns were found. The ranking rules should be reviewed."


def format_bias_report(report):
    names = (
        "genre_bias",
        "popularity_bias",
        "language_bias",
        "artist_concentration",
    )
    lines = ["=" * 55, "BIAS CHECK", "=" * 55]

    for name in names:
        item = report[name]
        status = "CHECK" if item["bias_detected"] else "OK"
        label = name.replace("_", " ").title()
        lines.append(f"{label:<22} [{status}] {item['details']}")

    lines.append("-" * 55)
    lines.append(report["summary"]["verdict"])
    return "\n".join(lines)


def get_llm_bias_summary(report, user_prefs):
    prompt = f"""Review this music recommendation bias report in two short sentences.
Give one practical improvement only when a bias was actually detected.

Genre: {report['genre_bias']['details']}
Popularity: {report['popularity_bias']['details']}
Language: {report['language_bias']['details']}
Artist repetition: {report['artist_concentration']['details']}
Requested genre: {user_prefs.get('genre')}
"""

    try:
        from llm_client import chat, TEXT_MODEL
        return chat(prompt, model=TEXT_MODEL, temperature=0.4, max_tokens=120)
    except Exception:
        return report["summary"]["verdict"]
