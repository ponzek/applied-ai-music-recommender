"""Confidence Scoring + Self-Critique — Rates and improves recommendations.

Each recommendation gets a confidence score (0-1). Low-confidence results
trigger the LLM to critique and suggest improvements.
"""

from typing import List, Dict, Tuple


# ---------------------------------------------------------------------------
# Confidence Scoring
# ---------------------------------------------------------------------------

def score_confidence(
    user_prefs: Dict,
    song: Dict,
    score: float,
    explanation: str,
) -> Dict:
    """Rate how confident the system is in a single recommendation.

    Factors:
    - Score strength (higher raw score = more confident)
    - Feature match count (more matching features = more confident)
    - Explanation richness (more reasons = more confident)

    Returns:
        Dict with confidence (0-1), level (high/medium/low), and reasoning.
    """
    reasons = []

    # Factor 1: Raw score strength (normalized to 0-1, max realistic score ~8)
    score_factor = min(score / 8.0, 1.0)
    reasons.append(f"score strength: {score_factor:.2f}")

    # Factor 2: Feature match count
    match_count = 0
    if song.get("genre") == user_prefs.get("genre"):
        match_count += 1
    if song.get("mood") == user_prefs.get("mood"):
        match_count += 1

    target_energy = user_prefs.get("energy", 0.5)
    if abs(song.get("energy", 0.5) - target_energy) < 0.15:
        match_count += 1

    likes_acoustic = user_prefs.get("likes_acoustic", False)
    acousticness = song.get("acousticness", 0.5)
    if (likes_acoustic and acousticness >= 0.6) or (not likes_acoustic and acousticness < 0.4):
        match_count += 1

    match_factor = match_count / 4.0
    reasons.append(f"feature matches: {match_count}/4")

    # Factor 3: Explanation richness
    reason_count = explanation.count(";") + 1 if explanation else 0
    explanation_factor = min(reason_count / 4.0, 1.0)
    reasons.append(f"explanation depth: {reason_count} reasons")

    # Weighted confidence
    confidence = (score_factor * 0.4) + (match_factor * 0.4) + (explanation_factor * 0.2)

    # Determine level
    if confidence >= 0.7:
        level = "high"
    elif confidence >= 0.4:
        level = "medium"
    else:
        level = "low"

    return {
        "confidence": round(confidence, 2),
        "level": level,
        "reasoning": "; ".join(reasons),
        "match_count": match_count,
    }


def score_all_confidence(
    user_prefs: Dict,
    recommendations: List[Tuple[Dict, float, str]],
) -> List[Dict]:
    """Score confidence for all recommendations."""
    return [
        score_confidence(user_prefs, song, score, explanation)
        for song, score, explanation in recommendations
    ]


# ---------------------------------------------------------------------------
# Self-Critique via LLM
# ---------------------------------------------------------------------------

def self_critique(
    user_prefs: Dict,
    recommendations: List[Tuple[Dict, float, str]],
    confidence_scores: List[Dict],
) -> Dict:
    """Ask the LLM to review and critique the recommendations.

    Returns a dict with critique text, issues found, and suggestions.
    """
    # Build the critique prompt
    rec_summary = []
    for i, ((song, score, _), conf) in enumerate(zip(recommendations, confidence_scores), 1):
        rec_summary.append(
            f"  {i}. {song['title']} by {song['artist']} "
            f"(genre={song['genre']}, mood={song['mood']}, energy={song['energy']}) "
            f"— score: {score:.2f}, confidence: {conf['confidence']:.2f} ({conf['level']})"
        )

    rec_block = "\n".join(rec_summary)
    avg_confidence = sum(c["confidence"] for c in confidence_scores) / len(confidence_scores) if confidence_scores else 0

    prompt = f"""You are a music recommendation quality reviewer. Critique these recommendations and identify any issues. Be specific and actionable. Respond in JSON format with keys: "issues" (list of strings), "suggestions" (list of strings), "overall_assessment" (string).

USER PROFILE:
- Genre: {user_prefs.get('genre')}
- Mood: {user_prefs.get('mood')}
- Energy: {user_prefs.get('energy')}
- Likes acoustic: {user_prefs.get('likes_acoustic')}

RECOMMENDATIONS:
{rec_block}

Average confidence: {avg_confidence:.2f}

Review these recommendations:"""

    try:
        from llm_client import chat_json, AGENT_MODEL
        result = chat_json(prompt, model=AGENT_MODEL)

        if result.get("parse_error"):
            return _fallback_critique(recommendations, confidence_scores, avg_confidence)

        return {
            "issues": result.get("issues", []),
            "suggestions": result.get("suggestions", []),
            "overall_assessment": result.get("overall_assessment", "No assessment available."),
            "avg_confidence": round(avg_confidence, 2),
            "source": "llm",
        }
    except Exception:
        return _fallback_critique(recommendations, confidence_scores, avg_confidence)


def _fallback_critique(
    recommendations: List[Tuple[Dict, float, str]],
    confidence_scores: List[Dict],
    avg_confidence: float,
) -> Dict:
    """Rule-based fallback critique when LLM is unavailable."""
    issues = []
    suggestions = []

    # Check for low confidence
    low_conf = [c for c in confidence_scores if c["level"] == "low"]
    if low_conf:
        issues.append(f"{len(low_conf)} recommendation(s) have low confidence")
        suggestions.append("Consider re-running with a different strategy or adjusting weights")

    # Check for same-genre domination
    genres = [s["genre"] for s, _, _ in recommendations]
    if len(set(genres)) == 1 and len(genres) > 2:
        issues.append(f"All recommendations are {genres[0]} — no genre diversity")
        suggestions.append("Enable diversity penalty or try Genre-First strategy")

    # Check for missing mood matches
    moods = [s["mood"] for s, _, _ in recommendations]
    unique_moods = set(moods)
    if len(unique_moods) == 1:
        issues.append("All recommendations share the same mood — limited emotional range")

    assessment = "Good" if avg_confidence >= 0.6 else "Needs improvement"
    if not issues:
        assessment = "Strong recommendations with no major issues detected"

    return {
        "issues": issues if issues else ["No significant issues found"],
        "suggestions": suggestions if suggestions else ["Results look solid"],
        "overall_assessment": assessment,
        "avg_confidence": round(avg_confidence, 2),
        "source": "rule-based",
    }


def should_refine(confidence_scores: List[Dict], threshold: float = 0.4) -> bool:
    """Determine if recommendations should be refined based on confidence."""
    if not confidence_scores:
        return False
    avg = sum(c["confidence"] for c in confidence_scores) / len(confidence_scores)
    low_count = sum(1 for c in confidence_scores if c["confidence"] < threshold)
    return avg < threshold or low_count >= len(confidence_scores) // 2


def format_confidence(
    recommendations: List[Tuple[Dict, float, str]],
    confidence_scores: List[Dict],
) -> str:
    """Format confidence scores as readable output."""
    lines = []
    lines.append("=" * 60)
    lines.append("  CONFIDENCE SCORES")
    lines.append("=" * 60)

    for i, ((song, score, _), conf) in enumerate(zip(recommendations, confidence_scores), 1):
        level_icon = {"high": "+", "medium": "~", "low": "!"}[conf["level"]]
        lines.append(
            f"  [{level_icon}] {i}. {song['title'][:25]:<25} "
            f"conf={conf['confidence']:.2f} ({conf['level']})"
        )
        lines.append(f"      {conf['reasoning']}")

    avg = sum(c["confidence"] for c in confidence_scores) / len(confidence_scores) if confidence_scores else 0
    lines.append("")
    lines.append(f"  Average confidence: {avg:.2f}")
    lines.append("=" * 60)

    return "\n".join(lines)


def format_critique(critique: Dict) -> str:
    """Format self-critique as readable output."""
    lines = []
    lines.append("=" * 60)
    lines.append(f"  SELF-CRITIQUE (source: {critique.get('source', 'unknown')})")
    lines.append("=" * 60)

    lines.append("  Issues:")
    for issue in critique.get("issues", []):
        lines.append(f"    - {issue}")

    lines.append("")
    lines.append("  Suggestions:")
    for suggestion in critique.get("suggestions", []):
        lines.append(f"    - {suggestion}")

    lines.append("")
    lines.append(f"  Assessment: {critique.get('overall_assessment', 'N/A')}")
    lines.append(f"  Avg confidence: {critique.get('avg_confidence', 0):.2f}")
    lines.append("=" * 60)

    return "\n".join(lines)
