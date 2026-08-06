"""RAG — Retrieval-Augmented Generation for music recommendations.

Searches the local knowledge base for relevant context about genres, moods,
and artists, then uses the LLM to generate richer explanations.
"""

import json
import os
import re
from typing import List, Dict, Tuple

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# ---------------------------------------------------------------------------
# Knowledge Base
# ---------------------------------------------------------------------------

_knowledge_cache: Dict = {}


def load_knowledge() -> Dict:
    """Load the music knowledge base from JSON."""
    global _knowledge_cache
    if _knowledge_cache:
        return _knowledge_cache

    kb_path = os.path.join(PROJECT_ROOT, "data", "music_knowledge.json")
    with open(kb_path, "r", encoding="utf-8") as f:
        _knowledge_cache = json.load(f)
    return _knowledge_cache


# ---------------------------------------------------------------------------
# Retrieval — Find Relevant Knowledge
# ---------------------------------------------------------------------------

def retrieve(query_terms: Dict, top_k: int = 5) -> List[Dict]:
    """Retrieve the most relevant knowledge entries for a user's preferences.

    Args:
        query_terms: Dict with keys like 'genre', 'mood', 'artist', 'decade'.
        top_k: Maximum number of knowledge entries to return.

    Returns:
        List of dicts, each with 'category', 'key', 'content', and 'relevance'.
    """
    kb = load_knowledge()
    results = []

    genre = query_terms.get("genre", "")
    mood = query_terms.get("mood", "")
    artist = query_terms.get("artist", "")

    # 1. Look up the genre
    if genre and genre in kb.get("genres", {}):
        genre_info = kb["genres"][genre]
        results.append({
            "category": "genre",
            "key": genre,
            "content": genre_info["description"],
            "relevance": 1.0,
        })
        # Check for era-specific info
        decade = query_terms.get("decade", "")
        if decade and decade in genre_info.get("era_notes", {}):
            results.append({
                "category": "genre_era",
                "key": f"{genre} ({decade})",
                "content": genre_info["era_notes"][decade],
                "relevance": 0.9,
            })

    # 2. Look up the mood
    if mood and mood in kb.get("moods", {}):
        mood_info = kb["moods"][mood]
        results.append({
            "category": "mood",
            "key": mood,
            "content": mood_info["description"],
            "relevance": 1.0,
        })
        # Include similar moods for context
        similar = mood_info.get("similar_moods", [])
        if similar:
            results.append({
                "category": "mood_context",
                "key": f"moods similar to {mood}",
                "content": f"Related moods: {', '.join(similar)}",
                "relevance": 0.7,
            })

    # 3. Look up the artist
    if artist:
        for artist_name, artist_info in kb.get("artists", {}).items():
            if artist.lower() in artist_name.lower():
                results.append({
                    "category": "artist",
                    "key": artist_name,
                    "content": artist_info["bio"],
                    "relevance": 0.95,
                })
                break

    # 4. Add relevant music concepts
    for concept_key, concept_info in kb.get("music_concepts", {}).items():
        concept_text = concept_info["description"].lower()
        if genre.lower() in concept_text or mood.lower() in concept_text:
            results.append({
                "category": "concept",
                "key": concept_key,
                "content": concept_info["description"],
                "relevance": 0.6,
            })

    # Sort by relevance and limit
    results.sort(key=lambda x: x["relevance"], reverse=True)
    return results[:top_k]


def get_similar_moods(mood: str) -> List[str]:
    """Return moods that are similar to the given mood."""
    kb = load_knowledge()
    mood_info = kb.get("moods", {}).get(mood, {})
    return mood_info.get("similar_moods", [])


def get_mood_similarity_score(mood_a: str, mood_b: str) -> float:
    """Return a similarity score (0.0 to 1.0) between two moods.

    1.0 = exact match, 0.7 = similar moods, 0.0 = unrelated.
    """
    if mood_a == mood_b:
        return 1.0

    kb = load_knowledge()
    mood_info = kb.get("moods", {}).get(mood_a, {})
    similar = mood_info.get("similar_moods", [])

    if mood_b in similar:
        return 0.7

    # Check reverse direction too
    mood_b_info = kb.get("moods", {}).get(mood_b, {})
    similar_b = mood_b_info.get("similar_moods", [])
    if mood_a in similar_b:
        return 0.7

    # Check if they share any opposite moods (weak signal)
    opposites_a = set(mood_info.get("opposite_moods", []))
    opposites_b = set(mood_b_info.get("opposite_moods", []))
    if opposites_a and opposites_b and opposites_a == opposites_b:
        return 0.4

    return 0.0


# ---------------------------------------------------------------------------
# Augmentation — Build Enhanced Prompts
# ---------------------------------------------------------------------------

def build_context_prompt(user_prefs: Dict, song: Dict, retrieved: List[Dict]) -> str:
    """Build an LLM prompt that includes retrieved knowledge for richer explanations.

    Args:
        user_prefs: The user's preference dict (genre, mood, energy, etc.).
        song: The recommended song dict.
        retrieved: List of retrieved knowledge entries.

    Returns:
        A formatted prompt string for the LLM.
    """
    # Format retrieved knowledge
    context_lines = []
    for entry in retrieved:
        context_lines.append(f"- [{entry['category']}] {entry['key']}: {entry['content']}")
    context_block = "\n".join(context_lines) if context_lines else "No additional context available."

    prompt = f"""You are a music recommendation expert. Using the context below, explain why this song is a good match for this listener. Be specific and reference the musical context. Keep it to 2-3 sentences.

MUSIC CONTEXT:
{context_block}

LISTENER PROFILE:
- Favorite genre: {user_prefs.get('genre', 'unknown')}
- Favorite mood: {user_prefs.get('mood', 'unknown')}
- Target energy: {user_prefs.get('energy', 0.5)}
- Likes acoustic: {user_prefs.get('likes_acoustic', False)}

RECOMMENDED SONG:
- Title: {song.get('title', 'Unknown')}
- Artist: {song.get('artist', 'Unknown')}
- Genre: {song.get('genre', 'unknown')}
- Mood: {song.get('mood', 'unknown')}
- Energy: {song.get('energy', 0.0)}

Write a brief, insightful explanation of why this song fits this listener. /no_think"""

    return prompt


def explain_with_context(user_prefs: Dict, song: Dict) -> str:
    """Generate a RAG-enhanced explanation for a recommendation.

    Retrieves relevant knowledge, builds a context-aware prompt,
    and asks the LLM for a rich explanation.

    Falls back to a basic explanation if the LLM is unavailable.
    """
    # Retrieve relevant knowledge
    query = {
        "genre": song.get("genre", ""),
        "mood": song.get("mood", ""),
        "artist": song.get("artist", ""),
        "decade": song.get("release_decade", ""),
    }
    retrieved = retrieve(query)

    # Build the prompt
    prompt = build_context_prompt(user_prefs, song, retrieved)

    # Try LLM, fall back to basic
    try:
        from src.llm_client import chat, TEXT_MODEL
        explanation = chat(prompt, model=TEXT_MODEL, temperature=0.6, max_tokens=200)
        return explanation
    except Exception as e:
        # Fallback: return a simple knowledge-based explanation
        fallback_parts = []
        for entry in retrieved[:2]:
            fallback_parts.append(entry["content"])
        if fallback_parts:
            return " | ".join(fallback_parts)
        return f"Matched on {song.get('genre', '?')} genre and {song.get('mood', '?')} mood."
