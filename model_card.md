# Model Card: Music Recommender -- Applied AI System

## Model Overview

| Field | Details |
|-------|---------|
| **System Name** | Applied AI Music Recommender |
| **Version** | 2.0 (Extended from Module 1-3 Music Recommender Simulation) |
| **Models Used** | Hermes3 (agentic planning, self-critique) + Qwen3 (text generation, explanations) |
| **Runtime** | Ollama (local, open-source, no API key required) |
| **Type** | Hybrid content-based recommendation system with agentic AI orchestration |
| **Dataset** | 26,399 songs sourced from Spotify Tracks Dataset (Kaggle/HuggingFace), 13 genres, 14 moods |

## Intended Use

This system recommends music based on user preferences (genre, mood, energy, acoustic preference) and provides AI-powered explanations, bias analysis, and confidence scoring. It is designed as a classroom simulation demonstrating applied AI concepts.

**Intended users:** Students, educators, and portfolio reviewers evaluating applied AI competency.

**Not intended for:** Production music streaming, commercial recommendations, or replacing human curation at scale.

## AI Features Integrated

### 1. Retrieval-Augmented Generation (RAG)
- **What it does:** Searches a local knowledge base (`music_knowledge.json`) for contextual information about genres, moods, and artists before generating explanations.
- **Why it matters:** Without RAG, the system can only say "mood match: melancholic (+3.0)." With RAG, it explains *why* the song fits â€” referencing the genre's history, the artist's style, and the emotional connection.
- **Before RAG:** "genre match: alt-rock (+1.0); mood match: melancholic (+3.0)"
- **After RAG:** "Cold's 'Cure My Tragedy' captures the introspective vulnerability that defines 2000s post-hardcore, perfectly matching your melancholic preference with its atmospheric guitars and haunting vocals."

### 2. Agentic Workflow
- **What it does:** An autonomous agent orchestrates an 8-step pipeline: Plan -> Retrieve -> Validate -> Recommend -> Evaluate -> Bias Check -> Confidence/Self-Critique -> Explain.
- **Self-checking:** The agent validates catalog data quality at runtime, detects missing mood tags or unknown genres, and flags issues before scoring.
- **Strategy selection:** Hermes3 analyzes user preferences and chooses the optimal scoring strategy (mood-first, genre-first, or energy-focused) with reasoning.
- **Refinement loop:** If average confidence drops below 0.4, the agent automatically switches strategies and re-runs.

### 3. Bias Detection
- **Four independent checks:** Genre concentration, popularity bias, language exclusion, artist domination.
- **Each check returns:** A detection flag, a 0-1 score, and human-readable details.
- **LLM summary:** Hermes3 generates a natural-language fairness analysis with actionable suggestions.

### 4. Confidence Scoring + Self-Critique
- **Per-recommendation confidence:** Based on score strength (40%), feature match count (40%), and explanation richness (20%).
- **Self-critique:** Hermes3 reviews all recommendations as a quality auditor, returning structured issues and suggestions.
- **Refinement trigger:** Low confidence triggers automatic strategy switching.

### Baseline vs. Specialized Output

Running the same profile through basic mode vs full agent pipeline shows measurable improvement:

| Metric | Basic Scoring | Agent Pipeline | Improvement |
|--------|--------------|----------------|-------------|
| Relevance | 0.78 | 0.96 | +23% |
| Diversity | 0.60 | 0.93 | +55% |
| Grade | C (0.62) | A (0.89) | 2 letter grades |
| Explanation | "mood match: melancholic (+3.0)" | "Cold's 'Cure My Tragedy' captures the introspective vulnerability that defines 2000s post-hardcore..." | Rich context |
| Bias checks | None | 4 independent fairness audits | New capability |
| Self-correction | None | Auto-refines when confidence < 0.4 | New capability |

The agent's strategy selection (via Hermes3) and RAG-enhanced explanations (via Qwen3) are the primary drivers of quality improvement.

## Limitations and Biases

### Known Limitations
1. **Catalog size:** While 26,399 songs is a solid demo dataset, it's still small compared to production systems (Spotify: 100M+ tracks). Some niche genres have limited representation.
2. **Content-based only:** No collaborative filtering. The system can't learn from user behavior because we simulate with preset profiles, not real interaction data.
3. **Rule-based mood assignment:** Moods are inferred from audio features (valence, energy, acousticness) using rules, not lyrical analysis. A song about heartbreak with an upbeat melody might be mis-tagged as "happy."
4. **LLM latency:** Each agent run takes 2-5 minutes due to multiple LLM calls through Ollama. Not suitable for real-time use without faster inference or caching.
5. **Language bias:** The knowledge base is English-centric. Spanish-language songs (reggaeton, corridos) have less contextual knowledge available.

### Known Biases
- **Genre coverage gap:** Some moods (e.g., "focused", "relaxed") have fewer songs in the catalog, leading to lower relevance scores for users who prefer them.
- **Popularity bias in data source:** The Spotify dataset skews toward popular tracks. The sampling strategy mitigates this by pulling from top, middle, and bottom popularity tiers.
- **Western-centric knowledge base:** Genre descriptions reflect Western music history. Non-Western genres are underrepresented in the knowledge base.

## Could This AI Be Misused?

### Potential Misuse
1. **Filter bubble reinforcement:** A recommender that only shows matching content could narrow users' musical exposure over time.
2. **Manipulative recommendations:** In a commercial setting, recommendations could be biased toward paid placements or label deals rather than user preference.
3. **Privacy concerns:** A production version with collaborative filtering would need user listening data, creating privacy risks.

### Mitigations Built In
- **Diversity penalty:** The scoring engine penalizes consecutive same-genre recommendations.
- **Bias detection:** Every run checks for and reports unfairness.
- **Transparency:** Confidence scores and explanations make the system's reasoning visible and auditable.

## What Surprised Me While Testing

1. **Qwen3's think blocks:** Qwen3 wraps its responses in `<think>...</think>` tags by default. Initially, stripping these left empty explanations. Adding `/no_think` to prompts and improving the stripping logic fixed this. This taught me that open-source LLMs have model-specific quirks that require defensive coding.

2. **The "D" grade:** The evaluation system gave a "D" grade to recommendations from a 20-song catalog. This wasn't a bug â€” it correctly identified that a tiny catalog can't provide good diversity or coverage. Expanding to 101 songs improved grades to B-range, validating the metric.

3. **Strategy selection matters:** Hermes3 consistently chose "mood-first" for emotional moods (melancholic, heartbreak) and "energy-focused" for activity-based profiles. This matched my intuition, showing the LLM was genuinely reasoning about the problem rather than guessing.

## AI Collaboration Reflection

### Helpful AI Suggestion
When building the RAG system, the AI suggested using mood similarity scoring â€” treating "melancholic" and "heartbreak" as related moods that deserve partial credit (0.7) rather than zero. This directly addressed the biggest weakness in my original project (binary mood matching) and significantly improved recommendation quality for edge cases.

### Flawed AI Suggestion
The AI initially suggested using Unicode box-drawing characters (â”€, â”€, âš , âœ“) for terminal output formatting. On Windows with cp1252 encoding, these characters crashed the entire application with a `UnicodeEncodeError`. This required replacing all Unicode characters with ASCII equivalents. The AI assumed a Linux/macOS environment where UTF-8 is default, which is a common oversight in cross-platform development.

## Ethical Considerations

Music recommendations shape what people listen to, which shapes culture. A biased recommender could:
- Systematically underexpose independent or non-English artists
- Reinforce existing popularity hierarchies
- Limit musical discovery

This project addresses these concerns through transparent bias detection, diversity penalties, and evaluation metrics that explicitly measure coverage and novelty. The bias detector doesn't just flag problems â€” it generates actionable suggestions for improvement.

## References

- Spotify Tracks Dataset: [maharshipandya/spotify-tracks-dataset](https://www.kaggle.com/datasets/maharshipandya/spotify-tracks-dataset)
- Hermes3 (NousResearch): Open-source LLM fine-tuned for agentic tasks and function calling
- Qwen3 (Alibaba): Open-source reasoning and text generation model
- Ollama: Local LLM runtime (https://ollama.com)
