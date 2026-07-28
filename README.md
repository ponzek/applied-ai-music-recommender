# Applied AI Music Recommender

A hybrid content-based music recommendation system with agentic AI orchestration, RAG-enhanced explanations, bias detection, confidence scoring, self-critique, and a Grover's algorithm quantum search simulation.

> **Extended from:** Module 1-3 Music Recommender Simulation  
> **Course:** AI110 â€” Applied AI  
> **Dataset:** 26,399 tracks sourced from the [Spotify Tracks Dataset](https://www.kaggle.com/datasets/maharshipandya/spotify-tracks-dataset) by MaharshiPandya. Audio features (energy, valence, danceability, acousticness) from Spotify's Web API. Mood tags derived via rule-based inference from valence and energy.

---

## What This Project Does

Takes a user profile (genre, mood, energy level, acoustic preference) and recommends music using a multi-stage AI pipeline:

1. **Plan** â€” Hermes3 analyzes preferences and selects the optimal scoring strategy
2. **Retrieve** â€” RAG searches a knowledge base for genre/mood/artist context
3. **Validate** â€” Agent checks catalog data quality (missing fields, unknown genres)
4. **Recommend** â€” Hybrid scoring engine ranks 26,399 songs
5. **Evaluate** â€” Measures relevance, diversity, coverage, and novelty
6. **Bias Check** â€” Context-aware fairness checks (won't flag genre bias if you asked for that genre)
7. **Confidence + Self-Critique** â€” Rates each recommendation's confidence, Hermes3 reviews for issues
8. **Refine** â€” If confidence is low, agent switches strategies and re-runs automatically
9. **Explain** â€” Qwen3 generates RAG-enhanced natural language explanations
10. **Quantum Search** â€” Grover's algorithm simulation demonstrates O(âˆšN) speedup over classical O(N) search

---

## Quick Start

### Prerequisites

- Python 3.10+
- [Ollama](https://ollama.com) installed with models:
  - `hermes3` (agentic planning, self-critique)
  - `qwen3` (text generation, explanations)

### Setup

```bash
# Clone and enter
cd applied-ai-music-recommender

# Create virtual environment
python -m venv .venv
.venv\Scripts\activate  # Windows
# source .venv/bin/activate  # macOS/Linux

# Install dependencies
pip install -r requirements.txt

# Start Ollama (if not running)
ollama serve

# Pull models
ollama pull hermes3
ollama pull qwen3
```

### Run

```bash
# Full AI agent pipeline (default: 3 profiles)
python -m src.main

# Single profile
python -m src.main --profile melancholic-rock

# Original scoring engine (no AI)
python -m src.main --mode basic

# Test harness
python -m src.main --test

# Quantum search comparison
python src/quantum_search.py

# Streamlit web UI
streamlit run src/app.py

# Run tests
python -m pytest tests/ -v
```

---

## Architecture

```
User Profile â”€â”€â–º Agent Pipeline (agent.py)
                    â”‚
                    â”œâ”€â”€ 1. Plan â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â–º Hermes3 (strategy selection)
                    â”œâ”€â”€ 2. Retrieve â”€â”€â”€â”€â”€â”€â–º RAG (music_knowledge.json)
                    â”œâ”€â”€ 2.5 Validate â”€â”€â”€â”€â”€â–º Data quality check
                    â”œâ”€â”€ 3. Recommend â”€â”€â”€â”€â”€â–º Scoring engine (recommender.py)
                    â”œâ”€â”€ 4. Evaluate â”€â”€â”€â”€â”€â”€â–º Quality metrics
                    â”œâ”€â”€ 5. Bias Check â”€â”€â”€â”€â–º Fairness audit
                    â”œâ”€â”€ 6. Confidence â”€â”€â”€â”€â–º Per-song rating
                    â”œâ”€â”€ 7. Self-Critique â”€â–º Hermes3 (quality review)
                    â”œâ”€â”€ 8. Refine â”€â”€â”€â”€â”€â”€â”€â”€â–º Strategy switch (if needed)
                    â””â”€â”€ 9. Explain â”€â”€â”€â”€â”€â”€â”€â–º Qwen3 + RAG context
                    
Quantum Search (quantum_search.py)
                    â”‚
                    â”œâ”€â”€ Classical: O(N) linear scan
                    â””â”€â”€ Grover's:  O(âˆšN) quantum simulation
                         26,399 songs â†’ 162 iterations (163x speedup)
```

See [diagrams/architecture.mmd](diagrams/architecture.mmd) for the full Mermaid diagram.

---

## Dataset

| Metric | Value |
|--------|-------|
| **Source** | [Spotify Tracks Dataset](https://www.kaggle.com/datasets/maharshipandya/spotify-tracks-dataset) (MaharshiPandya) |
| **Total songs** | 26,399 |
| **Genres** | 13 (pop, rock, alt-rock, lofi, ambient, jazz, r&b, hip-hop, reggaeton, indie pop, synthwave, prog-rock, corridos) |
| **Moods** | 14 (happy, chill, intense, melancholic, heartbreak, dark, hype, romantic, nostalgic, focused, energetic, relaxed, moody, confident) |
| **Features per song** | energy, valence, danceability, acousticness, tempo, popularity |
| **Languages** | English (en), Spanish (es) |

**Data pipeline:** The `scripts/build_dataset.py` script downloads from HuggingFace, maps Spotify's 125 genre tags to our 13 knowledge base genres, assigns mood tags using rule-based inference from valence + energy, and exports the final CSV.

**Mood assignment logic:**
- High valence + high energy = hype/happy
- Low valence + low energy = melancholic/heartbreak
- Low valence + high energy = dark/intense
- High valence + low energy = romantic/relaxed

---

## AI Models

| Model | Role | Why This Model |
|-------|------|----------------|
| **Hermes3** (NousResearch) | Agentic planning, strategy selection, self-critique | Fine-tuned for function calling and structured JSON output. Self-improving agent behavior. |
| **Qwen3** (Alibaba) | Text generation, RAG-enhanced explanations, bias summaries | Strong reasoning and text quality. Supports `/no_think` for direct output. |

Both models run locally via **Ollama** â€” no API keys, no cloud costs, no data leaving your machine.

---

## Stretch Features Implemented

### 1. Grover's Algorithm Simulation (Quantum Search)
- Simulates quantum amplitude amplification on classical hardware
- Demonstrates O(âˆšN) search: 26,399 songs â†’ 162 iterations vs 26,399 classical operations
- 153x theoretical speedup, scaling to 1,274x at 1M songs
- Finds the exact same result as classical search with P(target) = 1.0000

### 2. Test Harness
- Predefined input profiles with expected outputs
- Automated pass/fail evaluation
- Run with `python -m src.main --test`

### 3. Agentic Data Validation
- Agent checks catalog quality at runtime before scoring
- Flags missing moods, unknown genres, invalid energy values
- Demonstrates self-checking agentic behavior

### 4. Streamlit Web UI
- Interactive profile builder
- Real-time recommendations with evaluation metrics
- Bias detection and confidence visualization
- RAG-enhanced explanations via LLM

---

## Project Structure

```
applied-ai-music-recommender/
â”œâ”€â”€ data/
â”‚   â”œâ”€â”€ songs.csv                 # 26,399 songs from Spotify dataset
â”‚   â””â”€â”€ music_knowledge.json      # RAG knowledge base (genres, moods, artists)
â”œâ”€â”€ src/
â”‚   â”œâ”€â”€ main.py                   # CLI entry point (basic/agent/test modes)
â”‚   â”œâ”€â”€ recommender.py            # Scoring engine + strategy pattern
â”‚   â”œâ”€â”€ agent.py                  # Agentic workflow orchestrator
â”‚   â”œâ”€â”€ rag.py                    # Retrieval-Augmented Generation
â”‚   â”œâ”€â”€ bias_detector.py          # Fairness analysis (4 checks)
â”‚   â”œâ”€â”€ evaluation.py             # Quality metrics (relevance/diversity/coverage/novelty)
â”‚   â”œâ”€â”€ confidence.py             # Confidence scoring + self-critique
â”‚   â”œâ”€â”€ llm_client.py             # Ollama REST client (Hermes3 + Qwen3)
â”‚   â”œâ”€â”€ logger.py                 # Structured execution logging
â”‚   â”œâ”€â”€ quantum_search.py         # Grover's algorithm simulation
â”‚   â””â”€â”€ app.py                    # Streamlit web UI
â”œâ”€â”€ tests/
â”‚   â””â”€â”€ test_recommender.py       # 17 tests (scoring, RAG, bias, confidence, evaluation)
â”œâ”€â”€ scripts/
â”‚   â””â”€â”€ build_dataset.py          # Dataset download + processing pipeline
â”œâ”€â”€ diagrams/
â”‚   â””â”€â”€ architecture.mmd          # System architecture (Mermaid)
â”œâ”€â”€ logs/                         # Agent execution logs (per-run)
â”œâ”€â”€ model_card.md                 # Reflection + ethics + limitations
â”œâ”€â”€ ai_interactions.md            # AI collaboration documentation
â”œâ”€â”€ requirements.txt              # Python dependencies
â”œâ”€â”€ .env.example                  # Environment config template
â””â”€â”€ .gitignore
```

---

## Testing

```bash
# Unit tests (17 tests, all passing)
python -m pytest tests/ -v

# Coverage:
#   - Original scoring engine (3 tests)
#   - RAG retrieval + mood similarity (5 tests)
#   - Bias detection (3 tests)
#   - Confidence scoring (4 tests)
#   - Evaluation metrics (2 tests)
```

---

## Design Decisions

### Why Content-Based Over Collaborative Filtering?
In a production hybrid recommender, collaborative filtering ("users who liked X also liked Y") fills data gaps automatically â€” it doesn't need mood tags because it learns from behavior. But we're simulating with preset profiles, not real users. Content-based filtering with explicit features (genre, mood, energy) gives us:
- Full transparency into *why* a song was recommended
- Deterministic, reproducible results
- No cold-start problem for new users

### Why Rule-Based Mood Assignment Over LLM?
The LLM could infer mood from song titles and artist names (and would probably be more accurate for individual songs), but:
- Rule-based is deterministic and reproducible
- It processes 14K songs in milliseconds (vs. hours of LLM calls)
- Audio features (valence, energy) are scientifically grounded by Spotify's analysis
- The tradeoff: it can't detect lyrical content (a happy-sounding breakup song gets tagged "happy")

### Why Grover's Simulation Over Real Quantum?
Real quantum hardware (IBM Qiskit, etc.) requires cloud access and has noise/decoherence issues. The simulation demonstrates the *mathematical structure* of the algorithm â€” oracle marking + diffusion amplification â€” which is the conceptual insight. The speedup numbers are real: 162 operations vs 26,399 for our catalog.

---

## License

Educational project â€” MIT License.

## Acknowledgments

- [Spotify Tracks Dataset](https://www.kaggle.com/datasets/maharshipandya/spotify-tracks-dataset) by MaharshiPandya
- [Ollama](https://ollama.com) â€” local LLM runtime
- [Hermes3](https://nousresearch.com) â€” NousResearch
- [Qwen3](https://huggingface.co/Qwen) â€” Alibaba Cloud
