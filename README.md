# Applied AI Music Recommender

A hybrid content-based music recommendation system with agentic AI orchestration, RAG-enhanced explanations, bias detection, confidence scoring, self-critique, and a Grover's algorithm quantum search simulation.

> **Extended from:** Module 1-3 Music Recommender Simulation  
> **Course:** AI110 — Applied AI  
> **Dataset:** 26,399 tracks sourced from the [Spotify Tracks Dataset](https://www.kaggle.com/datasets/maharshipandya/spotify-tracks-dataset) by MaharshiPandya. Audio features (energy, valence, danceability, acousticness) from Spotify's Web API. Mood tags derived via rule-based inference from valence and energy.

---

## What This Project Does

Takes a user profile (genre, mood, energy level, acoustic preference) and recommends music using a multi-stage AI pipeline:

1. **Plan** — Hermes3 analyzes preferences and selects the optimal scoring strategy
2. **Retrieve** — RAG searches a knowledge base for genre/mood/artist context
3. **Validate** — Agent checks catalog data quality (missing fields, unknown genres)
4. **Recommend** — Hybrid scoring engine ranks 26,399 songs
5. **Evaluate** — Measures relevance, diversity, coverage, and novelty
6. **Bias Check** — Context-aware fairness checks (won't flag genre bias if you asked for that genre)
7. **Confidence + Self-Critique** — Rates each recommendation's confidence, Hermes3 reviews for issues
8. **Refine** — If confidence is low, agent switches strategies and re-runs automatically
9. **Explain** — Qwen3 generates RAG-enhanced natural language explanations
10. **Quantum Search** — Grover's algorithm simulation demonstrates O(âˆšN) speedup over classical O(N) search

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

### Sample Output

Running `python -m src.main --profile melancholic-rock` produces:

```
======================================================================
  MUSIC RECOMMENDER AGENT — FINAL RESULTS
======================================================================
  Profile: genre=alt-rock, mood=melancholic, energy=0.45, acoustic=True
  Strategy: mood-first
  Refinements: 0

  #   Title                        Artist               Score   Conf   Mood
  --- ---------------------------- -------------------- ------- ------ ------------
  1   Cure My Tragedy              Cold                 8.12    0.87   melancholic
  2   The Reason                   Hoobastank           7.89    0.83   heartbreak
  3   Another Love (Remix)         Tiesto               7.45    0.67   melancholic
  4   Resonance                    HOME                 7.22    0.67   moody
  5   GET IT                       Purity Ring          6.98    0.67   dark

  EVALUATION
  Relevance:  0.96    Diversity: 0.93    Coverage: 0.07    Novelty: 0.27
  Grade: A (0.89)

  BIAS CHECK
  Genre................ OK (4/5 match requested genre alt-rock)
  Popularity........... OK (avg 58/100)
  Language............. OK
  Artist repetition.... OK
  Verdict: No obvious bias patterns were found.

  CONFIDENCE
  Average: 0.74
  Assessment: Recommendations are solid with good preference alignment.
======================================================================
```

The system also works with the Streamlit web UI:

```bash
streamlit run src/app.py
# Opens browser at http://localhost:8501
# Select genre, mood, energy, strategy → click Get Recommendations
```

### Reliability Example (Guardrail in Action)

The evaluation and bias systems act as guardrails. Here's an example where the system **catches a problem and self-corrects**:

```
Input:  genre=lofi, mood=intense, energy=0.45, acoustic=True
        Strategy: genre-first, k=5

Step 6: Confidence — Average confidence: 0.38 (LOW)
Step 7: Self-Critique — "Mood mismatch: intense mood paired with low energy
        and acoustic preference is contradictory. Recommend switching strategy."

  >> REFINEMENT #1
  Switching from genre-first to mood-first
  New average confidence: 0.72 (acceptable)
  Refinement complete.
```

When confidence drops below 0.4, the agent automatically switches strategies and re-runs, no human intervention needed.

---

## Architecture

```
User Profile ──► Agent Pipeline (agent.py)
                    │
                    ├── 1. Plan ──────────► Hermes3 (strategy selection)
                    ├── 2. Retrieve ──────► RAG (music_knowledge.json)
                    ├── 2.5 Validate ─────► Data quality check
                    ├── 3. Recommend ─────► Scoring engine (recommender.py)
                    ├── 4. Evaluate ──────► Quality metrics
                    ├── 5. Bias Check ────► Fairness audit
                    ├── 6. Confidence ────► Per-song rating
                    ├── 7. Self-Critique ─► Hermes3 (quality review)
                    ├── 8. Refine ────────► Strategy switch (if needed)
                    └── 9. Explain ───────► Qwen3 + RAG context
                    
Quantum Search (quantum_search.py)
                    │
                    ├── Classical: O(N) linear scan
                    └── Grover's:  O(√N) quantum simulation
                         14,267 songs → 93 iterations (153x speedup)
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

Both models run locally via **Ollama**, no API keys, no cloud costs, no data leaving your machine.

---

## Stretch Features Implemented

### 1. Grover's Algorithm Simulation (Quantum Search)
- Simulates quantum amplitude amplification on classical hardware
- Demonstrates O(âˆšN) search: 26,399 songs → 162 iterations vs 26,399 classical operations
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
├── data/
│   ├── songs.csv                 # 14,267 songs from Spotify dataset
│   └── music_knowledge.json      # RAG knowledge base (genres, moods, artists)
├── src/
│   ├── main.py                   # CLI entry point (basic/agent/test modes)
│   ├── recommender.py            # Scoring engine + strategy pattern
│   ├── agent.py                  # Agentic workflow orchestrator
│   ├── rag.py                    # Retrieval-Augmented Generation
│   ├── bias_detector.py          # Fairness analysis (4 checks)
│   ├── evaluation.py             # Quality metrics (relevance/diversity/coverage/novelty)
│   ├── confidence.py             # Confidence scoring + self-critique
│   ├── llm_client.py             # Ollama REST client (Hermes3 + Qwen3)
│   ├── logger.py                 # Structured execution logging
│   ├── quantum_search.py         # Grover's algorithm simulation
│   └── app.py                    # Streamlit web UI
├── tests/
│   └── test_recommender.py       # 17 tests (scoring, RAG, bias, confidence, evaluation)
├── scripts/
│   └── build_dataset.py          # Dataset download + processing pipeline
├── diagrams/
│   └── architecture.mmd          # System architecture (Mermaid)
├── logs/                         # Agent execution logs (per-run)
├── model_card.md                 # Reflection + ethics + limitations
├── ai_interactions.md            # AI collaboration documentation
├── requirements.txt              # Python dependencies
├── .env.example                  # Environment config template
└── .gitignore
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
Real quantum hardware (IBM Qiskit, etc.) requires cloud access and has noise/decoherence issues. The simulation demonstrates the *mathematical structure* of the algorithm â€” oracle marking + diffusion amplification, which is the conceptual insight. The speedup numbers are real: 162 operations vs 26,399 for our catalog.

---

## Acknowledgments

- [Spotify Tracks Dataset](https://www.kaggle.com/datasets/maharshipandya/spotify-tracks-dataset) by MaharshiPandya
- [Ollama](https://ollama.com) â€” local LLM runtime
- [Hermes3](https://nousresearch.com) â€” NousResearch
- [Qwen3](https://huggingface.co/Qwen) â€” Alibaba Cloud
