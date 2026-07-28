# AI Interactions Log

Documentation of how AI tools were used throughout this project, including prompts, reasoning, and what was learned.

---

## Tool Usage Summary

| AI Tool | Purpose | Key Contribution |
|---------|---------|-----------------|
| **Hermes3** (via Ollama) | In-system agentic planning and self-critique | Strategy selection, recommendation quality review |
| **Qwen3** (via Ollama) | In-system text generation | RAG-enhanced explanations, bias summaries |

---

## Interaction 1: RAG System Design

**Prompt to ChatGPT**
> "How should the RAG system work for a music recommender? We have a knowledge base with genre descriptions and mood mappings."

**AI Suggestion:**
The AI suggested a three-stage RAG pipeline:
1. **Retrieve** â€” Match user preferences against the knowledge base using keyword overlap
2. **Augment** â€” Build a context-rich prompt with retrieved knowledge
3. **Generate** â€” Feed the augmented prompt to Qwen3 for natural language explanations

**Key insight the AI provided:** Use mood *similarity scoring* instead of exact matching. The AI suggested treating moods like "melancholic" and "heartbreak" as semantically related (similarity = 0.7) rather than binary match/mismatch. This dramatically improved recommendations for users whose mood preferences didn't exactly match available songs.

**What I learned:** RAG isn't just "find relevant text and paste it into a prompt." The retrieval quality directly determines the generation quality. A simple keyword matcher with domain-specific similarity scoring outperformed a more complex approach because it was tuned to our specific domain.

---

## Interaction 2: Bias Detection Architecture

**Prompt to ChatGPT:**
> "What types of bias should we check for in a music recommender? How can we measure them? Lastely, I need help coding this part."

**AI Suggestion:**
The AI identified four independent bias dimensions:
1. Genre concentration (>60% same genre = bias)
2. Popularity bias (all popular or all obscure)
3. Language exclusion (only one language represented)
4. Artist domination (same artist appears multiple times)

**What I modified:** I adjusted the popularity bias threshold. The AI suggested flagging when average popularity > 70, but I changed it to also flag when average < 30 (recommending only obscure tracks is also a form of bias â€” it ignores mainstream music the user might enjoy).

**What I learned:** Bias detection isn't just about catching obvious problems, it's about defining what "fair" means for your specific domain. A music recommender's fairness criteria are different from a hiring algorithm's.

---

## Interaction 3: Windows Encoding Crash

**The problem:** The entire agent pipeline crashed with `UnicodeEncodeError: 'charmap' codec can't encode characters` when trying to print Unicode box-drawing characters (â”€, âš , âœ“) on Windows.

**AI's initial approach:** Used Unicode characters for prettier terminal output â€” a reasonable choice on Linux/macOS where UTF-8 is default.

**The fix:** Replaced all Unicode characters with ASCII equivalents (`-` instead of `â”€`, `!!` instead of `âš `, `OK` instead of `âœ“`). All about data clearning.

**What I learned:** Cross-platform compatibility is a real engineering concern. The AI assumed UTF-8 encoding (common in tutorials and docs), but Windows PowerShell defaults to cp1252. This is the kind of bug that never appears in development on one OS but breaks everything on another.

---

## Interaction 4: Qwen3 Think Blocks

**The problem:** Qwen3 wraps its reasoning in `<think>...</think>` tags. When the model spent all its tokens on thinking, the "actual" response after stripping think blocks was empty â€” resulting in blank explanations.

**First fix attempt:** Simple regex: `re.sub(r"<think>.*?</think>", "", content)`. This worked when there was content after the think block, but failed when the entire response was inside `<think>`.

**Final fix:** A three-tier fallback:
1. Strip think blocks; use remaining content if non-empty
2. If empty, extract text after `</think>` tag
3. If still empty, strip just the tags and keep the thinking content

Also added `/no_think` to prompts to tell Qwen3 to skip internal reasoning.

**What I learned:** Open-source LLMs have model-specific quirks that aren't documented in the same way as commercial APIs. Defensive coding (fallback chains, not assuming output format) is essential when working with local models.

---

## Interaction 5: Dataset Engineering

**Decision:** Use the full Spotify Tracks Dataset (114K tracks â†’ 26,399 mapped to our genres) instead of a hand-curated 20-song catalog.

**AI's role:** Built the `build_dataset.py` pipeline that:
- Downloads from HuggingFace (no auth required)
- Maps Spotify's 125 genre tags to our 13 knowledge base genres
- Assigns mood tags using rule-based inference from valence + energy
- Adds manual entries for genres Spotify doesn't tag (prog-rock, corridos)

**Design discussion - Rule-based vs LLM mood assignment:**
- The AI suggested using the LLM to assign moods (more accurate for individual songs)
- I chose rule-based because it's deterministic, reproducible, and processes 14K songs in milliseconds
- We discussed having the agent fill gaps at runtime as a hybrid approach
- This became the "data validation" step in the agent pipeline

**What I learned:** Data engineering decisions cascade through the entire system. A bigger catalog improved evaluation grades (from D to B-range), gave the bias detector more interesting patterns to find, and made the Grover's quantum search demo more compelling (153x speedup with 14K songs vs 4x with 20).

---

## Interaction 6: Grover's Algorithm Connection

**My insight:** The connection between quantum search and music recommendation; both are about finding the best match in a large unstructured dataset.

**AI's contribution:** Built the Grover simulation with:
- Proper amplitude initialization (1/âˆšN)
- Oracle (amplitude flip) and diffusion (reflect about mean) operators
- Optimal iteration count (âŒŠÏ€/4 Â· âˆšNâŒ‹)
- Scale projections showing speedup at 1K, 10K, 100K, 1M songs

**Results:** On our 26,399-song catalog:
- Classical: 26,399 operations
- Grover's: 162 iterations
- Speedup: 163x
- Found the exact same song with P(target) = 1.0000

**What I learned:** Quantum algorithms aren't magic, they're mathematical structures that exploit superposition and interference. The simulation showed me *exactly* how the amplitude of the target item grows from 1/âˆšN to ~1.0 over âˆšN iterations. Understanding this on real data was more valuable than any textbook explanation.

---

## Reflection: AI as a Development Partner

### What worked well
- AI did well at boilerplate and architecture scaffolding; it generated the structure, I made the design decisions
- Pattern recognition: the AI caught edge cases I missed (like the mood similarity scoring) and helped with the bia detection.
- Debugging: the AI diagnosed the Windows encoding issue immediately after seeing the error, made sure everything is cleaned up .

### What required my judgment
- **Ethical tradeoffs:** The AI suggested features, but I decided what bias thresholds were appropriate for a music context
- **Scope management:** The AI would have built everything simultaneously; I had to prioritize what mattered for the rubric
- **Design philosophy:** Choosing rule-based over LLM for mood assignment was a human judgment about reproducibility vs accuracy

### What I'd do differently
- Start with a larger dataset from day one 
- The encoding bug wasted debugging time from the dataset but that was something I implemented.
- Use `/no_think` in all Qwen3 prompts from the start and learning these options.
