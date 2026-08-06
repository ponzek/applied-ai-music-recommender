"""Agent — Agentic workflow orchestrator for the music recommender.

Uses Hermes3 for planning and decision-making. Follows a
Plan → Retrieve → Recommend → Evaluate → Critique → Refine → Explain loop.
"""

import time
from typing import List, Dict, Tuple, Optional

from src.recommender import load_songs, recommend_songs, STRATEGIES
from src.rag import retrieve, explain_with_context, get_mood_similarity_score
from src.bias_detector import generate_bias_report, format_bias_report, get_llm_bias_summary
from src.evaluation import evaluate_recommendations, format_evaluation
from src.confidence import (
    score_all_confidence, self_critique, should_refine,
    format_confidence, format_critique,
)
from src.logger import AgentLogger



class MusicRecommenderAgent:
    """Agentic AI that oversees the full recommendation pipeline.

    Steps:
    1. Plan — Analyze user preferences and choose a strategy
    2. Retrieve — Pull relevant knowledge from the RAG system
    3. Recommend — Run the scoring engine
    4. Evaluate — Check quality metrics
    5. Critique — Score confidence and self-critique
    6. Refine — If confidence is low, adjust and re-run
    7. Explain — Generate rich, context-aware explanations
    """

    def __init__(self, songs: List[Dict] = None, verbose: bool = True):
        """Initialize the agent.

        Args:
            songs: Pre-loaded song catalog. If None, loads from CSV.
            verbose: If True, print detailed step-by-step output.
        """
        self.songs = songs if songs is not None else load_songs("data/songs.csv")
        self.verbose = verbose
        self.logger = AgentLogger()
        self.max_refinements = 2

    def run(self, user_prefs: Dict, k: int = 5) -> Dict:
        """Execute the full agent pipeline for a user profile.

        Args:
            user_prefs: Dict with genre, mood, energy, likes_acoustic.
            k: Number of recommendations to return.

        Returns:
            Dict with recommendations, evaluations, bias report,
            confidence scores, critique, and explanations.
        """
        self.logger.section("AGENT PIPELINE START")
        self.logger.log("Agent", f"Profile: genre={user_prefs.get('genre')}, "
                        f"mood={user_prefs.get('mood')}, "
                        f"energy={user_prefs.get('energy')}, "
                        f"acoustic={user_prefs.get('likes_acoustic')}")

        result = {
            "user_prefs": user_prefs,
            "strategy_used": None,
            "recommendations": [],
            "rag_context": [],
            "evaluation": {},
            "bias_report": {},
            "confidence_scores": [],
            "critique": {},
            "enhanced_explanations": [],
            "refinement_count": 0,
        }

        # Step 1: Plan
        strategy_key = self._plan(user_prefs)
        result["strategy_used"] = strategy_key

        # Step 2: Retrieve knowledge
        rag_context = self._retrieve(user_prefs)
        result["rag_context"] = rag_context

        # Step 2.5: Validate catalog data (agentic self-checking)
        validation = self._validate_data()
        result["data_validation"] = validation

        # Step 3: Recommend
        recommendations = self._recommend(user_prefs, k, strategy_key)
        result["recommendations"] = recommendations

        # Step 4: Evaluate
        evaluation = self._evaluate(user_prefs, recommendations)
        result["evaluation"] = evaluation

        # Step 5: Check bias
        bias_report = self._check_bias(recommendations, user_prefs)
        result["bias_report"] = bias_report

        # Step 6: Score confidence + self-critique
        confidence_scores = self._score_confidence(user_prefs, recommendations)
        result["confidence_scores"] = confidence_scores

        critique = self._self_critique(user_prefs, recommendations, confidence_scores)
        result["critique"] = critique

        # Step 7: Refine if needed
        refinement_count = 0
        while should_refine(confidence_scores) and refinement_count < self.max_refinements:
            refinement_count += 1
            self.logger.section(f"REFINEMENT #{refinement_count}")

            # Try a different strategy
            alt_strategy = self._pick_alternative_strategy(strategy_key)
            self.logger.log("Refine", f"Switching from {strategy_key} to {alt_strategy}")

            recommendations = self._recommend(user_prefs, k, alt_strategy)
            confidence_scores = self._score_confidence(user_prefs, recommendations)

            result["recommendations"] = recommendations
            result["confidence_scores"] = confidence_scores
            result["strategy_used"] = alt_strategy
            strategy_key = alt_strategy

        result["refinement_count"] = refinement_count

        # Step 8: Generate enhanced explanations
        enhanced = self._explain(user_prefs, recommendations)
        result["enhanced_explanations"] = enhanced

        # Finish
        self.logger.finish()
        result["log_file"] = self.logger.get_log_path()

        return result

    # ------------------------------------------------------------------
    # Pipeline Steps
    # ------------------------------------------------------------------

    def _plan(self, user_prefs: Dict) -> str:
        """Step 1: Analyze preferences and choose the best strategy."""
        with self.logger.step("Plan", "Analyzing user preferences") as s:
            # Use LLM to plan (with fallback to rule-based)
            strategy_key = self._choose_strategy_with_llm(user_prefs)
            s.set_output(f"Chosen strategy: {strategy_key}")
            return strategy_key

    def _retrieve(self, user_prefs: Dict) -> List[Dict]:
        """Step 2: Retrieve relevant knowledge from the RAG system."""
        with self.logger.step("Retrieve", "Searching knowledge base") as s:
            query = {
                "genre": user_prefs.get("genre", ""),
                "mood": user_prefs.get("mood", ""),
            }
            context = retrieve(query, top_k=5)
            s.set_output(f"Retrieved {len(context)} knowledge entries")
            return context

    def _validate_data(self) -> Dict:
        """Step 2.5: Validate catalog data quality (agentic self-checking).

        Checks for:
        - Missing or empty mood/genre fields
        - Energy values outside 0-1 range
        - Genres not in knowledge base
        - Mood distribution imbalance
        """
        with self.logger.step("Validate Data", "Checking catalog quality") as s:
            issues = []
            warnings = []
            from collections import Counter
            from src.rag import load_knowledge

            kb = load_knowledge()
            known_genres = set(kb.get("genres", {}).keys())
            known_moods = set(kb.get("moods", {}).keys())

            genres_found = Counter()
            moods_found = Counter()

            for song in self.songs:
                # Check for missing fields
                if not song.get("genre"):
                    issues.append(f"Song '{song.get('title', '?')}' missing genre")
                elif song["genre"] not in known_genres:
                    warnings.append(f"Song '{song['title']}' has unknown genre: {song['genre']}")

                if not song.get("mood"):
                    issues.append(f"Song '{song.get('title', '?')}' missing mood")
                elif song["mood"] not in known_moods:
                    warnings.append(f"Song '{song['title']}' has unknown mood: {song['mood']}")

                # Check energy range
                energy = song.get("energy", -1)
                if not (0.0 <= energy <= 1.0):
                    issues.append(f"Song '{song['title']}' has invalid energy: {energy}")

                genres_found[song.get("genre", "unknown")] += 1
                moods_found[song.get("mood", "unknown")] += 1

            # Check coverage
            missing_genres = known_genres - set(genres_found.keys())
            missing_moods = known_moods - set(moods_found.keys())

            if missing_genres:
                warnings.append(f"Catalog missing genres: {', '.join(missing_genres)}")
            if missing_moods:
                warnings.append(f"Catalog missing moods: {', '.join(missing_moods)}")

            # Summary
            total_issues = len(issues) + len(warnings)
            status = "clean" if total_issues == 0 else f"{len(issues)} issues, {len(warnings)} warnings"
            s.set_output(f"{len(self.songs)} songs checked - {status}")

            if issues:
                for issue in issues[:5]:
                    self.logger.warn("Validate Data", issue)
            if warnings:
                for warning in warnings[:5]:
                    self.logger.warn("Validate Data", warning)

            return {
                "songs_checked": len(self.songs),
                "issues": issues,
                "warnings": warnings,
                "genres_covered": len(genres_found),
                "moods_covered": len(moods_found),
                "missing_genres": list(missing_genres),
                "missing_moods": list(missing_moods),
                "status": "pass" if not issues else "fail",
            }

    def _recommend(self, user_prefs: Dict, k: int, strategy_key: str) -> List[Tuple[Dict, float, str]]:
        """Step 3: Run the scoring engine."""
        with self.logger.step("Recommend", f"Scoring with {strategy_key} strategy") as s:
            strategy = STRATEGIES.get(strategy_key, STRATEGIES["mood-first"])
            recommendations = recommend_songs(
                user_prefs, self.songs, k=k, strategy=strategy, diversity=True
            )
            if recommendations:
                top_song = recommendations[0][0]
                s.set_output(f"Top pick: {top_song['title']} by {top_song['artist']}")
            return recommendations

    def _evaluate(self, user_prefs: Dict, recommendations: List[Tuple[Dict, float, str]]) -> Dict:
        """Step 4: Run evaluation metrics."""
        with self.logger.step("Evaluate", "Running quality metrics") as s:
            evaluation = evaluate_recommendations(user_prefs, recommendations, self.songs)
            overall = evaluation.get("overall", {})
            s.set_output(f"Overall: {overall.get('score', 0):.2f} ({overall.get('grade', '?')})")

            if self.verbose:
                print(format_evaluation(evaluation))

            return evaluation

    def _check_bias(self, recommendations: List[Tuple[Dict, float, str]], user_prefs: Dict) -> Dict:
        """Step 5: Run bias detection."""
        with self.logger.step("Bias Check", "Scanning for unfairness") as s:
            report = generate_bias_report(recommendations, self.songs, user_prefs)
            summary = report.get("summary", {})
            s.set_output(f"{summary.get('biases_detected', 0)} biases detected")

            if self.verbose:
                print(format_bias_report(report))

            return report

    def _score_confidence(self, user_prefs: Dict, recommendations: List[Tuple[Dict, float, str]]) -> List[Dict]:
        """Step 6a: Score confidence for each recommendation."""
        with self.logger.step("Confidence", "Rating recommendation confidence") as s:
            scores = score_all_confidence(user_prefs, recommendations)
            avg = sum(c["confidence"] for c in scores) / len(scores) if scores else 0
            s.set_output(f"Average confidence: {avg:.2f}")

            if self.verbose:
                print(format_confidence(recommendations, scores))

            return scores

    def _self_critique(
        self,
        user_prefs: Dict,
        recommendations: List[Tuple[Dict, float, str]],
        confidence_scores: List[Dict],
    ) -> Dict:
        """Step 6b: Self-critique via LLM."""
        with self.logger.step("Self-Critique", "LLM reviewing recommendations") as s:
            critique = self_critique(user_prefs, recommendations, confidence_scores)
            source = critique.get("source", "unknown")
            issues_count = len(critique.get("issues", []))
            s.set_output(f"{issues_count} issues found (source: {source})")

            if self.verbose:
                print(format_critique(critique))

            return critique

    def _explain(self, user_prefs: Dict, recommendations: List[Tuple[Dict, float, str]]) -> List[str]:
        """Step 8: Generate RAG-enhanced explanations."""
        explanations = []
        with self.logger.step("Explain", "Generating enhanced explanations") as s:
            for i, (song, score, basic_explanation) in enumerate(recommendations):
                try:
                    enhanced = explain_with_context(user_prefs, song)
                    explanations.append(enhanced)
                except Exception as e:
                    self.logger.warn("Explain", f"LLM failed for {song['title']}: {e}")
                    explanations.append(basic_explanation)

            s.set_output(f"Generated {len(explanations)} explanations")
        return explanations

    # ------------------------------------------------------------------
    # Strategy Selection
    # ------------------------------------------------------------------

    def _choose_strategy_with_llm(self, user_prefs: Dict) -> str:
        """Use the LLM to pick the best scoring strategy. Falls back to rules."""
        try:
            from src.llm_client import chat_json, AGENT_MODEL

            prompt = f"""You are a music recommendation strategist. Given this user profile, choose the best scoring strategy. Respond in JSON with key "strategy" (one of: "mood-first", "genre-first", "energy-focused") and "reasoning" (one sentence why).

User profile:
- Favorite genre: {user_prefs.get('genre')}
- Favorite mood: {user_prefs.get('mood')}
- Target energy: {user_prefs.get('energy')}
- Likes acoustic: {user_prefs.get('likes_acoustic')}

Available strategies:
- "mood-first": Best when the user's emotional state is the primary signal
- "genre-first": Best when the user strongly identifies with a specific genre
- "energy-focused": Best for activity-based listening (workout, study, driving)

Choose the best strategy:"""

            result = chat_json(prompt, model=AGENT_MODEL)
            strategy = result.get("strategy", "mood-first")
            reasoning = result.get("reasoning", "")

            if strategy in STRATEGIES:
                self.logger.log("Plan", f"LLM reasoning: {reasoning}")
                return strategy

        except Exception as e:
            self.logger.warn("Plan", f"LLM planning failed: {e}, using rule-based fallback")

        # Rule-based fallback
        return self._choose_strategy_rules(user_prefs)

    def _choose_strategy_rules(self, user_prefs: Dict) -> str:
        """Rule-based strategy selection fallback."""
        energy = user_prefs.get("energy", 0.5)
        mood = user_prefs.get("mood", "")

        # High energy targets suggest activity-based listening
        if energy >= 0.8 or energy <= 0.3:
            return "energy-focused"

        # Strong mood signals
        emotional_moods = {"melancholic", "heartbreak", "dark", "romantic", "nostalgic"}
        if mood in emotional_moods:
            return "mood-first"

        return "genre-first"

    def _pick_alternative_strategy(self, current: str) -> str:
        """Pick a different strategy for refinement."""
        alternatives = [k for k in STRATEGIES if k != current]
        return alternatives[0] if alternatives else current


# ---------------------------------------------------------------------------
# Formatted Output
# ---------------------------------------------------------------------------

def format_agent_results(result: Dict) -> str:
    """Format the full agent output for terminal display."""
    lines = []

    lines.append(f"\n{'=' * 70}")
    lines.append(f"  MUSIC RECOMMENDER AGENT — FINAL RESULTS")
    lines.append(f"{'=' * 70}")

    prefs = result["user_prefs"]
    lines.append(f"  Profile: genre={prefs.get('genre')}, mood={prefs.get('mood')}, "
                 f"energy={prefs.get('energy')}, acoustic={prefs.get('likes_acoustic')}")
    lines.append(f"  Strategy: {result.get('strategy_used', '?')}")
    lines.append(f"  Refinements: {result.get('refinement_count', 0)}")
    lines.append("")

    # Recommendations table
    lines.append(f"  {'#':<3} {'Title':<28} {'Artist':<20} {'Score':<7} {'Conf':<6} {'Mood'}")
    lines.append(f"  {'-'*3} {'-'*28} {'-'*20} {'-'*7} {'-'*6} {'-'*12}")

    recs = result.get("recommendations", [])
    confs = result.get("confidence_scores", [])
    enhanced = result.get("enhanced_explanations", [])

    for i, (song, score, _) in enumerate(recs):
        conf = confs[i] if i < len(confs) else {}
        conf_val = conf.get("confidence", 0)
        conf_level = conf.get("level", "?")
        title = song["title"][:26]
        artist = song["artist"][:18]
        lines.append(f"  {i+1:<3} {title:<28} {artist:<20} {score:<7.2f} {conf_val:<6.2f} {song.get('mood', '?')}")

    # Enhanced explanations
    if enhanced:
        lines.append(f"\n  {'-' * 60}")
        lines.append(f"  RAG-ENHANCED EXPLANATIONS")
        lines.append(f"  {'-' * 60}")
        for i, explanation in enumerate(enhanced):
            song = recs[i][0] if i < len(recs) else {}
            lines.append(f"\n  {i+1}. {song.get('title', '?')} by {song.get('artist', '?')}")
            # Wrap explanation text
            words = explanation.split()
            current_line = "     "
            for word in words:
                if len(current_line) + len(word) + 1 > 70:
                    lines.append(current_line)
                    current_line = "     " + word
                else:
                    current_line += " " + word if current_line.strip() else "     " + word
            if current_line.strip():
                lines.append(current_line)

    # Summary
    overall = result.get("evaluation", {}).get("overall", {})
    bias_summary = result.get("bias_report", {}).get("summary", {})
    critique = result.get("critique", {})

    lines.append(f"\n  {'-' * 60}")
    lines.append(f"  SUMMARY")
    lines.append(f"  {'-' * 60}")
    lines.append(f"  Quality grade:    {overall.get('grade', '?')} ({overall.get('score', 0):.2f})")
    lines.append(f"  Biases detected:  {bias_summary.get('biases_detected', '?')}/4")
    lines.append(f"  Avg confidence:   {critique.get('avg_confidence', 0):.2f}")
    lines.append(f"  Assessment:       {critique.get('overall_assessment', 'N/A')}")

    log_file = result.get("log_file", "")
    if log_file:
        lines.append(f"  Log file:         {log_file}")

    lines.append(f"{'=' * 70}")

    return "\n".join(lines)
