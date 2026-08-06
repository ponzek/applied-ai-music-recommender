"""Grover's Algorithm Simulation -- Quantum-inspired search for music recommendations.

Demonstrates the conceptual speedup of Grover's algorithm applied to
music catalog search. While this runs on a classical computer (simulated
quantum), it shows the theoretical advantage:

  Classical search: O(N) operations to find the best match
  Grover's search:  O(sqrt(N)) operations to find the best match

For our catalog of N songs:
  - Classical: scans all N songs
  - Quantum:   ~sqrt(N) Grover iterations

Usage:
  python src/quantum_search.py              # Run comparison
  python src/quantum_search.py --visualize  # Show amplitude evolution
"""

import math
import random
import time
import argparse

from src.recommender import load_songs


# ---------------------------------------------------------------------------
# Classical Search (baseline)
# ---------------------------------------------------------------------------

def classical_search(songs: list, user_prefs: dict) -> tuple:
    """Linear scan through all songs, counting operations.
    
    Returns: (best_song, best_score, operations_count)
    """
    best_song = None
    best_score = -1
    operations = 0

    for song in songs:
        # Score calculation = 1 operation
        score = 0.0
        if song.get("genre") == user_prefs.get("genre"):
            score += 2.0
        if song.get("mood") == user_prefs.get("mood"):
            score += 3.0
        energy_diff = abs(song.get("energy", 0.5) - user_prefs.get("energy", 0.5))
        score += (1.0 - energy_diff) * 1.5
        operations += 1

        if score > best_score:
            best_score = score
            best_song = song

    return best_song, best_score, operations


# ---------------------------------------------------------------------------
# Grover's Algorithm Simulation
# ---------------------------------------------------------------------------

class GroverSimulator:
    """Simulates Grover's quantum search algorithm on a classical machine.
    
    This is NOT a real quantum computer -- it's a simulation that demonstrates
    the *mathematical structure* of Grover's algorithm:
    
    1. Initialize all N items in equal superposition (amplitude = 1/sqrt(N))
    2. Repeat ~sqrt(N) times:
       a. Oracle: flip the amplitude of the "marked" (best match) item
       b. Diffusion: reflect all amplitudes about their mean
    3. Measure: the marked item has the highest probability
    
    The key insight: after O(sqrt(N)) iterations, the probability of measuring
    the correct item approaches 1, compared to O(N) classical operations.
    """

    def __init__(self, songs: list, user_prefs: dict):
        self.songs = songs
        self.user_prefs = user_prefs
        self.n = len(songs)
        self.num_qubits = math.ceil(math.log2(self.n)) if self.n > 0 else 0
        
        # Pre-score all songs to identify the "marked" item
        self.scores = []
        for song in songs:
            score = 0.0
            if song.get("genre") == user_prefs.get("genre"):
                score += 2.0
            if song.get("mood") == user_prefs.get("mood"):
                score += 3.0
            energy_diff = abs(song.get("energy", 0.5) - user_prefs.get("energy", 0.5))
            score += (1.0 - energy_diff) * 1.5
            self.scores.append(score)
        
        # The "marked" item is the highest-scoring song
        self.marked_idx = self.scores.index(max(self.scores))
        
    def simulate(self, verbose: bool = False) -> dict:
        """Run the Grover simulation.
        
        Returns dict with:
            - found_song: the song found by Grover's
            - iterations: number of Grover iterations (oracle + diffusion calls)
            - classical_ops: number of classical operations for comparison
            - speedup: classical_ops / iterations
            - amplitude_history: amplitude of marked item per iteration
        """
        if self.n == 0:
            return {"error": "Empty catalog"}
        
        # Number of Grover iterations: floor(pi/4 * sqrt(N))
        optimal_iterations = max(1, int(math.floor(math.pi / 4 * math.sqrt(self.n))))
        
        # Initialize amplitudes: equal superposition
        amplitudes = [1.0 / math.sqrt(self.n)] * self.n
        
        amplitude_history = []
        
        if verbose:
            print(f"\n  Grover's Algorithm Simulation")
            print(f"  Catalog size (N): {self.n}")
            print(f"  Qubits needed: {self.num_qubits}")
            print(f"  Optimal iterations: {optimal_iterations}")
            print(f"  Classical operations: {self.n}")
            print(f"  Theoretical speedup: {self.n / optimal_iterations:.1f}x")
            print()
        
        for iteration in range(optimal_iterations):
            # Step 1: Oracle -- flip amplitude of marked item
            amplitudes[self.marked_idx] *= -1
            
            # Step 2: Diffusion -- reflect about mean
            mean = sum(amplitudes) / self.n
            amplitudes = [2 * mean - a for a in amplitudes]
            
            # Record probability of marked item
            prob = amplitudes[self.marked_idx] ** 2
            amplitude_history.append(prob)
            
            if verbose:
                print(f"  Iteration {iteration + 1}/{optimal_iterations}: "
                      f"P(target) = {prob:.4f}")
        
        # "Measurement" -- the marked item should have highest probability
        probabilities = [a ** 2 for a in amplitudes]
        measured_idx = probabilities.index(max(probabilities))
        found_correct = (measured_idx == self.marked_idx)
        
        found_song = self.songs[measured_idx]
        
        result = {
            "found_song": found_song,
            "found_score": self.scores[measured_idx],
            "correct": found_correct,
            "iterations": optimal_iterations,
            "classical_ops": self.n,
            "speedup": self.n / optimal_iterations,
            "final_probability": probabilities[self.marked_idx],
            "qubits_needed": self.num_qubits,
            "amplitude_history": amplitude_history,
        }
        
        return result


# ---------------------------------------------------------------------------
# Comparison
# ---------------------------------------------------------------------------

def run_comparison(songs: list, user_prefs: dict, verbose: bool = True) -> dict:
    """Run both classical and Grover's search, compare results."""
    
    print(f"\n{'=' * 70}")
    print(f"  QUANTUM vs CLASSICAL SEARCH COMPARISON")
    print(f"  Catalog: {len(songs)} songs")
    print(f"  Query: genre={user_prefs['genre']}, mood={user_prefs['mood']}, "
          f"energy={user_prefs['energy']}")
    print(f"{'=' * 70}")
    
    # Classical search
    print(f"\n  --- CLASSICAL (Linear Scan) ---")
    start = time.perf_counter()
    classical_song, classical_score, classical_ops = classical_search(songs, user_prefs)
    classical_time = time.perf_counter() - start
    print(f"  Found: {classical_song['title']} by {classical_song['artist']}")
    print(f"  Score: {classical_score:.2f}")
    print(f"  Operations: {classical_ops}")
    print(f"  Time: {classical_time * 1000:.2f}ms")
    
    # Grover's simulation
    print(f"\n  --- GROVER'S (Quantum Simulation) ---")
    start = time.perf_counter()
    grover = GroverSimulator(songs, user_prefs)
    result = grover.simulate(verbose=verbose)
    grover_time = time.perf_counter() - start
    print(f"\n  Found: {result['found_song']['title']} by {result['found_song']['artist']}")
    print(f"  Score: {result['found_score']:.2f}")
    print(f"  Iterations: {result['iterations']}")
    print(f"  Qubits: {result['qubits_needed']}")
    print(f"  Final P(target): {result['final_probability']:.4f}")
    print(f"  Correct: {'YES' if result['correct'] else 'NO'}")
    print(f"  Simulation time: {grover_time * 1000:.2f}ms")
    
    # Comparison
    print(f"\n  --- SPEEDUP ANALYSIS ---")
    print(f"  Classical operations: {classical_ops}")
    print(f"  Grover iterations:   {result['iterations']}")
    print(f"  Theoretical speedup: {result['speedup']:.1f}x")
    print(f"  Same result found:   {'YES' if classical_song['title'] == result['found_song']['title'] else 'NO'}")
    
    # Scale projection
    print(f"\n  --- SCALE PROJECTIONS ---")
    for scale in [1000, 10000, 100000, 1000000]:
        grover_ops = int(math.pi / 4 * math.sqrt(scale))
        print(f"  N={scale:>10,}: Classical={scale:>10,} ops | Grover={grover_ops:>6,} ops | Speedup={scale/grover_ops:.0f}x")
    
    print(f"\n{'=' * 70}")
    
    return {
        "classical": {"song": classical_song, "score": classical_score, "ops": classical_ops},
        "grover": result,
    }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Grover's Algorithm -- Quantum Search Simulation")
    parser.add_argument("--verbose", action="store_true", help="Show iteration details")
    args = parser.parse_args()
    
    songs = load_songs("data/songs.csv")
    
    profiles = [
        {"name": "Melancholic Alt-Rock", "genre": "alt-rock", "mood": "melancholic", "energy": 0.65, "likes_acoustic": True},
        {"name": "Chill Lofi", "genre": "lofi", "mood": "chill", "energy": 0.35, "likes_acoustic": True},
        {"name": "Hype Reggaeton", "genre": "reggaeton", "mood": "hype", "energy": 0.90, "likes_acoustic": False},
    ]
    
    for profile in profiles:
        name = profile.pop("name")
        print(f"\n  PROFILE: {name}")
        run_comparison(songs, profile, verbose=args.verbose)


if __name__ == "__main__":
    main()
