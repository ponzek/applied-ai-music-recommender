"""Download and sample from the Spotify Tracks Dataset (Kaggle).

Pulls diverse songs, maps Spotify genres to our knowledge base genres,
assigns moods based on valence/energy, and exports to songs.csv.
"""

import csv
import os
import random

# HuggingFace
csv_path = None

try:
    import kagglehub
    print("Downloading dataset via kagglehub...")
    path = kagglehub.dataset_download("maharshipandya/spotify-tracks-dataset")
    print(f"Dataset downloaded to: {path}")
    for root, dirs, files in os.walk(path):
        for f in files:
            if f.endswith(".csv"):
                csv_path = os.path.join(root, f)
                break
except Exception as e:
    print(f"kagglehub failed: {e}")
    print("Trying HuggingFace...")

if not csv_path:
    try:
        from datasets import load_dataset
        print("Downloading from HuggingFace...")
        ds = load_dataset("maharshipandya/spotify-tracks-dataset", split="train")
        # Save to a temp CSV
        temp_csv = os.path.join(os.path.dirname(os.path.abspath(__file__)), "_temp_spotify.csv")
        ds.to_csv(temp_csv)
        csv_path = temp_csv
        print(f"Downloaded {len(ds)} tracks")
    except Exception as e2:
        print(f"HuggingFace also failed: {e2}")
        exit(1)

print(f"Reading: {csv_path}")

# ---------------------------------------------------------------------------
# Genre mapping: Spotify dataset genre -> our genre
# ---------------------------------------------------------------------------
GENRE_MAP = {
    "pop": "pop",
    "power-pop": "pop",
    "synth-pop": "pop",
    "dance": "pop",
    "rock": "rock",
    "hard-rock": "rock",
    "punk-rock": "rock",
    "alt-rock": "alt-rock",
    "alternative": "alt-rock",
    "emo": "alt-rock",
    "grunge": "alt-rock",
    "ambient": "ambient",
    "new-age": "ambient",
    "chill": "lofi",
    "study": "lofi",
    "jazz": "jazz",
    "soul": "r&b",
    "r-n-b": "r&b",
    "reggaeton": "reggaeton",
    "latin": "reggaeton",
    "hip-hop": "hip-hop",
    "rap": "hip-hop",
    "indie": "indie pop",
    "indie-pop": "indie pop",
    "synthwave": "synthwave",
    "electronic": "synthwave",
    "progressive-rock": "prog-rock",
    # Regional Mexican
    "regional-mexican": "corridos",
    "mexican-regional": "corridos",
    "salsa": "reggaeton",
}

# ---------------------------------------------------------------------------
# Mood assignment: valence + energy -> mood
# ---------------------------------------------------------------------------
def assign_mood(valence, energy, acousticness):
    """Map Spotify audio features to our mood tags."""
    if valence >= 0.7 and energy >= 0.7:
        return "hype"
    elif valence >= 0.7 and energy >= 0.5:
        return "happy"
    elif valence >= 0.6 and energy < 0.5:
        return "romantic"
    elif valence >= 0.5 and energy >= 0.6:
        return "energetic"
    elif valence >= 0.5 and energy < 0.4:
        return "relaxed"
    elif valence >= 0.4 and energy < 0.5:
        return "chill"
    elif valence >= 0.3 and energy >= 0.7:
        return "intense"
    elif valence < 0.3 and energy >= 0.6:
        return "dark"
    elif valence < 0.3 and energy < 0.4:
        return "melancholic"
    elif valence < 0.25:
        return "heartbreak"
    elif acousticness > 0.7:
        return "nostalgic"
    elif energy >= 0.8:
        return "confident"
    elif energy < 0.3:
        return "focused"
    else:
        return "moody"


def estimate_decade(popularity):
    """Rough decade estimate (real dataset doesn't have release year)."""
    # Higher popularity tracks tend to be more recent
    if popularity >= 70:
        return "2020s"
    elif popularity >= 50:
        return "2010s"
    elif popularity >= 30:
        return "2000s"
    else:
        return "1990s"


# ---------------------------------------------------------------------------
# Read and sample
# ---------------------------------------------------------------------------
print("Parsing tracks...")

# Collect tracks by our genre
genre_buckets = {g: [] for g in set(GENRE_MAP.values())}

with open(csv_path, "r", encoding="utf-8") as f:
    reader = csv.DictReader(f)
    
    for row in reader:
        spotify_genre = row.get("track_genre", "").strip().lower()

        # -- FILTER 1: Map Spotify genre to our genre (two-tier) --
        # Tier 1: Exact match from our genre map
        our_genre = GENRE_MAP.get(spotify_genre)

        # Tier 2: Fuzzy keyword match — if no exact match,
        # check if the Spotify genre *contains* one of our genre keywords.
        # Example: "punk-rock" contains "rock" → maps to "rock"
        #          "deep-house" contains no keyword → skipped
        if not our_genre:
            FUZZY_KEYWORDS = {
                "rock": "rock",
                "punk": "rock",
                "metal": "rock",
                "pop": "pop",
                "jazz": "jazz",
                "blues": "r&b",
                "soul": "r&b",
                "ambient": "ambient",
                "lofi": "lofi",
                "lo-fi": "lofi",
                "hip": "hip-hop",
                "rap": "hip-hop",
                "trap": "hip-hop",
                "indie": "indie pop",
                "synth": "synthwave",
                "electro": "synthwave",
                "latin": "reggaeton",
                "reggae": "reggaeton",
                "salsa": "reggaeton",
                "prog": "prog-rock",
            }
            for keyword, mapped_genre in FUZZY_KEYWORDS.items():
                if keyword in spotify_genre:
                    our_genre = mapped_genre
                    break

        # If neither tier matched, skip this song
        if not our_genre:
            continue
        
        # -- FILTER 2: Convert audio features from strings to numbers --
        # The CSV stores everything as text. We need floats/ints for scoring.
        # If any value can't be converted (corrupted data), skip the row.
        try:
            energy = float(row.get("energy", 0))
            valence = float(row.get("valence", 0))
            danceability = float(row.get("danceability", 0))
            acousticness = float(row.get("acousticness", 0))
            popularity = int(row.get("popularity", 0))
            tempo = float(row.get("tempo", 120))
        except (ValueError, TypeError):
            continue

        # -- FILTER 3: Must have a title and artist --
        title = row.get("track_name", "").strip()
        artist = row.get("artists", "").strip().split(";")[0]  # Take first artist if multiple
        
        if not title or not artist:
            continue

        # -- FILTER 4: Skip very unpopular tracks --
        # Popularity < 15 is usually noise (test uploads, duplicates, etc.)
        if popularity < 15:
            continue

        # -- BUILD: Create our song record with derived fields --
        track = {
            "title": title,
            "artist": artist,
            "genre": our_genre,                                    # Mapped from Spotify genre
            "mood": assign_mood(valence, energy, acousticness),    # Rule-based: valence + energy -> mood
            "energy": round(energy, 2),
            "tempo_bpm": int(round(tempo)),
            "valence": round(valence, 2),
            "danceability": round(danceability, 2),
            "acousticness": round(acousticness, 2),
            "popularity": popularity,
            "release_decade": estimate_decade(popularity),         # Estimated since dataset lacks release year
            "language": "es" if our_genre in ("reggaeton", "corridos") else "en",
        }
        
        genre_buckets[our_genre].append(track)

# Print counts per genre
print("\nTracks found per genre:")
for genre, tracks in sorted(genre_buckets.items()):
    print(f"  {genre}: {len(tracks)}")

# ---------------------------------------------------------------------------
# Sample: aim for ~100 songs, balanced across genres
# ---------------------------------------------------------------------------

# Target distribution (more from popular genres, fewer from niche)
TARGET = {
    "pop": 12,
    "rock": 10,
    "alt-rock": 10,
    "lofi": 8,
    "ambient": 6,
    "jazz": 6,
    "r&b": 8,
    "hip-hop": 10,
    "reggaeton": 8,
    "indie pop": 8,
    "synthwave": 6,
    "prog-rock": 5,
    "corridos": 5,
}

# Manual entries for genres Spotify doesn't tag
MANUAL_SONGS = [
    {"title": "Make Like a Tree (Get Out)", "artist": "Thank You Scientist", "genre": "prog-rock",
     "mood": "energetic", "energy": 0.85, "tempo_bpm": 160, "valence": 0.65,
     "danceability": 0.45, "acousticness": 0.15, "popularity": 42, "release_decade": "2010s", "language": "en"},
    {"title": "Blood Sugar", "artist": "Thank You Scientist", "genre": "prog-rock",
     "mood": "intense", "energy": 0.88, "tempo_bpm": 170, "valence": 0.55,
     "danceability": 0.50, "acousticness": 0.10, "popularity": 38, "release_decade": "2010s", "language": "en"},
    {"title": "Chromology", "artist": "Haken", "genre": "prog-rock",
     "mood": "moody", "energy": 0.72, "tempo_bpm": 145, "valence": 0.42,
     "danceability": 0.38, "acousticness": 0.20, "popularity": 35, "release_decade": "2010s", "language": "en"},
    {"title": "Cockroach King", "artist": "Haken", "genre": "prog-rock",
     "mood": "energetic", "energy": 0.82, "tempo_bpm": 155, "valence": 0.60,
     "danceability": 0.42, "acousticness": 0.18, "popularity": 40, "release_decade": "2010s", "language": "en"},
    {"title": "Mr. Invisible", "artist": "Thank You Scientist", "genre": "prog-rock",
     "mood": "happy", "energy": 0.80, "tempo_bpm": 150, "valence": 0.70,
     "danceability": 0.55, "acousticness": 0.12, "popularity": 36, "release_decade": "2010s", "language": "en"},
    {"title": "TUQLO", "artist": "Fuerza Regida", "genre": "corridos",
     "mood": "confident", "energy": 0.82, "tempo_bpm": 135, "valence": 0.68,
     "danceability": 0.75, "acousticness": 0.12, "popularity": 85, "release_decade": "2020s", "language": "es"},
    {"title": "Bebe Dame", "artist": "Fuerza Regida", "genre": "corridos",
     "mood": "hype", "energy": 0.88, "tempo_bpm": 128, "valence": 0.78,
     "danceability": 0.80, "acousticness": 0.08, "popularity": 90, "release_decade": "2020s", "language": "es"},
    {"title": "Sabor Fresa", "artist": "Peso Pluma", "genre": "corridos",
     "mood": "intense", "energy": 0.80, "tempo_bpm": 130, "valence": 0.60,
     "danceability": 0.72, "acousticness": 0.15, "popularity": 82, "release_decade": "2020s", "language": "es"},
    {"title": "Ella Baila Sola", "artist": "Eslabon Armado & Peso Pluma", "genre": "corridos",
     "mood": "romantic", "energy": 0.65, "tempo_bpm": 118, "valence": 0.55,
     "danceability": 0.68, "acousticness": 0.35, "popularity": 92, "release_decade": "2020s", "language": "es"},
    {"title": "Ch y la Pizza", "artist": "Fuerza Regida", "genre": "corridos",
     "mood": "hype", "energy": 0.85, "tempo_bpm": 132, "valence": 0.72,
     "danceability": 0.78, "acousticness": 0.10, "popularity": 88, "release_decade": "2020s", "language": "es"},
    # Add back key songs from the original catalog
    {"title": "Fainting Spells", "artist": "AFI", "genre": "alt-rock",
     "mood": "melancholic", "energy": 0.65, "tempo_bpm": 110, "valence": 0.30,
     "danceability": 0.42, "acousticness": 0.25, "popularity": 55, "release_decade": "2000s", "language": "en"},
    {"title": "Cure My Tragedy", "artist": "Cold", "genre": "alt-rock",
     "mood": "melancholic", "energy": 0.60, "tempo_bpm": 100, "valence": 0.28,
     "danceability": 0.42, "acousticness": 0.30, "popularity": 50, "release_decade": "2000s", "language": "en"},
    {"title": "Bichiyal", "artist": "Bad Bunny & Yaviah", "genre": "reggaeton",
     "mood": "hype", "energy": 0.90, "tempo_bpm": 92, "valence": 0.82,
     "danceability": 0.83, "acousticness": 0.05, "popularity": 78, "release_decade": "2020s", "language": "es"},
    {"title": "Ex-Factor", "artist": "Lauryn Hill", "genre": "r&b",
     "mood": "heartbreak", "energy": 0.45, "tempo_bpm": 76, "valence": 0.20,
     "danceability": 0.58, "acousticness": 0.65, "popularity": 88, "release_decade": "1990s", "language": "en"},
    {"title": "Human Nature", "artist": "Michael Jackson", "genre": "pop",
     "mood": "romantic", "energy": 0.50, "tempo_bpm": 93, "valence": 0.62,
     "danceability": 0.70, "acousticness": 0.32, "popularity": 85, "release_decade": "1980s", "language": "en"},
]

random.seed(42)  # Reproducible

# Use ALL mapped songs (no sampling — take everything)
sampled = []
seen = set()

for genre, bucket in genre_buckets.items():
    for track in bucket:
        key = (track["title"].lower(), track["artist"].lower())
        if key not in seen:
            seen.add(key)
            sampled.append(track)

# Add manual songs (prog-rock, corridos, originals)
for manual in MANUAL_SONGS:
    key = (manual["title"].lower(), manual["artist"].lower())
    if key not in seen:
        sampled.append(manual)
        seen.add(key)

# Shuffle for variety, then assign IDs
random.shuffle(sampled)
for i, track in enumerate(sampled, 1):
    track["id"] = i

print(f"\nTotal: {len(sampled)} tracks (incl. {len(MANUAL_SONGS)} manual entries)")

# ---------------------------------------------------------------------------
# Export
# ---------------------------------------------------------------------------
output_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data", "songs.csv")
output_path = os.path.normpath(output_path)

fieldnames = ["id", "title", "artist", "genre", "mood", "energy", "tempo_bpm",
              "valence", "danceability", "acousticness", "popularity", "release_decade", "language"]

with open(output_path, "w", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(f, fieldnames=fieldnames)
    writer.writeheader()
    for track in sampled:
        writer.writerow(track)

print(f"Wrote {len(sampled)} songs to {output_path}")
print("\nGenre distribution:")
from collections import Counter
genres = Counter(t["genre"] for t in sampled)
for g, c in genres.most_common():
    print(f"  {g}: {c}")

moods = Counter(t["mood"] for t in sampled)
print("\nMood distribution:")
for m, c in moods.most_common():
    print(f"  {m}: {c}")
