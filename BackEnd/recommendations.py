import os
import requests
import pandas as pd
import numpy as np
from dotenv import load_dotenv
from sklearn.preprocessing import MultiLabelBinarizer, StandardScaler
from sklearn.metrics.pairwise import cosine_similarity

# === Load environment variable ===
load_dotenv()
TMDB_API_KEY = os.getenv("TMDB_API_KEY")

# === Fetch TMDB genre ID-to-name map ===
def get_genre_map():
    genre_url = f"https://api.themoviedb.org/3/genre/movie/list?api_key={TMDB_API_KEY}&language=en-US"
    response = requests.get(genre_url)
    data = response.json()
    return {genre["id"]: genre["name"] for genre in data.get("genres", [])}

# === Fetch movies from TMDB with genre names and release year ===
def fetch_tmdb_movies(pages=3):
    genre_map = get_genre_map()
    all_movies = []
    for page in range(1, pages + 1):
        url = f"https://api.themoviedb.org/3/discover/movie?api_key={TMDB_API_KEY}&language=en-US&sort_by=popularity.desc&include_adult=false&page={page}"
        response = requests.get(url)
        data = response.json()

        for item in data.get("results", []):
            genres = [genre_map.get(genre_id, "Unknown") for genre_id in item.get("genre_ids", [])]
            release_date = item.get("release_date", "2000-01-01")
            year = int(release_date.split("-")[0]) if release_date else 2000
            all_movies.append({
                'title': item.get('title', ''),
                'genres': genres,
                'tags': ["popular", "trending"],  # still placeholder
                'year': year,
                'duration': 100,  # placeholder; TMDB requires separate call to get actual runtime
                'actors': "Unknown"
            })
    return pd.DataFrame(all_movies)

# === Load dataset ===
movies_df = fetch_tmdb_movies()

# === Feature encoding ===
mlb_genres = MultiLabelBinarizer()
mlb_tags = MultiLabelBinarizer()

genre_encoded = mlb_genres.fit_transform(movies_df['genres'])
tag_encoded = mlb_tags.fit_transform(movies_df['tags'])

scaled_numeric = StandardScaler().fit_transform(movies_df[['year', 'duration']])
movie_vectors = np.hstack([genre_encoded, tag_encoded, scaled_numeric])

# === Search Function ===
def recommend_by_search(query: str, top_n=10):
    query_lower = query.lower()
    matched = movies_df[
        movies_df['title'].str.lower().str.contains(query_lower) |
        movies_df['genres'].astype(str).str.lower().str.contains(query_lower) |
        movies_df['actors'].str.lower().str.contains(query_lower)
    ]
    if matched.empty:
        print(f"❌ No movies found matching '{query}'")
        return pd.DataFrame(columns=['title', 'genres', 'year', 'duration'])

    return matched[['title', 'genres', 'year', 'duration']].head(top_n)

# === Preference-Based Recommendation Function ===
def recommend_movies(user_preferences: dict, top_n=10):
    liked_titles = user_preferences.get('liked_movies', [])
    preferred_genres = user_preferences.get('genres', [])
    preferred_tags = user_preferences.get('tags', [])
    year_min, year_max = user_preferences.get('year_range', (2000, 2024))
    preferred_duration = user_preferences.get('duration', 100)

    liked_movies = movies_df[movies_df['title'].isin(liked_titles)]
    if liked_movies.empty:
        print("⚠️ No liked movies matched the dataset.")
        return pd.DataFrame(columns=['title', 'genres', 'year', 'duration'])

    liked_indices = liked_movies.index.tolist()
    liked_vectors = movie_vectors[liked_indices]

    user_genres = mlb_genres.transform([preferred_genres])
    user_tags = mlb_tags.transform([preferred_tags])
    user_numeric = StandardScaler().fit_transform([[np.mean([year_min, year_max]), preferred_duration]])

    liked_vector_avg = np.mean(liked_vectors, axis=0).reshape(1, -1)
    user_vector = np.hstack([user_genres, user_tags, user_numeric])

    if liked_vector_avg.shape[1] < user_vector.shape[1]:
        liked_vector_avg = np.hstack([liked_vector_avg, np.zeros((1, user_vector.shape[1] - liked_vector_avg.shape[1]))])
    elif user_vector.shape[1] < liked_vector_avg.shape[1]:
        user_vector = np.hstack([user_vector, np.zeros((1, liked_vector_avg.shape[1] - user_vector.shape[1]))])

    final_user_vector = (liked_vector_avg + user_vector) / 2

    valid_indices = movies_df[(movies_df['year'] >= year_min) & (movies_df['year'] <= year_max)].index.tolist()
    valid_vectors = movie_vectors[valid_indices]

    similarities = cosine_similarity(final_user_vector, valid_vectors)[0]

    top_relative_indices = similarities.argsort()[-top_n:][::-1]
    top_absolute_indices = [valid_indices[i] for i in top_relative_indices]

    final_recommendations = movies_df.iloc[top_absolute_indices]
    final_recommendations = final_recommendations[~final_recommendations['title'].isin(liked_titles)]

    return final_recommendations[['title', 'genres', 'year', 'duration']]

# === Decision wrapper ===
def handle_user_input(user_input: dict, top_n=10):
    search_query = user_input.get("search_query", "").strip()
    if search_query:
        return recommend_by_search(search_query, top_n=top_n)
    elif user_input.get("genres") or user_input.get("year_range"):
        return recommend_movies(user_input, top_n=top_n)
    else:
        print("⚠️ No valid input provided.")
        return pd.DataFrame(columns=['title', 'genres', 'year', 'duration'])

if __name__ == "__main__":
    print("🔍 Search Test:")
    user_input = {"search_query": "Logan"}
    print(handle_user_input(user_input))

    print("\n🎯 Preference Test:")
    user_input = {
        "liked_movies": ["Logan"],
        "genres": ["Action"],
        "tags": ["popular"],
        "year_range": (1990, 2025),
        "duration": 120
    }
    print(handle_user_input(user_input))

