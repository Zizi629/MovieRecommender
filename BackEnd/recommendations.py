from dotenv import load_dotenv
import requests
import pandas as pd
import numpy as np
import os
from sklearn.preprocessing import MultiLabelBinarizer, MinMaxScaler
from sklearn.metrics.pairwise import cosine_similarity

# ===========================
# SETUP: Your TMDB API Key
# ===========================
load_dotenv()
TMDB_API_KEY = os.environ.get("TMDB_API_KEY")  # Replace with your TMDB API key

# ===========================
# FUNCTION 1: Fetch Movie Data from TMDB
# ===========================
def fetch_movies_from_tmdb(pages=2):
    """
    Fetch popular movies from TMDB.
    Each movie includes title, genres, tags (keywords), release year, and duration.
    """
    movies = []

    for page in range(1, pages + 1):
        url = f"https://api.themoviedb.org/3/movie/popular?api_key={TMDB_API_KEY}&language=en-US&page={page}"
        response = requests.get(url)
        data = response.json()

        for movie in data['results']:
            movies.append({
                'title': movie['title'],
                'genres': [g['name'] for g in get_genres_by_ids(movie['genre_ids'])],
                'tags': get_keywords(movie['id']),
                'year': int(movie['release_date'].split('-')[0]) if movie.get('release_date') else 2000,
                'duration': get_runtime(movie['id'])
            })

    return pd.DataFrame(movies)

# ===========================
# FUNCTION 2: Convert Genre IDs to Names
# ===========================
def get_genres_by_ids(ids):
    genre_map = {
        28: 'Action', 12: 'Adventure', 16: 'Animation', 35: 'Comedy', 80: 'Crime',
        99: 'Documentary', 18: 'Drama', 10751: 'Family', 14: 'Fantasy', 36: 'History',
        27: 'Horror', 10402: 'Music', 9648: 'Mystery', 10749: 'Romance',
        878: 'Sci-Fi', 10770: 'TV Movie', 53: 'Thriller', 10752: 'War', 37: 'Western'
    }
    return [{'id': g, 'name': genre_map.get(g, 'Unknown')} for g in ids]

# ===========================
# FUNCTION 3: Get Movie Runtime from TMDB
# ===========================
def get_runtime(movie_id):
    url = f"https://api.themoviedb.org/3/movie/{movie_id}?api_key={TMDB_API_KEY}"
    response = requests.get(url).json()
    return response.get('runtime', 100)

# ===========================
# FUNCTION 4: Get Movie Keywords (Tags)
# ===========================
def get_keywords(movie_id):
    url = f"https://api.themoviedb.org/3/movie/{movie_id}/keywords?api_key={TMDB_API_KEY}"
    response = requests.get(url).json()
    return [kw['name'] for kw in response.get('keywords', [])]

# ===========================
# LOAD & PROCESS MOVIE DATA
# ===========================
movies_df = fetch_movies_from_tmdb(pages=5)

# Convert genre and tags into list format if needed
movies_df['genres'] = movies_df['genres'].apply(lambda x: x if isinstance(x, list) else [])
movies_df['tags'] = movies_df['tags'].apply(lambda x: x if isinstance(x, list) else [])

# One-hot encode genres and tags
mlb_genres = MultiLabelBinarizer()
mlb_tags = MultiLabelBinarizer()

genres_encoded = mlb_genres.fit_transform(movies_df['genres'])
tags_encoded = mlb_tags.fit_transform(movies_df['tags'])

# Normalize year and duration
scaler = MinMaxScaler()
numerical_features = scaler.fit_transform(movies_df[['year', 'duration']])

# Combine all movie features into one matrix
movie_vectors = np.hstack([genres_encoded, tags_encoded, numerical_features])

# ===========================
# FUNCTION 5: Recommend Movies
# ===========================
def recommend_movies(user_preferences: dict, top_n=10):
    """
    Recommend top N movies based on:
    - Liked movies
    - Preferred genres and tags
    - Year range
    - Duration
    
    Parameters:
    - user_preferences (dict):
        - 'liked_movies': list of liked movie titles (must be in dataset)
        - 'genres': list of preferred genres (optional)
        - 'tags': list of preferred tags (optional)
        - 'year_range': (min_year, max_year) tuple
        - 'duration': preferred movie duration (int)
    - top_n: number of movies to recommend

    Returns:
    - DataFrame with top recommended movies
    """

    # Extract user preferences
    liked_titles = user_preferences.get('liked_movies', [])
    preferred_genres = user_preferences.get('genres', [])
    preferred_tags = user_preferences.get('tags', [])
    year_min, year_max = user_preferences.get('year_range', (2000, 2024))
    preferred_duration = user_preferences.get('duration', 100)

    # Step 1: Get liked movie vectors
    liked_movies = movies_df[movies_df['title'].isin(liked_titles)]
    if liked_movies.empty:
        print("⚠️ No liked movies matched the dataset.")
        return pd.DataFrame(columns=['title', 'genres', 'year', 'duration'])

    liked_indices = liked_movies.index.tolist()
    liked_vectors = movie_vectors[liked_indices]

    # Step 2: Encode genres and tags from user input
    user_genres = mlb_genres.transform([preferred_genres])
    user_tags = mlb_tags.transform([preferred_tags])
    user_numeric = scaler.transform([[np.mean([year_min, year_max]), preferred_duration]])

    # Step 3: Combine liked movies' vector and user preferences into one profile
    liked_vector_avg = np.mean(liked_vectors, axis=0).reshape(1, -1)
    user_vector = np.hstack([user_genres, user_tags, user_numeric])

    # Make sure vectors are same shape (pad if needed)
    if liked_vector_avg.shape[1] < user_vector.shape[1]:
        padding = np.zeros((1, user_vector.shape[1] - liked_vector_avg.shape[1]))
        liked_vector_avg = np.hstack([liked_vector_avg, padding])
    elif user_vector.shape[1] < liked_vector_avg.shape[1]:
        padding = np.zeros((1, liked_vector_avg.shape[1] - user_vector.shape[1]))
        user_vector = np.hstack([user_vector, padding])

    # Combine both (50% weight each)
    final_user_vector = (liked_vector_avg + user_vector) / 2

    # Step 4: Filter movies in the selected year range
    valid_indices = movies_df[
        (movies_df['year'] >= year_min) & (movies_df['year'] <= year_max)
    ].index.tolist()
    valid_vectors = movie_vectors[valid_indices]

    # Step 5: Compute cosine similarity
    similarities = cosine_similarity(final_user_vector, valid_vectors)[0]

    # Step 6: Get top N recommendations
    top_relative_indices = similarities.argsort()[-top_n:][::-1]
    top_absolute_indices = [valid_indices[i] for i in top_relative_indices]

    # Exclude movies the user already liked
    final_recommendations = movies_df.iloc[top_absolute_indices]
    final_recommendations = final_recommendations[~final_recommendations['title'].isin(liked_titles)]

    return final_recommendations[['title', 'genres', 'year', 'duration']]
user_input = {
    'liked_movies': ['Laila', 'Bad Influence'],
    'genres': ['Romance', 'Drama'],
    'tags': ['love', 'heartbreak'],
    'year_range': (2005, 2023),
    'duration': 120
}

recommended = recommend_movies(user_input, top_n=10)

print("\n🎬 Recommended Movies Based on Your Preferences:\n")
print(recommended)
