# ─────────────────────────────────────────────────────────────────────────────
#  Movie Recommender App  —  Real MovieLens Dataset Edition
#
#  HOW TO SET UP:
#  1. Download MovieLens Small dataset:
#     https://grouplens.org/datasets/movielens/latest/
#     → Download "ml-latest-small.zip" (~1MB)
#  2. Unzip it — you'll get a folder called "ml-latest-small/"
#  3. Place this script in the SAME folder as ml-latest-small/
#  4. Run:  streamlit run movie_recommender_app.py
#
#  REQUIRED FILES (inside ml-latest-small/):
#     ratings.csv  →  userId, movieId, rating, timestamp
#     movies.csv   →  movieId, title, genres
# ─────────────────────────────────────────────────────────────────────────────

import streamlit as st
import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import os

# ── CONSTANTS ─────────────────────────────────────────────────────────────────
DATASET_PATH   = "ml-latest-small"          # Folder with ratings.csv & movies.csv
MIN_USER_RATINGS = 50     # Only keep users who rated at least this many movies
MIN_MOVIE_RATINGS = 30    # Only keep movies that were rated by at least this many users
TOP_MOVIES_LIMIT  = 500   # Work with the top N most-rated movies (keeps UI fast)
SAMPLE_USERS      = 200   # Max number of users to include in similarity matrix


# ── 1. DATA LOADING ───────────────────────────────────────────────────────────

@st.cache_data
def load_data():
    """
    Load and preprocess the MovieLens dataset.

    WHY FILTERING MATTERS:
    Real data is extremely sparse — most users rate only a tiny fraction of all
    movies. If we keep all ~9,700 movies and ~610 users, two users who happen to
    rate completely different movies will have cosine similarity = 0, making the
    recommendations useless. By keeping only:
      - Active users  (rated many movies → richer profile)
      - Popular movies (rated by many users → more overlap between users)
    …we get a denser matrix where similarity scores are actually meaningful.
    """

    ratings_path = os.path.join(DATASET_PATH, "ratings.csv")
    movies_path  = os.path.join(DATASET_PATH, "movies.csv")

    if not os.path.exists(ratings_path) or not os.path.exists(movies_path):
        return None, None, None      # Signal that data is missing

    ratings = pd.read_csv(ratings_path)   # userId | movieId | rating | timestamp
    movies  = pd.read_csv(movies_path)    # movieId | title | genres

    # ── Step 1: Filter to popular movies ──────────────────────────────────────
    # Count how many users rated each movie; keep only well-known ones
    movie_rating_counts = ratings.groupby("movieId")["rating"].count()
    popular_movie_ids   = movie_rating_counts[movie_rating_counts >= MIN_MOVIE_RATINGS].index

    # Among those, keep only the TOP_MOVIES_LIMIT most-rated (manageable matrix size)
    top_movie_ids = (
        movie_rating_counts[popular_movie_ids]
        .nlargest(TOP_MOVIES_LIMIT)
        .index
    )
    ratings = ratings[ratings["movieId"].isin(top_movie_ids)]

    # ── Step 2: Filter to active users ────────────────────────────────────────
    # Keep users who have rated many movies (richer taste profiles)
    user_rating_counts = ratings.groupby("userId")["rating"].count()
    active_user_ids    = user_rating_counts[user_rating_counts >= MIN_USER_RATINGS].index

    # Sample to keep the matrix from being too large
    if len(active_user_ids) > SAMPLE_USERS:
        active_user_ids = active_user_ids[:SAMPLE_USERS]

    ratings = ratings[ratings["userId"].isin(active_user_ids)]

    # ── Step 3: Build the user-movie matrix ───────────────────────────────────
    # Rows = userId, Columns = movie title, Values = rating (0 = not watched)
    movie_id_to_title = movies.set_index("movieId")["title"]
    ratings["title"]  = ratings["movieId"].map(movie_id_to_title)

    matrix = ratings.pivot_table(
        index="userId", columns="title", values="rating"
    ).fillna(0)    # 0 = user hasn't rated this movie (treat as "not watched")

    # Build genre lookup (useful info to show in UI)
    movies["clean_title"] = movies["title"]
    genre_map = movies.set_index("title")["genres"].to_dict()

    return matrix, ratings, genre_map


# ── 2. RECOMMENDATION FUNCTION ────────────────────────────────────────────────

def recommend_movies(user_id, matrix, sim_df, top_n=5):
    """
    Collaborative filtering: predict ratings for unwatched movies.

    LOGIC (same as dummy version, just scaled up):
    For each movie the user hasn't rated (0 in matrix):
      predicted_score = Σ(similarity(user, other) × other's_rating)
                        ─────────────────────────────────────────────
                              Σ(similarity(user, other))

    Only other users who DID watch the movie contribute to the average.
    This gives a predicted score ∈ [0, 5].
    """

    sim_scores = sim_df[user_id].drop(user_id)      # similarity to everyone else
    user_row   = matrix.loc[user_id]
    unwatched  = user_row[user_row == 0].index.tolist()

    if not unwatched:
        return []

    # Vectorised prediction (much faster than a Python loop on large data)
    # Shape: (other_users,) dot (other_users × unwatched_movies) → (unwatched_movies,)
    other_ratings   = matrix.loc[sim_scores.index, unwatched]   # DataFrame
    sim_array       = sim_scores.values.reshape(1, -1)           # (1, n_users)

    # Weighted sum of ratings
    numerators   = sim_array @ other_ratings.values              # (1, n_unwatched)
    # Only sum similarities where the other user actually rated the movie
    rated_mask   = (other_ratings.values > 0).astype(float)      # (n_users, n_unwatched)
    denominators = sim_array @ rated_mask                        # (1, n_unwatched)

    # Avoid division by zero
    with np.errstate(invalid="ignore", divide="ignore"):
        predicted = np.where(denominators > 0, numerators / denominators, 0)

    scores = pd.Series(predicted[0], index=unwatched)
    scores = scores[scores > 0].sort_values(ascending=False)

    return list(scores.head(top_n).items())


# ── 3. STREAMLIT UI ───────────────────────────────────────────────────────────

st.set_page_config(page_title="Movie Recommender", page_icon="🎬", layout="wide")

st.title("🎬 Movie Recommender")
st.caption("Collaborative filtering on real MovieLens data — finds users with similar taste and recommends what they loved.")

# ── Load data ──────────────────────────────────────────────────────────────────
with st.spinner("Loading MovieLens dataset…"):
    matrix, ratings, genre_map = load_data()

if matrix is None:
    # ── Dataset not found → show setup instructions ────────────────────────────
    st.error("📂 MovieLens dataset not found!")
    st.markdown("""
    ### Quick Setup (takes ~2 minutes)

    **Step 1** — Download the dataset:
    👉 [https://grouplens.org/datasets/movielens/latest/](https://grouplens.org/datasets/movielens/latest/)
    → Click **"ml-latest-small.zip"** (about 1 MB)

    **Step 2** — Unzip it. You'll get a folder called `ml-latest-small/` containing:
    - `ratings.csv`
    - `movies.csv`
    - `tags.csv` (not used)
    - `links.csv` (not used)

    **Step 3** — Place `ml-latest-small/` in the **same folder** as this script.

    **Step 4** — Re-run the app:
    ```bash
    streamlit run movie_recommender_app.py
    ```

    ---
    **Dataset stats (ml-latest-small):**
    - 100,836 ratings
    - 9,742 movies
    - 610 users
    - Ratings scale: 0.5 → 5.0 ⭐
    """)
    st.stop()

# ── Sidebar ────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.header("⚙️ Settings")

    user_ids       = sorted(matrix.index.tolist())
    selected_user  = st.selectbox("Select User ID", user_ids)
    top_n          = st.slider("Recommendations to show", 1, 10, 5)

    st.divider()
    st.markdown("**Dataset info**")
    st.metric("Users in matrix",  matrix.shape[0])
    st.metric("Movies in matrix", matrix.shape[1])

    # Sparsity = fraction of cells that are 0 (not rated)
    sparsity = (matrix == 0).sum().sum() / matrix.size
    st.metric("Matrix sparsity", f"{sparsity:.1%}")

    st.divider()
    st.markdown("**How it works**")
    st.markdown(
        "1. Build a **User × Movie** ratings matrix\n"
        "2. Compute **cosine similarity** between all users\n"
        "3. For each unwatched movie, predict a score via **weighted average** of similar users\n"
        "4. Return top-N highest predicted scores"
    )

st.divider()

# ── Compute similarity matrix ──────────────────────────────────────────────────
# NOTE: Prefix 'index' with underscore so Streamlit skips hashing it
# (pandas Index is not hashable by st.cache_data)
@st.cache_data
def compute_similarity(matrix_values, _index):
    sim = cosine_similarity(matrix_values)
    return pd.DataFrame(sim, index=_index, columns=_index)

sim_df = compute_similarity(matrix.values, matrix.index.tolist())

# ── User's ratings summary ─────────────────────────────────────────────────────
st.subheader(f"User {selected_user} — Rating Profile")

user_row   = matrix.loc[selected_user]
watched    = user_row[user_row > 0].sort_values(ascending=False)
unwatched  = user_row[user_row == 0]

col1, col2, col3 = st.columns(3)
col1.metric("Movies rated",    len(watched))
col2.metric("Movies unrated",  len(unwatched))
col3.metric("Avg rating",      f"{watched.mean():.2f} ⭐" if len(watched) else "—")

with st.expander(f"📋 View all {len(watched)} rated movies"):
    watched_df = watched.reset_index()
    watched_df.columns = ["Movie", "Your Rating"]
    watched_df["Stars"] = watched_df["Your Rating"].apply(
        lambda r: "⭐" * int(r) + ("½" if r % 1 >= 0.5 else "")
    )
    st.dataframe(watched_df, use_container_width=True, hide_index=True)

st.divider()

# ── Similar users ──────────────────────────────────────────────────────────────
st.subheader("👥 Most Similar Users")

top_similar = sim_df[selected_user].drop(selected_user).sort_values(ascending=False).head(5)

cols = st.columns(5)
for col, (uid, score) in zip(cols, top_similar.items()):
    with col:
        st.metric(f"User {uid}", f"{score:.3f}")
        st.progress(float(score))
        n_rated = int((matrix.loc[uid] > 0).sum())
        st.caption(f"{n_rated} movies rated")

st.divider()

# ── Recommendations ────────────────────────────────────────────────────────────
st.subheader(f"🍿 Top {top_n} Recommendations for User {selected_user}")

with st.spinner("Crunching similarity scores…"):
    results = recommend_movies(selected_user, matrix, sim_df, top_n=top_n)

if not results:
    st.success("This user has rated every movie in the filtered dataset!")
else:
    for rank, (movie, score) in enumerate(results, start=1):
        with st.container():
            c1, c2 = st.columns([1, 8])
            with c1:
                st.metric("Rank", f"#{rank}")
            with c2:
                st.markdown(f"**{movie}**")
                # Show genre if available
                genre = genre_map.get(movie, "")
                if genre and genre != "(no genres listed)":
                    st.caption(f"🎭 {genre.replace('|', ' · ')}")
                bar_col, score_col = st.columns([5, 1])
                with bar_col:
                    st.progress(min(int((score / 5) * 100), 100))
                with score_col:
                    st.caption(f"{score:.2f} / 5.00")
        st.write("")

st.divider()
st.caption("Built with Python · pandas · scikit-learn · Streamlit · MovieLens dataset (GroupLens Research)")