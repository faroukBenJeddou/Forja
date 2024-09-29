import requests 

TMDB_API_KEY = '89a4748b3788935d5e08221e4ed6f7ef'  # Remplace par ta clé API

def get_similar_movies(movie_title):
    # Étape 1: Rechercher le film par son titre
    search_url = f"https://api.themoviedb.org/3/search/movie?api_key={TMDB_API_KEY}&query={movie_title}"
    search_response = requests.get(search_url).json()

    if search_response['results']:
        movie_id = search_response['results'][0]['id']
        
        # Étape 2: Obtenir les détails du film pour récupérer les genres
        movie_details_url = f"https://api.themoviedb.org/3/movie/{movie_id}?api_key={TMDB_API_KEY}"
        movie_details_response = requests.get(movie_details_url).json()
        
        # Récupérer les genres
        genres = [genre['id'] for genre in movie_details_response['genres']]
        
        # Étape 3: Obtenir les films similaires
        similar_url = f"https://api.themoviedb.org/3/movie/{movie_id}/similar?api_key={TMDB_API_KEY}"
        similar_response = requests.get(similar_url).json()
        
        # Récupérer les films similaires filtrés par genre et popularité
        similar_movies = [
            movie for movie in similar_response['results'] 
            if set(genres) & set(movie.get('genre_ids', []))  # Assurez-vous qu'ils partagent au moins un genre
        ]

        # Trier par popularité ou note
        similar_movies = sorted(similar_movies, key=lambda x: (x.get('popularity', 0), x.get('vote_average', 0)), reverse=True)

        # Limiter à 3 films
        return similar_movies[:3]

    return []
