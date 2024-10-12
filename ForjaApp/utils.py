import requests
from requests.exceptions import RequestException

TMDB_API_KEY = '89a4748b3788935d5e08221e4ed6f7ef'

def get_similar_movies(movie_title_or_keyword):
    try:
        # 1. Rechercher le film par son titre
        search_url = f"https://api.themoviedb.org/3/search/movie?api_key={TMDB_API_KEY}&query={movie_title_or_keyword}"
        search_response = requests.get(search_url).json()
        movie_results = search_response.get('results', [])

        # Si aucun film trouvé par le titre, passer à la recherche par mots-clés
        if not movie_results:
            return search_by_keyword(movie_title_or_keyword)

        # 2. Récupérer l'ID du premier film trouvé
        movie_id = movie_results[0]['id']

        # 3. Obtenir les recommandations basées sur ce film
        recommendations_url = f"https://api.themoviedb.org/3/movie/{movie_id}/recommendations?api_key={TMDB_API_KEY}&language=en-US&page=1"
        recommendations_response = requests.get(recommendations_url).json()
        recommended_movies = recommendations_response.get('results', [])

        # 4. Limiter à 5 films recommandés et retourner les résultats
        return recommended_movies[:5]

    except RequestException as e:
        print(f"Erreur lors de l'appel à l'API: {e}")
        return []

def search_by_keyword(keyword):
    try:
        # 1. Rechercher des films par mots-clés
        keyword_url = f"https://api.themoviedb.org/3/search/keyword?api_key={TMDB_API_KEY}&query={keyword}"
        keyword_response = requests.get(keyword_url).json()
        keyword_results = keyword_response.get('results', [])

        if not keyword_results:
            return []  # Aucun mot-clé trouvé

        # 2. Utiliser le premier mot-clé trouvé pour chercher des films
        keyword_id = keyword_results[0]['id']
        movies_url = f"https://api.themoviedb.org/3/discover/movie?api_key={TMDB_API_KEY}&with_keywords={keyword_id}&language=en-US&page=1"
        movies_response = requests.get(movies_url).json()

        # 3. Retourner les films trouvés avec ce mot-clé (limité à 5)
        return movies_response.get('results', [])[:5]

    except RequestException as e:
        print(f"Erreur lors de l'appel à l'API: {e}")
        return []