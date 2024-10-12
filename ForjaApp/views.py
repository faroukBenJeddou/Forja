from django.shortcuts import render
from django.shortcuts import render, redirect
from django.contrib.auth import login, authenticate, logout
from django.contrib import messages
from django.contrib.auth.forms import AuthenticationForm
from .forms import UserRegisterForm
from django.contrib.auth.decorators import login_required
from .utils import get_similar_movies
from .models import Recommendation
from .models import Movie 
import re
import requests
import random

API_KEY = '89a4748b3788935d5e08221e4ed6f7ef'
def range_1_to_5():
    return range(1, 6)
def index(request):
    url = f'https://api.themoviedb.org/3/movie/popular?api_key={API_KEY}&language=en-US&page=1'
    response = requests.get(url)
    movies = response.json().get('results', [])

    # Préparer l'URL de base des images
    image_base_url = 'https://image.tmdb.org/t/p/w500/'

    # Ajouter l'URL complète pour chaque image de film et calculer la note moyenne sur 2
    for movie in movies:
        movie['poster_url'] = image_base_url + movie.get('poster_path', '')
        movie['vote_average_div_2'] = movie['vote_average'] / 2

    context = {'movies': movies[:8], 'range_1_to_5': range(1, 6)}  # Limiter à 8 films
    return render(request, 'index.html', context)

# views.py


def register(request):
    if request.method == 'POST':
        form = UserRegisterForm(request.POST)
        if form.is_valid():
            form.save()
            username = form.cleaned_data.get('username')
            messages.success(request, f'Votre compte a été créé avec succès !')
            return redirect('login')
    else:
        form = UserRegisterForm()
    return render(request, 'register.html', {'form': form})

def login_view(request):
    if request.method == 'POST':
        form = AuthenticationForm(request, data=request.POST)
        if form.is_valid():
            username = form.cleaned_data.get('username')
            password = form.cleaned_data.get('password')
            user = authenticate(username=username, password=password)
            if user is not None:
                login(request, user)
                messages.success(request, f'Vous êtes connecté !')
                return redirect('index')
            else:
                messages.error(request, 'Nom d\'utilisateur ou mot de passe incorrect.')
    else:
        form = AuthenticationForm()
    return render(request, 'login.html', {'form': form})

def logout_view(request):
    logout(request)
    messages.success(request, 'Vous avez été déconnecté.')
    return redirect('login')

@login_required  # Ceci nécessite que l'utilisateur soit connecté
def profile(request):
    return render(request, 'profile.html', {'user': request.user})

from django.shortcuts import render

def custom_404_view(request, exception):
    return render(request, '404.html', status=404)
@login_required
def recommend_similar_movies(request):
    similar_movies = []
    movie_title = ""

    if request.method == 'POST':
        movie_title = request.POST.get('movie_title')

        # Obtenir les films similaires à l'aide de l'algorithme optimisé
        similar_movies = get_similar_movies(movie_title)

        # Vérifier si le film est trouvé dans la base de données
        movie = Movie.objects.filter(title__icontains=movie_title).first()

        # Si le film n'est pas trouvé, l'ajouter à la base de données
        if not movie:
            # Utiliser le premier film similaire pour créer une nouvelle entrée de film
            if similar_movies:  # Vérifier qu'il y a des films similaires
                movie_data = similar_movies[0]  # Prenons le premier film similaire

                # Créer une nouvelle instance de Movie
                movie = Movie(
                    title=movie_data['title'],
                    release_date=movie_data['release_date'],
                    overview=movie_data['overview'],
                    poster_path=movie_data['poster_path']
                )
                movie.save()  # Sauvegarder le nouveau film dans la base de données

        # Si le film est trouvé ou a été ajouté, enregistrer la recommandation
        if movie:
            recommendation = Recommendation.objects.create(
                user=request.user,
                movie=movie,
                similar_movies=[{"title": m['title'], "id": m['id']} for m in similar_movies]  # Enregistrer les titres et IDs des films similaires
            )

        return render(request, 'recommendations.html', {'similar_movies': similar_movies, 'movie_title': movie_title})

    return render(request, 'recommendations.html', {'similar_movies': similar_movies, 'movie_title': movie_title})
def generate_image(request):
    image_url = None
    error_message = None

    if request.method == 'POST':
        description = request.POST.get('description')
        selected_style = request.POST.get('style')  # Récupérer le style sélectionné

        # Liste de mots-clés associés aux films
        movie_keywords = [
            'film', 'movie', 'character', 'plot', 'scene', 'actor', 
            'actress', 'director', 'genre', 'cinema', 'trailer'
        ]

        # Vérifier si la description contient des mots-clés de film
        if any(re.search(r'\b' + keyword + r'\b', description, re.IGNORECASE) for keyword in movie_keywords):
            # Ajouter le style de dessin à la description
            modified_description = f"{description}, {selected_style}"
            image_url = f"https://image.pollinations.ai/prompt/{modified_description}"
        else:
            error_message = "Veuillez entrer une description liée aux films ou personnages de films."

    return render(request, 'image_generator.html', {'image_url': image_url, 'error_message': error_message})