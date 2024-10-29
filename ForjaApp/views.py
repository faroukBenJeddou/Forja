import base64

from PIL import Image
from django.shortcuts import render,redirect,get_object_or_404
from django.contrib.auth import login, authenticate, logout
from django.contrib import messages
from django.contrib.auth.forms import AuthenticationForm
from django.views.decorators.csrf import csrf_exempt
from platform import processor
import torch
from transformers import CLIPModel, CLIPProcessor
from django.db.models import Avg
from .models import Rating, WatchLater, CinemaRating, Artist, Role
from .forms import UserRegisterForm, RatingForm, CinemaRatingForm, ArtistForm, RoleForm, ChatBot
import google.generativeai as genai

from .forms import UserRegisterForm, GenreForm, MovieForm
from django.contrib.auth.decorators import login_required
from .utils import get_similar_movies
from .models import Movie, Recommendation, UserFeedback, Genre, Cinema
import re,requests,random,openai
from django.http import JsonResponse, HttpResponseRedirect
from groq import Groq
import requests
import time
import os
from django.conf import settings
from .models import Post, Comment
from .forms import PostForm, CommentForm
from django.core.paginator import Paginator
from django.urls import reverse
from nltk.sentiment import SentimentIntensityAnalyzer
import nltk
import speech_recognition as sr
from django.shortcuts import reverse

# Ensure VADER lexicon is downloaded
nltk.download('vader_lexicon', quiet=True)

# Create an instance of SentimentIntensityAnalyzer
sia = SentimentIntensityAnalyzer()


AI_API_URL = "https://api.openai.com/v1/chat/completions"
AI_API_KEY = os.getenv('OPENAI_API_KEY')

model = CLIPModel.from_pretrained("openai/clip-vit-base-patch16")
processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch16")
client = Groq(api_key='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')
openai.api_key = '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'
openai.api_base = "https://api.groq.com/openai/v1"
API_KEY = '89a4748b3788935d5e08221e4ed6f7ef'
BASE_URL = 'https://suno-apiv2-eight.vercel.app'

def range_1_to_5():
    return range(1, 6)
def index(request):
    # Fetch popular movies from the API
    url = f'https://api.themoviedb.org/3/movie/popular?api_key={API_KEY}&language=en-US&page=1'
    response = requests.get(url)
    api_movies = response.json().get('results', [])

    # Prepare the image base URL
    image_base_url = 'https://image.tmdb.org/t/p/w500/'

    # Add the complete URL for each movie image from the API
    for movie in api_movies:
        movie['poster_url'] = image_base_url + movie.get('poster_path', '')
        movie['vote_average_div_2'] = movie['vote_average'] / 2

    # Get all genres for filtering
    genres = Genre.objects.all()
    selected_genre = request.GET.get('genre')

    # Use movie_list logic to filter database movies
    if selected_genre:
        database_movies = Movie.objects.filter(genres__name=selected_genre)[:4]
    else:
        database_movies = Movie.objects.all()[:4]

    # Prepare context
    context = {
        'movies': api_movies[:8],  # Limit to 8 API movies
        'database_movies': database_movies,  # Movies from the database
        'genres': genres,  # All genres for the filter
        'selected_genre': selected_genre,
        'range_1_to_5': range(1, 6)
    }

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

@login_required
def user_feedback_management(request):
    feedbacks = UserFeedback.objects.filter(user=request.user)

    if request.method == 'POST':
        # Traitement de l'édition
        if 'save' in request.POST:
            feedback_id = request.POST.get('save')
            feedback_text = request.POST.get(f'feedback_text_{feedback_id}')
            rating = request.POST.get(f'rating_{feedback_id}')
            feedback = get_object_or_404(UserFeedback, id=feedback_id, user=request.user)
            feedback.feedback_text = feedback_text
            feedback.rating = rating
            feedback.save()

        # Traitement de la suppression
        elif 'delete' in request.POST:
            feedback_id = request.POST.get('delete')
            feedback = get_object_or_404(UserFeedback, id=feedback_id, user=request.user)
            feedback.delete()

        return redirect('user_feedback_management')  # Redirection après modification ou suppression

    return render(request, 'user_feedback_management.html', {
        'feedbacks': feedbacks
    })
@login_required
def recommend_similar_movies(request):
    similar_movies = []
    movie_title = ""
    recommendation = None  # Initialiser la variable recommendation

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

        return render(request, 'recommendations.html', {'similar_movies': similar_movies, 'movie_title': movie_title, 'recommendation': recommendation})

    return render(request, 'recommendations.html', {'similar_movies': similar_movies, 'movie_title': movie_title, 'recommendation': recommendation})

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

def generate_text(prompt):
    """Use Groq AI to generate a response based on a user prompt."""
    response = client.chat.completions.create(
        messages=[{"role": "user", "content": prompt}],
        model="llama3-8b-8192"
    )
    return response.choices[0].message.content.strip()

def text_to_speech(text):
    """Convert the given text to speech using Deepgram."""
    max_length = 2000
    if len(text) > max_length:
        text = text[:max_length]  # Truncate text to max_length characters

    url = "https://api.deepgram.com/v1/speak?model=aura-asteria-en"
    headers = {
        "Authorization": f"Token {'76dcd77db53a7659884675d0bfbad83679d5ab1f'}",
        "Content-Type": "application/json",
    }
    payload = {"text": text}

    response = requests.post(url, json=payload, headers=headers)

    print(f"Response Status Code: {response.status_code}")
    print(f"Response Body: {response.text}")

    if response.status_code == 200:
        return response.content  # Return audio bytes
    else:
        raise Exception(f"Failed to generate speech: {response.status_code}, {response.text}")






def movie_ending_view(request):
    """Handle user requests for custom movie endings."""
    if request.method == "POST":
        prompt = request.POST.get('prompt', "Imagine a new ending for Inception.")

        # Generate movie ending using Groq AI
        generated_text = generate_text(prompt)

        # Convert the generated text to speech
        audio_content = text_to_speech(generated_text)

        # Ensure the static directory exists
        static_dir = os.path.join(settings.BASE_DIR, "ForjaApp", "static")
        os.makedirs(static_dir, exist_ok=True)

        # Save the audio content to a file
        audio_file_path = os.path.join(static_dir, "generated_audio.mp3")
        with open(audio_file_path, "wb") as f:
            f.write(audio_content)

        return render(request, "movie_ending.html", {
            "generated_text": generated_text,
            "audio_path": f"{settings.STATIC_URL}generated_audio.mp3"  # Use STATIC_URL to construct the path
        })

    return render(request, "movie_ending.html")


def services(request):
    return render(request, 'services.html')


def generate_audio_by_prompt(prompt):
    """Send prompt to the Vercel API to generate audio."""
    url = f"{BASE_URL}/api/generate"
    payload = {
        "prompt": prompt,
        "make_instrumental": False,
        "wait_audio": True  # Synchronous generation
    }
    try:
        response = requests.post(url, json=payload, headers={'Content-Type': 'application/json'})
        response.raise_for_status()  # Raise an exception for 4xx/5xx responses
        return response.json()
    except Exception as e:
        print(f"Error in generate_audio_by_prompt: {e}")
        return None

def get_audio_information(audio_id):
    """Check the status and retrieve the audio URL."""
    url = f"{BASE_URL}/api/get?ids={audio_id}"
    try:
        response = requests.get(url)
        response.raise_for_status()  # Raise an exception for 4xx/5xx responses
        return response.json()
    except Exception as e:
        print(f"Error in get_audio_information: {e}")
        return None

def song_writer_view(request):
    """Django view to generate song lyrics and audio based on a prompt."""
    audio_url = None
    generated_lyrics = None

    if request.method == 'POST':
        prompt = request.POST.get('prompt', 'Write a pop song about dreams.')

        # Generate lyrics using your model
        try:
            generated_lyrics = client.chat.completions.create(
                messages=[{"role": "user", "content": prompt}],
                model="llama3-8b-8192"
            ).choices[0].message.content.strip()
        except Exception as e:
            return JsonResponse({"error": f"Failed to generate lyrics: {e}"}, status=500)

        # Call the Vercel API to generate audio from the lyrics
        audio_response = generate_audio_by_prompt(generated_lyrics)
        print("Audio Response:", audio_response)  # Debugging the raw response

        if audio_response and isinstance(audio_response, list) and len(audio_response) > 0:
            audio_id = audio_response[0].get('id')
            if not audio_id:
                return JsonResponse({"error": "Audio ID not found in the response"}, status=500)
        else:
            return JsonResponse({"error": "Invalid audio response"}, status=500)

        # Poll for audio status
        try:
            for _ in range(60):
                audio_info = get_audio_information(audio_id)
                print("Audio Info:", audio_info)  # Debugging the raw response

                if (audio_info and isinstance(audio_info, list) and len(audio_info) > 0 and 
                    audio_info[0].get("status") == "streaming"):
                    audio_url = audio_info[0].get("audio_url")
                    break
                time.sleep(5)
        except Exception as e:
            return JsonResponse({"error": f"Error retrieving audio information: {e}"}, status=500)

    return render(request, 'song_writer.html', {
        'generated_lyrics': generated_lyrics,
        'audio_url': audio_url
    }) 

def create_ratings_matrix():
    ratings = Rating.objects.all()
    ratings_data = {
        'user_id': [],
        'movie_id': [],
        'score': []
    }

    for rating in ratings:
        ratings_data['user_id'].append(rating.user.id)
        ratings_data['movie_id'].append(rating.movie.id)
        ratings_data['score'].append(rating.score)

    ratings_df = pd.DataFrame(ratings_data)
    
    # Créer une matrice d'évaluations
    ratings_matrix = ratings_df.pivot_table(index='user_id', columns='movie_id', values='score').fillna(0)

    return ratings_matrix

def get_recommendations_(user_id, num_recommendations=5):
    # Créer la matrice d'évaluations
    ratings_matrix = create_ratings_matrix()

    # Calculer la similarité cosinus entre les utilisateurs
    similarity_matrix = cosine_similarity(ratings_matrix)

    # Trouver l'indice de l'utilisateur
    user_index = ratings_matrix.index.get_loc(user_id)

    # Obtenir les scores de similarité pour l'utilisateur donné
    similar_scores = similarity_matrix[user_index]

    # Trouver les utilisateurs les plus similaires
    similar_users_indices = similar_scores.argsort()[::-1][1:num_recommendations + 1]

    recommended_movies = []
    
    # Parcourir les utilisateurs similaires pour recommander des films
    for similar_user_index in similar_users_indices:
        similar_user_id = ratings_matrix.index[similar_user_index]
        similar_user_ratings = ratings_matrix.loc[similar_user_id]

        # Obtenir les films non évalués par l'utilisateur courant
        non_rated_movies = similar_user_ratings[similar_user_ratings > 0].index

        # Ajouter les films recommandés à la liste
        recommended_movies.extend(non_rated_movies)

    # Éliminer les doublons et limiter le nombre de recommandations
    recommended_movies = list(set(recommended_movies))[:num_recommendations]

    return recommended_movies

@csrf_exempt
def get_movie_summary_and_ending(request):
    if request.method == 'POST':
        movie_title = request.POST.get('movie_title')
        if not movie_title:
            return JsonResponse({'error': 'No movie title provided'}, status=400)

        omdb_api_key = 'dc75851e'
        omdb_url = f'https://www.omdbapi.com/?t={movie_title}&apikey={omdb_api_key}'

        try:
            response = requests.get(omdb_url)
            data = response.json()

            if data.get('Response') == 'True':
                plot = data.get('Plot', 'No plot found.')
                # Generate a different ending using the correct function
                alternative_ending = generate_different_ending(plot)

                return JsonResponse({
                    'title': data['Title'],
                    'summary': plot,
                    'alternative_ending': alternative_ending
                })

            return JsonResponse({'error': 'Movie not found'}, status=404)

        except Exception as e:
            return JsonResponse({'error': str(e)}, status=500)

    return JsonResponse({'error': 'Invalid request method'}, status=400)

def generate_different_ending(original_plot):
    ai_api_token = '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'  # Your Hugging Face API token
    ai_api_url = 'https://api-inference.huggingface.co/models/distilgpt2'  # Using distilgpt2 model

    headers = {
        'Authorization': f'Bearer {ai_api_token}',
        'Content-Type': 'application/json'
    }

    # Clearer prompt
    prompt = (
        f"Plot Summary: {original_plot}\n"
        "Let's Imagine a different ending for this movie, one that changes the outcome entirely. "

    )

    payload = {
        'inputs': prompt,
        'parameters': {
            'max_length': 150,  # Increased for more room to generate
            'temperature': 0.9,  # More creative variations
            'top_k': 50,  # Sampling top k tokens for diversity
            'top_p': 0.95,  # Nucleus sampling
        }
    }

    response = requests.post(ai_api_url, headers=headers, json=payload)

    if response.status_code == 200:
        response_json = response.json()
        if isinstance(response_json, list) and len(response_json) > 0:
            return response_json[0]['generated_text'].strip()  # Remove leading/trailing whitespace
        else:
            return "Could not generate an ending at this time."
    else:
        return f"API Error: {response.status_code}, {response.text}"

def add_genre(request):
    if request.method == 'POST':
        form = GenreForm(request.POST)
        if form.is_valid():
            form.save()  # Save the new genre
            return redirect('add_genre')  # Redirect back to the genre add page
    else:
        form = GenreForm()

    return render(request, 'add_genre.html', {'form': form})

@login_required
def recommend_similar(request):
    similar_movies = []
    movie_title = ""

    if request.method == 'POST':
        movie_title = request.POST.get('movie_title')

        # Vous pouvez remplacer ceci par votre propre logique pour obtenir des films similaires
        similar_movies = get_similar_movies(movie_title)  # Assurez-vous d'avoir cette fonction définie

        # Vérifier si le film est trouvé dans la base de données
        movie = Movie.objects.filter(title__icontains=movie_title).first()

        # Si le film n'est pas trouvé, l'ajouter à la base de données
        if not movie:
            if similar_movies:
                movie_data = similar_movies[0]
                movie = Movie(
                    title=movie_data['title'],
                    release_date=movie_data['release_date'],
                    overview=movie_data['overview'],
                    poster_path=movie_data['poster_path']
                )
                movie.save()

        # Si le film est trouvé ou a été ajouté, enregistrer la recommandation
        if movie:
            # Obtenir les recommandations pour l'utilisateur
            recommended_movie_ids = get_recommendations(request.user.id)

            # Récupérer les films recommandés à partir des IDs
            similar_movies = Movie.objects.filter(id__in=recommended_movie_ids)

            Recommendation.objects.create(
                user=request.user,
                movie=movie,
                similar_movies=[{"title": m.title, "id": m.id} for m in similar_movies]
            )

        return render(request, 'recommendations.html', {'similar_movies': similar_movies, 'movie_title': movie_title})

    return render(request, 'recommendations.html', {'similar_movies': similar_movies, 'movie_title': movie_title,'recommendation': recommendation})

@csrf_exempt
def movie_recognition(request):
    if request.method == 'POST':
        image_file = request.FILES.get('image')
        if image_file:
            try:
                # Open the image using PIL
                image = Image.open(image_file).convert("RGB")

                # Prepare a more extensive list of relevant movie titles
                movie_titles = [
                    'The Dark Knight',
                    'Batman',
                    'Batman Begins',
                    'The Dark Knight Rises',
                    'Joker',
                    'Suicide Squad',
                    'The Killing Joke',
                    'Batman vs. Superman: Dawn of Justice',
                    'Titanic',
                    'Inception',
                    'Gladiator',
                    'Jurassic Park',
                    'Iron Man',
                    'The Avengers',

                    # Add more relevant titles as needed
                ]

                # Process the image and text
                inputs = processor(text=movie_titles, images=image, return_tensors="pt", padding=True)

                # Make predictions
                with torch.no_grad():
                    outputs = model(**inputs)
                    logits_per_image = outputs.logits_per_image
                    probs = logits_per_image.softmax(dim=1)

                # Get the top K predictions
                top_k_indices = probs[0].topk(5)
                results = [
                    {"title": movie_titles[i.item()], "probability": probs[0][i].item()}
                    for i in top_k_indices.indices
                ]

                # Adjust the probability threshold for filtering
                threshold = 0.15  # Try different values based on testing
                filtered_results = [
                    result for result in results if result['probability'] >= threshold
                ]

                # Optional: Add logic to include related titles if certain ones are found
                if any("The Dark Knight" in result['title'] for result in filtered_results):
                    filtered_results.append({"title": "Batman", "probability": 0.8})  # Example addition

                return JsonResponse({"results": filtered_results})
            except Exception as e:
                print(f"Error processing the image: {e}")
                return JsonResponse({"error": str(e)}, status=500)

    return render(request, 'movie_recognition.html')

@login_required
def submit_feedback(request):
    if request.method == 'POST':
        recommendation_id = request.POST.get('recommendation_id')
        feedback_text = request.POST.get('feedback_text')
        rating = request.POST.get('rating')

        if recommendation_id:  # Check if recommendation_id is not empty
            try:
                recommendation = Recommendation.objects.get(id=recommendation_id)
                UserFeedback.objects.create(user = request.user,  # Get the currently logged-in user
                    recommendation=recommendation,
                    feedback_text=feedback_text,
                    rating=rating
                )
                messages.success(request, 'Feedback submitted successfully!')
                return redirect('index')  # Or wherever you want to redirect
            except Recommendation.DoesNotExist:
                messages.error(request, 'Recommendation not found.')
        else:
            messages.error(request, 'Recommendation ID cannot be empty.')
    return redirect('index')  # Redirect if the method is not POST
def movie_list(request):
    genres = Genre.objects.all()  # Get all genres
    selected_genre = request.GET.get('genre')

    if selected_genre:
        movies = Movie.objects.filter(genres__name=selected_genre)
    else:
        movies = Movie.objects.all()  # Get all movies if no genre is selected

    return render(request, 'movie_list.html', {
        'movies': movies,
        'genres': genres,  # Make sure genres are passed here
        'selected_genre': selected_genre
    })
@login_required
def movie_create(request):
    if request.method == 'POST':
        form = MovieForm(request.POST, request.FILES)  # Include request.FILES for image upload
        if form.is_valid():
            movie = form.save(commit=False)  # Create the movie instance but don't save yet
            movie.save()  # Save the movie instance
            form.save_m2m()  # Save the many-to-many relationships (genres)
            print("Movie saved successfully!")  # Debug line
            return redirect('index')  # Redirect to the index page after saving
        else:
            print("Form errors:", form.errors)  # Debug line to see form errors
    else:
        form = MovieForm()

    return render(request, 'add_movie.html', {'form': form})  # Render the form
@login_required
def update_movie(request, movie_id):
    movie = get_object_or_404(Movie, id=movie_id)

    if request.method == 'POST':
        form = MovieForm(request.POST, request.FILES, instance=movie)  # Include request.FILES for file uploads
        if form.is_valid():
            form.save()
            messages.success(request, 'Movie updated successfully!')
            return redirect('index')  # Ensure 'index' is a valid URL name
    else:
        form = MovieForm(instance=movie)  # Pre-populate the form with existing movie data

    return render(request, 'update_movie.html', {'form': form, 'movie': movie})
# Delete Movie
@login_required
def delete_movie(request, movie_id):
    movie = get_object_or_404(Movie, id=movie_id)
    if request.method == 'POST':
        movie.delete()
        messages.success(request, 'Movie deleted successfully!')
        return redirect('index')
    return render(request, 'delete_movie.html', {'movie': movie})
def voice_search(request):
    if request.method == 'POST':
        # Here, implement the voice recognition logic
        recognizer = sr.Recognizer()
        with sr.Microphone() as source:
            print("Listening...")
            audio = recognizer.listen(source)
            try:
                query = recognizer.recognize_google(audio)
                print("You said: " + query)
                # Search your Movie model
                results = Movie.objects.filter(title__icontains=query)
                return render(request, 'search_results.html', {'results': results})
            except sr.UnknownValueError:
                return render(request, 'voice_search.html', {'error': "Could not understand audio"})
            except sr.RequestError:
                return render(request, 'voice_search.html',
                              {'error': "Could not request results from Google Speech Recognition service"})

VIDSRC_API_URL = "https://vidsrc.xyz/embed/movie"
TMDB_API_KEY = '89a4748b3788935d5e08221e4ed6f7ef'


def movie_detail(request, movie_id):
    # Get the movie from the database
    movie = get_object_or_404(Movie, id=movie_id)

    # Fetch the TMDB ID using the movie title
    tmdb_id = get_tmdb_id(movie.title)

    # If a TMDB ID is found, get the streaming URL
    streaming_url = None
    if tmdb_id:
        streaming_url = f"https://vidsrc.xyz/embed/movie?tmdb={tmdb_id}"

    return render(request, 'movie_detail.html', {
        'movie': movie,
        'streaming_url': streaming_url,
    })

def get_tmdb_id(movie_title):
    try:
        # Search for the movie by title in the TMDB API
        search_url = f"https://api.themoviedb.org/3/search/movie?api_key={TMDB_API_KEY}&query={movie_title}"
        search_response = requests.get(search_url).json()

        if search_response['results']:
            tmdb_id = search_response['results'][0]['id']  # Get the TMDB ID of the first result
            print(f"TMDB ID for '{movie_title}': {tmdb_id}")  # Log the TMDB ID
            return tmdb_id

    except Exception as e:
        print(f"Error retrieving TMDB ID: {e}")

    return None

@login_required
def rate_movie(request, movie_id):
    movie = get_object_or_404(Movie, id=movie_id)

    ratings = Rating.objects.filter(movie=movie).order_by('-date_rated')

    user_rating = Rating.objects.filter(movie=movie, user=request.user).first()

    if request.method == 'POST':
        form = RatingForm(request.POST)
        if form.is_valid():
            # Save or update the user's rating
            if user_rating:
                user_rating.score = form.cleaned_data['score']
                user_rating.review = form.cleaned_data['review']
                user_rating.save()
            else:
                rating = form.save(commit=False)
                rating.movie = movie
                rating.user = request.user
                rating.save()

            # Handle Watch Later checkbox
            if 'watch_later' in request.POST:
                # Add movie to Watch Later
                WatchLater.objects.get_or_create(user=request.user, movie=movie)
            else:
                # Remove movie from Watch Later
                WatchLater.objects.filter(user=request.user, movie=movie).delete()

            return redirect('rate_movie', movie_id=movie.id)
    else:
        form = RatingForm() if not user_rating else None

    return render(request, 'rate_movie.html', {
        'movie': movie,
        'form': form,
        'ratings': ratings,
        'user_rating': user_rating,
    })

def movieList(request):
    movies = Movie.objects.annotate(average_rating=Avg('rating__score'))
    return render(request, 'movie_list.html', {'movies': movies})



def cinema_list(request):
    cinemas = Cinema.objects.annotate(average_rating=Avg('cinemarating__score'))  # Annotate with average rating
    return render(request, 'cinema_list.html', {'cinemas': cinemas})


@login_required
def rate_cinema(request, cinema_id):
    cinema = get_object_or_404(Cinema, id=cinema_id)

    # Get all ratings for the cinema
    ratings = CinemaRating.objects.filter(cinema=cinema).order_by('-date_rated')

    # Get the user's rating if it exists
    user_rating = CinemaRating.objects.filter(cinema=cinema, user=request.user).first()

    if request.method == 'POST':
        form = CinemaRatingForm(request.POST)
        if form.is_valid():
            if user_rating:
                # Update existing rating
                user_rating.score = form.cleaned_data['score']
                user_rating.review = form.cleaned_data['review']
                user_rating.save()
            else:
                # Create a new rating
                cinema_rating = form.save(commit=False)
                cinema_rating.cinema = cinema
                cinema_rating.user = request.user  # Ensure that the user is logged in
                cinema_rating.save()

            return redirect('rate_cinema', cinema_id=cinema.id)
    else:
        form = CinemaRatingForm() if not user_rating else None  # Use None if user_rating exists

    return render(request, 'rate_cinema.html', {
        'cinema': cinema,
        'form': form,
        'ratings': ratings,
        'user_rating': user_rating,
    })

#########################################################################   khalil
def create_artist(request):
    if request.method == 'POST':
        form = ArtistForm(request.POST, request.FILES)  # Handle file uploads
        if form.is_valid():
            artist = form.save()  # Save the artist instance
            # Save the selected roles
            form.cleaned_data['roles']  # This will contain the selected roles
            artist.roles.set(form.cleaned_data['roles'])  # Assign the roles to the artist
            return redirect('artist_list')  # Redirect to the artist list after saving
    else:
        form = ArtistForm()

    return render(request, 'create_artist.html', {'form': form})


# Lire la liste des artistes
def artist_list(request):
    artists = Artist.objects.all()
    return render(request, 'artist_list.html', {'artists': artists})

# Lire un artiste spécifique
def artist_detail(request, pk):
    artist = get_object_or_404(Artist, pk=pk)
    return render(request, 'artist_detail.html', {'artist': artist})

# Mettre à jour un artiste
def update_artist(request, pk):
    artist = get_object_or_404(Artist, pk=pk)
    if request.method == 'POST':
        form = ArtistForm(request.POST, instance=artist)
        if form.is_valid():
            form.save()
            messages.success(request, 'Artiste mis à jour avec succès.')
            return redirect('artist_detail', pk=artist.pk)
        else:
            messages.error(request, 'Erreur lors de la mise à jour de l\'artiste.')
    else:
        form = ArtistForm(instance=artist)
    return render(request, 'update_artist.html', {'form': form, 'artist': artist})

# Supprimer un artiste
def delete_artist(request, pk):
    artist = get_object_or_404(Artist, pk=pk)
    if request.method == 'POST':
        artist.delete()
        messages.success(request, 'Artiste supprimé avec succès.')
        return redirect('artist_list')
    return render(request, 'delete_artist.html', {'artist': artist})
## VoRk6rXgydz4zAm7ng5F86Gt          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     4340a429-6c02-4795-9bf6-95f453bf84ba
# remover/views.py
def remove_background(request):
    image_base64 = None  # Initialiser la variable d'image

    if request.method == 'POST' and request.FILES.get('image'):
        # Récupérer le fichier image
        image_file = request.FILES['image']

        # API key de votre compte Remove.bg
        api_key = 'VoRk6rXgydz4zAm7ng5F86Gt'

        # Préparer la requête à l'API Remove.bg
        r = requests.post(
            "https://api.remove.bg/v1.0/removebg",
            files={'image_file': image_file},
            headers={'X-Api-Key': api_key},
            data={'size': 'auto'}
        )

        if r.status_code == requests.codes.ok:
            # Convertir l'image binaire en Base64
            image_base64 = base64.b64encode(r.content).decode('utf-8')

    return render(request, 'upload.html', {'image': image_base64})  # Passer l'image au template





def role_list(request):
    roles = Role.objects.all()
    return render(request, 'role_list.html', {'roles': roles})

def role_create(request):
    if request.method == 'POST':
        form = RoleForm(request.POST)
        if form.is_valid():
            form.save()
            return redirect('role_list')
    else:
        form = RoleForm()
    return render(request, 'role_form.html', {'form': form})

def role_update(request, pk):
    role = get_object_or_404(Role, pk=pk)
    if request.method == 'POST':
        form = RoleForm(request.POST, instance=role)
        if form.is_valid():
            form.save()
            return redirect('role_list')
    else:
        form = RoleForm(instance=role)
    return render(request, 'role_form.html', {'form': form})

def role_delete(request, pk):
    role = get_object_or_404(Role, pk=pk)
    if request.method == 'POST':
        role.delete()
        return redirect('role_list')
    return render(request, 'role_confirm_delete.html', {'role': role})

genai.configure(api_key="AIzaSyA0DzNxZWLn56ELHdr-WaPLigFAztEUOLg")

@login_required
def ask_question(request):
    if request.method == "POST":
        text = request.POST.get("text")
        model = genai.GenerativeModel("gemini-pro")
        chat = model.start_chat()
        response = chat.send_message(text)
        user = request.user
        ChatBot.objects.create(text_input=text, gemini_output=response.text, user=user)
        # Extract necessary data from response
        response_data = {
            "text": response.text,  # Assuming response.text contains the relevant response data
            # Add other relevant data from response if needed
        }
        return JsonResponse({"data": response_data})
    else:
        return HttpResponseRedirect(
            reverse("chat")
        )  # Redirect to chat page for GET requests


def chat(request):
    user = request.user
    chats = ChatBot.objects.filter(user=user)
    return render(request, "chat_bot.html", {"chats": chats})
#######################################################################  AIzaSyA0DzNxZWLn56ELHdr-WaPLigFAztEUOLg

@login_required
def post_list(request):
    """Display all posts with comments."""
    posts = Post.objects.all().order_by('-created_at')
    paginator = Paginator(posts, 2)
    page_number = request.GET.get("page")
    page_obj = paginator.get_page(page_number)

    return render(request, 'post_list.html', {"page_obj": page_obj})


@login_required
def post_detail(request, post_id):
    post = get_object_or_404(Post, id=post_id)
    comments = Comment.objects.filter(post=post)
    form = CommentForm(request.POST or None)

    if request.method == 'POST' and form.is_valid():
        comment = form.save(commit=False)
        comment.post = post
        comment.author = request.user
        comment.save()

        # Analyze the comment using VADER
        ai_analysis = analyze_comment_with_vader(comment.content)
        comment.analysis = ai_analysis  # Store analysis in the comment
        comment.save()

        return redirect('post_detail', post_id=post.id)

    return render(request, 'post_detail.html', {
        'post': post,
        'comments': comments,
        'form': form,
    })

@login_required
def update_comment(request, comment_id):
    """Update an existing comment."""
    comment = get_object_or_404(Comment, id=comment_id, author=request.user)
    if request.method == 'POST':
        form = CommentForm(request.POST, instance=comment)
        if form.is_valid():
            form.save()
            messages.success(request, "Comment updated successfully!")
            return redirect('post_detail', post_id=comment.post.id)
    else:
        form = CommentForm(instance=comment)
    return render(request, 'update_comment.html', {'form': form, 'comment': comment})

@login_required
def post_create(request):
    if request.method == 'POST':
        form = PostForm(request.POST)
        if form.is_valid():
            post = form.save(commit=False)
            post.author = request.user
            post.save()
            return redirect('post_list')
    else:
        form = PostForm()
    return render(request, 'post_form.html', {'form': form})

@login_required
def post_update(request, post_id):
    post = get_object_or_404(Post, id=post_id)
    if request.user != post.author:
        return redirect('post_detail', post_id=post.id)

    if request.method == 'POST':
        form = PostForm(request.POST, instance=post)
        if form.is_valid():
            form.save()
            return redirect('post_detail', post_id=post.id)
    else:
        form = PostForm(instance=post)

    return render(request, 'post_form.html', {'form': form})

@login_required
def post_delete(request, post_id):
    post = get_object_or_404(Post, id=post_id)
    if request.user == post.author:
        post.delete()
    return redirect('post_list')

@login_required
def delete_comment(request, comment_id):
    comment = get_object_or_404(Comment, id=comment_id)

    # Only allow the author of the comment to delete it
    if comment.author == request.user:
        post_id = comment.post.id  # Save the post ID to redirect after deletion
        comment.delete()
        return redirect(reverse('post_detail', args=[post_id]))
    else:
        return redirect('post_list')  # Or show a message that they can't delete it


def analyze_comment_with_vader(comment_text):
    try:
        sentiment = sia.polarity_scores(comment_text)
        return sentiment  # Return the sentiment analysis result
    except Exception as e:
        return {"error": str(e)}  # Handle any exceptions