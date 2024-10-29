"""
URL configuration for Forja project.

The `urlpatterns` list routes URLs to views. For more information please see:
    https://docs.djangoproject.com/en/5.1/topics/http/urls/
Examples:
Function views
    1. Add an import:  from my_app import views
    2. Add a URL to urlpatterns:  path('', views.home, name='home')
Class-based views
    1. Add an import:  from other_app.views import Home
    2. Add a URL to urlpatterns:  path('', Home.as_view(), name='home')
Including another URLconf
    1. Import the include() function: from django.urls import include, path
    2. Add a URL to urlpatterns:  path('blog/', include('blog.urls'))
"""
from django.contrib import admin
from django.urls import path
from django.conf.urls.static import static

from django.conf import settings
from ForjaApp import views
from django.conf.urls import handler404
from ForjaApp.views import submit_feedback, voice_search, cinema_list, rate_cinema,post_list, post_detail, post_create, post_update, post_delete, update_comment  # Import the view


urlpatterns = [path('', views.index, name='index'),
    path('admin/', admin.site.urls),  
    path('register/', views.register, name='register'),
    path('login/', views.login_view, name='login'),
    path('logout/', views.logout_view, name='logout'),     
    path('profile/', views.profile, name='profile'),
    path('recommend-similar/', views.recommend_similar_movies, name='recommend_similar_movies'),
    path('generate-image/', views.generate_image, name='generate_image'),
    path('movie-ending/', views.movie_ending_view, name='movie_ending'),
    path('song-writer/', views.song_writer_view, name='song_writer'),
    path('submit-feedback/', views.submit_feedback, name='submit_feedback'),
    path('user-feedback/', views.user_feedback_management, name='user_feedback_management'),
    path('posts/', post_list, name='post_list'),
    path('posts/<int:post_id>/', post_detail, name='post_detail'),
    path('posts/create/', post_create, name='post_create'),
    path('posts/update/<int:post_id>/', post_update, name='post_update'),
    path('posts/delete/<int:post_id>/', post_delete, name='post_delete'),
    path('comment/update/<int:comment_id>/', update_comment, name='update_comment'),
    path('comments/delete/<int:comment_id>/', views.delete_comment, name='delete_comment'),
    path('movies/create/', views.movie_create, name='create_movie'),
    path('movies/update/<int:movie_id>/', views.update_movie, name='update_movie'),
    path('movies/delete/<int:movie_id>/', views.delete_movie, name='delete_movie'),
    path('add_movie/', views.movie_create, name='add_movie'),  # Correct URL for adding a movie
    path('services/', views.services, name='services'),
    path('voice_search/', voice_search, name='voice_search'),
    path('movie_recognition/', views.movie_recognition, name='movie_recognition'),
    path('get_movie_summary/', views.get_movie_summary_and_ending, name='get_movie_summary'),
    path('add_genre/', views.add_genre, name='add_genre'),
    path('movies/', views.movie_list, name='movie_list'),
    path('movieList/', views.movieList, name='movieList'),
    path('movie-detail/<int:movie_id>/', views.movie_detail, name='movie_detail'),
    path('rate/<int:movie_id>/', views.rate_movie, name='rate_movie'),
    path('cinemas/', views.cinema_list, name='cinema_list'),
    path('cinemas/rate/<int:cinema_id>/', views.rate_cinema, name='rate_cinema'),
    path('artist', views.artist_list, name='artist_list'),
    path('artist/create/', views.create_artist, name='create_artist'),
    path('artist/<int:pk>/', views.artist_detail, name='artist_detail'),
    path('artist/<int:pk>/update/', views.update_artist, name='update_artist'),
    path('artist/<int:pk>/delete/', views.delete_artist, name='delete_artist'),
    path('remove-background/', views.remove_background, name='remove_background'),
    path('roles/', views.role_list, name='role_list'),
    path('roles/create/', views.role_create, name='role_create'),
    path('roles/update/<int:pk>/', views.role_update, name='role_update'),
    path('roles/delete/<int:pk>/', views.role_delete, name='role_delete'),
    path("ask_question/", views.ask_question, name="ask_question"),
    path("chat/", views.chat, name="chat"),
               ]
if settings.DEBUG:
    urlpatterns += static(settings.MEDIA_URL, document_root=settings.MEDIA_ROOT)

