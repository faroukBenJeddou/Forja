from django.db import models
from django.contrib.auth.models import User

from django.db import models

class Genre(models.Model):
    name = models.CharField(max_length=100)

    def __str__(self):
        return self.name
class Movie(models.Model):
    title = models.CharField(max_length=200)
    release_date = models.DateField()
    overview = models.TextField(blank=True, default='')  # Ajoute `blank=True` et `default=''`
    poster_path = models.ImageField(upload_to='posters/')  # Field for storing image paths
    genres = models.ManyToManyField(Genre, related_name='movies')

    def __str__(self):
        return self.title


class Rating(models.Model):
    user = models.ForeignKey(User, on_delete=models.CASCADE)
    movie = models.ForeignKey(Movie, on_delete=models.CASCADE)
    score = models.IntegerField()  # Score sur 5
    review = models.TextField(blank=True, null=True)
    date_rated = models.DateTimeField(auto_now_add=True)

    def __str__(self):
        return f"{self.user.username} - {self.movie.title} - {self.score}"

class Recommendation(models.Model):
    user = models.ForeignKey(User, on_delete=models.CASCADE)
    movie = models.ForeignKey(Movie, on_delete=models.CASCADE)
    similar_movies = models.JSONField()  # Pour stocker les informations des films similaires

    def __str__(self):
        return f"Recommendation for {self.user.username} based on {self.movie.title}"

class ArtisticDescription(models.Model):
    movie = models.ForeignKey(Movie, on_delete=models.CASCADE)
    description = models.TextField()

    def __str__(self):
        return f"Description for {self.movie.title}"



class UserFeedback(models.Model):
    user = models.ForeignKey(User, on_delete=models.CASCADE)
    recommendation = models.ForeignKey(Recommendation, on_delete=models.CASCADE)
    feedback_text = models.TextField()
    rating = models.IntegerField(choices=[(1, '1 étoile'), (2, '2 étoiles'), (3, '3 étoiles'), (4, '4 étoiles'), (5, '5 étoiles')])
    created_at = models.DateTimeField(auto_now_add=True)

    def __str__(self):
        return f"Feedback from {self.user.username} on {self.recommendation.movie.title}"


class Cinema(models.Model):
    name = models.CharField(max_length=255)
    description = models.TextField()  # Ensure this field is present
    location = models.CharField(max_length=255)
    date_created = models.DateTimeField(auto_now_add=True)

    def __str__(self):
        return self.name


class CinemaRating(models.Model):
    user = models.ForeignKey(User, on_delete=models.CASCADE)
    cinema = models.ForeignKey(Cinema, on_delete=models.CASCADE)
    score = models.IntegerField(choices=[(1, '1 étoile'), (2, '2 étoiles'), (3, '3 étoiles'), (4, '4 étoiles'), (5, '5 étoiles')])
    review = models.TextField(blank=True, null=True)
    date_rated = models.DateTimeField(auto_now_add=True)

    def __str__(self):
        return f"{self.user.username} - {self.cinema.name} - {self.score} étoiles"

class WatchLater(models.Model):
    user = models.ForeignKey(User, on_delete=models.CASCADE)
    movie = models.ForeignKey(Movie, on_delete=models.CASCADE)
    added_on = models.DateTimeField(auto_now_add=True)

    class Meta:
        unique_together = ('user', 'movie')

    def __str__(self):
        return f"{self.user.username} - Watch Later - {self.movie.title}"

class Role(models.Model):
    title = models.CharField(max_length=100)
    description = models.TextField(null=True, blank=True)

    def __str__(self):
        return self.title
class Artist(models.Model):
    name = models.CharField(max_length=100)
    biography = models.TextField()
    birth_date = models.DateField(null=True, blank=True)
    nationality = models.CharField(max_length=50, null=True, blank=True)
    death_date = models.DateField(null=True, blank=True)
    image = models.ImageField(upload_to='artists/', null=True, blank=True)  # Champ pour l'image
    roles = models.ManyToManyField(Role, related_name='artists')  # Relation Many-to-Many avec Role

    def __str__(self):
        return self.name

class Post(models.Model):
    title = models.CharField(max_length=200)
    content = models.TextField()
    author = models.ForeignKey(User, on_delete=models.CASCADE)
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)

    def __str__(self):
        return self.title

class Comment(models.Model):
    post = models.ForeignKey(Post, related_name='comments', on_delete=models.CASCADE)
    author = models.ForeignKey(User, on_delete=models.CASCADE)
    content = models.TextField()
    created_at = models.DateTimeField(auto_now_add=True)
    analysis = models.JSONField(default=dict, blank=True)

    def __str__(self):
        return f"Comment by {self.author} on {self.post}"