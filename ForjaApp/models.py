from django.db import models
from django.contrib.auth.models import User

from django.db import models

class Movie(models.Model):
    title = models.CharField(max_length=200)
    release_date = models.DateField()
    overview = models.TextField(blank=True, default='')  # Ajoute `blank=True` et `default=''`
    poster_path = models.CharField(max_length=200, blank=True)


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
