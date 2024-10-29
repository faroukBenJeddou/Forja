from django import forms
from django.contrib.auth.models import User
from django.contrib.auth.forms import UserCreationForm
from ForjaApp.models import Movie, CinemaRating, Rating, Artist, Role
from .models import Genre,Post, Comment
from django.db import models

class UserRegisterForm(UserCreationForm):
    email = forms.EmailField(required=True)

    class Meta:
        model = User
        fields = ['username', 'email', 'password1', 'password2']

class PostForm(forms.ModelForm):
    class Meta:
        model = Post
        fields = ['title', 'content']
        widgets = {
            'content': forms.Textarea(attrs={'class': 'form-control'}),
        }

class CommentForm(forms.ModelForm):
    class Meta:
        model = Comment
        fields = ['content']
        widgets = {
            'content': forms.Textarea(attrs={'class': 'form-control'}),
        }
        
class MovieForm(forms.ModelForm):
    genres = forms.ModelMultipleChoiceField(
        queryset=Genre.objects.all(),
        widget=forms.CheckboxSelectMultiple,  # Or another widget of your choice
        required=True
    )
    class Meta:
        model = Movie
        fields = ['title', 'release_date', 'overview', 'poster_path','genres']


class RatingForm(forms.ModelForm):
    class Meta:
        model = Rating
        fields = ['score', 'review']
        widgets = {
            'score': forms.NumberInput(attrs={'min': 1, 'max': 5}),
            'review': forms.Textarea(attrs={'placeholder': 'Write your review...'}),
        }

class CinemaRatingForm(forms.ModelForm):
    class Meta:
        model = CinemaRating
        fields = ['score', 'review']  # Fields you want the user to fill out
        widgets = {
            'review': forms.Textarea(attrs={'rows': 4, 'placeholder': 'Write your review here...'}),
        }


class GenreForm(forms.ModelForm):
    class Meta:
        model = Genre
        fields = ['name']

class ArtistForm(forms.ModelForm):
    class Meta:
        model = Artist
        fields = ['name', 'biography', 'birth_date', 'nationality', 'death_date', 'image', 'roles']
        widgets = {
            'roles': forms.CheckboxSelectMultiple(),  # Use checkboxes for role selection
        }
class TextInputForm(forms.Form):
    user_input = forms.CharField(label='Entrez votre texte', max_length=1000)
class ImageUploadForm(forms.Form):
    image = forms.ImageField()
class RoleForm(forms.ModelForm):
    class Meta:
        model = Role
        fields = ['title', 'description']
class ChatBot(models.Model):
    user = models.ForeignKey(
        User, on_delete=models.CASCADE, related_name="GeminiUser", null=True
    )
    text_input = models.CharField(max_length=500)
    gemini_output = models.TextField(null=True, blank=True)
    date = models.DateTimeField(auto_now_add=True, blank=True, null=True)

    def __str__(self):
        return self.text_input