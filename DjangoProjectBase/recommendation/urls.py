from django.urls import path
from . import views

urlpatterns = [
    path('', views.recommend_movie, name='recommendation'),
]
