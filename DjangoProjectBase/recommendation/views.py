from django.shortcuts import render
from django.http import HttpResponse
from movie.models import Movie  # Import your Movie model
import numpy as np
import openai
from openai.embeddings_utils import get_embedding, cosine_similarity
from dotenv import load_dotenv
import os

def recommend_movie(request):
    if request.method == 'POST':
        # Get the user's input from the form
        user_input = request.POST.get('user_input')

        # Load the OpenAI API key from the .env file
        _ = load_dotenv('../openAI.env')
        openai.api_key = os.environ['openAI_api_key']

        # Fetch movies from the database
        items = Movie.objects.all()

        # Calculate embeddings and find the most similar movie
        emb_req = get_embedding(user_input, engine='text-embedding-ada-002')
        sim = []
        for item in items:
            emb = item.emb
            emb = list(np.frombuffer(emb))
            sim.append(cosine_similarity(emb, emb_req))
        sim = np.array(sim)
        idx = np.argmax(sim)
        recommended_movie = items[int(idx)]

        # Render a template with the recommended movie
        return render(request, 'recommendation.html', {'recommended_movie': recommended_movie})

    return render(request, 'recommendation.html')
