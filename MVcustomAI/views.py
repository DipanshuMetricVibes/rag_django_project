from django.shortcuts import render, redirect
from django.http import JsonResponse
from django.contrib.auth import authenticate, logout
from django.views.decorators.csrf import csrf_exempt
import json
import os
import re
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
from google import genai
from typing import List, Dict
from .models import Conversation
from django.contrib.auth.models import User

# CONFIG
PROJECT_ID = "metricvibes-1718777660306"
REGION = "us-central1"
CHUNK_METADATA_DIR = "chunks"
FAISS_INDEX_DIR = "index"
USER_MAPPING_FILE = "user_mapping.json"

chat_histories: Dict[str, List[Dict[str, str]]] = {}  # Store chat histories in memory

# Initialize models
client = genai.Client(vertexai=True, project=PROJECT_ID, location=REGION)
embedding_model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

def load_user_mapping():
    with open(USER_MAPPING_FILE, "r") as f:
        return json.load(f)

def ask_gemini_rag(user_name: str, query: str) -> dict:
    user_mapping = load_user_mapping()
    allowed_reports = user_mapping.get(user_name)
    
    if not allowed_reports:
        return {"error": f"User '{user_name}' not found or not authorized."}

    # Initialize chat history if it doesn't exist
    if user_name not in chat_histories:
        chat_histories[user_name] = []

    # Get existing chat history
    chat_history = chat_histories[user_name]
    
    query_vector = embedding_model.encode([query]).astype("float32")
    all_matches = []

    for report in allowed_reports:
        index_path = os.path.join(FAISS_INDEX_DIR, f"{report}-index.index")
        metadata_path = os.path.join(CHUNK_METADATA_DIR, f"{report}-chunks.txt")

        if not os.path.exists(index_path) or not os.path.exists(metadata_path):
            continue

        index = faiss.read_index(index_path)
        with open(metadata_path, "r", encoding="utf-8") as f:
            chunks_raw = f.read()

        chunks = [chunk.strip() for chunk in chunks_raw.split("--- Chunk ") if chunk.strip()]
        chunks = [re.split(r'\n', c, maxsplit=1)[1] if '\n' in c else c for c in chunks]

        distances, indices = index.search(query_vector, 5)

        for i, idx in enumerate(indices[0]):
            if idx < len(chunks):
                all_matches.append((distances[0][i], chunks[idx], report, idx))

    if not all_matches:
        return {"error": "No matches found in allowed documents."}

    all_matches.sort(key=lambda x: x[0])
    top_matches = all_matches[:5]
    context = "\n\n".join([match[1] for match in top_matches])

    # Construct conversation context
    conversation_context = ""
    if chat_history:
        conversation_context = "Previous conversation:\n"
        for entry in chat_history[-3:]:  # Get last 3 exchanges
            conversation_context += f"Human: {entry['human']}\nAssistant: {entry['ai']}\n\n"

    # Modified prompt that includes conversation history but maintains original style
    prompt = f"""Answer the question considering both previous conversation and context. For general questions, answer naturally without context. For specific data questions, use the context if available or deny if information isn't present.

{conversation_context}
Context:
{context}

User Question:
{query}

Remember to maintain conversation continuity with previous exchanges while following these rules:
1. Answer general knowledge questions naturally
2. Use context only for specific data questions
3. Deny if asked about information not in context
4. Maintain a casual, helpful tone"""

    try:
        response = client.models.generate_content(
            model="gemini-2.0-flash-001",
            contents=[{"role": "user", "parts": [{"text": prompt}]}]
        )
        
        # Store the new exchange in chat history
        chat_histories[user_name].append({
            "human": query,
            "ai": response.text
        })
        
        return {
            "answer": response.text,
            "matches": [
                {
                    "report": match[2],
                    "distance": float(match[0]),
                    "content": match[1][:400]
                } for match in top_matches
            ],
            "chat_history": chat_histories[user_name]  # Include chat history in response
        }
    except Exception as e:
        return {"error": f"Gemini error: {str(e)}"}

@csrf_exempt
def chat_view(request):
    if request.method == 'POST':
        if not request.session.get('username'):
            return JsonResponse({'error': 'Session expired. Please log in again.'}, status=401)
        try:
            data = json.loads(request.body)
            query = data.get('query')
            user_name = request.session.get('username')
            result = ask_gemini_rag(user_name, query)

            # Save to DB
            user = User.objects.get(username=user_name)
            session_key = request.session.session_key or ''
            bot_response = result.get('answer', '')
            Conversation.objects.create(
                user=user,
                session_key=session_key,
                user_message=query,
                bot_response=bot_response
            )

            return JsonResponse(result)
        except Exception as e:
            return JsonResponse({"error": str(e)})
    if not request.session.get('username'):
        return redirect('login')
    return render(request, 'MVcustomAI/chatbot.html', {'username': request.session['username']})

def login_view(request):
    if request.method == 'POST':
        username = request.POST.get('username')
        password = request.POST.get('password')
        # You can use Django's authentication or your own logic
        user = authenticate(request, username=username, password=password)
        if user is not None:
            request.session['username'] = username
            return render(request, 'MVcustomAI/chatbot.html', {'username': username})
        else:
            return render(request, 'MVcustomAI/login.html', {'error': 'Invalid credentials'})
    return render(request, 'MVcustomAI/login.html')

@csrf_exempt
def logout_view(request):
    logout(request)
    request.session.flush()
    return redirect('login')