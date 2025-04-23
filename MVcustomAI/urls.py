from django.urls import path
from MVcustomAI.views import chat_view, login_view, logout_view

urlpatterns = [
    path('', login_view, name='login'),
    path('chat/', chat_view, name='chat'),
    path('logout/', logout_view, name='logout'),
]