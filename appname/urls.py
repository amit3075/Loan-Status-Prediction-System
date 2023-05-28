from django.urls import path
from . import views

urlpatterns = [
path('login/', views.LoginPage, name='login'),
path('SignupPage/', views.SignupPage, name='SignupPage'),
path('predict_output/',views.predict_output, name='predict_output')
]
