"""AI_DIESESE_PREDICTION URL Configuration

The `urlpatterns` list routes URLs to views. For more information please see:
    https://docs.djangoproject.com/en/3.2/topics/http/urls/
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

from MYAPP import views

urlpatterns = [
    path('',views.login,name='login'),
    path('login_post',views.login_post,name='login_post'),
    # path('admin_home',views.admin_home,name='admin_home'),
    # path('disease_info',views.disease_info,name='disease_info'),
    # path('symptoms',views.symptoms,name='symptoms'),
    # path('recommendations',views.recommendations,name='recommendations'),
    path('user_home', views.user_home, name='user_home'),
    # path('view_symptoms', views.view_symptoms, name='view_symptoms'),
    path('registration_page',views.registration_page,name='registration_page'),
    path('registration_post', views.registration_post, name='registration_post'),
    path('image_upload', views.image_upload, name='image_upload'),
    path('image_post', views.image_post, name='image_post'),
    path('history_home', views.history_home, name='history_home'),
    path('custom_logout_view', views.custom_logout_view, name='custom_logout_view'),
]
