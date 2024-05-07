from django.urls import path

from . import views

urlpatterns = [
    path('', views.HomeView),
    path('demo/', views.DemoView, name='demo'),
]

