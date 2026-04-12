import pygame

pygame.mixer.init()
pygame.mixer.music.load("alert.wav")

def play_alert():
    pygame.mixer.music.play()