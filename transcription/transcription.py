#!/usr/bin/env python
"""
Script 1 : Extraction de l’audio et transcription avec Whisper
Usage (en ligne de commande) : 
    python transcribe_audio.py [chemin_vers_video]
Si aucun argument n’est fourni, le script utilise "clip.mp4" dans le dossier courant.
La transcription est sauvegardée dans "transcription.json" et "transcription.txt".
"""

import time
import threading
import sys
import os
import json
from moviepy.editor import VideoFileClip, AudioFileClip
import whisper

print("librairies > importation > début")
start_time = time.time()
print("librairies > importation > fin")
print(f"Temps : {time.time() - start_time:.2f} secondes\n")
last_time = time.time()

# Variable globale pour la progress bar
done = False

# === UTILITAIRE : Barre de progression ===
def progress_bar(current, total, bar_length=100):
    fraction = current / total
    fraction = min(max(fraction, 0.0), 1.0)  # Contraindre entre 0 et 1
    arrow_count = int(fraction * bar_length - 1)
    if arrow_count < 0:
        arrow_count = 0
    arrow = '=' * arrow_count + '>'
    padding = ' ' * (bar_length - len(arrow))
    percent = round(fraction * 100, 2)
    sys.stdout.write(f"\r[{arrow}{padding}] {percent}%")
    sys.stdout.flush()

def progress_bar_thread(estimated_time, poll_interval):
    start = time.time()
    while not done:
        elapsed = time.time() - start
        fraction = elapsed / estimated_time
        progress_bar(fraction, 1)  # On considère 1 comme total pour utiliser la fraction directement
        if fraction >= 1.0:
            break
        time.sleep(poll_interval)
    progress_bar(1, 1)
    print()

# === FONCTIONS D'EXTRACTION ET DE TRANSCRIPTION ===
def extract_audio_from_video(video_path, output_audio_path):
    print("Extraction de l'audio depuis la vidéo...")
    video = VideoFileClip(video_path)
    audio = video.audio
    audio.write_audiofile(output_audio_path)
    video.close()
    print("Audio extrait.")

def transcribe_audio(audio_path, model_name):
    print(f"Chargement du modèle Whisper > début")
    print(f"Chargement du modèle Whisper > [{model_name}]")
    model = whisper.load_model(model_name)
    print("Chargement du modèle Whisper > fin")
    print("Transcription en cours...")

    # Estimation de la durée de transcription (l'estimation dépend du modèle et de la durée de l'audio)
    audio_clip = AudioFileClip(audio_path)
    audio_duration_sec = audio_clip.duration
    audio_clip.close()
    if model_name == 'large':
        time_factor = 3.0
    elif model_name == 'medium':
        time_factor = 2.0
    else:
        time_factor = 1.5
    estimated_time = audio_duration_sec * time_factor / 8.0
    progress_thread = threading.Thread(target=progress_bar_thread, args=(estimated_time, 0.1))
    print("Transcription estimée :", round(estimated_time, 0), "secondes")
    progress_thread.start()

    result = model.transcribe(audio_path)

    global done
    done = True

    time.sleep(1)
    return result["segments"]

def save_transcription_json(segments, output_json_file):
    with open(output_json_file, "w", encoding="utf-8") as f:
        json.dump(segments, f, indent=4, ensure_ascii=False)
    print(f"Transcription sauvegardée dans {output_json_file}")

def save_transcription_text(segments, output_txt_file):
    with open(output_txt_file, "w", encoding="utf-8") as f:
        for segment in segments:
            start_seg = segment["start"]
            end_seg = segment["end"]
            text = segment["text"]
            f.write(f"[{start_seg:.2f} - {end_seg:.2f}] {text}\n")
    print(f"Transcription texte sauvegardée dans {output_txt_file}")

# === MAIN ===
if __name__ == '__main__':
    # Récupération du chemin de la vidéo (argument en ligne de commande ou valeur par défaut)
    if len(sys.argv) > 1:
        video_path = sys.argv[1]
    else:
        video_path = "../clip.mp4"  # Fichier vidéo par défaut

    audio_path = "temp_audio.wav"
    output_json_file = "transcription.json"
    output_txt_file = "transcription.txt"
    model_name = "large"  # Options possibles : tiny, base, small, medium, large

    # Étape 1 : Extraire l'audio
    extract_audio_from_video(video_path, audio_path)
    print(f"Temps après extraction audio : {time.time() - last_time:.2f} secondes\n")
    last_time = time.time()

    # Étape 2 : Transcrire l'audio avec Whisper
    segments = transcribe_audio(audio_path, model_name)
    print("Transcription terminée.")
    print(f"Temps après transcription : {time.time() - last_time:.2f} secondes\n")
    last_time = time.time()

    # Étape 3 : Sauvegarder la transcription (JSON et texte)
    save_transcription_json(segments, output_json_file)
    save_transcription_text(segments, output_txt_file)

    # Nettoyage du fichier audio temporaire
    if os.path.exists(audio_path):
        os.remove(audio_path)

    print(f"Transcription complète. Temps total : {time.time() - start_time:.2f} secondes")
