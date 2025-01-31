import time
import sys
import threading
import os
import torch
import gc

from moviepy.editor import VideoFileClip, AudioFileClip
import whisper

from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

# -- GLOBAL FLAG TO STOP PROGRESS BAR --
done = False

# Simple progress bar utility
def progress_bar(current, total, bar_length=30):
    fraction = current / total
    fraction = min(max(fraction, 0.0), 1.0)  # clamp between 0 and 1
    arrow_count = int(fraction * bar_length - 1)
    if arrow_count < 0:
        arrow_count = 0
    arrow = '=' * arrow_count + '>'
    padding = ' ' * (bar_length - len(arrow))
    percent = int(fraction * 100)
    sys.stdout.write(f"\rProgress: [{arrow}{padding}] {percent}%")
    sys.stdout.flush()

# A separate thread that increments the progress bar based on an estimated duration
def progress_bar_thread(estimated_time, poll_interval=0.5):
    start = time.time()
    while not done:
        elapsed = time.time() - start
        fraction = elapsed / estimated_time
        progress_bar(fraction, 1)  # use fraction as 'current' vs 1 as 'total'
        if fraction >= 1.0:
            break
        time.sleep(poll_interval)
    # Make sure we end at 100%
    progress_bar(1, 1)
    print()

# Function to extract audio from video
def extract_audio_from_video(video_path, output_audio_path):
    print("Extraction de l'audio en cours...")
    video = VideoFileClip(video_path)
    audio = video.audio
    audio.write_audiofile(output_audio_path)
    video.close()
    print("Audio extrait avec succès!")

# Function to transcribe audio using Whisper with time-based estimation
def transcribe_audio(audio_path, model_name):
    print("Chargement du modèle de transcription...")
    model = whisper.load_model(model_name).to("cuda")
    print("Modèle chargé. Démarrage de la transcription...")

    # Pour estimer le temps de transcription, on utilise la durée de l'audio
    audio_clip = AudioFileClip(audio_path)
    audio_duration_sec = audio_clip.duration
    audio_clip.close()

    # Facteur d'estimation pour la vitesse du modèle (empirique, à ajuster)
    if model_name == 'large':
        time_factor = 3.0
    elif model_name == 'medium':
        time_factor = 2.0
    else:
        time_factor = 1.5

    estimated_time = audio_duration_sec * time_factor / 9.0  # 5x real-time factor

    # On lance la barre de progression dans un thread séparé
    progress_thread = threading.Thread(target=progress_bar_thread, args=(estimated_time, 0.3))
    print("Temps estimé pour la transcription:", round( estimated_time ,0), "secondes")
    progress_thread.start()

    # On effectue la transcription
    global done
    result = model.transcribe(audio_path)

    # Quand la transcription est terminée, on arrête la barre
    done = True
    progress_thread.join()

    del model
    torch.cuda.empty_cache()
    gc.collect()

    return result["segments"]

# Function to save transcription with timestamps to a text file
def save_transcription(translated_output, output_file):
    with open(output_file, "w", encoding="utf-8") as f:
        for segment in translated_output:
            start_time_s = segment["start"]
            end_time_s = segment["end"]
            text = segment["text"][0]
            f.write(f"[{start_time_s:.2f} - {end_time_s:.2f}] {text}\n")


def load_nllb():
    local_model = "translation/models--facebook--nllb-200-distilled-600M"

    tokenizer = AutoTokenizer.from_pretrained(local_model)
    
    # Chargement optimisé en 8-bit (réduit la consommation VRAM)
    model = AutoModelForSeq2SeqLM.from_pretrained(
        local_model, 
        torch_dtype=torch.float16,  # Réduit la mémoire et accélère sur RTX 3080
        device_map="auto"  # Envoie automatiquement sur le GPU
    )

    return model, tokenizer

def translate_nllb(sentences, model, tokenizer, src_lang="fra_Latn", tgt_lang="eng_Latn", batch_size=32):
    """ Traduit une liste de phrases en lot (batch) pour accélérer la traduction """
    translated_texts = []
    
    for i in range(0, len(sentences), batch_size):
        batch = sentences[i:i + batch_size]
        
        inputs = tokenizer(batch, return_tensors="pt", padding=True, truncation=True).to("cuda")
        with torch.no_grad():  # Désactive le calcul des gradients pour accélérer
            translated = model.generate(**inputs, forced_bos_token_id=tokenizer.convert_tokens_to_ids(tgt_lang))
        
        translated_texts.extend(tokenizer.batch_decode(translated, skip_special_tokens=True))
    
    return translated_texts



if __name__ == "__main__":
    start_time = time.time()

    # Paramètres
    video_path = "clip.mp4"        # Chemin vers la vidéo
    audio_path = "temp_audio.wav"  # Chemin vers l'audio temporaire
    output_file = "transcription.txt"
    model_name = "large"


    # -- Étape 1: Extraction de l'audio --
    extract_audio_from_video(video_path, audio_path)

    # -- Étape 2: Transcription avec estimation du temps --
    segments = transcribe_audio(audio_path, model_name)

    # -- Étape 3: Nettoyage de la transcription --
    print("Nettoyage de la transcription...")
    for segment in segments:
        if(segment["text"] == "..."):
            segment["text"] = ""

    # -- Étape 4: Traduction de la transcription --
    model, tokenizer = load_nllb()
    print("Traduction de la transcription...")

    translated_output = []
    for segment in segments:
        translated_output.append(translate_nllb(segment["text"], model, tokenizer, src_lang="fra_Latn", tgt_lang="eng_Latn"))


    # -- Étape 5: Formatage  --
    # print("Formatage de la transcription...")
    # for segment in segments:
    #     segment["text"] = segment["text"].replace(" .", ".").replace(" ,", ",").replace(" ?", "?").replace(" !", "!")

    # -- Étape 6: Sauvegarde de la transcription --
    print("Sauvegarde de la transcription...")
    print("DEBUG AVANT save_transcription :", type(translated_output), translated_output[:3])

    save_transcription(translated_output, output_file)

    # -- Étape 7: Nettoyage du fichier audio temporaire
    if os.path.exists(audio_path):
        os.remove(audio_path)

    print(f"La transcription a été sauvegardée dans {output_file}")
    print(f"Temps d'exécution total: {time.time() - start_time:.2f} secondes")