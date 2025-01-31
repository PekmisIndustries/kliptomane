print("librairies > importation > début")
import time
start_time = time.time()
from moviepy.editor import VideoFileClip, AudioFileClip
import whisper
import os
import sys
from transformers import WhisperProcessor, WhisperForConditionalGeneration
import torch
import torchaudio
import threading


from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
print("librairies > importation > fin")
print(f"Temps : {time.time() - start_time:.2f} secondes\n")
last_time = time.time()

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
    sys.stdout.write(f"\[{arrow}{padding}] {percent}%")
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
    video = VideoFileClip(video_path)
    audio = video.audio
    audio.write_audiofile(output_audio_path)
    video.close()

# Function to transcribe audio using Whisper

def transcribe_audio(audio_path, model_name):
    import threading
    import time
    from moviepy.editor import AudioFileClip
    from transformers import pipeline

    # Pour la barre de progression
    global done
    done = False

    def progress_bar(current, total, bar_length=30):
        fraction = current / total
        fraction = min(max(fraction, 0.0), 1.0)
        arrow_count = int(fraction * bar_length - 1)
        arrow_count = max(0, arrow_count)
        arrow = '=' * arrow_count + '>'
        padding = ' ' * (bar_length - len(arrow))
        percent = int(fraction * 100)
        sys.stdout.write(f"\r[{arrow}{padding}] {percent}%")
        sys.stdout.flush()

    def progress_bar_thread(estimated_time, poll_interval=0.5):
        start = time.time()
        while not done:
            elapsed = time.time() - start
            fraction = elapsed / estimated_time
            progress_bar(fraction, 1)
            if fraction >= 1.0:
                break
            time.sleep(poll_interval)
        # Forcer l'affichage final à 100%
        progress_bar(1, 1)
        print()

    print(f"chargement du modèle > début")

    # Mapping entre le nom de modèle Whisper (OpenAI) et celui hébergé sur Hugging Face
    hf_model_map = {
        "large": "openai/whisper-large",
        "medium": "openai/whisper-medium",
        "small": "openai/whisper-small",
        "base": "openai/whisper-base",
        "tiny": "openai/whisper-tiny",
    }
    hf_model_name = hf_model_map.get(model_name, "openai/whisper-base")

    print(f"chargement du modèle > [{hf_model_name}]")

    # Chargement de la pipeline Hugging Face (Whisper)
    # - return_timestamps=True permet de récupérer les découpes temporelles
    # - chunk_length_s et stride_length_s peuvent être ajustés selon la taille/longueur de l'audio
    asr_pipeline = pipeline(
        task="automatic-speech-recognition",
        model=hf_model_name,
        device=0,  # ou "cuda:0" / -1 pour CPU
        return_timestamps=True,
        chunk_length_s=30,
        stride_length_s=[10, 5]
    )

    print("chargement du modèle > fin\n")
    print("modele > traitement")

    # Calcule la durée de l'audio pour la barre de progression
    audio_clip = AudioFileClip(audio_path)
    audio_duration_sec = audio_clip.duration
    audio_clip.close()

    # Estimation du temps (similaire à ta logique originale)
    if model_name == 'large':
        time_factor = 3.0
    elif model_name == 'medium':
        time_factor = 2.0
    else:
        time_factor = 1.5
    estimated_time = audio_duration_sec * time_factor / 9.0

    # Démarre la barre de progression dans un thread séparé
    progress_thread = threading.Thread(target=progress_bar_thread, args=(estimated_time, 0.3))
    print("modele > traitement > estimé >", round(estimated_time, 0), "secondes")
    progress_thread.start()

    # Effectue la transcription via la pipeline
    output = asr_pipeline(audio_path)

    # Arrête la barre de progression
    done = True
    progress_thread.join()

    # La pipeline Hugging Face Whisper retourne en général un objet du type :
    #
    # {
    #   "text": "...",
    #   "chunks": [
    #       { "timestamp": (start, end), "text": "..." },
    #       { ... }
    #   ]
    # }
    #
    # On convertit ceci en une liste de dictionnaires [ { "start": ..., "end": ..., "text": ... }, ... ]
    segments = []
    if "chunks" in output:
        for i, chunk in enumerate(output["chunks"]):
            start_ts, end_ts = chunk["timestamp"]
            segments.append({
                "id": i,
                "start": start_ts,
                "end": end_ts,
                "text": chunk["text"]
            })
    else:
        # Cas où la pipeline ne renvoie qu'un seul bloc (pas de découpage).
        segments.append({
            "id": 0,
            "start": 0.0,
            "end": audio_duration_sec,
            "text": output["text"]
        })

    return segments


# Function to save transcription with timestamps to a text file
def save_transcription(segments, output_file):
    with open(output_file, "w") as f:
        for segment in segments:
            start_time = segment["start"]
            end_time = segment["end"]
            text = segment["text"]
            f.write(f"[{start_time:.2f} - {end_time:.2f}] {text}\n")


def load_nllb():
    local_model = "translation/models--facebook--nllb-200-distilled-600M"

    tokenizer = AutoTokenizer.from_pretrained(local_model)
    
    # Chargement optimisé en 8-bit (réduit la consommation VRAM)
    model = AutoModelForSeq2SeqLM.from_pretrained(
        local_model, 
        torch_dtype=torch.float32,  # Réduit la mémoire et accélère sur RTX 3080
        device_map="auto"  # Envoie automatiquement sur le GPU
    )

    return model, tokenizer

def translate_nllb(sentences, model, tokenizer, src_lang="fra_Latn", tgt_lang="eng_Latn", batch_size=1):
    """ Traduit une liste de phrases en lot (batch) pour accélérer la traduction """
    translated_texts = []
    
    for i in range(0, len(sentences), batch_size):

        batch = sentences[i:i + batch_size]
        
        if(batch[0] != "agreagreagreagre"):
            print("modele > traduit : ",batch)


        # Vérifie que le batch n'est pas vide
        if not batch or all(not text.strip() for text in batch or batch[0] == "agreagreagreagre"):  # ✅ Évite de traduire un batch vide
            continue
        
        inputs = tokenizer(batch, return_tensors="pt", padding=True, truncation=True).to("cuda")
        model.to("cuda")
        
        with torch.no_grad():  # Désactive le calcul des gradients pour accélérer
            translated = model.generate(**inputs, forced_bos_token_id=tokenizer.convert_tokens_to_ids(tgt_lang))

        translated_texts.extend(tokenizer.batch_decode(translated, skip_special_tokens=True))
    return translated_texts

def clean_text(text):
    cleanSegments = []
    for segment in segments:
        #replace '...' with ''
        text = segment["text"]
        if text == " ...":
            text = "agreagreagreagre"
        cleanSegments.append({"start": segment["start"], "end": segment["end"], "text": text})

    # for segment in cleanSegments:
    #     print(segment['text'])
    
    return cleanSegments



# Paths and parameters
video_path = "clip.mp4"  # Replace with your video path
audio_path = "temp_audio.wav"
output_file = "transcription.txt"
model_name = "large"  # Whisper model size: tiny, base, small, medium, large

# Step 1: Extract audio
print("Isolement de l'audio")
extract_audio_from_video(video_path, audio_path)
print("Audio extrait")

print(f"Temps : {time.time() - last_time:.2f} secondes\n")
last_time = time.time()

# Step 2: Transcribe audio
print("Transcription des voix")
segments = transcribe_audio(audio_path, model_name)
print("Voix transcrites")

print(f"Temps : {time.time() - last_time:.2f} secondes\n")
last_time = time.time()

# Step 3: Clean and normalize text
segments = clean_text(segments)

# Step 4: Translate transcription
model, tokenizer = load_nllb()
sentences = [segment["text"] for segment in segments]
translated_texts = translate_nllb(sentences, model, tokenizer)

print(f"Temps : {time.time() - last_time:.2f} secondes\n")
last_time = time.time()

#print translated texts
for i, text in enumerate(translated_texts):
    segments[i]["text"] = text

# Step 5: Clean and normalize text
for segment in segments:
    text = segment["text"]
    if text == "the agreement":
        text = "PEKMIGNORE"
    segment["text"] = text

# Step 6: Save transcription
save_transcription(segments, output_file)

# Clean up temporary audio file
if os.path.exists(audio_path):
    os.remove(audio_path)




print(f"Transcription terminée > {output_file}")
print(f"Temps total : {time.time() - start_time:.2f} secondes")
time.sleep(5)