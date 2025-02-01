#!/usr/bin/env python
"""
Script 2 : Traduction de la transcription obtenue par le script 1
Usage (en ligne de commande) :
    python translate_transcription.py
Ce script lit le fichier "transcription.json", traduit chaque segment via NLLB et sauvegarde
la transcription traduite dans "translated_transcription.txt".
"""

import sys
import os
import json
import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

# === FONCTIONS DE TRADUCTION ===
def load_nllb():
    local_model = "translation/models--facebook--nllb-200-distilled-600M"
    tokenizer = AutoTokenizer.from_pretrained(local_model)
    
    # Chargement optimisé en 8-bit (optionnel) et en mode GPU si disponible
    model = AutoModelForSeq2SeqLM.from_pretrained(
        local_model, 
        torch_dtype=torch.float16,
        device_map="auto"
    )
    return model, tokenizer

def translate_nllb(sentences, model, tokenizer, src_lang="fra_Latn", tgt_lang="eng_Latn", batch_size=1):
    """Traduit une liste de phrases en lots (batch) pour accélérer la traduction."""
    translated_texts = []
    for i in range(0, len(sentences), batch_size):
        batch = sentences[i:i + batch_size]
        # Affichage optionnel du lot en cours (sauf si c'est un placeholder)
        if batch and batch[0] != "agreagreagreagre":
            print("Traduction du batch :", batch)
        # Vérification d'un batch non vide
        if not batch or all(not text.strip() for text in batch):
            continue
        
        inputs = tokenizer(batch, return_tensors="pt", padding=True, truncation=True).to("cuda")
        model.to("cuda")
        
        with torch.no_grad():
            translated = model.generate(**inputs, forced_bos_token_id=tokenizer.convert_tokens_to_ids(tgt_lang))
        translated_texts.extend(tokenizer.batch_decode(translated, skip_special_tokens=True))
    
    return translated_texts

def clean_text(segments):
    """Nettoie les textes des segments en remplaçant par exemple les textes '...' par un placeholder."""
    clean_segments = []
    for segment in segments:
        text = segment["text"]
        if text.strip() in ["...", " ..."]:
            text = "agreagreagreagre"
        clean_segments.append({
            "start": segment["start"],
            "end": segment["end"],
            "text": text
        })
    return clean_segments

def save_transcription(segments, output_file):
    with open(output_file, "w", encoding="utf-8") as f:
        for segment in segments:
            start_seg = segment["start"]
            end_seg = segment["end"]
            text = segment["text"]
            f.write(f"[{start_seg:.2f} - {end_seg:.2f}] {text}\n")
    print(f"Transcription traduite sauvegardée dans {output_file}")

# === MAIN ===
if __name__ == '__main__':
    transcription_json = "transcription.json"
    output_file = "transcription.txt"

    # Vérifier que le fichier de transcription existe
    if not os.path.exists(transcription_json):
        print(f"Fichier {transcription_json} non trouvé. Veuillez exécuter d'abord le script de transcription.")
        sys.exit(1)

    # Charger les segments depuis le fichier JSON
    with open(transcription_json, "r", encoding="utf-8") as f:
        segments = json.load(f)

    # Optionnel : nettoyage des textes dans les segments
    segments = clean_text(segments)

    # Extraire la liste des phrases à traduire
    sentences = [segment["text"] for segment in segments]

    print("Chargement du modèle de traduction (NLLB)...")
    model, tokenizer = load_nllb()

    # Traduire les phrases
    translated_texts = translate_nllb(sentences, model, tokenizer)

    # Mettre à jour les segments avec le texte traduit
    for i, text in enumerate(translated_texts):
        segments[i]["text"] = text
        # Exemple d'ajustement éventuel (si le texte traduit vaut "the agreement", on remplace par un autre texte)
        if text == "the agreement":
            segments[i]["text"] = "PEKMIGNORE"

    # Sauvegarder la transcription traduite dans un fichier texte
    save_transcription(segments, output_file)
