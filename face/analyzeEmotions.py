# analyze_emotions.py
import os
import time
import json
import tensorflow as tf
import numpy as np
from tensorflow.keras.models import load_model

def load_emotion_model(model_path='fer2013_mini_XCEPTION.102-0.66.hdf5'):
    """
    Charge le modèle de reconnaissance d’émotion.
    Le modèle attend des images de taille 64x64 en niveaux de gris (forme (64,64,1)).
    """
    print("Chargement du modèle d'émotion...")
    model = load_model(model_path, compile=False)
    return model

def load_and_preprocess_with_filename(filename, target_size=(64, 64)):
    """
    Fonction de mappage pour tf.data.
    - Lit l'image à partir du fichier (PNG, 3 canaux).
    - Convertit en niveaux de gris.
    - Redimensionne à target_size.
    - Normalise les pixels en [0,1].
    Renvoie un tuple (image, nom_du_fichier).
    """
    # Lire le fichier image
    image_string = tf.io.read_file(filename)
    # Décoder l'image PNG en 3 canaux (RGB)
    image = tf.image.decode_png(image_string, channels=3)
    # Conversion en niveaux de gris
    image = tf.image.rgb_to_grayscale(image)  # résultat en (hauteur, largeur, 1)
    # Redimensionnement
    image = tf.image.resize(image, target_size)
    # Normalisation
    image = tf.cast(image, tf.float32) / 255.0
    return image, filename

def analyze_emotions_tfdata(input_folder, output_scores_file, batch_size=64, model_path='fer2013_mini_XCEPTION.102-0.66.hdf5'):
    """
    Parcourt toutes les images de visages dans input_folder en utilisant tf.data pour charger et prétraiter en parallèle,
    effectue la prédiction par lots sur le GPU, et enregistre un dictionnaire { nom_de_fichier: score } dans output_scores_file.
    
    Le score correspond à la probabilité maximale (sur 7 classes) multipliée par 10.
    """
    start_time = time.time()
    model = load_emotion_model(model_path)
    
    # Obtenir la liste triée des chemins de fichiers d'images
    all_files = sorted([
        os.path.join(input_folder, f)
        for f in os.listdir(input_folder) if f.endswith('.png')
    ])
    if not all_files:
        print("Aucun fichier trouvé dans", input_folder)
        return
    
    # Créer le dataset tf.data
    dataset = tf.data.Dataset.from_tensor_slices(all_files)
    # Utiliser le mapping avec parallélisme pour charger et prétraiter les images
    dataset = dataset.map(lambda f: load_and_preprocess_with_filename(f),
                          num_parallel_calls=tf.data.AUTOTUNE)
    dataset = dataset.batch(batch_size)
    dataset = dataset.prefetch(tf.data.AUTOTUNE)
    
    scores = {}
    total_processed = 0
    
    # Itérer sur les lots
    for batch in dataset:
        images, filenames = batch  # images: (batch_size, 64, 64, 1), filenames: (batch_size,)
        # Prédiction par lot sur le GPU
        preds = model.predict(images, verbose=0)
        # Pour chaque image du lot, récupérer la probabilité maximale (sur 7 classes)
        for pred, fname in zip(preds, filenames.numpy()):
            max_prob = float(np.max(pred))
            # Convertir le nom de fichier (bytes) en string et ne garder que le nom de fichier sans le chemin complet
            fname_str = os.path.basename(fname.decode('utf-8'))
            scores[fname_str] = max_prob * 10  # Échelle sur 10
        total_processed += images.shape[0]
        if total_processed % (batch_size * 1) == 0:
            print(f"Traité {total_processed} images sur {len(all_files)}. vitesse = {total_processed / (time.time() - start_time):.2f} visages/s", end='\r', flush=True)
    
    # Sauvegarde des scores dans le fichier JSON
    with open(output_scores_file, 'w') as f:
        json.dump(scores, f, indent=4)
    print(f"Analyse d'émotion terminée pour {len(scores)} images. Résultat enregistré dans {output_scores_file}.")

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(
        description="Analyse des émotions sur des images de visage via tf.data (optimisé pour GPU et parallélisme)"
    )
    parser.add_argument("--input_folder", type=str, default="temp_image/processing", help="Dossier contenant les images de visages")
    parser.add_argument("--output_file", type=str, default="face_scores.json", help="Fichier JSON de sortie pour les scores")
    parser.add_argument("--batch_size", type=int, default=2048, help="Taille de lot pour le traitement")
    parser.add_argument("--model_path", type=str, default="fer2013_mini_XCEPTION.102-0.66.hdf5", help="Chemin vers le modèle d'émotion")
    args = parser.parse_args()
    
    # Vérification que TensorFlow voit bien un GPU
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        print("GPU(s) détecté(s) :", gpus)
    else:
        print("Aucun GPU détecté, vérifiez votre installation de TensorFlow GPU.")
    
    analyze_emotions_tfdata(args.input_folder, args.output_file, args.batch_size, args.model_path)
