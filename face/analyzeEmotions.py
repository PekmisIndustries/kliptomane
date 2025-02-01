# analyze_emotions.py
import os
import cv2
from fer import FER
import json

def analyze_emotions(input_folder, output_scores_file):
    """Parcourt toutes les images de visages, analyse l'émotion dominante et enregistre le score (multiplié par 10)
    dans un fichier JSON.
    
    Le fichier JSON est un dictionnaire : { nom_de_fichier: score }
    """
    detector = FER(mtcnn=True)  # Utilisation de MTCNN pour une meilleure précision
    face_files = sorted([f for f in os.listdir(input_folder) if f.endswith('.png')])
    scores = {}

    fb = 0
    for face_file in face_files:
        fb += 1
        if fb % 100 == 0:
            print(f"Analyse de l'émotion pour {fb} images de visage...")

        file_path = os.path.join(input_folder, face_file)
        image = cv2.imread(file_path)

        if image is None:
            print(f"Erreur de lecture pour {face_file}, image ignorée.")
            continue

        # Convertir l'image en RGB (FER attend du RGB)
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        result = detector.top_emotion(image_rgb)

        if result is None or result[0] is None or result[1] is None:
            print(f"Aucune émotion détectée dans {face_file}, image ignorée.")
            os.remove(file_path)
            continue

        emotion, score = result
        scaled_score = score * 10  # Mise à l'échelle sur 10
        scores[face_file] = scaled_score

    # Sauvegarde des scores dans un fichier JSON
    with open(output_scores_file, 'w') as f:
        json.dump(scores, f, indent=4)
    
    print(f"Analyse d'émotion terminée pour {len(scores)} images de visage valides.")
    
if __name__ == '__main__':
    input_folder = os.path.join("temp_image", "processing")
    output_scores_file = "face_scores.json"
    analyze_emotions(input_folder, output_scores_file)
