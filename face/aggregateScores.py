# aggregate_scores.py
import os
import json
import cv2
import math

def aggregate_scores(scores_file, video_path, output_file, fps_override=None):
    """
    Lit le fichier JSON contenant les scores des visages et regroupe ces scores par seconde.
    
    Pour chaque seconde de la vidéo, si aucun score n'est présent (aucun visage détecté),
    le score moyen est fixé à 0. Le résultat est écrit dans output_file sous le format :
        [score_moyen à 2 décimales], [seconde]sec

    Exemple :
        9.00, 0sec
        7.50, 1sec
        0.00, 2sec
        ...
    """
    # Charger les scores depuis le fichier JSON
    with open(scores_file, 'r') as f:
        scores = json.load(f)
    
    # Récupérer le fps et le nombre total de frames depuis la vidéo
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("Erreur lors de l'ouverture de la vidéo.")
        return
    fps = fps_override if fps_override is not None else cap.get(cv2.CAP_PROP_FPS)
    total_frames = cap.get(cv2.CAP_PROP_FRAME_COUNT)
    total_seconds = math.ceil(total_frames / fps)
    cap.release()
    
    # Constitution d'un dictionnaire : seconde -> liste de scores
    # Le nom des fichiers est supposé être du type "imageXXXXXXfaceY.png"
    second_scores = {}
    for file_name, score in scores.items():
        try:
            if "face" in file_name:
                parts = file_name.split("face")
                frame_part = parts[0]  # Exemple : "image002450"
                frame_number_str = frame_part.replace("image", "")
                frame_number = int(frame_number_str)
            else:
                continue
        except Exception as e:
            print(f"Erreur lors du traitement du nom {file_name}: {e}")
            continue

        second_index = int(frame_number / fps)
        second_scores.setdefault(second_index, []).append(score)
    
    # Écriture du fichier final : une ligne par seconde, même si aucune image n'a été traitée à ce moment-là
    with open(output_file, 'w') as f_out:
        for sec in range(total_seconds):
            if sec in second_scores:
                scores_list = second_scores[sec]
                avg_score = sum(scores_list) / len(scores_list)
            else:
                avg_score = 0.0
            f_out.write(f"{avg_score:.2f}, {sec}sec\n")
    
    print(f"Scores agrégés sur {total_seconds} secondes. Résultat enregistré dans {output_file}.")

if __name__ == '__main__':
    scores_file = "face_scores.json"
    video_path = "../bigclip.mp4"
    output_file = "scored_intervals.txt"
    aggregate_scores(scores_file, video_path, output_file)
