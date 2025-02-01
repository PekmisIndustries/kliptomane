# aggregate_scores.py
import os
import json
import cv2

def aggregate_scores(scores_file, video_path, output_file, fps_override=None):
    """Lit le fichier des scores (JSON), regroupe les scores par seconde et calcule la moyenne.
    
    Le résultat est écrit dans output_file avec le format :
        [score_moyen: 2 décimales], [seconde]sec
    Par exemple :
        9.00, 56sec
    """
    with open(scores_file, 'r') as f:
        scores = json.load(f)
    
    # Récupérer le FPS de la vidéo (ou utiliser fps_override si déjà connu)
    if fps_override is not None:
        fps = fps_override
    else:
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            print("Erreur lors de l'ouverture de la vidéo.")
            return
        fps = cap.get(cv2.CAP_PROP_FPS)
        cap.release()
    
    # Dictionnaire : seconde -> liste de scores
    second_scores = {}
    # Les noms de fichiers ont le format "imageXXXXXXfaceY.png" où XXXXXX est l'indice de la frame
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

        seconde = int(frame_number / fps)
        second_scores.setdefault(seconde, []).append(score)
    
    # Écriture du fichier final
    with open(output_file, 'w') as f_out:
        for sec in sorted(second_scores.keys()):
            scores_list = second_scores[sec]
            avg_score = sum(scores_list) / len(scores_list)
            f_out.write(f"{avg_score:.2f}, {sec}sec\n")
    
    print(f"Scores agrégés sur {len(second_scores)} secondes. Résultat enregistré dans {output_file}.")

if __name__ == '__main__':
    scores_file = "face_scores.json"
    video_path = "clip.mp4"
    output_file = "scored_intervals.txt"
    aggregate_scores(scores_file, video_path, output_file)
