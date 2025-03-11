import cv2
import os

def extract_frames(video_path, output_folder, frame_rate=1):
    """
    Extrait des frames de la vidéo d'entrée.

    Paramètres:
    - video_path : chemin du fichier vidéo
    - output_folder : dossier où enregistrer les frames extraites
    - frame_rate : nombre de secondes entre deux extractions (1 = 1 frame par seconde)
    """
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    interval = int(fps * frame_rate)
    frame_count = 0
    saved_count = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if frame_count % interval == 0:
            frame_filename = os.path.join(output_folder, f"frame_{frame_count:06d}.jpg")
            cv2.imwrite(frame_filename, frame)
            saved_count += 1
        frame_count += 1

    cap.release()
    print(f"Extraction terminée : {saved_count} frames enregistrées dans '{output_folder}'.")
