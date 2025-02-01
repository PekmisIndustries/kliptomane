# extract_frames.py
import cv2
import os
from sys import argv

def extract_frames(video_path, output_folder):
    """Extrait toutes les frames d'une vidéo et les enregistre sous forme d'images.
    
    Les images sont nommées image000000.png, image000001.png, etc.
    Retourne le FPS de la vidéo.
    """
    os.makedirs(output_folder, exist_ok=True)
    
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("Erreur lors de l'ouverture de la vidéo. extractFrames")
        return None

    fps = cap.get(cv2.CAP_PROP_FPS)
    print(f"FPS de la vidéo : {fps}")
    
    frame_index = 0
    while True:
        if frame_index % 100 == 0:
            print(f"Extraction de l'image {frame_index}...")
        ret, frame = cap.read()
        if not ret:
            break
        filename = os.path.join(output_folder, f"image{frame_index:06d}.png")
        cv2.imwrite(filename, frame)
        frame_index += 1

    cap.release()
    print(f"Extraction terminée : {frame_index} images extraites.")
    return fps

if __name__ == '__main__':
    video_path = argv[1]
    output_folder = os.path.join("temp_image", "extracted")
    extract_frames(video_path, output_folder)
