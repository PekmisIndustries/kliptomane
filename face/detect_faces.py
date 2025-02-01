# detect_faces.py
import os
import cv2
import face_recognition
from concurrent.futures import ThreadPoolExecutor

def process_image(image_file, input_folder, output_folder):
    """Process a single image file to detect and crop faces."""

    image_path = os.path.join(input_folder, image_file)
    image = face_recognition.load_image_file(image_path)
    face_locations = face_recognition.face_locations(image)
    
    if len(face_locations) == 0:
        # Aucune détection de visage > on ignore cette image
        return 0
    
    total_faces = 0
    for i, (top, right, bottom, left) in enumerate(face_locations, start=1):
        face_image = image[top:bottom, left:right]
        # Conversion de RGB vers BGR pour cv2 (pour l'enregistrement)
        face_image = cv2.cvtColor(face_image, cv2.COLOR_RGB2BGR)
        base_filename = image_file.replace(".png", "")
        output_filename = f"{base_filename}face{i}.png"
        output_path = os.path.join(output_folder, output_filename)
        cv2.imwrite(output_path, face_image)
        total_faces += 1
    
    return total_faces

def detect_and_crop_faces(input_folder, output_folder):
    """Parcourt toutes les images d'un dossier, détecte les visages et enregistre chaque visage dans un nouveau fichier.
    
    Exemple : image002450.png contenant 2 visages donnera :
        image002450face1.png et image002450face2.png
    """
    os.makedirs(output_folder, exist_ok=True)
    image_files = sorted([f for f in os.listdir(input_folder) if f.endswith('.png')])
    total_faces = 0

    with ThreadPoolExecutor() as executor:
        futures = [executor.submit(process_image, image_file, input_folder, output_folder) for image_file in image_files]
        for i, future in enumerate(futures, start=1):
            total_faces += future.result()
            if i % 100 == 0:
                print(f"{i} images traitées.")

    print(f"{len(image_files)} images traitées et {total_faces} visages extraits.")

if __name__ == '__main__':
    input_folder = os.path.join("temp_image", "extracted")
    output_folder = os.path.join("temp_image", "processing")
    detect_and_crop_faces(input_folder, output_folder)
