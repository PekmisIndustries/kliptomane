import os
import glob
import argparse
import torch
import cv2

from videoExtractor import extract_frames
from silkModel import SilkSummarizer

def create_summary_video(frame_scores, output_video, fps=24):
    """
    Crée une vidéo résumé en sélectionnant les frames dont le score est élevé.
    
    Paramètres:
    - frame_scores : dictionnaire {chemin_frame: score}
    - output_video : chemin du fichier vidéo résumé de sortie
    - fps : fréquence d'image de la vidéo résumé
    """
    # Définir un seuil pour la sélection (par exemple, top 30% des scores)
    scores = list(frame_scores.values())
    if not scores:
        print("Aucune frame à traiter.")
        return
    threshold = sorted(scores, reverse=True)[max(1, int(len(scores) * 0.3)) - 1]
    selected_frames = [fp for fp, score in frame_scores.items() if score >= threshold]
    selected_frames.sort()  # On suppose que le nom des frames reflète leur ordre chronologique

    if len(selected_frames) == 0:
        print("Aucune frame sélectionnée pour la vidéo résumé.")
        return

    # Récupérer les dimensions de la première frame
    first_frame = cv2.imread(selected_frames[0])
    height, width, _ = first_frame.shape

    # Initialiser le writer vidéo
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_video, fourcc, fps, (width, height))

    for frame_path in selected_frames:
        frame = cv2.imread(frame_path)
        out.write(frame)

    out.release()
    print(f"Vidéo résumé sauvegardée dans '{output_video}'.")

def main():
    parser = argparse.ArgumentParser(description="Analyse vidéo avec le modèle SILK")
    parser.add_argument("video_path", type=str, help="Chemin vers la vidéo d'entrée")
    parser.add_argument("--frames_folder", type=str, default="extracted_frames", help="Dossier pour les frames extraites")
    parser.add_argument("--model_path", type=str, required=True, help="Chemin vers le modèle pré-entraîné SILK (fichier .pt)")
    parser.add_argument("--output_video", type=str, default="summary_video.mp4", help="Chemin vers la vidéo résumé de sortie")
    parser.add_argument("--frame_rate", type=float, default=1.0, help="Intervalle d'extraction des frames (secondes)")
    args = parser.parse_args()

    # Étape 1 : Extraction des frames
    extract_frames(args.video_path, args.frames_folder, frame_rate=args.frame_rate)

    # Étape 2 : Récupérer les chemins des frames
    frame_paths = sorted(glob.glob(os.path.join(args.frames_folder, "*.jpg")))
    if not frame_paths:
        print("Aucune frame n'a été extraite. Vérifie la vidéo d'entrée et le dossier de sortie.")
        return

    # Étape 3 : Initialisation du modèle SILK
    device = "cuda" if torch.cuda.is_available() else "cpu"
    summarizer = SilkSummarizer(args.model_path, device=device)

    # Étape 4 : Analyse des frames
    print("Analyse des frames en cours...")
    frame_scores = summarizer.analyze_frames(frame_paths)
    for fp, score in frame_scores.items():
        print(f"{fp}: {score:.4f}")

    # Étape 5 : Création de la vidéo résumé
    create_summary_video(frame_scores, args.output_video)

if __name__ == "__main__":
    main()
