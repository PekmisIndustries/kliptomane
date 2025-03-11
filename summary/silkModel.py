import torch
import torch.nn as nn
import torchvision.transforms as transforms
from PIL import Image

class SilkSummarizer(nn.Module):
    def __init__(self, model_path, device='cuda'):
        super(SilkSummarizer, self).__init__()
        self.device = device
        # Ici, nous définissons une architecture simplifiée pour l'exemple.
        # Remplace cette partie par l'architecture réelle du modèle SILK.
        self.model = self._build_model()
        
        # Chargement des poids pré-entraînés
        state_dict = torch.load(model_path, map_location=self.device)
        self.model.load_state_dict(state_dict)
        self.model.to(self.device)
        self.model.eval()

        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        ])

    def _build_model(self):
        """
        Construit un modèle simplifié pour la démonstration.
        Remplace cette méthode par la définition du modèle SILK.
        """
        model = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(16, 1),
            nn.Sigmoid()  # Sortie entre 0 et 1, représentant un score de "pertinence"
        )
        return model

    def analyze_frame(self, frame_path):
        """
        Analyse une frame et retourne un score indiquant sa pertinence.
        """
        image = Image.open(frame_path).convert('RGB')
        input_tensor = self.transform(image).unsqueeze(0).to(self.device)
        with torch.no_grad():
            score = self.model(input_tensor)
        return score.item()

    def analyze_frames(self, frame_paths):
        """
        Analyse une liste de frames et retourne un dictionnaire {chemin_frame: score}.
        """
        scores = {}
        for frame_path in frame_paths:
            score = self.analyze_frame(frame_path)
            scores[frame_path] = score
        return scores
