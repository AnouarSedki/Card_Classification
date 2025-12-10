import torch
import torch.nn as nn
import cv2
import numpy as np
import os
from modules.features import Extracteur

# --- CONFIGURATION ---
DOSSIER_TEST = "data/test"
CLASSES = ["CARTE ETUDIANT", "CARTE IDENTITE", "CARTE FIDELITE"]

class CarteClassifier(nn.Module):
    def __init__(self):
        super(CarteClassifier, self).__init__()
        self.fc1 = nn.Linear(6, 16)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(16, 3) 
    def forward(self, x):
        return self.fc2(self.relu(self.fc1(x)))

def main():
    if not os.path.exists("model.pth"):
        print("Erreur: Lance 'py train_pytorch.py' d'abord.")
        return

    # Chargement
    device = torch.device("cpu")
    model = CarteClassifier()
    model.load_state_dict(torch.load("model.pth", map_location=device))
    model.eval()

    extracteur = Extracteur()
    fichiers = [f for f in os.listdir(DOSSIER_TEST) if f.lower().endswith(('.jpg', '.png'))]
    
    print(f"\n--- ANALYSE DE {len(fichiers)} DOCUMENTS ---\n")

    for fichier in fichiers:
        chemin = os.path.join(DOSSIER_TEST, fichier)
        img = cv2.imread(chemin)
        if img is None: continue

        # 1. Extraction Features & Localisation Visage
        features, img_annotee, coords_visage = extracteur.obtenir_features(img)
        
        # 2. Prédiction Type
        input_tensor = torch.FloatTensor(features).unsqueeze(0)
        with torch.no_grad():
            outputs = model(input_tensor)
            probs = torch.softmax(outputs, dim=1)
            score, pred = torch.max(probs, 1)

        classe = CLASSES[pred.item()]
        confiance = score.item() * 100

        # 3. Extraction de contenu Spécifique (Le plus du projet)
        infos_extraites, img_finale = extracteur.analyser_contenu_specifique(img_annotee, classe, coords_visage)

        # --- AFFICHAGE TERMINAL PROPRE ---
        print(f"┌──────────────────────────────────────────┐")
        print(f"│ FICHIER : {fichier:<30} │")
        print(f"├──────────────────────────────────────────┤")
        print(f"│ [DETECTION]                              │")
        print(f"│ Type      : {classe:<28} │")
        print(f"│ Confiance : {confiance:.1f}%                          │")
        print(f"│ [CONTENU EXTRAIT]                        │")
        for k, v in infos_extraites.items():
            print(f"│ - {k:<10} : {v:<25} │")
        print(f"└──────────────────────────────────────────┘")
        print("")

        # --- AFFICHAGE VISUEL ---
        color = (0, 255, 0) if confiance > 80 else (0, 165, 255)
        
        # Bandeau titre
        cv2.rectangle(img_finale, (0, 0), (500, 40), (0, 0, 0), -1)
        cv2.putText(img_finale, f"{classe} ({confiance:.0f}%)", (10, 25), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
        
        cv2.imshow("Analyse Intelligente", img_finale)
        
        if cv2.waitKey(0) == 27: break

    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()