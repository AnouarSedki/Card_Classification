import torch
import torch.nn as nn
import cv2
import numpy as np
import os
from modules.features import Extracteur

# --- CONFIGURATION ---
# On change ici pour pointer vers le dossier d'entraînement
DOSSIER_CIBLE = "data/train" 
CLASSES = ["CARTE ETUDIANT", "CARTE IDENTITE", "CARTE FIDELITE"]

# --- DEFINITION DU MODELE ---
class CarteClassifier(nn.Module):
    def __init__(self):
        super(CarteClassifier, self).__init__()
        self.fc1 = nn.Linear(6, 16)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(16, 3) 
    def forward(self, x):
        return self.fc2(self.relu(self.fc1(x)))

def main():
    print("--- DEMARRAGE SUR LE DOSSIER TRAIN ---")

    # 1. Chargement du modèle
    if not os.path.exists("model.pth"):
        print("ERREUR: Lance d'abord 'py train_pytorch.py' !")
        return

    device = torch.device("cpu")
    model = CarteClassifier()
    try:
        model.load_state_dict(torch.load("model.pth", map_location=device))
    except:
        print("Erreur modèle. Relance l'entraînement.")
        return
        
    model.eval()
    print("Modèle chargé.")

    # 2. Récupération des images (Récursive)
    extracteur = Extracteur()
    fichiers_trouves = []

    print(f"Scan du dossier : {DOSSIER_CIBLE}...")
    # os.walk permet de descendre dans les sous-dossiers (etudiant, identite...)
    for root, dirs, files in os.walk(DOSSIER_CIBLE):
        for file in files:
            if file.lower().endswith(('.jpg', '.jpeg', '.png')):
                # On stocke le chemin complet
                chemin_complet = os.path.join(root, file)
                fichiers_trouves.append(chemin_complet)
    
    if not fichiers_trouves:
        print(f"Aucune image trouvée dans {DOSSIER_CIBLE}.")
        return

    print(f"\n{len(fichiers_trouves)} images trouvées. Appuie sur ESPACE pour défiler.\n")

    # 3. Boucle d'analyse
    for path in fichiers_trouves:
        img = cv2.imread(path)
        if img is None: continue

        # A. EXTRACTION (Avec les 3 valeurs retournées)
        features, img_base, coords_visage = extracteur.obtenir_features(img)

        # B. PREDICTION
        input_tensor = torch.FloatTensor(features).unsqueeze(0)
        with torch.no_grad():
            outputs = model(input_tensor)
            probs = torch.softmax(outputs, dim=1)
            score, index = torch.max(probs, 1)

        classe_predite = CLASSES[index.item()]
        confiance = score.item() * 100

        # C. OCR & ANALYSE
        infos, img_finale = extracteur.analyser_contenu_specifique(img_base, classe_predite, coords_visage)

        # D. AFFICHAGE
        nom_fichier = os.path.basename(path)
        dossier_parent = os.path.basename(os.path.dirname(path)) # ex: 'etudiant'

        print(f"┌──────────────────────────────────────────────────┐")
        print(f"│ Fichier : {nom_fichier:<38} │")
        print(f"│ Dossier : {dossier_parent:<38} │") # Pour vérifier si ça correspond
        print(f"├──────────────────────────────────────────────────┤")
        print(f"│ DETECTION IA : {classe_predite:<33} │")
        print(f"│ Confiance    : {confiance:.1f}%                          │")
        print(f"├──────────────────────────────────────────────────┤")
        for k, v in infos.items():
            print(f"│ {k:<12} : {v:<35} │")
        print(f"└──────────────────────────────────────────────────┘")

        # Titre sur l'image
        color = (0, 255, 0) if confiance > 80 else (0, 165, 255)
        cv2.rectangle(img_finale, (0, 0), (500, 60), (0, 0, 0), -1)
        cv2.putText(img_finale, f"{classe_predite}", (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
        cv2.putText(img_finale, f"Source: {dossier_parent}", (10, 50), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
        
        cv2.imshow("Demo IA + OCR", img_finale)
        
        key = cv2.waitKey(0)
        if key == 27: # Echap
            break

    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()