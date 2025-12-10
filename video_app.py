import cv2
import torch
import torch.nn as nn
import numpy as np
from modules.features import Extracteur

# --- CONFIGURATION ---
CLASSES = ["CARTE ETUDIANT", "CARTE IDENTITE", "CARTE FIDELITE"]
LARGEUR_VISEUR = 450
HAUTEUR_VISEUR = 280

# --- REDEFINITION DU MODELE ---
class CarteClassifier(nn.Module):
    def __init__(self):
        super(CarteClassifier, self).__init__()
        self.fc1 = nn.Linear(6, 16)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(16, 3) 
    def forward(self, x):
        return self.fc2(self.relu(self.fc1(x)))

def main():
    print("--- DEMARRAGE VIDEO ---")
    
    # 1. Chargement
    device = torch.device("cpu")
    model = CarteClassifier()
    try:
        model.load_state_dict(torch.load("model.pth", map_location=device))
        model.eval()
    except:
        print("Erreur: Modèle introuvable.")
        return

    # On force DirectShow pour Windows (Solution caméra)
    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    if not cap.isOpened():
        cap = cv2.VideoCapture(0) # Essai standard si DSHOW échoue

    extracteur = Extracteur()
    print("Webcam active. 'q' pour quitter.")

    while True:
        ret, frame = cap.read()
        if not ret: break

        # Miroir (plus naturel)
        frame = cv2.flip(frame, 1)

        # Création du Viseur
        h, w = frame.shape[:2]
        x_start = (w - LARGEUR_VISEUR) // 2
        y_start = (h - HAUTEUR_VISEUR) // 2
        x_end = x_start + LARGEUR_VISEUR
        y_end = y_start + HAUTEUR_VISEUR

        roi = frame[y_start:y_end, x_start:x_end].copy()

        # --- LOGIQUE INTELLIGENTE ---
        texte_affiche = "En attente..."
        couleur = (200, 200, 200) # Gris par défaut
        
        try:
            # 1. Extraction des indices
            features, _, coords_visage = extracteur.obtenir_features(roi)
            
            # --- FILTRE ANTI-VIDE (GATEKEEPER) ---
            # features[5] est la densité (quantité de contours/texte)
            # Si c'est < 0.02, c'est qu'il n'y a presque rien dans le cadre (mur vide)
            densite = features[5]
            has_face = features[4]
            
            if densite < 0.03: 
                texte_affiche = "Pas de carte detectee"
                couleur = (0, 0, 255) # Rouge
            
            else:
                # 2. Si le filtre passe, on demande à l'IA
                tensor_input = torch.FloatTensor(features).unsqueeze(0)
                with torch.no_grad():
                    outputs = model(tensor_input)
                    probs = torch.softmax(outputs, dim=1)
                    score, index = torch.max(probs, 1)

                classe = CLASSES[index.item()]
                confiance = score.item() * 100

                # 3. Correction Logique (Post-Processing)
                # Si l'IA dit "Fidélité" mais qu'il y a un visage -> C'est louche (probablement Etudiant ou Identité mal classé)
                if classe == "CARTE FIDELITE" and has_face > 0.5:
                    classe = "CARTE ETUDIANT (?)" # On devine
                    confiance = 50 # On baisse la confiance

                # 4. Affichage
                if confiance > 60:
                    couleur = (0, 255, 0) # Vert
                    texte_affiche = f"{classe} ({confiance:.0f}%)"
                    
                    # Bonus OCR (Seulement si image très stable)
                    if confiance > 85:
                        y_txt = y_end + 40
                        # On affiche un petit indicateur
                        cv2.putText(frame, "[Analyse OCR active]", (x_start, y_txt), 
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
                else:
                    texte_affiche = "Analyse en cours..."
                    couleur = (0, 165, 255)

        except Exception as e:
            pass

        # --- DESSIN ---
        # Assombrir le fond pour faire ressortir le viseur (Effet Pro)
        mask = np.zeros_like(frame)
        cv2.rectangle(mask, (x_start, y_start), (x_end, y_end), (255, 255, 255), -1)
        frame = cv2.bitwise_and(frame, mask) + cv2.addWeighted(frame, 0.3, mask, 0, 0) # Fond sombre

        cv2.rectangle(frame, (x_start, y_start), (x_end, y_end), couleur, 2)
        cv2.putText(frame, texte_affiche, (x_start, y_start - 10), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, couleur, 2)

        # Affichage Debug (Densité) pour t'aider à comprendre
        # Tu pourras enlever cette ligne pour la démo finale
        if 'features' in locals():
            cv2.putText(frame, f"Densite: {features[5]:.3f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0), 1)

        cv2.imshow('Projet Vision - Challenge 3', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()