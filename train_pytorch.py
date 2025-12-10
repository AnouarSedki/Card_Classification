import torch
import torch.nn as nn
import torch.optim as optim
import os
import cv2
import numpy as np
import matplotlib.pyplot as plt # Pour le graphique
from modules.features import Extracteur

# --- 1. MODELE ---
class CarteClassifier(nn.Module):
    def __init__(self):
        super(CarteClassifier, self).__init__()
        self.fc1 = nn.Linear(6, 16)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(16, 3) 
    def forward(self, x):
        return self.fc2(self.relu(self.fc1(x)))

# --- 2. DONNEES ---
def charger_donnees(dossier_train):
    X, y = [], []
    classes = ["etudiant", "identite", "fidelite"]
    extracteur = Extracteur()
    print("Chargement des images...")
    for idx, classe in enumerate(classes):
        path_classe = os.path.join(dossier_train, classe)
        if not os.path.exists(path_classe): continue
        for fichier in os.listdir(path_classe):
            img = cv2.imread(os.path.join(path_classe, fichier))
            if img is not None:
                feats, _, _ = extracteur.obtenir_features(img)
                X.append(feats)
                y.append(idx)
    return np.array(X), np.array(y)

# --- 3. ENTRAINEMENT ---
def train():
    X_train, y_train = charger_donnees("data/train")
    if len(X_train) == 0: return

    X_tensor = torch.FloatTensor(X_train)
    y_tensor = torch.LongTensor(y_train)

    model = CarteClassifier()
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.01)

    loss_history = [] # Pour stocker l'évolution de l'erreur

    print("\nDébut de l'entraînement...")
    epochs = 1000
    for i in range(epochs):
        optimizer.zero_grad()
        outputs = model(X_tensor)
        loss = criterion(outputs, y_tensor)
        loss.backward()
        optimizer.step()
        
        loss_history.append(loss.item()) # On enregistre le score

        if i % 100 == 0:
            print(f"Epoch {i}/{epochs} - Loss: {loss.item():.4f}")

    # --- GENERATION DE LA COURBE ---
    plt.figure(figsize=(10, 5))
    plt.plot(loss_history, label='Erreur (Loss)')
    plt.title("Courbe d'apprentissage du Réseau de Neurones")
    plt.xlabel('Epoques')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    plt.savefig("courbe_perte.png") # Sauvegarde l'image
    print("\nGraphique sauvegardé sous 'courbe_perte.png'")

    torch.save(model.state_dict(), "model.pth")
    print("Modèle sauvegardé !")

if __name__ == "__main__":
    train()