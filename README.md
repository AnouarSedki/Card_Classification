# Système d'Analyse et Classification de Documents

Projet académique -- Master 1 Informatique\
La Rochelle Université\
EC 260-1-81 -- Approches Expérimentales

------------------------------------------------------------------------

## Objectif du projet

Développement d'une pipeline complète de Computer Vision capable de :

1.  Classifier automatiquement une carte parmi trois catégories :
    -   Carte Étudiant\
    -   Carte d'Identité\
    -   Carte de Fidélité
2.  Extraire des informations textuelles spécifiques via OCR :
    -   Numéro INE\
    -   Zone MRZ\
    -   Année universitaire\
    -   Identifiants divers

Le projet combine Deep Learning, traitement d'image et OCR dans une
architecture modulaire.

------------------------------------------------------------------------

## Fonctionnalités principales

### Prétraitement d'image

-   Redimensionnement
-   Conversion HSV / niveaux de gris
-   Filtrage Gaussien pour réduction du bruit

### Extraction de caractéristiques (Feature Engineering)

-   Moyennes colorimétriques (RGB + saturation)
-   Détection de visage (Haar Cascade -- OpenCV)
-   Détection de contours (Canny) pour estimation de densité textuelle

### Classification

-   Implémentation d'un Perceptron Multi-Couches (MLP) avec PyTorch
-   Entraînement supervisé sur dataset personnalisé
-   Génération automatique de la courbe de perte

### OCR

-   Intégration de Tesseract OCR
-   Extraction ciblée selon la classe prédite
-   Localisation dynamique des zones d'intérêt

### Visualisation

-   Génération automatique de la courbe d'apprentissage
    (`courbe_perte.png`)
-   Interface interactive pour navigation entre images test

------------------------------------------------------------------------

## Architecture du projet

Classification/ │ ├── README.md ├── train_pytorch.py ├── main.py ├──
model.pth ├── courbe_perte.png ├── haarcascade_frontalface_default.xml │
├── modules/ │ ├── **init**.py │ └── features.py │ └── data/ ├── train/
│ ├── etudiant/ │ ├── identite/ │ └── fidelite/ └── test/

------------------------------------------------------------------------

## Architecture du modèle

Type : Perceptron Multi-Couches (MLP)

Entrée (6 features) : - Moyenne Rouge\
- Moyenne Verte\
- Moyenne Bleue\
- Saturation\
- Présence visage (0/1)\
- Densité textuelle

Couche cachée : - 16 neurones\
- Fonction d'activation ReLU

Sortie : - 3 neurones (probabilités pour Étudiant, Identité, Fidélité)

------------------------------------------------------------------------

## Installation

### Environnement Python

Python 3.8 ou supérieur recommandé.

pip install torch torchvision numpy opencv-python matplotlib pytesseract

### Installation de Tesseract OCR

Installer Tesseract OCR selon votre système d'exploitation.\
Configurer le chemin d'accès si nécessaire dans `modules/features.py`.

------------------------------------------------------------------------

## Utilisation

### 1. Entraînement du modèle

python train_pytorch.py

Résultats générés : - model.pth - courbe_perte.png

### 2. Lancer la démonstration

python main.py

Fonctionnement : - Prédiction de la classe - Détection des zones
d'intérêt - Lecture OCR - Navigation interactive (Espace pour image
suivante, Échap pour quitter)

------------------------------------------------------------------------

## Méthodologie

Approche hybride :

1.  Feature Engineering classique\
2.  Classification par réseau de neurones\
3.  Post-traitement basé sur règles géométriques\
4.  Extraction OCR ciblée

------------------------------------------------------------------------

## Compétences techniques mobilisées

-   Python
-   Deep Learning (PyTorch)
-   Traitement d'image (OpenCV)
-   OCR (Tesseract)
-   Feature Engineering
-   Gestion de dataset
-   Visualisation (Matplotlib)

------------------------------------------------------------------------

## Perspectives d'amélioration

-   Remplacer le MLP par un réseau convolutif (CNN)
-   Augmenter la taille du dataset
-   Déploiement via API (Flask ou FastAPI)
-   Intégration d'un modèle de détection d'objet (YOLO, Faster-RCNN)

------------------------------------------------------------------------

## Auteur

Anouar Sedki\
Master 1 Informatique -- La Rochelle Université
