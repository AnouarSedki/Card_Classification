# Système d'Analyse et Classification de Documents

**Projet de Synthèse - Master 1 Informatique (La Rochelle Université)**
*EC 260-1-81 - Approches Expérimentales*

## Description

Ce projet implémente une chaîne complète de vision par ordinateur ("Computer Vision Pipeline") capable d'analyser une image de carte, de déterminer son type via un **Réseau de Neurones Artificiel (Deep Learning)**, et d'en extraire les informations textuelles spécifiques via **OCR**.

Le système répond aux deux challenges du sujet :

1.  [cite_start]**Extraction d'information :** Localisation de photos, lecture de zones spécifiques (INE, MRZ, etc.)[cite: 48, 51, 52].
2.  [cite_start]**Classification :** Distinction automatique entre 3 classes (Carte Étudiant, Carte d'Identité, Carte de Fidélité)[cite: 54].

-----

## Fonctionnalités Clés

  * [cite_start]**Prétraitement d'image :** Redimensionnement, conversion sémantique (HSV/Gris), et réduction de bruit par filtrage Gaussien[cite: 37, 39, 40].
  * **Extraction de Caractéristiques (Features Engineering) :**
      * Analyse colorimétrique (Moyennes RGB + Saturation).
      * Détection de visage (Algorithme de Viola-Jones / Haar Cascade).
      * Analyse de densité textuelle (Détection de contours Canny).
  * **Classification Intelligente :** Utilisation d'un Perceptron Multi-Couches (MLP) sous **PyTorch**.
  * **Extraction OCR (Tesseract) :** Lecture ciblée des zones d'intérêt (Numéro INE, Année, Zone MRZ) selon la classe prédite.
  * **Visualisation :** Génération automatique de la courbe d'apprentissage (`Loss Curve`).

-----

## Structure du Projet

```text
Projet_Classification/
│
├── README.md                      # Documentation du projet
├── train_pytorch.py               # Script d'entraînement du réseau de neurones
├── main.py                        # Script de démonstration (Inférence + OCR)
├── model.pth                      # Le modèle entraîné (généré après entraînement)
├── courbe_perte.png               # Graphique de performance (généré après entraînement)
├── haarcascade_frontalface_default.xml  # Modèle OpenCV pour les visages
│
├── 📁 modules/                       # Bibliothèque de fonctions
│   ├── __init__.py
│   └── features.py                   # Moteur d'extraction et OCR
│
└── 📁 data/                          # Jeu de données
    ├── train/                        # Images pour l'apprentissage
    │   ├── etudiant/
    │   ├── identite/
    │   └── fidelite/
    └── test/                         # Images pour la validation
```

-----

##  Prérequis et Installation

### 1\. Environnement Python

Le projet nécessite Python 3.8+ et les librairies suivantes :

```bash
pip install torch torchvision numpy opencv-python matplotlib pytesseract
```

### 2\. Moteur OCR (Tesseract)

Ce projet utilise Tesseract pour la lecture de texte.

  * **Télécharger :** [Tesseract-OCR for Windows](https://www.google.com/search?q=https://github.com/UB-Mannheim/tesseract/wiki)
  * **Installation :** Installez-le dans le dossier par défaut (`C:\Program Files\Tesseract-OCR`).
  * **Configuration :** Le chemin est configuré dans `modules/features.py`.

### 3\. Fichiers de données

  * Placez vos images d'entraînement dans `data/train/{classe}`.
  * Placez le fichier `haarcascade_frontalface_default.xml` à la racine.

-----

##  Guide d'Utilisation

### Étape 1 : Entraînement du Modèle 

Avant de classifier, l'IA doit apprendre à partir de vos données.

```powershell
py train_pytorch.py
```

  * **Ce que ça fait :** Scanne le dossier `data/train`, extrait les vecteurs de caractéristiques, entraîne le réseau de neurones sur 1000 époques.
  * **Résultat :** Crée le fichier `model.pth` et le graphique `courbe_perte.png`.

### Étape 2 : Lancement de la Démonstration 

Une fois le modèle entraîné, lancez l'application principale.

```powershell
py main.py
```

  * **Ce que ça fait :** Scanne les images, prédit leur type, localise les informations (Cadres de couleur) et lit le texte (INE, Nom, etc.).
  * **Contrôles :** Appuyez sur `ESPACE` pour passer à l'image suivante, ou `ECHAP` pour quitter.

-----

## Méthodologie Technique

### Architecture du Réseau de Neurones

Le modèle est un **Perceptron Multi-Couches (MLP)** simple mais efficace pour ce volume de données :

  * **Entrée (6 neurones) :** Rouge, Vert, Bleu, Saturation, Présence Visage (0/1), Densité Texte.
  * **Couche Cachée (16 neurones) :** Fonction d'activation ReLU.
  * **Sortie (3 neurones) :** Probabilités pour [Étudiant, Identité, Fidélité].

### Stratégie d'Extraction (Post-Classification)

Une fois la classe déterminée par l'IA, le programme applique des règles géométriques pour l'OCR :

  * **Si Étudiant :** Recherche zone "INE" en bas à gauche + "Année" au milieu droite.
  * **Si Identité :** Recherche zone "MRZ" (bande optique) en bas.
  * **Si Fidélité :** Recherche code barre ou numéro client.

-----
