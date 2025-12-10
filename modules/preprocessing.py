import cv2
import numpy as np

def preparer_image(image, taille_cible=(500, 315)):
    """
    Redimensionne et nettoie l'image pour l'analyse.
    Respecte le ratio standard d'une carte (format ID-1 : 85.60 x 53.98 mm).
    """
    if image is None:
        return None
    
    # 1. Redimensionnement (Standardisation des entrées)
    img_resized = cv2.resize(image, taille_cible)
    
    # 2. Conversion en Gris
    gris = cv2.cvtColor(img_resized, cv2.COLOR_BGR2GRAY)
    
    # 3. Réduction de bruit (Filtre Gaussien)
    # Permet d'éliminer le grain de la photo ou de l'impression
    gris_flou = cv2.GaussianBlur(gris, (5, 5), 0)
    
    return img_resized, gris_flou