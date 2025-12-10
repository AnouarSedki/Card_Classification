import cv2
import numpy as np
import pytesseract

# --- CONFIGURATION WINDOWS ---
# Si tu as installé Tesseract ailleurs, change ce chemin !
chemin_tesseract = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

try:
    pytesseract.pytesseract.tesseract_cmd = chemin_tesseract
except:
    print("Attention : Tesseract introuvable. L'OCR ne marchera pas.")

class Extracteur:
    def __init__(self):
        self.face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

    def ocr_zone(self, image_crop):
        """
        Nettoie une zone d'image et lit le texte avec Tesseract.
        """
        if image_crop.size == 0: return ""
        
        # 1. Passage en gris
        gray = cv2.cvtColor(image_crop, cv2.COLOR_BGR2GRAY)
        
        # 2. Seuillage (Binarisation) : Texte noir sur fond blanc pur
        # C'est CRUCIAL pour que l'OCR ne se trompe pas avec le fond coloré de la carte
        _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

        # 3. Lecture
        # config='--psm 6' assume que c'est un bloc de texte unique (plus fiable pour des numéros)
        texte = pytesseract.image_to_string(thresh, config='--psm 6')
        
        # Nettoyage des caractères bizarres (sauts de ligne, etc)
        return texte.strip().replace("\n", " ")

    def obtenir_features(self, image_brute):
        """
        Extrait les caractéristiques pour le réseau de neurones (Inchangé)
        """
        img = cv2.resize(image_brute, (500, 315))
        gris = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

        # Features Couleur
        moyenne_couleur = cv2.mean(img)
        b, g, r = moyenne_couleur[0]/255.0, moyenne_couleur[1]/255.0, moyenne_couleur[2]/255.0
        sat_mean = cv2.mean(hsv)[1] / 255.0

        # Feature Visage
        faces = self.face_cascade.detectMultiScale(gris, 1.1, 4)
        has_face = 1.0 if len(faces) > 0 else 0.0
        coords_visage = faces[0] if len(faces) > 0 else None

        # Feature Densité
        edges = cv2.Canny(gris, 50, 150)
        nb_pixels_blancs = np.count_nonzero(edges)
        densite = min((nb_pixels_blancs / (edges.shape[0]*edges.shape[1])) * 5, 1.0) 

        features = np.array([b, g, r, sat_mean, has_face, densite], dtype=np.float32)
        return features, img, coords_visage

    def analyser_contenu_specifique(self, image, type_carte, coords_visage):
        """
        Découpe les zones et applique l'OCR.
        """
        infos = {}
        
        # Copie pour ne pas dessiner sur l'image qu'on envoie à l'OCR
        debug_img = image.copy()

        # 1. VISAGE
        if coords_visage is not None:
            x, y, wf, hf = coords_visage
            infos["Photo"] = "Presente"
            cv2.rectangle(image, (x, y), (x+wf, y+hf), (0, 0, 255), 2)
        else:
            infos["Photo"] = "Absente"

        # 2. EXTRACTION SELON LE TYPE + OCR
        try:
            if type_carte == "CARTE ETUDIANT":
                # --- INE (Bas Gauche) ---
                x, y, w, h = 20, 250, 250, 40 # Coordonnées approximatives
                zone_ine = debug_img[y:y+h, x:x+w]
                
                # On lit le texte
                texte_lu = self.ocr_zone(zone_ine)
                infos["OCR (INE)"] = texte_lu if texte_lu else "Non lu"
                
                # Dessin
                cv2.rectangle(image, (x, y), (x+w, y+h), (255, 0, 0), 2)
                cv2.putText(image, "INE", (x, y-5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)

                # --- ANNEE (Milieu Droite) ---
                x2, y2, w2, h2 = 380, 190, 100, 50
                zone_annee = debug_img[y2:y2+h2, x2:x2+w2]
                texte_annee = self.ocr_zone(zone_annee)
                infos["OCR (Annee)"] = texte_annee if texte_annee else "Non lu"
                
                cv2.rectangle(image, (x2, y2), (x2+w2, y2+h2), (0, 255, 255), 2)

            elif type_carte == "CARTE IDENTITE":
                # --- MRZ (Bas) ---
                # La bande MRZ contient Nom/Prénom/Numéro
                x, y, w, h = 20, 240, 460, 60
                zone_mrz = debug_img[y:y+h, x:x+w]
                
                texte_mrz = self.ocr_zone(zone_mrz)
                # On prend juste les 10 premiers caractères pour l'exemple
                infos["OCR (MRZ)"] = texte_mrz[:20] + "..." if len(texte_mrz) > 5 else "Non lu"
                
                cv2.rectangle(image, (x, y), (x+w, y+h), (255, 0, 0), 2)
                cv2.putText(image, "Zone Lecture Optique", (x, y-5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)

            elif type_carte == "CARTE FIDELITE":
                # --- NUMERO CLIENT (Souvent en bas) ---
                x, y, w, h = 50, 200, 400, 100
                zone_client = debug_img[y:y+h, x:x+w]
                texte_client = self.ocr_zone(zone_client)
                infos["OCR (Client)"] = texte_client if texte_client else "Non lu"
                cv2.rectangle(image, (x, y), (x+w, y+h), (0, 255, 0), 2)

        except Exception as e:
            print(f"Erreur OCR: {e}")

        return infos, image