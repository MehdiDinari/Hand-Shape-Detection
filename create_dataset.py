import os
import pickle
import mediapipe as mp
import cv2

# Initialisation de Mediapipe
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=True, min_detection_confidence=0.3)

DATA_DIR = './data'

data = []
labels = []

# Vérifier si le dossier DATA_DIR existe
if not os.path.exists(DATA_DIR):
    print(f"Erreur : Le dossier {DATA_DIR} n'existe pas.")
    exit()

for dir_ in os.listdir(DATA_DIR):
    path = os.path.join(DATA_DIR, dir_)

    # Vérifier si c'est un dossier
    if not os.path.isdir(path):
        continue  # Ignore les fichiers comme .gitignore

    print(f"Traitement de la classe : {dir_}")

    for img_path in os.listdir(path):
        data_aux = []
        x_ = []
        y_ = []

        # Charger l'image
        img = cv2.imread(os.path.join(path, img_path))
        if img is None:
            print(f"Erreur : Impossible de charger {img_path}. Fichier corrompu ou chemin invalide.")
            continue

        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        results = hands.process(img_rgb)

        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                for i in range(len(hand_landmarks.landmark)):
                    x = hand_landmarks.landmark[i].x
                    y = hand_landmarks.landmark[i].y

                    x_.append(x)
                    y_.append(y)

                for i in range(len(hand_landmarks.landmark)):
                    x = hand_landmarks.landmark[i].x
                    y = hand_landmarks.landmark[i].y
                    data_aux.append(x - min(x_))
                    data_aux.append(y - min(y_))

            data.append(data_aux)
            try:
                labels.append(int(dir_))  # Convertir le nom du dossier en entier
            except ValueError:
                print(f"Erreur : Le dossier '{dir_}' ne peut pas être converti en entier.")
                continue

# Sauvegarde des données avec Pickle
try:
    with open('data.pickle', 'wb') as f:
        pickle.dump({'data': data, 'labels': labels}, f)
    print("✅ Dataset sauvegardé dans 'data.pickle'.")
except Exception as e:
    print(f"Erreur lors de la sauvegarde du dataset : {e}")
