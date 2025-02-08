import pickle
import cv2
import mediapipe as mp
import numpy as np
import streamlit as st

# Charger le modèle entraîné
model_dict = pickle.load(open('./model.p', 'rb'))
model = model_dict['model']

# Initialiser MediaPipe Hands
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
hands = mp_hands.Hands(static_image_mode=False, min_detection_confidence=0.5)

# Dictionnaire des labels
labels_dict = {
    0: 'oui', 1: 'non', 2: 'peut-être', 3: 'haut', 4: 'stop',
    5: 'boule', 6: 'ok', 7: 'droit', 8: 'corne', 9: 'italien'
}

# Interface utilisateur Streamlit
st.set_page_config(page_title="Détecteur de Langue des Signes", layout="wide")
st.title("🖐 Détecteur de Langue des Signes en Temps Réel")
st.markdown("### 📷 Activez votre webcam et testez la reconnaissance des gestes.")

# Vérifier l'état du bouton
if "stop_detection" not in st.session_state:
    st.session_state.stop_detection = False

# Bouton pour arrêter la détection
if st.button("⏹ Arrêter la détection", key="stop_button"):
    st.session_state.stop_detection = True

# Activer la webcam
cap = cv2.VideoCapture(0)
frame_placeholder = st.empty()

# Boucle de capture vidéo
while cap.isOpened() and not st.session_state.stop_detection:
    ret, frame = cap.read()
    if not ret:
        st.error("Erreur de lecture de la webcam. Veuillez vérifier votre caméra.")
        break

    frame = cv2.flip(frame, 1)  # Miroir pour une meilleure expérience utilisateur
    H, W, _ = frame.shape
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Détection des mains
    results = hands.process(frame_rgb)
    data_aux = []
    x_, y_ = [], []

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            # Dessiner les repères
            mp_drawing.draw_landmarks(
                frame, hand_landmarks, mp_hands.HAND_CONNECTIONS,
                mp_drawing_styles.get_default_hand_landmarks_style(),
                mp_drawing_styles.get_default_hand_connections_style()
            )

            # Extraire les coordonnées normalisées
            for landmark in hand_landmarks.landmark:
                x_.append(landmark.x)
                y_.append(landmark.y)

            for landmark in hand_landmarks.landmark:
                data_aux.append(landmark.x - min(x_))
                data_aux.append(landmark.y - min(y_))

        # Prédiction si 42 caractéristiques sont présentes
        if len(data_aux) == 42:
            prediction = model.predict([np.asarray(data_aux)])
            predicted_character = labels_dict[int(prediction[0])]

            # Dessiner la boîte de détection
            x1 = int(min(x_) * W) - 10
            y1 = int(min(y_) * H) - 10
            x2 = int(max(x_) * W) - 10
            y2 = int(max(y_) * H) - 10

            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 4)
            cv2.putText(frame, predicted_character, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 1.3, (0, 255, 0), 3,
                        cv2.LINE_AA)

    # Affichage dans Streamlit
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    frame_placeholder.image(frame, channels="RGB", use_column_width=True)

# Libérer les ressources
cap.release()
cv2.destroyAllWindows()
st.success("Détection arrêtée !")

