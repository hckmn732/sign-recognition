import streamlit as st
import cv2
import tempfile
import time
import numpy as np
from keras.models import load_model
from tensorflow.keras.models import load_model
import mediapipe as mp
from ultralytics import YOLO
from sklearn.preprocessing import LabelEncoder

# Charger le modèle entraîné
#model = load_model('best_model_augmented_2.h5')

# Initialisation de MediaPipe
mp_holistic = mp.solutions.holistic
mp_drawing = mp.solutions.drawing_utils

# Extraction des keypoints
def extract_keypoints(image, holistic):
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = holistic.process(image_rgb)
    if results.pose_landmarks:
        return np.array([[lm.x, lm.y, lm.z] for lm in results.pose_landmarks.landmark]).flatten()
    return np.zeros(33 * 3)

# Prédiction avec le modèle
def predict_sign(keypoints):
    keypoints = keypoints.reshape(1, -1)
    prediction = model.predict(keypoints)
    return prediction.argmax(axis=1)[0]

# Config de la page
st.set_page_config(page_title="Reconnaissance de la langue des signes", layout="wide")

# Menu vertical dans la barre latérale
choix = st.sidebar.radio("Menu", ["Accueil", "Reconnaissance de Lettres (Alphabet)", "Reconnaissance de Mots", "Historique", "Modèle", "Contact"])

# Accueil
if choix == "Accueil":
    st.title("🏠 Accueil")
    st.markdown("Bienvenue dans l'Application de Reconnaissance de la Langue des Signes.")
    st.markdown("""
            Cette plateforme utilise l'intelligence artificielle pour détecter et interpréter les gestes de la langue des signes américaine (ASL) à partir de vidéos et de flux en temps réel.
            """)
    

    st.subheader("🔍 Fonctionnalités principales")
    st.markdown("""
    - **Téléversement de vidéos** : Analyse automatique de gestes dans une vidéo MP4.
    - **Reconnaissance en temps réel** : Interprétation des signes via webcam.
    - **Historique**
     : Suivi des prédictions passées.
    - **Informations sur le modèle** : Découvrez le fonctionnement du modèle IA.
    """)

    st.image("image1.gif", caption="Reconnaissance de la langue des signes en action")

    st.info("“La communication est un pont, pas une barrière.”")




# Reconnaissance de Lettres (Alphabet)
elif choix == "Reconnaissance de Lettres (Alphabet)":
    st.subheader("Reconnaissance de l'Alphabet")
    st.markdown("Utilisez une **image, vidéo** ou la **webcam** pour détecter une lettre signée à l’aide du modèle YOLO.")


    # Charger le modèle YOLO
    model_letters = YOLO("DetectionMask.pt")  # Remplace par ton chemin réel

    # Historique des lettres détectées
    if 'lettres_detectees' not in st.session_state:
        st.session_state.lettres_detectees = []

    mode = st.radio("Choisissez un mode :", ["Téléversement Image ou Vidéo", "Webcam Temps Réel"])

    def draw_detections(image, results):
        boxes = results[0].boxes
        for box in boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            cls = int(box.cls[0])
            conf = float(box.conf[0])
            label = model_letters.names[cls]
            st.session_state.lettres_detectees.append(label)
            cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(image, f"{label} {conf:.2f}", (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 0), 2)
        return image

    if mode == "Téléversement Image ou Vidéo":
        media_file = st.file_uploader("Téléversez une **image (jpeg/png)** ou une **vidéo MP4**", type=["jpeg", "png", "mp4"])
        if media_file:
            tfile = tempfile.NamedTemporaryFile(delete=False)
            tfile.write(media_file.read())
            if media_file.type.startswith("image"):
                image = cv2.imread(tfile.name)
                results = model_letters.predict(image)
                annotated = draw_detections(image.copy(), results)
                st.image(annotated, channels="BGR")
            else:
                cap = cv2.VideoCapture(tfile.name)
                stframe = st.empty()
                while cap.isOpened():
                    ret, frame = cap.read()
                    if not ret:
                        break
                    results = model_letters.predict(frame)
                    annotated = draw_detections(frame.copy(), results)
                    stframe.image(annotated, channels="BGR")
                cap.release()

    elif mode == "Webcam Temps Réel":
        run = st.checkbox("Activer la webcam")
        stframe = st.image([])
        cap = cv2.VideoCapture(0)
        while run:
            ret, frame = cap.read()
            if not ret:
                break
            results = model_letters.predict(frame)
            annotated = draw_detections(frame.copy(), results)
            stframe.image(annotated, channels='BGR')
        cap.release()

    # Section historique
    if st.session_state.lettres_detectees:
        dernieres = ' – '.join(st.session_state.lettres_detectees[-10:])
        st.markdown(f"📜 **Dernières lettres détectées :** {dernieres}")


elif choix == "Reconnaissance de Mots":
    st.subheader("Reconnaissance de Mots Signés")
    st.markdown("Téléversez une **vidéo** contenant un ou plusieurs signes correspondant à des mots complets. L'application analysera automatiquement les mouvements et prédira les mots.")

    # Initialiser l'historique
    if 'historique_phrase' not in st.session_state:
        st.session_state.historique_phrase = []

    # Deux colonnes
    col1, col2 = st.columns([2, 1])  # 2 parts pour la vidéo / 1 part pour l'info

    with col1:
        video_file = st.file_uploader("📁 Téléversez votre vidéo (MP4 uniquement)", type=["mp4"])
        
        if video_file:
            # Sauvegarder la vidéo
            tfile = tempfile.NamedTemporaryFile(delete=False)
            tfile.write(video_file.read())
            st.video(tfile.name)

            # Charger modèle et encoder
            model = load_model('best_model_augmented_2.h5')
            label_encoder = LabelEncoder()
            label_encoder.classes_ = np.load('label_encoder_classes.npy')

            # MediaPipe
            mp_hands = mp.solutions.hands
            hands = mp_hands.Hands(static_image_mode=False, max_num_hands=1, min_detection_confidence=0.5)

            def extract_keypoints_from_video(frame):
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                results = hands.process(frame_rgb)
                if results.multi_hand_landmarks:
                    hand_landmarks = results.multi_hand_landmarks[0]
                    keypoints = np.array([[lm.x, lm.y, lm.z] for lm in hand_landmarks.landmark]).flatten()
                    return keypoints
                return None


            def create_sequences(video_path, sequence_length=30):
                cap = cv2.VideoCapture(video_path)
                sequences = []
                frames = []
                while cap.isOpened():
                    ret, frame = cap.read()
                    if not ret:
                        break
                    keypoints = extract_keypoints_from_video(frame)
                    if keypoints is not None:
                        frames.append(keypoints)
                    if len(frames) >= sequence_length:
                        sequences.append(np.array(frames[-sequence_length:]))
                cap.release()
                return np.array(sequences)

            def predict_video(video_path):
                sequences = create_sequences(video_path)
                predictions = model.predict(sequences)
                return predictions

            def interpret_predictions(predictions):
                predicted_classes = np.argmax(predictions, axis=1)
                predicted_labels = label_encoder.inverse_transform(predicted_classes)
                return predicted_labels

            with st.spinner("Analyse en cours..."):
                preds = predict_video(tfile.name)
                labels = interpret_predictions(preds)
                unique_words = list(dict.fromkeys(labels))  # éviter les doublons
                st.session_state.historique_phrase.extend(unique_words)

    with col2:
        st.markdown("### 🧾 Mots détectés :")

        if st.button("🔄 Réinitialiser la phrase"):
            st.session_state.historique_phrase = []
            st.stop()  # ARRÊTER L’EXÉCUTION DU SCRIPT ICI

        if video_file:
            st.write(", ".join(unique_words))

        st.markdown("### 🧩 Phrase en construction :")
        st.write(" ".join(st.session_state.historique_phrase))
    
    # --- Section webcam ---
    # --- Chargement du modèle ---
    @st.cache_resource
    def charger_model():
        model = load_model('best_model_augmented_2.h5')
        label_encoder = LabelEncoder()
        label_encoder.classes_ = np.load('label_encoder_classes.npy')
        return model, label_encoder

    model, label_encoder = charger_model()

    # --- Initialisation MediaPipe ---
    mp_hands = mp.solutions.hands
    hands = mp_hands.Hands(static_image_mode=False, max_num_hands=2, min_detection_confidence=0.5)
    mp_drawing = mp.solutions.drawing_utils

    # --- Initialisation Session State ---
    if 'recording' not in st.session_state:
        st.session_state.recording = False
    if 'sequence' not in st.session_state:
        st.session_state.sequence = []
    if 'predicted_label' not in st.session_state:
        st.session_state.predicted_label = None

    # --- Interface utilisateur ---
    st.markdown("---")
    st.subheader("🎥 Détection des signes en direct")

    start = st.button("🎬 Démarrer")
    stop = st.button("⏹️ Arrêter")

    stframe = st.empty()

    if start:
        st.session_state.recording = True
        st.session_state.sequence = []
        st.session_state.predicted_label = None

    if stop:
        st.session_state.recording = False
        st.session_state.sequence = []
        st.session_state.predicted_label = None

    # --- Webcam logic ---
    if st.session_state.recording:
        cap = cv2.VideoCapture(0)
        prev_time = time.time()

        while st.session_state.recording:
            ret, frame = cap.read()
            if not ret:
                st.error("❌ Impossible d'accéder à la caméra.")
                break

            frame = cv2.flip(frame, 1)
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = hands.process(frame_rgb)

            keypoints = None
            if results.multi_hand_landmarks:
                keypoints = []
                for hand_landmarks in results.multi_hand_landmarks:
                    mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
                    keypoints.extend(np.array([[lm.x, lm.y, lm.z] for lm in hand_landmarks.landmark]).flatten())

            # Sauvegarde des keypoints
            if keypoints is not None and len(keypoints) == 126:
                st.session_state.sequence.append(keypoints)

            # Affichage
            stframe.image(frame, channels="BGR", use_container_width=True)

            # Dès qu'on a 30 frames => faire prédiction
            if len(st.session_state.sequence) == 30:
                sequence_array = np.array(st.session_state.sequence)

                pred = model.predict(np.expand_dims(sequence_array, axis=0))[0]
                label = label_encoder.inverse_transform([np.argmax(pred)])[0]

                st.session_state.predicted_label = label

                st.session_state.recording = False
                break  # sortir de la boucle car prédiction faite

        cap.release()

    # --- Affichage Résultat ---
    if st.session_state.predicted_label:
        st.success(f"🧠 Signe détecté : **{st.session_state.predicted_label}**")




# Historique
elif choix == "Historique":
    st.title("📚 Historique des Prédictions")
    st.info("Fonctionnalité à venir : affichage de l'historique.")

# Informations sur le modèle
elif choix == "Modèle":
    st.title("🧠 À propos du Modèle")
    st.write("Modèle entraîné à partir de keypoints extraits avec MediaPipe Holistic.")
    st.write("Format des données : vecteurs de keypoints 3D (x, y, z) pour chaque image.")

# Contact / À propos
elif choix == "Contact":
    st.title("📨 Contact / À propos")
    st.markdown("Développé par **[Votre Nom]**.")
    st.markdown("Email : votre.email@exemple.com")
    st.markdown("Projet de reconnaissance de la langue des signes américaine.")
