import streamlit as st
import cv2
import numpy as np
import pandas as pd
import os

# --- CONFIG ---
TRAINER_FILE = "trainer/trainer.yml"
EXCEL_FILE = "users.xlsx"
CASCADE_PATH = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"

st.set_page_config(page_title="Identity Scanner", layout="centered")
st.title("👤 Scanly")

# --- CAMERA CONTROL ---
system_on = st.toggle("Activate Scanner Camera", value=True)

if not system_on:
    st.info("📷 Camera is OFF.")
elif not os.path.exists(TRAINER_FILE) or not os.path.exists(EXCEL_FILE):
    st.warning("⚠️ System not ready. Admin must register and train first.")
else:
    recognizer = cv2.face.LBPHFaceRecognizer_create()
    recognizer.read(TRAINER_FILE)
    faceCascade = cv2.CascadeClassifier(CASCADE_PATH)
    df = pd.read_excel(EXCEL_FILE)

    img_file = st.camera_input("Scan your face")

    if img_file:
        bytes_data = img_file.getvalue()
        cv2_img = cv2.imdecode(np.frombuffer(bytes_data, np.uint8), cv2.IMREAD_COLOR)
        gray = cv2.cvtColor(cv2_img, cv2.COLOR_BGR2GRAY)
        
        # Improve detection by adjusting scaleFactor and minNeighbors
        faces = faceCascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=6, minSize=(100, 100))

        if len(faces) == 0:
            st.warning("🔍 No face detected. Move closer to the camera.")
        else:
            (x, y, w, h) = max(faces, key=lambda b: b[2] * b[3])
            id_found, confidence = recognizer.predict(gray[y:y+h, x:x+w])

            # --- ACCURACY LOGIC ---
            # 0 is a perfect match. 100+ is a total guess.
            # A threshold of 60-70 provides much higher accuracy (80%+)
            if confidence < 70: 
                user_match = df[df["Internal_ID"] == id_found]
                
                if not user_match.empty:
                    user_data = user_match.iloc[0]
                    st.success(f"✅ Identity Verified (Confidence: {round(100 - confidence, 1)}%)")
                    with st.container(border=True):
                        st.markdown(f"### {user_data['Name']}")
                        st.write(f"**ID:** `{user_data['Display_ID']}`")
                        st.write(f"**Parent:** {user_data['Parent']}")
                        st.write(f"**Phone:** {user_data['Phone']}")
                        st.write(f"**Blood:** :red[{user_data['Blood']}]")
                else:
                    st.error("ID found in model but missing from Excel.")
            else:
                # This handles faces not in your database
                st.error("❌ Face Not Recognized. Please register with Admin.")
