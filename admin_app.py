import streamlit as st
import cv2
import numpy as np
import pandas as pd
import os
from PIL import Image

# --- CONFIG ---
DATASET_DIR = "dataset"
TRAINER_DIR = "trainer"
TRAINER_FILE = "trainer/trainer.yml"
EXCEL_FILE = "users.xlsx"
CASCADE_PATH = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"

os.makedirs(DATASET_DIR, exist_ok=True)
os.makedirs(TRAINER_DIR, exist_ok=True)

def load_excel():
    if not os.path.exists(EXCEL_FILE):
        return pd.DataFrame(columns=["Internal_ID", "Display_ID", "Name", "Parent", "Phone", "Blood"])
    return pd.read_excel(EXCEL_FILE)

def save_to_excel(data):
    df = load_excel()
    new_row = pd.DataFrame([data])
    df = pd.concat([df, new_row], ignore_index=True)
    df.to_excel(EXCEL_FILE, index=False)

st.set_page_config(page_title="Admin Panel", layout="wide")
st.title("📋 User Registration Admin")

tab1, tab2 = st.tabs(["➕ Register User", "🗂️ View Database"])

with tab1:
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("User Details")
        display_id = st.text_input("User ID (e.g., ID-001)")
        name = st.text_input("Full Name")
        parent = st.text_input("Parent/Guardian Name")
        phone = st.text_input("Phone Number")
        blood = st.selectbox("Blood Group", ["A+", "A-", "B+", "B-", "O+", "O-", "AB+", "AB-"])

    with col2:
        st.subheader("Face Capture")
        if st.button("📸 Capture 30 Photos"):
            if not name or not display_id:
                st.error("Name and ID are required!")
            else:
                df = load_excel()
                # Ensures ID matches the next available row
                internal_id = int(df["Internal_ID"].max()) + 1 if not df.empty else 1
                
                cam = cv2.VideoCapture(0)
                detector = cv2.CascadeClassifier(CASCADE_PATH)
                count = 0
                FRAME_WINDOW = st.image([])
                
                while count < 30:
                    ret, frame = cam.read()
                    if not ret: break
                    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                    faces = detector.detectMultiScale(gray, 1.3, 5)
                    for (x,y,w,h) in faces:
                        count += 1
                        # Crucial: ID in filename must match Internal_ID in Excel
                        cv2.imwrite(f"{DATASET_DIR}/User.{internal_id}.{count}.jpg", gray[y:y+h,x:x+w])
                        cv2.rectangle(frame, (x,y), (x+w,y+h), (0,255,0), 2)
                    FRAME_WINDOW.image(frame, channels="BGR")
                
                cam.release()
                save_to_excel({
                    "Internal_ID": internal_id, "Display_ID": display_id, 
                    "Name": name, "Parent": parent, "Phone": phone, "Blood": blood
                })
                st.success(f"Registered {name} as ID {internal_id}!")

    if st.button("🧠 Update Face Model"):
        with st.spinner("Training model..."):
            recognizer = cv2.face.LBPHFaceRecognizer_create()
            detector = cv2.CascadeClassifier(CASCADE_PATH)
            imagePaths = [os.path.join(DATASET_DIR, f) for f in os.listdir(DATASET_DIR)]
            faceSamples, ids = [], []
            for path in imagePaths:
                img = Image.open(path).convert('L')
                img_np = np.array(img, 'uint8')
                id_num = int(os.path.split(path)[-1].split(".")[1])
                faceSamples.append(img_np)
                ids.append(id_num)
            
            if len(ids) > 0:
                recognizer.train(faceSamples, np.array(ids))
                recognizer.write(TRAINER_FILE)
                st.success("✅ Model updated and saved!")
            else:
                st.error("No images found in dataset to train.")

with tab2:
    st.dataframe(load_excel(), use_container_width=True)
