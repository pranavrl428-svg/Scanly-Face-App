 ## 👤 Scanly - An Emergency Identity Scanner
Scanly is a lightweight, Streamlit-based Face Recognition and Identity Verification system. It uses OpenCV's Local Binary Patterns Histograms (LBPH) for face recognition and a simple Excel database to manage user details.

This project consists of two parts:

- Admin Panel: For registering new users, capturing face datasets, and training the model.

- Identity Scanner: A real-time interface for verifying identities via webcam.

## 🚀 Features
- Real-time Face Detection: Uses Haar Cascades for fast face detection.

- LBPH Recognition: Robust face recognition using cv2.face.LBPHFaceRecognizer.

- Excel Database: Simple backend using .xlsx files—no SQL setup required.

- Streamlit UI: Clean, interactive web interface for both Admin and Scanner modes.

- Confidence Scoring: Displays accuracy percentage for every scan.

- Auto-Directory Management: Automatically handles dataset and model folder creation.

## 🛠️ Tech Stack
Python 3.x

Streamlit (Frontend)

- OpenCV (Computer Vision)

- Pandas (Data Management)

## 📖 Usage Guide
Step 1: Register Users (Admin Panel)
- You must register at least one user before using the scanner.

- Run the Admin app:
- streamlit run admin.py
- Go to the "➕ Register User" tab.

- Fill in the details (Name, ID, Phone, etc.).

- Click "📸 Capture 30 Photos". Look at the camera and rotate your head slightly for better data.
  Once captured, click "🧠 Update Face Model" to train the system.

Step 2: Verify Identity (Scanner)
- Run the Scanner app:
- streamlit run scanner.py
- Toggle "Activate Scanner Camera".
- Click "Scan your face" (Streamlit's camera input).
-If the face matches the trained model, the user's details will appear.  
- Numpy (Array Processing)
