import cv2
import mediapipe as mp
import pyautogui
import streamlit as st
from streamlit_webrtc import VideoTransformerBase, webrtc_streamer

# Initialize FaceMesh
face_mesh = mp.solutions.face_mesh.FaceMesh(refine_landmarks=True)
screen_w, screen_h = pyautogui.size()

class VideoTransformer(VideoTransformerBase):
    def __init__(self):
        self.face_mesh = mp.solutions.face_mesh.FaceMesh(refine_landmarks=True)
    
    def transform(self, frame):
        frame = frame.to_ndarray(format="bgr24")
        frame = cv2.flip(frame, 1)  # Flip the frame horizontally
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)  # Convert frame to RGB

        # Process the frame to get the landmarks
        output = self.face_mesh.process(rgb_frame)
        landmarks_points = output.multi_face_landmarks
        frame_h, frame_w, _ = frame.shape

        if landmarks_points:
            landmarks = landmarks_points[0].landmark
            for id, landmark in enumerate(landmarks[474:478]):
                # Convert normalized landmark coordinates to pixel coordinates
                x = int(landmark.x * frame_w)
                y = int(landmark.y * frame_h)
                cv2.circle(frame, (x, y), 3, (0, 255, 0), -1)

                if id == 1:
                    # Map normalized coordinates to screen coordinates
                    screen_x = int(landmark.x * screen_w)
                    screen_y = int(landmark.y * screen_h)
                    pyautogui.moveTo(screen_x, screen_y)

            # Define the landmarks for the left eye
            left = [landmarks[145], landmarks[159]]
            for landmark in left:
                x = int(landmark.x * frame_w)
                y = int(landmark.y * frame_h)
                cv2.circle(frame, (x, y), 3, (0, 255, 255), -1)

            # Check for blink (vertical distance between two points is small)
            if (left[0].y - left[1].y) < 0.014:
                pyautogui.click()
                pyautogui.sleep(1)

        return frame

# Custom CSS for the webpage
st.markdown("""
    <style>
        body {
            background-color: #1e3a5f;
            color: #1e3a5f;
            font-family: Arial, sans-serif;
        }
        .stApp {
            background-color: #1e3a5f;
            border-radius: 15px;
            padding: 20px;
            box-shadow: 0 4px 20px rgba(0, 0, 0, 0.1);
            max-width: 800px;
            margin: auto;
        }
        .title {
            color: #ffffff; 
            text-align: center;
            font-size: 36px;
            font-weight: bold;
            margin-bottom: 20px;
        }
        .description {
            color: #dfe6e9;
            text-align: center;
            font-size: 18px;
            margin-bottom: 40px;
        }
        .footer {
            color: #dfe6e9;
            text-align: center;
            font-size: 14px;
            margin-top: 40px;
        }
        .main-content {
            padding: 20px;
            border: 1px solid #ffffff;
            border-radius: 10px;
            background-color: #4a6572;
        }
        .footer a {
            color: #ffffff;
            text-decoration: none;
            font-weight: bold;
        }
    </style>
""", unsafe_allow_html=True)

# Streamlit interface
st.markdown('<div class="title">Eye Controlled Mouse</div>', unsafe_allow_html=True)
st.markdown('<div class="description">Move the mouse cursor using your eyes. Blink to click.</div>', unsafe_allow_html=True)

# Main content area with a border and background color
st.markdown('<div class="main-content">', unsafe_allow_html=True)
webrtc_streamer(key="example", video_transformer_factory=VideoTransformer)
st.markdown('</div>', unsafe_allow_html=True)

# Footer
st.markdown('<div class="footer">Developed with ❤️ by <a href="https://yourwebsite.com" target="_blank">Your Name</a></div>', unsafe_allow_html=True)
