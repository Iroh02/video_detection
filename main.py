import cv2
import numpy as np
import av
import streamlit as st
from streamlit_webrtc import VideoTransformerBase, webrtc_streamer

# Load built-in Haar cascades
face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)
eye_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_eye.xml"
)

class Transformer(VideoTransformerBase):
    def transform(self, frame: av.VideoFrame) -> av.VideoFrame:
        img = frame.to_ndarray(format="bgr24")
        h, w = img.shape[:2]
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # Detect faces
        faces = face_cascade.detectMultiScale(gray, 1.1, 5)
        for (x, y, fw, fh) in faces:
            # draw face box
            cv2.rectangle(img, (x, y), (x+fw, y+fh), (255, 0, 0), 2)

            # Is the face centered?
            face_cx = x + fw/2
            centered = abs(face_cx - w/2) < fw * 0.25  # within 25% of half-width
            cv2.putText(
                img, f"Centered: {'Yes' if centered else 'No'}",
                (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,255), 2
            )

            # Eyes detection inside face ROI
            roi_gray = gray[y:y+fh, x:x+fw]
            eyes = eye_cascade.detectMultiScale(roi_gray, 1.1, 10, minSize=(30,30))

            # Rough gaze check: require 2 eyes and roughly horizontal and central
            looking = False
            if len(eyes) >= 2:
                # get centers of first two eyes
                eye_centers = [(ex + ew/2, ey + eh/2) for (ex, ey, ew, eh) in eyes[:2]]
                avg_eye_x = sum([c[0] for c in eye_centers]) / 2
                # relative to face box
                looking = abs(avg_eye_x - fw/2) < fw * 0.15
            cv2.putText(
                img, f"Looking: {'Yes' if looking else 'No'}",
                (x, y + fh + 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,255), 2
            )

        return av.VideoFrame.from_ndarray(img, format="bgr24")

def main():
    st.title("📹 Live Face & Gaze Analytics (Haar Cascades)")
    st.write("No Mediapipe → pure OpenCV → zero dependency pain.")
    webrtc_streamer(
        key="face-analysis",
        video_transformer_factory=Transformer,
        media_stream_constraints={"video": True, "audio": False},
        async_processing=True,
        rtc_configuration={"iceServers":[{"urls":["stun:stun.l.google.com:19302"]}]},
    )

if __name__ == "__main__":
    main()
