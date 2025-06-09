import cv2
import numpy as np
import mediapipe as mp
import av
import streamlit as st
from streamlit_webrtc import VideoTransformerBase, webrtc_streamer

# --- Head-pose estimation helpers ---
# 3D model points of facial landmarks.
MODEL_POINTS = np.array([
    (0.0, 0.0, 0.0),             # Nose tip
    (0.0, -330.0, -65.0),        # Chin
    (-225.0, 170.0, -135.0),     # Left eye left corner
    (225.0, 170.0, -135.0),      # Right eye right corner
    (-150.0, -150.0, -125.0),    # Left Mouth corner
    (150.0, -150.0, -125.0)      # Right mouth corner
], dtype=np.float64)

# Corresponding MediaPipe mesh indices    
LANDMARK_IDS = {
    "nose_tip": 1,
    "chin": 152,
    "left_eye": 33,
    "right_eye": 263,
    "mouth_left": 61,
    "mouth_right": 291,
}

class Transformer(VideoTransformerBase):
    def __init__(self):
        self.face_mesh = mp.solutions.face_mesh.FaceMesh(
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5,
        )
        self.drawing = mp.solutions.drawing_utils

    def transform(self, frame: av.VideoFrame) -> av.VideoFrame:
        img = frame.to_ndarray(format="bgr24")
        h, w, _ = img.shape
        # downscale for speed
        small = cv2.resize(img, (320, 240))
        rgb = cv2.cvtColor(small, cv2.COLOR_BGR2RGB)

        results = self.face_mesh.process(rgb)
        if results.multi_face_landmarks:
            lm = results.multi_face_landmarks[0].landmark
            # get 2D image points
            image_points = np.array([
                (lm[LANDMARK_IDS["nose_tip"]].x * 320,
                 lm[LANDMARK_IDS["nose_tip"]].y * 240),
                (lm[LANDMARK_IDS["chin"]].x * 320,
                 lm[LANDMARK_IDS["chin"]].y * 240),
                (lm[LANDMARK_IDS["left_eye"]].x * 320,
                 lm[LANDMARK_IDS["left_eye"]].y * 240),
                (lm[LANDMARK_IDS["right_eye"]].x * 320,
                 lm[LANDMARK_IDS["right_eye"]].y * 240),
                (lm[LANDMARK_IDS["mouth_left"]].x * 320,
                 lm[LANDMARK_IDS["mouth_left"]].y * 240),
                (lm[LANDMARK_IDS["mouth_right"]].x * 320,
                 lm[LANDMARK_IDS["mouth_right"]].y * 240),
            ], dtype=np.float64)

            # Camera internals
            focal_length = 320
            center = (320 / 2, 240 / 2)
            camera_matrix = np.array([
                [focal_length, 0, center[0]],
                [0, focal_length, center[1]],
                [0, 0, 1]
            ], dtype=np.float64)
            dist_coeffs = np.zeros((4, 1))  # assume no lens distortion

            # Solve PnP for rotation vectors
            success, rot_vec, trans_vec = cv2.solvePnP(
                MODEL_POINTS, image_points, camera_matrix, dist_coeffs
            )

            # Project nose direction for visualization
            (nose_end_point2D, _) = cv2.projectPoints(
                np.array([(0.0, 0.0, 1000.0)]),
                rot_vec, trans_vec, camera_matrix, dist_coeffs
            )
            p1 = tuple(image_points[0].astype(int))
            p2 = tuple(nose_end_point2D[0][0].astype(int))
            cv2.line(small, p1, p2, (0, 255, 0), 2)

            # Determine yaw angle
            rmat, _ = cv2.Rodrigues(rot_vec)
            sy = np.sqrt(rmat[0,0]**2 + rmat[1,0]**2)
            yaw = np.arctan2(rmat[2,0], sy) * 57.3

            looking = abs(yaw) < 10  # within ±10° → looking
            # Bounding box
            xs = [pt[0] for pt in image_points]
            ys = [pt[1] for pt in image_points]
            x_min, x_max = int(min(xs)), int(max(xs))
            y_min, y_max = int(min(ys)), int(max(ys))
            cv2.rectangle(small, (x_min, y_min), (x_max, y_max), (255, 0, 0), 2)
            # Center check
            face_cx = (x_min + x_max) / 2
            centered = abs(face_cx - 160) < 50
            # Labels
            cv2.putText(small, f"Looking: {'Yes' if looking else 'No'}",
                        (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                        (0,255,255), 2)
            cv2.putText(small, f"Centered: {'Yes' if centered else 'No'}",
                        (10, 45), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                        (0,255,255), 2)

        # upscale back
        img_out = cv2.resize(small, (w, h))
        return av.VideoFrame.from_ndarray(img_out, format="bgr24")

def main():
    st.title("📹 Live Face & Posture Analytics")
    st.write("Detect face position, gaze, and centering in real time.")
    webrtc_streamer(
        key="face-analysis",
        video_transformer_factory=Transformer,
        media_stream_constraints={"video": True, "audio": False},
        async_processing=True,
        rtc_configuration={"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]},
    )

if __name__ == "__main__":
    main()
