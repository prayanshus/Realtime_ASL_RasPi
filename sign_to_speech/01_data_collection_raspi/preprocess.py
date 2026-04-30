#Imports and Configuration

import os
import cv2
import numpy as np
import mediapipe as mp

INPUT_DIR = "Video_Data_Liz"
FEATURES_DIR = "Extracted_Features"
OVERLAY_DIR = "Processed_Video_Data"
SEQUENCE_LENGTH = 30

BASE_FEATURES = 252
TOTAL_FEATURES = 504  # 252 base + 252 velocity

mp_holistic = mp.solutions.holistic
mp_drawing = mp.solutions.drawing_utils

def extract_landmarks(results):

    #Pose - Where hands are relative to face/chest
    pose = np.array([[r.x, r.y, r.z] for r in
           results.pose_landmarks.landmark]).flatten() \
           if results.pose_landmarks else np.zeros(33 * 3)

    #Raw hand points (needed for feature computation)
    if results.left_hand_landmarks:
        lh_pts = np.array([[r.x, r.y, r.z]
                  for r in results.left_hand_landmarks.landmark])
        lh_detected = True
    else:
        lh_pts = np.zeros((21, 3))
        lh_detected = False

    if results.right_hand_landmarks:
        rh_pts = np.array([[r.x, r.y, r.z]
                  for r in results.right_hand_landmarks.landmark])
        rh_detected = True
    else:
        rh_pts = np.zeros((21, 3))
        rh_detected = False

    # Wrist relative normalized positions
    # Landmark 0 = wrist, Landmark 9 = middle finger base (knuckle)

    rh_scale = np.linalg.norm(rh_pts[9] - rh_pts[0]) + 1e-6
    lh_scale = np.linalg.norm(lh_pts[9] - lh_pts[0]) + 1e-6

    rh_norm = ((rh_pts - rh_pts[0]) / rh_scale).flatten()  # 63 numbers
    lh_norm = ((lh_pts - lh_pts[0]) / lh_scale).flatten()  # 63 numbers

    # Fingertip distances
    # Landmarks 4,8,12,16,20 = tips of thumb, index, middle, ring, pinky
    # 5 tips → 10 unique pairs → 10 distances per hand

    def fingertip_distances(pts, scale):
        tips = [4, 8, 12, 16, 20]
        dists = []
        for i in range(len(tips)):
            for j in range(i + 1, len(tips)):
                d = np.linalg.norm(pts[tips[i]] - pts[tips[j]])
                dists.append(d / scale)  # normalise by hand size
        return np.array(dists)  # 10 numbers

    rh_dists = fingertip_distances(rh_pts, rh_scale)  # 10 numbers
    lh_dists = fingertip_distances(lh_pts, lh_scale)  # 10 numbers

    # Palm Orientation
    def palm_normal(pts, detected):
        if not detected:
            return np.zeros(3)
        v1 = pts[5] - pts[0]   # wrist → index base
        v2 = pts[17] - pts[0]  # wrist → pinky base
        normal = np.cross(v1, v2)
        magnitude = np.linalg.norm(normal) + 1e-6
        return normal / magnitude  # unit vector: 3 numbers

    rh_normal = palm_normal(rh_pts, rh_detected)  # 3 numbers
    lh_normal = palm_normal(lh_pts, lh_detected)  # 3 numbers

    # Inter-Hand distance (for 2 handed signs)
    inter_hand_dist = np.array([
        np.linalg.norm(rh_pts[0] - lh_pts[0])
    ])  # 1 number

    #Concatenate all features
    features = np.concatenate([
        pose,           # 99  — body context
        rh_norm,        # 63  — right hand shape (normalised)
        lh_norm,        # 63  — left hand shape (normalised)
        rh_dists,       # 10  — right fingertip distances
        lh_dists,       # 10  — left fingertip distances
        rh_normal,      # 3   — right palm orientation
        lh_normal,      # 3   — left palm orientation
        inter_hand_dist # 1   — distance between hands
    ]).astype(np.float32)

    # Total: 99+63+63+10+10+3+3+1 = 252 features

    return features


def pad_or_truncate_sequence(sequence, target_length):
    seq_len = len(sequence)
    if seq_len == target_length:
        return np.array(sequence)
    elif seq_len > target_length:
        return np.array(sequence[:target_length])
    else:
        padding = np.zeros((target_length - seq_len, sequence[0].shape[0]))
        return np.vstack((np.array(sequence), padding))


def add_velocity(sequence):
    
    #Compute velocity (frame-to-frame change) and stack with base features.
    velocity = np.diff(sequence, axis=0)  # shape: (29, 252)

    # Pad first row with zeros (no velocity before first frame)
    velocity = np.vstack([
        np.zeros((1, BASE_FEATURES)),
        velocity
    ])  # shape: (30, 252)

    # Stack: each frame now has both position AND velocity
    full_sequence = np.concatenate([sequence, velocity], axis=1)
    # shape: (30, 504)

    return full_sequence


#Preprocessing Loop
print(" Starting preprocessing with feature engineering")
print(f"Features per frame: {BASE_FEATURES} base + {BASE_FEATURES} velocity = {TOTAL_FEATURES} total")
print(f"Sequence shape: (30, {TOTAL_FEATURES})\n")

with mp_holistic.Holistic(
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5) as holistic:

    for action in sorted(os.listdir(INPUT_DIR)):
        action_path = os.path.join(INPUT_DIR, action)
        if not os.path.isdir(action_path):
            continue

        # Create output directories
        os.makedirs(os.path.join(FEATURES_DIR, action), exist_ok=True)
        os.makedirs(os.path.join(OVERLAY_DIR, action), exist_ok=True)

        video_files = [f for f in os.listdir(action_path) if f.endswith('.mp4')]
        print(f"Processing {action}: {len(video_files)} videos")

        for video_file in video_files:
            video_path = os.path.join(action_path, video_file)
            cap = cv2.VideoCapture(video_path)

            # Setup overlay video writer
            width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            fps    = cap.get(cv2.CAP_PROP_FPS) or 30.0

            overlay_path = os.path.join(
                OVERLAY_DIR, action, f"overlay_{video_file}")
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(overlay_path, fourcc, fps, (width, height))

            frame_landmarks = []

            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break

                # MediaPipe needs RGB
                image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                image_rgb.flags.writeable = False
                results = holistic.process(image_rgb)
                image_rgb.flags.writeable = True

                # Extract engineered features for this frame
                frame_landmarks.append(extract_landmarks(results))

                # Draw skeleton on overlay video
                mp_drawing.draw_landmarks(
                    frame, results.pose_landmarks,
                    mp_holistic.POSE_CONNECTIONS)
                mp_drawing.draw_landmarks(
                    frame, results.left_hand_landmarks,
                    mp_holistic.HAND_CONNECTIONS)
                mp_drawing.draw_landmarks(
                    frame, results.right_hand_landmarks,
                    mp_holistic.HAND_CONNECTIONS)
                out.write(frame)

            cap.release()
            out.release()

            if len(frame_landmarks) == 0:
                print(f" Skipped {video_file} — no frames extracted")
                continue

            # Step 1: Make sequence exactly 30 frames
            base_sequence = pad_or_truncate_sequence(
                frame_landmarks, SEQUENCE_LENGTH)  # shape: (30, 252)

            # Step 2: Add velocity features
            full_sequence = add_velocity(base_sequence)  # shape: (30, 504)

            # Step 3: Save as .npy file
            feature_path = os.path.join(
                FEATURES_DIR, action,
                video_file.replace('.mp4', '.npy'))
            np.save(feature_path, full_sequence)

        print(f" Done: {action}")

print(f"\n Preprocessing complete!")
print(f"Features saved in: {os.path.abspath(FEATURES_DIR)}")
print(f"Overlay videos in: {os.path.abspath(OVERLAY_DIR)}")

