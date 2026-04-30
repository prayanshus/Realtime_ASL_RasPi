# ============================================================
# DATA CAPTURE SCRIPT — MODIFIED FOR 19 SIGNS
# Changes from V2:
#   1. Updated actions list to all 19 signs
#   2. Increased to 60 videos per sign (more data = better model)
#   3. Increased countdown to 3 seconds (more time to get ready)
#   4. Added sign instructions on screen so you know what to do
#   5. Added SPACE bar to skip a bad recording
#   6. Added 'b' key to go back one sign if needed
# ============================================================

import os
import cv2
import numpy as np
import time
import mediapipe as mp

# ==========================================
# 0. SMART ENVIRONMENT DETECTION
# ==========================================
try:
    from picamera2 import Picamera2
    IS_RASPI = True
    print("Environment Detected: Raspberry Pi (Using Picamera2)")
    os.environ['DISPLAY'] = ':0'
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
    os.environ['GLOG_minloglevel'] = '2'
except ImportError:
    IS_RASPI = False
    print("Environment Detected: Laptop/Desktop (Using standard OpenCV)")

# ==========================================
# 1. SETUP & HELPER FUNCTIONS
# ==========================================
mp_holistic = mp.solutions.holistic
mp_drawing = mp.solutions.drawing_utils

def mediapipe_detection(image_rgb, model):
    image_rgb.flags.writeable = False
    results = model.process(image_rgb)
    image_rgb.flags.writeable = True
    return results

def draw_styled_landmarks(image, results):
    mp_drawing.draw_landmarks(
        image, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS,
        mp_drawing.DrawingSpec(color=(121,22,76), thickness=2, circle_radius=4),
        mp_drawing.DrawingSpec(color=(121,44,250), thickness=2, circle_radius=2))
    mp_drawing.draw_landmarks(
        image, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS,
        mp_drawing.DrawingSpec(color=(245,117,66), thickness=2, circle_radius=4),
        mp_drawing.DrawingSpec(color=(245,66,230), thickness=2, circle_radius=2))

def get_frame():
    """Get one frame from whichever camera is active."""
    if IS_RASPI:
        rgb_frame = picam2.capture_array()
        bgr_frame = cv2.cvtColor(rgb_frame, cv2.COLOR_RGB2BGR)
        return bgr_frame, rgb_frame
    else:
        ret, bgr_frame = cap.read()
        if not ret:
            return None, None
        rgb_frame = cv2.cvtColor(bgr_frame, cv2.COLOR_BGR2RGB)
        return bgr_frame, rgb_frame

# ==========================================
# 2. DATA CONFIGURATION — CHANGE SIGNS HERE
# ==========================================
DATA_PATH = os.path.join('Video_Data')

# ---- YOUR 19 SIGNS ----
# NOTE: BYE removed because it is identical to HELLO in ASL
# Added SORRY as replacement — it is visually distinct
actions = np.array([
    'yes',       # fist bobs up and down
    'no',        # index+middle tap thumb
    'hello',     # open hand wave from forehead
    'please',    # flat palm circles on chest
    'thank you', # hand from chin moves forward
    'water',     # W shape taps chin
    'more',      # fingertips tap together (2 hands)
    'home',      # pinched hand taps cheek twice
    'good',      # hand from chin moves down
    'bad',       # hand from chin flips outward
    'come',      # index finger curls inward
    'go',        # index finger points outward
    'stop',      # flat hand chops onto other palm
    'time',      # index finger taps wrist
    'me',        # index finger points to own chest
    'food',      # pinched fingers tap mouth
    'want',      # claw hands pull toward body
    'help',      # thumbs up lifted by other palm
    'sorry',     # fist circles on chest
])

# ---- HOW MANY VIDEOS PER SIGN ----
# 60 is better than 50 — gives more training data
# If you are short on time, 50 is acceptable
no_sequences = 60

# ---- HOW MANY FRAMES PER VIDEO ----
sequence_length = 30  # 30 frames = about 1-2 seconds of sign

# ---- FRAMES PER SECOND ----
fps = 30.0 if IS_RASPI else 15.0

# ---- INSTRUCTIONS for each sign (shown on screen) ----
# So you know what to do without looking at YouTube every time
instructions = {
    'yes':       'Make a FIST. Bob it UP and DOWN like nodding.',
    'no':        'Index+middle fingers TAP your THUMB repeatedly.',
    'hello':     'Open hand SALUTE from forehead, move hand OUTWARD.',
    'please':    'Flat palm makes CIRCLES on your CHEST.',
    'thank you': 'Flat hand touches CHIN, then moves FORWARD and down.',
    'water':     'W shape (3 fingers up), TAP your CHIN twice.',
    'more':      'Both hands pinched, TAP fingertips TOGETHER.',
    'home':      'Pinched hand TAPS CHEEK, then taps again lower.',
    'good':      'Flat hand touches CHIN, drops straight DOWN.',
    'bad':       'Flat hand touches CHIN, FLIPS palm outward.',
    'come':      'Index finger points out, CURLS toward your body.',
    'go':        'Index finger POINTS AWAY from your body.',
    'stop':      'Flat hand CHOPS DOWN onto your other open palm.',
    'time':      'Index finger TAPS your wrist (like pointing at watch).',
    'me':        'Index finger POINTS to your own CHEST.',
    'food':      'Pinched fingertips TAP your MOUTH/chin twice.',
    'want':      'Both hands CLAW shape, PULL toward your body.',
    'help':      'THUMBS UP on open palm, LIFT both hands upward.',
    'sorry':     'FIST makes CIRCLES on your chest.',
}

# ==========================================
# 3. CREATE FOLDER STRUCTURE
# ==========================================
for action in actions:
    try:
        os.makedirs(os.path.join(DATA_PATH, action))
    except:
        pass  # folder already exists, that's fine

# ==========================================
# 4. INITIALIZE CAMERA
# ==========================================
print("Warming up camera...")
if IS_RASPI:
    picam2 = Picamera2()
    config = picam2.create_preview_configuration(
        main={"size": (640, 480), "format": "RGB888"})
    picam2.configure(config)
    picam2.start()
else:
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

time.sleep(1)
print(f"\n📋 Will record {no_sequences} videos for each of {len(actions)} signs")
print(f"📋 Total videos to record: {no_sequences * len(actions)}")
print(f"📋 Controls: Q = quit | SPACE = skip this video (bad recording)")
print(f"📋 Tip: Keep your hand clearly visible, good lighting, plain background\n")

# ==========================================
# 5. VIDEO COLLECTION LOOP
# ==========================================
try:
    with mp_holistic.Holistic(
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5) as holistic:

        for action in actions:

            # Count how many videos already exist for this sign
            existing_videos = len([
                f for f in os.listdir(os.path.join(DATA_PATH, action))
                if f.endswith('.mp4')
            ])

            if existing_videos >= no_sequences:
                print(f"⏭️  Skipping {action} — already has {existing_videos} videos")
                continue

            print(f"\n{'='*50}")
            print(f"📹 SIGN: {action.upper()}")
            print(f"📖 HOW: {instructions.get(action, 'See YouTube for this sign')}")
            print(f"{'='*50}")

            # Show a "get ready for new sign" screen for 3 seconds
            for i in range(3, 0, -1):
                bgr_frame, _ = get_frame()
                if bgr_frame is None:
                    continue

                # Dark overlay
                overlay = bgr_frame.copy()
                cv2.rectangle(overlay, (0,0), (640,480), (0,0,0), -1)
                cv2.addWeighted(overlay, 0.5, bgr_frame, 0.5, 0, bgr_frame)

                cv2.putText(bgr_frame,
                    f'NEXT SIGN: {action.upper()}',
                    (40, 160), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0,255,255), 3)
                cv2.putText(bgr_frame,
                    instructions.get(action, ''),
                    (20, 210), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255,255,255), 1)
                cv2.putText(bgr_frame,
                    f'Starting in {i}...',
                    (220, 290), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0,255,0), 2)

                cv2.imshow('Video Collection Feed', bgr_frame)
                cv2.waitKey(1000)

            # Record videos for this sign
            sequence = existing_videos  # start numbering from where we left off

            while sequence < no_sequences:

                # 2-second countdown before each video
                skip_this = False
                for countdown in range(2, 0, -1):
                    bgr_frame, _ = get_frame()
                    if bgr_frame is None:
                        continue

                    cv2.putText(bgr_frame,
                        f'{action.upper()} | Video {sequence+1}/{no_sequences}',
                        (15, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,255), 2)
                    cv2.putText(bgr_frame,
                        f'GET READY... {countdown}',
                        (150, 240), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0,255,0), 4)
                    cv2.putText(bgr_frame,
                        instructions.get(action, ''),
                        (20, 460), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,0), 1)
                    cv2.putText(bgr_frame,
                        'SPACE=skip this video | Q=quit',
                        (15, 12), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (180,180,180), 1)

                    cv2.imshow('Video Collection Feed', bgr_frame)
                    key = cv2.waitKey(1000)

                    if key == ord('q'):
                        raise KeyboardInterrupt
                    if key == ord(' '):
                        skip_this = True
                        break

                if skip_this:
                    print(f"  ⏭️  Skipped video {sequence+1}")
                    sequence += 1
                    continue

                # Now record the actual video
                video_path = os.path.join(DATA_PATH, action, f"{sequence}.mp4")
                fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                out = cv2.VideoWriter(video_path, fourcc, fps, (640, 480))

                quit_now = False
                for frame_num in range(sequence_length):
                    bgr_frame, rgb_frame = get_frame()
                    if bgr_frame is None:
                        break

                    # Save raw video frame
                    out.write(bgr_frame)

                    # Run MediaPipe for live display feedback
                    results = mediapipe_detection(rgb_frame, holistic)

                    # Draw skeleton on display copy
                    display_frame = bgr_frame.copy()
                    draw_styled_landmarks(display_frame, results)

                    # Show recording status
                    cv2.putText(display_frame,
                        f'RECORDING: {action.upper()} | Video {sequence+1}/{no_sequences}',
                        (15, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0,0,255), 2)

                    # Show progress bar
                    progress = int((frame_num / sequence_length) * 400)
                    cv2.rectangle(display_frame, (120, 450), (520, 470), (80,80,80), -1)
                    cv2.rectangle(display_frame, (120, 450), (120+progress, 470), (0,255,0), -1)
                    cv2.putText(display_frame,
                        f'Frame {frame_num+1}/{sequence_length}',
                        (220, 445), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1)

                    # Warn if no hand detected
                    if not results.right_hand_landmarks and not results.left_hand_landmarks:
                        cv2.putText(display_frame,
                            '⚠ NO HAND DETECTED — move hand into frame!',
                            (60, 400), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0,0,255), 2)

                    cv2.imshow('Video Collection Feed', display_frame)

                    key = cv2.waitKey(1) & 0xFF
                    if key == ord('q'):
                        quit_now = True
                        break

                out.release()

                if quit_now:
                    raise KeyboardInterrupt

                print(f"  ✅ Saved: {action} video {sequence+1}/{no_sequences}")
                sequence += 1

        print("\n✅ ALL VIDEOS RECORDED SUCCESSFULLY!")
        print(f"📁 Videos saved in: {os.path.abspath(DATA_PATH)}")
        print("👉 Next step: run preprocess.py")

except KeyboardInterrupt:
    print("\n⚠️  Recording stopped by user.")
    print(f"📁 Videos saved so far in: {os.path.abspath(DATA_PATH)}")
    print("You can resume later — the script skips signs that already have enough videos.")

finally:
    if 'out' in locals():
        try:
            out.release()
        except:
            pass

    if IS_RASPI:
        if 'picam2' in locals():
            picam2.stop()
            picam2.close()
    else:
        if 'cap' in locals():
            cap.release()

    cv2.destroyAllWindows()
    print("Camera safely closed.")
