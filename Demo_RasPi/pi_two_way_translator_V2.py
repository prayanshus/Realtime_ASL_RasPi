# ============================================================
# STREAM_PI.PY — RASPBERRY PI VERSION (INTEGRATED 2-WAY)
# ============================================================

import os
import tkinter as tk
from tkinter import ttk
import PIL.Image, PIL.ImageTk
import numpy as np
import mediapipe as mp
import subprocess
import time
import threading
import queue
from collections import deque
import cv2                      
import speech_recognition as sr 

from picamera2 import Picamera2

# ==========================================
# DISPLAY SETUP
# ==========================================
if 'DISPLAY' not in os.environ:
    os.environ['DISPLAY'] = ':0'

try:
    import tflite_runtime.interpreter as tflite
    print("Using tflite_runtime")
except ImportError:
    from tensorflow import lite as tflite
    print("Using tensorflow.lite")

# ==========================================
# CONFIGURATION
# ==========================================
if os.path.exists('sign_labels.npy'):
    LABELS = list(np.load('sign_labels.npy', allow_pickle=True))
    print(f"Loaded {len(LABELS)} labels: {LABELS}")
else:
    LABELS = sorted([
        'bad', 'come', 'food', 'go', 'good', 'hello', 'help', 'home', 'me', 'more',
        'please', 'sorry', 'stop', 'thank you', 'time', 'want', 'water', 'yes', 'no'
    ])
    print("WARNING: sign_labels.npy not found — using hardcoded labels")

SEQUENCE_LENGTH = 30
OVERLAP_STEP    = 15
THRESHOLD       = 0.85
COOLDOWN        = 2.0
BASE_FEATURES   = 252
TOTAL_FEATURES  = 504
MIN_HAND_FRAMES = 8
USE_POSE        = True
MEDIAPIPE_FRAME_SKIP = 2


# ==========================================
# STT WORKER (LISTENER MODE)
# ==========================================
class STTWorker(threading.Thread):
    def __init__(self, ui_callback):
        super().__init__(daemon=True)
        self.recognizer = sr.Recognizer()
        self.ui_callback = ui_callback
        self.is_listening = False
        self._stop_event = threading.Event()

    def run(self):
        with sr.Microphone() as source:
            self.recognizer.adjust_for_ambient_noise(source, duration=1)
            while not self._stop_event.is_set():
                if self.is_listening:
                    try:
                        # Listen with timeouts so thread can exit cleanly if needed
                        audio = self.recognizer.listen(source, timeout=1, phrase_time_limit=5)
                        text = self.recognizer.recognize_google(audio)
                        self.ui_callback(text)
                    except sr.WaitTimeoutError:
                        continue # Nobody spoke, loop back
                    except sr.UnknownValueError:
                        pass # Audio wasn't clear
                    except sr.RequestError as e:
                        print(f"STT API Error: {e}")
                else:
                    time.sleep(0.1)

    def start_listening(self):
        self.is_listening = True

    def stop_listening(self):
        self.is_listening = False

    def stop(self):
        self._stop_event.set()


# ==========================================
# TTS WORKER
# ==========================================
class TTSWorker(threading.Thread):
    def __init__(self):
        super().__init__(daemon=True)
        self._q           = queue.Queue()
        self._active_proc = None

    def run(self):
        while True:
            text = self._q.get()
            if text is None:
                break
            if self._active_proc and self._active_proc.poll() is None:
                self._active_proc.terminate()
                self._active_proc.wait()
            try:
                # --- CLEARER VOICE SETTINGS ---
                self._active_proc = subprocess.Popen(
                    ['espeak-ng', '-v', 'en-in+f4', '-s', '140', '-p', '75', text],
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL)
                self._active_proc.wait()
            except FileNotFoundError:
                print("espeak-ng not found: sudo apt install espeak-ng")

    def speak(self, text):
        while not self._q.empty():
            try:
                self._q.get_nowait()
            except queue.Empty:
                break
        self._q.put(text)

    def stop(self):
        if self._active_proc and self._active_proc.poll() is None:
            self._active_proc.terminate()
        self._q.put(None)


# ==========================================
# INFERENCE WORKER
# ==========================================
class InferenceWorker(threading.Thread):
    def __init__(self, model_path, result_callback):
        super().__init__(daemon=True)
        self._q               = queue.Queue(maxsize=1)
        self._result_callback = result_callback
        self._stop_event      = threading.Event()
        self._load(model_path)

    def _load(self, model_path):
        self.interpreter = tflite.Interpreter(model_path=model_path)
        self.interpreter.allocate_tensors()
        self._in  = self.interpreter.get_input_details()
        self._out = self.interpreter.get_output_details()

    def run(self):
        while not self._stop_event.is_set():
            try:
                sequence, submit_time = self._q.get(timeout=0.1)
            except queue.Empty:
                continue
            tensor = np.expand_dims(sequence, axis=0)
            self.interpreter.set_tensor(self._in[0]['index'], tensor)
            self.interpreter.invoke()
            preds = self.interpreter.get_tensor(self._out[0]['index'])[0]
            self._result_callback(preds, submit_time)

    def submit(self, sequence, submit_time):
        if self._q.full():
            try:
                self._q.get_nowait()
            except queue.Empty:
                pass
        self._q.put((sequence, submit_time))

    def stop(self):
        self._stop_event.set()


# ==========================================
# MAIN APP 
# ==========================================
class ASLTranscriberApp:
    def __init__(self, window):
        self.window = window
        self.window.title("ASL Sign Language Translator")
        self.window.geometry("800x800")
        self.window.configure(bg="#f4f4f9")

        self.frame_buffer              = deque(maxlen=SEQUENCE_LENGTH)
        self.hand_present_frames       = 0
        self.frames_since_last_predict = 0
        self.last_spoken               = ""
        self.last_speech_time          = 0
        self.current_model_type        = tk.StringVar(value="FP32")
        self._inference_worker         = None
        self._frame_skip_counter       = 0
        self._last_results             = None   

        # --- Metrics Tracking ---
        self.last_frame_time           = time.time()
        self.fps_buffer                = deque(maxlen=15)
        self.last_inf_time_ms          = 0.0

        # Modes & Threads
        self.current_mode = "SPEAKER"
        self._stt = STTWorker(self._on_speech_recognized)
        self._stt.start()
        
        self._tts = TTSWorker()
        self._tts.start()

        # MediaPipe
        self.mp_hands   = mp.solutions.hands
        self.mp_drawing = mp.solutions.drawing_utils

        if USE_POSE:
            self.mp_holistic = mp.solutions.holistic
            self.holistic = self.mp_holistic.Holistic(
                min_detection_confidence=0.5,
                min_tracking_confidence=0.5)
        else:
            self.hands_detector = self.mp_hands.Hands(
                static_image_mode=False,
                max_num_hands=2,
                min_detection_confidence=0.6,
                min_tracking_confidence=0.5,
                model_complexity=0)

        # Picamera2
        self.picam2 = Picamera2()
        config = self.picam2.create_preview_configuration(
            main={"size": (640, 480), "format": "RGB888"})
        self.picam2.configure(config)
        self.picam2.start()
        time.sleep(0.5)

        test_frame = self.picam2.capture_array()
        self._cam_channels = test_frame.shape[2] if test_frame.ndim == 3 else 1

        self.setup_gui()
        self.load_model()
        self.update_frame()

    # ------------------------------------------
    # UI Mode Switching
    # ------------------------------------------
    def toggle_mode(self):
        if self.current_mode == "SPEAKER":
            self.current_mode = "LISTENER"
            self.mode_btn.config(text="📷 Switch to SPEAKER (Camera)")
            self.status_label.config(text="Microphone Active. Speak now.", foreground="#0056b3")
            self.transcript_box.delete(1.0, tk.END)
            self.transcript_box.insert(tk.END, "Listening for speech...")
            self.frame_buffer.clear()
            self._stt.start_listening()
        else:
            self.current_mode = "SPEAKER"
            self.mode_btn.config(text="🎤 Switch to LISTENER (Mic)")
            self.status_label.config(text="Camera Active. Show a hand.", foreground="#333333")
            self.transcript_box.delete(1.0, tk.END)
            self.transcript_box.insert(tk.END, "Camera active. Sign to speech enabled.")
            self._stt.stop_listening()

    def _on_speech_recognized(self, text):
        def update_text():
            self.transcript_box.delete(1.0, tk.END)
            self.transcript_box.insert(tk.END, text)
        self.window.after(0, update_text)

    # ------------------------------------------
    # Utils & GUI Setup
    # ------------------------------------------
    def _to_rgb(self, frame):
        if self._cam_channels == 4:
            frame = frame[:, :, :3]
        return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    def setup_gui(self):
        style = ttk.Style()
        style.configure("Big.TButton", font=("Helvetica", 16, "bold"), padding=15)
        style.configure("Custom.TRadiobutton", background="#f4f4f9", font=("Helvetica", 12))

        top_bar = tk.Frame(self.window, bg="#f4f4f9")
        top_bar.pack(side=tk.TOP, fill=tk.X, padx=20, pady=15)

        self.status_label = ttk.Label(top_bar, text="Camera Active. Show a hand.", font=('Helvetica', 16, 'bold'), background="#f4f4f9", foreground="#333333")
        self.status_label.pack(side=tk.LEFT)

        # Performance Metrics Label
        self.metrics_label = ttk.Label(top_bar, text="FPS: -- | E2E Inf: -- ms", font=('Helvetica', 12, 'bold'), background="#f4f4f9", foreground="#555555")
        self.metrics_label.pack(side=tk.LEFT, padx=20)

        model_frame = tk.Frame(top_bar, bg="#f4f4f9")
        model_frame.pack(side=tk.RIGHT)
        ttk.Label(model_frame, text="Model:", font=('Helvetica', 12, 'bold'), background="#f4f4f9").pack(side=tk.LEFT, padx=5)
        ttk.Radiobutton(model_frame, text="FP32", variable=self.current_model_type, value="FP32", command=self.load_model, style="Custom.TRadiobutton").pack(side=tk.LEFT, padx=5)
        ttk.Radiobutton(model_frame, text="INT8", variable=self.current_model_type, value="QAT", command=self.load_model, style="Custom.TRadiobutton").pack(side=tk.LEFT, padx=5)

        self.canvas = tk.Canvas(self.window, width=640, height=480, bg="black", highlightthickness=2, highlightbackground="#cccccc")
        self.canvas.pack(pady=10)

        self.transcript_box = tk.Text(self.window, height=4, width=45, font=('Helvetica', 22, 'bold'), bg="#ffffff", fg="#222222", wrap=tk.WORD, relief=tk.GROOVE, borderwidth=3)
        self.transcript_box.pack(pady=15, padx=20)
        self.transcript_box.insert(tk.END, "Camera active. Sign to speech enabled.")

        self.mode_btn = ttk.Button(self.window, text="🎤 Switch to LISTENER (Mic)", command=self.toggle_mode, style="Big.TButton")
        self.mode_btn.pack(pady=10)

    # ------------------------------------------
    # MODEL LOADING & PREDICTION
    # ------------------------------------------
    def load_model(self):
        model_path = ("sign_model_fp32.tflite" if self.current_model_type.get() == "FP32" else "sign_model_qat.tflite")
        try:
            if self._inference_worker is not None:
                self._inference_worker.stop()
                self._inference_worker.join(timeout=1.0)
            self._inference_worker = InferenceWorker(model_path, self._on_prediction)
            self._inference_worker.start()
            self.frame_buffer.clear()
            self.hand_present_frames = 0
            self.frames_since_last_predict = 0
        except Exception as e:
            print(f"Model load error: {e}")

    def _on_prediction(self, preds, submit_time):
        self.last_inf_time_ms = (time.time() - submit_time) * 1000

        idx = int(np.argmax(preds))
        confidence = float(preds[idx])
        word = LABELS[idx] if idx < len(LABELS) else "unknown"

        def _update_ui():
            if confidence > THRESHOLD:
                self.status_label.config(text=f"Detected: {word.upper()} ({confidence:.0%})", foreground="#28a745")
                curr_time = time.time()
                if (word.lower() != "background" and (word != self.last_spoken or curr_time - self.last_speech_time > COOLDOWN)):
                    self.transcript_box.delete(1.0, tk.END)
                    self.transcript_box.insert(tk.END, f'"{word.upper()}"')
                    self._tts.speak(word)
                    self.last_spoken = word
                    self.last_speech_time = curr_time
            else:
                self.status_label.config(text=f"Low confidence: {word} ({confidence:.0%})", foreground="#d35400")

        self.window.after(0, _update_ui)

    # ------------------------------------------
    # FEATURE EXTRACTION & VELOCITY STACKING
    # ------------------------------------------
    def extract_landmarks(self, results):
        lh_pts, rh_pts = np.zeros((21, 3)), np.zeros((21, 3))
        lh_detected, rh_detected = False, False

        if USE_POSE:
            if results.left_hand_landmarks:
                lh_pts = np.array([[l.x, l.y, l.z] for l in results.left_hand_landmarks.landmark])
                lh_detected = True
            if results.right_hand_landmarks:
                rh_pts = np.array([[l.x, l.y, l.z] for l in results.right_hand_landmarks.landmark])
                rh_detected = True
        else:
            if results.multi_hand_landmarks:
                for hand_lm, handedness in zip(results.multi_hand_landmarks, results.multi_handedness):
                    pts = np.array([[l.x, l.y, l.z] for l in hand_lm.landmark])
                    label = handedness.classification[0].label
                    if label == "Right":
                        lh_pts, lh_detected = pts, True
                    else:
                        rh_pts, rh_detected = pts, True

        if USE_POSE:
            pose = (np.array([[r.x, r.y, r.z] for r in results.pose_landmarks.landmark]).flatten() if results.pose_landmarks else np.zeros(33 * 3))

        rh_scale = np.linalg.norm(rh_pts[9] - rh_pts[0]) + 1e-6
        lh_scale = np.linalg.norm(lh_pts[9] - lh_pts[0]) + 1e-6
        rh_norm  = ((rh_pts - rh_pts[0]) / rh_scale).flatten()
        lh_norm  = ((lh_pts - lh_pts[0]) / lh_scale).flatten()

        def fingertip_distances(pts, scale):
            tips = [4, 8, 12, 16, 20]
            return np.array([np.linalg.norm(pts[tips[i]] - pts[tips[j]]) / scale for i in range(len(tips)) for j in range(i + 1, len(tips))])

        rh_dists = fingertip_distances(rh_pts, rh_scale)
        lh_dists = fingertip_distances(lh_pts, lh_scale)

        def palm_normal(pts, detected):
            if not detected: return np.zeros(3)
            n = np.cross(pts[5] - pts[0], pts[17] - pts[0])
            return n / (np.linalg.norm(n) + 1e-6)

        rh_normal  = palm_normal(rh_pts, rh_detected)
        lh_normal  = palm_normal(lh_pts, lh_detected)
        inter_hand = np.array([np.linalg.norm(rh_pts[0] - lh_pts[0])])

        if USE_POSE:
            return np.concatenate([pose, rh_norm, lh_norm, rh_dists, lh_dists, rh_normal, lh_normal, inter_hand]).astype(np.float32)
        else:
            return np.concatenate([rh_norm, lh_norm, rh_dists, lh_dists, rh_normal, lh_normal, inter_hand]).astype(np.float32)

    def compute_velocity_sequence(self):
        sequence = np.array(self.frame_buffer, dtype=np.float32)
        velocity = np.diff(sequence, axis=0)
        velocity = np.vstack([np.zeros((1, sequence.shape[1]), dtype=np.float32), velocity])
        return np.concatenate([sequence, velocity], axis=1)

    # ------------------------------------------
    # MAIN FRAME LOOP
    # ------------------------------------------
    def update_frame(self):
        curr_time = time.time()
        fps = 1.0 / (curr_time - self.last_frame_time + 1e-6)
        self.last_frame_time = curr_time
        self.fps_buffer.append(fps)
        avg_fps = sum(self.fps_buffer) / len(self.fps_buffer)
        
        self.metrics_label.config(text=f"FPS: {avg_fps:.1f} | E2E Inf: {self.last_inf_time_ms:.1f} ms")

        try:
            raw = self.picam2.capture_array()
            raw = np.array(raw)
            frame_rgb = self._to_rgb(raw)

            # ONLY PROCESS AI IF IN SPEAKER MODE
            if self.current_mode == "SPEAKER":
                self._frame_skip_counter += 1
                run_mediapipe = (self._frame_skip_counter > MEDIAPIPE_FRAME_SKIP)
                if run_mediapipe:
                    self._frame_skip_counter = 0
                    frame_rgb.flags.writeable = False
                    if USE_POSE:
                        self._last_results = self.holistic.process(frame_rgb)
                    else:
                        self._last_results = self.hands_detector.process(frame_rgb)
                    frame_rgb.flags.writeable = True

                results = self._last_results

                if results is None:
                    hand_present = False
                elif USE_POSE:
                    hand_present = (results.left_hand_landmarks is not None or results.right_hand_landmarks is not None)
                else:
                    hand_present = bool(results.multi_hand_landmarks)

                if results is not None:
                    if USE_POSE:
                        self.mp_drawing.draw_landmarks(frame_rgb, results.pose_landmarks, self.mp_holistic.POSE_CONNECTIONS)
                        self.mp_drawing.draw_landmarks(frame_rgb, results.left_hand_landmarks, self.mp_hands.HAND_CONNECTIONS)
                        self.mp_drawing.draw_landmarks(frame_rgb, results.right_hand_landmarks, self.mp_hands.HAND_CONNECTIONS)
                    elif hand_present:
                        for hand_lm in results.multi_hand_landmarks:
                            self.mp_drawing.draw_landmarks(frame_rgb, hand_lm, self.mp_hands.HAND_CONNECTIONS)

                if hand_present:
                    self.hand_present_frames += 1
                    landmarks = self.extract_landmarks(results)
                    self.frame_buffer.append(landmarks)
                    self.frames_since_last_predict += 1

                    if (len(self.frame_buffer) == SEQUENCE_LENGTH and self.frames_since_last_predict >= OVERLAP_STEP and self.hand_present_frames >= MIN_HAND_FRAMES):
                        full_sequence = self.compute_velocity_sequence()
                        self._inference_worker.submit(full_sequence, time.time())
                        self.frames_since_last_predict = 0
                else:
                    if self.hand_present_frames > 0:
                        self.frame_buffer.clear()
                        self.frames_since_last_predict = 0
                        self.status_label.config(text="Camera Active. Show a hand.", foreground="#333333")
                    self.hand_present_frames = 0

            # Render Camera Feed
            self.photo = PIL.ImageTk.PhotoImage(image=PIL.Image.fromarray(frame_rgb))
            self.canvas.create_image(0, 0, image=self.photo, anchor=tk.NW)

        except Exception as e:
            pass

        self.window.after(15, self.update_frame)

    # ------------------------------------------
    # SHUTDOWN
    # ------------------------------------------
    def on_close(self):
        print("Shutting down...")
        if self._inference_worker:
            self._inference_worker.stop()
        self._stt.stop()
        self._tts.stop()
        self.picam2.stop()
        self.picam2.close()
        if USE_POSE:
            self.holistic.close()
        else:
            self.hands_detector.close()
        self.window.destroy()
        print("Closed cleanly.")


if __name__ == "__main__":
    root = tk.Tk()
    app  = ASLTranscriberApp(root)
    root.protocol("WM_DELETE_WINDOW", app.on_close)
    root.mainloop()