# Real-Time Two-Way ASL Translator

A proof-of-concept wearable system that enables two-way communication between ASL (American Sign Language) users and non-signers. The system runs entirely on a Raspberry Pi 5 and supports two modes: **Sign-to-Speech** (camera captures signs → CNN classifies → audio output) and **Speech-to-Text** (microphone captures speech → speech recognition transcribes → text on display).

> **Course:** Edge AI (2026)

---

## Table of Contents

1. [Introduction](#1-introduction)
2. [Motivation](#2-motivation)
3. [System Architecture](#3-system-architecture)
4. [Hardware Setup](#4-hardware-setup)
5. [Dataset](#5-dataset)
6. [Feature Engineering](#6-feature-engineering)
7. [Model Training](#7-model-training)
8. [Model Optimization and Quantization](#8-model-optimization-and-quantization)
9. [Deployment on Raspberry Pi](#9-deployment-on-raspberry-pi)
10. [Application Interface](#10-application-interface)
11. [Results and Performance](#11-results-and-performance)
12. [Challenges and Learnings](#12-challenges-and-learnings)
13. [Future Work](#13-future-work)
14. [Reproducing the Project](#14-reproducing-the-project)
15. [Project Structure](#15-project-structure)
16. [References and Acknowledgements](#16-references-and-acknowledgements)

---

## 1. Introduction

Communication between ASL users and non-signers remains a significant barrier in everyday interactions. This project implements a real-time, portable, two-way ASL translator that runs on edge hardware (Raspberry Pi 5), requiring no cloud connectivity for sign language inference.

The system recognizes **19 common ASL signs** from a live camera feed using MediaPipe hand/pose landmarks and a CNN classifier, then speaks the detected word aloud via a Bluetooth speaker. In the reverse direction, it captures spoken English via a Bluetooth microphone, transcribes it using Google Speech Recognition, and displays the text on an attached touchscreen — enabling bidirectional conversation.

---

## 2. Motivation

- **Accessibility:** Bridge the communication gap between deaf/hard-of-hearing individuals and hearing people in everyday scenarios such as shopping, healthcare, and social interactions.
- **Edge deployment:** Demonstrate that real-time sign language recognition can run on low-power, portable hardware without cloud dependency for the ML inference pipeline.
- **Wearable proof-of-concept:** The compact hardware stack (Pi 5 + touchscreen + Bluetooth audio + ribbon camera) represents a form factor that could evolve into a practical wearable device.

---

## 3. System Architecture

### High-Level Pipeline

```
┌──────────────────────────────────────────────────────────────────┐
│                    TWO-WAY ASL TRANSLATOR                        │
├──────────────────────────────────────────────────────────────────┤
│                                                                  │
│  ┌─── SIGN-TO-SPEECH (Speaker Mode) ───────────────────────┐    │
│  │                                                          │    │
│  │  Pi Camera ──► MediaPipe ──► Feature ──► CNN ──► TTS   │    │
│  │  (640×480)     Holistic     Extraction   (TFLite)  espeak│    │
│  │                (Pose+Hands) (252+vel)              -ng   │    │
│  │                                                          │    │
│  └──────────────────────────────────────────────────────────┘    │
│                                                                  │
│  ┌─── SPEECH-TO-TEXT (Listener Mode) ──────────────────────┐    │
│  │                                                          │    │
│  │  Bluetooth ──► SpeechRecognition ──► Google STT ──► GUI  │    │
│  │  Mic (Sony      (PyAudio)            API          Display│    │
│  │   XB12)                                                  │    │
│  │                                                          │    │
│  └──────────────────────────────────────────────────────────┘    │
│                                                                  │
│  ┌─── SHARED COMPONENTS ───────────────────────────────────┐    │
│  │  Tkinter GUI  │  Threaded workers  │  Mode toggle btn   │    │
│  └──────────────────────────────────────────────────────────┘    │
│                                                                  │
└──────────────────────────────────────────────────────────────────┘
```

### Detailed Sign-to-Speech Pipeline

1. **Capture:** Picamera2 captures 640×480 RGB frames at ~30 FPS.
2. **Pose estimation:** MediaPipe Holistic extracts 33 pose landmarks + 21 left-hand landmarks + 21 right-hand landmarks per frame (run every 2nd frame to reduce CPU load).
3. **Feature engineering:** Raw landmarks are transformed into a 252-dimensional engineered feature vector per frame (wrist-normalized coordinates, fingertip distances, palm normals, inter-hand distance, pose context).
4. **Velocity stacking:** Frame-to-frame differences (velocity) are computed and concatenated, yielding **504 features per frame**.
5. **Sequence buffering:** A sliding window of 30 frames with 15-frame overlap is maintained in a deque.
6. **Inference:** The (30, 504) sequence tensor is fed to a TFLite CNN model running on a dedicated inference thread. If confidence exceeds **85%**, the sign is classified.
7. **TTS output:** `espeak-ng` speaks the detected word aloud via the Bluetooth speaker, with a 2-second cooldown to prevent repetitive speech.

### Speech-to-Text Pipeline

1. **Capture:** Bluetooth microphone (Sony SRS-XB12) captures audio via PyAudio.
2. **Ambient calibration:** The recognizer adjusts for ambient noise on startup (1-second calibration window).
3. **Transcription:** The `SpeechRecognition` library sends audio chunks to Google Speech Recognition API.
4. **Display:** Transcribed text appears in real-time on the touchscreen GUI.

---

## 4. Hardware Setup

| Component | Specification |
|---|---|
| Compute | Raspberry Pi 5 (4 GB / 8 GB RAM) |
| Camera | Raspberry Pi Camera Module (CSI ribbon cable) |
| Display | Official Raspberry Pi Touchscreen Display |
| Audio I/O | Sony SRS-XB12 Bluetooth speaker + microphone |
| OS | Raspberry Pi OS Bookworm (64-bit) |

### Connections

- **Camera:** CSI ribbon cable → Pi 5 camera connector.
- **Touchscreen:** DSI ribbon cable + USB power cable → Pi 5.
- **Bluetooth audio:** Paired via `bluetoothctl` — serves as both speaker (TTS output) and microphone (STT input).

<!-- TODO: Add a photo of your hardware setup -->
<!-- ![Hardware Setup](images/hardware_setup.jpg) -->

---

## 5. Dataset

### Data Collection

Data was collected **directly on the Raspberry Pi** using the custom `data_capture_raspi_V3.py` script. Collecting on the target device ensures the training data matches the deployment camera's characteristics (resolution, lens distortion, color profile), reducing the domain gap.

**Collection details:**

- **19 ASL signs** (common conversational words)
- **180 video clips per sign**, each 30 frames long (~1–2 seconds)
- **Total:** 3,420 video clips
- **Resolution:** 640×480 at 30 FPS
- **Format:** `.mp4` raw video → `.npy` extracted feature sequences

The script provides on-screen instructions for each sign, a countdown timer, progress bars, real-time hand detection warnings, and the ability to skip bad recordings.

### Supported Signs

| Sign | Gesture Description |
|---|---|
| yes | Fist bobs up and down |
| no | Index + middle fingers tap thumb |
| hello | Open hand wave from forehead |
| please | Flat palm circles on chest |
| thank you | Hand from chin moves forward |
| water | W-shape (3 fingers up) taps chin |
| more | Fingertips tap together (two hands) |
| home | Home symbol using both arms |
| good | Thumbs up sign |
| bad | Thumbs down sign |
| come | Index finger curls inward |
| go | Index finger points outward |
| stop | Flat hand chops onto other palm |
| time | Index finger taps wrist |
| me | Index finger points to own chest |
| food | Pinched fingers tap mouth |
| want | Claw hands pull toward body |
| help | Thumbs-up lifted by other palm |

### Downloading the Pre-Collected Dataset

```bash
python fetch_dataset.py
```

This downloads the training video data from Google Drive into `sign_to_speech/01_data_collection_raspi/`.

---

## 6. Feature Engineering

Rather than feeding raw pixel data or raw landmark coordinates to the model, the system uses a carefully designed feature vector that is robust to hand size, camera distance, and hand orientation.

### Per-Frame Feature Vector (252 dimensions)

| Feature Group | Dims | Description |
|---|---|---|
| Pose landmarks | 99 | 33 body keypoints × 3 (x, y, z) — spatial context for where hands are relative to face/chest |
| Right hand (normalized) | 63 | 21 keypoints × 3, translated to wrist origin and scaled by wrist-to-knuckle distance |
| Left hand (normalized) | 63 | Same normalization as right hand |
| Right fingertip distances | 10 | Pairwise Euclidean distances between 5 fingertips (C(5,2) = 10 pairs), normalized by hand scale |
| Left fingertip distances | 10 | Same as right hand |
| Right palm normal | 3 | Unit cross-product vector of the palm plane (captures hand orientation/flip) |
| Left palm normal | 3 | Same as right hand |
| Inter-hand distance | 1 | Euclidean distance between left and right wrist (important for two-handed signs like "more") |

### Velocity Features

Frame-to-frame differences of all 252 base features are computed via `np.diff` and concatenated, producing **504 features per frame**. The first frame's velocity is zero-padded. Velocity captures the temporal dynamics of each sign — for example, the "bobbing" motion in "yes" or the "forward sweep" in "thank you".

### Sequence Tensor

Each model input is a tensor of shape **(30, 504)** — a 30-frame sliding window with 504 features per frame.

---

## 7. Model Training

Training is performed on a GPU workstation using TensorFlow/Keras. The pipeline is fully automated in `train.py` and includes three stages.

### Stage 1: Neural Architecture Search (NAS)

Keras Tuner's Hyperband algorithm searches over:

- **Architecture type:** 1D-CNN (Conv1D + MaxPool + Flatten) vs. LSTM
- **Hyperparameters:**
  - LSTM units / Conv1D filters: 32, 64, 96, or 128
  - Conv1D kernel size: 3 or 5
  - Dense layer units: 32 or 64
  - Dropout rate: 0.2, 0.3, 0.4, or 0.5
  - Learning rate: 1e-3 or 1e-4

NAS runs for up to 50 epochs on an 80/20 internal split and selects the architecture + hyperparameter combination with the highest validation accuracy.

### Stage 2: 5-Fold Stratified Cross-Validation

The best architecture from NAS is trained for 100 epochs on each of 5 stratified folds. This provides:

- **Mean validation accuracy** (generalization estimate)
- **Standard deviation** (stability measure)
- **Mean best epoch** (used as the fixed epoch count for final training, avoiding the need for an early-stopping validation set)

### Stage 3: Learning Rate Scheduler Comparison

Four LR schedulers are compared on the full train+val data, each training for the mean-best-epoch count from Stage 2:

1. **Constant** — fixed learning rate
2. **Cosine Decay** — smooth annealing to zero
3. **Exponential Decay** — decay rate 0.9 per 1/10th of total steps
4. **ReduceLROnPlateau** — halves LR after 5 epochs of no improvement

The scheduler yielding the highest held-out test accuracy is selected for the final exported model.

### Data Split

- **85%** train + validation pool (used in k-fold CV and final training)
- **15%** held-out test set (stratified, never seen during training or NAS)

All training runs log to TensorBoard for visualization.

<!-- TODO: Add training curves / confusion matrix screenshots -->
<!-- ![Training Curves](images/training_curves.png) -->
<!-- ![Confusion Matrix](images/confusion_matrix.png) -->

---

## 8. Model Optimization and Quantization

Two TFLite models are exported for on-device deployment:

### FP32 Model (`sign_model_fp32.tflite`)

- Standard float32 conversion from the trained Keras model.
- **Size: 628 KB**
- Op sets: `TFLITE_BUILTINS` + `SELECT_TF_OPS`

### INT8 Quantized Model (`sign_model_qat.tflite`)

- Post-training integer quantization using a representative dataset of 200 calibration samples from the training set.
- **Size: 163 KB** (~74% reduction from FP32)
- Op sets: `TFLITE_BUILTINS_INT8` + `TFLITE_BUILTINS` + `SELECT_TF_OPS`
- Input/output interfaces remain float32 for compatibility.

Both models can be **swapped at runtime** via radio buttons in the GUI, allowing real-time comparison of accuracy vs. speed.

---

## 9. Deployment on Raspberry Pi

### Threading Architecture

The application uses a multi-threaded design to keep the GUI responsive while running compute-heavy tasks concurrently:

```
Main Thread (Tkinter event loop)
  │
  ├── update_frame() — called every 15 ms
  │     ├── Picamera2 frame capture
  │     ├── MediaPipe Holistic processing (every 2nd frame)
  │     ├── Feature extraction + sequence buffer management
  │     └── Tkinter canvas rendering
  │
  ├── InferenceWorker (daemon thread)
  │     └── TFLite interpreter — processes (30, 504) sequences
  │         from a maxsize-1 queue (drops stale sequences)
  │
  ├── TTSWorker (daemon thread)
  │     └── espeak-ng subprocess — queue-based, flushes old
  │         requests when new speech arrives
  │
  └── STTWorker (daemon thread)
        └── PyAudio microphone capture + Google Speech API
            (active only in Listener mode)
```

### Key Optimizations

| Optimization | Description |
|---|---|
| MediaPipe frame skip | Holistic runs every 2nd frame; cached results reused for intermediate frames |
| Sliding window overlap | 30-frame window advances by 15 frames, not 30 — balances latency vs. compute |
| Minimum hand frames | 8 frames of continuous hand presence required before triggering inference |
| Inference queue (maxsize=1) | Drops stale sequences; only the latest is processed |
| TTS queue flush | New speech cancels any pending TTS to avoid audio pileup |
| Cooldown timer | 2-second cooldown between speaking the same word |
| Clean shutdown | `on_close()` stops all workers, releases camera, and closes MediaPipe models |

---

## 10. Application Interface

The Tkinter GUI is designed for the Pi's touchscreen and provides:

- **Live camera feed** (640×480 canvas) with overlaid MediaPipe pose and hand skeleton drawings.
- **Transcript box** — large text area (22pt bold font) showing the currently detected sign or transcribed speech.
- **Mode toggle button** — switches between:
  -    **Speaker mode:** Camera active → sign recognition → spoken audio output
  -    **Listener mode:** Microphone active → speech recognition → text display
- **Model selector** — FP32 / INT8 radio buttons to swap TFLite models at runtime.
- **Performance metrics** — real-time FPS and end-to-end inference latency (ms).
- **Status bar** — shows detection confidence, hand tracking state, and current mode.

---

## 11. Results and Performance

### Model Performance


| Metric | Value |
|---|---|
| Number of Classes | 19 |
| Confidence Threshold for Detection | 85% |
| End-to-End Inference Latency | ~0.5 ms | ~0.5 ms |
| GUI Frame Rate | ~20-30 FPS | ~20-30 FPS |

### Runtime Performance on Raspberry Pi 5

| Metric | FP32 Model | INT8 Model |
|---|---|---|
| Model File Size | 628 KB | 163 KB |
| Total Tensor RAM | 771.9 KB | 252.5 KB |
| x86 inference latency | 0.082ms | 0.023 ms |
| Test Accuracy | 99.12% | 99.12% |


---

## 12. Challenges and Learnings

- **Domain gap:** Models trained on laptop webcam data performed poorly on the Pi's camera. Solution: collected all training data directly on the Pi to match the deployment camera.
- **MediaPipe compute cost:** MediaPipe Holistic (pose + hands) is heavy for the Pi's CPU. Solution: process MediaPipe every 2nd frame and reuse cached results, cutting compute by ~50%.
- **Bluetooth audio latency:** Bluetooth adds ~100–200 ms latency. `espeak-ng` was chosen for fast synthesis to minimize total end-to-end delay.
- **Wrist normalization:** Normalizing hand landmarks relative to the wrist and scaling by hand size made the model robust to varying camera distances and hand sizes.
- **Speech to text** Initially planned to implement knowledge distillation on OpenAI's Whisper model, but deferred this to future iterations due to current resource constraints and extended training times.

---

## 13. Future Work

- **Expand vocabulary:** Add more signs and fingerspelling (A–Z alphabet) for open-ended communication.
- **Fully offline STT:** Replace Google Speech API with an on-device model (e.g., distil-whisper) for complete offline operation.
- **Sentence construction:** Combine sequential sign detections into grammatically correct English sentences using a lightweight language model.
- **Wearable form factor:** Migrate to a smaller SBC or custom PCB with a head-mounted or chest-mounted camera.
- **Continuous learning:** Allow users to add custom signs via few-shot on-device fine-tuning.

---

## 14. Reproducing the Project

### Prerequisites

- Raspberry Pi 5 (4 GB+ RAM) running Raspberry Pi OS Bookworm (64-bit)
- Raspberry Pi Camera Module (CSI ribbon cable)
- Display (touchscreen or HDMI monitor)
- Bluetooth speaker/microphone (or USB audio alternatives)
- Internet connection (for initial setup and Speech-to-Text API)

### Step 1: Clone the Repository

```bash
git clone https://github.com/<your-username>/Realtime_ASL_RasPi.git
cd Realtime_ASL_RasPi
```

### Step 2: Install System Dependencies

```bash
sudo apt update && sudo apt install -y \
    python3-pip python3-opencv espeak-ng portaudio19-dev \
    python3-tk libatlas-base-dev bluetooth bluez
```

### Step 3: Install Python Dependencies

```bash
pip3 install mediapipe numpy opencv-python Pillow \
    SpeechRecognition tflite-runtime picamera2
```

### Step 4: Pair Bluetooth Audio (if using Bluetooth speaker/mic)
```

### Step 5: Run the Two-Way Translator

```bash
cd Demo_RasPi
python3 pi_two_way_translator_V2.py
```

The GUI launches on the attached display. Press the mode toggle button to switch between Speaker (sign → speech) and Listener (speech → text).

### (Optional) Retrain the Model

#### 5a. Download the dataset

```bash
python3 fetch_dataset.py
```

#### 5b. Collect your own data (on Pi)

```bash
cd sign_to_speech/01_data_collection_raspi
python3 data_capture_raspi_V3.py
```

#### 5c. Preprocess videos into feature sequences

```bash
cd sign_to_speech/01_data_collection_raspi
python3 preprocess.py
```

#### 5d. Train (on a GPU machine, not the Pi)

```bash
cd sign_to_speech/02_training
pip install tensorflow keras-tuner scikit-learn
python3 train.py
```

Copy the output `.tflite` and `sign_labels.npy` files to the `Demo_RasPi/` directory on the Pi.

---

## 15. Project Structure

```
Realtime_ASL_RasPi/
│
├── README.md                          # Project report (this file)
├── fetch_dataset.py                   # Downloads training data from Google Drive
├── .gitignore
│
├── Demo_RasPi/                        # ── Final deployment application ──
│   ├── pi_two_way_translator_V2.py    # Main app: two-way translator (Speaker + Listener modes)
│   ├── sign_model_fp32.tflite         # FP32 TFLite model (628 KB)
│   └── sign_model_qat.tflite          # INT8 quantized TFLite model (163 KB)
│
└── sign_to_speech/                    # ── Training pipeline ──
    │
    ├── 01_data_collection_raspi/
    │   ├── data_capture_raspi_V3.py   # On-device video collection (19 signs × 60 clips)
    │   └── preprocess.py              # MediaPipe feature extraction + velocity stacking
    │
    ├── 02_training/
    │   └── train.py                   # NAS + K-Fold CV + LR scheduler comparison
    │                                  #   → exports sign_model_fp32.tflite
    │                                  #   → exports sign_model_qat.tflite
    │                                  #   → exports sign_labels.npy
    │
    └── 03_inference_raspi/
        ├── Inference_on_raspi.py      # Sign-to-Speech standalone version (earlier prototype)
        ├── sign_model_fp32.tflite     # Model copy for standalone testing
        └── sign_model_qat.tflite      # Model copy for standalone testing
```

---

## 16. References and Acknowledgements

- [MediaPipe Holistic](https://google.github.io/mediapipe/solutions/holistic.html) — Real-time pose and hand landmark detection (Google)
- [TensorFlow Lite](https://www.tensorflow.org/lite/guide) — On-device ML inference framework
- [Keras Tuner](https://keras.io/keras_tuner/) — Hyperparameter tuning and neural architecture search
- [espeak-ng](https://github.com/espeak-ng/espeak-ng) — Open-source multilingual text-to-speech engine
- [SpeechRecognition](https://pypi.org/project/SpeechRecognition/) — Python library wrapping multiple speech-to-text APIs
- [Picamera2](https://github.com/raspberrypi/picamera2) — Official Raspberry Pi camera Python library

### AI Tools Disclosure

> GitHub Copilot was used for boilerplate Tkinter GUI layout. Claude was used to generate the README documentation. All model architecture decisions, feature engineering, and system design were done manually."

---

## Team Members

| Name | Affiliation | Email | Contribution |
| :--- | :--- | :--- | :--- |
| 1. Liz Maria George | * | lizgeorge@iisc.ac.in | Dataset collection, cleanup, feature engineering |
| 2. Naznin Amirul Haque | * | nazninhaque@iisc.ac.in | Dataset collection, cleanup, feature engineering |
| 3. Ayush Kumar | * | ayush1@iisc.ac.in | Top level integration, GUI development, Data collection GUI |
| 4. Prayanshu Sharma | * | prayanshus@iisc.ac.in | Dataset collection, model development |

*M.Tech. (MVLSI), ECE, IISc

---

## Video Presentation

In the folder **Demo_RasPi**
