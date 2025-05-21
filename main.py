from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import base64
import cv2
import numpy as np
import mediapipe as mp
import pickle
from typing import List, Optional
import tempfile
from gtts import gTTS
import os
import time
import pygame
import asyncio

app = FastAPI()

# Initialize pygame mixer for audio
pygame.mixer.init()

# Load the model
try:
    model_path = "model.p"
    model_dict = pickle.load(open(model_path, 'rb'))
    model = model_dict['model']
except Exception as e:
    raise RuntimeError(f"Failed to load model: {e}")

# Initialize MediaPipe Hands
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=2,
    min_detection_confidence=0.3
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global variables for state management
LAST_ENTER_TIME = 0
ENTER_COOLDOWN = 3.0  # 3 seconds cooldown for enter gesture
SPEECH_IN_PROGRESS = False  # Flag to track speech state
CAMERA_ENABLED = False  # Camera state flag

class ImageData(BaseModel):
    image: Optional[str] = None  # base64 image (optional when camera off)
    sentence: List[str] = []  # current sentence state
    last_char_time: float = 0  # timestamp of last character
    char_delay: float = 1.5  # delay between characters
    initial_delay: float = 2  # initial delay
    camera_state: Optional[bool] = None  # to sync camera state

class PredictionResult(BaseModel):
    prediction: str
    sentence: List[str]
    last_char_time: float
    camera_state: bool  # current camera state

class CameraControl(BaseModel):
    enable: bool

@app.post("/camera")
async def control_camera(control: CameraControl):
    """Endpoint to enable/disable camera"""
    global CAMERA_ENABLED
    CAMERA_ENABLED = control.enable
    return {"success": True, "camera_state": CAMERA_ENABLED}

def count_fingers(hand_landmarks, hand_label):
    """Detect open palm (5 fingers open)"""
    tips_ids = [4, 8, 12, 16, 20]
    fingers = []

    # Thumb
    if hand_label == "Right":
        fingers.append(hand_landmarks.landmark[tips_ids[0]].x < hand_landmarks.landmark[tips_ids[0] - 1].x)
    else:
        fingers.append(hand_landmarks.landmark[tips_ids[0]].x > hand_landmarks.landmark[tips_ids[0] - 1].x)

    # Other 4 fingers
    for i in range(1, 5):
        fingers.append(hand_landmarks.landmark[tips_ids[i]].y < hand_landmarks.landmark[tips_ids[i] - 2].y)

    return sum(fingers)

def detect_space_gesture(hand_landmarks, hand_label):
    """Detect space gesture (fist with thumb up)"""
    tips_ids = [4, 8, 12, 16, 20]
    
    # Check if thumb is extended
    thumb_extended = False
    if hand_label == "Right":
        thumb_extended = (hand_landmarks.landmark[tips_ids[0]].y < hand_landmarks.landmark[2].y)
    else:
        thumb_extended = (hand_landmarks.landmark[tips_ids[0]].y < hand_landmarks.landmark[2].y)
    
    # Check if other fingers are folded
    other_fingers_folded = True
    for i in range(1, 5):
        if hand_landmarks.landmark[tips_ids[i]].y < hand_landmarks.landmark[tips_ids[i] - 2].y:
            other_fingers_folded = False
            break
    
    return thumb_extended and other_fingers_folded

async def speak_and_reset(text: str):
    """Handle text-to-speech and reset the sentence"""
    global SPEECH_IN_PROGRESS
    
    if not text.strip():
        return []
        
    SPEECH_IN_PROGRESS = True
    try:
        tts = gTTS(text=text, lang='id')
        with tempfile.NamedTemporaryFile(delete=True, suffix=".mp3") as fp:
            fp_name = fp.name
            tts.save(fp_name)
            pygame.mixer.music.load(fp_name)
            pygame.mixer.music.play()
            
            # Wait for speech to complete
            while pygame.mixer.music.get_busy():
                await asyncio.sleep(0.1)
                
        return []  # Return empty list to reset the sentence
        
    except Exception as e:
        print(f"Error in text-to-speech: {e}")
        return []
    finally:
        SPEECH_IN_PROGRESS = False

@app.post("/predict")
async def predict(data: ImageData):
    global LAST_ENTER_TIME, SPEECH_IN_PROGRESS, CAMERA_ENABLED
    
    try:
        # Update camera state if provided
        if data.camera_state is not None:
            CAMERA_ENABLED = data.camera_state
        
        # Skip processing if speech is in progress or camera is disabled
        if SPEECH_IN_PROGRESS or not CAMERA_ENABLED:
            return PredictionResult(
                prediction="",
                sentence=data.sentence,
                last_char_time=data.last_char_time,
                camera_state=CAMERA_ENABLED
            )
        
        # Skip if no image provided when camera is enabled
        if not data.image:
            return PredictionResult(
                prediction="",
                sentence=data.sentence,
                last_char_time=data.last_char_time,
                camera_state=CAMERA_ENABLED
            )
        
        # Decode base64 image to NumPy array
        image_data = base64.b64decode(data.image.split(",")[1])
        npimg = np.frombuffer(image_data, np.uint8)
        frame = cv2.imdecode(npimg, cv2.IMREAD_COLOR)
        
        # Flip frame for mirror view
        frame = cv2.flip(frame, 1)
        
        # Convert to RGB for MediaPipe
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Process with MediaPipe
        results = hands.process(rgb_frame)
        
        current_char = ""
        new_sentence = data.sentence.copy()
        last_char_time = data.last_char_time
        current_time = time.time()

        if results.multi_hand_landmarks:
            hands_count = len(results.multi_hand_landmarks)
            data_aux = []
            x_ = []
            y_ = []

            for i, hand_landmarks in enumerate(results.multi_hand_landmarks):
                for lm in hand_landmarks.landmark:
                    x_.append(lm.x)
                    y_.append(lm.y)

                for lm in hand_landmarks.landmark:
                    data_aux.append(lm.x - min(x_))
                    data_aux.append(lm.y - min(y_))

                # Check gestures on first hand
                if i == 0 and results.multi_handedness:
                    label = results.multi_handedness[i].classification[0].label
                    
                    if detect_space_gesture(hand_landmarks, label):
                        current_char = "space"
                    elif count_fingers(hand_landmarks, label) == 5:
                        current_char = "enter"

            # If not a special gesture, predict letter
            if current_char != "enter" and current_char != "space":
                if hands_count == 1:
                    data_aux += [0] * 42  # padding
                if len(data_aux) == 84:
                    prediction = model.predict([np.asarray(data_aux)])
                    current_char = str(prediction[0]).lower()

            # Calculate time delay between characters
            time_since_last = current_time - last_char_time

            if time_since_last > data.initial_delay:
                if time_since_last > data.char_delay:
                    if current_char == "enter":
                        # Check cooldown period
                        if current_time - LAST_ENTER_TIME > ENTER_COOLDOWN:
                            spoken_text = "".join(new_sentence).strip()
                            if spoken_text:
                                print(f"Mengucapkan: {spoken_text}")
                                new_sentence = await speak_and_reset(spoken_text)
                            LAST_ENTER_TIME = current_time
                            last_char_time = current_time
                    elif current_char == "space":
                        new_sentence.append(" ")
                        print("Spasi ditambahkan")
                        last_char_time = current_time
                    elif current_char:  # Regular character
                        new_sentence.append(current_char.upper())
                        last_char_time = current_time

        return PredictionResult(
            prediction=current_char.upper() if current_char else "",
            sentence=new_sentence,
            last_char_time=last_char_time,
            camera_state=CAMERA_ENABLED
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/clear")
async def clear_sentence():
    """Endpoint to clear the sentence"""
    global LAST_ENTER_TIME
    LAST_ENTER_TIME = 0
    return {"sentence": [], "last_char_time": 0, "camera_state": CAMERA_ENABLED}