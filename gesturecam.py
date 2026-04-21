import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import pyautogui
import numpy as np
import time
from collections import deque
import os
import json
from datetime import datetime
import sys
import pygetwindow as gw
import threading
import subprocess
import webbrowser
import difflib
import re

# Fix Unicode encoding for Windows
if sys.platform == 'win32':
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

# Try to import speech recognition
try:
    import speech_recognition as sr
    VOICE_MODE_AVAILABLE = True
    print("[✓] speech_recognition imported successfully!")
except ImportError:
    VOICE_MODE_AVAILABLE = False
    print("[!] speech_recognition not installed. Voice mode disabled.")
    print("[*] Install with: pip install SpeechRecognition pydub pyaudio")
    print("[*] On Windows, you may also need: pip install pipwin && pipwin install pyaudio")
    sys.stdout.flush()

# ----------- DEFAULT SETTINGS -----------
DEFAULT_CONFIG = {
    'SMOOTHING': 0.92,
    'CLICK_THRESHOLD': 35,
    'STABILITY_FRAMES': 1,
    'SCROLL_THRESHOLD': 50,
    'DOUBLE_CLICK_DELAY': 0.15,
    'MIN_HAND_CONFIDENCE': 0.4,
    'CAM_WIDTH': 640,
    'CAM_HEIGHT': 480,
}

CONFIG_FILE = 'gesture_config.json'

# Load or create config
def load_config():
    if os.path.exists(CONFIG_FILE):
        try:
            with open(CONFIG_FILE, 'r') as f:
                return {**DEFAULT_CONFIG, **json.load(f)}
        except:
            return DEFAULT_CONFIG.copy()
    return DEFAULT_CONFIG.copy()

config = load_config()
SMOOTHING = config['SMOOTHING']
CLICK_THRESHOLD = config['CLICK_THRESHOLD']
STABILITY_FRAMES = config['STABILITY_FRAMES']
SCROLL_THRESHOLD = config['SCROLL_THRESHOLD']
DOUBLE_CLICK_DELAY = config['DOUBLE_CLICK_DELAY']
MIN_HAND_CONFIDENCE = config['MIN_HAND_CONFIDENCE']
CAM_WIDTH = config['CAM_WIDTH']
CAM_HEIGHT = config['CAM_HEIGHT']
SCREEN_WIDTH, SCREEN_HEIGHT = pyautogui.size()
CURSOR_RADIUS = 15

# Mediapipe initialization
model_path = 'hand_landmarker.task'
if not os.path.exists(model_path):
    print(f"Error: {model_path} not found!")
    print("\nDownloading hand_landmarker.task...")
    try:
        import urllib.request
        url = "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task"
        urllib.request.urlretrieve(url, model_path)
        print(f"✓ Successfully downloaded {model_path}")
        sys.stdout.flush()
    except Exception as e:
        print(f"✗ Failed to download model: {e}")
        print("\nManual download: Download from:")
        print("https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task")
        print("and place it in the same directory as gesture.py")
        sys.stdout.flush()
        exit(1)

BaseOptions = mp.tasks.BaseOptions
HandLandmarker = vision.HandLandmarker
HandLandmarkerOptions = vision.HandLandmarkerOptions
VisionRunningMode = vision.RunningMode

options = HandLandmarkerOptions(
    base_options=BaseOptions(model_asset_path=model_path),
    running_mode=VisionRunningMode.IMAGE,
    num_hands=2,
    min_hand_detection_confidence=MIN_HAND_CONFIDENCE,
    min_hand_presence_confidence=MIN_HAND_CONFIDENCE,
    min_tracking_confidence=MIN_HAND_CONFIDENCE
)

landmarker = HandLandmarker.create_from_options(options)

# State variables
prev_x, prev_y = SCREEN_WIDTH // 2, SCREEN_HEIGHT // 2
curr_x, curr_y = prev_x, prev_y
gesture_history = deque(maxlen=STABILITY_FRAMES)
hand_position_history = deque(maxlen=10)
last_click_time = time.time()
last_right_click_time = time.time()
last_double_click_time = time.time()
last_minimize_time = time.time()
is_dragging = False
drag_start_pos = None
fps_counter = deque(maxlen=30)
show_settings_menu = False
drawing_mode = False
canvas = None
voice_mode = False
voice_thread = None
listening = False
last_voice_toggle_time = time.time()
powerpoint_mode = False
last_powerpoint_open_time = time.time()
last_slide_change_time = time.time()
last_thumb_x = None


# Capture webcam
cap = cv2.VideoCapture(0)
cap.set(3, CAM_WIDTH)
cap.set(4, CAM_HEIGHT)
cap.set(cv2.CAP_PROP_FPS, 60)
cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)  # Minimize buffer for real-time response
if not cap.isOpened():
    print("Error: Could not open camera")
    sys.stdout.flush()
    exit(1)

print("[DEBUG] Camera initialized successfully")
sys.stdout.flush()

# ========== HELPER FUNCTIONS ==========

def get_distance(p1, p2):
    """Calculate Euclidean distance"""
    return np.sqrt((p1[0]-p2[0])**2 + (p1[1]-p2[1])**2)

def get_distance_squared(p1, p2):
    """Calculate squared distance (faster for comparisons, avoid sqrt)"""
    return (p1[0]-p2[0])**2 + (p1[1]-p2[1])**2

def get_hand_velocity(position_history):
    """Calculate hand movement velocity"""
    if len(position_history) < 2:
        return 0
    return get_distance(position_history[-1], position_history[0])

def is_finger_extended(landmarks, tip_idx, pip_idx):
    """Check if finger is extended"""
    return landmarks[tip_idx].y < landmarks[pip_idx].y

def count_extended_fingers(landmarks):
    """Count extended fingers (optimized with direct comparisons)"""
    count = 0
    if landmarks[4].x < landmarks[3].x: count += 1
    if landmarks[8].y < landmarks[6].y: count += 1
    if landmarks[12].y < landmarks[10].y: count += 1
    if landmarks[16].y < landmarks[14].y: count += 1
    if landmarks[20].y < landmarks[18].y: count += 1
    return count

def is_hand_in_frame(landmarks, margin=20):
    """Check if all hand landmarks are within frame boundaries"""
    for lm in landmarks:
        x = lm.x * CAM_WIDTH
        y = lm.y * CAM_HEIGHT
        if x < margin or x > CAM_WIDTH - margin or y < margin or y > CAM_HEIGHT - margin:
            return False
    return True

def are_both_hands_detected(results):
    """Check if both hands are detected"""
    if not results.hand_landmarks or len(results.hand_landmarks) < 2:
        return False
    return True

def detect_gesture(landmarks, lm_list, current_time):
    """Detect hand gestures (optimized with squared distances)"""
    # Only detect gestures if hand is fully within frame
    if not is_hand_in_frame(landmarks):
        return {
            'type': 'none',
            'confidence': 0,
            'distances': {
                'thumb_index': 0,
                'index_middle': 0
            }
        }
    
    # Pre-extract tips for faster access
    index_tip = lm_list[8]
    middle_tip = lm_list[12]
    thumb_tip = lm_list[4]
    
    # Use squared distances to avoid sqrt (1225 = 35^2)
    thumb_index_dist_sq = get_distance_squared(index_tip, thumb_tip)
    thumb_index_dist = np.sqrt(thumb_index_dist_sq) if thumb_index_dist_sq < 2000 else 100
    
    # Get extended fingers count
    extended_fingers = count_extended_fingers(landmarks)
    
    gesture = {
        'type': 'none',
        'confidence': 0,
        'distances': {
            'thumb_index': thumb_index_dist,
            'index_middle': 0
        }
    }
    
    # Pinch gesture (squared distance comparison is faster)
    if thumb_index_dist_sq < 1225 and not (landmarks[12].y < landmarks[10].y and landmarks[16].y < landmarks[14].y and landmarks[20].y < landmarks[18].y):
        gesture['type'] = 'pinch'
        gesture['confidence'] = 1.0
    # Open PowerPoint (3 fingers)
    elif extended_fingers == 3 and landmarks[8].y < landmarks[6].y and landmarks[12].y < landmarks[10].y and landmarks[16].y < landmarks[14].y:
        gesture['type'] = 'open_ppt'
        gesture['confidence'] = 1.0
    # Left click (2 fingers)
    elif extended_fingers == 2:
        gesture_history.append('left_click')
        if len(gesture_history) == STABILITY_FRAMES and all(g == 'left_click' for g in gesture_history):
            if current_time - last_click_time > DOUBLE_CLICK_DELAY:
                gesture['type'] = 'left_click'
                gesture['confidence'] = 1.0
    # Right click (pinky only)
    elif extended_fingers == 1 and landmarks[20].y < landmarks[18].y:
        gesture_history.append('right_click')
        if len(gesture_history) == STABILITY_FRAMES and all(g == 'right_click' for g in gesture_history):
            if current_time - last_right_click_time > DOUBLE_CLICK_DELAY:
                gesture['type'] = 'right_click'
                gesture['confidence'] = 1.0
    # Fist
    elif extended_fingers == 0:
        gesture['type'] = 'fist'
        gesture['confidence'] = 0.9
    # Open hand
    elif extended_fingers >= 4:
        gesture['type'] = 'open_hand'
        gesture['confidence'] = 0.7
    return gesture

def draw_hand_landmarks(image, landmarks):
    """Draw hand skeleton (optimized with pre-computed connections)"""
    h, w = image.shape[:2]
    connections = [
        (0, 1), (1, 2), (2, 3), (3, 4),
        (0, 5), (5, 6), (6, 7), (7, 8),
        (0, 9), (9, 10), (10, 11), (11, 12),
        (0, 13), (13, 14), (14, 15), (15, 16),
        (0, 17), (17, 18), (18, 19), (19, 20),
        (5, 9), (9, 13), (13, 17)
    ]
    
    # Convert all landmarks to pixel coordinates first (batch operation)
    pixels = [(int(lm.x * w), int(lm.y * h)) for lm in landmarks]
    
    # Draw connections
    for start, end in connections:
        cv2.line(image, pixels[start], pixels[end], (0, 255, 100), 2)
    
    # Draw joints (only key points)
    for i in [4, 8, 12, 16, 20]:
        cv2.circle(image, pixels[i], 5, (0, 255, 0), -1)
    
    # Draw wrist
    cv2.circle(image, pixels[0], 5, (200, 100, 0), -1)

def draw_ui(image, gesture, hand_confidence, fps, num_hands_detected=0, voice_mode_active=False):
    """Draw on-screen UI (optimized)"""
    h, w = image.shape[:2]

    # Cache color map
    color_map = {
        'left_click': (0, 255, 0),
        'right_click': (0, 0, 255),
        'pinch': (255, 0, 255),
        'scroll': (255, 255, 0),
        'fist': (255, 100, 0),
        'open_hand': (100, 255, 255)
    }

    # Top-left info (minimal text operations)
    cv2.putText(image, f"FPS: {fps:.1f}", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

    # Show hand count
    hand_text = f"Hands: {num_hands_detected}"
    hand_color = (0, 255, 0) if num_hands_detected >= 2 else (0, 165, 255)
    cv2.putText(image, hand_text, (10, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.7, hand_color, 2)

    # Show voice mode status
    voice_text = "VOICE: ON" if voice_mode_active else "VOICE: OFF"
    voice_color = (0, 255, 0) if voice_mode_active else (0, 0, 255)
    cv2.putText(image, voice_text, (10, 110), cv2.FONT_HERSHEY_SIMPLEX, 0.7, voice_color, 2)

    # Gesture feedback
    if gesture['type'] != 'none':
        color = color_map.get(gesture['type'], (200, 200, 200))
        text = f"GESTURE: {gesture['type'].upper()}"
        cv2.putText(image, text, (w//2 - 150, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)

    # Distance info
    cv2.putText(image, f"Thumb-Index: {gesture['distances']['thumb_index']:.1f}", (10, h-30), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 0, 0), 1)

def draw_settings_menu(image):
    """Draw settings adjustment menu"""
    h, w, c = image.shape
    overlay = image.copy()
    cv2.rectangle(overlay, (50, 50), (w-50, h-50), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.7, image, 0.3, 0, image)
    
    text = [
        "SETTINGS MENU",
        "",
        f"1: SMOOTHING = {SMOOTHING:.2f} (Press +/- to adjust)",
        f"2: CLICK_THRESHOLD = {CLICK_THRESHOLD} (Press I/O to adjust)",
        f"3: STABILITY_FRAMES = {STABILITY_FRAMES} (Press U/D to adjust)",
        "",
        "S: Save Config | C: Close Menu | ESC: Exit"
    ]
    
    y = 80
    for line in text:
        cv2.putText(image, line, (70, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 1)
        y += 40




def print_help():
    """Print help menu"""
    help_text = """
╔═══════════════════════════════════════════════════════════╗
║       HAND GESTURE MOUSE CONTROL - HELP MENU             ║
╠═══════════════════════════════════════════════════════════╣
║ GESTURES:                                                 ║
║   • Peace Sign (V - 2 Fingers) → LEFT CLICK              ║
║   • Pinky Pointing (1 Finger) → RIGHT CLICK              ║
║   • Three Fingers (Index+Middle+Ring) → OPEN POWERPOINT  ║
║   • Open Hand (5 Fingers) → STOP / PAUSE                 ║
║   • Fist (0 Fingers - Single) → MINIMIZE WINDOW          ║
║   • Index/Middle Swipe L/R (in PPT) → Change Slides      ║
║                                                           ║
║ VOICE COMMANDS:                                           ║
║   "click" / "left click" → LEFT CLICK                    ║
║   "right click" → RIGHT CLICK                            ║
║   "double click" → DOUBLE CLICK                          ║
║   "scroll up" / "scroll down" → SCROLL                   ║
║   "play" / "resume" / "start" → PLAY MEDIA              ║
║   "pause" → PAUSE MEDIA                                  ║
║   "set/increase/decrease volume <n>" → VOLUME CONTROL     ║
║   "close" / "minimize" → MINIMIZE WINDOW                 ║
║   "close <app>" → CLOSE SPECIFIC APP / WINDOW             ║
║   "open powerpoint" / "ppt" → OPEN POWERPOINT            ║
║   "slideshow" / "start slideshow" → START PRESENTATION   ║
║   "stop" / "quit voice" → STOP VOICE MODE                ║
║                                                           ║
║ KEYBOARD SHORTCUTS:                                       ║
║   ESC → Exit Program                                      ║
║   H → Show this help                                      ║
║   S → Settings Menu                                       ║
║   C → Calibrate (adjust for your hand size)              ║
║   V → Toggle Voice Mode                                  ║
║                                                           ║
║   +/- → Adjust Smoothing                                 ║
║   I/O → Adjust Click Threshold                           ║
║   U/D → Adjust Stability Frames                          ║
║                                                           ║
║ TIPS:                                                     ║
║   • Keep hand in good lighting                           ║
║   • Position hand 30-40cm from camera                    ║
║   • Move hand smoothly for better tracking               ║
║   • Hold gesture for 0.3s before click registers         ║
║   • Speak clearly for voice commands                     ║
║                                                           ║
║ CONFIG FILE: gesture_config.json                          ║
╚═══════════════════════════════════════════════════════════╝
"""
    print(help_text)

# Track opened applications for closing
opened_apps = {}


def normalize_name(s: str) -> str:
    s = s.lower()
    s = s.replace('_', ' ').replace('-', ' ')
    s = s.replace('  ', ' ')
    s = s.replace(' dot ', '.')
    s = s.replace(' dot', '.')
    s = s.replace('dot ', '.')
    s = s.strip()
    return s


def find_file(filename):
    """Search for a file in common directories (includes Start Menu and Program Files)."""
    search_paths = [
        os.path.expanduser("~\\Desktop"),
        os.path.expanduser("~\\Downloads"),
        os.path.expanduser("~\\Documents"),
        os.path.expanduser("~\\Videos"),
        os.path.expanduser("~\\Pictures"),
        os.path.expanduser("~\\Music"),
        os.path.expandvars(r"%APPDATA%\\Microsoft\\Windows\\Start Menu\\Programs"),
        os.path.expandvars(r"%ProgramData%\\Microsoft\\Windows\\Start Menu\\Programs"),
        os.path.join(os.environ.get('ProgramFiles', r'C:\\Program Files')),
        os.path.join(os.environ.get('ProgramFiles(x86)', r'C:\\Program Files (x86)')),
    ]

    # Normalize search name
    search_name = normalize_name(filename)
    print(f"[DEBUG] Searching for: {search_name}")

    candidates = []  # (full_path, display_name, norm_name)

    for search_path in search_paths:
        if not search_path or not os.path.exists(search_path):
            continue

        try:
            max_depth = 4
            base_depth = search_path.count(os.sep)
            for root, dirs, files in os.walk(search_path):
                depth = root.count(os.sep) - base_depth
                if depth > max_depth:
                    dirs[:] = []
                    continue

                for item in files:
                    full_path = os.path.join(root, item)
                    display = item
                    norm = normalize_name(os.path.splitext(item)[0])
                    candidates.append((full_path, display, norm))

                for item in dirs:
                    full_path = os.path.join(root, item)
                    display = item
                    norm = normalize_name(item)
                    candidates.append((full_path, display, norm))

        except PermissionError:
            continue
        except Exception as e:
            print(f"[DEBUG] Error searching {search_path}: {e}")
            continue

    # 1) Exact normalized match or substring/startswith
    for full_path, display, norm in candidates:
        if search_name == norm or norm.startswith(search_name) or (len(search_name) > 3 and search_name in norm):
            print(f"[DEBUG] Exact/sub match -> {full_path}")
            return full_path

    # 2) Contains match
    for full_path, display, norm in candidates:
        if search_name in norm:
            print(f"[DEBUG] Contains match -> {full_path}")
            return full_path

    # 3) Fuzzy match
    if candidates:
        names = [c[2] for c in candidates]
        matches = difflib.get_close_matches(search_name, names, n=3, cutoff=0.6)
        if matches:
            for m in matches:
                for full_path, display, norm in candidates:
                    if norm == m:
                        print(f"[DEBUG] Fuzzy match -> {full_path} (matched '{m}')")
                        return full_path

        closest = difflib.get_close_matches(search_name, names, n=3, cutoff=0.0)
        if closest:
            print(f"[DEBUG] No strong match. Closest candidates: {closest}")

    print(f"[DEBUG] File not found in search paths")
    return None

def open_file_by_name(filename):
    """Open a file by searching and launching it"""
    # Normalize spoken filename: remove filler words
    original = filename
    filename = filename.lower().strip()
    for prefix in ['the ', 'my ', 'please ', 'open ']:
        if filename.startswith(prefix):
            filename = filename[len(prefix):].strip()
    print(f"[VOICE] Attempting to open: '{original}' -> searching for '{filename}'")
    sys.stdout.flush()

    file_path = find_file(filename)
    if file_path:
        try:
            print(f"[VOICE] → Opening: {file_path}")
            sys.stdout.flush()
            os.startfile(file_path)
            time.sleep(0.5)
            print(f"[VOICE] ✓ Opened: {file_path}")
            sys.stdout.flush()
            return True
        except Exception as e:
            print(f"[!] Failed to open file: {e}")
            sys.stdout.flush()
            return False
    else:
        # Provide helpful debug suggestions (closest matches)
        print(f"[VOICE] ✗ File not found: {filename}")
        sys.stdout.flush()
        # gather some candidate names to show to user
        debug_candidates = []
        for search_path in [os.path.expanduser('~\\Desktop'), os.path.expanduser('~\\Downloads'), os.path.expanduser('~\\Documents'), os.path.expanduser('~\\Pictures'), os.path.expanduser('~\\Videos'), os.path.expandvars(r'%APPDATA%\\Microsoft\\Windows\\Start Menu\\Programs')]:
            try:
                if not os.path.exists(search_path):
                    continue
                for root, dirs, files in os.walk(search_path):
                    for item in list(files)[:20]:
                        debug_candidates.append(item)
                    for item in list(dirs)[:20]:
                        debug_candidates.append(item)
                    break
            except Exception:
                continue
        if debug_candidates:
            normalized = [normalize_name(os.path.splitext(x)[0]) for x in debug_candidates]
            close = difflib.get_close_matches(filename, normalized, n=5, cutoff=0.3)
            if close:
                print(f"[DEBUG] Closest names: {close}")
                sys.stdout.flush()
        return False

def close_active_window():
    """Close the currently active window"""
    try:
        # Add a small delay to ensure focus
        time.sleep(0.2)
        pyautogui.hotkey('alt', 'F4')
        time.sleep(0.3)
        print("[VOICE] ✓ Window closed")
        return True
    except Exception as e:
        print(f"[VOICE] ✗ Failed to close window: {e}")
        return False


# ---------------- Media & Volume Helpers ----------------
# Try to use pycaw/comtypes for precise volume control (Windows). If not available, we fallback to best-effort.
_audio_volume_iface = None

def _init_audio_iface():
    """Initialize and cache the audio endpoint interface (pycaw based) if available."""
    global _audio_volume_iface
    if _audio_volume_iface is not None:
        return True
    try:
        from ctypes import POINTER, cast
        from comtypes import CLSCTX_ALL  # type: ignore
        from comtypes.client import CreateObject  # type: ignore
        from pycaw.pycaw import AudioUtilities, IAudioEndpointVolume  # type: ignore
        devices = AudioUtilities.GetSpeakers()
        interface = devices.Activate(IAudioEndpointVolume._iid_, CLSCTX_ALL, None)
        _audio_volume_iface = cast(interface, POINTER(IAudioEndpointVolume))
        print('[Audio] pycaw audio interface initialized')
        return True
    except (ImportError, Exception):
        _audio_volume_iface = None
        print('[Audio] pycaw/comtypes not available - precise volume control disabled')
        return False


def get_current_volume_percent():
    """Return current master volume as 0-100 int, or None if unavailable."""
    if sys.platform != 'win32':
        return None
    if not _init_audio_iface():
        return None
    try:
        v = _audio_volume_iface.GetMasterVolumeLevelScalar()
        return int(round(v * 100))
    except Exception:
        return None


def set_volume_percent(percent):
    """Set master volume to percent (0-100). Returns True on success."""
    percent = max(0, min(100, int(percent)))
    if sys.platform != 'win32':
        print('[Audio] Volume set not supported on this platform')
        return False
    if _init_audio_iface():
        try:
            _audio_volume_iface.SetMasterVolumeLevelScalar(percent / 100.0, None)
            print(f'[Audio] Volume set to {percent}%')
            return True
        except Exception as e:
            print(f'[Audio] Failed to set volume via pycaw: {e}')
    # Fallback: approximate via 'volumeup'/'volumedown' key presses if we can read current
    cur = get_current_volume_percent()
    if cur is None:
        print('[Audio] Precise control not available. Install pycaw and comtypes for exact volume control.')
        return False
    diff = percent - cur
    key = 'volumeup' if diff > 0 else 'volumedown'
    steps = int(abs(diff) / 2)  # each hardware step approximated as ~2%
    steps = max(1, steps) if diff != 0 else 0
    for _ in range(steps):
        try:
            pyautogui.press(key)
        except Exception:
            pass
        time.sleep(0.01)
    print(f'[Audio] Approximated volume change by pressing {key} x {steps}')
    return True


def change_volume_by(delta):
    """Change volume by delta percentage points (positive or negative)"""
    cur = get_current_volume_percent()
    if cur is None:
        print('[Audio] Cannot read current volume; install pycaw/comtypes for precise control')
        return False
    return set_volume_percent(cur + int(delta))


def send_media_play():
    """Send a play media command (Windows broadcast)."""
    try:
        if sys.platform == 'win32':
            import ctypes
            user32 = ctypes.windll.user32
            HWND_BROADCAST = 0xFFFF
            WM_APPCOMMAND = 0x319
            APPCOMMAND_MEDIA_PLAY = 46
            user32.SendMessageW(HWND_BROADCAST, WM_APPCOMMAND, 0, APPCOMMAND_MEDIA_PLAY << 16)
            print('[Media] Play command sent')
            return True
        else:
            try:
                pyautogui.press('play')
                print('[Media] Play key pressed')
                return True
            except Exception:
                print('[Media] Play not available on this platform')
                return False
    except Exception as e:
        print(f"[Media] Failed to send play: {e}")
        return False


def send_media_pause():
    """Send a pause media command (Windows broadcast)."""
    try:
        if sys.platform == 'win32':
            import ctypes
            user32 = ctypes.windll.user32
            HWND_BROADCAST = 0xFFFF
            WM_APPCOMMAND = 0x319
            APPCOMMAND_MEDIA_PAUSE = 47
            user32.SendMessageW(HWND_BROADCAST, WM_APPCOMMAND, 0, APPCOMMAND_MEDIA_PAUSE << 16)
            print('[Media] Pause command sent')
            return True
        else:
            try:
                pyautogui.press('pause')
                print('[Media] Pause key pressed')
                return True
            except Exception:
                print('[Media] Pause not available on this platform')
                return False
    except Exception as e:
        print(f"[Media] Failed to send pause: {e}")
        return False


def send_media_play_pause():
    """Send a play/pause media command (Windows broadcast)."""
    try:
        if sys.platform == 'win32':
            import ctypes
            user32 = ctypes.windll.user32
            HWND_BROADCAST = 0xFFFF
            WM_APPCOMMAND = 0x319
            APPCOMMAND_MEDIA_PLAY_PAUSE = 14
            user32.SendMessageW(HWND_BROADCAST, WM_APPCOMMAND, 0, APPCOMMAND_MEDIA_PLAY_PAUSE << 16)
            print('[Media] Play/Pause command sent')
            return True
        else:
            try:
                pyautogui.press('playpause')
                print('[Media] Play/Pause key pressed')
                return True
            except Exception:
                print('[Media] Play/Pause not available on this platform')
                return False
    except Exception as e:
        print(f'[Media] Failed to send play/pause: {e}')
        return False


def close_app_by_name(name):
    """Close application windows or processes matching the given name."""
    name_lower = name.lower()
    found = False

    # Try to close windows whose title contains the name
    try:
        for w in gw.getAllWindows():
            title = (w.title or '').strip()
            if title and name_lower in title.lower():
                try:
                    print(f"[VOICE] → Closing window with title: {title}")
                    w.close()
                    found = True
                except Exception as e:
                    print(f"[DEBUG] Failed to close window '{title}': {e}")
                    try:
                        w.activate()
                        time.sleep(0.2)
                        pyautogui.hotkey('alt', 'F4')
                        found = True
                    except Exception:
                        pass
        if found:
            print(f"[VOICE] ✓ Closed window(s) matching: {name}")
            return True
    except Exception as e:
        print(f"[DEBUG] pygetwindow error: {e}")

    # Map common app names to executable names and try to kill process
    proc_map = {
        'chrome': 'chrome.exe',
        'edge': 'msedge.exe',
        'spotify': 'Spotify.exe',
        'zoom': 'Zoom.exe',
        'vlc': 'vlc.exe',
        'powerpoint': 'powerpnt.exe',
        'word': 'winword.exe',
        'excel': 'excel.exe',
        'notepad': 'notepad.exe',
        'explorer': 'explorer.exe',
        'file explorer': 'explorer.exe',
    }

    exe = None
    for key, exe_name in proc_map.items():
        if key in name_lower:
            exe = exe_name
            break

    if exe:
        try:
            subprocess.run(['taskkill', '/F', '/IM', exe], check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            print(f"[VOICE] ✓ Killed process: {exe}")
            return True
        except Exception as e:
            print(f"[!] Failed to kill {exe}: {e}")

    print(f"[VOICE] ✗ No matching window or process found for: {name}")
    return False


def process_voice_command(command, execute=True):
    """Interpret and execute a recognized voice command.

    Returns:
      (handled: bool, action: Optional[str]) where action can be 'stop_voice' to signal stopping.
    """
    cmd = command.lower().strip()
    # Normalize punctuation
    for ch in [',', '.', '!', '?', "'", '"']:
        cmd = cmd.replace(ch, '')
    print(f"[DEBUG] process_voice_command received: '{cmd}'")
    sys.stdout.flush()

    # STOP / VOICE MODE
    if any(x in cmd for x in ['stop voice', 'quit voice', 'exit voice']):
        print("[VOICE] → STOPPING VOICE MODE (via processor)")
        return True, 'stop_voice'
    if cmd in ['stop', 'quit', 'exit']:
        print("[VOICE] → STOPPING VOICE MODE (via processor)")
        return True, 'stop_voice'

    # MEDIA: Play / Pause (HIGH PRIORITY)
    if cmd in ['play', 'resume', 'start']:
        print('[VOICE] → Play')
        if execute:
            send_media_play()
        return True, None

    if cmd == 'pause':
        print('[VOICE] → Pause')
        if execute:
            send_media_pause()
        return True, None

    # MOUSE ACTIONS
    if 'double click' in cmd or 'double-click' in cmd:
        print('[VOICE] → Double click')
        if execute:
            pyautogui.doubleClick()
        return True, None
    if 'right click' in cmd or ('click' in cmd and 'right' in cmd):
        print('[VOICE] → Right click')
        if execute:
            pyautogui.click(button='right')
        return True, None
    if cmd == 'click' or 'left click' in cmd:
        print('[VOICE] → Left click')
        if execute:
            pyautogui.click()
        return True, None

    # SCROLL
    if 'scroll up' in cmd or 'scroll higher' in cmd:
        print('[VOICE] → Scroll up')
        if execute:
            pyautogui.scroll(300)
        return True, None
    if 'scroll down' in cmd or 'scroll lower' in cmd:
        print('[VOICE] → Scroll down')
        if execute:
            pyautogui.scroll(-300)
        return True, None

    # MINIMIZE
    if 'minimize' in cmd or 'minimise' in cmd:
        print('[VOICE] → Minimize window')
        if execute:
            # Minimize current window (Win+Down)
            pyautogui.hotkey('win', 'down')
        return True, None

    # CLOSE by name or current
    if cmd.startswith('close '):
        target = cmd.replace('close ', '', 1).strip()
        if target in ['', 'window', 'app', 'application', 'current', 'this', 'that', 'it']:
            print('[VOICE] → Closing current window (processor)')
            if execute:
                close_active_window()
            return True, None
        else:
            print(f"[VOICE] → Closing specific: {target} (processor)")
            if execute:
                closed = close_app_by_name(target)
                if not closed:
                    close_active_window()
            return True, None

    if cmd in ['close', 'close window', 'close app']:
        print('[VOICE] → Closing current window (processor)')
        if execute:
            close_active_window()
        return True, None

    # APPLICATION opens (explicit handlers)
    if 'open spotify' in cmd or cmd == 'spotify':
        print('[VOICE] → Opening Spotify')
        if execute:
            try:
                username = os.getenv('USERNAME')
                os.startfile(rf"C:\Users\{username}\AppData\Roaming\Spotify\Spotify.exe")
            except:
                try:
                    os.startfile(r"C:\Program Files\Spotify\Spotify.exe")
                except:
                    print('[!] Spotify not found')
        return True, None

    if 'open chrome' in cmd or cmd == 'chrome':
        print('[VOICE] → Opening Chrome (processor)')
        if execute:
            try:
                os.startfile(r"C:\\Program Files\\Google\\Chrome\\Application\\chrome.exe")
            except:
                try:
                    os.startfile(r"C:\\Program Files (x86)\\Google\\Chrome\\Application\\chrome.exe")
                except:
                    print("[!] Chrome not found")
        return True, None

    if 'open youtube' in cmd or 'youtube' in cmd:
        print('[VOICE] → Opening YouTube (processor)')
        if execute:
            try:
                webbrowser.open('https://www.youtube.com')
            except Exception as e:
                print(f"[!] Failed to open YouTube: {e}")
        return True, None

    if 'open whatsapp' in cmd or 'whatsapp' in cmd:
        print('[VOICE] → Opening WhatsApp (processor)')
        if execute:
            opened = False
            candidates = [
                os.path.expanduser(r"~\\AppData\\Local\\WhatsApp\\WhatsApp.exe"),
                os.path.join(os.environ.get('ProgramFiles', r"C:\\Program Files"), 'WhatsApp', 'WhatsApp.exe'),
                os.path.join(os.environ.get('ProgramFiles(x86)', r"C:\\Program Files (x86)"), 'WhatsApp', 'WhatsApp.exe')
            ]
            for p in candidates:
                try:
                    if p and os.path.exists(p):
                        os.startfile(p)
                        opened = True
                        break
                except:
                    pass
            if not opened:
                try:
                    webbrowser.open('https://web.whatsapp.com')
                except Exception:
                    print('[!] WhatsApp not found')
        return True, None

    if 'open edge' in cmd or 'microsoft edge' in cmd or cmd == 'edge':
        print('[VOICE] → Opening Microsoft Edge (processor)')
        if execute:
            try:
                os.startfile(r"C:\\Program Files\\Microsoft\\Edge\\Application\\msedge.exe")
            except:
                try:
                    os.startfile(r"C:\\Program Files (x86)\\Microsoft\\Edge\\Application\\msedge.exe")
                except:
                    try:
                        os.startfile('microsoft-edge:')
                    except:
                        print('[!] Microsoft Edge not found')
        return True, None

    if 'open vs code' in cmd or 'visual studio code' in cmd or 'vs code' in cmd:
        print('[VOICE] → Opening VS Code (processor)')
        if execute:
            try:
                os.startfile('code.exe')
            except Exception:
                print('[!] VS Code not found')
        return True, None

    if 'open firefox' in cmd or cmd == 'firefox':
        print('[VOICE] → Opening Firefox')
        if execute:
            try:
                os.startfile(r"C:\Program Files\Mozilla Firefox\firefox.exe")
            except:
                try:
                    os.startfile(r"C:\Program Files (x86)\Mozilla Firefox\firefox.exe")
                except:
                    print('[!] Firefox not found')
        return True, None

    if 'open discord' in cmd or cmd == 'discord':
        print('[VOICE] → Opening Discord')
        if execute:
            try:
                username = os.getenv('USERNAME')
                os.startfile(rf"C:\Users\{username}\AppData\Local\Discord\app-*\Discord.exe")
            except:
                print('[!] Discord not found')
        return True, None

    if 'open teams' in cmd or 'open microsoft teams' in cmd or cmd == 'teams':
        print('[VOICE] → Opening Microsoft Teams')
        if execute:
            try:
                username = os.getenv('USERNAME')
                os.startfile(rf"C:\Users\{username}\AppData\Local\Microsoft\Teams\Teams.exe")
            except:
                print('[!] Teams not found')
        return True, None

    if 'open telegram' in cmd or cmd == 'telegram':
        print('[VOICE] → Opening Telegram')
        if execute:
            try:
                username = os.getenv('USERNAME')
                os.startfile(rf"C:\Users\{username}\AppData\Local\Telegram\Telegram.exe")
            except:
                print('[!] Telegram not found')
        return True, None

    if 'open slack' in cmd or cmd == 'slack':
        print('[VOICE] → Opening Slack')
        if execute:
            try:
                username = os.getenv('USERNAME')
                os.startfile(rf"C:\Users\{username}\AppData\Local\slack\slack.exe")
            except:
                print('[!] Slack not found')
        return True, None

    if 'open settings' in cmd or 'windows settings' in cmd:
        print('[VOICE] → Opening Settings')
        if execute:
            os.startfile('ms-settings:')
        return True, None

    # OFFICE APPLICATIONS
    if 'open notepad' in cmd or cmd == 'notepad':
        print('[VOICE] → Opening Notepad')
        if execute:
            os.startfile('notepad.exe')
        return True, None

    if 'open calculator' in cmd or cmd == 'calculator' or cmd == 'calc':
        print('[VOICE] → Opening Calculator')
        if execute:
            os.startfile('calc.exe')
        return True, None

    if 'open file explorer' in cmd or 'file explorer' in cmd or cmd == 'explorer' or cmd == 'open files':
        print('[VOICE] → Opening File Explorer')
        if execute:
            os.startfile('explorer.exe')
        return True, None

    if 'open word' in cmd or cmd == 'word':
        print('[VOICE] → Opening Microsoft Word')
        if execute:
            os.startfile('winword.exe')
        return True, None

    if 'open excel' in cmd or cmd == 'excel':
        print('[VOICE] → Opening Microsoft Excel')
        if execute:
            os.startfile('excel.exe')
        return True, None

    if 'open paint' in cmd or cmd == 'paint' or cmd == 'mspaint':
        print('[VOICE] → Opening Paint')
        if execute:
            os.startfile('mspaint.exe')
        return True, None

    if 'open vlc' in cmd or cmd == 'vlc':
        print('[VOICE] → Opening VLC Media Player')
        if execute:
            try:
                os.startfile(r"C:\Program Files\VideoLAN\VLC\vlc.exe")
            except:
                try:
                    os.startfile(r"C:\Program Files (x86)\VideoLAN\VLC\vlc.exe")
                except:
                    print('[!] VLC not found')
        return True, None

    if 'open task manager' in cmd or 'task manager' in cmd or cmd == 'taskmgr':
        print('[VOICE] → Opening Task Manager')
        if execute:
            os.startfile('taskmgr.exe')
        return True, None

    if 'open control panel' in cmd or 'control panel' in cmd:
        print('[VOICE] → Opening Control Panel')
        if execute:
            os.startfile('control.exe')
        return True, None

    if 'open zoom' in cmd or cmd == 'zoom':
        print('[VOICE] → Opening Zoom')
        if execute:
            try:
                os.startfile('zoom.exe')
            except:
                print('[!] Zoom not found')
        return True, None

    # POWERPOINT / PRESENTATION
    if 'open powerpoint' in cmd or 'open ppt' in cmd or cmd == 'powerpoint' or cmd == 'ppt':
        print('[VOICE] → Opening PowerPoint')
        if execute:
            try:
                os.startfile('powerpnt.exe')
            except:
                print('[!] PowerPoint not found')
        return True, None

    if 'slideshow' in cmd or 'start slideshow' in cmd or 'start presentation' in cmd:
        print('[VOICE] → Starting Slideshow')
        if execute:
            pyautogui.press('f5')  # F5 starts slideshow in PowerPoint
        return True, None

    # FOLDERS - COMMON LOCATIONS
    if 'open desktop' in cmd:
        print('[VOICE] → Opening Desktop')
        if execute:
            desktop_path = os.path.expanduser('~\\Desktop')
            os.startfile(desktop_path)
        return True, None

    if 'open downloads' in cmd or cmd == 'downloads':
        print('[VOICE] → Opening Downloads')
        if execute:
            downloads_path = os.path.expanduser('~\\Downloads')
            os.startfile(downloads_path)
        return True, None

    if 'open documents' in cmd or cmd == 'documents':
        print('[VOICE] → Opening Documents')
        if execute:
            documents_path = os.path.expanduser('~\\Documents')
            os.startfile(documents_path)
        return True, None

    if 'open videos' in cmd or cmd == 'videos':
        print('[VOICE] → Opening Videos')
        if execute:
            videos_path = os.path.expanduser('~\\Videos')
            os.startfile(videos_path)
        return True, None

    if 'open pictures' in cmd or 'open photos' in cmd or cmd == 'pictures' or cmd == 'photos':
        print('[VOICE] → Opening Pictures')
        if execute:
            pictures_path = os.path.expanduser('~\\Pictures')
            os.startfile(pictures_path)
        return True, None

    if 'open music' in cmd or cmd == 'music':
        print('[VOICE] → Opening Music')
        if execute:
            music_path = os.path.expanduser('~\\Music')
            os.startfile(music_path)
        return True, None

    if 'open recents' in cmd or cmd == 'recents':
        print('[VOICE] → Opening Recent Files')
        if execute:
            recent_path = os.path.expanduser('%APPDATA%\\Microsoft\\Windows\\Recent')
            os.startfile(recent_path)
        return True, None

    if 'open this pc' in cmd or 'open my computer' in cmd:
        print('[VOICE] → Opening This PC / My Computer')
        if execute:
            os.startfile('explorer.exe')
        return True, None

    # DRIVES
    drive_match = re.search(r'open\s+([a-z])\s+drive', cmd)
    if drive_match:
        drive_letter = drive_match.group(1).upper()
        print(f'[VOICE] → Opening {drive_letter}: Drive')
        if execute:
            try:
                os.startfile(f'{drive_letter}:\\')
            except:
                print(f'[!] {drive_letter}: drive not found')
        return True, None

    # OPEN fallback for filenames: 'open <name>' but don't handle app-specific opens here
    if cmd.startswith('open '):
        # if it's a generic open not handled by main branch, attempt filename open
        keywords = ['open chrome', 'open settings', 'open notepad', 'open calculator', 'open files', 'open word', 'open excel', 'open paint', 'open vlc', 'open vs code', 'open task manager', 'open control panel', 'open whatsapp', 'open powerpoint', 'open slideshow', 'open desktop', 'open downloads', 'open videos', 'open pictures', 'open documents', 'open music', 'open recents', 'open this pc', 'open files', 'open zoom', 'open edge', 'open spotify', 'open youtube', 'open firefox', 'open discord', 'open teams', 'open telegram', 'open slack', 'open drive']
        if not any(k in cmd for k in keywords):
            filename = cmd.replace('open ', '', 1).strip()
            if filename:
                print(f"[VOICE] → Searching for file: {filename}")
                if execute:
                    open_file_by_name(filename)
                return True, None

    # MEDIA: Play / Pause
    # Spotify-specific pause/play
    if 'spotify' in cmd:
        if 'pause' in cmd or 'stop' in cmd:
            print('[VOICE] → Spotify Pause')
            if execute:
                # Send media pause to Spotify (Windows)
                try:
                    subprocess.run(['nircmd.exe', 'mediapause', 'Spotify'], check=True)
                except Exception:
                    send_media_play_pause()
            return True, None
        elif 'play' in cmd or 'resume' in cmd:
            print('[VOICE] → Spotify Play')
            if execute:
                try:
                    subprocess.run(['nircmd.exe', 'mediaplay', 'Spotify'], check=True)
                except Exception:
                    send_media_play_pause()
            return True, None
        elif 'volume' in cmd:
            m = re.search(r"\b(set|increase|decrease)?\s*volume(?:\s*(?:to|by))?\s*(\d{1,3})\b", cmd)
            if m:
                action = m.group(1) or 'set'
                val = int(m.group(2))
                # Use nircmd for Spotify volume if available
                print(f"[VOICE] → Spotify volume {action} {val}")
                if execute:
                    try:
                        if action == 'set':
                            subprocess.run(['nircmd.exe', 'setappvolume', 'Spotify.exe', str(val/100)], check=True)
                        elif action == 'increase':
                            subprocess.run(['nircmd.exe', 'changeappvolume', 'Spotify.exe', str(val/100)], check=True)
                        elif action == 'decrease':
                            subprocess.run(['nircmd.exe', 'changeappvolume', 'Spotify.exe', str(-val/100)], check=True)
                    except Exception:
                        print('[VOICE] Could not set Spotify volume')
                return True, None

    # VOLUME commands: set/increase/decrease
    m = re.search(r"\b(set|increase|decrease)?\s*volume(?:\s*(?:to|by))?\s*(\d{1,3})\b", cmd)
    if m:
        action = m.group(1) or 'set'
        val = int(m.group(2))
        # If user said 'by', treat increase/decrease as relative; otherwise treat as 'set to' for simplicity
        if action in ('increase','decrease') and re.search(r"\bby\b", cmd):
            cur = get_current_volume_percent()
            if cur is None:
                print('[VOICE] ⚠️ Precise volume control not available. Install pycaw/comtypes for exact control.')
                return True, None
            if action == 'increase':
                target = min(100, cur + val)
            else:
                target = max(0, cur - val)
        else:
            # Default: set volume to the provided percentage
            target = max(0, min(100, val))
        print(f"[VOICE] → Setting volume to {target}%")
        if execute:
            set_volume_percent(target)
        return True, None

    # Volume up/down (no number) -> +/-10%
    if re.search(r"\b(volume up|increase volume|volume increase)\b", cmd):
        print('[VOICE] → Volume up (10%)')
        if execute:
            change_volume_by(10)
        return True, None

    if re.search(r"\b(volume down|decrease volume|volume decrease)\b", cmd):
        print('[VOICE] → Volume down (10%)')
        if execute:
            change_volume_by(-10)
        return True, None

    # Not handled
    print('[DEBUG] Command not handled by process_voice_command')
    return False, None


def voice_control_worker():
    """Voice recognition worker thread"""
    global listening, voice_mode

    if not VOICE_MODE_AVAILABLE:
        print("[!] Voice mode not available - speech_recognition not installed")
        return

    try:
        recognizer = sr.Recognizer()
        recognizer.energy_threshold = 2000
        recognizer.dynamic_energy_threshold = True
        recognizer.pause_threshold = 0.8
        print("[VOICE] Initializing microphone...")
        sys.stdout.flush()

        # Test microphone availability
        try:
            with sr.Microphone() as source:
                print("[VOICE] ✓ Microphone found and initialized!")
                sys.stdout.flush()
        except Exception as e:
            print(f"[!] Microphone error: {e}")
            print("[!] Make sure your microphone is connected and working")
            sys.stdout.flush()
            return

        print("[*] Voice Control Active! Listening for commands...")
        print("[*] Media: 'play', 'pause', 'resume'")
        print("[*] Mouse: 'click', 'right click', 'double click'")
        print("[*] Scroll: 'scroll up', 'scroll down'")
        print("[*] Volume: 'volume 50', 'volume up', 'volume down'")
        print("[*] Apps: 'open chrome', 'open powerpoint', 'open spotify'")
        print("[*] Files: 'open filename' (searches Downloads, Documents, Desktop)")
        print("[*] Folders: 'open desktop', 'open downloads', 'open documents', 'open videos'")
        print("[*] Stop: 'stop', 'quit voice', 'exit voice'")
        sys.stdout.flush()

        with sr.Microphone() as source:
            print("[VOICE] Adjusting for ambient noise...")
            sys.stdout.flush()
            recognizer.adjust_for_ambient_noise(source, duration=1.0)
            print("[VOICE] Ready to listen!")
            sys.stdout.flush()

            while listening:
                try:
                    # Listen with timeout
                    print("[VOICE] Listening...", end='\r')
                    sys.stdout.flush()
                    audio = recognizer.listen(source, timeout=5.0, phrase_time_limit=5.0)

                    # Try to recognize speech using Google Speech Recognition
                    try:
                        print("[VOICE] Processing speech...         ")
                        sys.stdout.flush()
                        command = recognizer.recognize_google(audio).lower()
                        print(f"[VOICE] ✓ Recognized: '{command}'")
                        sys.stdout.flush()

                        # First pass: generic command processor (mouse, close, open-file fallback)
                        try:
                            handled, action = process_voice_command(command, execute=True)
                        except Exception as e:
                            print(f"[DEBUG] process_voice_command error: {e}")
                            sys.stdout.flush()
                            handled, action = False, None

                        if handled:
                            if action == 'stop_voice':
                                listening = False
                                voice_mode = False
                                break

                            # Already executed by processor
                            continue

                    except sr.UnknownValueError:
                        pass  # Silence repeated 'could not understand' noise
                    except sr.RequestError as e:
                        print(f"[!] API error: {e}")
                        print("[!] Make sure you have internet connection for Google Speech Recognition")
                        sys.stdout.flush()

                except sr.WaitTimeoutError:
                    pass  # Timeout, continue listening

    except Exception as e:
        print(f"[!] Voice control error: {e}")
        import traceback
        traceback.print_exc()
        sys.stdout.flush()

    finally:
        listening = False
        voice_mode = False
        print("[*] Voice control stopped")
        sys.stdout.flush()

def start_voice_mode():
    """Start voice control in a separate thread"""
    global voice_mode, voice_thread, listening

    if not VOICE_MODE_AVAILABLE:
        print("[!] Voice mode not available - speech_recognition not installed")
        print("[*] Install: pip install SpeechRecognition pydub pyaudio")
        return

    if voice_mode:
        print("[!] Voice mode already running")
        return

    print("[*] Starting voice mode...")
    sys.stdout.flush()
    voice_mode = True
    listening = True
    voice_thread = threading.Thread(target=voice_control_worker, daemon=True)
    voice_thread.start()
    print("[✓] Voice mode started!")
    sys.stdout.flush()

def stop_voice_mode():
    """Stop voice control"""
    global voice_mode, listening
    voice_mode = False
    listening = False
    if voice_thread:
        voice_thread.join(timeout=2)
    print("[*] Voice mode stopped")

def calibrate_mode(cap):
    """Calibration mode for adjusting sensitivity"""
    print("\n[*] CALIBRATION MODE STARTED")
    print("Move your hand around to find optimal settings")
    print("Press ESC to exit calibration\n")
    
    while True:
        success, img = cap.read()
        if not success:
            break
        
        img = cv2.flip(img, 1)
        h, w, c = img.shape
        
        # Draw grid
        for i in range(0, w, 50):
            cv2.line(img, (i, 0), (i, h), (50, 50, 50), 1)
        for i in range(0, h, 50):
            cv2.line(img, (0, i), (w, i), (50, 50, 50), 1)
        
        # Draw center crosshair
        cv2.line(img, (w//2-20, h//2), (w//2+20, h//2), (0, 255, 255), 2)
        cv2.line(img, (w//2, h//2-20), (w//2, h//2+20), (0, 255, 255), 2)
        
        cv2.putText(img, "CALIBRATION MODE - Press ESC to exit", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        cv2.imshow("Calibration", img)
        
        if cv2.waitKey(1) & 0xFF == 27:
            cv2.destroyWindow("Calibration")
            break
    
    print("✓ Calibration mode exited\n")

def air_drawing_mode(cap, landmarker):
    """Air drawing mode - draw while pinching thumb+index"""
    global drawing_mode, canvas
    
    print("\n[*] AIR DRAWING MODE STARTED")
    print("Pinch thumb+index to draw | C:Clear | ESC:Exit\n")
    
    canvas = np.ones((CAM_HEIGHT, CAM_WIDTH, 3), dtype=np.uint8) * 255
    drawing_mode = True
    prev_index_pos = None
    position_history = deque(maxlen=5)
    
    while drawing_mode:
        success, img = cap.read()
        if not success:
            break
        
        img = cv2.flip(img, 1)
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=img_rgb)
        
        results = landmarker.detect(mp_image)
        
        if results.hand_landmarks and len(results.hand_landmarks) >= 1:
            hand_landmarks = results.hand_landmarks[0]
            thumb_tip = hand_landmarks[4]
            index_tip = hand_landmarks[8]
            
            # Get positions
            thumb = (int(thumb_tip.x * CAM_WIDTH), int(thumb_tip.y * CAM_HEIGHT))
            index = (int(index_tip.x * CAM_WIDTH), int(index_tip.y * CAM_HEIGHT))
            
            # Detect pinch
            pinch_dist = np.linalg.norm(np.array(thumb) - np.array(index))
            is_pinching = pinch_dist < 35
            
            # Add to history for smoothing
            position_history.append(index)
            
            # Get smoothed position
            if position_history:
                smooth_x = int(np.mean([p[0] for p in position_history]))
                smooth_y = int(np.mean([p[1] for p in position_history]))
                
                # Draw only when pinching
                if is_pinching:
                    if prev_index_pos:
                        cv2.line(canvas, prev_index_pos, (smooth_x, smooth_y), (0, 0, 255), 4)
                    prev_index_pos = (smooth_x, smooth_y)
                    cursor_color = (0, 255, 0)  # Green when drawing
                else:
                    prev_index_pos = None
                    cursor_color = (0, 0, 255)  # Red when not pinching
                
                cv2.circle(canvas, (smooth_x, smooth_y), 10, cursor_color, -1)
        
        # Display with status
        display_canvas = canvas.copy()
        cv2.putText(display_canvas, "AIR DRAWING | Pinch to draw | C:Clear | ESC:Exit", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)
        cv2.imshow("Air Drawing Canvas", display_canvas)
        
        key = cv2.waitKey(1) & 0xFF
        if key == 27:  # ESC
            drawing_mode = False
            break
        elif key == ord('c') or key == ord('C'):
            canvas = np.ones((CAM_HEIGHT, CAM_WIDTH, 3), dtype=np.uint8) * 255
            prev_index_pos = None
            position_history.clear()
    
    cv2.destroyWindow("Air Drawing Canvas")
    print("✓ Air Drawing mode exited\n")
    drawing_mode = False

def save_config(config):
    try:
        with open(CONFIG_FILE, 'w') as f:
            json.dump(config, f, indent=2)
        print("[✓] Config saved.")
    except Exception as e:
        print(f"[!] Failed to save config: {e}")







try:
    pyautogui.FAILSAFE = True
    start_time = time.time()
    frame_count = 0

    print("[OK] Hand Gesture Control Started!")
    if VOICE_MODE_AVAILABLE:
        print("[✓] Voice mode: AVAILABLE")
    else:
        print("[!] Voice mode: DISABLED (speech_recognition not installed)")
    sys.stdout.flush()
    print("Press 'H' for help menu\n")
    sys.stdout.flush()
    
    while True:
        frame_start = time.time()
        success, img = cap.read()
        if not success:
            break
        
        frame_count += 1
        img = cv2.flip(img, 1)
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=img_rgb)
        
        # Detect hand landmarks
        results = landmarker.detect(mp_image)
        h, w, c = img.shape
        
        # Calculate FPS
        frame_end = time.time()
        fps_counter.append(1.0 / (frame_end - frame_start + 0.001))
        avg_fps = np.mean(fps_counter) if fps_counter else 0
        
        current_time = time.time()

        # Count hands (debug print removed to reduce console spam)
        num_hands = len(results.hand_landmarks) if results.hand_landmarks else 0

        # Check for both hands detected to activate voice mode (only if available)
        if VOICE_MODE_AVAILABLE:
            both_hands_detected = are_both_hands_detected(results)
            if both_hands_detected and (current_time - last_voice_toggle_time) > 3.0:
                if not voice_mode:
                    print("[*] Both hands detected - Activating voice mode...")
                    start_voice_mode()
                    last_voice_toggle_time = current_time
        
        if results.hand_landmarks:
            hand_landmarks = results.hand_landmarks[0]
            # Get hand confidence from handedness (it's already a list of category objects)
            hand_confidence = results.handedness[0][0].score if results.handedness else 0.7
            
            # Get landmark positions
            lm_list = []
            for lm in hand_landmarks:
                x, y = int(lm.x * CAM_WIDTH), int(lm.y * CAM_HEIGHT)
                lm_list.append((x, y))
            
            # Draw hand
            draw_hand_landmarks(img, hand_landmarks)
            
            # Get key landmarks
            index_tip = lm_list[8]
            hand_position_history.append(index_tip)
             
        # Map to screen
            screen_x = np.interp(index_tip[0], (0, CAM_WIDTH), (0, SCREEN_WIDTH))
            screen_y = np.interp(index_tip[1], (0, CAM_HEIGHT), (0, SCREEN_HEIGHT))
            
            # Apply smoothing (use multiplication instead of complex calculation)
            curr_x = SMOOTHING * prev_x + (1.0 - SMOOTHING) * screen_x
            curr_y = SMOOTHING * prev_y + (1.0 - SMOOTHING) * screen_y
            
            # Clamp to screen boundaries
            if curr_x < 0: curr_x = 0
            elif curr_x > SCREEN_WIDTH - 1: curr_x = SCREEN_WIDTH - 1
            if curr_y < 0: curr_y = 0
            elif curr_y > SCREEN_HEIGHT - 1: curr_y = SCREEN_HEIGHT - 1
            
            # Update mouse position with integer conversion
            pyautogui.moveTo(int(curr_x), int(curr_y))
            prev_x, prev_y = curr_x, curr_y
            
            # Detect gesture
            gesture = detect_gesture(hand_landmarks, lm_list, current_time)

            # --- Gesture actions (optimized) ---
            if gesture['type'] == 'open_ppt' and (current_time - last_powerpoint_open_time) > 0.8:
                try:
                    os.startfile("powerpnt.exe")
                    powerpoint_mode = True
                except Exception:
                    pass
                last_powerpoint_open_time = current_time
            
            # Exit PowerPoint mode on open hand gesture
            if gesture['type'] == 'open_hand' and powerpoint_mode and (current_time - last_powerpoint_open_time) > 0.8:
                powerpoint_mode = False
                last_powerpoint_open_time = current_time
            
            # PowerPoint slide navigation
            if powerpoint_mode and len(lm_list) >= 9:
                index_x = lm_list[8][0]
                if last_thumb_x is not None:
                    dx = index_x - last_thumb_x
                    if dx > 60 and (current_time - last_slide_change_time) > 0.7:
                        pyautogui.press('right')
                        last_slide_change_time = current_time
                    elif dx < -60 and (current_time - last_slide_change_time) > 0.7:
                        pyautogui.press('left')
                        last_slide_change_time = current_time
                last_thumb_x = index_x
            
            # Gesture-based actions
            if gesture['type'] == 'left_click' and current_time - last_click_time > DOUBLE_CLICK_DELAY:
                pyautogui.click()
                last_click_time = current_time
            elif gesture['type'] == 'right_click' and current_time - last_right_click_time > DOUBLE_CLICK_DELAY:
                pyautogui.click(button='right')
                last_right_click_time = current_time
            elif gesture['type'] == 'fist' and current_time - last_minimize_time > 0.6:
                if not powerpoint_mode:
                    try:
                        active_window = gw.getActiveWindow()
                        if active_window:
                            active_window.minimize()
                    except:
                        pass
                    last_minimize_time = current_time

            # AR Drawing Board activation: Hold 'pinch' gesture for 2 seconds
            if gesture['type'] == 'pinch':
                cv2.putText(img, "Pinch gesture detected!", (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
                print('[Gesture] Pinch gesture detected!')
            if gesture['type'] == 'open_ppt' and (current_time - last_powerpoint_open_time) > 0.8:
                try:
                    os.startfile("powerpnt.exe")
                    powerpoint_mode = True
                    print('[Gesture] Opening PowerPoint and entering PowerPoint mode')
                except Exception as e:
                    print(f"[!] Failed to open PowerPoint: {e}")
                last_powerpoint_open_time = current_time

            # AR Drawing Board activation: Hold 'pinch' gesture for 2 seconds
            if not hasattr(globals(), '_pinch_timer'):
                globals()['_pinch_timer'] = None
            pinch_timer = globals()['_pinch_timer']
            if gesture['type'] == 'pinch':
                if pinch_timer is None:
                    pinch_timer = current_time
                elif current_time - pinch_timer > 2.0:
                    air_drawing_mode(cap, landmarker)
                    pinch_timer = None
                globals()['_pinch_timer'] = pinch_timer
            else:
                globals()['_pinch_timer'] = None

            # Draw cursor indicator
            cv2.circle(img, index_tip, CURSOR_RADIUS, (0, 255, 255), 2)
            cv2.circle(img, index_tip, 5, (0, 255, 255), -1)
            
            # Draw UI
            draw_ui(img, gesture, hand_confidence, avg_fps, num_hands, voice_mode)

        else:
            cv2.putText(img, "No hand detected - Move your hand into frame", (w//2 - 250, h//2), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            draw_ui(img, {'type': 'none', 'distances': {'thumb_index': 0}}, 0, avg_fps, num_hands, voice_mode)
        
        # Draw settings menu if active
        if show_settings_menu:
            draw_settings_menu(img)
        
        cv2.imshow("Hand Gesture Mouse Control", img)
        
        # Keyboard controls
        key = cv2.waitKey(1) & 0xFF
        if key == 27:  # ESC
            break
        elif key == ord('h') or key == ord('H'):
            print_help()
        elif key == ord('s') or key == ord('S'):
            show_settings_menu = not show_settings_menu
        elif key == ord('v') or key == ord('V'):
            if not voice_mode:
                start_voice_mode()
            else:
                stop_voice_mode()
        elif key == ord('c') or key == ord('C'):
            if not show_settings_menu:
                calibrate_mode(cap)
            else:
                show_settings_menu = False
        elif key in (ord('+'), ord('=')):
            SMOOTHING = min(1.0, SMOOTHING + 0.05)
            config['SMOOTHING'] = SMOOTHING
        elif key in (ord('-'), ord('_')):
            SMOOTHING = max(0.1, SMOOTHING - 0.05)
            config['SMOOTHING'] = SMOOTHING
        elif key == ord('i') or key == ord('I'):
            CLICK_THRESHOLD = min(100, CLICK_THRESHOLD + 2)
            config['CLICK_THRESHOLD'] = CLICK_THRESHOLD
        elif key == ord('o') or key == ord('O'):
            CLICK_THRESHOLD = max(10, CLICK_THRESHOLD - 2)
            config['CLICK_THRESHOLD'] = CLICK_THRESHOLD

except KeyboardInterrupt:
    print("\n\n[!] Program interrupted by user")

finally:
    # Stop voice mode if active
    if voice_mode:
        stop_voice_mode()
    
    cap.release()
    cv2.destroyAllWindows()
    elapsed = time.time() - start_time
