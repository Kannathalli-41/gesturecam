# Hand Gesture Mouse Control with Voice Commands

A real-time hand gesture recognition system that allows you to control your computer using hand gestures and voice commands. Uses MediaPipe for hand detection and OpenCV for visualization.

## Features

✨ **Hand Gesture Control**
- Move mouse with index finger
- Left/Right click with hand gestures
- Scroll with gestures
- Open PowerPoint with specific gesture
- Air drawing mode

🎤 **Voice Commands**
- Play/Pause media
- Click, double-click, right-click
- Scroll, volume control
- Open applications (Chrome, Spotify, Teams, etc.)
- Open folders and files
- Close windows and applications

⚙️ **Customization**
- Adjustable smoothing for mouse movement
- Configurable click threshold
- Real-time FPS display
- Settings menu for tuning

## Installation

### Prerequisites
- Python 3.8+
- Webcam
- Windows 10/11

### Step 1: Install Dependencies

```bash
pip install opencv-python mediapipe pyautogui numpy pygetwindow
```

### Step 2: Install Voice Recognition (Optional but Recommended)

```bash
pip install SpeechRecognition pydub pyaudio
```

**On Windows, if pyaudio installation fails:**
```bash
pip install pipwin
pipwin install pyaudio
```

### Step 3: Download MediaPipe Hand Landmarker Model

The model will auto-download on first run. If it fails, manually download from:
https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task

Place it in the same directory as `gesture.py`

## Usage

### Basic Usage

```bash
python gesture.py
```

### Hand Gestures

| Gesture | Action |
|---------|--------|
| **Peace Sign (2 fingers)** | Left Click |
| **Pinky Pointing (1 finger)** | Right Click |
| **Three Fingers** | Open PowerPoint |
| **Open Hand (5 fingers)** | Stop/Pause |
| **Fist (0 fingers)** | Minimize Window |
| **Pinch (thumb+index)** | Air Drawing Mode (hold 2s) |

### Voice Commands

#### Media Control
```
"play"           → Start playback
"pause"          → Pause playback
"resume"         → Resume playback
```

#### Mouse Control
```
"click"          → Left click
"left click"     → Left click
"right click"    → Right click
"double click"   → Double click
"scroll up"      → Scroll up
"scroll down"    → Scroll down
```

#### Volume Control
```
"volume 50"      → Set volume to 50%
"volume up"      → Increase volume 10%
"volume down"    → Decrease volume 10%
"increase volume 20" → Increase by 20%
"decrease volume 15" → Decrease by 15%
```

#### Application Control
```
"open chrome"    → Open Google Chrome
"open powerpoint" → Open Microsoft PowerPoint
"open spotify"   → Open Spotify
"open edge"      → Open Microsoft Edge
"open discord"   → Open Discord
"open teams"     → Open Microsoft Teams
"open zoom"      → Open Zoom
"open notepad"   → Open Notepad
"open calculator" → Open Calculator
"open settings"  → Open Windows Settings
```

#### File & Folder Operations
```
"open desktop"   → Open Desktop folder
"open downloads" → Open Downloads folder
"open documents" → Open Documents folder
"open videos"    → Open Videos folder
"open pictures"  → Open Pictures folder
"open files"     → Open File Explorer
"open filename"  → Search and open a file
```

#### Window Control
```
"close"          → Close current window
"close chrome"   → Close Chrome
"minimize"       → Minimize window
"close app"      → Close current application
```

#### Voice Mode
```
"stop"           → Stop voice mode
"quit voice"     → Quit voice mode
"exit voice"     → Exit voice mode
```

### Keyboard Shortcuts

| Key | Action |
|-----|--------|
| **H** | Show help menu |
| **S** | Toggle settings menu |
| **V** | Toggle voice mode on/off |
| **C** | Calibration mode |
| **+** | Increase smoothing |
| **-** | Decrease smoothing |
| **I** | Increase click threshold |
| **O** | Decrease click threshold |
| **ESC** | Exit program |

## How to Activate Voice Mode

1. **Show 2 Hands** - Hold both hands visible in the camera frame
2. **Screen Indicator** - "VOICE: ON" will appear in green on screen
3. **Drop Hands** - You can now lower your hands and speak
4. **Say Commands** - Voice recognition will continue listening
5. **Stop** - Say "stop" or "quit voice", or press V key to stop

## Configuration

A `gesture_config.json` file will be created with these settings:

```json
{
  "SMOOTHING": 0.92,
  "CLICK_THRESHOLD": 35,
  "STABILITY_FRAMES": 1,
  "SCROLL_THRESHOLD": 50,
  "DOUBLE_CLICK_DELAY": 0.15,
  "MIN_HAND_CONFIDENCE": 0.4,
  "CAM_WIDTH": 640,
  "CAM_HEIGHT": 480
}
```

Edit these values to customize:
- **SMOOTHING**: Higher = smoother mouse (0.0-1.0)
- **CLICK_THRESHOLD**: Distance for pinch detection (lower = more sensitive)
- **MIN_HAND_CONFIDENCE**: Hand detection confidence (0.0-1.0, lower = more sensitive)

## Troubleshooting

### Voice Commands Not Working

1. **Check if speech_recognition is installed:**
   ```bash
   pip install SpeechRecognition pydub pyaudio
   ```

2. **Check microphone:**
   - Ensure microphone is connected and working
   - Test microphone in Windows Settings → Sound
   - Check microphone permissions in Windows

3. **Check internet connection:**
   - Google Speech Recognition requires internet
   - Check your network connection

4. **Check console for errors:**
   - Look for `[VOICE]` messages in console
   - Check for microphone initialization errors

### Hands Not Detected

1. **Lighting:**
   - Ensure good lighting (natural light is best)
   - Avoid backlighting

2. **Hand Position:**
   - Hold hand 30-40cm from camera
   - Keep hand fully visible in frame
   - Move hand slowly and smoothly

3. **Adjust Confidence:**
   - Edit `gesture_config.json`
   - Lower `MIN_HAND_CONFIDENCE` to 0.3
   - Restart program

4. **Check Camera:**
   - Test camera with other applications
   - Ensure camera has permission in Windows Settings

### Camera Not Opening

1. Ensure only one application is using the camera
2. Restart the program
3. Check camera permissions in Windows Settings
4. Try a different USB port if using external camera

### Low FPS / Lag

1. Close other applications
2. Reduce camera resolution in `gesture_config.json`
3. Ensure good lighting (MediaPipe runs faster with clear hands)
4. Close browser tabs and background processes

## File Structure

```
hand_landmarker.task/
├── gesture.py                 # Main program
├── hand_landmarker.task      # MediaPipe model (auto-downloaded)
├── gesture_config.json        # Configuration file (auto-created)
└── README.md                  # This file
```

## Performance Tips

- ✅ Keep hand in good lighting
- ✅ Position hand 30-40cm from camera
- ✅ Move hand smoothly for better tracking
- ✅ Close unnecessary applications
- ✅ Speak clearly for voice commands
- ✅ Keep microphone clean and unobstructed

## Known Limitations

- Requires Windows 10/11
- Works best with clear hand visibility
- Voice recognition requires internet connection (Google Speech API)
- Microphone audio quality affects voice recognition accuracy

## System Requirements

- **OS**: Windows 10/11
- **RAM**: 4GB minimum (8GB recommended)
- **CPU**: Intel i5 or equivalent
- **Camera**: Any USB webcam
- **Microphone**: Any microphone (optional, for voice mode)

## Credits

- **MediaPipe**: Hand detection model
- **OpenCV**: Computer vision processing
- **SpeechRecognition**: Voice command processing

## License

This project is open source and available for personal and educational use.

## Support

For issues or suggestions:
1. Check the troubleshooting section above
2. Verify all dependencies are installed
3. Check console output for error messages
4. Ensure camera and microphone are working

---

**Happy Gesturing! 🖐️**
