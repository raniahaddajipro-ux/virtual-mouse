✋ Virtual Mouse - Gesture Control System

A real-time computer vision application that controls system cursor and volume using hand gestures detected via a standard webcam.

Demo:
https://drive.google.com/drive/folders/17PgJw1x7pIKSwcyMxnbq-tnpGRUymwdn?usp=sharing

🚀 Features :
Cursor Movement: Move your index finger to control the mouse position with smoothing.

Gesture Clicking: Peace sign (✌️) triggers a single mouse click.

Volume Control: Pinch (👌) for Volume Down, Wide grasp (🖐️) for Volume Up.

Optimized Performance: Uses MediaPipe Lite model for low-latency tracking.

🛠️ Tech Stack :
Language: Python
Libraries: OpenCV, MediaPipe, PyAutoGUI, NumPy, Pycaw

⚙️ Installation :
1.Clone the repo
git clone https://github.com/YOUR_USERNAME/Virtual-Mouse-Python.gitcd Virtual-Mouse-Python
2.Create a Virtual Environment (Recommended):
python -m venv venv
# Windows
venv\Scripts\activate
# Mac/Linux
source venv/bin/activate
3.Install Dependencies :
pip install opencv-python mediapipe pyautogui numpy pycaw comtypes
4.Run the App :
python virtual_mouse.py

🎮 Controls / Gestures :
| Gesture | Action |
| :---: | :---: |
| ☝️ **Index Up** | Move Cursor |
| ✌️ **Index + Middle** | Left Click |
| 👌 **Thumb + Index (Close)** | Volume Down |
| 🖐️ **Thumb + Index (Wide)** | Volume Up |
| **Press 'Q'** | Quit Application |

📄 License :
This project is open source and available under the MIT License.



