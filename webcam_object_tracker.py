import subprocess
import sys
import os

_FaceEyeTracker_PATH = os.getcwd()


command = ["python", _FaceEyeTracker_PATH + "/object_tracker.py",
           "--video", str(0),
           "--output", str("./outputs/FaceEyeTracker-Webcam.mp4"),
           "--output_format", str("mp4v"),
           "--count", str("True")]

subprocess.run(command, cwd=_FaceEyeTracker_PATH, stdout=sys.stdout, stderr=sys.stderr)
