# Autonomous Robotics EX1
## Table of Contents
- [Part 1](#part-1)
- [Part 2](#part-2)

## Part 1

This part demonstrates the detection and pose estimation of Aruco QR markers in a video file. The detected markers are logged into a CSV file along with their 2D and 3D positions, and the processed video with detected markers is saved as an MP4 file.

### Prerequisites

Before running this code, ensure you have the following libraries installed:

- `opencv-contrib-python==4.7.0.68`
- `opencv-python==4.7.0.68`
- `numpy`
- `pandas`

You can install these libraries using pip:

```bash
pip install opencv-contrib-python==4.7.0.68 opencv-python==4.7.0.68 numpy pandas
```
#### How to Run the Code
Ensure you have the required libraries installed.

Update the input_video_path variable with the path to your input video file.

## Part 2

In this part, we demonstrate a method to capture and align a camera's position using ArUco markers. The goal is to capture an initial frame, move the camera, and then guide the user to return it to its initial position using directional commands. The final frame is captured and saved for comparison once the camera is back in the initial position.

## Features

- Detects ArUco markers in real-time using a webcam.
- Captures and saves an initial frame when prompted.
- Guides the user to return the camera to the initial position with directional commands.
- Captures and saves the final frame once the camera is aligned with the initial position.
- Ensures the final frame is mirrored back to match the initial frame.

## Instructions

Press 'c' to capture the initial frame with the ArUco marker.
Move the camera to a different position.
Press 'm' to start the alignment process. Follow the printed movement commands to align the camera to the initial position.
Press 'q' to quit the application.

## Required Dependencies

pip install opencv-python-headless numpy
