import cv2
import pandas as pd
import numpy as np
import math

# Dictionary of available ArUco marker types
ARUCO_DICTS = {
    "DICT_4X4_50": cv2.aruco.DICT_4X4_50,
    "DICT_4X4_100": cv2.aruco.DICT_4X4_100,
    "DICT_4X4_250": cv2.aruco.DICT_4X4_250,
    "DICT_4X4_1000": cv2.aruco.DICT_4X4_1000,
    "DICT_5X5_50": cv2.aruco.DICT_5X5_50,
    "DICT_5X5_100": cv2.aruco.DICT_5X5_100,
    "DICT_5X5_250": cv2.aruco.DICT_5X5_250,
    "DICT_5X5_1000": cv2.aruco.DICT_5X5_1000,
    "DICT_6X6_50": cv2.aruco.DICT_6X6_50,
    "DICT_6X6_100": cv2.aruco.DICT_6X6_100,
    "DICT_6X6_250": cv2.aruco.DICT_6X6_250,
    "DICT_6X6_1000": cv2.aruco.DICT_6X6_1000,
    "DICT_7X7_50": cv2.aruco.DICT_7X7_50,
    "DICT_7X7_100": cv2.aruco.DICT_7X7_100,
    "DICT_7X7_250": cv2.aruco.DICT_7X7_250,
    "DICT_7X7_1000": cv2.aruco.DICT_7X7_1000,
    "DICT_ARUCO_ORIGINAL": cv2.aruco.DICT_ARUCO_ORIGINAL,
    "DICT_APRILTAG_16h5": cv2.aruco.DICT_APRILTAG_16h5,
    "DICT_APRILTAG_25h9": cv2.aruco.DICT_APRILTAG_25h9,
    "DICT_APRILTAG_36h10": cv2.aruco.DICT_APRILTAG_36h10,
    "DICT_APRILTAG_36h11": cv2.aruco.DICT_APRILTAG_36h11
}

# Select the dictionary to use
selected_dict = ARUCO_DICTS["DICT_4X4_100"]

# Initialize the ArUco dictionary
aruco_dict = cv2.aruco.getPredefinedDictionary(selected_dict)
aruco_params = cv2.aruco.DetectorParameters()

# Setup video input and output
input_video_path = '/home/ori/Desktop/robotos ex2/challengeB(1).mp4'  # Replace with a path to your video
cap = cv2.VideoCapture(input_video_path)

# Get the original video frame rate
fps = cap.get(cv2.CAP_PROP_FPS)

# Define the codec and create VideoWriter object to save output as MP4
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
output_video = cv2.VideoWriter('output_video.mp4', fourcc, fps, (int(cap.get(3)), int(cap.get(4))))

# Prepare to log detections to a CSV file
csv_data = []
csv_columns = ['Frame ID', 'QR ID', 'QR 2D', 'QR 3D: dist', 'QR 3D: yaw', 'QR 3D: pitch', 'QR 3D: roll']
frame_id = 0

camera_matrix = np.array([[921.170702, 0.000000, 459.904354], [0.000000, 919.018377, 351.238301], [0.000000, 0.000000, 1.000000]], dtype=np.float32)
dist_coeffs = np.array([-0.033458, 0.105152, 0.001256, -0.006647, 0.000000], dtype=np.float32)

def rotationMatrixToEulerAngles(R):
    sy = math.sqrt(R[0, 0] ** 2 + R[1, 0] ** 2)
    singular = sy < 1e-6

    if not singular:
        x = math.atan2(R[2, 1], R[2, 2])
        y = math.atan2(-R[2, 0], sy)
        z = math.atan2(R[1, 0], R[0, 0])
    else:
        x = math.atan2(-R[1, 2], R[1, 1])
        y = math.atan2(-R[2, 0], sy)
        z = 0

    return np.degrees(x), np.degrees(y), np.degrees(z)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    corners, ids, rejectedImgPoints = cv2.aruco.detectMarkers(gray, aruco_dict, parameters=aruco_params)

    if ids is not None:
        rvecs, tvecs, _ = cv2.aruco.estimatePoseSingleMarkers(corners, 0.05, cameraMatrix=camera_matrix, distCoeffs=dist_coeffs)

        for i, corner in enumerate(corners):
            qr_id = ids[i][0]
            qr_2d = corner.reshape(4, 2)

            # Extract the rotation and translation vectors for this marker
            rvec = rvecs[i][0]
            tvec = tvecs[i][0]

            # Calculate distance, yaw, pitch, and roll from rvec and tvec
            dist = np.linalg.norm(tvec)
            rmat, _ = cv2.Rodrigues(rvec)
            yaw, pitch, roll = rotationMatrixToEulerAngles(rmat)

            csv_data.append([frame_id, qr_id, qr_2d.tolist(), dist, yaw, pitch, roll])

        cv2.aruco.drawDetectedMarkers(frame, corners, ids)

    output_video.write(frame)
    frame_id += 1

cap.release()
output_video.release()

# Save detection data to a CSV file
df = pd.DataFrame(csv_data, columns=csv_columns)
df.to_csv('detected_markers.csv', index=False)
print("Detection complete. CSV and video file have been saved.")
