import cv2
import numpy as np
import math
import time

# Dictionary of available ArUco marker types
ARUCO_DICTS = {
    "DICT_4X4_50": cv2.aruco.DICT_4X4_50,
    "DICT_4X4_100": cv2.aruco.DICT_4X4_100,
    "DICT_4X4_250": cv2.aruco.DICT_4X4_250,
    "DICT_4X4_1000": cv2.aruco.DICT_4X4_1000,
}

# Select the dictionary to use
selected_dict = ARUCO_DICTS["DICT_4X4_100"]

# Initialize the ArUco dictionary and parameters
aruco_dict = cv2.aruco.getPredefinedDictionary(selected_dict)
aruco_params = cv2.aruco.DetectorParameters()

# Setup webcam input
cap = cv2.VideoCapture(0)  # 0 is the ID for the default camera

camera_matrix = np.array([[1000, 0, 640], [0, 1000, 360], [0, 0, 1]], dtype=np.float32)
dist_coeffs = np.zeros((5, 1), dtype=np.float32)


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


initial_data = None
alignment_in_progress = False
last_command = None
last_command_time = time.time()


def get_movement_command(initial, current):
    tvec_diff = current["tvec"] - initial["tvec"]
    yaw_diff = current["yaw"] - initial["yaw"]
    pitch_diff = current["pitch"] - initial["pitch"]
    roll_diff = current["roll"] - initial["roll"]

    if np.abs(tvec_diff[2]) > 0.1:
        return "Move forward" if tvec_diff[2] > 0 else "Move backward"

    if np.abs(tvec_diff[0]) > 0.02:
        return (
            "Move left" if tvec_diff[0] > 0 else "Move right"
        )  # Corrected for mirroring

    if np.abs(tvec_diff[1]) > 0.02:
        return "Move down" if tvec_diff[1] > 0 else "Move up"

    if np.abs(yaw_diff) > 5:
        return "Turn right" if yaw_diff > 0 else "Turn left"

    if np.abs(pitch_diff) > 5:
        return "Tilt down" if pitch_diff > 0 else "Tilt up"

    if np.abs(roll_diff) > 5:
        return "Roll clockwise" if roll_diff > 0 else "Roll counterclockwise"

    return "Successfully returned to initial position!"


def poses_aligned(initial, current):
    tvec_diff = np.linalg.norm(current["tvec"] - initial["tvec"])
    yaw_diff = np.abs(current["yaw"] - initial["yaw"])
    pitch_diff = np.abs(current["pitch"] - initial["pitch"])
    roll_diff = np.abs(current["roll"] - initial["roll"])

    return tvec_diff < 0.02 and yaw_diff < 2 and pitch_diff < 2 and roll_diff < 2


print(
    "Press 'c' to capture the initial frame. Press 'm' to start the alignment process. Press 'q' to quit."
)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    corners, ids, _ = cv2.aruco.detectMarkers(gray, aruco_dict, parameters=aruco_params)

    if ids is not None:
        cv2.aruco.drawDetectedMarkers(frame, corners, ids)

        rvecs, tvecs, _ = cv2.aruco.estimatePoseSingleMarkers(
            corners, 0.05, cameraMatrix=camera_matrix, distCoeffs=dist_coeffs
        )

        for i, corner in enumerate(corners):
            qr_id = ids[i][0]
            rvec = rvecs[i][0]
            tvec = tvecs[i][0]

            dist = np.linalg.norm(tvec)
            rmat, _ = cv2.Rodrigues(rvec)
            yaw, pitch, roll = rotationMatrixToEulerAngles(rmat)

            current_data = {
                "qr_id": qr_id,
                "dist": dist,
                "yaw": yaw,
                "pitch": pitch,
                "roll": roll,
                "tvec": tvec,
            }

            if alignment_in_progress and initial_data is not None:
                if poses_aligned(initial_data, current_data):
                    print("Successfully returned to initial position!")
                    cv2.imwrite("final_position.jpg", cv2.flip(frame, 1))  # Save the final image
                    alignment_in_progress = False
                    break  # Stop further calculations
                else:
                    command = get_movement_command(initial_data, current_data)
                    current_time = time.time()

                    # Check if the user has moved sufficiently in the indicated direction
                    if command != last_command and (
                        last_command_time is None
                        or (current_time - last_command_time) > 2
                    ):
                        print(command)
                        last_command = command
                        last_command_time = current_time
                    elif (
                        command == last_command
                        and (current_time - last_command_time) > 2
                    ):
                        last_command_time = current_time

    # Mirror the camera feed for display
    frame = cv2.flip(frame, 1)

    cv2.imshow("Webcam Feed", frame)

    key = cv2.waitKey(1) & 0xFF
    if key == ord("c"):  # 'c' key is pressed to capture initial frame
        if ids is not None:
            initial_data = current_data
            cv2.imwrite("initial_position.jpg", frame)  # Save the initial image
            print(
                f"Initial capture - QR ID: {initial_data['qr_id']}, Distance: {initial_data['dist']:.2f}, Yaw: {initial_data['yaw']:.2f}, Pitch: {initial_data['pitch']:.2f}, Roll: {initial_data['roll']:.2f}\n"
            )
            print(
                "Move the camera to a different position and press 'm' to start the alignment process."
            )
        else:
            print("No ArUco markers detected.")
    elif key == ord("m"):  # 'm' key is pressed to start alignment
        if initial_data is not None:
            alignment_in_progress = True
            last_command = None
            last_command_time = time.time()
            print("Starting alignment process. Follow the instructions.")
        else:
            print("Capture the initial frame first by pressing 'c'.")
    elif key == ord("q"):  # 'q' key is pressed to quit
        break

cap.release()
cv2.destroyAllWindows()
