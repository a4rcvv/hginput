import cv2
import mediapipe as mp
import numpy as np
from mediapipe import solutions
from mediapipe.framework.formats import landmark_pb2

MARGIN = 10  # pixels
FONT_SIZE = 1
FONT_THICKNESS = 2
HANDEDNESS_TEXT_COLOR = (88, 205, 54)  # vibrant green


def draw_landmarks(
    rgb_image: np.ndarray, detection_result: mp.tasks.vision.HandLandmarkerResult
) -> np.ndarray:
    """Draw landmarks on the captured image."""

    annotated_image = np.copy(rgb_image)

    # Loop through the detected hands to visualize.
    for idx in range(len(detection_result.hand_landmarks)):
        hand_landmarks = detection_result.hand_landmarks[idx]
        handedness = detection_result.handedness[idx]

        # Draw the hand landmarks.
        hand_landmarks_proto = landmark_pb2.NormalizedLandmarkList()
        hand_landmarks_proto.landmark.extend(
            [
                landmark_pb2.NormalizedLandmark(
                    x=landmark.x, y=landmark.y, z=landmark.z
                )
                for landmark in hand_landmarks
            ]
        )
        solutions.drawing_utils.draw_landmarks(
            annotated_image,
            hand_landmarks_proto,
            solutions.hands.HAND_CONNECTIONS,
            solutions.drawing_styles.get_default_hand_landmarks_style(),
            solutions.drawing_styles.get_default_hand_connections_style(),
        )

    return annotated_image


def draw_gesture_id(rgb_image: np.ndarray, gesture_id: str) -> np.ndarray:
    """Draw gesture ID on the captured image."""

    annotated_image = np.copy(rgb_image)

    text_x = 0
    text_y = 30

    # Draw gesture ID on the image.
    cv2.putText(
        annotated_image,
        f"{gesture_id}",
        (text_x, text_y),
        cv2.FONT_HERSHEY_DUPLEX,
        FONT_SIZE,
        HANDEDNESS_TEXT_COLOR,
        FONT_THICKNESS,
        cv2.LINE_AA,
    )

    return annotated_image
