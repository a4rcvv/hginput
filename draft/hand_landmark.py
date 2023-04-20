"""
An example of hand landmark detection.

Usage:
    1. Download a hand landmark model bundle.
       You can download it at here: https://developers.google.com/mediapipe/solutions/vision/hand_landmarker/index#models
    2. Save the model to ${WORKDIR}/hand_landmarker.task.
    3. Run this script.
    4. You can quit the script by pressing Q.

References:
    https://developers.google.com/mediapipe/solutions/vision/hand_landmarker/python
"""

import cv2
import mediapipe as mp
import time
import numpy as np
from mediapipe.framework.formats import landmark_pb2
from mediapipe import solutions
import queue

# modification of https://github.com/googlesamples/mediapipe/blob/main/examples/hand_landmarker/python/hand_landmarker.ipynb
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


# constants
DEVICE_ID = 0
WIDTH = 480
HEIGHT = 270
FPS = 30
MARGIN = 10  # pixels
FONT_SIZE = 2
FONT_THICKNESS = 2
HANDEDNESS_TEXT_COLOR = (88, 205, 54)  # vibrant green

img_queue = queue.Queue()


def current_ms():
    return round(time.time() * 1000)


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

        # Get the top left corner of the detected hand's bounding box.
        height, width, _ = annotated_image.shape
        x_coordinates = [landmark.x for landmark in hand_landmarks]
        y_coordinates = [landmark.y for landmark in hand_landmarks]
        text_x = int(min(x_coordinates) * width)
        text_y = int(min(y_coordinates) * height) - MARGIN

        # Draw handedness (left or right hand) on the image.
        cv2.putText(
            annotated_image,
            f"{handedness[0].category_name}",
            (text_x, text_y),
            cv2.FONT_HERSHEY_DUPLEX,
            FONT_SIZE,
            HANDEDNESS_TEXT_COLOR,
            FONT_THICKNESS,
            cv2.LINE_AA,
        )

    return annotated_image


def callback(
    result: mp.tasks.vision.HandLandmarkerResult, image: mp.Image, timestamp_ms: int
):
    print(result.handedness)

    annotated_image = draw_landmarks(image.numpy_view(), result)

    # Queue the annotated image.
    # Since this function will be called in a sub thread and
    # accessing to the GUI from a sub thread is prohibited,
    # we put the image to the queue.
    # The image will be popped and rendered in the main thread.
    img_queue.put(annotated_image)


options = mp.tasks.vision.HandLandmarkerOptions(
    base_options=mp.tasks.BaseOptions(model_asset_path="./hand_landmarker.task"),
    running_mode=mp.tasks.vision.RunningMode.LIVE_STREAM,
    result_callback=callback,
    min_hand_detection_confidence=0.1,
    min_hand_presence_confidence=0.5,
    min_tracking_confidence=0.1,
)

# initialize a webcam
cam = cv2.VideoCapture(DEVICE_ID)
cam.set(cv2.CAP_PROP_FRAME_WIDTH, WIDTH)
cam.set(cv2.CAP_PROP_FRAME_HEIGHT, HEIGHT)
cam.set(cv2.CAP_PROP_FPS, FPS)

start_time = current_ms()

with mp.tasks.vision.HandLandmarker.create_from_options(options) as landmarker:
    while True:
        succeeded, frame = cam.read()
        if not succeeded:
            continue
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame)
        current_time = current_ms()
        # Execute a detection task.
        landmarker.detect_async(mp_image, current_time - start_time)
        # Pop the image from the queue, then show it.
        img = img_queue.get(block=True)
        cv2.imshow("image", img)
        # Quit if "Q" is pressed.
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
