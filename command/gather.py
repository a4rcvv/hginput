import datetime
import queue
from pathlib import Path
import mediapipe as mp

import cv2
import pandas as pd

from util.image import draw_landmarks
from util.time import current_ms
from logging import getLogger

logger = getLogger(__name__)


def gather(label: str, device: int, width: int, height: int, fps: int):
    img_queue = queue.Queue()
    is_recording = False
    time = datetime.datetime.now().strftime("%Y%m%D%H%M%S")
    path = Path(f"./model/data/{label}_{time}.parquet.zstd")

    def on_detection_completed(
        result: mp.tasks.vision.HandLandmarkerResult, image: mp.Image, timestamp_ms: int
    ):
        annotated_image = draw_landmarks(image.numpy_view(), result)

        # Queue the annotated image.
        # Since this function will be called in a sub thread and
        # accessing to the GUI from a sub thread is prohibited,
        # we put the image to the queue.
        # The image will be popped and rendered in the main thread.
        img_queue.put(annotated_image)

        if is_recording:
            print(result)

    def on_start_recording():
        logger.info("Recording is started.")

    def on_end_recording():
        logger.info("Recording is ended. saving landmarks data...")
        # save

    # initialize a webcam
    cap = cv2.VideoCapture(device)
    if cap is None or not cap.isOpened():
        raise RuntimeError("Failed to open a video source")
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    cap.set(cv2.CAP_PROP_FPS, fps)

    options = mp.tasks.vision.HandLandmarkerOptions(
        base_options=mp.tasks.BaseOptions(model_asset_path="./hand_landmarker.task"),
        running_mode=mp.tasks.vision.RunningMode.LIVE_STREAM,
        result_callback=on_detection_completed,
        min_hand_detection_confidence=0.1,
        min_hand_presence_confidence=0.5,
        min_tracking_confidence=0.1,
    )

    start_time = current_ms()

    with mp.tasks.vision.HandLandmarker.create_from_options(options) as landmarker:
        while True:
            succeeded, frame = cap.read()
            if not succeeded:
                continue
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame)
            current_time = current_ms()

            # Execute a detection task.
            landmarker.detect_async(mp_image, current_time - start_time)

            # Pop the image from the queue, then show it.
            img = img_queue.get(block=True)
            cv2.imshow("image", img)

            key = cv2.waitKey(1)
            if key & 0xFF == ord("q"):
                # Quit if "Q" is pressed.
                break
            if key & 0xFF == ord("r"):
                # Switch a recording state if "R" is pressed.
                is_recording = not is_recording
                if is_recording:
                    on_start_recording()
                else:
                    on_end_recording()
