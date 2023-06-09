import datetime
import queue
from pathlib import Path
import mediapipe as mp

import cv2
import polars as pl

from hginput.util.image import draw_landmarks
from hginput.util.time import current_ms
from hginput.util.const import hand_landmarker_model_path
from hginput.util.mediapipe import get_hand_landmarker_option
from hginput.util.camera import get_camera
import numpy as np
from logging import getLogger

logger = getLogger(__name__)
n_landmarks = 21


def _make_empty_record_dict() -> dict[str, list]:
    columns = (
        ["label", "hand"]
        + [f"x_{n}" for n in range(n_landmarks)]
        + [f"y_{n}" for n in range(n_landmarks)]
        + [f"z_{n}" for n in range(n_landmarks)]
    )
    return {column: [] for column in columns}


def gather(label: str, device: int, width: int, height: int, fps: int):
    img_queue: queue.Queue[np.ndarray] = queue.Queue()
    is_recording = False
    records = _make_empty_record_dict()

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
        hand_exists = True if len(result.handedness) != 0 else False

        if is_recording and hand_exists:
            records["label"].append(label)
            records["hand"].append(result.handedness[0][0].category_name)
            world_landmarks = result.hand_world_landmarks[0]
            for i in range(n_landmarks):
                landmark = world_landmarks[i]
                records[f"x_{i}"].append(landmark.x)
                records[f"y_{i}"].append(landmark.y)
                records[f"z_{i}"].append(landmark.z)

    def on_start_recording():
        logger.info("Recording is started.")

    def on_end_recording():
        logger.info("Recording is ended. saving landmarks data...")
        time = datetime.datetime.now().strftime("%Y%m%d%H%M%S")
        file_name = f"{time}.parquet.zstd"
        path = Path(f"./hginput/model/data/raw/{label}/{file_name}")
        df = pl.DataFrame(records)
        df.write_parquet(path)
        logger.info(f"Succeeded to save landmarks. path: {path}")
        records.clear()
        records.update(_make_empty_record_dict())

    cap = get_camera(device, width, height, fps)
    options = get_hand_landmarker_option(on_detection_completed)
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
