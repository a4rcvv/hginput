import json
import logging
import queue
import time
from datetime import datetime
from typing import Optional

import cv2
import mediapipe as mp
import numpy as np
import pyautogui
import torch

from hginput.datatypes.config import ConfigV1, GestureConfig
from hginput.datatypes.metadata import MetaData
from hginput.model.model import GestureClassifier
from hginput.util.camera import get_camera
from hginput.util.const import hand_landmarker_model_path
from hginput.util.image import draw_landmarks
from hginput.util.mediapipe import get_hand_landmarker_option
from hginput.util.time import current_ms

logger = logging.getLogger(__name__)


def should_execute_command(
    detected_time_ns: int,
    executed_time_ns: Optional[int],
    current_time_ns: int,
    gestureConfig: GestureConfig,
    commandExecutionIntervalSec: float,
) -> bool:
    """コマンドを実行する必要があるか判定"""
    if current_time_ns - detected_time_ns < commandExecutionIntervalSec * 1e9:
        return False

    if gestureConfig.onlyOnce:
        if executed_time_ns is None:
            return True
        else:
            return False
    else:
        if executed_time_ns is None:
            return True
        else:
            return current_time_ns - executed_time_ns > gestureConfig.intervalSec * 1e9


def send_key_event(keys: list[str]):
    """キーイベントを送信"""
    pyautogui.hotkey(*keys)


def launch(config_path: str):
    # 現在検知されているジェスチャー，正常に検知されてない場合は None
    detected_gesture: Optional[str] = None
    # detected_gesture がいつ最初に検知されたか(epoch time, ns)
    detected_time_ns: Optional[int] = None
    # detected_gestureに対応するコマンドが最後に実行された時刻，実行されてない場合はNone(epoch time, ns)
    executed_time_ns: Optional[int] = None
    img_queue: queue.Queue[np.ndarray] = queue.Queue()

    # load config file
    with open(config_path, "r") as f:
        config_json = json.load(f)
        config = ConfigV1.from_dict(config_json)
        with open(config.model.metadata_path, "r") as f2:
            metadata_json = json.load(f2)
            metadata = MetaData.from_dict(metadata_json)
    logging.info("succeeded to load config file")

    gestures_config_map: dict[str, GestureConfig] = {
        gesture.gesture: gesture for gesture in config.gestures
    }

    # initialize camera
    cap = get_camera(
        device=config.camera.device,
        width=config.camera.width,
        height=config.camera.height,
        fps=config.camera.fps,
    )
    # load model
    model = GestureClassifier.load_from_checkpoint(config.model.model_path)
    model.eval()

    def on_detection_completed(
        result: mp.tasks.vision.HandLandmarkerResult, image: mp.Image, timestamp_ms: int
    ):
        nonlocal detected_gesture, detected_time_ns, executed_time_ns
        annotated_image = draw_landmarks(image.numpy_view(), result)
        img_queue.put(annotated_image)
        hand_exists = True if len(result.handedness) != 0 else False
        if hand_exists:
            world_landmarks = result.hand_world_landmarks[0]
            # create input tensor
            input_list = (
                [0 if result.handedness[0][0].category_name == "Left" else 1]
                + [landmark.x for landmark in world_landmarks]
                + [landmark.y for landmark in world_landmarks]
                + [landmark.z for landmark in world_landmarks]
            )
            input_tensor = torch.FloatTensor(input_list)

            # predict
            with torch.no_grad():
                output_value = model(input_tensor)
                output_index = torch.argmax(output_value)
                confidence = torch.softmax(output_value, dim=0)[output_index]
            gesture_id = metadata.labels[output_index]
            logger.debug(f"detected gesture: {gesture_id}, confidence: {confidence}")

            # update states
            current_time_ns = time.time_ns()
            if confidence > config.detection.min_confidence:
                # valid prediction
                if detected_gesture != gesture_id:
                    # reset
                    detected_gesture = gesture_id
                    detected_time_ns = current_time_ns
                    executed_time_ns = None
            else:
                # invalid prediction
                detected_gesture = None
                detected_time_ns = None
                executed_time_ns = None

            # execute command if needed
            if detected_gesture is not None and detected_time_ns is not None:
                gesture_config = gestures_config_map.get(detected_gesture, None)
                if gesture_config is None:
                    logger.error(
                        f"gesture {detected_gesture} is not defined in config file"
                    )
                else:
                    if should_execute_command(
                        detected_time_ns,
                        executed_time_ns,
                        current_time_ns,
                        gesture_config,
                        config.detection.command_execution_interval_sec,
                    ):
                        logger.info(f"execute command. keys: {gesture_config.keys}")
                        send_key_event(gesture_config.keys)
                        executed_time_ns = current_time_ns
        else:
            # no hand
            # reset
            detected_gesture = None
            detected_time_ns = None
            executed_time_ns = None

    options = get_hand_landmarker_option(on_detection_completed)
    start_time = current_ms()

    with mp.tasks.vision.HandLandmarker.create_from_options(options) as landmarker:
        while True:
            succeeded, frame = cap.read()
            if not succeeded:
                logger.warning("failed to read frame")
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
