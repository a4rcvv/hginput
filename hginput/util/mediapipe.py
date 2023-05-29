import mediapipe as mp
from hginput.util.const import hand_landmarker_model_path
from typing import Callable


def get_hand_landmarker_option(
    on_detection_completed: Callable[
        [mp.tasks.vision.HandLandmarkerResult, mp.Image, int], None
    ]
) -> mp.tasks.vision.HandLandmarkerOptions:
    return mp.tasks.vision.HandLandmarkerOptions(
        base_options=mp.tasks.BaseOptions(model_asset_path=hand_landmarker_model_path),
        running_mode=mp.tasks.vision.RunningMode.LIVE_STREAM,
        result_callback=on_detection_completed,
        min_hand_detection_confidence=0.1,
        min_hand_presence_confidence=0.5,
        min_tracking_confidence=0.1,
    )
