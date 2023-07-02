from dataclasses import dataclass
from typing import Optional

import dataclasses_json
from dataclasses_json import LetterCase, dataclass_json


@dataclass_json(letter_case=LetterCase.CAMEL)
@dataclass
class GestureConfig:
    gesture: str
    keys: list[str]
    onlyOnce: bool
    intervalSec: float


@dataclass_json(letter_case=LetterCase.CAMEL)
@dataclass
class MouseGestureConfig:
    point_gesture: str
    click_gesture: str
    move_scale: float


@dataclass_json(letter_case=LetterCase.CAMEL)
@dataclass
class CameraConfig:
    device: int
    width: int
    height: int
    fps: int


@dataclass_json(letter_case=LetterCase.CAMEL)
@dataclass
class ModelConfig:
    model_path: str
    metadata_path: str


@dataclass_json(letter_case=LetterCase.CAMEL)
@dataclass
class DetectionConfig:
    min_confidence: float
    command_execution_interval_sec: float


@dataclass_json(letter_case=LetterCase.CAMEL)
@dataclass
class ConfigV1(dataclasses_json.DataClassJsonMixin):
    version: int
    gestures: list[GestureConfig]
    mouse_gestures: MouseGestureConfig
    camera: CameraConfig
    model: ModelConfig
    detection: DetectionConfig

    # @classmethod
    # def from_json(cls, json: dict) -> "ConfigV1":
    #     return ConfigV1(
    #         version=json["version"],
    #         gestures=[
    #             GestureConfig(
    #                 gesture=gesture["gesture"],
    #                 keys=gesture["keys"],
    #                 onlyOnce=gesture["onlyOnce"],
    #                 interval=gesture["interval"],
    #             )
    #             for gesture in json["gestures"]
    #         ],
    #         camera=CameraConfig(
    #             device=json["camera"]["device"],
    #             width=json["camera"]["width"],
    #             height=json["camera"]["height"],
    #             fps=json["camera"]["fps"],
    #         ),
    #         model=ModelConfig(
    #             model_path=json["model"]["modelPath"],
    #             metadata_path=json["model"]["metadataPath"],
    #         ),
    #         detection=DetectionConfig(
    #             min_confidence=json["detection"]["minConfidence"]
    #         ),
    #     )
