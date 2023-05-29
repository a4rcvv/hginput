import cv2


def get_camera(device: int, width: int, height: int, fps: int) -> cv2.VideoCapture:
    """Initialize a video capture device with the specified parameters.

    Args:
        device (int): Device ID of the video capture device.
        width (int): Width of the video capture.
        height (int): Height of the video capture.
        fps (int): FPS of the video capture.

    Returns:
        cv2.VideoCapture: Initialized video capture device.
    """
    cap = cv2.VideoCapture(device)
    if cap is None or not cap.isOpened():
        raise RuntimeError("Failed to open a video source")
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    cap.set(cv2.CAP_PROP_FPS, fps)
    return cap
