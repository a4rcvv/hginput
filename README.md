# HGInput: Hand Gesture Input

## Install Dependencies

Since `mediapipe` doesn't support Apple Silicon officially,
We use `mediapipe-silicon` ([GitHub](https://github.com/cansik/mediapipe-silicon)) instead
in the Apple Silicon environment.

### Apple Silicon

```shell
poetry install -E arm
```

### Others

```shell
poetry install -E x86
```

## Run the app

```shell
python entry.py [COMMAND] [OPTIONS]
```

You can show help messages by `python entry.py --help`.

## Directory Structure

### command

The entry points of commands.

### model

Training data and trained model

### util

Utility functions.

### entry.py

The entry point of this system.

## Model Structure

### Data Structure

Training data is some parquet files, which are compressed by zstandard.

Each files has these 65 columns (There are 21 hand landmarks(0th~20th), so 21*3 + 2 = 65).

- label: the gesture name of this sample.
- hand: "Left" or "Right".
  - we record this column **as MediaPipe says**(somehow MediaPipe recognizes a left hand as a right hand, and vice versa).
- x_{n}: the x axis value of {n}th landmark. presented in world coordinates.
- y_{n}: the y axis value of {n}th landmark. presented in world coordinates.
- z_{n}: the z axis value of {n}th landmark. presented in world coordinates.

#### Check

You can check the contents of the file by using [parquet-cli](https://github.com/chhantyal/parquet-cli)

