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