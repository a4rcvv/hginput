# HGInput: Hand Gesture Input

## For User

## For Developer

### Install Dependencies

Since `mediapipe` doesn't support Apple Silicon officially,
We use `mediapipe-silicon` ([GitHub](https://github.com/cansik/mediapipe-silicon)) instead
in the Apple Silicon environment.

#### Apple Silicon

```shell
poetry install -E arm
```

#### Others

```shell
poetry install -E x86
```