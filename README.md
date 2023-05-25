# HGInput: Hand Gesture Input

## Install Dependencies

```shell
poetry install
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

### Raw Data Structure

Raw data is a parquet file, which is compressed by zstandard.

Each file has these 65 columns (There are 21 hand landmarks(0th~20th), so 21*3 + 2 = 65).

- label: the gesture name of this sample.
- hand: "Left" or "Right".
  - we record this column **as MediaPipe says**(somehow MediaPipe recognizes a left hand as a right hand, and vice versa).
- x_{n}: the x axis value of {n}th landmark. presented in world coordinates.
- y_{n}: the y axis value of {n}th landmark. presented in world coordinates.
- z_{n}: the z axis value of {n}th landmark. presented in world coordinates.

### Dataset Structure

- label: integer-encoded gesture name.
- hand: 0 if "Left", 1 if "Right".
- x_{n}, y_{n}, z_{n}: as same as raw data.

#### Check

You can check the contents of the file by using [parquet-cli](https://github.com/chhantyal/parquet-cli)

