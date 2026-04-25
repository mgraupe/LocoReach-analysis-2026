# LocoReach Dataset — Raw Electrophysiology and Behavior Recordings

## Overview

This dataset contains raw in vivo extracellular recordings from the cerebellum
(molecular layer interneurons, MLI, and Purkinje cells, PC) of head-fixed mice
walking on a motorized, runged treadmill (LocoReach). Recordings were performed 
while animals walked on the wheel with regular (equally-spaced) rungs requiring 
precise step adjustments.

Each trial lasts approximately 60 seconds and contains:
- Single-unit electrophysiology (MLI or PC)
- Behavioral video (side- and bottom view of the animal on the ladder)
- Rotary encoder (treadmill speed)
- DAQ synchronization signals

## Animals and Recordings

| Animal ID    | Sex | MLI cells | MLI trials | PC cells | PC trials | Recording days |
|--------------|-----|-----------|------------|----------|-----------|----------------|
| 220211_f38   | F   | 6         | 27         | 3        | 13        | 8              |
| 220214_f43   | F   | 3         | 7          | 6        | 19        | 8              |
| 220205_f57   | F   | 4         | 20         | 3        | 15        | 6              |
| 220205_f61   | F   | 10        | 36         | 2        | 7         | 7              |
| 220507_m81   | M   | 8         | 27         | 3        | 13        | 8              |
| 220507_m90   | M   | 8         | 40         | 2        | 6         | 8              |
| 220525_m19   | M   | 5         | 18         | 4        | 20        | 8              |
| 220525_m27   | M   | 6         | 30         | 5        | 21        | 10             |
| 220525_m28   | M   | 7         | 34         | 4        | 19        | 10             |
| 220716_f65   | F   | 4         | 17         | 1        | 5         | 5              |
| 220716_f67   | F   | 3         | 12         | 1        | 5         | 4              |
| **Total**    |     | **64**    | **268**    | **34**   | **143**   | **—**          |

Animal ID encodes implantation date (YYMMDD) and animal number.
Total: 410 trial folders across 11 animals.

## Dataset Structure

The dataset is distributed as one `.tar.gz` archive per animal:

```
220211_f38.tar.gz
220214_f43.tar.gz
...
220716_f67.tar.gz
```

Each archive expands to:

```
<ANIMAL_ID>/
  <YYYY.MM.DD_NNN>/            # recording session folder
    locomotionEphys2Motor60sec_TTT/   # one folder per trial (TTT = trial number)
      AxoPatch200_2.ma               # electrophysiology patch-clamp amplifier signal
      DaqDevice.ma                   # DAQ synchronization and analog channels
      RotaryEncoder.ma               # rotary encoder (running speed)
      CameraGigEBehavior/
        video_000.avi                # behavioral video (MJPG, converted from .ma)
        daqResult.ma                 # camera trigger / frame timing
```

## File Formats

All `.ma` files are HDF5 files recorded by ACQ4 (open-source data acquisition software,
https://github.com/acq4/acq4). They can be opened with any HDF5 reader (e.g. h5py in
Python, HDFView, MATLAB's `h5read`).

### Electrophysiology files (`AxoPatch200_2.ma`, `DaqDevice.ma`, `RotaryEncoder.ma`)

Key HDF5 datasets:
- `/data` — 1-D array of samples (float32 or int16, depending on channel)
- `/info/0/` — channel metadata including sampling rate (`rate`) and physical units (`units`)

### Camera trigger file (`CameraGigEBehavior/daqResult.ma`)

- `/data` — 1-D array of the camera trigger signal (TTL pulses), same format as above

### Behavioral video (`CameraGigEBehavior/video_000.avi`)

Videos were originally recorded as ACQ4 `.ma` (HDF5) files and converted to MJPG
`.avi` for broad compatibility. Each frame is an 8-bit grayscale image. Frame
timestamps are embedded in the original `.ma` file (`/info/0/values`). Videos were 
recorded at 200 fps.

To read frames in Python:
```python
import cv2
cap = cv2.VideoCapture('video_000.avi')
while True:
    ret, frame = cap.read()
    if not ret:
        break
    # frame is (H, W, 3) uint8 BGR; convert with cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
cap.release()
```

## Acquisition Details

- Electrophysiology: patch-clamp amplifier (Axopatch 200B), loose-seal cell-attached
  configuration; signals digitized at 20 kHz
- Camera: GigE machine-vision camera, 200 fps, side and bottom view
- Locomotion apparatus: motorized rung ladder (LocoReach); rung spacing either
  regular (equal spacing) or irregular (randomized per trial)

## Related Resources

- Analysis code: https://github.com/mgraupe/LocoRungs
- ACQ4 documentation: https://acq4.readthedocs.io
