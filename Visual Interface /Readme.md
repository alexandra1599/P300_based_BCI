# Task, Control panel and UDP Streams

In this project, we use LSL for EEG data acquisition. The amplifiers and caps are provided by AntNeuro.
In this folder you will find 3 codes.

## Task Code

**nback.py** is the code to run the visual interface. The Nback task is coded such that the subject will see a sequence of numbers on the screen followed by 3 crosses (white, green, white in this order).
The goal is to answer by a mouse button press if the number just seen is the same (right click) or not (left click) as the one seen N times back.

The green cross is implemented to give time to the subject to blink in the trial and lasts 1500 ms. The two white crosses are implemented so that the subject fixates on them and prepares for the next trial. They last 800 ms.
At the end of the task, the score will be shown. 
The parameters that can be changed are :
- Duration of the fixation crosses (white or green)
- N (Nback difficulty)
- % of total trials that are target trials (usually 30% for it to be considered "rare" stimuli)
- Total number of trials per run
- Markers (these are values to lcok specific events to time-points in the EEG data)

## LSL Marker Stream Utility - UTIL_marker_stream.py
### Overview
This utility provides synchronized marker injection into EEG data streams using Lab Streaming Layer (LSL). It listens for marker commands via UDP and timestamps them precisely against the EEG stream timeline, ensuring accurate event marking for neuroscience experiments.

### Features:
- LSL Integration: Creates an LSL outlet stream (MarkerStream) for sending event markers
- EEG Synchronization: Pulls timestamps directly from the EEG LSL stream to ensure precise temporal alignment
- UDP Interface: Listens on port 12345 for remote marker commands
- Predefined Markers: Validates marker values against a whitelist (0, 100, 200, 300, 400, 500, 1, 2, 4, 11, 12)
- Thread-Safe: Runs UDP listener in a separate daemon thread

### How It Works:
- Resolves and connects to an available EEG LSL stream
- Creates an LSL outlet for marker events (2-channel: marker value + timestamp)
- Listens for UDP messages containing marker values
- When a marker is received:
    - Flushes the EEG inlet buffer
    - Pulls the most recent EEG sample timestamp
    - Pushes the marker value and EEG-aligned timestamp to the LSL outlet
 
### Usage:
python UTIL_marker_stream.py

### Requirements:
- pylsl - Lab Streaming Layer Python interface
- An active EEG LSL stream (type: 'EEG')

### Install dependencies: pip install pylsl

### Notes:
- The utility continuously runs until interrupted (Ctrl+C)
- EEG stream must be available before starting this utility
- Timestamps are synchronized to the EEG clock, not system time
- Stream offset is calculated and logged for each marker


## Device Initialization Script - initialize_devices.sh
### Overview
This bash script automates the initialization and startup of all required components for EEG data acquisition experiments. It launches each application in a separate GNOME terminal window with the appropriate conda environment activated.

### Features: 
- **Automated Startup:** Launches all experimental components in the correct order
- **Conda Environment Management:** Automatically activates the mne conda environment for each process
- **Separate Terminals:** Opens each component in its own terminal window for easy monitoring
- **Optional MNE-LSL Viewer:** Prompts user to optionally start the real-time EEG visualization tool
- **Interactive:** Keeps terminals open after process completion for debugging

### Components Initialized:
**1. eegoSports** EEG acquisition software for eegoSports amplifier system
**2. Marker Stream Utility** LSL marker stream handler (UTIL_marker_stream.py) that synchronizes event markers with EEG timestamps
**3. LabRecorder** LSL recording application for saving all synchronized data streams to disk
**4. MNE-LSL Viewer (Optional)** Real-time EEG visualization tool for monitoring signal quality during acquisition





