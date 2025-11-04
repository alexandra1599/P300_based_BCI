# Visual Interface and UDP Streams

In this project, we use LSL for EEG data acquisition. The amplifiers and caps are provided by AntNeuro.
In this folder you will find 3 codes.

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

**UTIL_marker_stream.py** is the code that will receive the markers from nback.py and synchronize them to the EEG time-series. This way, it will pair every marker received to a specific EEG timepoint when it happens and save that 
in the .xdf file alongside the EEG raw data.

