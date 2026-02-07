#!/bin/bash

#Path to Conda activation script
CONDA_SCRIPT="/home/alexandra-admin/opt/miniconda/etc/profile.d/conda.sh"

#Set this to the path of script directory
SCRIPT_DIR="$(dirname "$(realpath "$0")")"

#Function to open a new terminal and run a command
open_terminal() {
    local cmd=$1
    gnome-terminal -- bash -c "source $CONDA_SCRIPT && conda activate mne && $cmd; exec bash"
}

read -p "Do you want to start mne-lsl viewer? (default no) (y/n): " start_mne

echo "initializing experiment devices ..."

#STEP 1: Start eegoSports
echo "Starting eegoSports..."
open_terminal "eegoSports"

#STEP 2: Run UTIL_marker_stream.py
echo "Running UTIL_marker_stream.py..."
open_terminal "python3 $SCRIPT_DIR/UTIL_marker_stream.py"

#STEP 3: Start LabRecorder
echo "Starting LabRecorder..."
open_terminal "LabRecorder"

#Step 4: Optionally start mne-lsl viewer
if [[ "$start_mne" == "y" ]]; then
    echo "Starting mne-lsl viewer..."
    sleep 30
    open_terminal "mne-lsl viewer"
else
    echo "Skipping mne-lsl viewer."
fi

echo "Initialization complete. Check the opened terminals for individual processes."
