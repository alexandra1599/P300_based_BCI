import numpy as np

def remove_aux(eeg_list):
    """
    Removes AUX channels and extracts EOG channels from EEG data.
    
    Assumes:
    - VEOG is at column index 36 (AUX 7)
    - HEOG is at column index 37 (AUX 8)
    - Last 7 columns are AUX channels to be removed
    
    Parameters:
        eeg_list (list): List of EEG numpy arrays (time x channels)
    
    Returns:
        eeg_clean (list): EEG data with AUX channels removed
        EOG (list): Extracted EOG signals (time x 2) per run
    """
    eeg_clean = []
    EOG = []
    AUX_COLUMNS = 7

    for eeg in eeg_list:
        VEOG = eeg[:, 36] if eeg.shape[1] > 36 else np.zeros(eeg.shape[0])
        HEOG = eeg[:, 37] if eeg.shape[1] > 37 else np.zeros(eeg.shape[0])
        EOG.append(np.column_stack((VEOG, HEOG)))
        
        eeg_wo_aux = eeg[:, :-AUX_COLUMNS]  # remove last 7 columns
        eeg_clean.append(eeg_wo_aux)

    return eeg_clean, EOG
