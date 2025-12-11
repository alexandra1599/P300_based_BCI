import numpy as np

def filter_trials(data, always_positive_threshold = 80, peak_to_peak_percentile=90):
  """
  Filter P300 single trial data to remove artifact or high noise trials.
  Inputs:
    - data (ndarray (time,trials)) : EEG single trial data at a specific electrode (i.e Pz)
    - always_positive_threshold (float): Rejects trials where more than this % of points are positive (or negative)
    - peak_to_peak_percentile (float): Reject trials above this percentile peak-to-peak amplitude (0-100)

  Returns:
    - clean_data (ndarray): Filtered trials, shaope (time,n_clean_trials)
    - rejected_indices (ndarray): Indices of rejected trials
  """
  n_timepoints, n_trials = data.shape
  
  #Calculate metrics for each trial
  peak_to_peak = np.max(data,axis=0) - np.min(data,axis=0)
  fraction_positive = np.sum(data > 0, axis=0) / n_timepoints * 100
  fraction_negative = np.sum(data < 0, axis=0) / n_timepoints *100
  
  #Determine threshold
  p2p_threshold = np.percentile(peak_to_peak, peak_to_peak_percentile)
  
  #Find bad trials 
  bad_polarity = (fraction_positive > always_positive_threshold) | (fraction_negative > always_positive_threshold)
  bad_p2p = peak_to_peak > p2p_threshold
  rejected_mask = bad_polarity | bad_p2p
  clean_data = data[:,~rejected_mask]
  rejected_indices = np.where(rejected_mask)[0]
  print(f"P300 Filtering: Rejected {len(rejected_indices)}/{n_trials} trials"
        f"({len(rejected_indices)/n_trials*100:.1f}%)")

  return clean_data, rejected_indices
