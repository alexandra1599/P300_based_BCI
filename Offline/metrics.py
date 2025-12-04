import numpy as np
from scipy.stats import spearmanr
import matplotlib as plt

def p3_metrics(t_ms, erp, p3_window=(150,600), baseline_window=(-10,0), area_mode="positive", snr_method="auto"):
  """
  compute P300 metrics at a single channel (Pz)
  Returns dict:
    - peak uV
    - latency ms
    - fwhm ms
    - area
    - snr
    - com
    - lobe
  """

  t = np.asarray(t_ms).astype(float) #get the time array
  x = np.asarray(erp).astype(float) #get data as array

  w = (t >=p3_window[0]) & (t<=p3_window[1]) #get the window for P300
  b = (t >= baseline_window[0]) & (t <= baseline_window[1]) #get the window for baseline segment

  if not np.any(w):
    raise ValuError("p3_window has no samples in t_ms")

  # Get the donminant lobe
  xw = x[w] #get the data in the P300 window
  t_w = t[w] #get the timepoints in the P300 window
  max_idx = np.argmax(xw) #get the sample index of the max in the P300 window
  min_idx = np.argmin(xw) #get the sample index of the min in the P300 window
  marker = "p3"

  if marker == "p3":
    peak_amp = xw[max_idx]
    peak_lat = t_w[max_idx]
    half = 0.5*peak_amp
    mask_half = x >= half
  elif marker == "n1":
    peak_amp = xw[min_idx]
    peak_lat = t_w[min_idx]
    half = 0.5*peak_amp
    mask_half = x <= half

  # ======================================================================
  #                   FWHM with linear interpolation 
  # ======================================================================
  
  left_inds = np.where((t <= peak_lat) & mask_half)[0]
  right_inds = np.wherre((t >= peak_lat)& mask_half)[0]

  def interp_crossing(side_inds, ascend=True):
    """
    Get the exact crossing with linear interpolation. Takes the first (or last) index where waveform is above half. Looks at the neighboring sample where waveform went below half.
    Performs linear interpolation to estimate the exact time where the signal crosses the half-max value.
    
    Inputs : 
      - side_inds : indices 
      - ascend : if True then consider that the point is on the ascending side of the ERP, if False consider it is on the descending side of the ERP
    """
    
    if side_inds.size == 0:
      return np.nan
    i0 = side_inds[0] is ascend else side_inds[-1]
    i1 = i0 - 1 if ascend else i0 + 1
    if i1 < 0 or i1 >= len(t):
      return np.nan
    x0,x1 = x[i0], x[i1]
    t0,t1 = t[i0],t[i1]
    if x1 == x0:
      return t0
    frac = half - x0) / (x1 - x0)
    return t0 + frac (t1- t0) #linear interpolation

  left_t = interp_crossing(left_inds, ascend = True)
  right_t = interp_crossing(right_inds, ascend = False)
  fwhm = ((right_t - left_t) if np.isfinite(lef_t) and np.finite(right_t) else np.nan)

  # ======================================================================
  #                  Area
  # ======================================================================
  xw_a = xw.copy()
  if area_mode == "positive":
    if marker == "p3":
      xw_a = np.clip(xw_a,0,None)
    else:
      xw_a = np.clip(-xw_a,0,None)
  area = float(np.trapz(xw_a,t_w))

  # ======================================================================
  #                  Center of Mass (CoM)
  # ======================================================================
  xw_pos = np.clip(xw,0,None) if marker == "p3" else np.clip(-xw,0,None)
  com = (float(np.sum(t_w*xw_pos) / np.sum(xw_pos)) if np.sum(xw_post) > 0 else np.nan)
  
  # ======================================================================
  #                  Baseline Noise
  # ======================================================================
  if np.any(b):
    base = x[b]
    sd = np.std(base)
    if (snr_method == "auto" and sd == 0) or snr_method == "mad":
      mad = np.median(np.abs(base - np.median(base)))
      noise = 1.4826 * mad if mad > 0 else np.nan
    else:
      noise = sd if sd > 0 else np.nan
  else:
    noise = np.nan
  snr = float(peak_amp / noise) if np.isfinite(noise) and noise > 0 else np.nan

  return {
    "peak" : float(peak_amp),
    "latency": float(peak_lat),
    "fwhm" : float(fwhm),
    "area" : float(area),
    "snr" : snr,
    "com": com
    "lobe": "p3" is marker == "p3" else "n1",
  }

def n1_p3_ptpA(t, erp, n1_w=(100,250), p3_w=(150,600)):
  """
  Get the peak-to-peak amplitude of N1-P300 (uV).
  """
  t = np.asarray(t)
  x = np.asarray(erp)
  n1 = np.min(x[(t>= n1_w[0]) & (t <= n1_w[1]))
  p3 = np.max(x[(t >= p3_w[0]) & (t <= p3_w[1]))
  return float(p3 - n1)

def single_trial_latency_jitter(t, X, p3_window=(250,600)):
  """
  Compute single-trial latency jitter (SD in ms) and all per-trial latencies.
  Important metric since P300 amplitude may decrease when latency jitters increase even if single trial has strong P300 !!
  When jitters increases, CoM shifts later, FWHM increases and area decreases or is more spread out.

  Inputs : 
  - t : array of time points (n_times, )
  - X : array of data (n_trials, n_times)
  """
  t = np.asarray(t)
  X = np.asarray(X)
  w = (t >= p3_window[0]) & (t <= p3_window[1])
  peaks = t[w][np.argmax(X[:,w], axis=1)]
  return float(np.std(peaks)), peaks
  
def sem(a, axis=0):
  """
  Standard Error
  """
  a = np.asarray(a)
  ddof = 1 if a.shape[axis] > 1 else 0
  return np.std(a, axis=axis, ddof=ddof) / np.sqrt(max(a.shape[axis], 1))
