import numpy as np
from scipy.stats import spearmanr


def p3_metrics(
    t_ms,
    erp,
    p3_window=(300, 600),
    baseline_window=(-200, 0),
    area_mode="positive",
    snr_method="auto",
):
    """
    Compute P3 metrics at a single channel (e.g., Pz).

    Returns dict:
        peak_uv, latency_ms, fwhm_ms, area_uv_ms, snr, com_ms, lobe
    """
    t = np.asarray(t_ms).astype(float)
    x = np.asarray(erp).astype(float)

    w = (t >= p3_window[0]) & (t <= p3_window[1])
    b = (t >= baseline_window[0]) & (t <= baseline_window[1])

    if not np.any(w):
        raise ValueError("p3_window has no samples in t_ms")

    # dominant lobe (positive vs negative)
    xw = x[w]
    t_w = t[w]
    max_idx = np.argmax(xw)
    min_idx = np.argmin(xw)
    use_positive = abs(xw[max_idx]) >= abs(xw[min_idx])

    if use_positive:
        peak_amp = xw[max_idx]
        peak_lat = t_w[max_idx]
        half = 0.5 * peak_amp
        mask_half = x >= half
    else:
        peak_amp = xw[min_idx]
        peak_lat = t_w[min_idx]
        half = 0.5 * peak_amp
        mask_half = x <= half

    # --- FWHM with interpolation
    left_inds = np.where((t <= peak_lat) & mask_half)[0]
    right_inds = np.where((t >= peak_lat) & mask_half)[0]

    def interp_crossing(side_inds, ascend=True):
        if side_inds.size == 0:
            return np.nan
        i0 = side_inds[0] if ascend else side_inds[-1]
        i1 = i0 - 1 if ascend else i0 + 1
        if i1 < 0 or i1 >= len(t):
            return np.nan
        x0, x1 = x[i0], x[i1]
        t0, t1 = t[i0], t[i1]
        if x1 == x0:
            return t0
        frac = (half - x0) / (x1 - x0)
        return t0 + frac * (t1 - t0)

    left_t = interp_crossing(left_inds, ascend=True)
    right_t = interp_crossing(right_inds, ascend=False)
    fwhm = (
        (right_t - left_t) if np.isfinite(left_t) and np.isfinite(right_t) else np.nan
    )

    # --- Area
    xw_for_area = xw.copy()
    if area_mode == "positive":
        if use_positive:
            xw_for_area = np.clip(xw_for_area, 0, None)
        else:
            xw_for_area = -np.clip(-xw_for_area, 0, None)
    area = float(np.trapz(xw_for_area, t_w))

    # --- Center of mass
    xw_pos = np.clip(xw, 0, None) if use_positive else np.clip(-xw, 0, None)
    com_ms = (
        float(np.sum(t_w * xw_pos) / np.sum(xw_pos)) if np.sum(xw_pos) > 0 else np.nan
    )

    # --- Baseline noise
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
        "peak_uv": float(peak_amp),
        "latency_ms": float(peak_lat),
        "fwhm_ms": float(fwhm),
        "area_uv_ms": float(area),
        "snr": snr,
        "com_ms": com_ms,
        "lobe": "positive" if use_positive else "negative",
    }


def n2_p3_ptp(t_ms, erp, n2_w=(180, 300), p3_w=(300, 600)):
    """Peak-to-peak N2–P3 amplitude (µV)."""
    t = np.asarray(t_ms)
    x = np.asarray(erp)
    n2 = np.min(x[(t >= n2_w[0]) & (t <= n2_w[1])])
    p3 = np.max(x[(t >= p3_w[0]) & (t <= p3_w[1])])
    return float(p3 - n2)


def single_trial_latency_jitter(t_ms, X, p3_window=(300, 600)):
    """
    Compute single-trial latency jitter (SD in ms) and all per-trial latencies.

    Parameters
    ----------
    t_ms : array-like, shape (n_times,)
    X : array-like, shape (n_trials, n_times)
    """
    t = np.asarray(t_ms)
    X = np.asarray(X)
    w = (t >= p3_window[0]) & (t <= p3_window[1])
    peaks = t[w][np.argmax(X[:, w], axis=1)]
    return float(np.std(peaks)), peaks


import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans


def trials_time_from_tc_tr(data_tc_tr, ch_idx):
    """data_tc_tr: (time, chan, trials) → returns X: (trials, time) at ch_idx"""
    return data_tc_tr[:, ch_idx, :].T


def sem(a, axis=0):
    a = np.asarray(a)
    ddof = 1 if a.shape[axis] > 1 else 0
    return np.std(a, axis=axis, ddof=ddof) / np.sqrt(max(a.shape[axis], 1))


def cluster_latency_and_plot(
    t_ms,
    X,
    title,
    fast_window=(250, 400),
    slow_window=(420, 620),
    n_clusters=2,
    random_state=0,
    show=True,
):
    """
    X: (trials, time) at one channel.
    Returns: dict with jitter_sd, per_trial_lat, labels, centers, n_per_cluster
    and the cluster ERPs (erp0, erp1, sem0, sem1).
    """
    # latencies & jitter
    w = (t_ms >= 300) & (t_ms <= 600)
    per_trial_lat = t_ms[w][np.argmax(X[:, w], axis=1)]
    jitter_sd = float(np.std(per_trial_lat))

    # cluster on latency
    lat = per_trial_lat.reshape(-1, 1)
    km = KMeans(n_clusters=n_clusters, n_init=20, random_state=random_state).fit(lat)
    labels = km.labels_
    centers = sorted(km.cluster_centers_.ravel())
    # cluster splits
    X0, X1 = X[labels == 0], X[labels == 1]
    erp0, erp1 = X0.mean(axis=0), X1.mean(axis=0)
    sem0 = sem(X0, axis=0) if len(X0) > 1 else np.zeros_like(erp0)
    sem1 = sem(X1, axis=0) if len(X1) > 1 else np.zeros_like(erp1)

    if show:
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.plot(t_ms, erp0, label=f"Cluster 0 (n={len(X0)})")
        ax.fill_between(t_ms, erp0 - sem0, erp0 + sem0, alpha=0.2)
        ax.plot(t_ms, erp1, label=f"Cluster 1 (n={len(X1)})")
        ax.fill_between(t_ms, erp1 - sem1, erp1 + sem1, alpha=0.2)
        ax.axvspan(
            fast_window[0],
            fast_window[1],
            alpha=0.10,
            label=f"FAST window ({fast_window[0]}–{fast_window[1]} ms)",
        )
        ax.axvspan(
            slow_window[0],
            slow_window[1],
            alpha=0.10,
            label=f"SLOW window ({slow_window[0]}–{slow_window[1]} ms)",
        )
        ax.set_title(title)
        ax.set_xlabel("Time (ms)")
        ax.set_ylabel("Amplitude (µV)")
        ax.axvline(0, linestyle="--", linewidth=1)
        ax.legend(loc="best")
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.show()

    return {
        "jitter_sd": jitter_sd,
        "per_trial_lat": per_trial_lat,
        "labels": labels,
        "centers_ms": centers,
        "n0": int((labels == 0).sum()),
        "n1": int((labels == 1).sum()),
        "erp0": erp0,
        "erp1": erp1,
        "sem0": sem0,
        "sem1": sem1,
    }


import numpy as np
from scipy.signal import correlate


def shift_no_wrap(x, shift, pad=0.0):
    """Shift 1D array by integer samples; pad with `pad` instead of wrap."""
    y = np.full_like(x, pad)
    if shift > 0:
        y[shift:] = x[:-shift]
    elif shift < 0:
        y[:shift] = x[-shift:]
    else:
        y[:] = x
    return y


def woody_align_window(
    t, X, window=(300, 600), max_shift_ms=80, pad=0.0, rebaseline=(-200, 0)
):
    """
    X: (trials × time). Aligns only using the window; shifts the WHOLE trial with no wrap.
    Then re-baselines to keep amplitude comparable.
    """
    t = np.asarray(t)
    X = np.asarray(X)
    dt = t[1] - t[0]
    w = (t >= window[0]) & (t <= window[1])
    ref = np.nanmean(X, axis=0)  # reference ERP
    ref_w = ref[w] - np.mean(ref[(t >= rebaseline[0]) & (t <= rebaseline[1])])

    max_shift = int(round(max_shift_ms / dt))
    shifts = np.zeros(X.shape[0], dtype=int)
    X_aligned = np.empty_like(X)

    for i in range(X.shape[0]):
        xi = X[i].copy()
        # re-baseline first
        base_mask = (t >= rebaseline[0]) & (t <= rebaseline[1])
        xi -= np.mean(xi[base_mask])

        # xcorr inside the window
        c = correlate(xi[w], ref_w, mode="full")
        lags = np.arange(-len(ref_w) + 1, len(ref_w))
        m = (lags >= -max_shift) & (lags <= max_shift)
        best_lag = lags[m][np.argmax(c[m])]
        shifts[i] = best_lag

        # shift entire trial with zero-padding (no wrap)
        X_aligned[i] = shift_no_wrap(xi, best_lag, pad=0.0)

        # re-baseline again (keeps comparability after shift)
        X_aligned[i] -= np.mean(X_aligned[i][base_mask])

    return X_aligned, shifts * dt


def run_analysis(
    ID,
    session,
    labels,
    p300_pre,
    p300_post,
    nop300_pre,
    nop300_post,
    p300_online,
    nop300_online,
    fs=512,
    comparison=None,
):
    import numpy as np
    import matplotlib.pyplot as plt
    from scipy.ndimage import gaussian_filter1d

    # === Channel indices (adjust based on session) ===
    if session == "tACS":
        pz, cz, c1, c2 = 11, 15, 19, 21
    elif session == "Relaxation":
        pz, cz, cpz, c1, c2 = 26, 15, 20, 19, 21

    prompt = "CPP[1] or P300[2] or Target[3] analysis: "
    c = int(input(prompt))

    time_ms = np.arange(p300_pre.shape[0]) * 1000 / fs

    if c == 1:  # === CPP Analysis ===
        if comparison == None:
            cpp_pre = np.mean(nop300_pre[:, [c1, c2], :], axis=1)  # (samples, trials)
            cpp_pre = np.mean(cpp_pre, axis=1)  # (samples,)

            cpp_post = np.mean(nop300_post[:, [c1, c2], :], axis=1)
            cpp_post = np.mean(cpp_post, axis=1)

            stdpre = np.std(nop300_pre[:, [c1, c2], :], axis=1) / np.sqrt(
                nop300_pre.shape[2]
            )
            stdpre = np.mean(stdpre, axis=1)

            stdpost = np.std(nop300_post[:, [c1, c2], :], axis=1) / np.sqrt(
                nop300_post.shape[2]
            )
            stdpost = np.mean(stdpost, axis=1)

            plt.figure()
            plt.plot(
                time_ms,
                gaussian_filter1d(cpp_pre, sigma=2),
                label="CPP Pre",
                color="blue",
            )
            plt.plot(
                time_ms,
                gaussian_filter1d(cpp_post, sigma=2),
                label="CPP Post",
                color="red",
            )
            plt.fill_between(
                time_ms, cpp_pre - stdpre, cpp_pre + stdpre, color="blue", alpha=0.2
            )
            plt.fill_between(
                time_ms, cpp_post - stdpost, cpp_post + stdpost, color="red", alpha=0.2
            )
            plt.title("CPP Analysis (CP1 + CP2)")
            plt.xlabel("Time (ms)")
            plt.ylabel("Amplitude (µV)")
            plt.legend()
            plt.grid(True)

            plt.show()

        elif comparison == 1:
            cpp_pre = np.mean(nop300_pre[:, [c1, c2], :], axis=1)  # (samples, trials)
            cpp_pre = np.mean(cpp_pre, axis=1)  # (samples,)

            cpp_post = np.mean(nop300_post[:, [c1, c2], :], axis=1)
            cpp_post = np.mean(cpp_post, axis=1)

            cpp_online = np.mean(nop300_online[:, [c1, c2], :], axis=1)
            cpp_online = np.mean(cpp_online, axis=1)

            stdpre = np.std(nop300_pre[:, [c1, c2], :], axis=1) / np.sqrt(
                nop300_pre.shape[2]
            )
            stdpre = np.mean(stdpre, axis=1)

            stdpost = np.std(nop300_post[:, [c1, c2], :], axis=1) / np.sqrt(
                nop300_post.shape[2]
            )
            stdpost = np.mean(stdpost, axis=1)

            std_online = np.std(nop300_online[:, [c1, c2], :], axis=1) / np.sqrt(
                nop300_online.shape[2]
            )
            std_online = np.mean(std_online, axis=1)

            plt.figure()
            plt.plot(
                time_ms,
                gaussian_filter1d(cpp_pre, sigma=2),
                label="CPP Pre Offline",
                color="blue",
            )
            plt.plot(
                time_ms,
                gaussian_filter1d(cpp_post, sigma=2),
                label="CPP Post Offline",
                color="red",
            )
            plt.plot(
                time_ms,
                gaussian_filter1d(cpp_online, sigma=2),
                label="CPP Online",
                color="green",
            )
            plt.fill_between(
                time_ms, cpp_pre - stdpre, cpp_pre + stdpre, color="blue", alpha=0.2
            )
            plt.fill_between(
                time_ms, cpp_post - stdpost, cpp_post + stdpost, color="red", alpha=0.2
            )
            plt.fill_between(
                time_ms,
                cpp_online - std_online,
                cpp_online + std_online,
                color="green",
                alpha=0.2,
            )
            plt.title("CPP Analysis Comparison (CP1 + CP2)")
            plt.xlabel("Time (ms)")
            plt.ylabel("Amplitude (µV)")
            plt.legend()
            plt.grid(True)

            plt.show()

    elif c == 2:  # === P300 Target Analysis ===
        if comparison == None:
            pz_pre_target = np.mean(p300_pre[:, pz, :], axis=1)
            pz_post_target = np.mean(p300_post[:, pz, :], axis=1)

            stdpre = np.std(p300_pre[:, pz, :], axis=1) / np.sqrt(p300_pre.shape[2])
            stdpost = np.std(p300_post[:, pz, :], axis=1) / np.sqrt(p300_post.shape[2])

            plt.figure()
            plt.plot(
                time_ms,
                gaussian_filter1d(pz_pre_target, sigma=2),
                label="P300 Pre",
                color="blue",
            )
            plt.plot(
                time_ms,
                gaussian_filter1d(pz_post_target, sigma=2),
                label="P300 Post",
                color="red",
            )
            plt.fill_between(
                time_ms,
                pz_pre_target - stdpre,
                pz_pre_target + stdpre,
                color="blue",
                alpha=0.2,
            )
            plt.fill_between(
                time_ms,
                pz_post_target - stdpost,
                pz_post_target + stdpost,
                color="red",
                alpha=0.2,
            )
            plt.title("P300 Analysis at Pz")
            plt.xlabel("Time (ms)")
            plt.ylabel("Amplitude (µV)")
            plt.legend()
            plt.grid(True)
            plt.show()

            m_pre = p3_metrics(time_ms, pz_pre_target)  # blue
            m_post = p3_metrics(time_ms, pz_post_target)  # red
            print("P300 pre metrics :", m_pre)
            print("P300 post metrics :", m_post)

        elif comparison == 1:
            pz_pre_target = np.mean(p300_pre[:, pz, :], axis=1)
            pz_post_target = np.mean(p300_post[:, pz, :], axis=1)
            pz_online_target = np.mean(p300_online[:, pz, :], axis=1)

            stdpre = np.std(p300_pre[:, pz, :], axis=1) / np.sqrt(p300_pre.shape[2])
            stdpost = np.std(p300_post[:, pz, :], axis=1) / np.sqrt(p300_post.shape[2])
            stdonline = np.std(p300_online[:, pz, :], axis=1) / np.sqrt(
                p300_online.shape[2]
            )

            plt.figure()
            plt.plot(
                time_ms,
                gaussian_filter1d(pz_pre_target, sigma=2),
                label="P300 Pre Offline",
                color="blue",
            )
            plt.plot(
                time_ms,
                gaussian_filter1d(pz_post_target, sigma=2),
                label="P300 Post Offline",
                color="red",
            )
            plt.plot(
                time_ms,
                gaussian_filter1d(pz_online_target, sigma=2),
                label="P300 Online",
                color="green",
            )
            plt.fill_between(
                time_ms,
                pz_pre_target - stdpre,
                pz_pre_target + stdpre,
                color="blue",
                alpha=0.2,
            )
            plt.fill_between(
                time_ms,
                pz_post_target - stdpost,
                pz_post_target + stdpost,
                color="red",
                alpha=0.2,
            )
            plt.fill_between(
                time_ms,
                pz_online_target - stdonline,
                pz_online_target + stdonline,
                color="green",
                alpha=0.2,
            )
            plt.title("P300 Analysis Comparison at Pz")
            plt.xlabel("Time (ms)")
            plt.ylabel("Amplitude (µV)")
            plt.legend()
            plt.grid(True)
            plt.show()

            m_pre = p3_metrics(time_ms, pz_pre_target)  # blue
            m_post = p3_metrics(time_ms, pz_post_target)  # red
            m_on = p3_metrics(time_ms, pz_online_target)  # green
            print("\n=== P300 Metrics ===")
            print(f"Pre metrics : {m_pre}")
            print(f"Post metrics : {m_post}")
            print(f"Online metrics : {m_on}")

            ptp_pre = n2_p3_ptp(time_ms, pz_pre_target)
            ptp_post = n2_p3_ptp(time_ms, pz_post_target)
            ptp_on = n2_p3_ptp(time_ms, pz_online_target)

            t_ms = np.arange(p300_online.shape[0]) * 1000 / fs

            # ensure trials×time
            X = p300_online[
                :, pz, :
            ].T  # was (n_times, n_trials) -> now (n_trials, n_times)

            print(f"X shape (trials×time): {X.shape}")

            # compute jitter
            jitter_sd, per_trial_lat = single_trial_latency_jitter(t_ms, X)

            print(
                f"Online jitter SD: {jitter_sd:.2f} ms; n={len(per_trial_lat)} trials"
            )

            # ---- Peak-to-peak amplitude (N2–P3) ----
            print("\n=== N2–P3 Peak-to-Peak Amplitude (µV) ===")
            print(f"Offline PRE:   {ptp_pre:.3f} µV")
            print(f"Offline POST:  {ptp_post:.3f} µV")
            print(f"Online (BCI):  {ptp_on:.3f} µV")

            # ---- Latency jitter ----
            print("\n=== Single-Trial Latency Jitter ===")
            print(f"Online jitter SD: {jitter_sd:.2f} ms")
            print(f"Number of trials: {len(per_trial_lat)}")

            # Optional: Show summary statistics of per-trial latencies
            print(f"Mean latency:  {np.mean(per_trial_lat):.2f} ms")
            print(f"Min latency:   {np.min(per_trial_lat):.2f} ms")
            print(f"Max latency:   {np.max(per_trial_lat):.2f} ms")

            plt.hist(per_trial_lat, bins=10, edgecolor="black")
            plt.xlabel("P3 Latency (ms)")
            plt.ylabel("Trial Count")
            plt.title("Online BCI: Single-Trial P3 Latency Distribution")
            plt.show()

            # Use with your online trials at Pz (trials×time)
            X = p300_online[:, pz, :].T
            Xw, shifts_ms = woody_align_window(
                t_ms, X, window=(300, 600), max_shift_ms=80
            )
            erp_online_aligned = Xw.mean(axis=0)

            # Recompute metrics on the aligned ERP
            m_on_aligned = p3_metrics(t_ms, erp_online_aligned)
            print("Aligned online:", m_on_aligned)
            print(
                f"Alignment SD: {np.std(shifts_ms):.1f} ms (should be close to jitter SD)"
            )
            # per_trial_lat already computed (ms)
            trial_idx = np.arange(len(per_trial_lat)) + 1
            rho, p = spearmanr(trial_idx, per_trial_lat)
            print(f"Latency drift: Spearman r={rho:.2f}, p={p:.3g}")

            plt.plot(trial_idx, per_trial_lat, ".-")
            plt.xlabel("Trial #")
            plt.ylabel("P3 latency (ms)")
            plt.title("P3 latency across trials (online)")
            plt.show()

            X = p300_online[:, pz, :].T
            Xw, shifts_ms = woody_align_window(
                t_ms, X, window=(300, 600), max_shift_ms=80
            )
            erp_online_aligned = Xw.mean(axis=0)

            erp_on_raw = X.mean(axis=0)
            erp_on_aln = Xw.mean(axis=0)

            m_on_raw = p3_metrics(
                t_ms, erp_on_raw, baseline_window=(-400, -250), snr_method="mad"
            )
            m_on_aln = p3_metrics(
                t_ms, erp_on_aln, baseline_window=(-400, -250), snr_method="mad"
            )
            print("Online (raw):    ", m_on_raw)
            print("Online (aligned):", m_on_aln)
            print(f"Alignment SD (ms): {np.std(shifts_ms):.1f}")

            from sklearn.cluster import KMeans

            lat = per_trial_lat.reshape(-1, 1)
            km = KMeans(n_clusters=2, n_init=20, random_state=0).fit(lat)
            labels = km.labels_
            print("Cluster means (ms):", sorted(km.cluster_centers_.ravel()))

            # Compare ERPs of the two latency clusters
            erp_fast = X[labels == 0].mean(axis=0)
            erp_slow = X[labels == 1].mean(axis=0)

            print("Fast cluster:", p3_metrics(t_ms, erp_fast))
            print("Slow cluster:", p3_metrics(t_ms, erp_slow))

            sign_ref = np.sign(np.max(p300_post) - abs(np.min(p300_post))) or 1.0
            erp_online_pz_fixed = sign_ref * X.mean(axis=0)  # X.mean: online ERP at Pz
            print(p3_metrics(t_ms, erp_online_pz_fixed, snr_method="mad"))

            erp_online_pz = X.mean(axis=0)  # grand-average (online) at Pz

            def sem(a, axis=0):
                a = np.asarray(a)
                return np.std(a, axis=axis, ddof=1) / np.sqrt(a.shape[axis])

            # ====== Figure 1: Pre vs Post vs Online ======
            fig1, ax1 = plt.subplots(figsize=(8, 4))

            # Online SEM from trials
            online_sem = sem(X, axis=0)

            ax1.plot(t_ms, pz_pre_target, label="Pre (offline)")
            ax1.plot(t_ms, pz_post_target, label="Post (offline)")
            ax1.plot(t_ms, erp_online_pz, label="Online day1 (BCI)")

            ax1.fill_between(
                t_ms,
                pz_pre_target - stdpre,
                pz_pre_target + stdpre,
                alpha=0.2,
                color="blue",
            )

            ax1.fill_between(
                t_ms,
                pz_post_target - stdpost,
                pz_post_target + stdpost,
                alpha=0.2,
                color="red",
            )

            ax1.fill_between(
                t_ms,
                erp_online_pz - online_sem,
                erp_online_pz + online_sem,
                alpha=0.2,
                color="green",
            )

            ax1.set_title("ERP at Pz: Pre vs Post vs Online day 1")
            ax1.set_xlabel("Time (ms)")
            ax1.set_ylabel("Amplitude (µV)")
            ax1.axvline(0, linestyle="--", linewidth=1)
            ax1.legend(loc="best")
            ax1.grid(True, alpha=0.3)
            plt.tight_layout()
            plt.show()

            # ====== Figure 2: Online FAST vs SLOW clusters at Pz ======
            # Determine which label is earlier (FAST) based on cluster mean latencies you computed
            # If you don’t have per_trial_lat here, infer FAST/SLOW by ERP center-of-mass or keep 0/1 as-is.
            # For simplicity, we’ll compute means by label 0 and 1 and you can swap if needed.

            X0 = X[labels == 0]  # cluster 0 trials
            X1 = X[labels == 1]  # cluster 1 trials

            erp0 = X0.mean(axis=0)
            erp1 = X1.mean(axis=0)
            sem0 = sem(X0, axis=0) if len(X0) > 1 else np.zeros_like(erp0)
            sem1 = sem(X1, axis=0) if len(X1) > 1 else np.zeros_like(erp1)

            fig2, ax2 = plt.subplots(figsize=(8, 4))
            ax2.plot(t_ms, erp0, label=f"Cluster 0 (n={len(X0)})")
            ax2.fill_between(t_ms, erp0 - sem0, erp0 + sem0, alpha=0.2)

            ax2.plot(t_ms, erp1, label=f"Cluster 1 (n={len(X1)})")
            ax2.fill_between(t_ms, erp1 - sem1, erp1 + sem1, alpha=0.2)

            # Highlight canonical P3 windows you used
            ax2.axvspan(250, 400, alpha=0.1, label="FAST window (250–400 ms)")
            ax2.axvspan(420, 620, alpha=0.1, label="SLOW window (420–620 ms)")

            ax2.set_title("Online day 1 ERP at Pz: FAST vs SLOW clusters")
            ax2.set_xlabel("Time (ms)")
            ax2.set_ylabel("Amplitude (µV)")
            ax2.axvline(0, linestyle="--", linewidth=1)
            ax2.legend(loc="best")
            ax2.grid(True, alpha=0.3)
            plt.tight_layout()
            plt.show()

            # ===== Clustered ERPs by condition (PRE, POST, ONLINE) =====
            # Build trials×time matrices at Pz
            X_pre = trials_time_from_tc_tr(p300_pre, pz)
            X_post = trials_time_from_tc_tr(p300_post, pz)
            X_on = trials_time_from_tc_tr(p300_online, pz)

            print("\n--- OFFLINE PRE: latency clustering at Pz ---")
            res_pre = cluster_latency_and_plot(
                time_ms, X_pre, title="Offline PRE ERP at Pz: FAST vs SLOW clusters"
            )
            print(
                f"PRE jitter SD: {res_pre['jitter_sd']:.2f} ms | centers: {np.array(res_pre['centers_ms'])}"
            )
            print("  Cluster 0 metrics:", p3_metrics(time_ms, res_pre["erp0"]))
            print("  Cluster 1 metrics:", p3_metrics(time_ms, res_pre["erp1"]))

            print("\n--- OFFLINE POST: latency clustering at Pz ---")
            res_post = cluster_latency_and_plot(
                time_ms, X_post, title="Offline POST ERP at Pz: FAST vs SLOW clusters"
            )
            print(
                f"POST jitter SD: {res_post['jitter_sd']:.2f} ms | centers: {np.array(res_post['centers_ms'])}"
            )
            print("  Cluster 0 metrics:", p3_metrics(time_ms, res_post["erp0"]))
            print("  Cluster 1 metrics:", p3_metrics(time_ms, res_post["erp1"]))

            print("\n--- ONLINE: latency clustering at Pz ---")
            res_on = cluster_latency_and_plot(
                time_ms, X_on, title="Online ERP at Pz: FAST vs SLOW clusters"
            )
            print(
                f"ONLINE jitter SD: {res_on['jitter_sd']:.2f} ms | centers: {np.array(res_on['centers_ms'])}"
            )
            print("  Cluster 0 metrics:", p3_metrics(time_ms, res_on["erp0"]))
            print("  Cluster 1 metrics:", p3_metrics(time_ms, res_on["erp1"]))

    elif c == 3:  # === Target vs Non-Target (Pre and Post) ===
        if comparison == None:
            for label, targ, nontarg in zip(
                ["Pre", "Post"], [p300_pre, p300_post], [nop300_pre, nop300_post]
            ):
                target = np.mean(targ[:, pz, :], axis=1)
                nontarget = np.mean(nontarg[:, pz, :], axis=1)

                std_target = np.std(targ[:, pz, :], axis=1) / np.sqrt(targ.shape[2])
                std_nontarget = np.std(nontarg[:, pz, :], axis=1) / np.sqrt(
                    nontarg.shape[2]
                )

                plt.figure()
                plt.plot(
                    time_ms,
                    gaussian_filter1d(target, sigma=2),
                    label=f"{label} Target",
                    color="blue",
                )
                plt.plot(
                    time_ms,
                    gaussian_filter1d(nontarget, sigma=2),
                    label=f"{label} Non-Target",
                    color="red",
                )
                plt.fill_between(
                    time_ms,
                    target - std_target,
                    target + std_target,
                    color="blue",
                    alpha=0.2,
                )
                plt.fill_between(
                    time_ms,
                    nontarget - std_nontarget,
                    nontarget + std_nontarget,
                    color="red",
                    alpha=0.2,
                )
                plt.title(f"{label} P300: Target vs Non-Target")
                plt.xlabel("Time (ms)")
                plt.ylabel("Amplitude (µV)")
                plt.legend()
                plt.grid(True)

                plt.show()

        elif comparison == 1:
            for label, targ, nontarg in zip(
                ["Pre", "Post", "Online"],
                [p300_pre, p300_post, p300_online],
                [nop300_pre, nop300_post, nop300_online],
            ):
                target = np.mean(targ[:, pz, :], axis=1)
                nontarget = np.mean(nontarg[:, pz, :], axis=1)

                std_target = np.std(targ[:, pz, :], axis=1) / np.sqrt(targ.shape[2])
                std_nontarget = np.std(nontarg[:, pz, :], axis=1) / np.sqrt(
                    nontarg.shape[2]
                )

                plt.figure()
                plt.plot(
                    time_ms,
                    gaussian_filter1d(target, sigma=2),
                    label=f"{label} Target",
                    color="blue",
                )
                plt.plot(
                    time_ms,
                    gaussian_filter1d(nontarget, sigma=2),
                    label=f"{label} Non-Target",
                    color="red",
                )
                plt.fill_between(
                    time_ms,
                    target - std_target,
                    target + std_target,
                    color="blue",
                    alpha=0.2,
                )
                plt.fill_between(
                    time_ms,
                    nontarget - std_nontarget,
                    nontarget + std_nontarget,
                    color="red",
                    alpha=0.2,
                )
                plt.title(f"{label} P300: Target vs Non-Target")
                plt.xlabel("Time (ms)")
                plt.ylabel("Amplitude (µV)")
                plt.legend()
                plt.grid(True)

                plt.show()
