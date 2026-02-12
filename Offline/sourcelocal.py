"""
© 2026 Alexandra Mikhael. All Rights Reserved.
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import mne
from mne.minimum_norm import make_inverse_operator, apply_inverse
from scipy import stats
import warnings

warnings.filterwarnings("ignore")

import scipy.io as sio


def load_channel_locations(mat_file):
    """
    Load channel locations from .mat file

    Parameters:
    -----------
    mat_file : str
        Path to ch32Locations.mat

    Returns:
    --------
    ch_pos : dict
        Dictionary mapping channel names to (x, y, z) positions
    ch_names : list
        List of channel names in order
    """

    print(f"\nLoading channel locations from: {mat_file}")

    # Load .mat file
    mat_data = sio.loadmat(mat_file)

    # Print structure to see what's inside
    print("Keys in .mat file:", list(mat_data.keys()))

    # Get the channel data
    chanlocs = mat_data["ch32Locations"]

    print(f"Channel data shape: {chanlocs.shape}")
    print(f"Channel data dtype: {chanlocs.dtype}")

    # Extract channel names and positions
    ch_names = []
    ch_pos = {}

    # Handle MATLAB struct array
    if chanlocs.dtype.names:
        print(f"Fields in struct: {chanlocs.dtype.names}")

        # Iterate through each channel
        for i in range(len(chanlocs[0])):  # Note: MATLAB structs come as (1, n) array
            ch = chanlocs[0][i]

            # Extract label - handle different possible formats
            if "labels" in chanlocs.dtype.names:
                label = ch["labels"]
            elif "label" in chanlocs.dtype.names:
                label = ch["label"]
            else:
                label = f"Ch{i+1}"

            # Clean up label
            if isinstance(label, np.ndarray):
                if label.dtype.kind in ["U", "S", "O"]:  # String types
                    # Handle nested arrays
                    while isinstance(label, np.ndarray) and label.size == 1:
                        label = label[0]
                    label = str(label)
                else:
                    label = str(label[0]) if len(label) > 0 else f"Ch{i+1}"

            # Remove quotes, brackets, etc.
            label = str(label).strip()
            label = (
                label.replace("[", "")
                .replace("]", "")
                .replace("'", "")
                .replace('"', "")
            )

            # Extract position
            pos = None

            # Try X, Y, Z fields
            if (
                "X" in chanlocs.dtype.names
                and "Y" in chanlocs.dtype.names
                and "Z" in chanlocs.dtype.names
            ):
                x = (
                    float(ch["X"][0])
                    if isinstance(ch["X"], np.ndarray)
                    else float(ch["X"])
                )
                y = (
                    float(ch["Y"][0])
                    if isinstance(ch["Y"], np.ndarray)
                    else float(ch["Y"])
                )
                z = (
                    float(ch["Z"][0])
                    if isinstance(ch["Z"], np.ndarray)
                    else float(ch["Z"])
                )
                pos = np.array([x, y, z])

            # Try theta, radius, phi (spherical coordinates)
            elif "theta" in chanlocs.dtype.names and "radius" in chanlocs.dtype.names:
                theta = (
                    float(ch["theta"][0])
                    if isinstance(ch["theta"], np.ndarray)
                    else float(ch["theta"])
                )
                radius = (
                    float(ch["radius"][0])
                    if isinstance(ch["radius"], np.ndarray)
                    else float(ch["radius"])
                )

                # Check for phi (3D) or assume 2D
                if "phi" in chanlocs.dtype.names:
                    phi = (
                        float(ch["phi"][0])
                        if isinstance(ch["phi"], np.ndarray)
                        else float(ch["phi"])
                    )
                else:
                    phi = 0.0

                # Convert spherical to Cartesian
                # EEGLAB uses: theta (azimuth), phi (elevation), radius
                theta_rad = np.deg2rad(theta)
                phi_rad = np.deg2rad(phi)

                x = radius * np.cos(theta_rad) * np.cos(phi_rad)
                y = radius * np.sin(theta_rad) * np.cos(phi_rad)
                z = radius * np.sin(phi_rad)

                pos = np.array([x, y, z])

            if pos is not None and label:
                ch_names.append(label)
                ch_pos[label] = pos

    else:
        # Simple numeric array
        print("⚠️ Expected struct array but got numeric array")
        if chanlocs.shape[1] >= 3:
            for i in range(chanlocs.shape[0]):
                label = f"Ch{i+1}"
                ch_names.append(label)
                ch_pos[label] = chanlocs[i, :3]

    print(f"\n✓ Loaded {len(ch_names)} channels")
    if len(ch_names) > 0:
        print(f"Sample channels: {ch_names[:5]}")
        print(f"Sample position ({ch_names[0]}): {ch_pos[ch_names[0]]}")
    else:
        print("❌ No channels loaded!")

    return ch_pos, ch_names


def setup_source_space_and_forward(labels, sfreq=512, ch_pos_file=None):
    """
    Setup source space and forward model using fsaverage template

    Parameters:
    -----------
    labels : list
        Channel names
    sfreq : float
        Sampling frequency
    ch_pos_file : str
        Path to .mat file with channel positions (optional)
    """

    print("\n" + "=" * 70)
    print("SETTING UP SOURCE SPACE AND FORWARD MODEL")
    print("=" * 70)

    # Convert labels to list
    if isinstance(labels, np.ndarray):
        labels = labels.tolist()
    elif not isinstance(labels, list):
        labels = list(labels)

    print(f"Number of channels: {len(labels)}")
    print(f"Channels: {labels}")

    # Hardcoded correct paths
    subjects_dir = "/home/alexandra-admin/mne_data/MNE-fsaverage-data"
    subject = "fsaverage"

    print(f"✓ Subjects directory: {subjects_dir}")
    print(f"✓ Subject: {subject}")

    # Create info
    info = mne.create_info(ch_names=labels, sfreq=sfreq, ch_types=["eeg"] * len(labels))

    # Set channel positions
    print("\nSetting channel locations...")

    if ch_pos_file is not None and os.path.exists(ch_pos_file):
        # Use custom channel locations from .mat file
        print(f"Using custom positions from: {ch_pos_file}")

        ch_pos_dict, ch_names_from_file = load_channel_locations(ch_pos_file)

        # Match channel names
        matched_pos = {}
        for label in labels:
            # Try exact match
            if label in ch_pos_dict:
                matched_pos[label] = ch_pos_dict[label]
            # Try case-insensitive match
            else:
                for ch_name, pos in ch_pos_dict.items():
                    if ch_name.upper() == label.upper():
                        matched_pos[label] = pos
                        break

        if len(matched_pos) == 0:
            print("⚠️ No channels matched between data and position file")
            print(f"   Your labels: {labels[:5]}...")
            print(f"   Position file channels: {list(ch_pos_dict.keys())[:5]}...")
            raise ValueError("Cannot match channel names!")

        print(f"✓ Matched {len(matched_pos)}/{len(labels)} channels")

        # Keep only channels with positions
        valid_labels = [ch for ch in labels if ch in matched_pos]
        valid_indices = [i for i, ch in enumerate(labels) if ch in matched_pos]

        # Create montage from matched positions
        montage = mne.channels.make_dig_montage(ch_pos=matched_pos, coord_frame="head")

        # Recreate info with only valid channels
        info = mne.create_info(
            ch_names=valid_labels, sfreq=sfreq, ch_types=["eeg"] * len(valid_labels)
        )

        info.set_montage(montage)

        print(f"✓ Set custom positions for {len(valid_labels)} channels")

        labels = valid_labels

    else:
        # Fallback to standard montage
        print("Using standard_1020 montage...")

        montage = mne.channels.make_standard_montage("standard_1020")

        # Standardize names and match
        labels_std = standardize_channel_names(labels)

        montage_ch_names = set(montage.ch_names)
        valid_labels = [ch for ch in labels_std if ch in montage_ch_names]
        valid_indices = [i for i, ch in enumerate(labels_std) if ch in montage_ch_names]

        if len(valid_labels) == 0:
            raise ValueError(
                "No channels match standard montage and no custom position file provided!"
            )

        print(
            f"✓ Matched {len(valid_labels)}/{len(labels)} channels to standard montage"
        )

        # Recreate info
        info = mne.create_info(
            ch_names=valid_labels, sfreq=sfreq, ch_types=["eeg"] * len(valid_labels)
        )

        info.set_montage(montage, on_missing="ignore")
        labels = valid_labels

    # Verify all channels have locations
    for ch in info["chs"]:
        if np.allclose(ch["loc"][:3], 0):
            raise RuntimeError(f"Channel {ch['ch_name']} has no location!")

    print(f"✓ All {len(labels)} channels have valid 3D locations")

    # Setup source space
    print("\nSetting up source space...")
    src = mne.setup_source_space(
        subject,
        spacing="oct6",
        subjects_dir=subjects_dir,
        add_dist=False,
        verbose=False,
    )
    print(f"✓ Source space: {len(src[0]['vertno'])} + {len(src[1]['vertno'])} vertices")

    # Setup forward model
    print("\nSetting up forward model...")

    bem_path = os.path.join(
        subjects_dir, subject, "bem", f"{subject}-5120-5120-5120-bem-sol.fif"
    )

    if os.path.exists(bem_path):
        bem = mne.read_bem_solution(bem_path, verbose=False)
        bem_method = "3-layer BEM (accurate)"
        print(f"✓ Found BEM solution")
    else:
        bem = mne.make_sphere_model(
            r0="auto", head_radius="auto", info=info, verbose=False
        )
        bem_method = "Sphere model (approximation)"
        print("⚠️ BEM not found, using sphere model")

    print(f"✓ Using: {bem_method}")

    # Compute forward solution
    print("\nComputing forward solution...")

    trans = "fsaverage"

    fwd = mne.make_forward_solution(
        info,
        trans=trans,
        src=src,
        bem=bem,
        eeg=True,
        mindist=5.0,
        n_jobs=1,
        verbose=False,
    )

    print(f"✓ Forward solution: {fwd['nsource']} sources, {fwd['nchan']} channels")
    print(f"✓ Method: {bem_method}")
    print("=" * 70)

    return src, fwd, subjects_dir, info, valid_indices


def setup_source_space_and_forward(labels, sfreq=512, ch_pos_file=None):
    """
    Setup source space and forward model using fsaverage template

    Parameters:
    -----------
    labels : list
        Channel names
    sfreq : float
        Sampling frequency
    ch_pos_file : str
        Path to .mat file with channel positions (optional)
    """

    print("\n" + "=" * 70)
    print("SETTING UP SOURCE SPACE AND FORWARD MODEL")
    print("=" * 70)

    # Convert labels to list
    if isinstance(labels, np.ndarray):
        labels = labels.tolist()
    elif not isinstance(labels, list):
        labels = list(labels)

    print(f"Number of channels: {len(labels)}")
    print(f"Channels: {labels}")

    # Hardcoded correct paths
    subjects_dir = "/home/alexandra-admin/mne_data/MNE-fsaverage-data"
    subject = "fsaverage"

    print(f"✓ Subjects directory: {subjects_dir}")
    print(f"✓ Subject: {subject}")

    # Create info
    info = mne.create_info(ch_names=labels, sfreq=sfreq, ch_types=["eeg"] * len(labels))

    # Set channel positions
    print("\nSetting channel locations...")

    if ch_pos_file is not None and os.path.exists(ch_pos_file):
        # Use custom channel locations from .mat file
        print(f"Using custom positions from: {ch_pos_file}")

        ch_pos_dict, ch_names_from_file = load_channel_locations(ch_pos_file)

        # Match channel names
        matched_pos = {}
        for label in labels:
            # Try exact match
            if label in ch_pos_dict:
                matched_pos[label] = ch_pos_dict[label]
            # Try case-insensitive match
            else:
                for ch_name, pos in ch_pos_dict.items():
                    if ch_name.upper() == label.upper():
                        matched_pos[label] = pos
                        break

        if len(matched_pos) == 0:
            print("⚠️ No channels matched between data and position file")
            print(f"   Your labels: {labels[:5]}...")
            print(f"   Position file channels: {list(ch_pos_dict.keys())[:5]}...")
            raise ValueError("Cannot match channel names!")

        print(f"✓ Matched {len(matched_pos)}/{len(labels)} channels")

        # Keep only channels with positions
        valid_labels = [ch for ch in labels if ch in matched_pos]
        valid_indices = [i for i, ch in enumerate(labels) if ch in matched_pos]

        # Create montage from matched positions
        montage = mne.channels.make_dig_montage(ch_pos=matched_pos, coord_frame="head")

        # Recreate info with only valid channels
        info = mne.create_info(
            ch_names=valid_labels, sfreq=sfreq, ch_types=["eeg"] * len(valid_labels)
        )

        info.set_montage(montage)

        print(f"✓ Set custom positions for {len(valid_labels)} channels")

        labels = valid_labels

    else:
        # Fallback to standard montage
        print("Using standard_1020 montage...")

        montage = mne.channels.make_standard_montage("standard_1020")

        # Standardize names and match
        labels_std = standardize_channel_names(labels)

        montage_ch_names = set(montage.ch_names)
        valid_labels = [ch for ch in labels_std if ch in montage_ch_names]
        valid_indices = [i for i, ch in enumerate(labels_std) if ch in montage_ch_names]

        if len(valid_labels) == 0:
            raise ValueError(
                "No channels match standard montage and no custom position file provided!"
            )

        print(
            f"✓ Matched {len(valid_labels)}/{len(labels)} channels to standard montage"
        )

        # Recreate info
        info = mne.create_info(
            ch_names=valid_labels, sfreq=sfreq, ch_types=["eeg"] * len(valid_labels)
        )

        info.set_montage(montage, on_missing="ignore")
        labels = valid_labels

    # Verify all channels have locations
    for ch in info["chs"]:
        if np.allclose(ch["loc"][:3], 0):
            raise RuntimeError(f"Channel {ch['ch_name']} has no location!")

    print(f"✓ All {len(labels)} channels have valid 3D locations")

    # Setup source space
    print("\nSetting up source space...")
    src = mne.setup_source_space(
        subject,
        spacing="oct6",
        subjects_dir=subjects_dir,
        add_dist=False,
        verbose=False,
    )
    print(f"✓ Source space: {len(src[0]['vertno'])} + {len(src[1]['vertno'])} vertices")

    # Setup forward model
    print("\nSetting up forward model...")

    bem_path = os.path.join(
        subjects_dir, subject, "bem", f"{subject}-5120-5120-5120-bem-sol.fif"
    )

    if os.path.exists(bem_path):
        bem = mne.read_bem_solution(bem_path, verbose=False)
        bem_method = "3-layer BEM (accurate)"
        print(f"✓ Found BEM solution")
    else:
        bem = mne.make_sphere_model(
            r0="auto", head_radius="auto", info=info, verbose=False
        )
        bem_method = "Sphere model (approximation)"
        print("⚠️ BEM not found, using sphere model")

    print(f"✓ Using: {bem_method}")

    # Compute forward solution
    print("\nComputing forward solution...")

    trans = "fsaverage"

    fwd = mne.make_forward_solution(
        info,
        trans=trans,
        src=src,
        bem=bem,
        eeg=True,
        mindist=5.0,
        n_jobs=1,
        verbose=False,
    )
    # Ensure valid_indices is a proper list of integers
    if not isinstance(valid_indices, list):
        valid_indices = list(valid_indices)

    # If all channels matched, create full range
    if len(labels) == info["nchan"]:
        valid_indices = list(range(len(labels)))
        print(f"All {len(labels)} channels used, indices: 0-{len(labels)-1}")

    # Verify all elements are integers
    valid_indices = [int(i) for i in valid_indices]

    print(f"✓ Method: {bem_method}")
    print(f"✓ Valid indices (final): {valid_indices}")
    print("=" * 70)

    return src, fwd, subjects_dir, info, valid_indices


def standardize_channel_names(labels):
    """Convert channel names to standard_1020 format"""
    standardized = []
    for ch in labels:
        ch_str = str(ch).strip()
        if ch_str.upper() in ["FPZ", "FP1", "FP2"]:
            standardized.append("Fp" + ch_str[2:])
        elif ch_str.upper() in ["FZ", "CZ", "PZ", "OZ", "FCZ", "CPZ"]:
            standardized.append(ch_str[0].upper() + ch_str[1:].lower())
        elif ch_str.upper() == "POZ":
            standardized.append("POz")
        else:
            if len(ch_str) >= 2:
                standardized.append(ch_str[0].upper() + ch_str[1:])
            else:
                standardized.append(ch_str.upper())
    return standardized


def compute_sources(
    p300_data,
    labels,
    time_ms,
    fwd,
    info,
    baseline_window=(-200, 0),
    component_name="P300",
):
    """
    Compute source reconstruction for ERP data

    Parameters:
    -----------
    p300_data : ndarray
        Shape (time, channels) - should already be filtered to valid channels
    labels : list
        Channel names (should match info['ch_names'])
    time_ms : ndarray
        Time vector
    fwd : Forward
        Forward solution
    info : Info
        MNE info structure
    """

    print(f"\n{'='*70}")
    print(f"COMPUTING SOURCES: {component_name}")
    print(f"{'='*70}")

    # Ensure 2D array
    if p300_data.ndim == 3:
        print(f"Averaging across {p300_data.shape[2]} trials...")
        data_avg = p300_data.mean(axis=2)  # (time, channels)
    elif p300_data.ndim == 1:
        # Single channel? Reshape
        data_avg = p300_data.reshape(-1, 1)
    else:
        data_avg = p300_data

    print(f"Data shape: {data_avg.shape}")
    print(f"Info expects: (time={len(time_ms)}, channels={info['nchan']})")

    # Verify dimensions match
    if data_avg.shape[0] != len(time_ms):
        print(
            f"⚠️ Time dimension mismatch: data has {data_avg.shape[0]}, expected {len(time_ms)}"
        )
        print(f"   This is OK for averaged component data (single time point)")

    if data_avg.shape[1] != info["nchan"]:
        raise ValueError(
            f"Channel mismatch: data has {data_avg.shape[1]} channels "
            f"but info expects {info['nchan']} channels.\n"
            f"Data channels: {labels[:5] if isinstance(labels, list) else 'N/A'}...\n"
            f"Info channels: {info['ch_names'][:5]}..."
        )

    # Create Evoked object
    # MNE expects (channels, time) in Volts
    evoked = mne.EvokedArray(
        data_avg.T * 1e-6,  # Transpose to (channels, time), convert µV to V
        info,
        tmin=time_ms[0] / 1000,  # Convert to seconds
        comment=component_name,
        verbose=False,
    )

    print(f"✓ Created Evoked object: {evoked.data.shape} (channels, time)")

    evoked.set_eeg_reference(projection=True, verbose=False)
    print(f"✓ Set EEG average reference")

    # Use identity covariance for single time point data
    print(f"\nUsing identity covariance matrix...")
    noise_cov = mne.make_ad_hoc_cov(info)

    print(f"✓ Noise covariance computed")

    # Compute inverse operator
    print("\nComputing inverse operator...")

    inverse_operator = make_inverse_operator(
        evoked.info, fwd, noise_cov, loose=0.2, depth=0.8, fixed=False, verbose=False
    )

    print(f"✓ Inverse operator created")

    # Apply inverse solution
    print("\nApplying inverse solution (sLORETA)...")

    method = "sLORETA"
    snr = 3.0
    lambda2 = 1.0 / snr**2

    stc = apply_inverse(
        evoked, inverse_operator, lambda2, method=method, pick_ori=None, verbose=False
    )

    print(f"✓ Source reconstruction complete")
    print(f"  Method: {method}")
    print(f"  Sources: {stc.data.shape[0]}")
    print(f"  Time points: {stc.data.shape[1]}")
    print(f"  Peak activation: {np.max(np.abs(stc.data)):.2e} A·m")

    print("=" * 70)

    return stc, evoked


def extract_component_sources(
    p300_data, labels, time_ms, fwd, info, valid_indices, condition_name="Condition"
):
    """
    Extract sources for P3a, P3b, and Slow Wave components
    """

    print("\n" + "=" * 70)
    print(f"EXTRACTING COMPONENT SOURCES: {condition_name}")
    print("=" * 70)

    print(f"Input data shape: {p300_data.shape}")
    print(f"Info channels: {info['nchan']}")

    # Define component windows (in ms)
    components_windows = {"P3a": (250, 350), "P3b": (300, 500), "SW": (400, 700)}

    components = {}

    for comp_name, (tmin, tmax) in components_windows.items():
        print(f"\n--- {comp_name} ({tmin}-{tmax} ms) ---")

        # Extract time window
        time_mask = (time_ms >= tmin) & (time_ms <= tmax)

        if p300_data.ndim == 3:
            # (time, channels, trials)
            comp_data = p300_data[time_mask, :, :].mean(axis=(0, 2))
        else:
            # (time, channels)
            comp_data = p300_data[time_mask, :].mean(axis=0)

        print(f"Averaged data shape: {comp_data.shape}")

        # NO FILTERING - use all channels since 32/32 matched
        # Just verify dimensions
        if comp_data.shape[0] != info["nchan"]:
            raise ValueError(
                f"Channel count mismatch: data has {comp_data.shape[0]}, "
                f"info expects {info['nchan']}"
            )

        # Reshape to (1, channels)
        comp_data_2d = comp_data.reshape(1, -1)

        print(f"Final shape: {comp_data_2d.shape}")

        # Compute sources
        stc, evoked = compute_sources(
            comp_data_2d,
            info["ch_names"],
            np.array([0]),
            fwd,
            info,
            component_name=f"{condition_name}_{comp_name}",
        )

        components[comp_name] = {"stc": stc, "evoked": evoked, "window": (tmin, tmax)}

    return components


def extract_roi_values(stc, subjects_dir, src=None, parcellation="aparc"):
    """
    Extract source values from anatomical ROIs

    Parameters:
    -----------
    stc : SourceEstimate
        Source time course
    subjects_dir : str
        Path to subjects directory
    src : SourceSpaces
        Source space (optional, will be loaded if not provided)
    parcellation : str
        Parcellation name

    Returns:
    --------
    roi_dict : dict
        ROI names and their mean source strength
    """

    print(f"\nExtracting ROI values using {parcellation} parcellation...")

    # Read labels (ROIs) from parcellation
    labels = mne.read_labels_from_annot(
        "fsaverage", parc=parcellation, subjects_dir=subjects_dir, verbose=False
    )

    # Load source space if not provided
    if src is None:
        print("Loading source space...")
        src = mne.read_source_spaces(
            os.path.join(subjects_dir, "fsaverage", "bem", "fsaverage-oct-6-src.fif")
        )

    # ROI name mapping (from paper)
    roi_mapping = {
        "ACC": "caudalanteriorcingulate",
        "CNG": "rostralanteriorcingulate",
        "Cns": "cuneus",
        "FFG": "fusiform",
        "IFG": "parsorbitalis",
        "IPL": "inferiorparietal",
        "MiFG": "rostralmiddlefrontal",
        "MiTG": "middletemporal",
        "MFG": "superiorfrontal",
        "MTG": "inferiortemporal",
        "PCns": "precuneus",
    }

    roi_dict = {}

    for roi_abbr, roi_name in roi_mapping.items():
        # Find matching labels (left and right hemispheres)
        matching_labels = [l for l in labels if roi_name in l.name.lower()]

        if matching_labels:
            # Extract data from this ROI (combine hemispheres)
            roi_data = []
            for label in matching_labels:
                # Get source time course in this label
                label_tc = stc.extract_label_time_course(
                    label, src, mode="mean"  # Pass src here
                )[0]
                roi_data.append(np.abs(label_tc).mean())  # Mean absolute activation

            # Average across hemispheres
            roi_dict[roi_abbr] = np.mean(roi_data) * 1e9  # Convert to nA·m
        else:
            roi_dict[roi_abbr] = 0.0

    print(f"✓ Extracted {len(roi_dict)} ROIs")

    return roi_dict


def plot_sources_on_brain(
    stc,
    subjects_dir,
    title="Source Activity",
    output_file=None,
    views=["lateral", "medial"],
):
    """
    Plot source activity on brain surface
    """

    print(f"\nGenerating brain plots: {title}")

    # Create figure with multiple views
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    # Get brain plot for each hemisphere and view
    for hemi_idx, hemi in enumerate(["lh", "rh"]):
        for view_idx, view in enumerate(views):
            ax = axes[hemi_idx, view_idx]

            # Plot on inflated brain
            # Note: stc.plot() returns a Brain object, not directly plottable
            # We'll use a simpler approach with screenshots

            try:
                brain = stc.plot(
                    subject="fsaverage",
                    subjects_dir=subjects_dir,
                    hemi=hemi,
                    views=view,
                    time_label=title,
                    colormap="hot",
                    clim=dict(kind="percent", lims=[90, 95, 99]),
                    background="white",
                    size=(400, 400),
                    # Remove 'show' parameter - not supported
                )

                # Capture screenshot
                screenshot = brain.screenshot()
                brain.close()

                # Display in matplotlib
                ax.imshow(screenshot)
                ax.axis("off")
                ax.set_title(f"{hemi.upper()} - {view}", fontsize=11)

            except Exception as e:
                print(f"⚠️ Could not plot {hemi}-{view}: {e}")
                # Show blank axis if plotting fails
                ax.axis("off")
                ax.text(
                    0.5,
                    0.5,
                    f"Plot error:\n{hemi}-{view}",
                    ha="center",
                    va="center",
                    transform=ax.transAxes,
                )

    plt.suptitle(title, fontsize=14, fontweight="bold")
    plt.tight_layout()

    if output_file:
        plt.savefig(output_file, dpi=300, bbox_inches="tight")
        print(f"✓ Saved: {output_file}")

    plt.show()
    plt.close()


def plot_roi_bars(roi_dict, title="ROI Activity", output_file=None, color="steelblue"):
    """
    Plot bar chart of ROI activations
    """

    # Sort ROIs by activation
    sorted_rois = sorted(roi_dict.items(), key=lambda x: x[1], reverse=True)
    roi_names = [r[0] for r in sorted_rois]
    roi_values = [r[1] for r in sorted_rois]

    # Create bar plot
    fig, ax = plt.subplots(figsize=(10, 5))

    bars = ax.bar(roi_names, roi_values, color=color, alpha=0.7, edgecolor="black")

    # Highlight top ROIs
    max_val = max(roi_values)
    for i, bar in enumerate(bars):
        if roi_values[i] > max_val * 0.7:
            bar.set_color("crimson")
            bar.set_alpha(0.8)

    ax.set_ylabel("Current Source Density (nA·m)", fontsize=12)
    ax.set_xlabel("Brain Region", fontsize=12)
    ax.set_title(title, fontsize=14, fontweight="bold")
    ax.grid(True, alpha=0.3, axis="y")

    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()

    if output_file:
        plt.savefig(output_file, dpi=300, bbox_inches="tight")
        print(f"✓ Saved: {output_file}")

    plt.show()
    plt.close()


def compare_conditions_sources(components_dict, subjects_dir, src, output_dir):
    """
    Compare sources across conditions (Pre, Post, Online)

    Parameters:
    -----------
    components_dict : dict
        Nested dict: {condition: {component: {'stc': stc, ...}}}
    subjects_dir : str
        Path to subjects directory
    src : SourceSpaces
        Source space
    output_dir : str
        Output directory
    """

    print("\n" + "=" * 70)
    print("COMPARING SOURCES ACROSS CONDITIONS")
    print("=" * 70)

    # For each component, compare conditions
    component_names = ["P3a", "P3b", "SW"]

    for comp_name in component_names:
        print(f"\n--- {comp_name} Comparison ---")

        # Create figure for this component
        fig, axes = plt.subplots(1, 3, figsize=(15, 4))

        condition_names = list(components_dict.keys())

        for idx, cond_name in enumerate(condition_names):
            if comp_name in components_dict[cond_name]:
                stc = components_dict[cond_name][comp_name]["stc"]

                # Extract ROI values - PASS src HERE
                roi_dict = extract_roi_values(stc, subjects_dir, src=src)

                # Plot
                ax = axes[idx]
                sorted_rois = sorted(roi_dict.items(), key=lambda x: x[1], reverse=True)
                roi_names = [r[0] for r in sorted_rois]
                roi_values = [r[1] for r in sorted_rois]

                ax.bar(roi_names, roi_values, alpha=0.7)
                ax.set_title(f"{cond_name}", fontsize=12, fontweight="bold")
                ax.set_ylabel("Current Source Density (nA·m)", fontsize=10)
                ax.tick_params(axis="x", rotation=45, labelsize=9)
                ax.grid(True, alpha=0.3, axis="y")

        plt.suptitle(f"{comp_name} Source Comparison", fontsize=14, fontweight="bold")
        plt.tight_layout()

        output_file = os.path.join(output_dir, f"comparison_{comp_name}.png")
        plt.savefig(output_file, dpi=300, bbox_inches="tight")
        print(f"✓ Saved: {output_file}")

        plt.show()
        plt.close()


def run_source_localization_analysis(
    p300_data_dict, labels, time_ms, output_dir, sfreq=512, ch_pos_file=None
):
    """
    Complete source localization pipeline
    """

    print("\n" + "=" * 70)
    print("P300 SOURCE LOCALIZATION ANALYSIS")
    print("=" * 70)

    # Create output directory
    os.makedirs(output_dir, exist_ok=True)

    # Setup source space and forward model (do once)
    src, fwd, subjects_dir, info, valid_indices = setup_source_space_and_forward(
        labels, sfreq, ch_pos_file=ch_pos_file
    )

    print(f"\n✓ Using {len(valid_indices)}/{len(labels)} channels for source analysis")

    # DEBUG: Check valid_indices here too
    print(f"DEBUG in run_source_localization:")
    print(f"  valid_indices type: {type(valid_indices)}")
    print(f"  valid_indices length: {len(valid_indices)}")
    print(f"  valid_indices content: {valid_indices}")

    # Store all results
    all_components = {}

    # Process each condition
    for cond_name, p300_data in p300_data_dict.items():
        if p300_data is None:
            print(f"\n⚠️ Skipping {cond_name} (no data)")
            continue

        print(f"\n{'='*70}")
        print(f"PROCESSING: {cond_name}")
        print(f"{'='*70}")

        # Create condition output directory
        cond_dir = os.path.join(output_dir, cond_name.lower())
        os.makedirs(cond_dir, exist_ok=True)

        # Extract component sources
        components = extract_component_sources(
            p300_data, labels, time_ms, fwd, info, valid_indices, cond_name
        )

        all_components[cond_name] = components

        # For each component, create visualizations
        for comp_name, comp_data in components.items():
            stc = comp_data["stc"]

            print(f"\n--- Visualizing {comp_name} ---")

            # Plot sources on brain
            brain_file = os.path.join(cond_dir, f"brain_{comp_name}.png")
            plot_sources_on_brain(
                stc,
                subjects_dir,
                title=f"{cond_name} {comp_name} Sources",
                output_file=brain_file,
            )

            # Extract and plot ROI values
            roi_dict = extract_roi_values(stc, subjects_dir, src=src)

            roi_file = os.path.join(cond_dir, f"roi_{comp_name}.png")
            plot_roi_bars(
                roi_dict,
                title=f"{cond_name} {comp_name} - ROI Activity",
                output_file=roi_file,
                color=(
                    "steelblue"
                    if cond_name == "Pre"
                    else "coral" if cond_name == "Post" else "mediumseagreen"
                ),
            )

    # Compare across conditions
    if len(all_components) > 1:
        compare_conditions_sources(
            all_components, subjects_dir, src, output_dir
        )  # ✅ Add src

    print("\n" + "=" * 70)
    print("✅ SOURCE LOCALIZATION ANALYSIS COMPLETE")
    print(f"📁 Results saved to: {output_dir}")
    print("=" * 70)

    return all_components
