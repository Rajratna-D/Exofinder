import os
import numpy as np
import pandas as pd
from scipy.stats import skew, kurtosis
from astropy.timeseries import BoxLeastSquares

def get_binned_profile(phase, flux, num_bins=80):
    """
    Bins the folded light curve between phases -0.15 and 0.15.
    
    Parameters:
    -----------
    phase : np.ndarray
        Folded phases (typically between -0.5 and 0.5).
    flux : np.ndarray
        Normalized flux values.
    num_bins : int
        Number of bins.
        
    Returns:
    --------
    bin_centers : np.ndarray
        Centers of the bins.
    binned_flux : np.ndarray
        Median flux in each bin.
    """
    bins = np.linspace(-0.15, 0.15, num_bins + 1)
    bin_centers = 0.5 * (bins[:-1] + bins[1:])
    binned_flux = []
    for i in range(len(bins)-1):
        mask = (phase >= bins[i]) & (phase < bins[i+1])
        binned_flux.append(np.median(flux[mask]) if np.any(mask) else 1.0)
    return bin_centers, np.array(binned_flux)

def extract_features_for_star(star_id, label, cat_period, bls_row, detrended_df, raw_df):
    """
    Extracts a feature vector for a single star based on its detrended light curve
    and BoxLeastSquares (BLS) period search results.
    
    Parameters:
    -----------
    star_id : int
        TIC ID of the star.
    label : str
        The catalog class label.
    cat_period : float
        The expected catalog period (from labels.csv).
    bls_row : dict/pd.Series
        Row containing recovered BLS parameters (period, epoch, depth, duration, power).
    detrended_df : pd.DataFrame
        DataFrame of detrended light curve with columns ['time', 'detrended_flux'].
    raw_df : pd.DataFrame
        DataFrame of raw light curve with columns ['time', 'flux'].
        
    Returns:
    --------
    dict
        Feature dictionary for the star.
    """
    time = detrended_df['time'].values
    flux = detrended_df['detrended_flux'].values
    
    raw_flux = raw_df['flux'].values
    
    # Retrieve BLS parameters
    period = float(bls_row['recovered_period'])
    epoch = float(bls_row['recovered_epoch'])
    depth = float(bls_row['recovered_depth'])
    duration = float(bls_row['recovered_duration'])
    bls_power = float(bls_row['bls_power'])
    
    # If BLS failed, fill features with defaults
    if np.isnan(period) or period <= 0:
        return {
            'star_id': star_id,
            'label': label,
            'catalog_period': cat_period,
            'bls_period': 0.0,
            'bls_depth': 0.0,
            'bls_duration': 0.0,
            'bls_power': 0.0,
            'transit_snr': 0.0,
            'mean_to_max_depth_ratio': 0.0,
            'in_transit_skew': 0.0,
            'in_transit_kurtosis': 0.0,
            'std_raw': np.std(raw_flux) if len(raw_flux) > 0 else 0.0,
            'std_detrended': np.std(flux) if len(flux) > 0 else 0.0,
            'transit_count': 0.0,
            'flux_kurtosis': kurtosis(flux) if len(flux) > 0 else 0.0,
            'ingress_egress_symmetry': 0.0,
            'secondary_eclipse_check': 0.0,
            'period_ratio': 0.0,
            'depth_to_noise': 0.0
        }
        
    # Calculate phase folding to isolate in-transit vs out-of-transit points
    phase = ((time - epoch) % period) / period
    phase = np.where(phase > 0.5, phase - 1.0, phase)
    
    # Define transit window mask (within half of the transit duration on either side of phase 0)
    # duration is in days, period is in days.
    half_dur_phase = (duration / 2.0) / period
    in_transit_mask = np.abs(phase) <= half_dur_phase
    
    # Out of transit mask (buffer of 1.5x duration to prevent ingress/egress overlaps)
    out_transit_mask = np.abs(phase) >= (1.5 * half_dur_phase)
    
    # 1. Calculate Out-Of-Transit (OOT) scatter to evaluate significance (SNR)
    flux_oot = flux[out_transit_mask]
    if len(flux_oot) > 10:
        std_oot = np.std(flux_oot)
    else:
        std_oot = np.std(flux) if len(flux) > 0 else 1e-5
        
    std_oot = max(std_oot, 1e-6)  # Prevent division by zero
    transit_snr = depth / std_oot
    
    # 2. Extract Shape Metrics (U-vs-V shape to separate EBs from planets)
    # Calculate binned profile between -0.15 and 0.15
    bin_centers, binned_flux = get_binned_profile(phase, flux, num_bins=80)
    
    # Identify in-transit points in the binned profile
    binned_in_transit_mask = np.abs(bin_centers) <= half_dur_phase
    binned_flux_in = binned_flux[binned_in_transit_mask]
    
    if len(binned_flux_in) > 3:
        # Depth at each point: distance from normalized baseline 1.0
        depths = 1.0 - binned_flux_in
        max_depth = np.max(depths)
        mean_depth = np.mean(depths)
        
        # Mean-to-Max depth ratio:
        # V-shaped dips (EBs) have a peak shape, giving mean/max ratio ~0.5.
        # U-shaped dips (planets) have a boxy flat bottom, giving mean/max ratio > 0.75.
        mean_to_max_ratio = mean_depth / max_depth if max_depth > 0 else 0.0
        
        in_transit_skew = skew(binned_flux_in)
        in_transit_kurt = kurtosis(binned_flux_in)
        
        # Clean any NaNs
        if np.isnan(mean_to_max_ratio):
            mean_to_max_ratio = 0.0
        if np.isnan(in_transit_skew):
            in_transit_skew = 0.0
        if np.isnan(in_transit_kurt):
            in_transit_kurt = 0.0
    else:
        mean_to_max_ratio = 0.0
        in_transit_skew = 0.0
        in_transit_kurt = 0.0
        
    # 3. Time Series and Overall Stats
    std_raw = np.std(raw_flux) if len(raw_flux) > 0 else 0.0
    std_detrend = np.std(flux) if len(flux) > 0 else 0.0
    total_time = np.max(time) - np.min(time) if len(time) > 0 else 0.0
    transit_count = total_time / period
    
    flux_kurt = kurtosis(flux) if len(flux) > 0 else 0.0
    if np.isnan(flux_kurt):
        flux_kurt = 0.0

    # Calculate ingress_egress_symmetry
    left_half_flux = flux[(phase < 0) & in_transit_mask]
    right_half_flux = flux[(phase > 0) & in_transit_mask]
    if len(left_half_flux) < 3 or len(right_half_flux) < 3:
        ingress_egress_symmetry = 0.0
    else:
        skew_left = skew(left_half_flux)
        skew_right = skew(right_half_flux)
        if np.isnan(skew_left) or np.isnan(skew_right):
            ingress_egress_symmetry = 0.0
        else:
            ingress_egress_symmetry = abs(skew_left) - abs(skew_right)

    # Calculate secondary_eclipse_check
    secondary_eclipse_check = 0.0
    try:
        # Only check for secondary eclipse if period/2 is physically meaningful (>= 0.5 days)
        if period / 2.0 >= 0.5:
            clean_mask = np.isfinite(time) & np.isfinite(flux)
            t_clean = time[clean_mask]
            f_clean = flux[clean_mask]
            if len(t_clean) >= 100:
                dy_clean = detrended_df['flux_err'].values[clean_mask] if 'flux_err' in detrended_df.columns else None
                bls_sec = BoxLeastSquares(t_clean, f_clean, dy=dy_clean)
                duration_grid = np.linspace(0.01, min(0.25, max(0.02, period / 4.0)), 10)
                results_sec = bls_sec.power([period / 2.0], duration_grid)
                peak_power = float(np.nanmax(results_sec.power))
                if not np.isnan(peak_power):
                    secondary_eclipse_check = peak_power
    except Exception:
        secondary_eclipse_check = 0.0
    
    return {
        'star_id': star_id,
        'label': label,
        'catalog_period': cat_period,
        'bls_period': period,
        'bls_depth': depth,
        'bls_duration': duration,
        'bls_power': bls_power,
        'transit_snr': transit_snr,
        'mean_to_max_depth_ratio': mean_to_max_ratio,
        'in_transit_skew': in_transit_skew,
        'in_transit_kurtosis': in_transit_kurt,
        'std_raw': std_raw,
        'std_detrended': std_detrend,
        'transit_count': transit_count,
        'flux_kurtosis': flux_kurt,
        'ingress_egress_symmetry': ingress_egress_symmetry,
        'secondary_eclipse_check': secondary_eclipse_check,
        'period_ratio': period / cat_period if cat_period > 0 and not np.isnan(cat_period) else 0.0,
        'depth_to_noise': depth / std_detrend if std_detrend > 0 else 0.0
    }


def extract_features_all(labels_path="data/labels.csv", bls_results_path="outputs/period_search_results.csv",
                         raw_dir="data/raw", detrended_dir="data/detrended", output_csv="outputs/features.csv"):
    """
    Loads all datasets, extracts features for each star, and saves them to a CSV file.
    """
    if not os.path.exists(labels_path):
        raise FileNotFoundError(f"Labels file not found: {labels_path}")
    if not os.path.exists(bls_results_path):
        raise FileNotFoundError(f"BLS search results file not found: {bls_results_path}")
        
    labels_df = pd.read_csv(labels_path)
    bls_df = pd.read_csv(bls_results_path)
    
    features_list = []
    failed_list = []
    
    print(f"Extracting features for {len(labels_df)} targets...")
    
    for idx, row in labels_df.iterrows():
        star_id = int(row['star_id'])
        label = row['label']
        cat_period = float(row['catalog_period'])
        
        # Load BLS row
        bls_sub = bls_df[bls_df['star_id'] == star_id]
        if len(bls_sub) == 0:
            print(f"Warning: No BLS results for TIC {star_id}")
            failed_list.append({'star_id': star_id, 'reason': 'No BLS results'})
            continue
        bls_row = bls_sub.iloc[0]
        
        # Load light curve files
        raw_file = os.path.join(raw_dir, f"TIC_{star_id}.csv")
        det_file = os.path.join(detrended_dir, f"TIC_{star_id}.csv")
        
        if not (os.path.exists(raw_file) and os.path.exists(det_file)):
            print(f"Warning: Missing files for TIC {star_id}")
            failed_list.append({'star_id': star_id, 'reason': 'Missing raw/detrended CSV files'})
            continue
            
        try:
            raw_df = pd.read_csv(raw_file)
            det_df = pd.read_csv(det_file)
            
            feat = extract_features_for_star(
                star_id=star_id,
                label=label,
                cat_period=cat_period,
                bls_row=bls_row,
                detrended_df=det_df,
                raw_df=raw_df
            )
            features_list.append(feat)
            
        except Exception as e:
            print(f"Error extracting features for TIC {star_id}: {e}")
            failed_list.append({'star_id': star_id, 'reason': str(e)})
            
    # Save features
    features_df = pd.DataFrame(features_list)
    
    # Sanity check: Ensure no missing values (NaNs) in the feature matrix
    # Loop through all columns that have NaN values; if numeric, impute with median, else fallback to 0.0
    nan_cols = features_df.columns[features_df.isna().any()].tolist()
    if nan_cols:
        print(f"Warning: Found NaN values in columns: {nan_cols}. Imputing with median/fallback.")
        for col in nan_cols:
            if pd.api.types.is_numeric_dtype(features_df[col]):
                col_median = features_df[col].median()
                if pd.isna(col_median):
                    features_df[col] = features_df[col].fillna(0.0)
                else:
                    features_df[col] = features_df[col].fillna(col_median)
            else:
                features_df[col] = features_df[col].fillna(0.0)
        
    os.makedirs(os.path.dirname(output_csv), exist_ok=True)
    features_df.to_csv(output_csv, index=False)
    print(f"Features extracted successfully! Saved to {output_csv} (Rows: {len(features_df)})")
    
    # Save failures log
    if failed_list:
        failed_df = pd.DataFrame(failed_list)
        failed_csv = os.path.join(os.path.dirname(output_csv), "feature_failures.csv")
        failed_df.to_csv(failed_csv, index=False)
        print(f"Logged {len(failed_list)} failures to {failed_csv}")
        
    return features_df
