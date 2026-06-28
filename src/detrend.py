import os
import pandas as pd
import numpy as np
from scipy.signal import savgol_filter

def detrend_light_curve(df, window_length=1001, polyorder=2, gap_threshold=0.5):
    """
    Applies a Savitzky-Golay filter to detrend a light curve, split by gaps
    to avoid edge artifacts near data downlinks.
    
    Parameters:
    -----------
    df : pd.DataFrame
        DataFrame with columns ['time', 'flux'] and optional ['flux_err']
    window_length : int
        The length of the filter window (must be an odd integer).
    polyorder : int
        The order of the polynomial used to fit the samples.
    gap_threshold : float
        Time gap threshold in days to split light curve into separate segments.
        
    Returns:
    --------
    pd.DataFrame
        DataFrame with columns ['time', 'flux', 'flux_err', 'trend', 'detrended_flux']
    """
    if df is None or df.empty:
        return pd.DataFrame(columns=['time', 'flux', 'trend', 'detrended_flux'])
        
    # Make a copy to avoid warnings and drop any potential missing values
    df_clean = df.dropna(subset=['time', 'flux']).copy()
    
    # Sort by time to ensure chronological order
    df_clean = df_clean.sort_values(by='time').reset_index(drop=True)
    
    # Identify gaps in time to split segments (e.g., data downlink in middle of sector)
    time_diffs = np.diff(df_clean['time'])
    gap_indices = np.where(time_diffs > gap_threshold)[0] + 1
    
    # Split the DataFrame into contiguous segments natively in pandas
    segments = []
    prev_idx = 0
    for idx in gap_indices:
        segments.append(df_clean.iloc[prev_idx:idx])
        prev_idx = idx
    segments.append(df_clean.iloc[prev_idx:])
    
    detrended_segments = []
    
    for seg in segments:
        # Adaptively cap window length to ~15% of segment length (minimum 51, must be odd)
        adaptive_window = min(window_length, max(51, int(len(seg) * 0.15)))
        if adaptive_window % 2 == 0:
            adaptive_window -= 1
        
        if len(seg) < adaptive_window:
            # If segment is too short for the adaptive window, adjust further
            seg_window = len(seg)
            if seg_window % 2 == 0:
                seg_window -= 1
            if seg_window <= polyorder:
                # Segment too small to fit polynomial — normalize by median
                seg = seg.copy()
                median_val = np.median(seg['flux'])
                median_val = median_val if median_val > 0 else 1.0
                seg['trend'] = median_val
                seg['detrended_flux'] = seg['flux'] / median_val
                detrended_segments.append(seg)
                continue
        else:
            seg_window = adaptive_window
            
        # Apply Savitzky-Golay filter to the segment
        seg = seg.copy()
        try:
            trend = savgol_filter(seg['flux'].values, window_length=seg_window, polyorder=polyorder)
            # Guard against zero or negative trend values (prevent division issues)
            trend = np.where(trend <= 0, np.median(seg['flux']), trend)
            seg['trend'] = trend
            # Normalize flux by dividing by the trend
            seg['detrended_flux'] = seg['flux'].values / trend
        except Exception as e:
            # Fallback if SG filter fails
            print(f"Warning: SG filter failed for a segment: {e}. Falling back to median.")
            median_val = np.median(seg['flux'])
            median_val = median_val if median_val > 0 else 1.0
            seg['trend'] = median_val
            seg['detrended_flux'] = seg['flux'] / median_val
            
        detrended_segments.append(seg)
        
    # Recombine all segments
    final_df = pd.concat(detrended_segments, ignore_index=True)
    return final_df

def process_all_stars(labels_path="data/labels.csv", raw_dir="data/raw", output_dir="data/detrended"):
    """
    Processes all stars in labels_path, detrends their light curves, and saves them.
    """
    if not os.path.exists(labels_path):
        raise FileNotFoundError(f"Labels file not found: {labels_path}")
        
    labels_df = pd.read_csv(labels_path)
    os.makedirs(output_dir, exist_ok=True)
    
    processed_count = 0
    print(f"Detrending light curves for {len(labels_df)} stars...")
    
    for idx, row in labels_df.iterrows():
        star_id = int(row['star_id'])
        raw_file = os.path.join(raw_dir, f"TIC_{star_id}.csv")
        
        if not os.path.exists(raw_file):
            print(f"Warning: Raw file not found for TIC {star_id}")
            continue
            
        # Load raw data
        df = pd.read_csv(raw_file)
        
        # Apply detrending
        detrended_df = detrend_light_curve(df)
        
        # Save detrended data
        out_file = os.path.join(output_dir, f"TIC_{star_id}.csv")
        detrended_df.to_csv(out_file, index=False)
        processed_count += 1
        
    print(f"Successfully detrended and saved {processed_count}/{len(labels_df)} light curves to {output_dir}")
