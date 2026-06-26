import os
import numpy as np
import pandas as pd
from astropy.timeseries import BoxLeastSquares

def find_transit_period(time, flux, flux_err=None, min_period=0.5, max_period=15.0, num_durations=10):
    """
    Runs Astropy's BoxLeastSquares (BLS) periodogram on a light curve
    to identify the best candidate period, transit epoch, depth, and duration.
    
    Parameters:
    -----------
    time : np.ndarray
        Array of time values (typically BTJD).
    flux : np.ndarray
        Array of normalized flux values (baseline around 1.0).
    flux_err : np.ndarray, optional
        Array of flux uncertainties.
    min_period : float
        Minimum orbital period to search (days).
    max_period : float
        Maximum orbital period to search (days).
    num_durations : int
        Number of transit durations to test in the search grid.
        
    Returns:
    --------
    dict
        Dictionary containing best fit parameters:
        - 'period': Best fit period (days)
        - 'epoch': Transit epoch T0 (BTJD)
        - 'depth': Transit depth (fractional)
        - 'duration': Transit duration (days)
        - 'power': Peak power of the BLS periodogram
        - 'periodogram': The raw BLS power result object
    """
    # Clean arrays to guarantee no NaNs or infs
    clean_mask = np.isfinite(time) & np.isfinite(flux)
    if flux_err is not None:
        clean_mask = clean_mask & np.isfinite(flux_err)
        dy = flux_err[clean_mask]
    else:
        dy = None
        
    t = time[clean_mask]
    y = flux[clean_mask]
    
    if len(t) < 100:
        # Too few points to run BLS
        return {
            'period': np.nan, 'epoch': np.nan, 'depth': np.nan,
            'duration': np.nan, 'power': np.nan, 'periodogram': None
        }
        
    # Set up BLS periodogram
    bls = BoxLeastSquares(t, y, dy=dy)
    
    # Generate optimal period grid automatically
    periods = bls.autoperiod(
        0.1,  # Representative duration in days
        minimum_period=min_period,
        maximum_period=max_period,
        frequency_factor=5.0  # Controls grid density (5.0 is dense enough for discovery)
    )
    
    # Grid of transit durations to search (typically 1% to 15% of the orbital period)
    # TESS transits are usually a few hours long (e.g. 0.02 to 0.25 days)
    durations = np.linspace(0.01, 0.25, num_durations)
    
    # Run the period search
    results = bls.power(periods, durations)
    
    # Find the peak power index
    best_idx = np.argmax(results.power)
    
    return {
        'period': float(results.period[best_idx]),
        'epoch': float(results.transit_time[best_idx]),
        'depth': float(results.depth[best_idx]),
        'duration': float(results.duration[best_idx]),
        'power': float(results.power[best_idx]),
        'periodogram': results
    }

def fold_light_curve(time, flux, period, epoch):
    """
    Phase folds a light curve at the specified period and epoch.
    Shifts phases to be centered at 0.0 (ranging from -0.5 to 0.5).
    
    Parameters:
    -----------
    time : np.ndarray
        Time array.
    flux : np.ndarray
        Flux array.
    period : float
        Orbital period to fold at.
    epoch : float
        Transit epoch T0 (midpoint).
        
    Returns:
    --------
    tuple
        (phases, sorted_fluxes)
    """
    # Calculate phase
    phase = ((time - epoch) % period) / period
    # Shift phase to range [-0.5, 0.5]
    phase = np.where(phase > 0.5, phase - 1.0, phase)
    
    # Sort by phase
    sort_idx = np.argsort(phase)
    return phase[sort_idx], flux[sort_idx]

def run_period_search_all(labels_path="data/labels.csv", detrended_dir="data/detrended", output_csv="outputs/period_search_results.csv"):
    """
    Runs the BLS search on all detrended files and saves results.
    """
    if not os.path.exists(labels_path):
        raise FileNotFoundError(f"Labels file not found: {labels_path}")
        
    labels_df = pd.read_csv(labels_path)
    os.makedirs(os.path.dirname(output_csv), exist_ok=True)
    
    results = []
    success_count = 0
    
    print(f"Beginning BLS period search on {len(labels_df)} targets...")
    
    for idx, row in labels_df.iterrows():
        star_id = int(row['star_id'])
        label = row['label']
        cat_period = float(row['catalog_period'])
        
        file_path = os.path.join(detrended_dir, f"TIC_{star_id}.csv")
        if not os.path.exists(file_path):
            print(f"Warning: Detrended file not found for TIC {star_id}")
            continue
            
        df = pd.read_csv(file_path)
        
        time_arr = df['time'].values
        flux_arr = df['detrended_flux'].values
        flux_err = df['flux_err'].values if 'flux_err' in df.columns else None
        
        # Run BLS
        try:
            bls_res = find_transit_period(time_arr, flux_arr, flux_err)
            
            rec_period = bls_res['period']
            rec_epoch = bls_res['epoch']
            rec_depth = bls_res['depth']
            rec_duration = bls_res['duration']
            rec_power = bls_res['power']
            
            # Calculate error
            if not np.isnan(rec_period) and not np.isnan(cat_period):
                period_error = np.abs(rec_period - cat_period)
                relative_error = period_error / cat_period
            else:
                period_error = np.nan
                relative_error = np.nan
                
            results.append({
                'star_id': star_id,
                'label': label,
                'catalog_period': cat_period,
                'recovered_period': rec_period,
                'period_error': period_error,
                'relative_error': relative_error,
                'recovered_epoch': rec_epoch,
                'recovered_depth': rec_depth,
                'recovered_duration': rec_duration,
                'bls_power': rec_power
            })
            
            success_count += 1
            print(f"[{success_count}/{len(labels_df)}] TIC {star_id} ({label}): Cat Period={cat_period:.4f} d, Rec Period={rec_period:.4f} d (Error: {relative_error*100:.2f}%)")
            
        except Exception as e:
            print(f"Failed period search for TIC {star_id}: {e}")
            
    # Save to CSV
    results_df = pd.DataFrame(results)
    results_df.to_csv(output_csv, index=False)
    print(f"\nPeriod search results saved to {output_csv}")
    
    # Print accuracy statistics for planets
    planets_df = results_df[results_df['label'] == 'confirmed_planet']
    accurate_recoveries = planets_df[planets_df['relative_error'] < 0.05]
    print(f"Confirmed Planets: recovered {len(accurate_recoveries)}/{len(planets_df)} periods within 5% tolerance.")
    
    return results_df
