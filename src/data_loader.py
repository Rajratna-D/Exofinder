import os
import pandas as pd
import numpy as np
import lightkurve as lk
from astroquery.ipac.nexsci.nasa_exoplanet_archive import NasaExoplanetArchive

def fetch_labeled_list(num_per_class=10):
    """
    Queries the ExoFOP TESS TOI table for a balanced set of stars.
    
    Parameters:
    -----------
    num_per_class : int
        Number of stars to fetch per class (confirmed_planet, eclipsing_binary, false_positive).
        
    Returns:
    --------
    pd.DataFrame
        DataFrame with columns: star_id (TIC ID), label, catalog_period
    """
    print(f"Fetching TESS Objects of Interest from ExoFOP (fetching {num_per_class} per class)...")
    url = "https://exofop.ipac.caltech.edu/tess/download_toi.php?sort=toi&output=csv"
    
    try:
        df = pd.read_csv(url)
    except Exception as e:
        print(f"Error fetching ExoFOP CSV: {e}")
        print("Attempting fallback using Astroquery.ipac.nexsci.nasa_exoplanet_archive...")
        # Fallback to query Astroquery (which won't have EB, but will get confirmed and false positive)
        print("Fetching confirmed planet targets via Astroquery...")
        cp_table = NasaExoplanetArchive.query_criteria(
            table="toi",
            where="tfopwg_disp = 'CP' or tfopwg_disp = 'KP'",
            select="tid, tfopwg_disp, pl_orbper"
        )
        df_cp = cp_table.to_pandas().drop_duplicates(subset=['tid'])
        df_cp = df_cp.rename(columns={'tid': 'TIC ID', 'pl_orbper': 'Period (days)'})
        df_cp['TFOPWG Disposition'] = 'CP'
        df_cp['TESS Disposition'] = 'CP'
        
        print("Fetching false positive targets via Astroquery...")
        fp_table = NasaExoplanetArchive.query_criteria(
            table="toi",
            where="tfopwg_disp = 'FP'",
            select="tid, tfopwg_disp, pl_orbper"
        )
        df_fp = fp_table.to_pandas().drop_duplicates(subset=['tid'])
        df_fp = df_fp.rename(columns={'tid': 'TIC ID', 'pl_orbper': 'Period (days)'})
        df_fp['TFOPWG Disposition'] = 'FP'
        df_fp['TESS Disposition'] = 'FP'
        
        # We can't fetch EBs via Astroquery easily because the TAP table doesn't have TESS Disposition 'EB'.
        # We will create an empty DataFrame for EB in this fallback case.
        df_eb = pd.DataFrame(columns=['TIC ID', 'Period (days)', 'TESS Disposition', 'TFOPWG Disposition'])
        
        df = pd.concat([df_cp, df_eb, df_fp], ignore_index=True)

    # We filter and sample for each class
    # 1. Confirmed Planets: TFOPWG Disposition is CP or KP
    df_cp = df[df['TFOPWG Disposition'].isin(['CP', 'KP'])].copy()
    df_cp['label'] = 'confirmed_planet'
    
    # 2. Eclipsing Binaries: TESS Disposition is EB
    df_eb = df[df['TESS Disposition'] == 'EB'].copy()
    df_eb['label'] = 'eclipsing_binary'
    
    # 3. False Positives: TFOPWG Disposition is FP and not EB
    df_fp = df[(df['TFOPWG Disposition'] == 'FP') & (df['TESS Disposition'] != 'EB')].copy()
    df_fp['label'] = 'false_positive'
    
    # Drop duplicates by TIC ID
    df_cp = df_cp.drop_duplicates(subset=['TIC ID'])
    df_eb = df_eb.drop_duplicates(subset=['TIC ID'])
    df_fp = df_fp.drop_duplicates(subset=['TIC ID'])
    
    # Take a sample of each class
    sample_cp = df_cp.head(num_per_class).copy()
    sample_eb = df_eb.head(num_per_class).copy()
    sample_fp = df_fp.head(num_per_class).copy()
    
    # Combine the samples
    combined = pd.concat([sample_cp, sample_eb, sample_fp], ignore_index=True)
    
    # Clean up column names and types
    combined = combined.rename(columns={
        'TIC ID': 'star_id',
        'Period (days)': 'catalog_period'
    })
    
    # Ensure proper data types
    combined['star_id'] = combined['star_id'].astype(int)
    combined['catalog_period'] = pd.to_numeric(combined['catalog_period'], errors='coerce')
    
    # Rearrange and return the final dataframe
    final_df = combined[['star_id', 'label', 'catalog_period']]
    
    print(f"Successfully retrieved a list of {len(final_df)} stars:")
    print(final_df['label'].value_counts())
    
    return final_df


def download_light_curves(labels_df, output_dir, failures_path):
    """
    Downloads raw TESS light curve data from MAST using lightkurve and saves as CSV.
    Logs failed downloads to failures_path.
    
    Parameters:
    -----------
    labels_df : pd.DataFrame
        DataFrame containing 'star_id' (TIC ID).
    output_dir : str
        Directory to save raw light curves to.
    failures_path : str
        Path to save download failures CSV file.
    """
    import socket
    import threading
    import concurrent.futures
    socket.setdefaulttimeout(30)  # Set standard socket timeout to prevent indefinite hangs
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Filter out stars that are already downloaded
    star_ids_to_download = []
    success_count = 0
    for idx, row in labels_df.iterrows():
        star_id = int(row['star_id'])
        filename = f"TIC_{star_id}.csv"
        filepath = os.path.join(output_dir, filename)
        if os.path.exists(filepath) and os.path.getsize(filepath) > 100:
            success_count += 1
        else:
            star_ids_to_download.append(star_id)
            
    total_stars = len(labels_df)
    print(f"Beginning download of {len(star_ids_to_download)} light curves in parallel (Cached: {success_count}/{total_stars})...")
    
    failures = []
    
    def download_single_star(star_id):
        filepath = os.path.join(output_dir, f"TIC_{star_id}.csv")
        try:
            # Search for TESS light curves. SPOC is preferred.
            search_result = lk.search_lightcurve(f"TIC {star_id}", mission="TESS", author="SPOC")
            if len(search_result) == 0:
                search_result = lk.search_lightcurve(f"TIC {star_id}", mission="TESS")
            if len(search_result) == 0:
                raise ValueError("No TESS light curve data available on MAST for this target.")
                
            # Download first sector
            lc = search_result[0].download()
            if lc is None:
                raise ValueError("Lightkurve download returned None.")
                
            time_val = np.asarray(lc.time.value).astype(np.float64)
            flux_raw = lc.flux.value if hasattr(lc.flux, 'value') else lc.flux
            flux_val = np.asarray(flux_raw).astype(np.float64)
            
            flux_err_raw = lc.flux_err.value if hasattr(lc.flux_err, 'value') else lc.flux_err
            flux_err_val = np.asarray(flux_err_raw).astype(np.float64)
            
            lc_df = pd.DataFrame({
                'time': time_val,
                'flux': flux_val,
                'flux_err': flux_err_val
            }).dropna(subset=['time', 'flux'])
            
            lc_df.to_csv(filepath, index=False)
            try:
                print(f"  [SUCCESS] TIC {star_id} saved ({len(lc_df)} points)")
            except Exception:
                pass
            return {'star_id': star_id, 'success': True}
        except Exception as e:
            error_msg = str(e)
            try:
                print(f"  [FAILED] TIC {star_id}: {error_msg}")
            except Exception:
                pass
            return {'star_id': star_id, 'success': False, 'error': error_msg}

    failures_lock = threading.Lock()

    # Execute parallel downloads with 10 workers
    with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
        futures = {executor.submit(download_single_star, sid): sid for sid in star_ids_to_download}
        for future in concurrent.futures.as_completed(futures):
            res = future.result()
            if res['success']:
                success_count += 1
            else:
                with failures_lock:
                    failures.append({
                        'star_id': res['star_id'],
                        'error_message': res['error']
                    })
                    # Incrementally save failures list
                    os.makedirs(os.path.dirname(failures_path), exist_ok=True)
                    pd.DataFrame(failures).to_csv(failures_path, index=False)
                
    # Write final failures file
    os.makedirs(os.path.dirname(failures_path), exist_ok=True)
    failures_df = pd.DataFrame(failures)
    failures_df.to_csv(failures_path, index=False)
    
    print(f"\nDownload summary:")
    print(f"  Successful downloads: {success_count}/{total_stars}")
    print(f"  Failures: {len(failures)}/{total_stars} (logged to {failures_path})")


def normalize_light_curve(df):
    """
    Normalizes the flux column of a light curve DataFrame around 1.0.
    Divides both 'flux' and 'flux_err' (if present) by the median flux.
    
    Parameters:
    -----------
    df : pd.DataFrame
        DataFrame containing 'flux' and optionally 'flux_err'.
        
    Returns:
    --------
    pd.DataFrame
        DataFrame with normalized flux and flux_err.
    """
    df_norm = df.copy()
    median_flux = df_norm['flux'].median()
    
    if median_flux <= 0:
        raise ValueError(f"Median flux is non-positive ({median_flux}), cannot normalize.")
        
    df_norm['flux'] = df_norm['flux'] / median_flux
    if 'flux_err' in df_norm.columns:
        df_norm['flux_err'] = df_norm['flux_err'] / median_flux
        
    return df_norm


def load_light_curve(file_path, normalize=True):
    """
    Source-agnostic loader that reads a CSV file containing time and flux data.
    
    Parameters:
    -----------
    file_path : str
        Path to the light curve CSV file.
    normalize : bool, default True
        If True, normalizes the flux (and flux_err) around 1.0 using median normalization.
        
    Returns:
    --------
    pd.DataFrame
        DataFrame with standardized columns ['time', 'flux', 'flux_err' (optional)]
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Light curve file not found: {file_path}")
        
    # Read the file
    df = pd.read_csv(file_path)
    
    # Normalize column names to lowercase to make it source-agnostic
    df = df.rename(columns={col: col.lower() for col in df.columns})
    
    # Verify required columns are present
    if 'time' not in df.columns or 'flux' not in df.columns:
        raise KeyError(
            f"The light curve file must contain 'time' and 'flux' columns (case-insensitive). "
            f"Found columns: {list(df.columns)}"
        )
        
    # Standardize columns to return
    columns_to_return = ['time', 'flux']
    if 'flux_err' in df.columns:
        columns_to_return.append('flux_err')
        
    # Drop rows that don't have valid values for time or flux
    df_clean = df[columns_to_return].dropna(subset=['time', 'flux']).reset_index(drop=True)
    
    # Apply normalization if requested
    if normalize:
        df_clean = normalize_light_curve(df_clean)
        
    return df_clean
