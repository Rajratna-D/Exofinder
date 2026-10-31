import os
import pandas as pd
import numpy as np
import lightkurve as lk
from astroquery.ipac.nexsci.nasa_exoplanet_archive import NasaExoplanetArchive

def fetch_labeled_list(num_per_class=700, existing_labels_path=None):
    """
    Queries the ExoFOP TESS TOI table for a balanced set of stars.
    Preserves any existing labeled targets if existing_labels_path is supplied.
    
    Parameters:
    -----------
    num_per_class : int
        Number of stars to fetch per class (confirmed_planet, eclipsing_binary, false_positive).
    existing_labels_path : str, optional
        Path to an existing labels.csv to retain already downloaded stars.
        
    Returns:
    --------
    pd.DataFrame
        DataFrame with columns: star_id (TIC ID), label, catalog_period
    """
    print(f"Fetching TESS Objects of Interest from ExoFOP (targeting up to {num_per_class} per class)...")
    url = "https://exofop.ipac.caltech.edu/tess/download_toi.php?sort=toi&output=csv"
    
    try:
        df = pd.read_csv(url)
    except Exception as e:
        print(f"Error fetching ExoFOP CSV: {e}")
        print("Attempting fallback using Astroquery.ipac.nexsci.nasa_exoplanet_archive...")
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
        
        df_eb = pd.DataFrame(columns=['TIC ID', 'Period (days)', 'TESS Disposition', 'TFOPWG Disposition'])
        df = pd.concat([df_cp, df_eb, df_fp], ignore_index=True)

    # 1. Confirmed Planets
    df_cp = df[df['TFOPWG Disposition'].isin(['CP', 'KP'])].copy().drop_duplicates(subset=['TIC ID'])
    # 2. Eclipsing Binaries
    df_eb = df[df['TESS Disposition'] == 'EB'].copy().drop_duplicates(subset=['TIC ID'])
    # 3. False Positives
    df_fp = df[(df['TFOPWG Disposition'] == 'FP') & (df['TESS Disposition'] != 'EB')].copy().drop_duplicates(subset=['TIC ID'])

    # Read existing labels if present
    existing_df = None
    if existing_labels_path and os.path.exists(existing_labels_path):
        try:
            existing_df = pd.read_csv(existing_labels_path)
            print(f"Preserving {len(existing_df)} previously cataloged targets from {existing_labels_path}.")
        except Exception as e:
            print(f"Could not read existing labels: {e}")
            existing_df = None

    def sample_with_preservation(df_source, class_label):
        existing_class = pd.DataFrame()
        known_sids = set()
        if existing_df is not None and not existing_df.empty and 'label' in existing_df.columns:
            existing_class = existing_df[existing_df['label'] == class_label].copy()
            known_sids = set(existing_class['star_id'].astype(int))

        df_source['TIC ID'] = df_source['TIC ID'].astype(int)
        df_new = df_source[~df_source['TIC ID'].isin(known_sids)].copy()
        
        needed = max(0, num_per_class - len(existing_class))
        if needed > 0 and len(df_new) > 0:
            sample_count = min(needed, len(df_new))
            sample_new = df_new.sample(n=sample_count, random_state=42).copy()
            sample_new = sample_new.rename(columns={'TIC ID': 'star_id', 'Period (days)': 'catalog_period'})
            sample_new['label'] = class_label
            sample_new['star_id'] = sample_new['star_id'].astype(int)
            sample_new['catalog_period'] = pd.to_numeric(sample_new['catalog_period'], errors='coerce')
            sample_new = sample_new[['star_id', 'label', 'catalog_period']]
            combined_class = pd.concat([existing_class, sample_new], ignore_index=True)
            print(f"  {class_label}: {len(existing_class)} retained + {sample_count} new = {len(combined_class)} total")
            return combined_class
        else:
            print(f"  {class_label}: {len(existing_class)} retained (target cap reached or no new)")
            return existing_class

    res_cp = sample_with_preservation(df_cp, 'confirmed_planet')
    res_eb = sample_with_preservation(df_eb, 'eclipsing_binary')
    res_fp = sample_with_preservation(df_fp, 'false_positive')

    final_df = pd.concat([res_cp, res_eb, res_fp], ignore_index=True).drop_duplicates(subset=['star_id'])
    final_df['star_id'] = final_df['star_id'].astype(int)
    final_df['catalog_period'] = pd.to_numeric(final_df['catalog_period'], errors='coerce')
    
    print(f"\nFinal unified target catalog ready: {len(final_df)} stars across 3 classes:")
    print(final_df['label'].value_counts())
    
    return final_df


def download_light_curves(labels_df, output_dir, failures_path, progress_json_path=None):
    """
    Downloads raw TESS light curve data from MAST using lightkurve and saves as CSV.
    Logs failed downloads to failures_path and writes live progress.
    """
    import socket
    import threading
    import concurrent.futures
    import json
    socket.setdefaulttimeout(30)
    
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
    initial_cached = success_count
    total_to_download = len(star_ids_to_download)
    print(f"Beginning download of {total_to_download} new light curves in parallel (Cached: {success_count}/{total_stars})...", flush=True)
    
    failures = []
    completed_in_batch = 0
    new_success_count = 0
    progress_lock = threading.Lock()
    
    def update_progress_file(current_sid=None, is_success=True, status=None):
        if not progress_json_path:
            return
        try:
            os.makedirs(os.path.dirname(progress_json_path), exist_ok=True)
            pct = round((success_count / total_stars) * 100, 1) if total_stars > 0 else 100.0
            cur_status = status if status else ("complete" if completed_in_batch >= total_to_download else "downloading")
            data = {
                "total_catalog": total_stars,
                "already_cached": initial_cached,
                "batch_total": total_to_download,
                "batch_completed": completed_in_batch,
                "new_downloaded": new_success_count,
                "total_available": success_count,
                "failed": len(failures),
                "percent": pct,
                "current_star": current_sid,
                "status": cur_status
            }
            with open(progress_json_path, 'w') as f:
                json.dump(data, f, indent=2)
        except Exception:
            pass

    update_progress_file()
    
    def download_single_star(star_id):
        filepath = os.path.join(output_dir, f"TIC_{star_id}.csv")
        last_err = ""
        for attempt in range(3):
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
                return {'star_id': star_id, 'success': True, 'points': len(lc_df)}
            except Exception as e:
                last_err = str(e)
                import time, random
                time.sleep(random.uniform(0.5, 1.2))
                
        return {'star_id': star_id, 'success': False, 'error': last_err}

    # Execute parallel downloads with 10 workers
    with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
        futures = {executor.submit(download_single_star, sid): sid for sid in star_ids_to_download}
        for future in concurrent.futures.as_completed(futures):
            res = future.result()
            with progress_lock:
                completed_in_batch += 1
                if res['success']:
                    success_count += 1
                    new_success_count += 1
                else:
                    failures.append({
                        'star_id': res['star_id'],
                        'error_message': res.get('error', 'unknown error')
                    })
                    if os.path.dirname(failures_path):
                        os.makedirs(os.path.dirname(failures_path), exist_ok=True)
                    pd.DataFrame(failures).to_csv(failures_path, index=False)
                
                update_progress_file(res['star_id'], res['success'])
                
                # Print clean, continuous update every 10 downloads or at completion
                if completed_in_batch % 10 == 0 or completed_in_batch == total_to_download:
                    pct = (success_count / total_stars) * 100.0 if total_stars > 0 else 0
                    print(
                        f"[DOWNLOAD UPDATE] {success_count}/{total_stars} ({pct:.1f}%) | "
                        f"New in Batch: {completed_in_batch}/{total_to_download} "
                        f"(Success: {new_success_count}, Failed: {len(failures)})",
                        flush=True
                    )
                
    # Write final failures file
    if os.path.dirname(failures_path):
        os.makedirs(os.path.dirname(failures_path), exist_ok=True)
    failures_df = pd.DataFrame(failures)
    failures_df.to_csv(failures_path, index=False)
    update_progress_file(status="complete")
    
    print(f"\nDownload summary:")
    print(f"  Total available light curves: {success_count}/{total_stars}")
    print(f"  New downloads in this run: {new_success_count}")
    print(f"  Failures: {len(failures)}/{total_to_download} (logged to {failures_path})", flush=True)


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


def load_detrended_star(star_id, base_dir=None):
    """
    Loads detrended light curve for a given TIC star_id.
    
    Checks:
    1. data/detrended/TIC_{star_id}.csv if available locally.
    2. data/detrended_parquet/ compressed archive using PyArrow predicate pushdown.
    
    Parameters:
    -----------
    star_id : int or str
        TIC star identifier (e.g. 297153209 or "TIC 297153209").
    base_dir : str, optional
        Project root path. If None, auto-detected relative to this file.
        
    Returns:
    --------
    pd.DataFrame
        DataFrame with columns ['time', 'detrended_flux', 'flux', 'flux_err', 'trend', 'star_id']
    """
    if isinstance(star_id, str):
        star_id = int(star_id.replace("TIC", "").replace("tic", "").replace("_", "").strip())
        
    if base_dir is None:
        base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
        
    # 1. Check local CSV first
    csv_path = os.path.join(base_dir, "data", "detrended", f"TIC_{star_id}.csv")
    if os.path.exists(csv_path):
        df = pd.read_csv(csv_path)
        if 'star_id' not in df.columns:
            df['star_id'] = star_id
        return df

    # 2. Check Parquet partitioned dataset
    pq_dir = os.path.join(base_dir, "data", "detrended_parquet")
    if os.path.exists(pq_dir):
        try:
            import pyarrow.parquet as pq
            dataset = pq.ParquetDataset(pq_dir, filters=[('star_id', '=', star_id)])
            table = dataset.read()
            if table.num_rows > 0:
                return table.to_pandas()
        except Exception as e:
            print(f"[DataLoader] Warning: Could not read TIC {star_id} from Parquet archive: {e}")
            
    raise FileNotFoundError(f"Light curve data for TIC {star_id} not found in CSV or Parquet archives.")


def unpack_parquet_to_csv(parquet_dir=None, output_dir=None, unpack_raw=True, raw_output_dir=None):
    """
    Extracts all individual CSV light curves from data/detrended_parquet/ into data/detrended/
    and optionally data/raw/.
    """
    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    if parquet_dir is None:
        parquet_dir = os.path.join(base_dir, "data", "detrended_parquet")
    if output_dir is None:
        output_dir = os.path.join(base_dir, "data", "detrended")
    if raw_output_dir is None:
        raw_output_dir = os.path.join(base_dir, "data", "raw")
        
    if not os.path.exists(parquet_dir):
        raise FileNotFoundError(f"Parquet directory not found at: {parquet_dir}")
        
    os.makedirs(output_dir, exist_ok=True)
    if unpack_raw:
        os.makedirs(raw_output_dir, exist_ok=True)
        
    import glob
    parts = sorted(glob.glob(os.path.join(parquet_dir, "*.parquet")))
    print(f"Unpacking {len(parts)} Parquet chunks into {output_dir} (and raw in {raw_output_dir})...")
    
    total_unpacked = 0
    for p in parts:
        df_part = pd.read_parquet(p)
        for sid, group in df_part.groupby('star_id'):
            # Save detrended curve
            out_csv = os.path.join(output_dir, f"TIC_{sid}.csv")
            cols_det = [c for c in ['time', 'flux', 'flux_err', 'trend', 'detrended_flux'] if c in group.columns]
            group[cols_det].to_csv(out_csv, index=False)
            
            # Save raw curve
            if unpack_raw:
                raw_csv = os.path.join(raw_output_dir, f"TIC_{sid}.csv")
                cols_raw = [c for c in ['time', 'flux', 'flux_err'] if c in group.columns]
                group[cols_raw].to_csv(raw_csv, index=False)
                
            total_unpacked += 1
            
    print(f"Successfully unpacked {total_unpacked} star CSVs into {output_dir} and {raw_output_dir}!")


if __name__ == "__main__":
    import sys
    if "--unpack-csv" in sys.argv:
        unpack_parquet_to_csv()
    else:
        print("ExoFinder DataLoader Module. Run with --unpack-csv to extract Parquet archive to CSVs.")

