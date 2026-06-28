import os
import textwrap
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import streamlit as st

# 1. Set page config for a premium wide layout
st.set_page_config(
    page_title="ExoFinder | AI-Enabled Exoplanet Detection",
    page_icon="🪐",
    layout="wide",
    initial_sidebar_state="expanded"
)

# 2. Custom CSS styling for a dark-mode space aesthetic (Outfit typography, gradients, glassmorphism cards)
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;600;800&display=swap');
    
    html, body, [class*="css"] {
        font-family: 'Outfit', sans-serif;
    }
    
    .main-title {
        background: linear-gradient(90deg, #ff007f 0%, #7928ca 50%, #00dfd8 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-size: 2.8rem;
        font-weight: 800;
        margin-bottom: 0.2rem;
    }
    
    .subtitle {
        color: #a0aec0;
        font-size: 1.1rem;
        margin-bottom: 1.8rem;
    }
    
    .card {
        background-color: #111827;
        border-radius: 12px;
        padding: 1.5rem;
        border: 1px solid #1f2937;
        box-shadow: 0 4px 10px rgba(0, 0, 0, 0.3);
        margin-bottom: 1.5rem;
        transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
    }
    
    .card:hover {
        transform: translateY(-2px);
        border-color: #3b82f6;
        box-shadow: 0 8px 24px rgba(59, 130, 246, 0.15);
    }
    
    .metric-val {
        font-size: 1.8rem;
        font-weight: 800;
        color: #f3f4f6;
    }
    
    .metric-lbl {
        font-size: 0.9rem;
        color: #9ca3af;
        text-transform: uppercase;
        letter-spacing: 0.05em;
    }
    
    .badge {
        padding: 0.35rem 0.75rem;
        border-radius: 20px;
        font-size: 0.85rem;
        font-weight: 600;
        display: inline-block;
        text-align: center;
    }
    
    .badge-confirmed_planet {
        background-color: rgba(16, 185, 129, 0.15);
        color: #10b981;
        border: 1px solid #10b981;
    }
    
    .badge-eclipsing_binary {
        background-color: rgba(236, 72, 153, 0.15);
        color: #ec4899;
        border: 1px solid #ec4899;
    }
    
    .badge-false_positive {
        background-color: rgba(239, 68, 68, 0.15);
        color: #ef4444;
        border: 1px solid #ef4444;
    }
    
    .badge-high {
        background-color: rgba(59, 130, 246, 0.15);
        color: #3b82f6;
        border: 1px solid #3b82f6;
    }
    
    .badge-low_significance {
        background-color: rgba(245, 158, 11, 0.15);
        color: #f59e0b;
        border: 1px solid #f59e0b;
    }
    
    /* Summary table styling */
    .summary-table {
        width: 100%;
        border-collapse: collapse;
        margin-top: 1rem;
        color: #f3f4f6;
    }
    .summary-table th {
        padding: 12px;
        color: #9ca3af;
        font-size: 0.85rem;
        text-transform: uppercase;
        border-bottom: 2px solid #374151;
        text-align: left;
        letter-spacing: 0.05em;
    }
    .summary-table td {
        padding: 12px;
        border-bottom: 1px solid #1f2937;
        font-size: 0.95rem;
    }
    .summary-table tr:hover {
        background-color: rgba(31, 41, 55, 0.4);
    }
    
    /* Premium Sidebar Radio Buttons Navigation Style */
    div[data-testid="stSidebar"] div[role="radiogroup"] {
        display: flex;
        flex-direction: column;
        gap: 0.5rem;
    }

    div[data-testid="stSidebar"] div[role="radiogroup"] label {
        display: flex;
        align-items: center;
        background-color: #111827 !important;
        border: 1px solid #1f2937 !important;
        border-radius: 10px !important;
        padding: 0.75rem 1rem !important;
        margin: 0 !important;
        cursor: pointer !important;
        transition: all 0.25s cubic-bezier(0.4, 0, 0.2, 1) !important;
        width: 100% !important;
    }

    /* Hover state with slide effect */
    div[data-testid="stSidebar"] div[role="radiogroup"] label:hover {
        background-color: #1f2937 !important;
        border-color: #3b82f6 !important;
        transform: translateX(4px);
    }

    /* Active state gradient & glow */
    div[data-testid="stSidebar"] div[role="radiogroup"] label:has(input:checked) {
        background: linear-gradient(90deg, #1e3a8a 0%, #1e40af 100%) !important;
        border-color: #3b82f6 !important;
        box-shadow: 0 4px 12px rgba(59, 130, 246, 0.25) !important;
    }

    /* Hide the default radio circle button */
    div[data-testid="stSidebar"] div[role="radiogroup"] [data-baseweb="radio"] > div:first-child {
        display: none !important;
    }

    /* Text styling inside navigation buttons */
    div[data-testid="stSidebar"] div[role="radiogroup"] label span {
        color: #9ca3af !important;
        font-weight: 600 !important;
        font-size: 1rem !important;
        transition: color 0.2s ease !important;
    }

    div[data-testid="stSidebar"] div[role="radiogroup"] label:has(input:checked) span {
        color: #ffffff !important;
    }
</style>
""", unsafe_allow_html=True)

# Define paths relative to the directory of this script to allow running from any CWD
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
PREDICTIONS_PATH = os.path.join(PROJECT_ROOT, "outputs", "predictions.csv")
RAW_DIR = os.path.join(PROJECT_ROOT, "data", "raw")
DETRENDED_DIR = os.path.join(PROJECT_ROOT, "data", "detrended")

@st.cache_data
def load_predictions():
    if not os.path.exists(PREDICTIONS_PATH):
        return None
    preds = pd.read_csv(PREDICTIONS_PATH)
    # Ensure 'significance' column exists (backwards compatibility)
    if 'significance' not in preds.columns:
        preds['significance'] = 'high'
    if 'confidence' not in preds.columns:
        preds['confidence'] = 0.5
    
    # Merge with period search results to get recovered_epoch
    results_path = os.path.join(PROJECT_ROOT, "outputs", "period_search_results.csv")
    if os.path.exists(results_path):
        res = pd.read_csv(results_path)
        if 'recovered_epoch' in res.columns and 'star_id' in res.columns:
            preds = pd.merge(preds, res[['star_id', 'recovered_epoch']], on='star_id', how='left')
            
    return preds

def fold_lc(time, flux, period, epoch):
    phase = ((time - epoch) % period) / period
    phase = np.where(phase > 0.5, phase - 1.0, phase)
    sort_idx = np.argsort(phase)
    return phase[sort_idx], flux[sort_idx]

def get_binned_profile(phase, flux, num_bins=80):
    bins = np.linspace(-0.15, 0.15, num_bins)
    bin_centers = 0.5 * (bins[:-1] + bins[1:])
    binned_flux = []
    for i in range(len(bins)-1):
        mask = (phase >= bins[i]) & (phase < bins[i+1])
        binned_flux.append(np.nanmedian(flux[mask]) if np.any(mask) else 1.0)
    return bin_centers, np.array(binned_flux)

def main():
    # Header block
    st.markdown('<div class="main-title">🪐 ExoFinder</div>', unsafe_allow_html=True)
    st.markdown('<div class="subtitle">AI-Enabled Detection of Exoplanets from Stellar Light Curves</div>', unsafe_allow_html=True)
    
    preds_df = load_predictions()
    
    if preds_df is None:
        st.error("Predictions file not found! Please run the pipeline scripts first to generate predictions.")
        st.info("Run in terminal:\n`.venv\\Scripts\\python src/run_acquisition.py`\n`.venv\\Scripts\\python src/run_detrending.py`\n`.venv\\Scripts\\python src/run_period_search.py`\n`.venv\\Scripts\\python src/run_features.py`\n`.venv\\Scripts\\python src/run_classifier.py`")
        return
        
    # Sidebar selectors
    st.sidebar.markdown("### Navigation")
    page = st.sidebar.radio("Go to:", ["Dashboard Summary", "Stellar Inspector"])
    st.sidebar.markdown("---")
    
    # Filter selection
    st.sidebar.markdown("### Filters")
    filter_sig = st.sidebar.checkbox(
        "Only High-Significance Planets", 
        value=True, 
        help="Hides planet candidates with low SNR (< 5.0) or low model confidence (< 60%) from summary lists."
    )
    st.sidebar.markdown("---")
    
    st.sidebar.markdown("### Pipeline Status")
    st.sidebar.success(f"Active ({len(preds_df)} Stars Processed)")
    
    # Class count summary in Sidebar
    st.sidebar.markdown("### Dataset Summary")
    class_counts = preds_df['predicted_label'].value_counts()
    for class_name, count in class_counts.items():
        st.sidebar.markdown(f"**{class_name.replace('_', ' ').title()}**: {count}")
        

    
    if page == "Dashboard Summary":
        # ==========================================
        # PAGE 1: DASHBOARD SUMMARY
        # ==========================================
        total_stars = len(preds_df)
        cp_count = class_counts.get('confirmed_planet', 0)
        eb_count = class_counts.get('eclipsing_binary', 0)
        fp_count = class_counts.get('false_positive', 0)
        
        cp_pct = (cp_count / total_stars) * 100 if total_stars > 0 else 0
        eb_pct = (eb_count / total_stars) * 100 if total_stars > 0 else 0
        fp_pct = (fp_count / total_stars) * 100 if total_stars > 0 else 0
        
        # Grid of summary cards
        sum_col1, sum_col2, sum_col3, sum_col4 = st.columns(4)
        with sum_col1:
            st.html(textwrap.dedent(f"""
            <div class="card">
                <div class="metric-lbl">Total Processed</div>
                <div class="metric-val" style="color: #60a5fa;">{total_stars}</div>
            </div>
            """))
        with sum_col2:
            st.html(textwrap.dedent(f"""
            <div class="card">
                <div class="metric-lbl">Confirmed Planets</div>
                <div class="metric-val" style="color: #10b981;">{cp_count} <span style="font-size: 1.1rem; color: #a0aec0;">({cp_pct:.1f}%)</span></div>
            </div>
            """))
        with sum_col3:
            st.html(textwrap.dedent(f"""
            <div class="card">
                <div class="metric-lbl">Eclipsing Binaries</div>
                <div class="metric-val" style="color: #ec4899;">{eb_count} <span style="font-size: 1.1rem; color: #a0aec0;">({eb_pct:.1f}%)</span></div>
            </div>
            """))
        with sum_col4:
            st.html(textwrap.dedent(f"""
            <div class="card">
                <div class="metric-lbl">False Positives</div>
                <div class="metric-val" style="color: #ef4444;">{fp_count} <span style="font-size: 1.1rem; color: #a0aec0;">({fp_pct:.1f}%)</span></div>
            </div>
            """))
            
        # Two panels: Category Breakdown Progress bars & candidates list
        panel_col1, panel_col2 = st.columns([2, 3])
        
        with panel_col1:
            st.markdown("### Category Breakdown")
            st.html(textwrap.dedent(f"""
            <div class="card">
                <div style="margin-bottom: 1.25rem;">
                    <div style="display: flex; justify-content: space-between; font-weight: 600; color: #10b981; margin-bottom: 0.3rem; font-size: 0.95rem;">
                        <span>Confirmed Planets</span>
                        <span>{cp_count} stars ({cp_pct:.1f}%)</span>
                    </div>
                    <div style="background-color: #1f2937; height: 12px; border-radius: 6px; overflow: hidden;">
                        <div style="background-color: #10b981; width: {cp_pct}%; height: 100%;"></div>
                    </div>
                </div>
                <div style="margin-bottom: 1.25rem;">
                    <div style="display: flex; justify-content: space-between; font-weight: 600; color: #ec4899; margin-bottom: 0.3rem; font-size: 0.95rem;">
                        <span>Eclipsing Binaries (EBs)</span>
                        <span>{eb_count} stars ({eb_pct:.1f}%)</span>
                    </div>
                    <div style="background-color: #1f2937; height: 12px; border-radius: 6px; overflow: hidden;">
                        <div style="background-color: #ec4899; width: {eb_pct}%; height: 100%;"></div>
                    </div>
                </div>
                <div style="margin-bottom: 0.5rem;">
                    <div style="display: flex; justify-content: space-between; font-weight: 600; color: #ef4444; margin-bottom: 0.3rem; font-size: 0.95rem;">
                        <span>False Positives</span>
                        <span>{fp_count} stars ({fp_pct:.1f}%)</span>
                    </div>
                    <div style="background-color: #1f2937; height: 12px; border-radius: 6px; overflow: hidden;">
                        <div style="background-color: #ef4444; width: {fp_pct}%; height: 100%;"></div>
                    </div>
                </div>
            </div>
            """))
            
            # ISRO Readiness Note
            st.html(textwrap.dedent(f"""
            <div class="card" style="background: linear-gradient(135deg, rgba(31, 41, 55, 0.4) 0%, rgba(17, 24, 39, 0.7) 100%); border-left: 5px solid #6366f1;">
                <div style="display: flex; align-items: flex-start; gap: 0.8rem;">
                    <span style="font-size: 1.8rem; line-height: 1;">🚀</span>
                    <div>
                        <div style="font-size: 1.15rem; font-weight: 800; color: #818cf8; margin-bottom: 0.3rem;">Pipeline ready for ISRO dataset</div>
                        <div style="color: #9ca3af; font-size: 0.9rem; line-height: 1.4;">
                            ExoFinder has been regularized, calibrated, and scaled up. Ready to process high-volume exoplanet datasets for the ISRO Bharatiya Antariksh Hackathon 2026 (Problem Statement PS-07).
                        </div>
                    </div>
                </div>
            </div>
            """))
            
        with panel_col2:
            if filter_sig:
                st.markdown("### Top 5 High-Significance Planet Candidates")
                cp_candidates = preds_df[
                    (preds_df['predicted_label'] == 'confirmed_planet') & 
                    (preds_df['significance'] == 'high')
                ].copy()
            else:
                st.markdown("### Top 5 Confirmed Planet Candidates (All)")
                cp_candidates = preds_df[preds_df['predicted_label'] == 'confirmed_planet'].copy()
                
            top5 = cp_candidates.sort_values(by='confidence', ascending=False).head(5)
            
            if len(top5) == 0:
                st.info("No candidates matching the criteria found.")
            else:
                table_rows = ""
                for _, row in top5.iterrows():
                    tic_id = int(row['star_id'])
                    period = f"{row['bls_period']:.5f} days"
                    depth = f"{row['bls_depth']*100:.4f}% ({row['bls_depth']*1e6:.0f} ppm)"
                    conf_score = f"{row['confidence']*100:.1f}%"
                    status = row['significance'].replace('_', ' ').title()
                    status_badge = f'<span class="badge badge-{row["significance"]}">{status}</span>'
                    
                    table_rows += f"""
                    <tr>
                        <td style="font-weight: bold; color: #60a5fa;">TIC {tic_id}</td>
                        <td>{period}</td>
                        <td style="color: #10b981;">{depth}</td>
                        <td style="font-weight: bold; color: #f59e0b;">{conf_score}</td>
                        <td>{status_badge}</td>
                    </tr>
                    """
                
                table_html = textwrap.dedent(f"""
                <div class="card" style="padding: 1rem 1.5rem;">
                    <table class="summary-table">
                        <thead>
                            <tr>
                                <th>TIC ID</th>
                                <th>Period</th>
                                <th>Depth</th>
                                <th>Confidence</th>
                                <th>Significance</th>
                            </tr>
                        </thead>
                        <tbody>
                            {table_rows}
                        </tbody>
                    </table>
                </div>
                """)
                st.html(table_html)
                
    elif page == "Stellar Inspector":
        # ==========================================
        # PAGE 2: STELLAR INSPECTOR (ORIGINAL PAGE)
        # ==========================================
        st.sidebar.markdown("---")
        st.sidebar.markdown("### Target Selector")
        
        # List of processed targets
        star_ids = preds_df['star_id'].tolist()
        selected_id = st.sidebar.selectbox("Select TIC ID", star_ids, format_func=lambda x: f"TIC {x}")
        
        # Extract selected star record
        star_row = preds_df[preds_df['star_id'] == selected_id].iloc[0]
        
        true_label = star_row['label']
        pred_label = star_row['predicted_label']
        confidence = star_row['confidence']
        sig_status = star_row['significance']
        
        # Load light curve data
        raw_path = os.path.join(RAW_DIR, f"TIC_{selected_id}.csv")
        det_path = os.path.join(DETRENDED_DIR, f"TIC_{selected_id}.csv")
        
        if not (os.path.exists(raw_path) and os.path.exists(det_path)):
            st.error(f"Data files for TIC {selected_id} are missing. Please verify the raw and detrended directories.")
            st.stop()
            
        df_raw = pd.read_csv(raw_path)
        df_det = pd.read_csv(det_path)
        
        # Layout panels: Left column (1.5 width) for Prediction Metadata, Right column (3.5 width) for Tabs
        inspector_col1, inspector_col2 = st.columns([1.5, 3.5])
        
        # Panel 1: Target Prediction Card
        with inspector_col1:
            st.html(textwrap.dedent(f'<div class="card"><div class="metric-lbl">Target Star</div><div class="metric-val" style="color: #60a5fa;">TIC {selected_id}</div></div>'))
            
            # Display Prediction Badge
            badge_class = f"badge-{pred_label}"
            st.html(textwrap.dedent(f"""
            <div class="card">
                <div class="metric-lbl">Model Prediction</div>
                <div class="metric-val" style="margin-bottom: 0.5rem;">{pred_label.replace('_', ' ').title()}</div>
                <span class="badge {badge_class}">Classification</span>
            </div>
            """))
            
            # Confidence Score
            st.html(textwrap.dedent(f"""
            <div class="card">
                <div class="metric-lbl">Confidence Score</div>
                <div class="metric-val" style="color: #f59e0b;">{confidence*100:.1f}%</div>
                <div style="margin-top: 0.5rem; background-color: #374151; height: 8px; border-radius: 4px; overflow: hidden;">
                    <div style="background-color: #f59e0b; width: {confidence*100}%; height: 100%;"></div>
                </div>
            </div>
            """))
            

        with inspector_col2:
            tab_charts, tab_params, tab_shape = st.tabs([
                "📈 Light Curve Charts",
                "📋 Recovered Physical Parameters",
                "📊 Transit Shape Metrics"
            ])
            
            with tab_charts:
                plot_col1, plot_col2 = st.columns([1, 1])
                with plot_col1:
                    st.markdown("#### Raw vs. Detrended Light Curve")
                    fig, ax = plt.subplots(figsize=(8, 4.5))
                    # Raw flux plotted as background gray
                    raw_med = np.nanmedian(df_raw['flux'])
                    raw_med = raw_med if raw_med > 0 else 1.0
                    ax.plot(df_raw['time'], df_raw['flux'] / raw_med, color='#9ca3af', alpha=0.4, label='Raw Flux (Normalized)')
                    # Detrended flux plotted as bold blue
                    ax.plot(df_det['time'], df_det['detrended_flux'], color='#3b82f6', alpha=0.75, label='Detrended Flux')
                    # Trend line overplotted
                    ax.plot(df_det['time'], df_det['trend'] / raw_med, color='#ef4444', lw=1.5, linestyle='--', label='SG Polynomial Trend')
                    
                    ax.set_xlabel("Time (BTJD)")
                    ax.set_ylabel("Normalized Flux")
                    ax.legend()
                    ax.grid(True, alpha=0.2)
                    
                    # Apply space theme styling to plot
                    fig.patch.set_facecolor('#111827')
                    ax.set_facecolor('#1f2937')
                    ax.xaxis.label.set_color('#f3f4f6')
                    ax.yaxis.label.set_color('#f3f4f6')
                    ax.tick_params(colors='#f3f4f6')
                    for spine in ax.spines.values():
                        spine.set_color('#374151')
                        
                    st.pyplot(fig)
                    plt.close(fig)
                    
                with plot_col2:
                    st.markdown("#### Phase-Folded Light Curve")
                    # Calculate folded phase
                    period = star_row['bls_period']
                    epoch = star_row.get('recovered_epoch', 0.0)
                    if pd.isna(epoch):
                        epoch = 0.0
                    
                    if np.isnan(period) or period <= 0:
                        st.warning("Cannot fold light curve: BLS period search did not yield a valid period.")
                    else:
                        phase, flux_folded = fold_lc(df_det['time'].values, df_det['detrended_flux'].values, period, epoch)
                        bin_centers, binned_flux = get_binned_profile(phase, flux_folded)
                        
                        fig, ax = plt.subplots(figsize=(8, 4.5))
                        # Individual cadences plotted as scatter
                        ax.scatter(phase, flux_folded, color='#60a5fa', s=2.5, alpha=0.35, label='Individual Cadence')
                        # Binned median profile plotted in red
                        ax.plot(bin_centers, binned_flux, color='#ef4444', lw=3.0, label='Binned Median Profile')
                        
                        # Set limits zoom on transit window
                        ax.set_xlim(-0.25, 0.25)
                        min_val = np.nanmin(binned_flux) if np.any(~np.isnan(binned_flux)) else 0.998
                        min_y = min(min_val - 0.002, 0.998)
                        if not np.isfinite(min_y):
                            min_y = 0.998
                        ax.set_ylim(min_y, 1.002)
                        
                        ax.set_xlabel("Phase")
                        ax.set_ylabel("Detrended Normalized Flux")
                        ax.legend()
                        ax.grid(True, alpha=0.2)
                        
                        # Apply space theme styling
                        fig.patch.set_facecolor('#111827')
                        ax.set_facecolor('#1f2937')
                        ax.xaxis.label.set_color('#f3f4f6')
                        ax.yaxis.label.set_color('#f3f4f6')
                        ax.tick_params(colors='#f3f4f6')
                        for spine in ax.spines.values():
                            spine.set_color('#374151')
                            
                        st.pyplot(fig)
                        plt.close(fig)
            
            with tab_params:
                st.markdown("### Recovered Physical Parameters")
                param_grid = [
                    ("Recovered Period", f"{star_row['bls_period']:.6f} days"),
                    ("Transit Depth", f"{star_row['bls_depth']*100:.4f}% ({star_row['bls_depth']*1e6:.0f} ppm)"),
                    ("Transit Duration", f"{star_row['bls_duration']*24:.2f} hours ({star_row['bls_duration']:.4f} days)"),
                    ("Transit Signal-to-Noise Ratio (SNR)", f"{star_row['transit_snr']:.2f}"),
                    ("BLS Power Metric", f"{star_row['bls_power']:.2f}"),
                    ("Expected Catalog Period", f"{star_row['catalog_period']:.6f} days")
                ]
                
                for name, value in param_grid:
                    st.markdown(f"""
                    <div style="padding: 0.8rem; border-bottom: 1px solid #1f2937; display: flex; justify-content: space-between;">
                        <span style="color: #9ca3af;">{name}</span>
                        <span style="font-weight: bold; color: #f3f4f6;">{value}</span>
                    </div>
                    """, unsafe_allow_html=True)
                    
            with tab_shape:
                st.markdown("### Transit Shape Metrics")
                shape_grid = [
                    ("U-vs-V Symmetry Score (Mean-to-Max)", f"{star_row['mean_to_max_depth_ratio']:.3f}"),
                    ("In-Transit Skewness", f"{star_row['in_transit_skew']:.4f}"),
                    ("In-Transit Kurtosis", f"{star_row['in_transit_kurtosis']:.4f}"),
                    ("Out-of-Transit Scatter (Detrended)", f"{star_row['std_detrended']:.6f}"),
                    ("Raw Flux Variance (Stellar Noise)", f"{star_row['std_raw']:.6f}"),
                    ("Total Transits Spanned", f"{star_row['transit_count']:.1f}")
                ]
                
                for name, value in shape_grid:
                    st.markdown(f"""
                    <div style="padding: 0.8rem; border-bottom: 1px solid #1f2937; display: flex; justify-content: space-between;">
                        <span style="color: #9ca3af;">{name}</span>
                        <span style="font-weight: bold; color: #10b981;">{value}</span>
                    </div>
                    """, unsafe_allow_html=True)
            
            # Ground Truth & Low Significance Warning Cards below the tabs
            st.markdown("---")
            if pred_label == 'confirmed_planet' and sig_status == 'low_significance':
                bottom_col1, bottom_col2 = st.columns([1.5, 2.5])
                with bottom_col1:
                    sig_badge = f"badge-{sig_status}"
                    st.html(textwrap.dedent(f"""
                    <div class="card">
                        <div class="metric-lbl">Ground Truth</div>
                        <div style="font-size: 1.25rem; font-weight: 600; margin-bottom: 0.6rem; color: #f3f4f6;">{true_label.replace('_', ' ').title()}</div>
                        <div class="metric-lbl">Significance Status</div>
                        <div style="margin-top: 0.3rem;"><span class="badge {sig_badge}">{sig_status.replace('_', ' ').title()}</span></div>
                    </div>
                    """))
                with bottom_col2:
                    st.html(textwrap.dedent(f"""
                    <div class="card" style="border: 1px solid #ef4444; background-color: rgba(239, 68, 68, 0.05); padding: 1.2rem;">
                        <div style="font-weight: 800; color: #ef4444; font-size: 0.95rem; margin-bottom: 0.4rem; text-transform: uppercase; letter-spacing: 0.05em;">⚠️ Low Significance Warning</div>
                        <div style="font-size: 0.85rem; color: #cbd5e1; line-height: 1.45;">
                            Although the model predicts a planet transit shape with <b>{confidence*100:.1f}%</b> confidence, the transit Signal-to-Noise Ratio (<b>SNR = {star_row['transit_snr']:.2f}</b>) is below the threshold of <b>5.0</b>. This means the signal is weak and could be a random noise fluctuation.
                        </div>
                    </div>
                    """))
            else:
                sig_badge = f"badge-{sig_status}"
                st.html(textwrap.dedent(f"""
                <div class="card" style="max-width: 350px;">
                    <div class="metric-lbl">Ground Truth</div>
                    <div style="font-size: 1.25rem; font-weight: 600; margin-bottom: 0.6rem; color: #f3f4f6;">{true_label.replace('_', ' ').title()}</div>
                    <div class="metric-lbl">Significance Status</div>
                    <div style="margin-top: 0.3rem;"><span class="badge {sig_badge}">{sig_status.replace('_', ' ').title()}</span></div>
                </div>
                """))

if __name__ == "__main__":
    main()
