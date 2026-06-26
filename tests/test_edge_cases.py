import os
import sys
import unittest
import numpy as np
import pandas as pd

# Add the project root to sys.path to resolve src imports
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.detrend import detrend_light_curve
from src.period_search import find_transit_period, fold_light_curve

class TestPipelineEdgeCases(unittest.TestCase):
    
    # ----------------------------------------------------
    # 1. EMPTY DATAFRAME / INPUT TESTS
    # ----------------------------------------------------
    
    def test_detrend_empty_df(self):
        """Verify detrend_light_curve handles empty DataFrames gracefully."""
        df_empty = pd.DataFrame()
        result = detrend_light_curve(df_empty)
        
        self.assertIsInstance(result, pd.DataFrame)
        self.assertTrue(result.empty)
        # Verify default columns exist in returned empty DataFrame
        for col in ['time', 'flux', 'trend', 'detrended_flux']:
            self.assertIn(col, result.columns)
            
    def test_detrend_none_input(self):
        """Verify detrend_light_curve handles None input gracefully."""
        result = detrend_light_curve(None)
        self.assertIsInstance(result, pd.DataFrame)
        self.assertTrue(result.empty)
        for col in ['time', 'flux', 'trend', 'detrended_flux']:
            self.assertIn(col, result.columns)
            
    def test_period_search_empty_input(self):
        """Verify find_transit_period handles empty array inputs gracefully."""
        t_empty = np.array([])
        y_empty = np.array([])
        
        result = find_transit_period(t_empty, y_empty)
        self.assertIsInstance(result, dict)
        self.assertTrue(np.isnan(result['period']))
        self.assertTrue(np.isnan(result['epoch']))
        self.assertTrue(np.isnan(result['depth']))
        self.assertTrue(np.isnan(result['duration']))
        self.assertTrue(np.isnan(result['power']))
        self.assertIsNone(result['periodogram'])

    # ----------------------------------------------------
    # 2. DATASETS WITH NaNs IN TIME OR FLUX
    # ----------------------------------------------------
    
    def test_detrend_nan_inputs_partial(self):
        """Verify detrend_light_curve drops NaN values and detrends the rest."""
        # Create a df with some NaNs
        np.random.seed(42)
        n_points = 200
        time = np.linspace(0, 10, n_points)
        flux = 1.0 + 0.01 * np.random.randn(n_points)
        
        # Inject NaNs
        time[10] = np.nan
        flux[50] = np.nan
        
        df = pd.DataFrame({'time': time, 'flux': flux})
        result = detrend_light_curve(df, window_length=51, polyorder=2)
        
        # We expect the 2 rows with NaNs to be dropped, leaving 198 rows
        self.assertEqual(len(result), 198)
        self.assertFalse(result['time'].isna().any())
        self.assertFalse(result['flux'].isna().any())
        self.assertIn('trend', result.columns)
        self.assertIn('detrended_flux', result.columns)
        
    def test_detrend_nan_inputs_all(self):
        """Verify detrend_light_curve does not crash when all values are NaN."""
        df_all_nan = pd.DataFrame({
            'time': [np.nan, np.nan, np.nan],
            'flux': [np.nan, np.nan, np.nan]
        })
        
        # Should drop all rows and return empty dataframe without crashing
        result = detrend_light_curve(df_all_nan)
        self.assertTrue(result.empty)
        
    def test_period_search_nan_inputs_partial(self):
        """Verify find_transit_period filters NaNs and executes BLS."""
        np.random.seed(42)
        n_points = 200
        time = np.linspace(0, 10, n_points)
        flux = 1.0 + 0.001 * np.random.randn(n_points)
        
        # Inject NaNs
        time[5] = np.nan
        flux[15] = np.nan
        
        # Result should run BLS on remaining valid points
        result = find_transit_period(time, flux, min_period=1.0, max_period=5.0)
        
        self.assertIsInstance(result, dict)
        # Since we have > 100 valid points, BLS should execute and return valid numbers
        self.assertFalse(np.isnan(result['period']))
        self.assertFalse(np.isnan(result['power']))
        self.assertIsNotNone(result['periodogram'])

    def test_period_search_nan_inputs_all(self):
        """Verify find_transit_period returns NaNs when all inputs are NaN."""
        time = np.full(150, np.nan)
        flux = np.full(150, np.nan)
        
        result = find_transit_period(time, flux)
        self.assertIsInstance(result, dict)
        self.assertTrue(np.isnan(result['period']))
        self.assertIsNone(result['periodogram'])

    # ----------------------------------------------------
    # 3. PURE GAUSSIAN NOISE (NO TRANSITS)
    # ----------------------------------------------------
    
    def test_gaussian_noise_pipeline(self):
        """Verify both detrending and period search run successfully on pure Gaussian noise."""
        np.random.seed(42)
        n_points = 500
        time = np.linspace(0, 20, n_points)
        # Generate pure Gaussian noise around 1.0
        flux = 1.0 + 0.005 * np.random.randn(n_points)
        
        df = pd.DataFrame({'time': time, 'flux': flux})
        
        # 1. Run detrending
        detrended_df = detrend_light_curve(df, window_length=101, polyorder=2)
        self.assertEqual(len(detrended_df), n_points)
        self.assertFalse(detrended_df['detrended_flux'].isna().any())
        
        # 2. Run BLS period search on detrended light curve
        t_arr = detrended_df['time'].values
        f_arr = detrended_df['detrended_flux'].values
        
        bls_res = find_transit_period(t_arr, f_arr, min_period=0.5, max_period=10.0)
        
        self.assertIsInstance(bls_res, dict)
        self.assertFalse(np.isnan(bls_res['period']))
        self.assertFalse(np.isnan(bls_res['epoch']))
        self.assertIsNotNone(bls_res['periodogram'])
        
        # 3. Verify fold_light_curve also doesn't crash on Gaussian noise results
        phases, folded_flux = fold_light_curve(t_arr, f_arr, bls_res['period'], bls_res['epoch'])
        self.assertEqual(len(phases), n_points)
        self.assertEqual(len(folded_flux), n_points)
        # Phases should be sorted between -0.5 and 0.5
        self.assertTrue(np.all(phases >= -0.5))
        self.assertTrue(np.all(phases <= 0.5))
        self.assertTrue(np.all(np.diff(phases) >= 0.0))  # check if sorted

if __name__ == "__main__":
    unittest.main()
