import sys
import os
import time
import json
import urllib.request
import urllib.error

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

_client = None

def get_test_client():
    global _client
    if _client is None:
        from fastapi.testclient import TestClient
        from src.server import app, load_data_and_models, state
        if not state.get("target_dict"):
            load_data_and_models()
        _client = TestClient(app)
    return _client

def test_endpoint(name, url, method='GET', data=None):
    start = time.perf_counter()
    path = url.replace("http://127.0.0.1:8001", "")
    
    # Try live server first; fall back to in-process TestClient if not running
    try:
        req = urllib.request.Request(url, method=method)
        if data:
            req.add_header('Content-Type', 'application/json')
            body = json.dumps(data).encode('utf-8')
        else:
            body = None
        with urllib.request.urlopen(req, data=body, timeout=0.8) as resp:
            duration_ms = (time.perf_counter() - start) * 1000
            res = json.loads(resp.read().decode('utf-8'))
            print(f"[{name}] HTTP {resp.status} (Live) in {duration_ms:.1f}ms")
            return res
    except Exception:
        client = get_test_client()
        if method == 'GET':
            resp = client.get(path)
        else:
            resp = client.post(path, json=data)
        duration_ms = (time.perf_counter() - start) * 1000
        assert resp.status_code == 200, f"Expected 200, got {resp.status_code}: {resp.text}"
        print(f"[{name}] HTTP {resp.status_code} (In-Process) in {duration_ms:.1f}ms")
        return resp.json()

def run_all_api_tests():
    print("=== EXOFINDER NOVA BACKEND VERIFICATION SUITE ===\n")
    
    # 1. Health
    h = test_endpoint('Health Check', 'http://127.0.0.1:8001/api/health')
    print("  Status:", h.get("status"), "| Indexed Targets:", h.get("targets_indexed"))
    assert h.get("status") == "online"
    assert h.get("targets_indexed") > 0

    # 2. Targets List
    t = test_endpoint('Targets List', 'http://127.0.0.1:8001/api/targets?page=1&page_size=5')
    print(f"  Total targets: {t.get('total')}, Page count: {len(t.get('targets', []))}")
    assert t.get("total") > 0
    assert len(t.get("targets", [])) == 5

    # 3. Target Details
    d = test_endpoint('Target Details', 'http://127.0.0.1:8001/api/target/77202722')
    print(f"  Target TIC 77202722 label: {d.get('ground_truth')}, SNR: {d.get('transit_snr')}")
    assert d.get("tic_id") == "TIC 77202722"
    assert "vetting_diagnostics" in d
    assert "model_comparison" in d

    # 4. Lightcurve
    lc = test_endpoint('Lightcurve Data', 'http://127.0.0.1:8001/api/lightcurve/77202722?max_points=500')
    ts_pts = len(lc["time_series"]["time"])
    fold_pts = len(lc["phase_folded"]["phase"])
    g_bins = len(lc["cnn_views"]["global_view"])
    l_bins = len(lc["cnn_views"]["local_view"])
    print(f"  Time series points: {ts_pts}, Folded points: {fold_pts}")
    print(f"  CNN Global view bins: {g_bins}, Local view bins: {l_bins}")
    assert ts_pts > 0 and fold_pts > 0
    assert g_bins == 201 and l_bins == 61

    # 5. Classify
    clf = test_endpoint('Classify POST', 'http://127.0.0.1:8001/api/classify', method='POST', data={
        'star_id': 77202722,
        'transit_snr': 12.5,
        'mean_to_max_depth_ratio': 0.85
    })
    print(f"  Predicted Label: {clf['predicted_label']} ({clf['confidence']*100:.1f}%)")
    assert "predicted_label" in clf
    assert "probabilities" in clf
    assert "vetting_status" in clf

    # 6. Transit Model
    tm = test_endpoint('Transit Model', 'http://127.0.0.1:8001/api/transit-model?period=3.35&radius_ratio=0.1')
    print(f"  Theoretical Model pts: {len(tm['phase'])}, Depth: {tm['transit_depth_ppm']} ppm")
    assert len(tm["phase"]) == 200

    print("\n>>> ALL 6 API ENDPOINTS PASSED VERIFICATION WITH 100% SUCCESS! <<<")

def test_api_endpoints():
    run_all_api_tests()

if __name__ == '__main__':
    run_all_api_tests()
