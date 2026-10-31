import sys
import os
import io
import time
import requests
import scipy.io.wavfile as wavfile

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

BASE_URL = "http://127.0.0.1:8001"
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

def fetch_api(path):
    try:
        res = requests.get(f"{BASE_URL}{path}", timeout=0.8)
        return res.status_code, res.headers, res.content, (res.json() if "application/json" in res.headers.get("content-type", "") else None)
    except Exception:
        client = get_test_client()
        res = client.get(path)
        return res.status_code, res.headers, res.content, (res.json() if "application/json" in res.headers.get("content-type", "") else None)

def test_audio_metadata():
    star_ids = [77202722, 100100827, 240613164]
    for sid in star_ids:
        t0 = time.perf_counter()
        status, headers, content, data = fetch_api(f"/api/star/{sid}/audio?format=json&base_freq=440.0&duration=4.0&contrast=1.5")
        latency = (time.perf_counter() - t0) * 1000
        assert status == 200, f"Expected 200, got {status}"
        assert data["star_id"] == sid
        assert data["base_frequency_hz"] == 440.0
        assert data["minimum_frequency_hz"] < 440.0, "Expected pitch drop during transit"
        assert data["max_pitch_drop_hz"] > 0, "Pitch drop should be positive"
        assert "audio_download_url" in data
        print(f"✓ Audio Metadata for TIC {sid}: Pitch drop -{data['max_pitch_drop_hz']} Hz ({latency:.2f}ms)")

def test_audio_wav_stream():
    star_id = 77202722
    t0 = time.perf_counter()
    status, headers, content, _ = fetch_api(f"/api/star/{star_id}/audio?format=wav&base_freq=440.0&duration=3.0&contrast=1.0")
    latency = (time.perf_counter() - t0) * 1000
    assert status == 200
    assert headers["content-type"] == "audio/wav"
    assert "attachment" in headers.get("content-disposition", "")
    
    # Parse WAV file in memory
    sample_rate, audio_data = wavfile.read(io.BytesIO(content))
    assert sample_rate == 44100
    assert audio_data.ndim == 2, "Expected stereo audio (2 channels)"
    assert audio_data.shape[1] == 2
    assert len(audio_data) == int(3.0 * 44100)
    print(f"✓ Audio WAV Binary: {len(content)} bytes, {sample_rate} Hz stereo, valid PCM ({latency:.2f}ms)")

if __name__ == "__main__":
    print("Testing ExoFinder Audio Sonification Pipeline...")
    test_audio_metadata()
    test_audio_wav_stream()
    print("All Audio Tests Passed Successfully!")
