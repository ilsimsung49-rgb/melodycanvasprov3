import requests
import json
import time

url_base = 'http://localhost:5050/api'
file_id = '053d67be'
filename = 'mp3'

sections = [
    {"name": "Intro", "type": "inst", "bars": 8},
    {"name": "Verse 1", "type": "vocal", "bars": 16, "lyrics": "오지 않은 그날을 이미 온 것처럼 사네"},
    {"name": "Pre-Chorus", "type": "vocal", "bars": 8, "lyrics": "세상이 아무리 거칠게 나를 밀어내도"},
    {"name": "Chorus", "type": "vocal", "bars": 16, "lyrics": "나의 꿈은 현실이 되겠지"}
]

print("1. Getting health...")
try:
    r = requests.get(f"{url_base}/health")
    print(r.json())
except Exception as e:
    print(e)
    exit(1)

print("2. Extracting melody (t_start=0, t_end=30) for fast test...")
payload = {
    "file_id": file_id,
    "filename": filename,
    "t_start": 0,
    "t_end": 30
}
r = requests.post(f"{url_base}/extract_melody", json=payload)
data = r.json()
if 'error' in data:
    print("Error:", data['error'])
    exit(1)

print("3. Building score...")
build_payload = {
    "title": "Test Score",
    "file_id": file_id,
    "beat": data['beat'],
    "key": {"root": 0, "root_name": "C", "mode": "major", "key_str": "C"},
    "chords": [{"bar": i, "t": i*2.5, "chord": "C"} for i in range(100)],
    "melody": data['melody'],
    "sections": sections
}
r2 = requests.post(f"{url_base}/build_score", json=build_payload)
abc = r2.json().get('abc', '')
print("--- ABC OUTPUT ---")
print('\n'.join(abc.split('\n')[:30]))
