import os
import uuid
import numpy as np
import librosa
import soundfile as sf
import gc
import traceback
from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

UPLOAD_FOLDER = 'uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Shared state (In-memory, clears on restart)
data_store = {}

def get_musical_key(y, sr):
    """Memory-efficient musical key detection."""
    try:
        # Use a large hop length to save RAM
        chroma = librosa.feature.chroma_cqt(y=y, sr=sr, hop_length=1024)
        mean_chroma = np.mean(chroma, axis=1)
        
        keys = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']
        # Simple profiles
        major_profile = np.array([6.35, 2.23, 3.48, 2.33, 4.38, 4.09, 2.52, 5.19, 2.39, 3.66, 2.29, 2.88])
        minor_profile = np.array([6.33, 2.68, 3.52, 5.38, 2.60, 3.53, 2.54, 4.75, 3.98, 2.69, 3.34, 3.17])
        
        major_corrs = [np.corrcoef(mean_chroma, np.roll(major_profile, i))[0, 1] for i in range(12)]
        minor_corrs = [np.corrcoef(mean_chroma, np.roll(minor_profile, i))[0, 1] for i in range(12)]
        
        maj_idx = np.argmax(major_corrs)
        min_idx = np.argmax(minor_corrs)
        
        if major_corrs[maj_idx] > minor_corrs[min_idx]:
            return {"key": keys[maj_idx], "scale": "major"}
        else:
            return {"key": keys[min_idx], "scale": "minor"}
    except:
        return {"key": "C", "scale": "major"}
    finally:
        if 'chroma' in locals(): del chroma
        gc.collect()

@app.route('/api/analyze', methods=['POST'])
def analyze():
    try:
        if 'file' not in request.files:
            return jsonify({"error": "No file uploaded"}), 400
        
        file = request.files['file']
        file_id = str(uuid.uuid4())[:8]
        safe_filename = "".join([c for c in file.filename if c.isalnum() or c in "._-"])
        filepath = os.path.join(UPLOAD_FOLDER, f"{file_id}_{safe_filename}")
        file.save(filepath)
        
        print(f"[*] Stage 1 (Load): {filepath}")
        # Use low SR (11025) for initial metadata check to save RAM
        y, sr = librosa.load(filepath, sr=11025, duration=120, mono=True)
        gc.collect()

        print(f"[*] Stage 2 (Beats)")
        tempo_raw, beats = librosa.beat.beat_track(y=y, sr=sr)
        tempo = float(np.mean(tempo_raw)) if isinstance(tempo_raw, np.ndarray) else float(tempo_raw)
        gc.collect()

        print(f"[*] Stage 3 (Key)")
        key_info = get_musical_key(y, sr)
        
        # CLEAR RAM IMMEDIATELY - only store path
        del y
        gc.collect()
        
        data_store[file_id] = {"path": filepath}
        
        return jsonify({
            "file_id": file_id,
            "filename": file.filename,
            "beat": {"bpm": round(tempo), "beats": beats.tolist()},
            "key": key_info,
            "chords": []
        })
    except Exception as e:
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500

@app.route('/api/extract_melody', methods=['POST'])
def extract_melody():
    try:
        data = request.json
        fid = data.get('file_id')
        if not fid or fid not in data_store:
            return jsonify({"error": "File not found"}), 404
            
        audio_path = data_store[fid]["path"]
        
        print(f"[*] Extraction Load: {audio_path}")
        # Use 16000 for melody analysis (good balance of precision and RAM)
        y, sr = librosa.load(audio_path, sr=16000, duration=180.0)
        gc.collect()
        
        hop = 512
        pitches, magnitudes = librosa.piptrack(y=y, sr=sr, hop_length=hop, fmin=75, fmax=1000)
        gc.collect()
        
        onset_env = librosa.onset.onset_strength(y=y, sr=sr)
        onsets = librosa.onset.onset_detect(onset_envelope=onset_env, sr=sr, hop_length=hop)
        onset_times = librosa.frames_to_time(onsets, sr=sr, hop_length=hop)
        gc.collect()
        
        melody = []
        max_mag = np.max(magnitudes) if magnitudes.size > 0 else 1.0
        threshold = max_mag * 0.1 

        if len(onset_times) > 0:
            for i in range(len(onset_times)):
                t_start = onset_times[i]
                t_end = onset_times[i+1] if i+1 < len(onset_times) else t_start + 0.5
                center_frame = librosa.time_to_frames((t_start + t_end)/2, sr=sr, hop_length=hop)
                if center_frame < pitches.shape[1]:
                    idx = magnitudes[:, center_frame].argmax()
                    mag = magnitudes[idx, center_frame]
                    if mag > threshold:
                        pitch = pitches[idx, center_frame]
                        if pitch > 50:
                            melody.append({'t': float(t_start), 'dur': float(t_end - t_start), 'pitch': float(pitch)})
        
        # Cleanup
        del y, pitches, magnitudes, onset_env
        gc.collect()
        
        return jsonify({"melody": melody})
    except Exception as e:
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500

def pitch_to_abc(pitch, key_info):
    if pitch <= 0: return ""
    midi = int(round(librosa.hz_to_midi(pitch)))
    # Vocal range 60-79 (C4 to G5)
    midi = max(60, min(79, midi))
    
    notes_map = {0: "C", 1: "_D", 2: "D", 3: "_E", 4: "E", 5: "F", 6: "_G", 7: "G", 8: "_A", 9: "A", 10: "_B", 11: "B"}
    octave = (midi // 12) - 4
    note_name = notes_map[midi % 12]
    
    if octave == 0: return note_name
    if octave == 1: return note_name.lower()
    if octave == 2: return note_name.lower() + "'"
    if octave < 0: return note_name + ("," * abs(octave))
    return note_name

@app.route('/api/build_score', methods=['POST'])
def build_score():
    try:
        data = request.json
        melody_data = data.get('melody', [])
        bpm = data.get('bpm', 120)
        key_info = data.get('key', {"key": "C", "scale": "major"})
        lyrics = data.get('lyrics', "")
        
        # Syllabification (Basic)
        import re
        lines = lyrics.split('\n')
        sections = []
        current_section = None
        
        for line in lines:
            line = line.strip()
            if not line: continue
            if line.startswith('[') and line.endswith(']'):
                current_section = line[1:-1]
                continue
            
            # Clean and split Korean syllables
            chars = [c for c in line if c.strip()]
            if chars:
                sections.append({"name": current_section or "Verse", "lyrics": chars})

        # ABC Construction
        key_char = key_info['key']
        if key_info['scale'] == 'minor': key_char += "m"
        
        abc = f"X:1\nT:Melody Canvas Score\nM:4/4\nL:1/8\nQ:1/4={int(bpm)}\nK:{key_char}\n"
        
        # Grid settings (8th note resolution)
        sec_per_beat = 60.0 / bpm
        sec_per_8th = sec_per_beat / 2
        
        lyric_idx = 0
        total_lyrics = []
        for s in sections: total_lyrics.extend(s['lyrics'])
        
        current_time = 0.0
        measure_8ths = 0
        
        # Group melody into 8th note slots
        for note in melody_data:
            start_8th = round(note['t'] / sec_per_8th)
            # Add rests
            while current_time < start_8th * sec_per_8th:
                abc += "z"
                measure_8ths += 1
                if measure_8ths >= 8:
                    abc += " | "
                    measure_8ths = 0
                current_time += sec_per_8th
            
            # Add note
            abc_note = pitch_to_abc(note['pitch'], key_info)
            dur_8ths = max(1, round(note['dur'] / sec_per_8th))
            
            abc += abc_note
            if dur_8ths > 1: abc += str(dur_8ths)
            
            # Add lyric if available
            if lyric_idx < len(total_lyrics):
                # Using inline lyrics for ABCJS
                abc += f'w: {total_lyrics[lyric_idx]}\n'
                lyric_idx += 1
            
            measure_8ths += dur_8ths
            while measure_8ths >= 8:
                abc += " | "
                measure_8ths -= 8
            current_time += dur_8ths * sec_per_8th

        return jsonify({"abc": abc})
    except Exception as e:
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
