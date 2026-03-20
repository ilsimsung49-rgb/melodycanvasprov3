import os
import uuid
import numpy as np
import librosa
import soundfile as sf
import traceback
from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS

# Get the absolute path to the frontend directory
FRONTEND_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'frontend'))

app = Flask(__name__, static_folder=FRONTEND_DIR)
CORS(app)

UPLOAD_FOLDER = 'uploads'
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

# Global memory to store analyzed data
data_store = {}

def get_musical_key(y, sr):
    try:
        # Use stft instead of cqt for better stability and lower memory usage
        chroma = librosa.feature.chroma_stft(y=y, sr=sr)
        chroma_sum = np.sum(chroma, axis=1)
        keys = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']
        major_profile = np.array([6.35, 2.23, 3.48, 2.33, 4.38, 4.09, 2.52, 5.19, 2.39, 3.66, 2.29, 2.88])
        best_corr = -1
        best_key = 0
        for i in range(12):
            corr = np.corrcoef(np.roll(chroma_sum, -i), major_profile)[0, 1]
            if corr > best_corr:
                best_corr = corr
                best_key = i
        return {"root": best_key, "root_name": keys[best_key], "mode": "major", "key_str": keys[best_key]}
    except:
        return {"root": 0, "root_name": "C", "mode": "major", "key_str": "C"}

# ----------------- API ROUTES (Defined FIRST to avoid conflict) -----------------

@app.route('/api/health', methods=['GET'])
def health():
    return jsonify({"status": "healthy", "version": "v3.1.0"})

@app.route('/api/analyze', methods=['POST'])
def analyze():
    try:
        if 'file' not in request.files:
            return jsonify({"error": "No file uploaded"}), 400
        
        file = request.files['file']
        file_id = str(uuid.uuid4())[:8]
        # Sanitize filename for security
        safe_filename = "".join([c for c in file.filename if c.isalnum() or c in "._-"])
        filepath = os.path.join(UPLOAD_FOLDER, f"{file_id}_{safe_filename}")
        file.save(filepath)
        
        print(f"[*] Analyzing file (Stage 1: Load): {filepath}")
        # Use duration limit and mono for faster/safer analysis
        try:
            y, sr = librosa.load(filepath, sr=22050, duration=120, mono=True)
        except Exception as load_error:
            print(f"[!] Load error: {load_error}")
            return jsonify({"error": f"Audio load failed: {str(load_error)}"}), 500

        print(f"[*] Analyzing file (Stage 2: Beat Tracking)")
        try:
            tempo_raw, beats = librosa.beat.beat_track(y=y, sr=sr)
            # Handle librosa 0.10+ tempo return type (can be array or float)
            tempo = float(np.mean(tempo_raw)) if isinstance(tempo_raw, np.ndarray) else float(tempo_raw)
        except Exception as beat_error:
            print(f"[!] Beat error: {beat_error}")
            tempo, beats = 120.0, np.array([]) # Default on failure

        print(f"[*] Analyzing file (Stage 3: Key Detection)")
        key_info = get_musical_key(y, sr)
        
        data_store[file_id] = {"path": filepath, "y": y, "sr": sr}
        
        print(f"[*] Analysis Success: {file_id}")
        return jsonify({
            "file_id": file_id,
            "filename": file.filename,
            "beat": {"bpm": round(tempo), "beats": beats.tolist()},
            "key": key_info,
            "chords": []
        })
    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500

@app.route('/api/extract_melody', methods=['POST'])
def extract_melody():
    try:
        data = request.json
        fid = data.get('file_id')
        if not fid or fid not in data_store:
            return jsonify({"error": "File not found"}), 404
            
        y_orig = data_store[fid]["y"]
        sr_orig = data_store[fid]["sr"]
        
        # Stability: 1. Resample to 22050 for lighter processing
        # Limit to first 180 seconds to avoid memory crash
        max_samples = 180 * 22050
        y = librosa.resample(y_orig[:180 * sr_orig], orig_sr=sr_orig, target_sr=22050)
        sr = 22050
        
        # 1. Pitch Tracking (Efficient Hop)
        hop = 512
        pitches, magnitudes = librosa.piptrack(y=y, sr=sr, hop_length=hop, n_fft=2048)
        
        # 2. Onset Detection (Finding where notes start)
        onset_env = librosa.onset.onset_strength(y=y, sr=sr)
        onsets = librosa.onset.onset_detect(onset_envelope=onset_env, sr=sr, hop_length=hop)
        onset_times = librosa.frames_to_time(onsets, sr=sr, hop_length=hop)
        
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
                            melody.append({
                                't': float(t_start), 
                                'dur': float(t_end - t_start),
                                'pitch': float(pitch)
                            })
        
        return jsonify({"melody": melody})
    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500

@app.route('/api/build_score', methods=['POST'])
def build_score():
    try:
        data = request.json
        if not data:
            return jsonify({"error": "No JSON payload received"}), 400
            
        title = data.get('title') or 'Lead Sheet Title'
        composer = data.get('composer') or 'Leon Kerner'
        style = data.get('style') or 'Medium Swing'
        beat_data = data.get('beat') or {}
        bpm = beat_data.get('bpm', 120)
        key_data = data.get('key') or {}
        key_str = key_data.get('key_str') or 'C'
        sections = data.get('sections', [])
        melody_data = list(data.get('melody', []))

        # ---------------------------------------------------------------------
        # ABC Header (Professional Lead Sheet Formatting)
        # ---------------------------------------------------------------------
        abc_header = [
            f"X:1",
            f"T:{title}",
            f"C:{composer}",
            f"M:4/4",
            f"L:1/4", # Use quarter notes as the default length for cleaner look
            f"Q:1/4={bpm}",
            f"K:{key_str}",
            "%%titlefont Outfit 25",
            "%%composerfont Outfit 14",
            "%%vocalfont Inter 13",
            "%%wordsfont Inter 13",
            "%%staffwidth 700",
            "%%measurenb 0",
            f"P: {style}"
        ]

        # Helper: Pitch to ABC with comfortable vocal range
        def pitch_to_abc(p):
            if p <= 0: return "z"
            # Constrain to comfortable vocal range (Middle C 60 to approx G5 79)
            midi = int(round(12 * np.log2(p / 440.0) + 69))
            midi = max(60, min(midi, 79)) 
            
            notes = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']
            n, oct = notes[midi % 12], (midi // 12) - 5 # Shifted one octave for treble preference
            if oct == 0: return n
            elif oct == 1: return n.lower()
            elif oct > 1: return n.lower() + "'" * (oct - 1)
            else: return n + "," * abs(oct)

        lines, bars_in_line = [], 0

        # Syllabification Helper (Exclude [...] labels)
        import re
        def syllabify(text):
            # 1. Remove [Verse 1] style labels for mapping to notes
            text = re.sub(r'\[.*?\]', '', text)
            tokens = []
            for char in text:
                if char.strip():
                    tokens.append(char)
            return tokens

        # ---------------------------------------------------------------------
        # Generating Music Lines with Rhythmic Variety (Eighth Note Grid)
        # ---------------------------------------------------------------------
        # Resolution: Eighth notes (8 slots per bar)
        half_beat_dur = 60.0 / bpm / 2.0
        current_time_offset = 0.0

        for sec in sections:
            raw_lyric = sec.get('lyrics', '')
            lyric_tokens = syllabify(raw_lyric)
            token_ptr = 0
            
            abc_line_notes, abc_line_lyrics = [], []
            bars_to_gen = int(sec.get('bars', 4))
            section_label = f'^"{sec.get("name", "Section")}"'

            for b in range(bars_to_gen):
                bar_notes, bar_lyrics = [], []
                chord = "Cmaj7" if b % 2 == 0 else "Am7" 
                
                # Starting label and chord
                if b == 0:
                    bar_notes.append(f'{section_label}"{chord}"')
                else:
                    bar_notes.append(f'"{chord}"')
                
                # Iterate through 8 half-beat slots
                for h in range(8):
                    t_slot = current_time_offset
                    
                    # Find note closest to this slot
                    note_found = None
                    if melody_data:
                        # Find potential note starts within this 8th note window
                        candidates = [m for m in melody_data if abs(m.get('t', 0) - t_slot) < half_beat_dur]
                        if candidates:
                            # Pick the one closest to the start of the slot
                            note_found = min(candidates, key=lambda x: abs(x.get('t', 0) - t_slot))
                    
                    if note_found:
                        note = pitch_to_abc(float(note_found.get('pitch', 0)))
                        # Duration in ABC: /2 for 8th note, 1 for 4th note
                        # For simplicity, we make them 8th notes to allow syncopation
                        note = note + "/2"
                    else:
                        note = "z/2" # Eighth rest
                    
                    bar_notes.append(note)
                    current_time_offset += half_beat_dur
                    
                    # Mapping lyrics ONLY when a non-rest note is present
                    if note != "z/2" and token_ptr < len(lyric_tokens):
                        abc_line_lyrics.append(lyric_tokens[token_ptr])
                        token_ptr += 1
                    elif note == "z/2":
                        pass # Rest - no lyric mapping
                    else:
                        abc_line_lyrics.append("*") 

                abc_line_notes.append("".join(bar_notes))
                bars_in_line += 1
                
                if bars_in_line >= 4:
                    lines.append(" | ".join(abc_line_notes) + " |")
                    lines.append("w: " + " ".join(abc_line_lyrics))
                    abc_line_notes, abc_line_lyrics = [], []
                    bars_in_line = 0

        # Remaining cleanup
        if abc_line_notes:
            lines.append(" | ".join(abc_line_notes) + " |")
            lines.append("w: " + " ".join(abc_line_lyrics))

        full_abc = "\n".join(abc_header) + "\n" + "\n".join(lines)
        return jsonify({"abc": full_abc, "xml": ""})
        
    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500

# ----------------- STATIC ROUTES (Defined LAST) -----------------

@app.route('/')
def index():
    return send_from_directory(FRONTEND_DIR, 'index.html')

@app.route('/<path:path>')
def serve_static(path):
    if path.startswith('api/'): # Security check to let API pass through if needed
        return jsonify({"error": "API route not found"}), 404
    return send_from_directory(FRONTEND_DIR, path)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5050, debug=True)
