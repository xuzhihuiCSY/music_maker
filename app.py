import torch
from flask import Flask, request, jsonify
from transformers import pipeline, AutoProcessor, MusicgenForConditionalGeneration
import scipy.io.wavfile
import numpy as np
import soundfile as sf
from pydub import AudioSegment
import os
from flask_cors import CORS
import base64
from TTS.api import TTS
import parselmouth
import pyworld
import librosa
from music21 import stream, note

app = Flask(__name__)
CORS(app)

processor_music = AutoProcessor.from_pretrained("facebook/musicgen-small")
music_model = MusicgenForConditionalGeneration.from_pretrained("facebook/musicgen-small", attn_implementation="eager")

generator = pipeline("text-generation", model="EleutherAI/gpt-neo-1.3B", device=-1)

tts_model = "tts_models/en/ljspeech/glow-tts"
tts = TTS(model_name=tts_model)

def detect_key_naive(music_path):
    y, sr = librosa.load(music_path, sr=None)
    chroma = librosa.feature.chroma_cqt(y=y, sr=sr)
    chroma_sum = chroma.sum(axis=1)

    note_names = ['C', 'C#', 'D', 'D#', 'E', 'F',
                  'F#', 'G', 'G#', 'A', 'A#', 'B']
    most_common_note_index = int(np.argmax(chroma_sum))
    most_common_note = note_names[most_common_note_index]

    melody = stream.Stream()
    for _ in range(8):
        melody.append(note.Note(most_common_note, quarterLength=1))

    detected_key = melody.analyze('key')
    print(f"[INFO] Detected key: {detected_key}")

    scale_notes = [n.frequency for n in detected_key.pitches]
    return scale_notes

def autotune_audio(input_wav_path, output_wav_path, scale_notes=None, sr=32000):
    if scale_notes is None:
        scale_notes = [130.81, 146.83, 164.81, 174.61, 196.00, 220.00, 246.94,
                       261.63, 293.66, 329.63, 349.23, 392.00, 440.00, 493.88,
                       523.25]

    y, actual_sr = sf.read(input_wav_path)
    if actual_sr != sr:
        print(f"[INFO] Resampling from {actual_sr} to {sr}")
        y = librosa.resample(y.T, orig_sr=actual_sr, target_sr=sr).T

    if y.ndim > 1:
        y = y[:, 0]
    y = y.astype(np.float64)

    snd = parselmouth.Sound(y, sr)
    pitch_obj = snd.to_pitch(time_step=0.01, pitch_floor=50, pitch_ceiling=800)
    pitch_values = pitch_obj.selected_array['frequency']

    # Melody-driven correction
    melody_freqs = [261.63, 293.66, 329.63, 392.00, 440.00, 392.00, 329.63, 293.66]
    total_frames = len(pitch_values)
    melody_len = len(melody_freqs)
    frames_per_note = max(1, total_frames // melody_len)

    corrected_pitches = np.zeros_like(pitch_values)
    for i in range(total_frames):
        note_idx = min(i // frames_per_note, melody_len - 1)
        corrected_pitches[i] = melody_freqs[note_idx] if pitch_values[i] > 0 else 0.0

    _f0, t = pyworld.harvest(y, sr, f0_floor=50.0, f0_ceil=800.0, frame_period=10.0)
    sp = pyworld.cheaptrick(y, _f0, t, sr)
    ap = pyworld.d4c(y, _f0, t, sr)

    min_len = min(len(_f0), len(corrected_pitches))
    _f0 = _f0[:min_len]
    sp = sp[:min_len, :]
    ap = ap[:min_len, :]
    corrected_pitches = corrected_pitches[:min_len]

    y_out = pyworld.synthesize(corrected_pitches, sp, ap, sr)
    if np.max(np.abs(y_out)) > 1e-6:
        y_out /= np.max(np.abs(y_out))
    y_out_int16 = np.int16(y_out * 32767)
    sf.write(output_wav_path, y_out_int16, sr)


@app.route('/generate-song', methods=['POST'])
def generate_song():
    try:
        data = request.json
        style = data.get("style", "pop")
        theme = data.get("theme", "summer")
        emotion = data.get("emotion", "happy")

        lyrics_prompt = f"Write poetic lyrics for a {emotion} {style} song about {theme}.\nLyrics:\n"
        lyrics_output = generator(
            lyrics_prompt,
            max_length=150,
            do_sample=True,
            top_p=0.9,
            temperature=0.9,
            repetition_penalty=1.3,
            truncation=True
        )
        lyrics = lyrics_output[0]['generated_text'].split('Lyrics:')[-1].strip()

        music_description = f"{emotion} {style} instrumental music about {theme}"
        music_inputs = processor_music(text=[music_description], return_tensors="pt")
        with torch.no_grad():
            audio_tokens = music_model.generate(**music_inputs, max_new_tokens=1024)

        music_array = audio_tokens[0].cpu().numpy().flatten()
        music_array = np.clip(music_array, -1, 1)
        music_path = "generated_music.wav"
        scipy.io.wavfile.write(
            music_path,
            32000,
            (music_array * 32767).astype(np.int16)
        )

        scale_notes = detect_key_naive(music_path)
        raw_vocals_path = "generated_vocals.wav"
        tts.tts_to_file(
            text=lyrics,
            file_path=raw_vocals_path,
            speed=0.5
        )

        autotuned_vocals_path = "autotuned_vocals.wav"
        autotune_audio(
            input_wav_path=raw_vocals_path,
            output_wav_path=autotuned_vocals_path,
            scale_notes=scale_notes,
            sr=32000
        )

        music = AudioSegment.from_wav(music_path).set_frame_rate(32000).set_channels(2)
        vocals = AudioSegment.from_wav(autotuned_vocals_path).set_frame_rate(32000).set_channels(2)
        vocals = vocals + 5

        combined_audio = music.overlay(vocals)
        final_path = "final_song.wav"
        combined_audio.export(final_path, format="wav")

        with open(final_path, 'rb') as f:
            b64_audio = base64.b64encode(f.read()).decode()

        return jsonify({
            "lyrics": lyrics,
            "audio_base64": b64_audio,
            "sampling_rate": 32000,
            "message": "Song generated successfully with melody-driven autotune."
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)
