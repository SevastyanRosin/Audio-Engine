import librosa
import os
import pandas as pd
import numpy as np
from tqdm import tqdm

def detect_key(y, sr):
    chroma = librosa.feature.chroma_cqt(y=y, sr=sr)
    chroma_mean = np.mean(chroma, axis=1)

    major_template = np.array([1, 0, 1, 0, 1, 1, 0, 1, 0, 1, 0, 1])
    minor_template = np.array([1, 0, 1, 1, 0, 1, 0, 1, 1, 0, 1, 0])

    scores = []
    for i in range(12):
        major_score = np.dot(np.roll(major_template, i), chroma_mean)
        minor_score = np.dot(np.roll(minor_template, i), chroma_mean)
        scores.append((major_score, minor_score))

    best_idx = np.argmax([max(mj, mn) for mj, mn in scores])
    best_major, best_minor = scores[best_idx]
    mode = 'major' if best_major >= best_minor else 'minor'
    key_list = ['C', 'C#', 'D', 'D#', 'E', 'F', 
                'F#', 'G', 'G#', 'A', 'A#', 'B']
    key = key_list[best_idx]
    return f"{key} {mode}"

def detect_structure(y, sr):
    energy = librosa.feature.rms(y=y)[0]
    times = librosa.frames_to_time(np.arange(len(energy)), sr=sr)

    max_energy = np.max(energy)
    threshold = 0.2 * max_energy

    # Интро: пока энергия < threshold
    intro_end_idx = next((i for i, e in enumerate(energy) if e >= threshold), len(energy)-1)
    intro_end_time = times[intro_end_idx]

    # Аутро: от момента, где энергия падает ниже threshold до конца
    outro_start_idx = next((i for i in reversed(range(len(energy))) if energy[i] >= threshold), 0)
    outro_start_time = times[outro_start_idx]

    return round(intro_end_time, 1), round(outro_start_time, 1)

def analyze_bpm_key_structure(file_path):
    try:
        y, sr = librosa.load(file_path, sr=None, mono=True)
        tempo, _ = librosa.beat.beat_track(y=y, sr=sr)
        tempo = float(np.median(tempo)) if isinstance(tempo, np.ndarray) else float(tempo)
        key = detect_key(y, sr)
        intro_end, outro_start = detect_structure(y, sr)

        return {
            "file": os.path.basename(file_path),
            "bpm": round(tempo, 1),
            "key": key,
            "intro_end_sec": intro_end,
            "outro_start_sec": outro_start
        }
    except Exception as e:
        print(f"Ошибка при обработке {file_path}: {str(e)}")
        return {
            "file": os.path.basename(file_path),
            "bpm": None,
            "key": None,
            "intro_end_sec": None,
            "outro_start_sec": None
        }

def batch_analyze(input_folder, output_csv="results.csv"):
    supported_formats = ('.mp3', '.wav', '.flac', '.ogg', '.m4a')
    files = [f for f in os.listdir(input_folder) if f.lower().endswith(supported_formats)]
    
    if not files:
        print("Нет поддерживаемых аудиофайлов в указанной папке.")
        return

    results = []
    print(f"Начинаем анализ BPM, тональности и структуры для {len(files)} файлов...")

    for file in tqdm(files, desc="Processing"):
        file_path = os.path.join(input_folder, file)
        result = analyze_bpm_key_structure(file_path)
        results.append(result)

    df = pd.DataFrame(results)
    df.to_csv(output_csv, index=False)
    print(f"Результаты сохранены в {output_csv}")
    print("\nРезультаты анализа:")
    print(df.to_string(index=False))
    return df

if __name__ == "__main__":
    input_folder = r"tracks\mp3"
    batch_analyze(input_folder)
