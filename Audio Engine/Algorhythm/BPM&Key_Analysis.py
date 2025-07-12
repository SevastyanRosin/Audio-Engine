import librosa
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
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
    key_list = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']
    key = key_list[best_idx]
    return f"{key} {mode}"

def detect_structure(y, sr, filename=None):
    hop_length = 512
    frame_length = 2048
    rms = librosa.feature.rms(y=y, frame_length=frame_length, hop_length=hop_length)[0]
    times = librosa.frames_to_time(np.arange(len(rms)), sr=sr, hop_length=hop_length)

    # Сглаживание
    window = 10
    rms_smooth = np.convolve(rms, np.ones(window)/window, mode='same')

    threshold = 0.2 * np.max(rms_smooth)
    min_duration_sec = 2.5
    min_frames = int(min_duration_sec * sr / hop_length)

    # Интро: пока средняя энергия < threshold в течение ≥ N фреймов
    intro_end_idx = 0
    for i in range(min_frames, len(rms_smooth)):
        if np.all(rms_smooth[i-min_frames:i] > threshold):
            intro_end_idx = i
            break
    intro_end_time = times[intro_end_idx]

    # Аутро: от конца назад ищем участок низкой энергии
    outro_start_idx = len(rms_smooth) - 1
    for i in range(len(rms_smooth) - min_frames - 1, 0, -1):
        if np.all(rms_smooth[i:i+min_frames] > threshold):
            outro_start_idx = i + min_frames
            break
    outro_start_time = times[outro_start_idx]

    # Визуализация
    if filename:
        os.makedirs("structure_plots", exist_ok=True)
        plt.figure(figsize=(10, 4))
        plt.plot(times, rms_smooth, label="Smoothed RMS Energy", color='blue')
        plt.axvline(intro_end_time, color='green', linestyle='--', label="Intro End")
        plt.axvline(outro_start_time, color='red', linestyle='--', label="Outro Start")
        plt.title(f"Structure - {filename}")
        plt.xlabel("Time (s)")
        plt.ylabel("Energy")
        plt.legend()
        plt.tight_layout()
        plot_path = os.path.join("structure_plots", f"{os.path.splitext(filename)[0]}.png")
        plt.savefig(plot_path)
        plt.close()

    return round(intro_end_time, 1), round(outro_start_time, 1)

def analyze_bpm_key_structure(file_path):
    try:
        y, sr = librosa.load(file_path, sr=None, mono=True)
        tempo, _ = librosa.beat.beat_track(y=y, sr=sr)
        tempo = float(np.median(tempo)) if isinstance(tempo, np.ndarray) else float(tempo)
        key = detect_key(y, sr)
        filename = os.path.basename(file_path)
        intro_end, outro_start = detect_structure(y, sr, filename=filename)

        return {
            "file": filename,
            "bpm": round(tempo, 1),
            "key": key,
            "intro_end_sec": intro_end,
            "outro_start_sec": outro_start
        }
    except Exception as e:
        tqdm.write(f"Ошибка при обработке {file_path}: {str(e)}")
        return {
            "file": os.path.basename(file_path),
            "bpm": None,
            "key": None,
            "intro_end_sec": None,
            "outro_start_sec": None
        }

def batch_analyze(input_folder, output_csv="bpm_results.csv"):
    supported_formats = ('.mp3', '.wav', '.flac', '.ogg', '.m4a')
    files = [f for f in os.listdir(input_folder) if f.lower().endswith(supported_formats)]
    
    if not files:
        print("Нет поддерживаемых аудиофайлов в указанной папке.")
        return

    results = []
    tqdm.write(f"Начинаем анализ BPM, тональности и структуры для {len(files)} файлов...")

    for file in tqdm(files, desc="Processing", ncols=80):
        file_path = os.path.join(input_folder, file)
        result = analyze_bpm_key_structure(file_path)
        results.append(result)

    df = pd.DataFrame(results)
    df.to_csv(output_csv, index=False)
    tqdm.write(f"Результаты сохранены в {output_csv}")
    tqdm.write("\nРезультаты анализа:")
    tqdm.write(df.to_string(index=False))
    return df

if __name__ == "__main__":
    input_folder = r"Audio Engine\tracks\mp3"
    batch_analyze(input_folder)
