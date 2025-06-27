import librosa
import os
import pandas as pd
import numpy as np
from tqdm import tqdm

def detect_key(y, sr):
    # Получаем хромограмму
    chroma = librosa.feature.chroma_cqt(y=y, sr=sr)
    chroma_mean = np.mean(chroma, axis=1)
    
    # Шаблоны для мажора и минора (C - первая нота)
    major_template = np.array([1, 0, 1, 0, 1, 1, 0, 1, 0, 1, 0, 1])
    minor_template = np.array([1, 0, 1, 1, 0, 1, 0, 1, 1, 0, 1, 0])

    # Корреляция для каждого сдвига
    scores = []
    for i in range(12):
        major_score = np.dot(np.roll(major_template, i), chroma_mean)
        minor_score = np.dot(np.roll(minor_template, i), chroma_mean)
        scores.append((major_score, minor_score))

    # Определение максимальной корреляции
    best_idx = np.argmax([max(mj, mn) for mj, mn in scores])
    best_major, best_minor = scores[best_idx]
    if best_major >= best_minor:
        mode = 'major'
    else:
        mode = 'minor'

    key_list = ['C', 'C#', 'D', 'D#', 'E', 'F', 
                'F#', 'G', 'G#', 'A', 'A#', 'B']
    key = key_list[best_idx]
    
    return f"{key} {mode}"

def analyze_bpm_key(file_path):
    try:
        y, sr = librosa.load(file_path, sr=None, mono=True)
        tempo, _ = librosa.beat.beat_track(y=y, sr=sr)
        tempo = float(np.median(tempo)) if isinstance(tempo, np.ndarray) else float(tempo)
        key = detect_key(y, sr)
        return {
            "file": os.path.basename(file_path),
            "bpm": round(tempo, 1),
            "key": key
        }
    except Exception as e:
        print(f"Ошибка при обработке {file_path}: {str(e)}")
        return {
            "file": os.path.basename(file_path),
            "bpm": None,
            "key": None
        }

def batch_analyze_bpm_key(input_folder, output_csv="bpm&key_results.csv"):
    supported_formats = ('.mp3', '.wav', '.flac', '.ogg', '.m4a')
    files = [f for f in os.listdir(input_folder) if f.lower().endswith(supported_formats)]
    
    if not files:
        print("Нет поддерживаемых аудиофайлов в указанной папке.")
        return

    results = []
    print(f"Начинаем анализ BPM и тональности для {len(files)} файлов...")

    for file in tqdm(files, desc="Processing"):
        file_path = os.path.join(input_folder, file)
        result = analyze_bpm_key(file_path)
        results.append(result)

    df = pd.DataFrame(results)
    df.to_csv(output_csv, index=False)
    print(f"Результаты сохранены в {output_csv}")
    print("\nРезультаты анализа BPM и тональности:")
    print(df.to_string(index=False))
    return df

if __name__ == "__main__":
    input_folder = r"tracks\mp3"
    batch_analyze_bpm_key(input_folder)
