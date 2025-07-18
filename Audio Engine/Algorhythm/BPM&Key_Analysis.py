from typing import Dict, List
import librosa
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import json

class TrackAnalyzer:
    def __init__(self):
        self.hop_length = 512
        self.frame_length = 2048
        
    def detect_key(self, y: np.ndarray, sr: int) -> str:
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
        mode = 'major' if scores[best_idx][0] >= scores[best_idx][1] else 'minor'
        key = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B'][best_idx]
        return f"{key} {mode}"

    def analyze_track(self, file_path: str) -> Dict:
        try:
            y, sr = librosa.load(file_path, sr=None, mono=True)
            
            # Исправление предупреждения NumPy
            tempo = librosa.beat.beat_track(y=y, sr=sr)[0]
            tempo = float(tempo[0] if isinstance(tempo, np.ndarray) else tempo)
            
            key = self.detect_key(y, sr)
            beats = librosa.beat.beat_track(y=y, sr=sr, units='time')[1]
            segments = self._analyze_structure(y, sr)
            
            return {
                'file': os.path.basename(file_path),
                'bpm': round(tempo, 1),
                'key': key,
                'beats': beats.tolist() if isinstance(beats, np.ndarray) else beats,
                'segments': segments,
                'duration': float(librosa.get_duration(y=y, sr=sr))
            }
        except Exception as e:
            print(f"Error analyzing {file_path}: {str(e)}")
            return None

    def _analyze_structure(self, y: np.ndarray, sr: int) -> List[Dict]:
        """Анализирует музыкальную структуру трека"""
        rms = librosa.feature.rms(y=y, frame_length=self.frame_length, 
                                 hop_length=self.hop_length)[0]
        novelty = librosa.onset.onset_strength(y=y, sr=sr, hop_length=self.hop_length)
        
        # Нормализация
        rms_norm = (rms - np.min(rms)) / (np.max(rms) - np.min(rms) + 1e-6)
        novelty_norm = (novelty - np.min(novelty)) / (np.max(novelty) - np.min(novelty) + 1e-6)
        
        # Комбинированный сигнал
        combined = 0.6 * rms_norm + 0.4 * novelty_norm
        
        # Поиск сегментов
        boundaries = librosa.segment.agglomerative(combined, 15)
        times = librosa.frames_to_time(boundaries, sr=sr, hop_length=self.hop_length)
        
        segments = []
        for i in range(len(times)-1):
            start = times[i]
            end = times[i+1]
            energy = np.mean(rms_norm[int(start*sr/self.hop_length):int(end*sr/self.hop_length)])
            
            segment_type = "verse" if energy < 0.5 else "chorus"
            segments.append({
                'start': float(start),
                'end': float(end),
                'type': segment_type,
                'energy': float(energy)
            })
        
        return segments

class TransitionFinder:
    def __init__(self):
        self.analyzer = TrackAnalyzer()
        
    def find_best_transition(self, track1: Dict, track2: Dict) -> Dict:
        """Находит оптимальную точку перехода между двумя треками"""
        if not track1 or not track2:
            return None
            
        key_compat = self._check_key_compatibility(track1['key'], track2['key'])
        bpm_ratio = track2['bpm'] / track1['bpm']
        
        best_score = -1
        best_transition = None
        
        for seg1 in track1['segments']:
            if seg1['type'] != 'chorus':
                continue
                
            for seg2 in track2['segments']:
                if seg2['type'] != 'intro':
                    continue
                    
                score = self._calculate_transition_score(
                    seg1, seg2, key_compat, bpm_ratio
                )
                
                if score > best_score:
                    best_score = score
                    best_transition = {
                        'from_track': track1['file'],
                        'to_track': track2['file'],
                        'exit_point': seg1['end'],
                        'enter_point': seg2['start'],
                        'duration': min(seg1['end'] - seg1['start'], 
                                      seg2['end'] - seg2['start']),
                        'score': score,
                        'bpm_ratio': bpm_ratio,
                        'key_compatibility': key_compat
                    }
        
        return best_transition

    def _check_key_compatibility(self, key1: str, key2: str) -> float:
        key_map = {'C':0, 'C#':1, 'D':2, 'D#':3, 'E':4, 'F':5, 
                  'F#':6, 'G':7, 'G#':8, 'A':9, 'A#':10, 'B':11}
        
        try:
            k1, mode1 = key1.split()
            k2, mode2 = key2.split()
            diff = (key_map[k2] - key_map[k1]) % 12
            compatible = {0:1.0, 5:0.9, 7:0.9, 3:0.7, 4:0.7, 9:0.6}
            return compatible.get(diff, 0.2)
        except:
            return 0.2

    def _calculate_transition_score(self, seg1: Dict, seg2: Dict, 
                                 key_compat: float, bpm_ratio: float) -> float:
        bpm_score = 1 - min(abs(bpm_ratio - 1), 0.5)
        length_score = min(seg1['end'] - seg1['start'], 
                         seg2['end'] - seg2['start']) / 10
        energy_score = (seg1['energy'] + seg2['energy']) / 2
        return 0.4 * key_compat + 0.3 * bpm_score + 0.2 * energy_score + 0.1 * length_score

def analyze_folder(folder_path: str) -> List[Dict]:
    analyzer = TrackAnalyzer()
    files = [f for f in os.listdir(folder_path) 
             if f.lower().endswith(('.mp3', '.wav', '.flac', '.ogg', '.m4a'))]
    
    tracks = []
    for file in tqdm(files, desc="Analyzing tracks"):
        track = analyzer.analyze_track(os.path.join(folder_path, file))
        if track:
            tracks.append(track)
    
    return tracks

def find_all_transitions(tracks: List[Dict]) -> List[Dict]:
    finder = TransitionFinder()
    transitions = []
    
    for i in tqdm(range(len(tracks)-1), desc="Finding transitions"):
        transition = finder.find_best_transition(tracks[i], tracks[i+1])
        if transition and transition['score'] > 0.6:
            transitions.append(transition)
    
    return sorted(transitions, key=lambda x: x['score'], reverse=True)

def save_results(tracks: List[Dict], transitions: List[Dict], 
                output_prefix: str = "analysis_results"):
    # Сохранение в JSON
    with open(f"{output_prefix}_tracks.json", 'w') as f:
        json.dump(tracks, f, indent=2)
    
    with open(f"{output_prefix}_transitions.json", 'w') as f:
        json.dump(transitions, f, indent=2)
    
    try:
        import openpyxl
        tracks_df = pd.DataFrame([{
            'file': t['file'],
            'bpm': t['bpm'],
            'key': t['key'],
            'duration': t['duration'],
            'segments': len(t['segments'])
        } for t in tracks])
        
        trans_df = pd.DataFrame(transitions)
        
        with pd.ExcelWriter(f"{output_prefix}.xlsx") as writer:
            tracks_df.to_excel(writer, sheet_name='Tracks', index=False)
            trans_df.to_excel(writer, sheet_name='Transitions', index=False)
    except ImportError:
        print("Note: openpyxl not installed. Excel export skipped.")

    print(f"Results saved to {output_prefix}_*.json")

if __name__ == "__main__":
    folder_path = r"tracks\mp3"
    
    print("Analyzing tracks...")
    tracks = analyze_folder(folder_path)
    
    print("\nFinding best transitions...")
    transitions = find_all_transitions(tracks)
    
    print("\nSaving results...")
    save_results(tracks, transitions)
    
    if transitions:
        print("\nTop transitions:")
        for i, t in enumerate(transitions[:5]):
            print(f"{i+1}. {t['from_track']} ({t['exit_point']:.1f}s) → "
                  f"{t['to_track']} ({t['enter_point']:.1f}s) | "
                  f"Score: {t['score']:.2f}")
    else:
        print("No valid transitions found.")