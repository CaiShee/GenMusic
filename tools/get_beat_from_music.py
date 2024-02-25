import librosa
import numpy as np
import os
import glob


def get_beat_from_music(music_folder: str, sv_folder: str):
    music_paths = glob.glob(os.path.join(music_folder, "*"))
    for mus_pth in music_paths:
        y, sr = librosa.load(mus_pth)
        file_name = os.path.basename(mus_pth).split('.')[0]
        tempo, beat_frames = librosa.beat.beat_track(y=y[:], sr=sr)
        beat_times = librosa.frames_to_time(beat_frames, sr=sr)
        beat_times[1:] -= beat_times[:-1]
        sv_pth = os.path.join(sv_folder, file_name)
        np.save(sv_pth, beat_times)


if __name__ == "__main__":
    music_folder = "Data/PMEmo2019/chorus"
    sv_folder = "Data/Beat_dataset/Beat_after_PMEmo"
    get_beat_from_music(music_folder, sv_folder)
