import mido
from mido import MidiFile, MidiTrack, Message
import numpy as np
import os
import glob


def get_feature(file_name: str) -> "tuple[list[dict],np.ndarray]":
    midi_file = MidiFile(file_name)
    active_notes = {}
    notes_info = []
    for i, track in enumerate(midi_file.tracks):
        time = 0
        for msg in track:
            time += msg.time
            if msg.type == 'note_on' and msg.velocity > 0:
                active_notes[msg.note] = {
                    'start': time, 'duration': 0, 'pitch': msg.note, "velocity": msg.velocity}

            elif msg.type == 'note_off' or (msg.type == 'note_on' and msg.velocity == 0):
                if msg.note in active_notes:
                    duration = time - active_notes[msg.note]['start']
                    active_notes[msg.note]['duration'] = duration
                    notes_info.append({
                        'pitch': active_notes[msg.note]['pitch'],
                        'start': active_notes[msg.note]['start'],
                        'duration': duration,
                        'velocity': active_notes[msg.note]['velocity']
                    })
                    del active_notes[msg.note]
    notes_info = sorted(notes_info, key=lambda x: x['start'])
    np_nodes = np.zeros((len(notes_info), 4))
    for i in range(len(notes_info)):
        np_nodes[i, 0] = notes_info[i]["pitch"]
        np_nodes[i, 1] = notes_info[i]["start"]
        np_nodes[i, 2] = notes_info[i]["duration"]
        np_nodes[i, 3] = notes_info[i]["velocity"]
    np_nodes[1:, 1] -= np_nodes[:-1, 1]
    return notes_info, np_nodes


def getCovInv(folder_name: str) -> np.ndarray:
    paths = glob.glob(os.path.join(folder_name, "*"))
    datas = []
    for path in paths:
        datas += list(get_feature(path)[1])
    datas = np.array(datas)
    cov = np.cov(datas.T)
    return np.linalg.inv(cov)


if __name__ == "__main__":
    covInv = getCovInv("Data/mini_midi/midi")
    np.save("Data/covInv", covInv)

    paths = glob.glob(os.path.join("Data/mini_midi/happy", "*"))
    for i in range(len(paths)):
        np.save("Data/mini_midi/npy_file/happy/" +
                str(i), get_feature(paths[i])[1])

    paths = glob.glob(os.path.join("Data/mini_midi/sad", "*"))
    for i in range(len(paths)):
        np.save("Data/mini_midi/npy_file/sad/" +
                str(i), get_feature(paths[i])[1])
