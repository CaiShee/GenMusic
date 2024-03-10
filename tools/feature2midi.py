import numpy as np
import torch
from midiutil import MIDIFile
from mido import MidiFile, MidiTrack, MetaMessage, Message


def get_midifile(feature_ori: "np.ndarray", sv_path: str = 'grand_piano_midi_file.mid'):
    midi_file = MidiFile(type=0, ticks_per_beat=96)
    track = MidiTrack()
    track.append(MetaMessage('track_name', name='Grand Piano 3', time=0))
    track.append(MetaMessage('time_signature', numerator=4, denominator=4,
                 clocks_per_click=36, notated_32nd_notes_per_beat=8, time=0))

    feature = feature_ori.copy()
    kk = np.zeros((2*len(feature), 4))
    for i in range(len(feature)-1):
        feature[i+1, 1] += feature[i, 1]
    for i in range(len(feature)):
        kk[i] = feature[i]
        kk[i, 2] = 1

        kk[i+len(feature), 0] = feature[i, 0]
        kk[i+len(feature), 1] = feature[i, 1]+feature[i, 2]
        kk[i+len(feature), 3] = feature[i, 3]
    kk = sorted(kk, key=lambda x: x[1])
    kk = np.array(kk)
    kk[1:, 1] -= kk[:-1, 1]
    midi_file.tracks.append(track)
    cc = ['note_off', 'note_on']
    for k in kk:
        track.append(Message(cc[int(k[2])], channel=0,
                     note=int(k[0]), velocity=int(k[3]), time=int(k[1])))
    track.append(MetaMessage('end_of_track', time=0))
    midi_file.save(sv_path)
