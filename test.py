import random
from midiutil import MIDIFile

# 定义可能的节奏和音高范围
rhythms = ['全音符', '二分音符', '四分音符', '八分音符']
pitches = ['C3', 'C#3', 'D3', 'D#3', 'E3',
           'F3', 'F#3', 'G3', 'G#3', 'A3', 'A#3', 'B3']

# 将音符名称转换为MIDI音符编号


def note_name_to_midi(note_name):
    midi_notes = {
        'C3': 0, 'C#3': 1, 'D3': 2, 'D#3': 3, 'E3': 4,
        'F3': 5, 'F#3': 6, 'G3': 7, 'G#3': 8, 'A3': 9, 'A#3': 10, 'B3': 11
    }
    return midi_notes[note_name]

# 将节奏转换为MIDI持续时间（以四分音符为单位）


def rhythm_to_midi(rhythm):
    if rhythm == '全音符':
        return 4
    elif rhythm == '二分音符':
        return 2
    elif rhythm == '四分音符':
        return 1
    elif rhythm == '八分音符':
        return 0.5
    else:
        return 1

# 评分函数，评估乐谱的快乐程度


def score_composition(melody_rhythm, melody_pitches):
    score = 0
    # 假设快乐的乐谱倾向于使用大调音符和较快的节奏
    for pitch in melody_pitches:
        if pitch in ['C3', 'D3', 'E3', 'F3', 'G3', 'A3']:
            score += 1
    for rhythm in melody_rhythm:
        if rhythm == '四分音符' or rhythm == '八分音符':
            score += 0.5
    return score

# 创建MIDI文件


def create_midi(melody_rhythm, melody_pitches):
    midi_file = MIDIFile(1)  # 创建一个包含1个轨道的MIDI文件
    track = 0
    time = 0

    # 设置钢琴音色
    midi_file.addProgramChange(track, time, 0, 0)  # 音色编号0对应钢琴

    midi_file.addTrackName(track, time, "Happy Melody")
    midi_file.addTempo(track, time, 120)  # 设置BPM为120

    for note_name, rhythm in zip(melody_pitches, melody_rhythm):
        midi_note = note_name_to_midi(note_name)
        duration = rhythm_to_midi(rhythm)
        midi_file.addNote(track, 0, midi_note, time, int(duration * 4), 100)
        time += duration

    with open("happy_melody.mid", 'wb') as output_file:
        midi_file.writeFile(output_file)

# 主程序


def main():
    best_score = -1
    best_composition = None
    iterations = 100  # 设置迭代次数

    for _ in range(iterations):
        # 生成随机乐谱
        melody_rhythm = [random.choice(rhythms) for _ in range(8)]
        melody_pitches = [random.choice(pitches) for _ in range(8)]

        # 评分
        current_score = score_composition(melody_rhythm, melody_pitches)

        # 如果当前乐谱的评分更高，则保存为最佳乐谱
        if current_score > best_score:
            best_score = current_score
            best_composition = (melody_rhythm, melody_pitches)
    print(best_composition)

    # 创建并保存最佳乐谱
    if best_composition:
        create_midi(best_composition[0], best_composition[1])
        print("最佳快乐乐谱已生成: happy_melody.mid")
    else:
        print("未能生成快乐乐谱。")


if __name__ == "__main__":
    main()
