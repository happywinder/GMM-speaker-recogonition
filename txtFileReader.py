def file_reader(filename):
    audio = []
    label = []
    with open(filename, 'r', encoding='utf-8') as f:
        for line in f.readlines():
            label.append(line[:-9])
            audio.append(line.strip())
    return audio, label


if __name__ == '__main__':
    file_reader('train_audio_speech.txt')
