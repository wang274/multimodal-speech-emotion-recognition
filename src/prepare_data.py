"""
This script preprocesses data and prepares data to be actually used in training
"""
import re
import os
import pickle
import unicodedata
import pandas as pd
from IPython.display import display

from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split


def read_file():
    df = pd.read_csv('G:/IEMOCAP/pre-processed/audio_features.csv')
    df = df[df['label'].isin([0, 1, 2, 3, 4, 5, 6, 7])]
    print(df.shape)
    display(df.head())

    # change 7 to 2
    df['label'] = df['label'].map({0: 0, 1: 1, 2: 1, 3: 2, 4: 2, 5: 3, 6: 4, 7: 5})
    df.head()
    df.to_csv('G:/IEMOCAP/pre-processed/no_sample_df.csv')

    # oversample fear
    fear_df = df[df['label'] == 3]
    for i in range(30):
        df = df.append(fear_df)

    sur_df = df[df['label'] == 4]
    for i in range(10):
        df = df.append(sur_df)

    df.to_csv('G:/IEMOCAP/pre-processed/modified_df.csv')
    emotion_dict = {'ang': 0,
                    'hap': 1,
                    'sad': 2,
                    'neu': 3, }
    scalar = MinMaxScaler()
    df[df.columns[2:]] = scalar.fit_transform(df[df.columns[2:]])
    df.head()
    return df

def unicodeToAscii(s):
    return ''.join(
        c for c in unicodedata.normalize('NFD', s)
        if unicodedata.category(c) != 'Mn'
    )


def normalizeString(s):
    """
    Lowercase, trim, and remove non-letter characters
    """
    s = unicodeToAscii(s.lower().strip())
    s = re.sub(r"([.!?])", r" \1", s)
    s = re.sub(r"[^a-zA-Z.!?]+", r" ", s)
    return s


def transcribe_sessions():
    file2transcriptions = {}
    useful_regex = re.compile(r'^(\w+)', re.IGNORECASE)

    for sess in range(1, 6):
        transcript_path = 'G:/IEMOCAP/Session{}/dialog/transcriptions/'
        transcript_path = transcript_path.format(sess)
        print(transcript_path)
        for f in os.listdir(transcript_path):
            with open('{}{}'.format(transcript_path, f), 'r') as f:
                all_lines = f.readlines()

            for l in all_lines:
                audio_code = useful_regex.match(l).group()
                transcription = l.split(':')[-1].strip()
                # assuming that all the keys would be unique and hence no `try`
                file2transcriptions[audio_code] = transcription
    with open('G:/IEMOCAP/t2e/audiocode2text.pkl', 'wb') as file:
        pickle.dump(file2transcriptions, file)
    print(len(file2transcriptions))
    return file2transcriptions



def prepare_text_data(audiocode2text):
    df = read_file()
    x_train, x_test = train_test_split(df, test_size=0.20)
    x_train.to_csv('G:/IEMOCAP/s2e/audio_train.csv', index=False)
    x_test.to_csv('G:/IEMOCAP/s2e/audio_test.csv', index=False)

    print(x_train.shape, x_test.shape)
    # Prepare text data
    text_train = pd.DataFrame()
    text_train['wav_file'] = x_train['wav_file']
    text_train['label'] = x_train['label']
    text_train['transcription'] = [normalizeString(audiocode2text[code]) for code in x_train['wav_file']]

    text_test = pd.DataFrame()
    text_test['wav_file'] = x_test['wav_file']
    text_test['label'] = x_test['label']
    text_test['transcription'] = [normalizeString(audiocode2text[code]) for code in x_test['wav_file']]

    text_train.to_csv('G:/IEMOCAP/t2e/text_train.csv', index=False)
    text_test.to_csv('G:/IEMOCAP/t2e/text_test.csv', index=False)

    print(text_train.shape, text_test.shape)


def main():
    prepare_text_data(transcribe_sessions())


if __name__ == '__main__':
    main()
