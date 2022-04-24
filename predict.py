"""
Example usage of boundary prediction.
"""
import numpy as np
import librosa
import UnsupSeg
from UnsupSeg import ModelTag


def main():
    wav, sr = librosa.load("./test.wav", sr=16000)
    wav = wav.astype(np.float32)
    segmenter = UnsupSeg.load_model_from_tag(ModelTag.BUCKEYE)
    boundaries = segmenter.predict(wav)
    print(len(boundaries))
    print(boundaries)


if __name__ == "__main__":
    main()