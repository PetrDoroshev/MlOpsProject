import numpy as np
from librosa import beat, effects, feature, get_duration  # type: ignore[attr-defined]

from ..config.core import config


def extract_features(audio_data, sample_rate) -> dict:

    start_sample = int(sample_rate * get_duration(y=audio_data) // 2)
    finish_sample = start_sample + sample_rate * 3

    audio_data = audio_data[start_sample:finish_sample]

    chroma_stft = feature.chroma_stft(y=audio_data, sr=sample_rate)
    rms = feature.rms(y=audio_data)
    spec_cent = feature.spectral_centroid(y=audio_data, sr=sample_rate)
    spec_bw = feature.spectral_bandwidth(y=audio_data, sr=sample_rate)
    rolloff = feature.spectral_rolloff(y=audio_data, sr=sample_rate)
    zcr = feature.zero_crossing_rate(audio_data)
    harmony = effects.harmonic(audio_data)
    tempo = beat.tempo(y=audio_data, sr=sample_rate)
    mfcc = feature.mfcc(y=audio_data, sr=sample_rate, n_mfcc=20)

    features = [
        np.mean(chroma_stft),
        np.var(chroma_stft),
        np.mean(rms),
        np.var(rms),
        np.mean(spec_cent),
        np.var(spec_cent),
        np.mean(spec_bw),
        np.var(spec_bw),
        np.mean(rolloff),
        np.var(rolloff),
        np.mean(zcr),
        np.var(zcr),
        np.mean(harmony),
        np.var(harmony),
        np.mean(tempo),
    ]

    for coef in mfcc:
        features.append(np.mean(coef))
        features.append(np.var(coef))

    features_values = [[val] for val in features]
    features_data = dict(zip(config.ml_model_config.features, features_values))

    return features_data
