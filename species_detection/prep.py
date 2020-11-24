import numpy as np
import librosa, librosa.display
import matplotlib.pyplot as plt

# see audio_theory.md for more colour on this exercise
file = "blues.00000.wav"

# waveform
signal, sr = librosa.load(file, sr=22050) # signal is np array sr * T -> 22050 * 30s = ~600K
librosa.display.waveplot(signal, sr=sr)

# plt.xlabel("Time")
# plt.ylabel("Amplitude")
# plt.show()

# fft -> spectrum
fft = np.fft.fft(signal)
magnitude = np.abs(fft)
frequency = np.linspace(0, sr, len(magnitude))

left_frequency = frequency[:int(len(frequency/2))]
left_magnitude = magnitude[:int(len(frequency/2))]

plt.plot(left_frequency, left_magnitude, linestyle="-")
# plt.xlabel("Frequency")
# plt.ylabel("Magnitude")
# plt.show()

# stft -> spectrogram
args = {"n_fft": 2048, "hop_length": 512}
stft = librosa.core.stft(signal, **args)

spectrogram = np.abs(stft)
log_spectrogram = librosa.amplitude_to_db(spectrogram)
librosa.display.specshow(log_spectrogram, sr=sr, hop_length=args["hop_length"])
# plt.xlabel("Time")
# plt.ylabel("Frequency")
# plt.colorbar()
# plt.show() 

# MFCCs
mfcc_args = args
mfcc_args["n_mfcc"] = 13
mfcc = librosa.feature.mfcc(signal, **mfcc_args)
librosa.display.specshow(mfcc, sr=sr, hop_length=args["hop_length"])
plt.xlabel("Time")
plt.ylabel("MFCC")
plt.show()