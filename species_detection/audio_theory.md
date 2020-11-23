## Audio theory for Kaggle competition

### Aspects of sound theory
- waveform
    - has a frequency (=pitch) φ = 1/Τ
    - has an amplitude (=loudness)
- waveforms are continuous, analogue. To represent them on a computer, we need to digitalise them (to store them in _discrete_ memory). To do so:
    - we sample the signal at uniform time intervals (e.g. 44KHz)
    - we quantise the amplitude with a certain # of bits (e.g. 8-bit music)
- obviously, real-world signals are not simple sine waves—they're a combination of waves. 
- how do we decompose this?
    - (fast) Fourier transform: decompose complex periodic sound into sum of oscillating sine waves 
      - Amplitude = how much each frequency contributes to the complex sound
      - ==> magnitude as a function of frequency 
    - but time does not come into it——we lose sound information
      - so do a short time Fourier transform: compute FFT at different intervals
      - ==> spectrogram: time, frequency, magnitude

### Tying it all together
- feature selection for our model
  - Mel frequency cepstral coefficients (MFCCs)
    - useful in speech recognition
    - music genre classification
    - instrument classification