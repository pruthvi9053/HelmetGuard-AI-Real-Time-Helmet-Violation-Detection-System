'''For downloading the beep sound'''

import numpy as np
from scipy.io.wavfile import write

# Generate 1 second beep sound (1000 Hz tone)
sample_rate = 44100
duration = 1.0
frequency = 1000
t = np.linspace(0, duration, int(sample_rate * duration), False)
tone = np.sin(frequency * 2 * np.pi * t)
audio = (tone * 32767).astype(np.int16)

write("alert.wav", sample_rate, audio)
print("✅ alert.wav created!")
