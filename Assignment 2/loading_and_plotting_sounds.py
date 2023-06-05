import matplotlib.pyplot as plt
import numpy as np
from scipy.io import wavfile

# Load the female and male speech signal and the music signal
# Get the sampling frequency
female_fs, female_speech = wavfile.read('Assignment 2/Sounds/female.wav')
male_fs, male_speech = wavfile.read('Assignment 2/Sounds/male.wav')
music_fs, music = wavfile.read('Assignment 2/Sounds/music.wav')

# Plot the female speech signal
t_female_speech = np.arange(len(female_speech))/female_fs  # Time axis in seconds
plt.figure()
plt.plot(t_female_speech, female_speech)
plt.xlabel('Time (s)')
plt.ylabel('Amplitude')
plt.title('Female Speech Signal')

# Plot the male speech signal
t_male_speech = np.arange(len(male_speech))/male_fs  # Time axis in seconds
plt.figure()
plt.plot(t_male_speech, male_speech)
plt.xlabel('Time (s)')
plt.ylabel('Amplitude')
plt.title('Male Speech Signal')

# Plot the music signal
t_music = np.arange(len(music))/music_fs  # Time axis in seconds
plt.figure()
plt.plot(t_music, music)
plt.xlabel('Time (s)')
plt.ylabel('Amplitude')
plt.title('Music Signal')

# Zoom in on a range of 20 ms in different regions of the signals
start_time = [0.5, 1, 1.5]  # Start time of the zoomed-in range
duration = 0.02  # Duration of the zoomed-in range in seconds

# Zoom in on the female speech signal
for i in range(len(start_time)):
    idx_start = int(start_time[i]*female_fs)
    idx_end = idx_start + int(duration*female_fs)
    t_zoom = t_female_speech[idx_start:idx_end]
    female_speech_zoom = female_speech[idx_start:idx_end]
    plt.figure()
    plt.plot(t_zoom, female_speech_zoom)
    plt.xlabel('Time (s)')
    plt.ylabel('Amplitude')
    plt.title(f'Female Speech Signal (Zoomed in at {start_time[i]:.2f} s)')

# Zoom in on the male speech signal
for i in range(len(start_time)):
    idx_start = int(start_time[i]*male_fs)
    idx_end = idx_start + int(duration*male_fs)
    t_zoom = t_male_speech[idx_start:idx_end]
    male_speech_zoom = male_speech[idx_start:idx_end]
    plt.figure()
    plt.plot(t_zoom, male_speech_zoom)
    plt.xlabel('Time (s)')
    plt.ylabel('Amplitude')
    plt.title(f'Male Speech Signal (Zoomed in at {start_time[i]:.2f} s)')

# Zoom in on the music signal
for i in range(len(start_time)):
    idx_start = int(start_time[i]*music_fs)
    idx_end = idx_start + int(duration*music_fs)
    t_zoom = t_music[idx_start:idx_end]
    music_zoom = music[idx_start:idx_end]
    plt.figure()
    plt.plot(t_zoom, music_zoom)
    plt.xlabel('Time (s)')
    plt.ylabel('Amplitude')
    plt.title(f'Music Signal (Zoomed in at {start_time[i]:.2f} s)')

plt.show()
