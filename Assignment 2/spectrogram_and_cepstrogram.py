import numpy as np
import matplotlib.pyplot as plt
import scipy.io.wavfile as wavfile
from scipy.signal import hann, spectrogram
from scipy.fftpack import dct
from python_speech_features import mfcc
from python_speech_features import logfbank


# Load the female and male speech signal and the music signal
female_fs, female_speech = wavfile.read('Assignment 2/Sounds/female.wav')
male_fs, male_speech = wavfile.read('Assignment 2/Sounds/male.wav')
music_fs, music = wavfile.read('Assignment 2/Sounds/music.wav')

windowSize = 0.03
overlap = 0.015
# Compute spectrogram
female_windowSize = round(windowSize * female_fs)
female_overlap = round(female_windowSize * 0.5)
female_F, female_T, female_S = spectrogram(female_speech, fs=female_fs, window=hann(female_windowSize),
                                            nperseg=female_windowSize, noverlap=female_overlap)

male_windowSize = round(windowSize * male_fs)
male_overlap = round(male_windowSize * 0.5)
male_F, male_T, male_S = spectrogram(male_speech, fs=male_fs, window=hann(male_windowSize),
                                      nperseg=male_windowSize, noverlap=male_overlap)

music_windowSize = round(windowSize * music_fs)
music_overlap = round(music_windowSize * 0.5)
music_F, music_T, music_S = spectrogram(music, fs=music_fs, window=hann(music_windowSize),
                                          nperseg=music_windowSize, noverlap=music_overlap)


# Plot spectrograms

plt.subplot(3,3,1)
plt.imshow(20*np.log10(abs(female_S)), cmap='jet', aspect='auto', origin='lower',
           extent=[female_T.min(), female_T.max(), female_F.min(), female_F.max()])
plt.colorbar()
plt.xlabel('Time (s)')
plt.ylabel('Frequency (Hz)')
plt.title('spectrogram female speech')

plt.subplot(3,3,2)
plt.imshow(20*np.log10(abs(male_S)), cmap='jet', aspect='auto', origin='lower',
           extent=[male_T.min(), male_T.max(), male_F.min(), male_F.max()])
plt.colorbar()
plt.xlabel('Time (s)')
plt.ylabel('Frequency (Hz)')
plt.title('spectrogram male speech')

plt.subplot(3,3,3)
plt.imshow(20*np.log10(abs(music_S)), cmap='jet', aspect='auto', origin='lower',
           extent=[music_T.min(), music_T.max(), music_F.min(), music_F.max()])
plt.colorbar()
plt.xlabel('Time (s)')
plt.ylabel('Frequency (Hz)')
plt.title('spectrogram music')


# Compute MFCC coefficients
female_coeffs = mfcc(female_speech, female_fs, winlen=windowSize, winstep=overlap, numcep=13)
male_coeffs = mfcc(male_speech, male_fs, winlen=windowSize, winstep=overlap, numcep=13)
music_coeffs = mfcc(music, music_fs, winlen=windowSize, winstep=overlap, numcep=13)
print(len(female_coeffs),len(female_coeffs[0]))
print(len(female_S), len(female_S[0]))




# normalize cepstrum
def normalize(mfcc):
    mean_cep = np.mean(mfcc, axis=0)
    std_cep = np.std(mfcc, axis=0)
    cepstrogram_centered = mfcc - mean_cep
    return cepstrogram_centered / std_cep

female_coeffs_norm = normalize(female_coeffs)
male_coeffs_norm = normalize(male_coeffs)
music_coeffs_norm = normalize(music_coeffs)

print(np.min(female_coeffs))
# plot MFCC
plt.subplot(3, 3, 4)
plt.imshow(female_coeffs.T, aspect='auto', cmap='jet', origin='lower', 
           extent=[female_T.min(), female_T.max(), np.min(female_coeffs), np.max(female_coeffs)])
plt.xlabel('Time (s)')
plt.ylabel('MFCC coeffiencit')
plt.title('cepstrogtam female speech')
plt.colorbar()

plt.subplot(3, 3, 5)
plt.imshow(male_coeffs.T, aspect='auto', cmap='jet', origin='lower',
           extent=[male_T.min(), male_T.max(), np.min(male_coeffs), np.max(male_coeffs)])
plt.xlabel('Time (s)')
plt.ylabel('MFCC coeffiencit')
plt.title('cepstrogtam male speech')
plt.colorbar()

plt.subplot(3, 3, 6)
plt.imshow(music_coeffs.T, aspect='auto', cmap='jet',origin='lower',
           extent=[music_T.min(), music_T.max(), np.min(music_coeffs), np.max(music_coeffs)])
plt.xlabel('Time (s)')
plt.ylabel('MFCC coeffiencit')
plt.title('cepstrogtam music')
plt.colorbar()

#plot normalized mfcc
plt.subplot(3, 3, 7)
plt.imshow(female_coeffs_norm.T, aspect='auto', cmap='jet', origin='lower', 
           extent=[female_T.min(), female_T.max(), np.min(female_coeffs_norm), np.max(female_coeffs_norm)])
plt.xlabel('Time (s)')
plt.ylabel('MFCC coefficient normalized')
plt.title('normalized cepstrogtam female speech')
plt.colorbar()

plt.subplot(3, 3, 8)
plt.imshow(male_coeffs_norm.T, aspect='auto', cmap='jet', origin='lower',
           extent=[male_T.min(), male_T.max(), np.min(male_coeffs_norm), np.max(male_coeffs_norm)])
plt.xlabel('Time (s)')
plt.ylabel('MFCC coefficient normalized')
plt.title('normalized cepstrogtam male speech')
plt.colorbar()

plt.subplot(3, 3, 9)
plt.imshow(music_coeffs_norm.T, aspect='auto', cmap='jet',origin='lower',
           extent=[music_T.min(), music_T.max(), np.min(music_coeffs_norm), np.max(music_coeffs_norm)])
plt.xlabel('Time (s)')
plt.ylabel('MFCC coefficient normalized')
plt.title('cepstrogtam music speech')
plt.colorbar()

plt.show()

#compute correlations
female_corr_spec = np.corrcoef(20*np.log10(abs(female_S)), rowvar=True)
male_corr_spec = np.corrcoef(20*np.log10(abs(male_S)), rowvar=True)
music_corr_spec = np.corrcoef(20*np.log10(abs(music_S)), rowvar=True)

female_corr_ceps = np.corrcoef(female_coeffs.T, rowvar=True)
male_corr_ceps = np.corrcoef(male_coeffs.T, rowvar=True)
music_corr_ceps = np.corrcoef( music_coeffs.T, rowvar=True)

female_corr_ceps_norm = np.corrcoef(female_coeffs_norm.T, rowvar=True)
male_corr_ceps_norm = np.corrcoef(male_coeffs_norm.T, rowvar=True)
music_corr_ceps_norm = np.corrcoef( music_coeffs_norm.T, rowvar=True)

plt.subplot(3,3,1)
plt.imshow(np.abs(female_corr_spec), aspect='auto', origin='lower', cmap='gray')
plt.title('spectrum correltion female')
plt.colorbar()

plt.subplot(3,3,2)
plt.imshow(np.abs(male_corr_spec), aspect='auto', origin='lower', cmap='gray')
plt.title('spectrum correltion male')
plt.colorbar()

plt.subplot(3,3,3)
plt.imshow(np.abs(music_corr_spec), aspect='auto', origin='lower', cmap='gray')
plt.title('spectrum correltion music')
plt.colorbar()

plt.subplot(3,3,4)
plt.imshow(np.abs(female_corr_ceps), aspect='auto', origin='lower', cmap='gray')
plt.title('cepstrum correltion female')
plt.colorbar()

plt.subplot(3,3,5)
plt.imshow(np.abs(male_corr_ceps), aspect='auto', origin='lower', cmap='gray')
plt.title('cepstrum correltion male')
plt.colorbar()

plt.subplot(3,3,6)
plt.imshow(np.abs(music_corr_ceps), aspect='auto', origin='lower', cmap='gray')
plt.title('cepstrum correltion music')
plt.colorbar()

plt.subplot(3,3,7)
plt.imshow(np.abs(female_corr_ceps_norm), aspect='auto', origin='lower', cmap='gray')
plt.title('normalized cepstrum correltion female')
plt.colorbar()

plt.subplot(3,3,8)
plt.imshow(np.abs(male_corr_ceps_norm), aspect='auto', origin='lower', cmap='gray')
plt.title('normalized cepstrum correltion male')
plt.colorbar()

plt.subplot(3,3,9)
plt.imshow(np.abs(music_corr_ceps_norm), aspect='auto', origin='lower', cmap='gray')
plt.title('norlamized cepstrum correltion music')
plt.colorbar()

plt.show()


