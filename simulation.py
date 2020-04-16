import os
import time
import pydub 
import numpy as np
import scipy.signal as signal
import matplotlib.pyplot as plt
from operator import add
from easing_functions import QuarticEaseInOut
from scipy.fftpack import fft
from pylab import repeat, arange, cumsum, unwrap, arctan2, imag, real

os.system("")

"""SETUP VARIABLES"""
input_song = 'test-config.mp3'
gain = +0.0
demo_mode = False

# Ear loss is measured hearing threshold for frequencies specified in FILTER_CENTERS
# Ear loss is given in a logarythmic scale, where 0 is perfect hearing and 100 is complete deaf
left_ear_loss = [10, 60, 90, 50, 15, 10, 0, 0]
right_ear_loss = [0, 0, 0, 20, 40, 60, 75, 100]

class style():
    BLACK = '\033[30m'
    RED = '\033[31m'
    GREEN = '\033[32m'
    YELLOW = '\033[33m'
    ORANGE = '\033[34m'
    MAGENTA = '\033[35m'
    CYAN = '\033[36m'
    WHITE = '\033[37m'
    UNDERLINE = '\033[4m'
    RESET = '\033[0m'

def read(f, normalized=True):
    """MP3 to numpy array"""
    print(style.MAGENTA + 'Reading input audio:')
    a = pydub.AudioSegment.from_mp3(f)
    a_bis = a.apply_gain(gain)
    # print(a.max)
    print(style.YELLOW + 'A max:\t', style.WHITE, a_bis.max) #maximum representation: 32767
    print(style.YELLOW + 'dBFS:\t', style.WHITE, a_bis.dBFS, '\n')
    y = np.array(a_bis.get_array_of_samples())
    if a.channels == 2:
        y = y.reshape((-1, 2))
    if normalized:
        # print(style.YELLOW + 'A max:\t', style.WHITE, max(np.float32(y) / 2**15), '\n')
        return a.frame_rate, np.float32(y) / 2**15
    else:
        return a.frame_rate, y

def write(f, sr, x, normalized=False):
    """numpy array to MP3"""
    channels = 2 if (x.ndim == 2 and x.shape[1] == 2) else 1
    if normalized:  # normalized array - each item should be a float in [-1, 1)
        y = np.int16(x * 2 ** 15)
    else:
        y = np.int16(x)
    song = pydub.AudioSegment(y.tobytes(), frame_rate=sr, sample_width=2, channels=channels)
    song.export(f, format="mp3", bitrate="320k")

def plot_mfreqz(b, a=1):
    """Plot frequency and phase response """
    w, h = signal.freqz(b, a)
    h_dB = 20 * np.log10(abs(h))
    plt.subplot(211)
    plt.plot(w/max(w), h_dB)
    plt.ylim(-150, 5)
    plt.ylabel('Magnitude (db)')
    plt.xlabel(r'Normalized Frequency (x$\pi$rad/sample)')
    plt.title(r'Frequency response')
    plt.subplot(212)
    h_Phase = unwrap(arctan2(imag(h), real(h)))
    plt.plot(w/max(w), h_Phase)
    plt.ylabel('Phase (radians)')
    plt.xlabel(r'Normalized Frequency (x$\pi$rad/sample)')
    plt.title(r'Phase response')
    plt.subplots_adjust(hspace=0.5)

def plot_impz(b, a=1):
    """Plot step and impulse response """
    l = len(b)
    impulse = np.repeat(0., l)
    impulse[0] = 1.
    x = np.arange(0, l)
    response = signal.lfilter(b, a, impulse)
    plt.subplot(211)
    plt.stem(x, response, use_line_collection=True)
    plt.ylabel('Amplitude')
    plt.xlabel(r'n (samples)')
    plt.title(r'Impulse response')
    plt.subplot(212)
    step = cumsum(response)
    plt.stem(x, step, use_line_collection=True)
    plt.ylabel('Amplitude')
    plt.xlabel(r'n (samples)')
    plt.title(r'Step response')
    plt.subplots_adjust(hspace=0.5)

def seconds(x_count, fs):
  return x_count/fs

def chunkz(source_array, chunk_size, chunk_overlap=0):
  return [source_array[i:i+chunk_size] for i in range(0, len(source_array), chunk_size - chunk_overlap)]

sampling_frequency, song_data = read(input_song, normalized=True)
nyquist_frequency = sampling_frequency / 2.0
filter_length = 2073

time_window = .08 # in seconds. because of window, we are losing frequencies below 1.25Hz => fortunately, they do not matter
time_window_crossfade = .01 # Fade length
time_window_samples = int(sampling_frequency * time_window) # Window length in seconds
time_window_crossfade_samples = int(sampling_frequency * time_window_crossfade) # Crossfde window in seconds

print(style.MAGENTA + 'Initial configuration:')
print(style.YELLOW + 'Sampling frequency: \t', style.RESET, sampling_frequency, ' Hz')
print(style.YELLOW + 'Nyquist frequency: \t', style.RESET, nyquist_frequency, ' Hz')
print(style.YELLOW + 'Filter length: \t\t', style.RESET, filter_length, ' samples')
print(style.YELLOW + 'Crossfade length: \t', style.RESET, int(time_window_crossfade*1000), '  miliseconds')
print(style.YELLOW + 'Time block length: \t', style.RESET, time_window_crossfade_samples, ' samples')
print(style.YELLOW + 'Time block length: \t', style.RESET, int(time_window*1000), '  miliseconds')
print(style.YELLOW + 'Time block length: \t', style.RESET, time_window_samples, ' samples')
print(style.RESET)

#* Filter centers designate filter band
filter_centers = [250, 500, 1000, 2000, 3000, 4000, 6000, 8000]
bands = ['band_1', 'band_2', 'band_3', 'band_4', 'band_5', 'band_6', 'band_7', 'band_8']
filter_limits = {
  "band_1" : [0, 375],
  "band_2" : [375, 750],
  "band_3" : [750, 1500],
  "band_4" : [1500, 2500],
  "band_5" : [2500, 3500],
  "band_6" : [3500, 4500],
  "band_7" : [4500, 7000],
  "band_8" : [7000, nyquist_frequency]}

print(style.MAGENTA + 'Initial Data:')
print(style.YELLOW + 'Left ear loss: \t\t', style.RESET, left_ear_loss)
print(style.YELLOW + 'Right ear loss: \t', style.RESET, right_ear_loss)
print(style.RESET)

#* Filter design
print(style.CYAN + 'Designing filters...')
filter_band_1 = signal.firwin(filter_length, cutoff=375, window="blackmanharris", fs=sampling_frequency)                            # Low-pass filter
filter_band_2 = signal.firwin(filter_length, cutoff=[375, 750], window="blackmanharris", pass_zero=False, fs=sampling_frequency)    # Band-pass filter
filter_band_3 = signal.firwin(filter_length, cutoff=[750, 1500], window="blackmanharris", pass_zero=False, fs=sampling_frequency)   # Band-pass filter
filter_band_4 = signal.firwin(filter_length, cutoff=[1500, 2500], window="blackmanharris", pass_zero=False, fs=sampling_frequency)  # Band-pass filter
filter_band_5 = signal.firwin(filter_length, cutoff=[2500, 3500], window="blackmanharris", pass_zero=False, fs=sampling_frequency)  # Band-pass filter
filter_band_6 = signal.firwin(filter_length, cutoff=[3500, 4500], window="blackmanharris", pass_zero=False, fs=sampling_frequency)  # Band-pass filter
filter_band_7 = signal.firwin(filter_length, cutoff=[4500, 7000], window="blackmanharris", pass_zero=False, fs=sampling_frequency)  # Band-pass filter
filter_band_8 = signal.firwin(filter_length, cutoff=7000, window="blackmanharris", pass_zero=False, fs=sampling_frequency)          # High-pass filter
print(style.GREEN + 'Done.\n', style.RESET)

#* Print figures of designed filters with normalized spectrum
def plot_filters():
  plt.figure()
  plot_mfreqz(filter_band_1)
  plt.figure()
  plot_mfreqz(filter_band_2)
  plt.figure()
  plot_mfreqz(filter_band_3)
  plt.figure()
  plot_mfreqz(filter_band_4)
  plt.figure()
  plot_mfreqz(filter_band_5)
  plt.figure()
  plot_mfreqz(filter_band_6)
  plt.figure()
  plot_mfreqz(filter_band_7)
  plt.figure()
  plot_mfreqz(filter_band_8)
plot_filters()
plt.show() if demo_mode == True else plt.close('all') # Todo: Call to demonstrate filters

#* Splitting song_data to separate channels

print(style.CYAN + 'Splitting sounds...')
print(style.YELLOW + 'Left channel... \t', style.RESET)
init_left_channel = np.asarray([i[0] for i in song_data])

print(style.YELLOW + 'Right channel... \t', style.RESET)
init_right_channel = np.asarray([i[1] for i in song_data])
print(style.GREEN + 'Done.\n', style.RESET)


#?Check length of arrays -> should be the same
print(style.CYAN + 'Checking length of each channel...')
print(style.RESET + 'left channel length:  ', seconds(len(init_left_channel), sampling_frequency),'s')
print('right channel length: ', seconds(len(init_right_channel), sampling_frequency),'s')
print(style.GREEN + 'Length OK.\n') if len(init_left_channel) == len(init_right_channel) else print(style.RED + 'LENGTH FAILURE!\n')


#? Using Convoition theorem ->  convolution in the time domain corresponds to multiplication in the frequency domain
def filter_test():
  fs = sampling_frequency
  N = len(init_left_channel)/150
  noise_power = 1 * fs / 2
  time = np.arange(N) / float(fs)
  noise = np.random.normal(scale=np.sqrt(noise_power), size=time.shape)

  x_1 = signal.oaconvolve(noise, filter_band_1)
  x_2 = signal.oaconvolve(noise, filter_band_2)
  x_3 = signal.oaconvolve(noise, filter_band_3)
  x_4 = signal.oaconvolve(noise, filter_band_4)
  x_5 = signal.oaconvolve(noise, filter_band_5)
  x_6 = signal.oaconvolve(noise, filter_band_6)
  x_7 = signal.oaconvolve(noise, filter_band_7)
  x_8 = signal.oaconvolve(noise, filter_band_8)
  
  f, t, Sxx = signal.spectrogram(noise, sampling_frequency, scaling="density")
  plt.figure()
  plt.pcolormesh(t, f, Sxx)
  plt.ylabel('Frequency [Hz]')
  plt.xlabel('Time [sec]')
  plt.title('Noise')

  f, t, Sxx = signal.spectrogram(x_1, sampling_frequency, scaling="density")
  plt.figure()
  plt.pcolormesh(t, f, Sxx)
  plt.ylabel('Frequency [Hz]')
  plt.xlabel('Time [sec]')
  plt.title('Filter 1')

  f, t, Sxx = signal.spectrogram(x_2, sampling_frequency, scaling="density")
  plt.figure()
  plt.pcolormesh(t, f, Sxx)
  plt.ylabel('Frequency [Hz]')
  plt.xlabel('Time [sec]')
  plt.title('Filter 2')

  f, t, Sxx = signal.spectrogram(x_3, sampling_frequency, scaling="density")
  plt.figure()
  plt.pcolormesh(t, f, Sxx)
  plt.ylabel('Frequency [Hz]')
  plt.xlabel('Time [sec]')
  plt.title('Filter 3')

  f, t, Sxx = signal.spectrogram(x_4, sampling_frequency, scaling="density")
  plt.figure()
  plt.pcolormesh(t, f, Sxx)
  plt.ylabel('Frequency [Hz]')
  plt.xlabel('Time [sec]')
  plt.title('Filter 4')

  f, t, Sxx = signal.spectrogram(x_5, sampling_frequency, scaling="density")
  plt.figure()
  plt.pcolormesh(t, f, Sxx)
  plt.ylabel('Frequency [Hz]')
  plt.xlabel('Time [sec]')
  plt.title('Filter 5')

  f, t, Sxx = signal.spectrogram(x_6, sampling_frequency, scaling="density")
  plt.figure()
  plt.pcolormesh(t, f, Sxx)
  plt.ylabel('Frequency [Hz]')
  plt.xlabel('Time [sec]')
  plt.title('Filter 6')

  f, t, Sxx = signal.spectrogram(x_7, sampling_frequency, scaling="density")
  plt.figure()
  plt.pcolormesh(t, f, Sxx)
  plt.ylabel('Frequency [Hz]')
  plt.xlabel('Time [sec]')
  plt.title('Filter 7')

  f, t, Sxx = signal.spectrogram(x_8, sampling_frequency, scaling="density")
  plt.figure()
  plt.pcolormesh(t, f, Sxx)
  plt.ylabel('Frequency [Hz]')
  plt.xlabel('Time [sec]')
  plt.title('Filter 8')

  finale = x_1 + x_2 + x_3 + x_4 + x_5 + x_6 + x_7 + x_8
  f, t, Sxx = signal.spectrogram(finale, sampling_frequency, scaling="density")
  plt.figure()
  plt.pcolormesh(t, f, Sxx)
  plt.ylabel('Frequency [Hz]')
  plt.xlabel('Time [sec]')
  plt.title('Sum of signals')
filter_test() 
plt.show() if demo_mode == True else plt.close('all') #Todo: Call to demonstrate filters' working
plt.close('all')

print(style.CYAN + 'Testing convolution...')
test = signal.oaconvolve(init_left_channel, filter_band_1, mode='same')
print(style.RESET + 'Initial sample count: \t',len(init_left_channel))
print('Filtered sample count: \t',len(test))
print(style.GREEN + 'Count OK.\n') if len(init_left_channel) == len(test) else print(style.RED + 'COUNT FAILURE!\n')

#* Splitting each channel of track to 8 bands
print(style.CYAN + 'Creating band-tracks (Filtering)...')
left_channel = {}
right_channel = {}
print(style.YELLOW + 'Left channel... \t', style.RESET)
left_channel['band_1'] = signal.oaconvolve(init_left_channel, filter_band_1, mode='same')
left_channel['band_2'] = signal.oaconvolve(init_left_channel, filter_band_2, mode='same')
left_channel['band_3'] = signal.oaconvolve(init_left_channel, filter_band_3, mode='same')
left_channel['band_4'] = signal.oaconvolve(init_left_channel, filter_band_4, mode='same')
left_channel['band_5'] = signal.oaconvolve(init_left_channel, filter_band_5, mode='same')
left_channel['band_6'] = signal.oaconvolve(init_left_channel, filter_band_6, mode='same')
left_channel['band_7'] = signal.oaconvolve(init_left_channel, filter_band_7, mode='same')
left_channel['band_8'] = signal.oaconvolve(init_left_channel, filter_band_8, mode='same')

print(style.YELLOW + 'Right channel... \t', style.RESET)
right_channel['band_1'] = signal.oaconvolve(init_right_channel, filter_band_1, mode='same')
right_channel['band_2'] = signal.oaconvolve(init_right_channel, filter_band_2, mode='same')
right_channel['band_3'] = signal.oaconvolve(init_right_channel, filter_band_3, mode='same')
right_channel['band_4'] = signal.oaconvolve(init_right_channel, filter_band_4, mode='same')
right_channel['band_5'] = signal.oaconvolve(init_right_channel, filter_band_5, mode='same')
right_channel['band_6'] = signal.oaconvolve(init_right_channel, filter_band_6, mode='same')
right_channel['band_7'] = signal.oaconvolve(init_right_channel, filter_band_7, mode='same')
right_channel['band_8'] = signal.oaconvolve(init_right_channel, filter_band_8, mode='same')
print(style.GREEN + 'Done.\n', style.RESET)


#* Test Chunkization
print(style.CYAN + 'Test chunkization of filtered data in time domain...')
temp_blck = []
temp_blck.append(chunkz(left_channel['band_1'], time_window_samples, time_window_crossfade_samples))
temp_blck.append(chunkz(left_channel['band_2'], time_window_samples, time_window_crossfade_samples))
temp_blck.append(chunkz(left_channel['band_3'], time_window_samples, time_window_crossfade_samples))
temp_blck.append(chunkz(left_channel['band_4'], time_window_samples, time_window_crossfade_samples))
temp_blck.append(chunkz(left_channel['band_5'], time_window_samples, time_window_crossfade_samples))
temp_blck.append(chunkz(left_channel['band_6'], time_window_samples, time_window_crossfade_samples))
temp_blck.append(chunkz(left_channel['band_7'], time_window_samples, time_window_crossfade_samples))
temp_blck.append(chunkz(left_channel['band_8'], time_window_samples, time_window_crossfade_samples))
print(style.YELLOW + 'Band count in channel: \t', style.WHITE, len(temp_blck))
print(style.YELLOW + 'Chunk count in band: \t', style.WHITE, len(temp_blck[0]))
print(style.GREEN + 'Done.\n', style.RESET)

print(style.CYAN + 'Checking overlapping...')
temp_1 = temp_blck[1][0][-time_window_crossfade_samples:]
temp_2 = temp_blck[1][1][0:time_window_crossfade_samples]
print(style.GREEN + 'Overlapping OK.\n') if np.array_equal(temp_1, temp_2) else print(style.RED + 'Overlapping FAILURE!\n')


#* Test block spectrum extraction
print(style.CYAN + 'Test attempt to extract spectrum data from a chunk across all bands...')

def spectrum_extraction(block, band):
  # Generate time in [ms] for time signal
  dt = 1/sampling_frequency
  t = np.arange(0, time_window, dt)

  plt.figure()
  plt.subplot(2, 1, 1)
  plt.plot(t, block) 
  plt.title('Time signal')
  plt.ylim(-1, 1)
  plt.xlabel('Time [ms]')
  plt.ylabel('Amplitude')

  fr = nyquist_frequency * np.linspace(0, 1, 1764)
  temp_freq = fft(block)
  temp_freq_norm = np.abs(temp_freq[0:np.size(fr)]) * 4 / len(block)
  # temp_freq_norm = np.abs(temp_freq[0:np.size(fr)])

  plt.subplot(2, 1, 2)
  plt.plot(fr, temp_freq_norm)
  plt.title('Spectrum')
  plt.xlabel('Frequency')
  plt.ylabel('Amplitude')
  plt.xlim(filter_limits[band][0], filter_limits[band][1])
  plt.ylim(0, 1)
  
  print(style.YELLOW + 'Band: ', style.RESET, band, style.YELLOW, '\tAmpl maximum: ', style.RESET, max(abs(block)))
  print(style.YELLOW + 'Band: ', style.RESET, band, style.YELLOW, '\tFreq maximum: ', style.RESET, max(temp_freq_norm))

block = 580
plt.close('all')
spectrum_extraction(temp_blck[0][block], 'band_1')
spectrum_extraction(temp_blck[1][block], 'band_2')
spectrum_extraction(temp_blck[2][block], 'band_3')
spectrum_extraction(temp_blck[3][block], 'band_4')
spectrum_extraction(temp_blck[4][block], 'band_5')
spectrum_extraction(temp_blck[5][block], 'band_6')
spectrum_extraction(temp_blck[6][block], 'band_7')
spectrum_extraction(temp_blck[7][block], 'band_8')
plt.show() if demo_mode == True else plt.close('all') #Todo: Call to demonstrate filtered chunk
print(style.GREEN + 'Done.\n')


#* Test thresholding
print(style.CYAN + 'Test proceeding to thresholding of chunked data...')
print(style.RESET + 'Maximum of chunk before zeroing: \t', max(abs(temp_blck[0][876])))
temp_blck[0][876][:]=0
print(style.RESET + 'Maximum of zeroed chunk: \t\t', max(temp_blck[0][876]))
print(style.GREEN + 'Thresholding OK.\n') if max(temp_blck[0][876]) == 0.0 else print(style.RED + 'Thresholding failure!\n')

#* Chunkization
print(style.CYAN + 'Chunkization of full data...')
print(style.YELLOW + 'Left channel... \t', style.RESET)
left_chunks = []
left_chunks.append(chunkz(left_channel['band_1'], time_window_samples, time_window_crossfade_samples))
left_chunks.append(chunkz(left_channel['band_2'], time_window_samples, time_window_crossfade_samples))
left_chunks.append(chunkz(left_channel['band_3'], time_window_samples, time_window_crossfade_samples))
left_chunks.append(chunkz(left_channel['band_4'], time_window_samples, time_window_crossfade_samples))
left_chunks.append(chunkz(left_channel['band_5'], time_window_samples, time_window_crossfade_samples))
left_chunks.append(chunkz(left_channel['band_6'], time_window_samples, time_window_crossfade_samples))
left_chunks.append(chunkz(left_channel['band_7'], time_window_samples, time_window_crossfade_samples))
left_chunks.append(chunkz(left_channel['band_8'], time_window_samples, time_window_crossfade_samples))

print(style.YELLOW + 'Right channel... \t', style.RESET)
right_chunks = []
right_chunks.append(chunkz(right_channel['band_1'], time_window_samples, time_window_crossfade_samples))
right_chunks.append(chunkz(right_channel['band_2'], time_window_samples, time_window_crossfade_samples))
right_chunks.append(chunkz(right_channel['band_3'], time_window_samples, time_window_crossfade_samples))
right_chunks.append(chunkz(right_channel['band_4'], time_window_samples, time_window_crossfade_samples))
right_chunks.append(chunkz(right_channel['band_5'], time_window_samples, time_window_crossfade_samples))
right_chunks.append(chunkz(right_channel['band_6'], time_window_samples, time_window_crossfade_samples))
right_chunks.append(chunkz(right_channel['band_7'], time_window_samples, time_window_crossfade_samples))
right_chunks.append(chunkz(right_channel['band_8'], time_window_samples, time_window_crossfade_samples))
print(style.GREEN + 'Done.\n', style.RESET)


#* FFT Extraction && Thresholding
# Define a window for easing blocks
x = np.arange(0, 1, 1/time_window_crossfade_samples)
a = QuarticEaseInOut(start=0, end=1)
b = QuarticEaseInOut(start=1, end=0)

t_in = list(map(a.ease, x))
t_out = list(map(b.ease, x))

rising_edge = np.asarray(map(a.ease, x))
falling_edge = np.asarray(map(b.ease, x))

def lin_to_log(volume):
  #* Convert volume hearing loss to loudness threshold
  return np.float_power(10, -(100-volume)/20)

def max_of_fft(block):
  fr = nyquist_frequency * np.linspace(0, 1, 1764)
  temp_freq = fft(block)
  return max(np.abs(temp_freq[0:np.size(fr)]) * 2 / 1764) # Times two: compensation for half the window

def fade_block_edges_demo(block): 
  if len(block) == 3528: # If true, fade block from both sides
    temp = np.split(block, [441, 3087])
    intro = np.multiply(temp[0], t_in)
    outro = np.multiply(temp[2], t_out)
    sum_in_out = intro + outro

    plt.figure()
    plt.plot(intro, label='rising edge')
    plt.plot(outro, label='falling edge')
    plt.plot(sum_in_out, label='sum')
    plt.legend()
    plt.title('Ease in and Ease out curves')
    plt.xlabel('Samples')
    plt.ylabel('Gain')

    h = np.concatenate((intro, temp[1], outro), axis=0)
    plt.figure()
    plt.plot(h)
    plt.title('Block with in and out fading')
    plt.xlabel('Samples')
    plt.ylabel('Normalized Volume')
  elif len(block) > 440: # If not full block, fade-in only
    temp = np.split(block, [441])
    intro = np.multiply(temp[0], t_in)
    h = np.concatenate((intro, temp[1]), axis=0)
    plt.figure()
    plt.plot(h)
    plt.title('Block with in fading')
    plt.xlabel('Samples')
    plt.ylabel('Normalized Volume')

def fade_block_edges(block):
  if len(block) == 3528:  # If true, fade block from both sides
    temp = np.split(block, [441, 3087])   
    intro = np.multiply(temp[0], t_in)
    outro = np.multiply(temp[2], t_out)
    return np.concatenate((intro, temp[1], outro), axis=0)
  elif len(block) > 440: # If not possible to fade from both sides, fade-in only
    temp = np.split(block, [441])
    intro = np.multiply(temp[0], t_in)
    return np.concatenate((intro, temp[1]), axis=0)
  else: # If not possible to fade at all, zero that chunk.
    return np.zeros(len(block))

plt.close('all')
print(style.CYAN + 'Block fading demo...')
print(style.YELLOW + 'Demo 1\t', style.RESET)
fade_block_edges_demo(np.ones(3528))
print(style.YELLOW + 'Demo 2')
fade_block_edges_demo(np.ones(2430))
plt.show() if demo_mode == True else plt.close('all') #Todo: Call to demonstrate filters work
print(style.GREEN + 'Done.\n')

print(style.CYAN + 'FFT Extraction && Thresholding...')

def threshold_proc(band, threshold):
  temp_band = []
  for chunk in band:
    if max_of_fft(chunk) < threshold:
      temp_band.append(np.zeros(len(chunk)))
    else:
      temp_band.append(fade_block_edges(chunk))
  return temp_band

print(style.YELLOW + 'Left channel... \t', style.RESET)
thrsld_left_chunks = []
thrsld_left_chunks.append(threshold_proc(left_chunks[0], lin_to_log(left_ear_loss[0])))
thrsld_left_chunks.append(threshold_proc(left_chunks[1], lin_to_log(left_ear_loss[1])))
thrsld_left_chunks.append(threshold_proc(left_chunks[2], lin_to_log(left_ear_loss[2])))
thrsld_left_chunks.append(threshold_proc(left_chunks[3], lin_to_log(left_ear_loss[3])))
thrsld_left_chunks.append(threshold_proc(left_chunks[4], lin_to_log(left_ear_loss[4])))
thrsld_left_chunks.append(threshold_proc(left_chunks[5], lin_to_log(left_ear_loss[5])))
thrsld_left_chunks.append(threshold_proc(left_chunks[6], lin_to_log(left_ear_loss[6])))
thrsld_left_chunks.append(threshold_proc(left_chunks[7], lin_to_log(left_ear_loss[7])))

print(style.YELLOW + 'Right channel... \t', style.RESET)
thrsld_right_chunks = []
thrsld_right_chunks.append(threshold_proc(right_chunks[0], lin_to_log(right_ear_loss[0])))
thrsld_right_chunks.append(threshold_proc(right_chunks[1], lin_to_log(right_ear_loss[1])))
thrsld_right_chunks.append(threshold_proc(right_chunks[2], lin_to_log(right_ear_loss[2])))
thrsld_right_chunks.append(threshold_proc(right_chunks[3], lin_to_log(right_ear_loss[3])))
thrsld_right_chunks.append(threshold_proc(right_chunks[4], lin_to_log(right_ear_loss[4])))
thrsld_right_chunks.append(threshold_proc(right_chunks[5], lin_to_log(right_ear_loss[5])))
thrsld_right_chunks.append(threshold_proc(right_chunks[6], lin_to_log(right_ear_loss[6])))
thrsld_right_chunks.append(threshold_proc(right_chunks[7], lin_to_log(right_ear_loss[7])))
print(style.GREEN + 'Done.\n', style.RESET)


#* Assembling Bands
print(style.CYAN + 'Assembling bands <dechunkization>...')
print(style.YELLOW + 'Left channel... \t', style.RESET)
processed_left_bands = []

for band in thrsld_left_chunks:
  temp_band = []
  temp_overlap = []
  # print(style.RESET + 'Band len: \t', len(band))
  for chunk in band:
    if len(temp_band) == 0:
      temp_band = list(chunk[:-time_window_crossfade_samples])
      temp_overlap = list(chunk[-time_window_crossfade_samples:])
    else:
      chunk_first_part = list(chunk[:time_window_crossfade_samples])
      chunk_middle_part = list(chunk[time_window_crossfade_samples:-time_window_crossfade_samples])
      chunk_last_part = list(chunk[-time_window_crossfade_samples:])

      overlap = list(map(add, temp_overlap, chunk_first_part))
      temp_band.extend(overlap + chunk_middle_part)
      temp_overlap = chunk_last_part
  temp_band.extend(temp_overlap)
  processed_left_bands.append(temp_band)

print(style.GREEN + 'Count OK.\n') if len(init_left_channel) == len(processed_left_bands[0]) else print(style.RED + 'COUNT FAILURE!\n')


print(style.YELLOW + 'Right channel... \t', style.RESET)
processed_right_bands = []

for band in thrsld_right_chunks:
  temp_band = []
  temp_overlap = []
  # print(style.RESET + 'Band len: \t', len(band))
  for chunk in band:
    if len(temp_band) == 0:
      temp_band = list(chunk[:-time_window_crossfade_samples])
      temp_overlap = list(chunk[-time_window_crossfade_samples:])
    else:
      chunk_first_part = list(chunk[:time_window_crossfade_samples])
      chunk_middle_part = list(chunk[time_window_crossfade_samples:-time_window_crossfade_samples])
      chunk_last_part = list(chunk[-time_window_crossfade_samples:])

      overlap = list(map(add, temp_overlap, chunk_first_part))
      temp_band.extend(overlap + chunk_middle_part)
      temp_overlap = chunk_last_part
  temp_band.extend(temp_overlap)
  processed_right_bands.append(temp_band)

print(style.GREEN + 'Count OK.') if len(init_left_channel) == len(processed_right_bands[0]) else print(style.RED + 'COUNT FAILURE!')
print(style.GREEN + 'Done.\n', style.RESET)


#* Channel assembly
print(style.CYAN + 'Assembling channels from bands...')
print(style.YELLOW + 'Left channel assembly...')

processed_left_channel = []
for i in range(len(processed_left_bands[0])):
  temp_sum = processed_left_bands[0][i] + processed_left_bands[1][i] + processed_left_bands[2][i] + processed_left_bands[3][i] + processed_left_bands[4][i] + processed_left_bands[5][i] + processed_left_bands[6][i] + processed_left_bands[7][i]
  processed_left_channel.append(temp_sum)

print(style.YELLOW + 'Right channel assembly...')
processed_right_channel = []
for i in range(len(processed_right_bands[0])):
  temp_sum = processed_right_bands[0][i] + processed_right_bands[1][i] + processed_right_bands[2][i] + processed_right_bands[3][i] + processed_right_bands[4][i] + processed_right_bands[5][i] + processed_right_bands[6][i] + processed_right_bands[7][i]
  processed_right_channel.append(temp_sum)

print(style.GREEN + 'Done.\n', style.RESET)


#* Removal of DC offset
print(style.CYAN + 'Calculating DC offset...')
left_offset = np.mean(processed_left_channel) / len(processed_left_channel)
right_offset = np.mean(processed_right_channel) / len(processed_right_channel)
print(style.RESET, left_offset)
print(style.RESET, right_offset)

print(style.CYAN + 'Mitigating offset...')
processed_left_channel = processed_left_channel - left_offset
processed_right_channel = processed_right_channel - right_offset

print(style.RESET, np.mean(processed_left_channel) / len(processed_left_channel))
print(style.RESET, np.mean(processed_right_channel) / len(processed_right_channel))
print(style.GREEN + 'Done.\n', style.RESET)


#* Channel normalization
print(style.CYAN + 'Channel normlization...')
norm_max = max(max(np.abs(processed_left_channel)), max(np.abs(processed_right_channel)))
print(style.YELLOW + 'Left channel... \t', style.RESET)
normalized_left_channel = processed_left_channel / norm_max
print(style.YELLOW + 'Right channel... \t', style.RESET)
normalized_right_channel = processed_right_channel / norm_max

print(style.YELLOW, 'Max before normalization ', style.RESET, norm_max)

norm_max = max(max(np.abs(normalized_left_channel)), max(np.abs(normalized_right_channel)))
print(style.YELLOW, 'Max after normalization: ', style.RESET, norm_max)
print(style.GREEN + 'Done.\n', style.RESET)


#* Construe song for output
print(style.CYAN + 'Song construction...')

output_song_data = np.asarray([[normalized_left_channel[i], normalized_right_channel[i]] for i in range(len(normalized_left_channel))])
print(style.GREEN + 'Done.\n', style.RESET)


#* Export settings
print(style.CYAN + 'Export settings:')

export_time = time.localtime()
export_name = "-".join((str(export_time.tm_year), str(export_time.tm_mon), str(export_time.tm_mday),":".join((str(export_time.tm_hour), str(export_time.tm_min))), 'export.mp3'))

print(style.YELLOW + 'Filename: \t\t', style.WHITE + export_name, '\n')
print(style.YELLOW + 'Initial Length: \t', style.WHITE, len(song_data))
print(style.WHITE, song_data[-10:], '\n')
print(style.YELLOW + 'Output Length: \t\t', style.WHITE, len(output_song_data))
print(style.WHITE, output_song_data[-10:], '\n')


#* Export song
print(style.CYAN + 'Exporting song...')
write(export_name, sampling_frequency, output_song_data, normalized=True)
print(style.GREEN + 'Done.\n', style.RESET)
print(style.MAGENTA + 'END OF SIMULATION\n')
