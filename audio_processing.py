import os
import glob
import shutil
import math
import numpy as np
from scipy.io import wavfile
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
import torch
import torch.nn as nn
import torchaudio
from pydub import AudioSegment
from moviepy.editor import AudioFileClip

class ConvAutoencoder(nn.Module):
    def __init__(self):
        super(ConvAutoencoder, self).__init__()
        
        # Encoder layers
        self.conv1 = nn.Conv1d(201, 100, 3, padding=1)  
        self.conv2 = nn.Conv1d(100, 50, 3, padding=1)
        self.relu = nn.ReLU()
        self.maxpool = nn.MaxPool1d(2, 2)
        
        # Decoder layers
        self.t_conv1 = nn.ConvTranspose1d(50, 100, 2, stride=2)
        self.t_conv2 = nn.ConvTranspose1d(100, 201, 2, stride=2)

    def forward(self, x):
        # Reshape input to (batch_size, frequency_bins, time_steps)
        x = x.view(x.shape[0], x.shape[2], x.shape[3])

        # Encoder
        x = self.relu(self.conv1(x))
        x = self.maxpool(x)
        x = self.relu(self.conv2(x))
        x = self.maxpool(x)
        
        # Decoder
        x = self.relu(self.t_conv1(x))
        x = self.t_conv2(x)

        # Reshape output to (batch_size, frequency_bins, time_steps)
        x = x.view(x.shape[0], x.shape[1], x.shape[2])
        
        return x

class AudioProcessor:
    def __init__(self, directory, exclude=[]):
        self.directory = directory
        self.files = glob.glob(os.path.join(directory, '*.wav')) + glob.glob(os.path.join(directory, '*.WAV'))
        for filename in exclude:
            if os.path.join(directory, filename) in self.files:
                self.files.remove(os.path.join(directory, filename))
        
        # add .mp4 files
        x = glob.glob(os.path.join(directory, '*.mp4')) + glob.glob(os.path.join(directory, '*.MP4'))
        for filename in x:
            clip = AudioFileClip(filename)
            clip.write_audiofile(filename.replace('.mp4', '.wav'))
        
        # device
        self.device = (
            "cuda"
            if torch.cuda.is_available()
            else "mps"
            if torch.backends.mps.is_available()
            else "cpu"
        )

    def load_audio(self, filename):
        #resample to 96kHz
        audio, sr = torchaudio.load(filename)
        if sr != 96000:
            resampler = torchaudio.transforms.Resample(sr, 96000)
            audio = resampler(audio)
        #convert to mono if necessary
        if audio.shape[0] > 1 and audio.shape[1] > 1:
            audio = torch.mean(audio, dim=0, keepdim=True)
        return audio, 96000

    def calculatePower(self, plot=False, power_threshold=float('-inf'), new_dir=''):
        if new_dir != '' and not os.path.exists(new_dir):
            os.makedirs(new_dir, exist_ok=True)

        max_power = float('-inf')
        min_power = float('inf')
        max_file = min_file = ''
        power_values = []

        for filename in self.files:
            # samplerate, data = self.load_audio(filename)    
            samplerate, data = wavfile.read(filename)
            data = data.astype(np.float32)
            power = 10 * np.log10(np.sum(data ** 2))
            power_values.append(power)

            if power > max_power:
                max_power = power
                max_file = filename
            if power < min_power:
                min_file = filename
                min_power = power

            if power < power_threshold and new_dir != '':
                print(f'Copying "{filename}" to new directory...')
                shutil.copy(filename, new_dir)
        if plot: self.plotHist(power_values, 'Distribution of Logarithmically Scaled Total Power')
        return max_power, max_file, min_power, min_file
    
    def calculateLength(self, plot=False, length_threshold=float('inf'), new_dir=''):
        if new_dir != '' and not os.path.exists(new_dir):
            os.makedirs(new_dir, exist_ok=True)

        max_length = float('-inf')
        min_length = float('inf')
        max_file = min_file = ''
        length_values = []

        for filename in self.files:
            samplerate, data = wavfile.read(filename)
            length = data.shape[0]
            length_values.append(length)

            if length > max_length:
                max_length = length
                max_file = filename
            if length < min_length:
                min_file = filename
                min_length = length

            if length > length_threshold and new_dir != '':
                print(f'Copying "{filename}" to new directory...')
                shutil.copy(filename, new_dir)
        if plot: self.plotHist(length_values, 'Distribution of Audio Lengths', 'Length (samples)', 'Frequency')
        return max_length, max_file, min_length, min_file
    
    def plotHist(self, data, title, xlabel='Logarithmically Scaled Total Power (dB)', ylabel='Frequency'):
        plt.hist(data, bins=100)
        plt.title(title)
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        plt.show()

    def adjust_power_and_save(self, new_directory, gain_dB=0):
        if not os.path.exists(new_directory):
            os.makedirs(new_directory, exist_ok=True)

        for file in self.files:
            audio = AudioSegment.from_file(file)
            louder_audio = audio + gain_dB
            base, ext = os.path.splitext(os.path.basename(file))
            new_file_name = f"{base}_power_{gain_dB}{ext}"
            louder_audio.export(os.path.join(new_directory, new_file_name), format='wav')

    def generate_combined_files(self, noise_processor, output_directory, shift_amount_ms=1000):
        if not os.path.exists(output_directory):
            os.makedirs(output_directory, exist_ok=True)
        
        for bird_file in self.files:
            bird_sound = AudioSegment.from_file(bird_file)
            for noise_file in noise_processor.files:
                noise_sound = AudioSegment.from_file(noise_file)

                if shift_amount_ms != False:
                    for i in range(0, len(noise_sound) - len(bird_sound) + 1, shift_amount_ms):
                        if i+len(bird_sound) > len(noise_sound):
                            break # don't cut off the bird sound
                        
                        output_sound = noise_sound[:i]
                        output_sound += noise_sound[i:i+len(bird_sound)].overlay(bird_sound)
                        output_sound += noise_sound[i+len(bird_sound):]

                        noise_file_name = os.path.splitext(os.path.basename(noise_file))[0]
                        bird_file_name = os.path.splitext(os.path.basename(bird_file))[0]
                        new_file_name = f"{bird_file_name}_{noise_file_name}_shift{i}.wav"

                else:
                    # when shift is false, shorten the noise to the length of the bird sound and overlay
                    noise_sound = noise_sound[:len(bird_sound)]
                    output_sound = noise_sound.overlay(bird_sound)

                    noise_file_name = os.path.splitext(os.path.basename(noise_file))[0]
                    bird_file_name = os.path.splitext(os.path.basename(bird_file))[0]
                    new_file_name = f"{bird_file_name}_{noise_file_name}.wav"

                output_sound.export(os.path.join(output_directory, new_file_name), format='wav')

    def resize_audio(self, desired_length_ms, new_directory, pad=True, cut=True):
        if not os.path.exists(new_directory):
            os.makedirs(new_directory, exist_ok=True)
        
        for file in self.files:
            audio = AudioSegment.from_file(file)
            if len(audio) < desired_length_ms:
                if pad:
                    padding = AudioSegment.silent(duration=desired_length_ms-len(audio))
                    padded_audio = audio + padding
                    padded_audio.export(os.path.join(new_directory, os.path.basename(file)), format='wav')
            elif len(audio) > desired_length_ms:
                if cut:
                    cut_audio = audio[:desired_length_ms]
                    cut_audio.export(os.path.join(new_directory, os.path.basename(file)), format='wav')
            else:
                audio.export(os.path.join(new_directory, os.path.basename(file)), format='wav')
 
    def normalize_audio_rms(self, new_directory='', target_rms=1000):
        # audio clips should contain no silence
        if not os.path.exists(new_directory):
            os.makedirs(new_directory, exist_ok=True)
            print('Created new directory')

        normalized_audio_files = []

        for file in self.files:
            audio = AudioSegment.from_file(file)
            current_rms = audio.rms
            print(f'Current RMS for {file}: {current_rms}')

            if current_rms == 0: # avoid divide by zero
                normalized_audio = audio

            else:
                gain_dB = 20 * math.log10(target_rms / current_rms)
                normalized_audio  = audio.apply_gain(gain_dB)

            if new_directory != '':
                normalized_audio.export(os.path.join(new_directory, os.path.basename(file)), format='wav')
            normalized_audio_files.append(normalized_audio)

        return normalized_audio_files

    def normalize_audio_peak(self, new_directory, target_dBFS=0.0):
        # =<0.0 will maximise the volume without clipping peaks
        if not os.path.exists(new_directory):
            os.makedirs(new_directory, exist_ok=True)

        for file in self.files:
            audio = AudioSegment.from_file(file)
            normalized_audio = audio.apply_gain(target_dBFS - audio.dBFS)
            normalized_audio.export(os.path.join(new_directory, os.path.basename(file)), format='wav')

    def run_through_autoencoder(self, model_path, new_directory, plot=False):
        if not os.path.exists(new_directory):
            os.makedirs(new_directory)

        model = ConvAutoencoder()
        model.load_state_dict(torch.load(model_path))
        model = model.to(self.device)
        model.eval()

        for file in self.files:
            waveform, sample_rate = self.load_audio(file)
            spectrogram = torchaudio.transforms.Spectrogram()(waveform)
            spectrogram_complex = torchaudio.transforms.Spectrogram(power=None)(waveform)
            magnitude = torch.abs(spectrogram_complex)
            phase = torch.angle(spectrogram_complex)

            # Inference
            with torch.no_grad():
                spectrogram = spectrogram.to(self.device)
                recreated_spectrogram = model(spectrogram.unsqueeze(0))  # Add a batch dimension

            spectrogram = spectrogram.to('cpu').squeeze()
            recreated_spectrogram = recreated_spectrogram.squeeze().to('cpu').detach()

            # Transpose the matrices
            recreated_spectrogram = recreated_spectrogram.unsqueeze(0)  # add mono channel
            # change from 4800 to 4801 (model outputs wrong size)
            x = recreated_spectrogram[:, :, -1].unsqueeze(2)
            recreated_spectrogram = torch.cat((recreated_spectrogram, x), dim=2)

            # convert back to waveform
            recreated_spectrogram_complex = recreated_spectrogram * torch.cos(phase) + recreated_spectrogram * torch.sin(phase) * 1j
            recreated_waveform = torch.istft(recreated_spectrogram_complex, n_fft=400, hop_length=200)

            # Save the resulting waveform as a new audio file
            torchaudio.save(os.path.join(new_directory, os.path.basename(file)), recreated_waveform, sample_rate)

            # Plot the original and recreated spectrogram
            if plot:
                fig, axs = plt.subplots(2, 1, figsize=(10, 10))
                axs[0].imshow(spectrogram, aspect='auto', cmap='inferno', norm=LogNorm())
                axs[0].set_title('Original Spectrogram')
                axs[1].imshow(recreated_spectrogram.squeeze(), aspect='auto', cmap='inferno', norm=LogNorm())
                axs[1].set_title('Recreated Spectrogram')
                plt.show()

    def segment_audio_files(self, segment_length, new_directory='', pad_end=False, sample_rate=96000):
        if new_directory!='' and not os.path.exists(new_directory):
            os.makedirs(new_directory)
            print('created directory')
        segment_length = int(segment_length * sample_rate)
        print(f'segment length: {segment_length}')

        segments = []
        for file in self.files:
            waveform, sample_rate = torchaudio.load(file)
            num_segments = math.ceil(waveform.shape[1] / segment_length)
            print(f'number of segments: {num_segments}')
            for i in range(num_segments):
                start_sample = i * segment_length
                end_sample = start_sample + segment_length
                # if last segment is shorter than segment_length
                if end_sample > waveform.shape[1]:
                    if pad_end:
                        padding = torch.zeros((waveform.shape[0], end_sample - waveform.shape[1]))
                        padded_waveform = torch.cat((waveform[:, start_sample:], padding), dim=1)
                        print(padded_waveform)
                        print(os.path.join(new_directory, os.path.basename(file)))
                        #convert to 16-bit
                        padded_waveform = (padded_waveform * 32767).to(torch.int16)
                        if new_directory!='': torchaudio.save(os.path.join(new_directory, os.path.basename(file)), padded_waveform, sample_rate)
                        print(f'{i}: {padded_waveform.shape}')
                        segments.append(padded_waveform)
                    else:
                        continue
                else:
                    segment_waveform = waveform[:, start_sample:end_sample]

                    segment_filename = os.path.join(new_directory, os.path.basename(file)[:-4] + '_' + str(i) + '.wav')
                    #convert to 16-bit
                    segment_waveform = (segment_waveform * 32767).to(torch.int16)
                    if new_directory!='': torchaudio.save(segment_filename, segment_waveform, sample_rate)
                    segments.append(segment_waveform)
        return segments

# Usage
raw_files_path = '/Users/kaspar/Desktop/eco_commons/rāpaki_temp_store/rāpaki_001_0224_15_06_2023_0207_16_06_2023'
exclude = ['20230615_022500.WAV', '20230615_022454.WAV', '20230616_020700.WAV', '20230615_174700.WAV']
manually_isolated_path = '/Users/kaspar/Desktop/AEDI/data/rap_001_isolated_manual'
test_noise_path = 'data/test_noise'
test_song_path = 'data/test_song'
test_combined_path = 'data/test_combined'

print('y')