from audio_processing import AudioProcessor
import os

# clear the directories which are used during augmentation
directories_to_clear = ['./songs_normalized', './noise_normalized', './noise_segments', './combined_files']
for directory in directories_to_clear:
    for file_name in os.listdir(directory):
        file_path = os.path.join(directory, file_name)
        try:
            if os.path.isfile(file_path):
                os.remove(file_path)
        except Exception as e:
            print(f"Error deleting file: {file_path} - {e}")

# load the song files
songs = AudioProcessor('./songs_real')
# normalize the song files
songs.normalize_audio_rms(per_second=False, new_directory='./songs_normalized')
songs = AudioProcessor('./songs_normalized')

# load the noise files
noise = AudioProcessor('./noise_real')
# normalize the noise files to 200 rms per second power
noise = noise.normalize_audio_rms(per_second=False, target_rms=200, new_directory='./noise_normalized')
# segment the noise files to 5 second segments
AudioProcessor('./noise_normalized').segment_audio_files(5, new_directory='./noise_segments')
noise = AudioProcessor('./noise_segments')

# augment the song files with the noise files
songs.generate_combined_files(noise, output_directory='./combined_files', shift_amount_ms=False)