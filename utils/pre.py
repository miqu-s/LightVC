import os
import crepe
import numpy as np
from scipy.io import wavfile
import pandas as pd
import matplotlib.pyplot as plt
from scipy import signal


def process(input_folder, save_dir, model):
    """
    PRE-PROCESSES THE AUDIO TO SPECTOGRAMS AND CSV FILES
    """
    specs_dir = os.path.join(save_dir, model, 'specs')
    f0_dir = os.path.join(save_dir, model, 'f0')
    
    os.makedirs(specs_dir, exist_ok=True)
    os.makedirs(f0_dir, exist_ok=True)
    
    for filename in os.listdir(input_folder):
        if filename.endswith('.wav'):
            audio_file_path = os.path.join(input_folder, filename)
            sr, audio = wavfile.read(audio_file_path)

            chunk_size = sr
            num_chunks = len(audio) // chunk_size
            for i in range(num_chunks):
                chunk_audio = audio[i * chunk_size : (i + 1) * chunk_size]

                time, frequency, confidence, _ = crepe.predict(chunk_audio, sr, viterbi=True)

                csv_filename = os.path.splitext(filename)[0] + f'_chunk_{i+1}_f0.csv'
                csv_path = os.path.join(f0_dir, csv_filename)
                df_f0 = pd.DataFrame({'time': time, 'frequency': frequency, 'confidence': confidence})
                df_f0.to_csv(csv_path, index=False)
                
                print(f"Processed chunk {i+1} of {filename}, f0 at {csv_path}")

                spec_filename = os.path.splitext(filename)[0] + f'_chunk_{i+1}_spec.npy'
                spec_path = os.path.join(specs_dir, spec_filename)

                nfft = 256
                noverlap = 128
                frequencies, times, spec = signal.spectrogram(chunk_audio, fs=sr, nperseg=nfft, noverlap=noverlap)
                plt.figure(figsize=(244/100, 244/100), dpi=100)
                plt.imshow(np.log(spec), aspect='auto', cmap='viridis')
                plt.axis('off')  # Turn off axis
                plt.subplots_adjust(left=0, right=1, top=1, bottom=0)
                plt.savefig('spectrogram.png', bbox_inches='tight', pad_inches=0)
                plt.close()
                spectrogram_img = plt.imread('spectrogram.png')
                spec_array = np.array(spectrogram_img)
                np.save(spec_path, spec_array)
                
                print(f"Processed chunk {i+1} of {filename}, saved spectrogram to {spec_path}")
    return 'Finished.'
