import argparse
from pydub import AudioSegment
import numpy as np
from scipy.signal import iirpeak, sosfilt

def create_peak_filter(freq, fs, q):
    # Create the peak filter coefficients
    b, a = iirpeak(freq / (0.5 * fs), q)
    sos = np.vstack([b, a]).reshape(1, -1)
    return sos

def apply_filter(data, sos):
    # Apply the SOS filter to the data
    return sosfilt(sos, data)

def apply_gain(data, gain_dB):
    gain_linear = 10 ** (gain_dB / 20)
    return data * gain_linear

def normalize_audio(data):
    peak = np.max(np.abs(data))
    if peak == 0:
        return data
    return data / peak

def apply_peak_eq(audio_file_path, output_file_path, freq, q, gain_dB, sample_rate=44100):
    # Load audio file
    audio = AudioSegment.from_file(audio_file_path)
    
    # Convert to raw audio data
    samples = np.array(audio.get_array_of_samples()).astype(np.float32)
    
    # Debug: Print original sample statistics
    print(f"Original Samples - Min: {np.min(samples)}, Max: {np.max(samples)}, Mean: {np.mean(samples)}")
    
    # Create the peak filter
    sos = create_peak_filter(freq, sample_rate, q)
    
    # Apply the filter
    filtered_samples = apply_filter(samples, sos)
    
    # Apply gain
    filtered_samples = apply_gain(filtered_samples, gain_dB)
    
    # Debug: Print filtered sample statistics
    print(f"Filtered Samples - Min: {np.min(filtered_samples)}, Max: {np.max(filtered_samples)}, Mean: {np.mean(filtered_samples)}")
    
    # Normalize the filtered audio
    normalized_filtered_samples = normalize_audio(filtered_samples)
    
    # Scale to 16-bit PCM range
    scaled_samples = normalized_filtered_samples * (2**15 - 1)
    
    # Ensure the data is within the valid range
    scaled_samples = np.clip(scaled_samples, -2**15, 2**15 - 1)
    
    # Debug: Print scaled sample statistics
    print(f"Scaled Samples - Min: {np.min(scaled_samples)}, Max: {np.max(scaled_samples)}, Mean: {np.mean(scaled_samples)}")
    
    # Convert back to audio segment
    filtered_audio = audio._spawn(scaled_samples.astype(np.int16).tobytes())
    
    # Export the processed audio
    filtered_audio.export(output_file_path, format="wav")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Apply peak EQ to an audio file.')
    parser.add_argument('input_file', type=str, help='Path to the input audio file')
    parser.add_argument('output_file', type=str, help='Path to the output audio file')
    parser.add_argument('--peak_freq', type=float, default=200.0, help='Peak frequency in Hz')
    parser.add_argument('--q_factor', type=float, default=1.0, help='Quality factor (controls the width of the peak)')
    parser.add_argument('--gain_dB', type=float, default=4.0, help='Gain in dB')
    
    args = parser.parse_args()

    apply_peak_eq(args.input_file, args.output_file, args.peak_freq, args.q_factor, args.gain_dB)
