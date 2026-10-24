import os
from pydub import AudioSegment
from pydub.silence import split_on_silence

def preprocess_swahili_audio(input_folder, output_folder, segment_length_ms=5000):
    """
    Processes raw copyrighted audio files by normalizing volume, 
    removing long silences, and splitting them into uniform segments for AI training.
    """
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
        print(f"Created output directory: {output_folder}")

    # Supported audio and video extensions (ffmpeg extracts audio from video formats automatically)
    valid_extensions = ('.mp3', '.wav', '.flac', '.m4a', '.aac', '.ogg', '.opus', '.mp4', '.mkv', '.mov', '.webm')
    file_counter = 0

    # Walk recursively through all files and subdirectories
    for root, _, files in os.walk(input_folder):
        for filename in files:
            if filename.lower().endswith(valid_extensions):
                file_path = os.path.join(root, filename)
                rel_path = os.path.relpath(file_path, input_folder)
                print(f"Processing: {rel_path}")
            
            try:
                # 1. Load the audio asset
                audio = AudioSegment.from_file(file_path)
                
                # 2. Normalize volume (Target -20 dBFS for uniform training data)
                change_in_dBFS = -20.0 - audio.dBFS
                normalized_audio = audio.apply_gain(change_in_dBFS)
                
                # 3. Split on long silences to keep high-density speech/dialogue
                # Adjust silence_thresh based on the background noise of your movies
                chunks = split_on_silence(
                    normalized_audio,
                    min_silence_len=700,  # millisecond silence threshold
                    silence_thresh=-40,    # quietness threshold in dB
                    keep_silence=200       # leave a natural breath cushion
                )
                
                # Recombine valid chunks
                combined_audio = AudioSegment.empty()
                for chunk in chunks:
                    combined_audio += chunk
                
                # 4. Slice the dense audio into uniform chunks (e.g., 5-second tokens)
                for i in range(0, len(combined_audio), segment_length_ms):
                    segment = combined_audio[i:i + segment_length_ms]
                    
                    # Only save complete segments to avoid partial training steps
                    if len(segment) == segment_length_ms:
                        output_filename = f"tz_swahili_{file_counter:06d}.wav"
                        output_path = os.path.join(output_folder, output_filename)
                        
                        # Export as standard 16kHz mono WAV for clean machine learning inputs
                        segment.set_frame_rate(16000).set_channels(1).export(output_path, format="wav")
                        file_counter += 1
                        
            except Exception as e:
                print(f"Could not process file {filename}. Error: {e}")

    print(f"\nPreprocessing complete! Generated {file_counter} clean audio segments in: {output_folder}")

if __name__ == "__main__":
    # Define your local or cloud storage paths
    RAW_DATA_DIR = "./data/raw_copyrighted_assets"
    PROCESSED_DATA_DIR = "./data/processed_training_chunks"
    
    # Create dummy folders locally if they don't exist yet for testing
    if not os.path.exists(RAW_DATA_DIR):
        os.makedirs(RAW_DATA_DIR)
        print(f"Please drop your raw movie/song audio files into: {RAW_DATA_DIR}")
    else:
        preprocess_swahili_audio(RAW_DATA_DIR, PROCESSED_DATA_DIR)
