from moviepy.editor import VideoFileClip
import librosa
import numpy as np
import os

# Function to extract audio from video
def extract_audio_from_video(video_path, output_audio_path):
    video = VideoFileClip(video_path)
    audio = video.audio
    audio.write_audiofile(output_audio_path)
    video.close()

# Function to evaluate audio coherence and score segments
def evaluate_audio_coherence(audio_path, threshold_variation=0.2, energy_threshold=0.1, analysis_window=0.1):
    y, sr = librosa.load(audio_path, sr=None) 
    
    #set audio_lenght in seconds
    audio_length = len(y) / sr

    # Calculate frame length and hop length based on analysis window
    frame_length = int(sr * analysis_window)  # Number of samples in one window
    hop_length = frame_length // 2  # Overlap of 50%
    
    # Calculate energy in sliding windows
    energy = np.array([
        sum(abs(y[i:i+frame_length]**2)) for i in range(0, len(y), hop_length)
    ])
    
    # Calculate energy variations
    energy_diff = np.diff(energy) / np.max(np.abs(np.diff(energy)))  # Normalize variations
    
    # Identify significant variations
    positive_variations = np.where(energy_diff > threshold_variation)[0]
    negative_variations = np.where(energy_diff < -threshold_variation)[0]
    
    # Convert indices to time in seconds
    positive_times = positive_variations * hop_length / sr
    negative_times = negative_variations * hop_length / sr
    
    # Scoring system
    segment_scores = {}
    
    # Add 3 points for positive variations
    for time in positive_times:
        segment_scores[time] = segment_scores.get(time, 0) + 3
    
    # Add 3 points for negative variations
    for time in negative_times:
        segment_scores[time] = segment_scores.get(time, 0) + 3
    
    # Add 2 points for high-energy segments
    for i, e in enumerate(energy):
        if e / np.max(energy) > energy_threshold:
            time = i * hop_length / sr
            segment_scores[time] = segment_scores.get(time, 0) + 2
    
    #for every second not rated in the audio, rate 0
    for i in range(0, int(audio_length)+2):
        time = i
        if time not in segment_scores.keys():
            segment_scores[time] = 0

    #video length


    # Sort by time
    scored_segments = sorted(segment_scores.items())
    return scored_segments

# Function to save scored segments to a text file
def save_scores_to_file(scored_segments, output_file):
    with open(output_file, "w") as f:
        for time, score in scored_segments:
            f.write(f"{time:.2f}, {score}\n")

# Paths and parameters
video_path = "clip.mp4"
# video_path = "bigclip.mp4"
audio_path = "temp_audio.wav"
output_file = "scored_segments.txt"
threshold_variation = 0.4 
energy_threshold = 0.2
analysis_window = 2

# Run the analysis
extract_audio_from_video(video_path, audio_path)
scored_segments = evaluate_audio_coherence(audio_path, threshold_variation, energy_threshold, analysis_window)

# Save results to a text file
save_scores_to_file(scored_segments, output_file)

# Clean up temporary audio file
if os.path.exists(audio_path):
    os.remove(audio_path)

#print(scored_segments)
scored_segments