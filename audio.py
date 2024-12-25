import sys
import os
import numpy as np
import librosa
import soundfile as sf
from pydub import AudioSegment
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QLabel, QPushButton, QFileDialog, QVBoxLayout, 
    QHBoxLayout, QWidget, QTextEdit, QSlider
)
from PyQt5.QtCore import Qt

# ==============================
# Audio Functions
# ==============================

# Text-to-Audio using basic synthetic wave generation
def generate_audio(text, output_path="generated_audio.wav"):
    try:
        import pyttsx3
        engine = pyttsx3.init()
        engine.save_to_file(text, output_path)
        engine.runAndWait()
        return output_path
    except Exception as e:
        print(f"Error generating audio: {e}")
        return None

# Voice Changer: Change pitch and speed
def change_voice(input_path, output_path, pitch_factor=1.0, speed_factor=1.0):
    try:
        data, sr = librosa.load(input_path, sr=None)  # Load audio with original sampling rate
        data = librosa.effects.pitch_shift(data, sr, n_steps=pitch_factor)
        data = librosa.effects.time_stretch(data, speed_factor)
        sf.write(output_path, data, sr)
        return output_path
    except Exception as e:
        print(f"Error changing voice: {e}")
        return None

# Trim Audio
def trim_audio(input_path, start_time, end_time, output_path):
    try:
        audio = AudioSegment.from_file(input_path)
        trimmed_audio = audio[start_time * 1000:end_time * 1000]
        trimmed_audio.export(output_path, format="wav")
        return output_path
    except Exception as e:
        print(f"Error trimming audio: {e}")
        return None

# Enhance Audio
def enhance_audio(input_path, output_path):
    try:
        data, sr = librosa.load(input_path, sr=None)
        enhanced_audio = librosa.effects.preemphasis(data)
        sf.write(output_path, enhanced_audio, sr)
        return output_path
    except Exception as e:
        print(f"Error enhancing audio: {e}")
        return None

# Noise Cancellation
def noise_cancellation(input_path, output_path, aggressiveness=2):
    try:
        import webrtcvad
        vad = webrtcvad.Vad(aggressiveness)
        data, sample_rate = sf.read(input_path)

        if sample_rate != 16000:
            raise ValueError("Sample rate must be 16 kHz for WebRTC VAD.")

        frame_size = int(sample_rate * 0.01)  # 10ms frames
        frames = [data[i:i + frame_size] for i in range(0, len(data), frame_size)]

        filtered_audio = np.concatenate([
            frame for frame in frames if vad.is_speech(frame.tobytes(), sample_rate)
        ])
        sf.write(output_path, filtered_audio, sample_rate)
        return output_path
    except Exception as e:
        print(f"Error in noise cancellation: {e}")
        return None

# Audio Mixing
def mix_audio(tracks, output_path):
    try:
        audio_segments = [AudioSegment.from_file(track) for track in tracks]
        mixed_audio = audio_segments[0]
        for segment in audio_segments[1:]:
            mixed_audio = mixed_audio.overlay(segment)
        mixed_audio.export(output_path, format="wav")
        return output_path
    except Exception as e:
        print(f"Error in audio mixing: {e}")
        return None

# ==============================
# GUI Implementation
# ==============================

class AudioStudio(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Audio AI Studio")
        self.setGeometry(100, 100, 800, 600)

        self.central_widget = QWidget()
        self.setCentralWidget(self.central_widget)
        self.layout = QVBoxLayout()
        self.central_widget.setLayout(self.layout)

        self.text_input = QTextEdit()
        self.text_input.setPlaceholderText("Enter text for audio generation...")
        self.layout.addWidget(self.text_input)

        self.upload_button = QPushButton("Upload Audio File")
        self.upload_button.clicked.connect(self.upload_file)
        self.layout.addWidget(self.upload_button)

        self.generate_button = QPushButton("Generate Audio")
        self.generate_button.clicked.connect(self.generate_audio)
        self.layout.addWidget(self.generate_button)

        self.voice_change_button = QPushButton("Change Voice")
        self.voice_change_button.clicked.connect(self.change_voice)
        self.layout.addWidget(self.voice_change_button)

        self.trim_button = QPushButton("Trim Audio")
        self.trim_button.clicked.connect(self.trim_audio)
        self.layout.addWidget(self.trim_button)

        self.enhance_button = QPushButton("Enhance Audio")
        self.enhance_button.clicked.connect(self.enhance_audio)
        self.layout.addWidget(self.enhance_button)

        self.noise_cancel_button = QPushButton("Noise Cancellation")
        self.noise_cancel_button.clicked.connect(self.noise_cancellation)
        self.layout.addWidget(self.noise_cancel_button)

        self.mix_button = QPushButton("Mix Audio")
        self.mix_button.clicked.connect(self.mix_audio)
        self.layout.addWidget(self.mix_button)

        self.output_label = QLabel("Output will appear here.")
        self.layout.addWidget(self.output_label)

    def upload_file(self):
        self.file_path, _ = QFileDialog.getOpenFileName(self, "Open Audio File")
        self.output_label.setText(f"File Selected: {self.file_path}")

    def generate_audio(self):
        text = self.text_input.toPlainText()
        output_path = "generated_audio.wav"
        result = generate_audio(text, output_path)
        self.output_label.setText(f"Audio Generated: {output_path}" if result else "Failed to generate audio.")

    def change_voice(self):
        if hasattr(self, 'file_path'):
            output_path = "voice_changed_audio.wav"
            pitch = 2  # Adjust as needed
            speed = 1.0  # Adjust as needed
            result = change_voice(self.file_path, output_path, pitch, speed)
            self.output_label.setText(f"Voice Changed: {output_path}" if result else "Failed to change voice.")

    def trim_audio(self):
        if hasattr(self, 'file_path'):
            output_path = "trimmed_audio.wav"
            result = trim_audio(self.file_path, 5, 10, output_path)
            self.output_label.setText(f"Trimmed Audio: {output_path}" if result else "Failed to trim audio.")

    def enhance_audio(self):
        if hasattr(self, 'file_path'):
            output_path = "enhanced_audio.wav"
            result = enhance_audio(self.file_path, output_path)
            self.output_label.setText(f"Enhanced Audio: {output_path}" if result else "Failed to enhance audio.")

    def noise_cancellation(self):
        if hasattr(self, 'file_path'):
            output_path = "noise_cancelled_audio.wav"
            result = noise_cancellation(self.file_path, output_path)
            self.output_label.setText(f"Noise Cancelled: {output_path}" if result else "Failed to cancel noise.")

    def mix_audio(self):
        tracks = QFileDialog.getOpenFileNames(self, "Select Audio Tracks to Mix")[0]
        if tracks:
            output_path = "mixed_audio.wav"
            result = mix_audio(tracks, output_path)
            self.output_label.setText(f"Mixed Audio: {output_path}" if result else "Failed to mix audio.")


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = AudioStudio()
    window.show()
    sys.exit(app.exec_())
