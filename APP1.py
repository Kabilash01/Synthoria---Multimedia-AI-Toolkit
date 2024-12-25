import sys
import os
import cv2
import torch
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QLabel, QFileDialog, QPushButton, QTextEdit,
    QVBoxLayout, QWidget, QHBoxLayout, QDialog, QLineEdit, QMessageBox, QInputDialog
)
from PyQt5.QtGui import QPixmap, QFont
from PyQt5.QtCore import Qt
from PIL import Image
from diffusers import StableDiffusionPipeline, DPMSolverMultistepScheduler, DiffusionPipeline
from diffusers.utils import export_to_video
from transformers import BlipProcessor, BlipForConditionalGeneration
from image_eraser import erase
from pydub import AudioSegment
from gtts import gTTS

# Google Gemini API configuration
try:
    from google.generativeai import configure as configure_google_gemini, GenerativeModel
    configure_google_gemini(api_key="AIzaSyCcvSMuS2wD6H6kuU3oYm3NyzzMh3Plg3I")  # Replace with your API key
except ImportError:
    configure_google_gemini = None
    GenerativeModel = None
    print("Google Gemini API not available. AI chat functionality will be disabled.")

# Check CUDA availability
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")


class ImageEraserDialog(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Image Eraser")
        self.setFixedSize(400, 300)

        layout = QVBoxLayout(self)

        self.source_folder = QLineEdit(self)
        self.source_folder.setPlaceholderText("Enter source folder path (e.g., 'images/cat/')")
        layout.addWidget(self.source_folder)

        self.output_folder = QLineEdit(self)
        self.output_folder.setPlaceholderText("Enter output folder path (e.g., 'images/output/')")
        layout.addWidget(self.output_folder)

        self.image_file = QLineEdit(self)
        self.image_file.setPlaceholderText("Enter image file name (e.g., 'input_1.jpg')")
        layout.addWidget(self.image_file)

        self.mask_file = QLineEdit(self)
        self.mask_file.setPlaceholderText("Enter mask file name (e.g., 'mask_1.jpg')")
        layout.addWidget(self.mask_file)

        self.result_file = QLineEdit(self)
        self.result_file.setPlaceholderText("Enter result file name (e.g., 'result_1.png')")
        layout.addWidget(self.result_file)

        self.erase_button = QPushButton("Erase Image", self)
        self.erase_button.clicked.connect(self.erase_image)
        layout.addWidget(self.erase_button)

    def erase_image(self):
        source_folder = self.source_folder.text().strip()
        output_folder = self.output_folder.text().strip()
        image_name = self.image_file.text().strip()
        mask_name = self.mask_file.text().strip()
        result_name = self.result_file.text().strip()

        if not (source_folder and output_folder and image_name and mask_name and result_name):
            QMessageBox.warning(self, "Input Error", "Please fill in all fields.")
            return

        try:
            os.makedirs(output_folder, exist_ok=True)
        except OSError as e:
            QMessageBox.critical(self, "Folder Error", f"Error creating output folder: {e}")
            return

        image_path = os.path.join(source_folder, image_name)
        mask_path = os.path.join(source_folder, mask_name)

        image = cv2.imread(image_path)
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)

        if image is None or mask is None:
            QMessageBox.critical(self, "File Error", "Could not load the image or mask. Please check the file paths.")
            return

        try:
            output = erase(image, mask, window=(22, 22))
            result_path = os.path.join(output_folder, result_name)
            cv2.imwrite(result_path, output)
            QMessageBox.information(self, "Success", f"Result saved as {result_name} in {output_folder}")
        except Exception as e:
            QMessageBox.critical(self, "Processing Error", f"Error processing the image: {e}")


class AIChatDialog(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Google Gemini AI Chat")
        self.setFixedSize(600, 400)

        layout = QVBoxLayout(self)

        self.chat_history = QTextEdit(self)
        self.chat_history.setReadOnly(True)
        self.chat_history.setStyleSheet("background-color: #404040; color: white;")
        layout.addWidget(self.chat_history)

        self.user_input = QLineEdit(self)
        self.user_input.setPlaceholderText("Type your message here...")
        self.user_input.setStyleSheet("background-color: #404040; color: white;")
        layout.addWidget(self.user_input)

        send_button = QPushButton("Send", self)
        send_button.setStyleSheet("background-color: #1E90FF; color: white;")
        send_button.clicked.connect(self.send_message)
        layout.addWidget(send_button)

        self.chat_model = None
        self.init_gemini_chat()
        
  
            
    def init_gemini_chat(self):
        try:
            self.chat_model = GenerativeModel()
            self.chat_history.append("AI Chat initialized. Start a conversation!")
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to initialize Gemini AI Chat: {e}")
            self.chat_model = None

    def send_message(self):
        user_message = self.user_input.text().strip()
        if not user_message:
            return

        try:
            response = self.chat_model.generate_content(user_message)
            self.chat_history.append(f"You: {user_message}")
            self.chat_history.append(f"AI: {response.text}")
            self.user_input.clear()
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Error sending message: {e}")



class MultimediaApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("SYNTHORIA Multimedia AI Toolkit")
        self.setGeometry(100, 100, 1000, 800)
        self.setStyleSheet("background-color: #2B2B2B; color: white;")

        self.central_widget = QWidget()
        self.setCentralWidget(self.central_widget)
        self.layout = QVBoxLayout(self.central_widget)

        self.add_section_label("SYNTHORIA AI STUDIO")
        self.image_label = self.add_preview_area()

        self.prompt_input = QTextEdit(self)
        self.prompt_input.setPlaceholderText("Enter your prompt here...")
        self.prompt_input.setStyleSheet("background-color: #404040; color: white; border: none;")
        self.layout.addWidget(self.prompt_input)

        button_layout = QHBoxLayout()
        self.layout.addLayout(button_layout)

        self.add_button(button_layout, "Load Image", self.load_image)
        self.add_button(button_layout, "Generate Image", self.generate_image)
        self.add_button(button_layout, "Generate Video", self.generate_video)
        self.add_button(button_layout, "Generate Caption", self.generate_caption)
        self.add_button(button_layout, "Erase Image", self.open_image_eraser)

        self.add_section_label("Audio AI Features")
        audio_button_layout = QHBoxLayout()
        self.layout.addLayout(audio_button_layout)

        self.add_button(audio_button_layout, "Upload Audio", self.upload_audio)
        self.add_button(audio_button_layout, "Generate Audio", self.generate_audio)
        self.add_button(audio_button_layout, "Change Voice", self.change_voice)
        self.add_button(audio_button_layout, "Trim Audio", self.trim_audio)

        self.add_section_label("CHATBOT (Google Gemini Flash 1.5)")
        self.chat_button = QPushButton("Open AI Chat", self)
        self.chat_button.setStyleSheet("background-color: #1E90FF; color: white; font-size: 14px;")
        self.chat_button.clicked.connect(self.open_ai_chat)
        self.layout.addWidget(self.chat_button, alignment=Qt.AlignCenter)

    def add_section_label(self, text):
        label = QLabel(text, self)
        label.setAlignment(Qt.AlignCenter)
        label.setFont(QFont("Arial", 16, QFont.Bold))
        self.layout.addWidget(label)

    def add_preview_area(self):
        preview = QLabel("Image/Video Preview", self)
        preview.setAlignment(Qt.AlignCenter)
        preview.setStyleSheet("border: 2px solid #555; padding: 10px;")
        self.layout.addWidget(preview)
        return preview

    def add_button(self, layout, text, callback):
        button = QPushButton(text, self)
        button.setStyleSheet("background-color: #404040; color: white;")
        button.clicked.connect(callback)
        layout.addWidget(button)

    def open_image_eraser(self):
        dialog = ImageEraserDialog(self)
        dialog.exec_()
        
    def open_ai_chat(self):
        if configure_google_gemini is None or GenerativeModel is None:
            QMessageBox.warning(self, "Unavailable", "Google Gemini AI Chat is not available. Check API setup.")
            return

        chat_dialog = AIChatDialog(self)
        chat_dialog.exec_()
        
    
    def load_image(self):
        file_name, _ = QFileDialog.getOpenFileName(self, "Open Image", "", "Images (*.png *.jpg *.bmp)")
        if file_name:
            pixmap = QPixmap(file_name)
            self.image_label.setPixmap(pixmap.scaled(500, 500, Qt.KeepAspectRatio))

  
    def generate_image(self):
        prompt = self.prompt_input.toPlainText().strip()
        if not prompt:
            QMessageBox.warning(self, "Input Error", "Please enter a prompt.")
            return

        if self.stable_diffusion_model is None:
            self.stable_diffusion_model = StableDiffusionPipeline.from_pretrained("CompVis/stable-diffusion-v1-4-original", torch_dtype=torch.float16)
            self.stable_diffusion_model.to(device)

        with torch.no_grad():
            generated_image = self.stable_diffusion_model(prompt).images[0]
            generated_image.save("generated_image.png")
            image = QPixmap("generated_image.png")
            self.image_label.setPixmap(image.scaled(500, 500, Qt.KeepAspectRatio))

    def generate_video(self):
        prompt = self.prompt_input.toPlainText().strip()
        if not prompt:
            QMessageBox.warning(self, "Input Error", "Please enter a prompt.")
            return

        if self.video_gen_model is None:
            self.video_gen_model = DiffusionPipeline.from_pretrained("CompVis/stable-diffusion-v1-4-video")

        generated_video = self.video_gen_model(prompt).images  # Assuming video generation is similar to image generation for demonstration purposes
        generated_video[0].save("generated_video.mp4")
        QMessageBox.information(self, "Success", "Video saved as generated_video.mp4")


        # Initialize BLIP processor and model
        self.caption_processor = None
        self.caption_model = None
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Layout and buttons
        button_layout = QVBoxLayout()
        self.add_button(button_layout, "Generate Caption", self.generate_caption)

    def add_button(self, layout, text, callback):
        button = QPushButton(text)
        button.clicked.connect(callback)
        layout.addWidget(button)

    def generate_caption(self):
        # Check if the processor and model are loaded
        if self.caption_processor is None or self.caption_model is None:
            self.caption_processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
            self.caption_model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base").to(self.device)

        # Try to load and caption the image
        try:
            image = Image.open("generated_image.png").convert("RGB")
            inputs = self.caption_processor(images=image, return_tensors="pt").to(self.device)
            outputs = self.caption_model.generate(**inputs)
            caption = self.caption_processor.decode(outputs[0], skip_special_tokens=True)
            QMessageBox.information(self, "Generated Caption", caption)
        except FileNotFoundError:
            QMessageBox.warning(self, "Error", "Image file not found.")
        except Exception as e:
            QMessageBox.warning(self, "Error", f"An error occurred: {str(e)}")






    def upload_audio(self):
        pass  # Implement audio upload functionality here

    def generate_audio(self):
        pass  # Implement audio generation functionality here

    def change_voice(self):
        pass  # Implement voice change functionality here

    def trim_audio(self):
        pass  # Implement audio trimming functionality here
    # Add other functions as needed...


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MultimediaApp()
    window.show()
    sys.exit(app.exec_())
