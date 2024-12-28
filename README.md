Here's a detailed README for your Synthoria project based on our chat:

---

# Synthoria - Multimedia AI Toolkit

Synthoria is a powerful multimedia AI toolkit that integrates a variety of advanced features for generating, enhancing, and manipulating images, videos, and audio. This project provides an all-in-one solution for creating stunning visuals, video content, and personalized audio using the latest AI technologies.

## Features

### Image Generation
- **Stable Diffusion Model Integration**: Generate high-quality images from textual prompts using the Stable Diffusion model.
- **Image Captioning**: Automatically generate captions for images using the BLIP captioning model.
- **Image Eraser**: Remove unwanted objects from images by using the `image_eraser` functionality.
- **Image Upload**: Allows users to upload images for further processing or generation.

### Video Generation
- **Text-to-Video Generation**: Generate video content from text descriptions using advanced video generation models.
- **Video Preview**: View the generated video preview before saving it.

### Audio Features
- **Audio Upload**: Upload audio files to the application for processing.
- **Audio Generation**: Generate custom audio clips using the `gTTS` (Google Text-to-Speech) API.
- **Voice Changing**: Modify the voice in audio files using various transformations (TBD).
- **Audio Trimming**: Trim audio files to select desired segments.
- **Audio Upscaling and Enhancement**: Improve audio quality by increasing the bitrate and enhancing the overall sound.

### Chatbot Functionality
- **Google Gemini Chatbot**: A fully integrated chatbot powered by Google Gemini's Flash 1.5 API. This allows users to interact with an AI chatbot, ask questions, and receive responses.

## Requirements

- **Python 3.x**
- **PyQt5**: Used for the graphical user interface.
- **Torch**: For running deep learning models.
- **Pillow**: For handling image operations.
- **diffusers**: For text-to-image and text-to-video functionalities.
- **transformers**: For BLIP image captioning.
- **gTTS**: For generating audio from text.
- **pydub**: For audio processing tasks such as trimming and enhancing.
- **huggingface_hub**: For downloading models from Hugging Face.
- **moviepy**: For handling video creation and processing.

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/Kabilash01/synthoria.git
   cd synthoria
   ```

2. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```

   If you encounter issues with `moviepy` or `ffmpeg`, you may need to manually install `ffmpeg`:
   ```bash
   pip install moviepy
   ```

3. Make sure you have an API key for Hugging Face and Google Gemini if you plan to use these services.

## Usage

### 1. Launch the Application

Run the main application using the following command:
```bash
python main.py
```

### 2. Features Overview

- **Image Generation**: Enter a text prompt, and click "Generate Image" to create a custom image.
- **Image Captioning**: Upload an image, and the app will generate a caption for it.
- **Video Generation**: Input a text prompt for video creation and view the video preview.
- **Audio Features**:
  - Upload your audio and perform actions like trimming, changing voices, or upscaling.
  - Generate audio by typing text and selecting a voice option.
- **Chatbot**: Use the integrated Google Gemini chatbot by clicking "Open AI Chat" and entering your query.

### 3. Google Gemini API Configuration

Ensure you have a valid Google Gemini API key. Replace the `api_key` in the `configure_google_gemini` function within the code.

### 4. Stable Diffusion Model Integration

The image and video generation functionalities rely on the Hugging Face `StableDiffusionPipeline` and other models. You'll need to authenticate with your Hugging Face account and provide the necessary API tokens.

### 5. Audio Enhancement

The `pydub` library is used for audio manipulation, including trimming, upscaling, and enhancement. Ensure you have `ffmpeg` installed on your system to handle audio operations.

## Development and Contributions

### Contributing

Feel free to contribute to the project by opening issues, forking the repository, and submitting pull requests. Here are some ways you can contribute:

- **Add new AI features**: Improve existing features or add new ones like emotion-based voice generation.
- **Bug fixes**: Help debug any issues that arise during usage.
- **Documentation**: Update and improve the documentation for better usability.

### Project Structure

- `main.py`: The main file that runs the application.
- `image_eraser.py`: Contains the logic for erasing unwanted objects from images.
- `audio_processing.py`: Handles various audio functionalities like trimming and enhancement.
- `video_processing.py`: Manages video generation and editing tasks.
- `chatbot.py`: Integrates the Google Gemini chatbot for conversational AI.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- **Hugging Face**: For providing various pretrained models for text-to-image and text-to-video generation.
- **Google Gemini**: For providing the chatbot API.
- **PyQt5**: For building the GUI.

---

This README provides a comprehensive guide to using the features in the Synthoria project. Let me know if you'd like any changes or additional information!
