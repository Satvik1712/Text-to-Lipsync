from transformers import AutoModel, AutoTokenizer
import torch
import numpy as np
import scipy.io.wavfile
from IPython.display import Audio

# Dictionary to map languages to model names
language_models = {
    "english": "facebook/mms-tts-eng",
    "hindi": "facebook/mms-tts-hin",
    "kannada": "facebook/mms-tts-kan",
    "malayalam": "facebook/mms-tts-mal",
    "tamil": "facebook/mms-tts-tam",
    "telugu": "facebook/mms-tts-tel"
}

# Get user inputs
print("Available languages:", ", ".join(language_models.keys()))
language = input("Enter the language (e.g., english, hindi): ").strip().lower()

if language not in language_models:
    raise ValueError("Selected language is not supported!")

text = input("Enter the text to convert to audio: ").strip()
output_path = input("Enter the path to save the audio file (e.g., example.wav): ").strip()

# Load the model and tokenizer for the selected language
model_name = language_models[language]
model = AutoModel.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)

# Tokenize the input text
inputs = tokenizer(text, return_tensors="pt")

# Generate the audio waveform
with torch.no_grad():
    output = model(**inputs).waveform

# Convert waveform to proper format for saving
data = output.squeeze().float().numpy()  # Remove batch dimension
scaled_data = np.int16(data * 32767)  # Scale data to int16 range

# Validate and retrieve the sampling rate
sampling_rate = model.config.sampling_rate
if not (8000 <= sampling_rate <= 48000):
    raise ValueError(f"Invalid sampling rate: {sampling_rate}")

# Save the audio file
scipy.io.wavfile.write(output_path, rate=sampling_rate, data=scaled_data)

print(f"Audio saved successfully at {output_path}.")

# Optionally play the audio in Jupyter Notebook
Audio(output_path, rate=sampling_rate)
