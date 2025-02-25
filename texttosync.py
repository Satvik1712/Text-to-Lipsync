from os import listdir, path
import numpy as np
import scipy, cv2, os, sys
import json, subprocess, random, string
from tqdm import tqdm
from glob import glob
import torch, face_detection
from models import Wav2Lip
import platform
import argparse
import audio
from transformers import AutoModel, AutoTokenizer
import scipy.io.wavfile
from IPython.display import Audio

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Using {device} for inference.")

# Dictionary to map languages to model names for TTS
language_models = {
    "english": "facebook/mms-tts-eng",
    "hindi": "facebook/mms-tts-hin",
    "kannada": "facebook/mms-tts-kan",
    "malayalam": "facebook/mms-tts-mal",
    "tamil": "facebook/mms-tts-tam",
    "telugu": "facebook/mms-tts-tel"
}

def tts_generate_audio():
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
    return output_path

def get_user_inputs():
    print("Welcome to Wav2Lip Inference!")
    use_tts = input("Do you want to generate audio using TTS? (yes/no): ").strip().lower()
    if use_tts == "yes":
        audio_path = tts_generate_audio()
    else:
        audio_path = input("Enter the path to the audio file (WAV format preferred): ").strip()
        if not path.exists(audio_path):
            raise ValueError("Audio file path does not exist!")

    checkpoint_path = input("Enter the path to the checkpoint file: ").strip()
    face_path = input("Enter the path to the video/image file containing the face: ").strip()
    outfile_path = input("Enter the output file path (e.g., results/result_voice.mp4): ").strip()

    if not path.exists(checkpoint_path):
        raise ValueError("Checkpoint file path does not exist!")
    if not path.exists(face_path):
        raise ValueError("Video/Image file path does not exist!")

    return checkpoint_path, face_path, audio_path, outfile_path

args = argparse.Namespace()
args.static = False
args.fps = 25.0
args.pads = [0, 10, 0, 0]
args.face_det_batch_size = 16
args.wav2lip_batch_size = 128
args.resize_factor = 1
args.crop = [0, -1, 0, -1]
args.box = [-1, -1, -1, -1]
args.rotate = False
args.nosmooth = False
args.img_size = 96

checkpoint_path, face_path, audio_path, outfile_path = get_user_inputs()
args.checkpoint_path = checkpoint_path
args.face = face_path
args.audio = audio_path
args.outfile = outfile_path

def get_smoothened_boxes(boxes, T):
    for i in range(len(boxes)):
        if i + T > len(boxes):
            window = boxes[len(boxes) - T:]
        else:
            window = boxes[i : i + T]
        boxes[i] = np.mean(window, axis=0)
    return boxes

def face_detect(images):
    detector = face_detection.FaceAlignment(face_detection.LandmarksType._2D, 
                                            flip_input=False, device=device)

    batch_size = args.face_det_batch_size
    
    while True:
        predictions = []
        try:
            for i in tqdm(range(0, len(images), batch_size)):
                predictions.extend(detector.get_detections_for_batch(np.array(images[i:i + batch_size])))
        except RuntimeError:
            if batch_size == 1: 
                raise RuntimeError('Image too big to run face detection on GPU. Please use the --resize_factor argument')
            batch_size //= 2
            print('Recovering from OOM error; New batch size: {}'.format(batch_size))
            continue
        break

    results = []
    pady1, pady2, padx1, padx2 = args.pads
    for rect, image in zip(predictions, images):
        if rect is None:
            cv2.imwrite('temp/faulty_frame.jpg', image)
            raise ValueError('Face not detected! Ensure the video contains a face in all the frames.')

        y1 = max(0, rect[1] - pady1)
        y2 = min(image.shape[0], rect[3] + pady2)
        x1 = max(0, rect[0] - padx1)
        x2 = min(image.shape[1], rect[2] + padx2)
        
        results.append([x1, y1, x2, y2])

    boxes = np.array(results)
    if not args.nosmooth: boxes = get_smoothened_boxes(boxes, T=5)
    results = [[image[y1: y2, x1:x2], (y1, y2, x1, x2)] for image, (x1, y1, x2, y2) in zip(images, boxes)]

    del detector
    return results 

def load_model(path):
    model = Wav2Lip()
    print("Load checkpoint from: {}".format(path))
    checkpoint = torch.load(path, map_location=device)
    model.load_state_dict({k.replace('module.', ''): v for k, v in checkpoint["state_dict"].items()})
    return model.to(device).eval()

def datagen(frames, mels):
    img_batch, mel_batch, frame_batch, coords_batch = [], [], [], []

    if args.box[0] == -1:
        if not args.static:
            face_det_results = face_detect(frames)
        else:
            face_det_results = face_detect([frames[0]])
    else:
        print('Using the specified bounding box instead of face detection...')
        y1, y2, x1, x2 = args.box
        face_det_results = [[f[y1:y2, x1:x2], (y1, y2, x1, x2)] for f in frames]

    for i, m in enumerate(mels):
        idx = 0 if args.static else i % len(frames)
        frame_to_save = frames[idx].copy()
        face, coords = face_det_results[idx].copy()

        face = cv2.resize(face, (args.img_size, args.img_size))

        img_batch.append(face)
        mel_batch.append(m)
        frame_batch.append(frame_to_save)
        coords_batch.append(coords)

        if len(img_batch) >= args.wav2lip_batch_size:
            img_batch, mel_batch = np.asarray(img_batch), np.asarray(mel_batch)

            img_masked = img_batch.copy()
            img_masked[:, args.img_size // 2:] = 0

            img_batch = np.concatenate((img_masked, img_batch), axis=3) / 255.
            mel_batch = np.reshape(mel_batch, [len(mel_batch), mel_batch.shape[1], mel_batch.shape[2], 1])

            yield img_batch, mel_batch, frame_batch, coords_batch
            img_batch, mel_batch, frame_batch, coords_batch = [], [], [], []

    if len(img_batch) > 0:
        img_batch, mel_batch = np.asarray(img_batch), np.asarray(mel_batch)

        img_masked = img_batch.copy()
        img_masked[:, args.img_size // 2:] = 0

        img_batch = np.concatenate((img_masked, img_batch), axis=3) / 255.
        mel_batch = np.reshape(mel_batch, [len(mel_batch), mel_batch.shape[1], mel_batch.shape[2], 1])

        yield img_batch, mel_batch, frame_batch, coords_batch

mel_step_size = 16

def main():
    if not os.path.isfile(args.face):
        raise ValueError('--face argument must be a valid path to video/image file')

    full_frames = []
    if args.face.split('.')[-1].lower() in ['jpg', 'png', 'jpeg']:
        full_frames = [cv2.imread(args.face)]
        fps = args.fps
    else:
        video_stream = cv2.VideoCapture(args.face)
        fps = video_stream.get(cv2.CAP_PROP_FPS)

        print('Reading video frames...')
        while True:
            still_reading, frame = video_stream.read()
            if not still_reading:
                video_stream.release()
                break
            if args.resize_factor > 1:
                frame = cv2.resize(frame, (frame.shape[1] // args.resize_factor, frame.shape[0] // args.resize_factor))
            if args.rotate:
                frame = cv2.rotate(frame, cv2.cv2.ROTATE_90_CLOCKWISE)
            y1, y2, x1, x2 = args.crop
            if x2 == -1: x2 = frame.shape[1]
            if y2 == -1: y2 = frame.shape[0]
            frame = frame[y1:y2, x1:x2]
            full_frames.append(frame)

    print("Number of frames available for inference: ", len(full_frames))

    # Convert audio if needed
    if not args.audio.endswith('.wav'):
        print('Extracting raw audio...')
        command = f'ffmpeg -y -i {args.audio} -strict -2 temp/temp.wav'
        subprocess.call(command, shell=True)
        args.audio = 'temp/temp.wav'

    wav = audio.load_wav(args.audio, 16000)
    mel = audio.melspectrogram(wav)

    mel_chunks = []
    mel_idx_multiplier = 80. / fps 
    i = 0
    while True:
        start_idx = int(i * mel_idx_multiplier)
        if start_idx + mel_step_size > len(mel[0]):
            mel_chunks.append(mel[:, len(mel[0]) - mel_step_size:])
            break
        mel_chunks.append(mel[:, start_idx: start_idx + mel_step_size])
        i += 1

    full_frames = full_frames[:len(mel_chunks)]

    # Inference
    model = load_model(args.checkpoint_path)
    print("Model loaded")

    frame_h, frame_w = full_frames[0].shape[:-1]
    out = cv2.VideoWriter('temp/result.avi', cv2.VideoWriter_fourcc(*'DIVX'), fps, (frame_w, frame_h))

    for i, (img_batch, mel_batch, frames, coords) in enumerate(tqdm(datagen(full_frames.copy(), mel_chunks))):
        img_batch = torch.FloatTensor(np.transpose(img_batch, (0, 3, 1, 2))).to(device)
        mel_batch = torch.FloatTensor(np.transpose(mel_batch, (0, 3, 1, 2))).to(device)

        with torch.no_grad():
            pred = model(mel_batch, img_batch)

        pred = pred.cpu().numpy().transpose(0, 2, 3, 1) * 255.

        for p, f, c in zip(pred, frames, coords):
            y1, y2, x1, x2 = c
            p = cv2.resize(p.astype(np.uint8), (x2 - x1, y2 - y1))
            f[y1:y2, x1:x2] = p
            out.write(f)

    out.release()

    # Combine audio and video
    command = f'ffmpeg -y -i {args.audio} -i temp/result.avi -strict -2 -q:v 1 {args.outfile}'
    subprocess.call(command, shell=True)
    print(f"Video saved successfully at {args.outfile}")

if __name__ == '__main__':
    main()
