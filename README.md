# Text-to-Lipsync

-> Divided into 2 parts: Text to Speech and Speech to Lip-Sync Generation

Text to Speech:

textcode.py -> Code for the TTS in 6 languages

TTS Evaluation -> Folder has the code, data used for the evaluation of the TTS model and Results of the evaluation in the form of csv files.

Speech to Lip-Sync Model: (Refer https://github.com/Rudrabha/Wav2Lip for models)

checkpoints -> Trained models will be saved here ( wav2lip_lrs2pretrainedmodel.pth is pretrained model and wav2lip_griddataset.pth is the model obtained for us after training)

face_detection -> Code for the face detection used in the preprocessing and training

models -> code for the models

results -> The generated videos will be stored here.

syncnet -> pretrained model used for the evaluation of the model.

audio.py -> to generate mel-spectograms used for training

calculate_scores_LRS.y -> Code for the evaluation metrics

hparams.py -> code for hyperparameters used in the training

lipsync.py -> code for generating lipsync video.


Publication: D. Kaushik, K. P. Karthik, T. Satvik Gupta and S. Vekkot, "Realistic Lip-Sync Generation from Text for Multimodal Applications," 2025 IEEE International Conference on Interdisciplinary Approaches in Technology and Management for Social Innovation (IATMSI), Gwalior, India, 2025, pp. 1-7, doi: 10.1109/IATMSI64286.2025.10984895.
preprocess,py, Preprocessing.ipynb -> code for pre-processing

wav2lip_train.py -> code for training the model.


texttosync.py -> Integrated code to get lip-sync video using a text input.
