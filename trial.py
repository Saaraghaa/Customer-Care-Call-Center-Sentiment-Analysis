pip install transformers
import librosa
import torch
from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor
import IPython.display as display
processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-base-960h")
model = Wav2Vec2ForCTC.from_pretrained("facebook/wav2vec2-base-960h")
display.Audio("harvard.wav")
speech, _ = librosa.load('harvard.wav', sr=16000)
input_values = processor(speech, return_tensors = 'pt').input_values
input_values
logits = model(input_values).logits
logits
predicted_ids = torch.argmax(logits, dim = -1)
transcription = processor.decode(predicted_ids[0])
print(transcription)
