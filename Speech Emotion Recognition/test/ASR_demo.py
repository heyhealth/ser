from transformers import Wav2Vec2Processor, Wav2Vec2ForCTC
from transformers.models.wav2vec2 import Wav2Vec2CTCTokenizer,Wav2Vec2ForCTC


from datasets import load_dataset
import torch
from main.opts import ARGS

dataset = load_dataset("hf-internal-testing/librispeech_asr_demo", "clean", split="validation")
dataset = dataset.sort("id")
sampling_rate = dataset.features["audio"].sampling_rate

# text : "BE QUICK THERE'S A GOOD FELLOW I WANT TO GET AWAY AT ONCE"


processor = Wav2Vec2Processor.from_pretrained(ARGS.MODEL_NAME_Wav2Vec2)
model = Wav2Vec2ForCTC.from_pretrained(ARGS.MODEL_NAME_Wav2Vec2)

# audio file is decoded on the fly
inputs = processor(dataset[0]["audio"]["array"], sampling_rate=sampling_rate, return_tensors="pt")
with torch.no_grad():
    logits = model(**inputs).logits
predicted_ids = torch.argmax(logits, dim=-1)

# transcribe speech
transcription = processor.batch_decode(predicted_ids)
print(transcription[0])
# 'MISTER QUILTER IS THE APOSTLE OF THE MIDDLE CLASSES AND WE ARE GLAD TO WELCOME HIS GOSPEL'

# compute loss
inputs["labels"] = processor(text=dataset[0]["text"], return_tensors="pt").input_ids

# compute loss
loss = model(**inputs).loss
print(round(loss.item(), 2))
