from datasets import load_dataset
from transformers import (
    AutoFeatureExtractor,
    AutoTokenizer,
)

model_id = "openai/whisper-tiny"
feature_extractor = AutoFeatureExtractor.from_pretrained(model_id)
tokenizer = AutoTokenizer.from_pretrained(model_id)

dataset = load_dataset("google/fleurs", "en_us", split="train", streaming=True)

audio_column_name = "audio"
model_input_name = feature_extractor.model_input_names[0]
text_column_name = "raw_transcription"

def prepare_dataset(batch):
    # process audio
    sample = batch[audio_column_name]
    inputs = feature_extractor(sample["array"], sampling_rate=sample["sampling_rate"])
    # process audio length
    batch[model_input_name] = inputs.get(model_input_name)[0]
    batch["input_length"] = len(sample["array"])

    # process targets
    input_str = batch[text_column_name].lower()
    batch["labels"] = tokenizer(input_str).input_ids
    return batch

dataset = dataset.map(prepare_dataset)
for batch in dataset:
    print(batch)
    break