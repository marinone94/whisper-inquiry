""" Whisper training script using Hugging Face Transformers. """

import os  # used to create output directory
from dataclasses import dataclass  # used to define data collator
from math import ceil  # used to round up decimals

import evaluate  # used to import and compute evaluation metrics
import torch  # used to know if a GPU with CUDA is available
import wandb  # used for experiment tracking
from datasets import IterableDatasetDict, load_dataset  # used to load the dataset in streaming mode
from transformers import (
    AutoConfig,  # used to load model configurations
    AutoModelForSpeechSeq2Seq,  # used to load the model architecture and weights
    AutoProcessor,  # used to load the Whisper processor, which includes a feature extractor and a tokenizer
    Seq2SeqTrainer,  # used to perform training and evaluation loops
    Seq2SeqTrainingArguments,  # used to define training hyperparameters
    TrainerCallback,  # used to shuffle the training data after each epoch
    WhisperProcessor  # used for static data typing 
)
from transformers import set_seed  # used for reproducibility
from transformers.models.whisper.english_normalizer import BasicTextNormalizer  # used to normalize transcript and reference before evaluation
from transformers.trainer_pt_utils import IterableDataset, IterableDatasetShard  # used to shuffle the training data after each epoch

"""Then, we will load processor, model configuration, architecture and weights, and the dataset (in streaming mode). The English split of Fleurs is not a massive dataset, thus we could easily download it and store it in memory, but it is good to learn how to use the streaming mode if you were to fine-tune your model on larger datasets. """

model_id = "openai/whisper-tiny"
processor = AutoProcessor.from_pretrained(model_id)
config = AutoConfig.from_pretrained(model_id)
model = AutoModelForSpeechSeq2Seq.from_pretrained(model_id)

dataset_id = "google/fleurs"
dataset_language_code = "sv_se"
dataset = load_dataset(dataset_id, dataset_language_code, streaming=True)

"""The first time you run this code, make sure everything works fine using a small sample and low number of training steps. Just uncomment the next cell and run it. One note: since the dataset is loaded in streaming mode, the instruction will not be executed immediately. Instead, the dataset will be subsampled only when data will be needed during training."""

test_script = True
# test_script = False

## Sample dataset for testing
if test_script is True:
    dataset["train"] = dataset["train"].shuffle(seed=42).take(8)
    dataset["validation"] = dataset["validation"].shuffle(seed=42).take(4)
    dataset["test"] = dataset["test"].shuffle(seed=42).take(4)

"""The raw dataset is not yet ready for training. As described in my first about Whisper, the input audio waveform needs to be transformed into a Log-mel Spectrogram. I recommend you to read the [Audio Preprocessing section](https://marinone94.github.io/Whisper-paper/#audio-preprocessing) to understand the process. For the scope of this article, you should just know that the audio is translated from the time domain to its frequency representation using a sliding window, and adjusted to simulate human hearing. The Whisper Feature Extractor included in the Whisper Processor will take care of the rest.

Furthermore, the reference transcripts need to be tokenized, since the model outputs one token at the time and they are used to compute the loss during training. Again, the Tokenizer will take care of that, but the task needs to be included in the preprocessing step.

When we introduced the WER metric, we learned about the importance of normalizing the texts. But should we do that also before training? That is up to you, but you should remember that Whisper models have been pretrained to predict Capitalization, digits, and punctuation. So if you normalize the reference teanscripts before fine-tuning, you will teach model not to predict capital letters, digits, and punctuations. This does not mean that the model will never predict them, since it has been extensively pretrained to do so. To wrap up, your choice should depend on the final application and the dataset size, but in general I recommend not to normalize the references before training.

Finally, by storing the input features in the default model input name, the trainer will automatically pick the correct ones during training. Thus, don't hard-code it!
"""

normalizer = BasicTextNormalizer()
# model_input_name = 'input_features'
model_input_name = processor.feature_extractor.model_input_names[0]

def prepare_dataset(batch, normalize=False):
    # process audio
    sample = batch["audio"]
    inputs = processor.feature_extractor(sample["array"], sampling_rate=sample["sampling_rate"])
    # process audio length
    batch[model_input_name] = inputs.get(model_input_name)[0]
    batch["input_length"] = len(sample["array"])

    # process targets
    if normalize is True:
        labels = batch["raw_transcription"].lower()
        labels = normalizer(labels).strip()
    else:
        labels = batch["raw_transcription"].strip()
    batch["labels"] = processor.tokenizer(labels).input_ids
    return batch

"""We will use the `.map` method to apply our preprocessing function to the whole dataset. At the same time, we will drop all the columns which are not strictly needed during training. Since `input_features`, `ìnput_length` and `labels` are not features of the raw dataset, we can remove all the original ones. Finally, we will convert the dataset features to `torch` type since the dataset has no `__len__`property (again, we are in streaming mode). """

# dataset["train"].features is like a dict
# train, validation and test splits have the same features
raw_datasets_features = list(dataset["train"].features.keys())
preprocessed_dataset = IterableDatasetDict()

preprocessed_dataset["train"] = dataset["train"].map(
    prepare_dataset,
    remove_columns=raw_datasets_features,
    fn_kwargs={"normalize": False},  # needed only if default value and provided value differ
).with_format("torch")
preprocessed_dataset["validation"] = dataset["validation"].map(
    prepare_dataset,
    remove_columns=raw_datasets_features,
    fn_kwargs={"normalize": False},  # reference transripts are normalized in the evaluation function
).with_format("torch")
preprocessed_dataset["test"] = dataset["test"].map(
    prepare_dataset,
    remove_columns=raw_datasets_features,
    fn_kwargs={"normalize": False},  # reference transripts are normalized in the evaluation function
).with_format("torch")

"""Since we want to evaluate our model on the validation set during training, we also need to provide a method that computes the metrics given the model predictions. It looks very similar to the function we introduced above, but since it will receive a single prediction object, we need to extract the predicted tokens and the corresponding labels. Furthermore, we replace the label ids equal to -100 with the padding token. A couple of minutes of patience and you will understand why.

When decoding the prediction and the labels, we need to discard the special tokens. Those are used to force the model to perform specific tasks. You can read more [here](https://marinone94.github.io/Whisper-paper/#tasks).
"""

metric = evaluate.load("wer")

def compute_metrics(pred):
    # extract predicted tokens 
    pred_ids = pred.predictions
    label_ids = pred.label_ids

    # pad tokens will then be discarded by the tokenizer with all other special tokens
    label_ids[label_ids == -100] = processor.tokenizer.pad_token_id

    # decode transcripts and reference
    pred_str = processor.batch_decode(pred_ids, skip_special_tokens=True)
    label_str = processor.batch_decode(label_ids, skip_special_tokens=True)

    # normalize transcript and reference
    pred_str = [normalizer(pred) for pred in pred_str]
    label_str = [normalizer(label) for label in label_str]

    # only evaluate the samples that correspond to non-zero references
    pred_str = [pred_str[i] for i in range(len(pred_str)) if len(label_str[i]) > 0]
    label_str = [label_str[i] for i in range(len(label_str)) if len(label_str[i]) > 0]

    # express WER as percentage
    wer = 100 * metric.compute(predictions=pred_str, references=label_str)

    return {"wer": wer}

"""Alright, we are almost done preparing our dataset. Quite a lot of work, I know, but that is most of the job.

The last step is to define a data collator, which will build data btaches from the datasets during training using the Whisper Processor. It will also pad input features and labels.

Also, in the metrics computation method we replaced the labels with id equal to -100. It was done because the data collator **must** set the padding tokens to -100 so that the trainer will ignore them when computing the loss. That was the reverse step.
"""

@dataclass
class DataCollatorSpeechSeq2SeqWithPadding:

    processor: WhisperProcessor
    decoder_start_token_id: int

    def __call__(self, features):
        # split inputs and labels since they have to be of different lengths and need
        # different padding methods
        model_input_name = self.processor.model_input_names[0]
        input_features = [{model_input_name: feature[model_input_name]} for feature in features]
        label_features = [{"input_ids": feature["labels"]} for feature in features]

        batch = self.processor.feature_extractor.pad(input_features, return_tensors="pt")

        labels_batch = self.processor.tokenizer.pad(label_features, return_tensors="pt")

        # replace padding with -100 to ignore loss correctly
        labels = labels_batch["input_ids"].masked_fill(labels_batch.attention_mask.ne(1), -100)

        # if bos token is appended in previous tokenization step,
        # cut bos token here as it's append later anyways
        if (labels[:, 0] == self.decoder_start_token_id).all().cpu().item():
            labels = labels[:, 1:]

        batch["labels"] = labels

        return batch

data_collator = DataCollatorSpeechSeq2SeqWithPadding(
    processor=processor,
    decoder_start_token_id=model.config.decoder_start_token_id,
)

"""Next step was something I would have definitely missed I had not attended the 🤗 Whisper Fine-Tuning Event. Thanks, guys, I learned a ton!

Still, there is something misterious to me, so I would love if someone explained it to me. Streaming datasets are not automatically shuffled after each epoch, therefore we define a Callback to do so. However, if we set the number of epochs in the Training Arguments (which we will see shortly), the Trainer complains that the datset has no length, and it asks us to define the maximum number of training steps. So, will this Callback ever be used? Or the Trainer will not be aware of having completed an epoch? Thanks in advance to whoever will clarify this to me! 
"""

# Trainer callback to reinitialise and reshuffle the streamable datasets at the beginning of each epoch
# Only required for streaming: Trainer automatically shuffles non-streaming datasets
class ShuffleCallback(TrainerCallback):
    def on_epoch_begin(self, args, state, control, train_dataloader, **kwargs):
        if isinstance(train_dataloader.dataset, IterableDatasetShard):
            pass  # set_epoch() is handled by the Trainer
        elif isinstance(train_dataloader.dataset, IterableDataset):
            train_dataloader.dataset.set_epoch(train_dataloader.dataset._epoch + 1)

"""We are finally done preparing our data! But do you remember that Whisper is a multi-task Speech Recognition model? And that the task is simply induced using special prefix tokens? Good, now it is time to instruct the model. To do so, we can set those special tokens using the Tokenizer embedded in the Processor.

In our specific case, we could skip this step since English transcription is the default behaviour. Still, this is how you would do if you were in a multilingual setting.
"""

# processor.tokenizer.set_prefix_tokens(language="en", task="transcribe")

## If you wanted to transcribe in Swedish
## (Of course, you'd need a Swedish dataset)
processor.tokenizer.set_prefix_tokens(language="sv", task="transcribe")

## If you wanted to get an English transcription from Swedish audio
# processor.tokenizer.set_prefix_tokens(language="sv", task="translate")

"""(Here you can see what happens if we define only the number of epochs. Scroll down a bit to see explanation and working implementation of Training Arguments and Trainer)."""

# output_dir = "./model"
# os.makedirs(output_dir, exist_ok=True)
# training_args = Seq2SeqTrainingArguments(
#     output_dir=output_dir,
#     num_train_epochs=2,
#     do_train=True,
#     do_eval=True,
#     evaluation_strategy="steps",
#     eval_steps=1,
#     logging_strategy="steps",
#     logging_steps=1,
#     per_device_train_batch_size=4,
#     per_device_eval_batch_size=2
# )

# Initialize Trainer
# trainer = Seq2SeqTrainer(
#     model=model,
#     args=training_args,
#     train_dataset=preprocessed_dataset["train"],
#     eval_dataset=preprocessed_dataset["validation"],
#     tokenizer=processor.feature_extractor,
#     data_collator=data_collator,
#     compute_metrics=compute_metrics,
#     callbacks=[ShuffleCallback()]
# )

"""Cool, we are almost ready for training! Let's define (and create, if missing) the output directory and define some Training Arguments. You can read about all the parameterse on the [🤗 docs](https://huggingface.co/docs/transformers/main_classes/trainer#transformers.Seq2SeqTrainingArguments).

Here, we will instruct the trainer to both train and evaluate the model, define how often metrics should be logged, evaluation should be performed on the evaluation set, model saved, and what batch size to use. The model - in this configuration - **will not be** pushed to the 🤗 hub since it is quite slow. Make sure to authenticate, create a repo and push your model if you train a large model, or use a large dataset!

We will also use mixed precision (16-bit floating point, or fp16) if we are running our training on a GPU.

We will also instruct the model to use the `generate` method for evaluation. That method is used for inference, and it applies a decoding technique to the predicted logits. In this case, it will use greedy search, since we set the number of beams to 1. I briefly introduced decoding algorithgms in the [Decoder paragraph](https://marinone94.github.io/Whisper-paper/#decoder) of my first article, but for now you can simply think of it as selecting the next token as the highest probability, after applying a softmax to the logits. I am considering writing a post about the impact of decoding algorithms on Whisper performance, so let me know you are interested!

Last, we can track our training using several experiment tracking tools. I use Weights and Biases - great tool, you should definitely have a look - but 🤗 supports also "azure_ml", "comet_ml", "mlflow", "neptune" and "tensorboard". You can use "all" (default) to report to all integrations installed, "none" for no integrations. Since WandB is installed in this environment, you should explicitely set it to "none" if you don't have an account.
"""

## If you don't want to track your experiment with WandB, run this!
# os.environ["WANDB_DISABLED"] = "true"

# If you have a wandb account, login!
# Otherwise, edit this cell to loging with your favourite experiment tracker(s)
wandb.login()
wandb.init(project="whisper-training-post")

# Check if we have a GPU.
# In case, we will use mixed precision
# to reduce memory footprint with
# with minimal to no harm to performance
device = "cuda" if torch.cuda.is_available() else "cpu"
use_fp16 = (device == "cuda")

# Let's first define the batch sizes
# Adapt it to your hardware
train_bs = 4 if test_script is True else 64
eval_bs = 2 if test_script is True else 32

# Then we infer the number of steps
# TODO: how did I find it?
num_training_samples = 2385
num_epochs = 3
max_steps_full_training = ceil(num_training_samples * num_epochs / train_bs)
max_steps = 2 if test_script is True else max_steps_full_training

# We don't want to evaluate too often since it slows down training a lot
# but neither too little, since we want to see how the model is training
eval_steps = 1 if test_script is True else int(max_steps / 10)
logging_steps = 1 if test_script is True else int(max_steps / 100)

training_args = Seq2SeqTrainingArguments(
    output_dir=".",
    do_train=True,
    do_eval=True,
    max_steps=max_steps,  
    evaluation_strategy="steps",
    eval_steps=eval_steps,
    logging_strategy="steps",
    logging_steps=logging_steps,
    save_strategy="steps",
    save_steps=eval_steps,
    save_total_limit=2,
    learning_rate=1e-5,
	warmup_ratio=0.5 if test_script is True else 0.2,
    per_device_train_batch_size=train_bs,
    per_device_eval_batch_size=eval_bs,
    # important
    fp16=use_fp16,
    predict_with_generate=True,
    generation_num_beams=1,
    # track experiment
    report_to="wandb",  # edit this line to track with your favourite experiment tracker(s)
    # report_to="none",  # uncomment this line to AVOID tracking the experiment with WandB
)

"""Now we can provide the trainer with the model, tokenizer (important: use the one you set language and task to! In this example, it is `processor.tokenizer`), training arguments, datasets, data collator, callback, and the method to compute metrics during evaluation.

Note that we don't need to place the model to the accelerator device, nor we had to do it in the data collator with the dataset! The trainer will take care of it, if a GPU is available.
"""

# Initialize Trainer
trainer = Seq2SeqTrainer(
    model=model,
    args=training_args,
    train_dataset=preprocessed_dataset["train"],
    eval_dataset=preprocessed_dataset["validation"],
    tokenizer=processor.feature_extractor,
    data_collator=data_collator,
    compute_metrics=compute_metrics,
    callbacks=[ShuffleCallback()]
)

"""Let's

I hope you haven't left yet. If you have, bad for you, as we are ready for training our model! 🍾
As Whisper is a pretrained model ready to be used off-the-shelf, it is advisable to evaluate it before training on both the validation and test sets. Let's make sure we make no harm to it.
"""

eval_metrics = trainer.evaluate(
    eval_dataset=preprocessed_dataset["validation"],
    metric_key_prefix="eval",
    max_length=448,
    num_beams=1,
    # gen_kwargs={"key": value}  to provide additional generation specific arguments by keyword
)

trainer.log_metrics("eval", eval_metrics)
trainer.save_metrics("eval", eval_metrics)
print(eval_metrics)

test_metrics = trainer.evaluate(
    eval_dataset=preprocessed_dataset["test"],
    metric_key_prefix="test",
    max_length=448,
    num_beams=1,
    # gen_kwargs={"key": value}  to provide additional generation specific arguments by keyword
)

trainer.log_metrics("test", test_metrics)
trainer.save_metrics("test", test_metrics)
print(test_metrics)

train_result = trainer.train()
trainer.save_model()

metrics = train_result.metrics
trainer.log_metrics("train", metrics)
trainer.save_metrics("train", metrics)
trainer.save_state()
print(metrics)

"""ADD SOMETHING ABOUT THE TRAINING.

Now let's evaluate the 
"""

final_metrics = trainer.evaluate(
    eval_dataset=preprocessed_dataset["test"],
    metric_key_prefix="test",
    max_length=448,
    num_beams=1,
    # gen_kwargs={"key": value}  to provide additional generation specific arguments by keyword
)

trainer.log_metrics("test", final_metrics)
trainer.save_metrics("test", final_metrics)
print(final_metrics)

# Pushing to hub during training slows down training
# so we push it only in the end.
trainer.push_to_hub()