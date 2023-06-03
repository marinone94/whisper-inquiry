# Training Whisper with Huggingface 🤗

In my previous article, we learned about the new end-to-end speech recognition model developed by OpenAI: Whisper.

Today, we will go through the steps required to fine-tune a Whisper model using several Huggingface libraries. Furthermore, we will explore how the Huggingface libraries take care of all the different steps under the hood, and how the model learns from the training examples.

But first, let me clarify an important point: Whisper models are already trained on downstream tasks, which means that they can be used out-of-the-box to perform several tasks like language-to-language transcription, language-to-english translation, and language identification. However, you will achieve better performance on specific distributions (language, domain, a particular background noise, ...) if the model is fine-tuned on a a specific dataset.

If you want to learn more about approach used, training data, model architecture, and the extensive evaluation performed by the OpenAI team, [this](https://marinone94.github.io/Whisper-paper/) is the place to start! You will benefit much more from this post afterwards.

Since the scope of the article is to learn about the training process, we will fine-tune the smallest model available - [Whisper Tiny](https://huggingface.co/openai/whisper-tiny) - on the Swedish subset of the [Fleurs](https://huggingface.co/datasets/google/fleurs) dataset. Whisper models have been trained largely on English data, so they have more margin for improvement on lower-resource languages.

## How to read this article

This article follows a quite standard flow and starts by introducing the datastet - loaded in streaming mode - and the evaluation metric. Then, it will dive deep into the training chapter, which is made of two sections. The first one is a step-by-step guide on how to fine-tune the model using the 🤗 Trainer. The second one looks inside the engine, and explains what happens in a single training step: loading a data batch, forward pass, loss computation, backpropagation, and optimization. It does not use the 🤗 Trainer, instead it is implemented in PyTorch. Of course, you should read both! But if you are just interested into getting a fine-tuning job to work, then skip the second section. If instead you are just curious to know what happens inside the engine but you don't plan to fine-tune the model using the 🤗 Trainer any time soon, then get back to the first section when the time comes!

I strongly advise you to play with the [notebook](link) to truly understand all the bits and pieces. To smoothly run the code, you can follow the instructions in the [Environment setup](##environment-setup) chapter.

The following chapters are somewhat self-contained, meaning that they import all the required packages and define (and re-define) methods and classes so that they can run independently from each other.

## Environment setup

**⚠ Skip this chapter if you will just read the article without running any code.**

To reproduce the following examples, I recommend you to setup a virtual environment if you are running this notebook on your local machine. The code has been tested with the packages listed in the `requirements.txt` file on `Colab` and `Python 3.9.16`, so I cannot guarantee that it will run smoothly with other packages or Python versions (although it will mostly do with minimal adjustments).


```python
## Clean directories
## RUN WITH CARE!
# !rm -rf assets/ model/ wandb/
```


```python
# !python3.8 -m venv venv
# !source venv/bin/activate
```


```python
!pip install --upgrade pip
```


```python
!pip install huggingface-hub==0.15.1
```

Last, since the purpose of training a model is to use it afterwards, we will leverage the 🤗 Hub to store it, so we can load it anywhere. Model and repo will be used in future articles, so don't skip these steps! First of all, you need to create an account on https://huggingface.co (even though - if you are reading this - chance is high you already have one).

Then, you can authenticate and create a repository. It is a standard git repo, and it leverages git-lfs to manage large files like the model weights. If git-lfs is not installed, let's install it now so we can push our model later. Since on Colab it comes pre-installed, I commented out the installation cell.


```python
!git-lfs -v
```


```python
# !curl -s https://packagecloud.io/install/repositories/github/git-lfs/script.deb.sh | sudo bash
# !sudo apt-get install git-lfs
```

Then we can login to 🤗, create, and clone the repo. The login token can be set on the [🤗 site](https://huggingface.co/settings/tokens). Make sure to give it `write` permissions, and a meaningful name. It is recommended to use differnt tokens for different scopes.


```python
!git config --global credential.helper store
!huggingface-cli login
```


```python
# feel free to edit the name
repo_name = "whisper-training-blog"
```


```python
!huggingface-cli repo create $repo_name -y
```


```python
# this returns a IPython.utils.text.SList
hf_user = !huggingface-cli whoami | head -1
# so we take the first (and only) element
hf_user = hf_user[0]
repo_url = f"https://huggingface.co/{hf_user}/{repo_name}"
```


```python
!echo {repo_url}
```

    https://huggingface.co/marinone94/whisper-training-blog



```python
!git clone $repo_url
```

This will allow us to push the model during training, or after training is completed!


```python
!pip install --upgrade -r whisper-training-blog/colab_requirements.txt
```

Let's check that the packages have been installed correctly, and see if a GPU is available.


```python
!pip3 freeze
```


```python
!nvidia-smi
```

Now we are ready. Let's move on to explore the dataset!

## Training dataset

Fleurs is a speech dataset open-sourced by Google which contains approximately 2000 examples for each language. Each training set has around 10 hours of supervision, and speakers of the training sets are different from the speakers of the dev and test sets.

This dataset has also been used to evaluate the translation capabilities of Whisper models, since most sentences are translated in all languages and can be matched using their ids.

Before proceeding with the training, let's take a look at the data.


```python
from datasets import load_dataset

dataset = load_dataset("google/fleurs", "sv_se", streaming=True)
```


```python
dataset
```




    {'train': <datasets.iterable_dataset.IterableDataset at 0x7fbee4575d80>,
     'validation': <datasets.iterable_dataset.IterableDataset at 0x7fbee4576320>,
     'test': <datasets.iterable_dataset.IterableDataset at 0x7fbe3e3ab7c0>}



As you can see, the dataset contains three splits. Each split is an IterableDataset, since we have loaded it in streaming mode. This means that the dataset is not downloaded, but it is loaded on the fly when needed. This is useful when the dataset occupies too much space on the disk, or if you want to avoid waiting for the whole dataset to be downloaded. Huggingface [docs](https://huggingface.co/docs/datasets/stream) are excellent to learn more about the datasets library and the streaming mode.

We can check the dataset features without downloading the data. So let's have a look.


```python
from pprint import pprint

features = dataset['train'].features
pprint(features)
```

    {'audio': Audio(sampling_rate=16000, mono=True, decode=True, id=None),
     'gender': ClassLabel(names=['male', 'female', 'other'], id=None),
     'id': Value(dtype='int32', id=None),
     'lang_group_id': ClassLabel(names=['western_european_we',
                                        'eastern_european_ee',
                                        'central_asia_middle_north_african_cmn',
                                        'sub_saharan_african_ssa',
                                        'south_asian_sa',
                                        'south_east_asian_sea',
                                        'chinese_japanase_korean_cjk'],
                                 id=None),
     'lang_id': ClassLabel(names=['af_za',
                                  'am_et',
                                  'ar_eg',
                                  'as_in',
                                  'ast_es',
                                  'az_az',
                                  'be_by',
                                  'bg_bg',
                                  'bn_in',
                                  'bs_ba',
                                  'ca_es',
                                  'ceb_ph',
                                  'ckb_iq',
                                  'cmn_hans_cn',
                                  'cs_cz',
                                  'cy_gb',
                                  'da_dk',
                                  'de_de',
                                  'el_gr',
                                  'en_us',
                                  'es_419',
                                  'et_ee',
                                  'fa_ir',
                                  'ff_sn',
                                  'fi_fi',
                                  'fil_ph',
                                  'fr_fr',
                                  'ga_ie',
                                  'gl_es',
                                  'gu_in',
                                  'ha_ng',
                                  'he_il',
                                  'hi_in',
                                  'hr_hr',
                                  'hu_hu',
                                  'hy_am',
                                  'id_id',
                                  'ig_ng',
                                  'is_is',
                                  'it_it',
                                  'ja_jp',
                                  'jv_id',
                                  'ka_ge',
                                  'kam_ke',
                                  'kea_cv',
                                  'kk_kz',
                                  'km_kh',
                                  'kn_in',
                                  'ko_kr',
                                  'ky_kg',
                                  'lb_lu',
                                  'lg_ug',
                                  'ln_cd',
                                  'lo_la',
                                  'lt_lt',
                                  'luo_ke',
                                  'lv_lv',
                                  'mi_nz',
                                  'mk_mk',
                                  'ml_in',
                                  'mn_mn',
                                  'mr_in',
                                  'ms_my',
                                  'mt_mt',
                                  'my_mm',
                                  'nb_no',
                                  'ne_np',
                                  'nl_nl',
                                  'nso_za',
                                  'ny_mw',
                                  'oc_fr',
                                  'om_et',
                                  'or_in',
                                  'pa_in',
                                  'pl_pl',
                                  'ps_af',
                                  'pt_br',
                                  'ro_ro',
                                  'ru_ru',
                                  'sd_in',
                                  'sk_sk',
                                  'sl_si',
                                  'sn_zw',
                                  'so_so',
                                  'sr_rs',
                                  'sv_se',
                                  'sw_ke',
                                  'ta_in',
                                  'te_in',
                                  'tg_tj',
                                  'th_th',
                                  'tr_tr',
                                  'uk_ua',
                                  'umb_ao',
                                  'ur_pk',
                                  'uz_uz',
                                  'vi_vn',
                                  'wo_sn',
                                  'xh_za',
                                  'yo_ng',
                                  'yue_hant_hk',
                                  'zu_za',
                                  'all'],
                           id=None),
     'language': Value(dtype='string', id=None),
     'num_samples': Value(dtype='int32', id=None),
     'path': Value(dtype='string', id=None),
     'raw_transcription': Value(dtype='string', id=None),
     'transcription': Value(dtype='string', id=None)}


Alright, so we can see that the dataset contains a bunch of features we could use for different purposes. To train our speech recognition model, we will use only the `audio` and `raw_transcription` features. But let's look at a sample to see what values the other fields can take. Remember that we loaded the dataset in streaming mode, so we can't access it through indexes since we don't have it in memory yet! By casting it to a list though, the data points will be loaded.


```python
# Get a random sample
dataset_head = dataset['test'].shuffle(1994).take(1)
# Actually download the first item and pprint it
sample = list(dataset_head)[0]
pprint(sample)
```

    {'audio': {'array': array([ 0.00000000e+00, -5.96046448e-08,  5.96046448e-08, ...,
            3.28928232e-03,  2.80916691e-03,  2.16770172e-03]),
               'path': 'test/8553188658088956431.wav',
               'sampling_rate': 16000},
     'gender': 0,
     'id': 1851,
     'lang_group_id': 0,
     'lang_id': 85,
     'language': 'Swedish',
     'num_samples': 113280,
     'path': None,
     'raw_transcription': 'Ripströmmar är det återvändande flödet från vågor som '
                          'bryter av vid stranden, ofta vid ett rev eller '
                          'liknande.',
     'transcription': 'ripströmmar är det återvändande flödet från vågor som '
                      'bryter av vid stranden ofta vid ett rev eller liknande'}


So, the `audio` feature is a dictionary containing the audio waveform stored as numpy array, the file path, and the sampling rate. The `raw_transcription` feature is a string containing the transcription of the audio file. Let's listen to it.


```python
# This download function is used to embed the files in the blog post.
import os

from scipy.io.wavfile import write

def save_wav(data, filename, sample_rate, overwrite=False):
    # Create dir if missing
    if not os.path.exists(filename) or overwrite is True:
        os.makedirs(os.path.dirname(filename), exist_ok=True)
        write(filename, sample_rate, data)
        print(f"Saved {filename} with sampling rate {sample_rate}")
    else:
        print(f"File {filename} already exists. Skipping.")

```


```python
# Notebook visualization
from IPython.display import Audio

print(sample["raw_transcription"])
Audio(sample["audio"]["array"], rate=sample["audio"]["sampling_rate"])
```

    Ripströmmar är det återvändande flödet från vågor som bryter av vid stranden, ofta vid ett rev eller liknande.






<audio  controls="controls" >
    <source src="data:audio/wav;base64,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AQASADUAHwC8/7D/5f/N/3H/if+G/47/yf/u/6P/Yf+///n/0/85AOUAcAAyAN//v/+2/6//7/8MAOX/FAAZAOv/3/+i/+T/5f/4/83/9v/l/xgANACo/1b/Uv+8/8L/yP/B/+v/HQAIAOD/3P8sAKEAygCbANwAGAHLAIEAhgCbAG0AnAA9AP//KwAGACgAdQCkAIQAfQCIAHsANABeAAEAz/8DAMT/BAAeAPP/8v8kALUAxAB+AJUAkgClAEcA7/8+ANYAAgEYAWQAIQDWAOAAZgDf/zwAmgA9AMf/3v9QAK0ARwAHABAAmQDgAJYAOQAHAJIA2QDQAF8AqwDfABABJQFqACgAPACfAKEAOwC9AAQBCgGvAEYAnAC3AAEBggBaALoAwADrAJUAlgAdAXwBeQEYARQBYwF6AVwBKAEbAWwBqQG1AXABLAEFAf0ACgGdAFsAYQB+AKgAggBiANAAIQFwAR8BwwAMAV8BGgHcABkBPQFoAfQA5QDVAOAAGgHNAJUAxQAXAZsBXwHvADQBaAE3AbQAyADWAAIB7ADnALgAxwAWAesAsgCrANsALQFTAc4AwADXAPEA7QDRAMYADgFAAUUBQgExATUBCQHJAMEAGwFQAXkB5QCnAMoA8QDpAGQAKgD1/0IAZgBPAG8AYAB3AM0A7QD4AM8A+QAmASkBCAGZAL0AuQBrAEIAkQC3AFgATQA6AG4AwwDBAH0AJgB0ALAAZAC+/+T/MQAbADQAUAB9AJIAnABfADQAGQAxACEAnf/M/+z/r//X/77/vv/O/7T/mf+q/9b/BQDT/4v/nf/q/w8A9P/P/5//5P/l/9j/AQD0/6T/K/9r/47/sP+8/1r/gP+a/8j/nv9g/4D/0P+//4f/w//H/2L/U/9V/xv/Kv+E/5X/jv9c/0j/jf9y/4T/ff+F/1r/CgATALL/xv+7//H/0P/5/7L/uf/n/+r/+P/1/woANwAHAN//vv+z/5b/Pv8l/0//Vv9B/2X/Y/+r/6f/b/90/8P/xv9n/5X/qv+K/4X/f/+7//D/0f+Y//7/7/8HAEQABgCS/2b/of98/4b/lP92/1v/VP86/0b/JP9h/yX/bP9r/zH/U/90/7b/df+X/23/U/8t/y///f4J/+7+WP+g/0v/a/+e/47/Sf+L/1D/cf+w//z/wf9a/1L/if+R/3b/g/+T/8L/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bwBxAF8AYAB9AKoAhwD0AAMBlAAjAA0AJwDM/3f/Vf+Y/7D/qf/q/0oAaACOALQAvQCoAJAAawAdAPr/xP+w/5r/cf+C/5L/e/+//9v/4/8rAIYAmwCOAKUAdwBvAIYAVQAaAEIAAADA/6T/jP/H/8b/tv/B/8b/5v8JADAAbABfAGYAYACdALQArACqALoA9QDZAOsAJwEuAYEBqAF/AbABjAFPAfYA5wDKAJkAOQAfABUAJgAHAA8AagB0AL0AsQBVAE8AMwCh/6T/Iv8+/xT/RP9I/2P/3P99/+L/yv8AAD8ACAANAOT/m/+L/6T/mv/S/yMA3v////j/5/+5//3/KABAAGUAlwCJABoAFgCw/5//cv+P/23/Wf9n/7L/vf/b/xsA9/8XAEsAkQC6AM8AxQC7AIwAXwASAMv/mv8k/xP/Kf/t/hP/GP81/y7/Z/9T/17/Yv99/3v/Hv8//xH/6v7S/gj/4v4C/xT/L/9c/1n/VP8//z7/Sv/n/pX+3P4j/4j/mv/d/wMAOgCTAKkAoADLAJUARwAoAAQA2/+u/7H/rP/u//H/RgBaAI4AxwDQACABJwFOAfIA0QDvANcAxAC4ALoAjgCAAJQAkACuACAB9gABAUkBLgFRAScB2wCgAHsARAAxANP/vv+s/3D/d/+E////JwBQAIsAqgDlAEQBCAHqAPAAwACeALsAxQDLAJcA4wD8AMwAAgGFANkA9gALAfwAAQEEAQ0BMAFJATkBNQEEAQ0BZgFDAZQBMwFIAWsBZgFhASMB8gAOAbEAdgCgAGgAfgBeABkA9v8rAGoAagCIAKMARABBAEIAAwAiACAAJAAwACIAvADvACMB2wAUATkBHgFXAQ8BTQEEATMBxgCaAI4AYQAhABoAKgAiACIACQApAAwAmABUALEA+gD7ABIBcwGWAWEBdQEaASgB2ADIAAMBHAEnAeYAhABxAG8AYgA6ADMArf+P/3P/Nf9X/zf/mf+E/6H/f//I//n/CgBlABcAcwCyAM0AzQCYAMMA3gADAfcA1ACxAK8AsQA/AD8ASQBNAPv/7/8iACYARQBuAIwAnQDZALcA1gAJAd0AtADeAI0AZQBeAFsAjgBIACQA7/+4/5//cf8v/xL/Kv8V/yj/Rf8h/0P/Mv8H/0X/Uf9m/3X/mf+F/8v/TAArAKUApACoAJ0AlgCgADcA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dP9a/2f/fv+D/3//lf9y/4v/4/+c/5H/rv+F/9L/rv+U/5X/jf+//7n/nf+R/67/hv+p/6r/pP+g/6T/6v+8/5f/k/+c/4H/aP9U/2L/lP90/5j/t//A/7X/q//j/7v/rP/B/+v/tv+X/5f/of/A/4D/pf/I/67/uv+l/1b/O/9T///+Gf9X/yr/K/8j/y//WP92/1D/oP+a/6H/kf+3/9//Zv90/3P/hP+w/8r/u//y/73/0f/F/7n/vv9s/6D/f/9G/43/lf9P/2X/P/9C/0//eP9a/yz/Sf87/w//Qv9r/3P/jf+K/5j/af9t/4H/av8r/27/W/9q/63/if/C/8P/pv+h/4H/V/9T/1r/YP99/4n/ZP+S/5r/iP+a/2H/hf+i/4v/lP+o/5b/U/+R/5f/hf+0/8P/3v8KAPL/ov/M/8D/8//u//D/8P/K//j/1//6/8f/vP/N/8P/5//z/9H/gv+g/8v/qv+J/43/pv+r/5P/mP93/3H/bP9Y/2z/cf+p/+3/of+i/5v/W/+v/5H/oP+D/4b/s/+P/6z/2v/Y/8H/of+N/5b/uv/y/7j/5P8LALn/qv+2/57/g/+H/5L/fv9w/zv/af+J/53/0f+y/83/3f/S/8v/qf/k/xkALABUAAcAAwDI/6j/wP+j/3b/c/93/zz/R/9N/zX/Z/+I/1T/ev9p/yL/HP9Z/1n/Sf9K/2r/gP+J/1T/Jv8n/z7/Wf9O/5n/g/+U/6v/wv+s/9T/4v/T/+7/BQDl/83/BADj/xwAGQAFABsAHgASAAEACQA4ADAAUQA+AEMASgBhAE4AHgBnAFcAbAA/ACgAUgBXAHcAkwBhAGUAVgCDAGMANgCxALAAsACMALwAkACVAL8AkACjAKAAbAAXAIoAkQCiAJYAagCmAKwAnACDAMIAngDJAOQAswD/ANoAqAA=" type="audio/wav" />
    Your browser does not support the audio element.
</audio>





```python
save_wav(
    data=sample["audio"]["array"],
    filename="src/whisper_training/assets/audio/fleurs_sv/random_sample.wav",
    sample_rate=sample["audio"]["sampling_rate"]
)
```

As you can hear, the audio is clear but the reader speaks quite fast. The model should perform well, but most likely it won't be perfect. Let's have a look at the transcription. I will not explain each step here, but you can read all the details in my previous [article](https://marinone94.github.io/Whisper-paper/).


```python
import time

import torch
from transformers import (
    WhisperProcessor,
    WhisperForConditionalGeneration
)

# If you have a GPU, it will be faster...
device = "cuda" if torch.cuda.is_available() else "cpu"

# Load processor for audio and text
# feature processor (audio) + tokenizer (text)
processor = WhisperProcessor.from_pretrained(
    "openai/whisper-tiny")

# Load model architecture and weights
model = WhisperForConditionalGeneration.from_pretrained(
    "openai/whisper-tiny").to(device)

```


```python
model_def_max_length = model.config.max_length

def transcribe(sample, return_pred_and_ref=False, print_results=False):
    if print_results is False and return_pred_and_ref is False:
        raise ValueError("Results are not printed nor returned.\n" +
                         "Set return_pred_and_ref to return results.\n" +
                         "Set print_results to print results to console.")

    # Load audio file (see previous cells)
    data = sample["audio"]["array"]
    sampling_rate = sample["audio"]["sampling_rate"]
    print("Audio loaded. Sample rate: ", sampling_rate)

    # Convert input audio to log-mel spectrogram
    input_features = processor(
        data, sampling_rate=sampling_rate, return_tensors="pt").input_features.to(device)

    # Get tokens to initiate transcription
    # and store them in the model config.
    # Under the hood, this will force the model to predict
    # these tokens in the beginning of the transcription.
    init_tokens = processor.get_decoder_prompt_ids(
        language="sv", task="transcribe", no_timestamps=True)
    model.config.forced_decoder_ids = init_tokens

    # Generate transcription tokens
    print("Generating transcription...")
    st = time.perf_counter_ns()
    transcription_token_ids = model.generate(
        input_features, max_new_tokens=model_def_max_length)
    et = time.perf_counter_ns()
    dt = (et - st) / 1e9
    print(f"Transcription generated in {dt} [s]")

    # Decode transcription
    # NOTE: input_features is a batch of one element,
    # and the returned token ids are batched analogously
    transcription_token_ids = transcription_token_ids[0]
    whisper_transcript = processor.decode(
        transcription_token_ids, skip_special_tokens=True).strip()

    # Print results and return if required
    reference_transcript = sample["raw_transcription"]
    if print_results is True:
        print("Whisper transcript:\n", whisper_transcript)
        print("Reference transcript:\n", reference_transcript)
    if return_pred_and_ref is True:
        return {
            "prediction": whisper_transcript,
            "label": reference_transcript
        }

```


```python
transcribe(sample, print_results=True)
```

    Audio loaded. Sample rate:  16000
    Generating transcription...
    Transcription generated in 7.358902868 [s]
    Whisper transcript:
     Ljipströmmer är åtvänderna för att få gågår från butra av i ständen. Ofta vi treväll liknande.
    Reference transcript:
     Ripströmmar är det återvändande flödet från vågor som bryter av vid stranden, ofta vid ett rev eller liknande.


We can see that the generated and the reference transcripts are far from identical. As said, we are using the smallest model available, and I can guarantee you the large and medium models are mindblowing! Still, one might want to use the tiny model for different reasons (latency, memory footprint, learning, ...): then fine-tuning is definitely required, at least in Swedish! I will never get tired to repeat that also the largest models benefit from fine-tuning, especially on very specific distributions and for low-resource languages. Since there are multiple reasons to fine-tune pretrained models, let's learn everything about it.

The first pillar of improving anything is measuring! If you don't measure something, you won't know if the direction you have taken is the right one. The question is then straightforward: how do we measure the performance of the model? It is almost time to introduce the standard evaluation metric for speech recognition: the [Word Error Rate](#word-error-rate).

But before that, we should inspect the dataset a bit more. Let's load 100 random samples and look at the distribution of audio duration and transcription length (measured in number of tokens).

The audio duration is not directly provided, but we can calculate it as the ratio between the number of samples and the sampling rate. Let's keep only the features used for the analysis.


```python
# Remove all unnecessary columns
cols_to_keep = ["audio", "raw_transcription", "num_samples"]
cols_to_remove = [col for col in sample.keys() if col not in cols_to_keep]
dataset = dataset.remove_columns(cols_to_remove)
# First shuffle, then take 100 samples
samples = dataset['train'].shuffle(94).take(100)
sampling_rate = sample["audio"]["sampling_rate"]
samples.features
```




    {'num_samples': Value(dtype='int32', id=None),
     'audio': Audio(sampling_rate=16000, mono=True, decode=True, id=None),
     'raw_transcription': Value(dtype='string', id=None)}



Since the dataset is in streaming mode, we cannot use the `dataset.to_pandas()` method. Instead, we loop over the dataset and store each item in a list: this forces the data to be downloaded. Only then we can create our dataframe.


```python
import pandas as pd
from tqdm import tqdm


rows = [row for row in tqdm(samples)]
df = pd.DataFrame(rows, columns=list(samples.features.keys()))

def audio_length(ns):
    return round(ns / sampling_rate, 2)

def tokenize(text):
    return processor.tokenizer.tokenize(text)

df["audio_length"] = df["num_samples"].apply(audio_length)
df["tokens"] = df["raw_transcription"].apply(tokenize)
df["num_tokens"] = df["tokens"].apply(len)

df.head()
```

    100it [00:45,  2.21it/s]






  <div id="df-1a914dc2-286c-41b2-ae62-1d5fe0eec367">
    <div class="colab-df-container">
      <div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>num_samples</th>
      <th>audio</th>
      <th>raw_transcription</th>
      <th>audio_length</th>
      <th>tokens</th>
      <th>num_tokens</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>182400</td>
      <td>{'path': None, 'array': [0.0, -3.0517578125e-0...</td>
      <td>Detta är särskilt populärt bland nyutexaminera...</td>
      <td>11.40</td>
      <td>[D, etta, ĠÃ¤r, Ġs, Ã¤r, sk, ilt, Ġpopul, Ã¤rt...</td>
      <td>45</td>
    </tr>
    <tr>
      <th>1</th>
      <td>193920</td>
      <td>{'path': None, 'array': [0.0, 0.0, 0.0, 0.0, 0...</td>
      <td>Samtidigt försökte den tyska flottan, främst m...</td>
      <td>12.12</td>
      <td>[S, amt, id, igt, ĠfÃ¶rsÃ¶, kte, Ġden, Ġ, t, y...</td>
      <td>38</td>
    </tr>
    <tr>
      <th>2</th>
      <td>243840</td>
      <td>{'path': None, 'array': [0.0, 0.0, 0.0, 0.0, 0...</td>
      <td>Att bära åt andra - Lämna aldrig dina väskor u...</td>
      <td>15.24</td>
      <td>[Att, Ġb, Ã¤, ra, ĠÃ¥t, Ġandra, Ġ-, ĠL, Ã¤m, n...</td>
      <td>40</td>
    </tr>
    <tr>
      <th>3</th>
      <td>201600</td>
      <td>{'path': None, 'array': [0.0, 0.0, 0.0, 0.0, 0...</td>
      <td>Sådana framgångssagor minskade förändringsskrä...</td>
      <td>12.60</td>
      <td>[S, Ã¥, d, ana, Ġfram, g, Ã¥ng, ss, ag, or, Ġm...</td>
      <td>43</td>
    </tr>
    <tr>
      <th>4</th>
      <td>139200</td>
      <td>{'path': None, 'array': [0.0, 0.0, -3.05175781...</td>
      <td>En välbärgad resenär skulle kanske överväga en...</td>
      <td>8.70</td>
      <td>[En, ĠvÃ¤l, b, Ã¤r, g, ad, Ġres, en, Ã¤r, Ġsku...</td>
      <td>31</td>
    </tr>
  </tbody>
</table>
</div>
      <button class="colab-df-convert" onclick="convertToInteractive('df-1a914dc2-286c-41b2-ae62-1d5fe0eec367')"
              title="Convert this dataframe to an interactive table."
              style="display:none;">

  <svg xmlns="http://www.w3.org/2000/svg" height="24px"viewBox="0 0 24 24"
       width="24px">
    <path d="M0 0h24v24H0V0z" fill="none"/>
    <path d="M18.56 5.44l.94 2.06.94-2.06 2.06-.94-2.06-.94-.94-2.06-.94 2.06-2.06.94zm-11 1L8.5 8.5l.94-2.06 2.06-.94-2.06-.94L8.5 2.5l-.94 2.06-2.06.94zm10 10l.94 2.06.94-2.06 2.06-.94-2.06-.94-.94-2.06-.94 2.06-2.06.94z"/><path d="M17.41 7.96l-1.37-1.37c-.4-.4-.92-.59-1.43-.59-.52 0-1.04.2-1.43.59L10.3 9.45l-7.72 7.72c-.78.78-.78 2.05 0 2.83L4 21.41c.39.39.9.59 1.41.59.51 0 1.02-.2 1.41-.59l7.78-7.78 2.81-2.81c.8-.78.8-2.07 0-2.86zM5.41 20L4 18.59l7.72-7.72 1.47 1.35L5.41 20z"/>
  </svg>
      </button>

  <style>
    .colab-df-container {
      display:flex;
      flex-wrap:wrap;
      gap: 12px;
    }

    .colab-df-convert {
      background-color: #E8F0FE;
      border: none;
      border-radius: 50%;
      cursor: pointer;
      display: none;
      fill: #1967D2;
      height: 32px;
      padding: 0 0 0 0;
      width: 32px;
    }

    .colab-df-convert:hover {
      background-color: #E2EBFA;
      box-shadow: 0px 1px 2px rgba(60, 64, 67, 0.3), 0px 1px 3px 1px rgba(60, 64, 67, 0.15);
      fill: #174EA6;
    }

    [theme=dark] .colab-df-convert {
      background-color: #3B4455;
      fill: #D2E3FC;
    }

    [theme=dark] .colab-df-convert:hover {
      background-color: #434B5C;
      box-shadow: 0px 1px 3px 1px rgba(0, 0, 0, 0.15);
      filter: drop-shadow(0px 1px 2px rgba(0, 0, 0, 0.3));
      fill: #FFFFFF;
    }
  </style>

      <script>
        const buttonEl =
          document.querySelector('#df-1a914dc2-286c-41b2-ae62-1d5fe0eec367 button.colab-df-convert');
        buttonEl.style.display =
          google.colab.kernel.accessAllowed ? 'block' : 'none';

        async function convertToInteractive(key) {
          const element = document.querySelector('#df-1a914dc2-286c-41b2-ae62-1d5fe0eec367');
          const dataTable =
            await google.colab.kernel.invokeFunction('convertToInteractive',
                                                     [key], {});
          if (!dataTable) return;

          const docLinkHtml = 'Like what you see? Visit the ' +
            '<a target="_blank" href=https://colab.research.google.com/notebooks/data_table.ipynb>data table notebook</a>'
            + ' to learn more about interactive tables.';
          element.innerHTML = '';
          dataTable['output_type'] = 'display_data';
          await google.colab.output.renderOutput(dataTable, element);
          const docLink = document.createElement('div');
          docLink.innerHTML = docLinkHtml;
          element.appendChild(docLink);
        }
      </script>
    </div>
  </div>





```python
import matplotlib.pyplot as plt

# Plot the histogram
df.audio_length.plot(kind='hist', bins=10)

# Show the plot
plt.title(f"Audio length [s] distribution - {len(df)} samples")
plt.show()
```


    
![png](./src/whisper_training/assets/images/audio_length_histogram.png)
    



```python
# Plot the histogram
df.num_tokens.plot(kind='hist', bins=10)

# Show the plot
plt.title(f"Number of tokens distribution - {len(df)} samples")
plt.show()
```


    
![png](./src/whisper_training/assets/images/num_tokens_histogram.png)
    



```python
stats_df = pd.DataFrame({
    "mean": [df.audio_length.mean(), df.num_tokens.mean()],
    "std": [df.audio_length.std(), df.num_tokens.std()],
    "min": [df.audio_length.min(), df.num_tokens.min()],
    "max": [df.audio_length.max(), df.num_tokens.max()],
}, index=["audio_length", "num_tokens"]).T
stats_df
stats_df.boxplot()
plt.show()
```


    
![png](./src/whisper_training/assets/images/audio_length_num_tokens_distributions.png)
    


Most of the randomly selected examples are shorter than 30 seconds. It is relatively safe to assume then that only a few examples will be truncated during training. Most of the samples have between 20 and 50 tokens, corresponding to audios between 5 and 20 seconds. Therefore, this dataset might not be the best to fine-tune a model which will then be used to transcribe single words. This is just to remind us that we rarely have one dataset which is perfect for all the tasks we want to perform, and it is worth spending some time to understand the dataset before jumping into the training.

## Word Error Rate

Now that we know a bit more about the dataset, we can introduce the [Word Error Rate](https://en.wikipedia.org/wiki/Word_error_rate) (WER) metric. WER measures the amount of errors in the generated transcript - the transcript, from now on - compared with the reference transcript - the reference, from now on - which we assume is the correct one.

Errors can be of three kinds: Insertions (i.e. words that are added in the transcript, but were not present in the reference), Deletions (i.e. words that are present in the reference, but  not in the transcript), and Substitutions (ie. words which are wrongly transcribed, for example `The meat was great!` vs `The meal was great!` has 1 substitution (`meal` -> `meat`).

WER is then calculated as follows:
$$
WER = (S + I + D) / N
$$

where N is the number of words of the reference.

As you can see from the definition, the WER metric measures **only** the amount of words which are wrongly transcribed, with no understanding at all about those differences. Have a look at the following examples, where the first word is the reference, and the second is the transcript:

* `Hi` -> `hi`
* `5` -> `five`
* `cool`-> `cool!`

This list could go longer (currency, units of measure, ...) but you can already see my point. If the text is not normalized before calculating the metric, the results will look much worse than they actually are! So don't forget that important step. Whisper comes with a default normalizer to facilitate experiments comparison, therefore will use that.

Still, if two words are different, one is added, or missed, they will count as 1 Substitution, 1 Insertion, or 1 Deletion respectively, regardlesss of their meanings in the context. Take those two examples:

* `The meal was great` -> `The meal was not great`
* `Ì studied Spanish at school` -> `Ì have studied Spanish at school`

The first Deletion completely swap the sentence meaning, while the second one is basically harmless, but they both impact WER the same way!

Let's now calculate the WER of the long example. We will use the WER implementation provided by 🤗 Evaluate, as we should not reinvent the wheel unless we can make a better one. Note that the BasicTextNormalizer is language agnostic. For accurate evaluation, I advise implementing language specific normalizers.


```python
import evaluate
from transformers.models.whisper.english_normalizer import BasicTextNormalizer

metric = evaluate.load("wer")
normalizer = BasicTextNormalizer()
```


```python
def wer(pred, label):
    # wer <= 1, we make it as %
    # and it wants list of predictions and references, not strings
    norm_pred = normalizer(pred)
    norm_label = normalizer(label)
    wer = 100 * metric.compute(predictions=[pred], references=[label])
    wer = round(wer, 2)

    print("-"*45)
    print("Prediction:", pred)
    print("Reference:", label)
    print("WER", wer)
    print("-"*45)

    norm_wer = 100 * metric.compute(predictions=[norm_pred], references=[norm_label])
    wer = round(wer, 2)

    print("Normalised prediction:", norm_pred)
    print("Normalised reference:", norm_label)
    print("WER", norm_wer)

    return wer, norm_wer
```


```python
response = transcribe(sample, return_pred_and_ref=True)
pred, label = response["prediction"], response["label"]
raw_wer, norm_wer = wer(pred, label)
```

    Audio loaded. Sample rate:  16000
    Generating transcription...
    Transcription generated in 0.341825522 [s]
    ---------------------------------------------
    Prediction: Ljipströmmer är åtvänderna för att få gågår från butra av i ständen. Ofta vi treväll liknande.
    Reference: Ripströmmar är det återvändande flödet från vågor som bryter av vid stranden, ofta vid ett rev eller liknande.
    WER 83.33
    ---------------------------------------------
    Normalised prediction: ljipströmmer är åtvänderna för att få gågår från butra av i ständen ofta vi treväll liknande 
    Normalised reference: ripströmmar är det återvändande flödet från vågor som bryter av vid stranden ofta vid ett rev eller liknande 
    WER 77.77777777777779


We can observe again that the model has a hard time transcribing this sample. But we can also see how normalization can affect the WER. Here a basic normalization reduces the WER of about 9%, since a substitution was "." for "," which is extremely hard for the model to predict and either could be correct. Therefore, these normalization techniques allow for better model's capabilities assessment.

Yeah, WER probably is not the best metric possible, but it does not require human judgment, so it is scalable and somehow consistent. And it is the standard metric used in ASR research, so we will use it. For now.

## Beyond WER

Having seen the limitations of WER, it is worth mentioning that other approaches can be used to either replace or integrate WER. This is a research topic on itself, so I will not write about here, but let me know if you are interested about it and I can make another article about it! Little spolier: I am thinking about using NER to discard Entities like personal or organization names, and using word and sentence embeddings to weight errors. But keep this little secret between the two of us 😉

## Training

In this paragraph, we will finally fine-tune our model on the dataset we just introduced. Then, we will compare the performance of the original and the fine-tuned models, and we hope to see a decline in the WER.

The paragraph is split into two main parts: in the first one, we will go through the steps required to fine-tune the model using multiple 🤗 libraries. In the second part, we will open the engine and figure out what happens under the hood when the model learns from the training samples.

If you are familiar with fine-tuning Speech Sequence to Sequence models with 🤗, or if you are just curious to know what happens during training, you can jump to the second part. If you don't care about what happens, and you are happy enough with being able to fine-tune a model yourself, the first part will be enough. The full training script and requirement file are available on my [GitHub](https://github.com/marinone94/whisper-inquiry/tree/main/src/whisper_training), and it should run out of the box (but go through it, you might need to change a couple of hyperparameters and set the correct flags).

Most of the code comes from the script provided during the 🤗 Whisper Fine-Tuning Event, but it has been slimmed to focus on the essential parts, and it will be expained along the way. In my way.

### Fine-tuning Whisper with 🤗


To get started, let's import the required libraries.


```python
import os  # used to create output directory
import shutil  # used to remove checkpoints before pushing
from dataclasses import dataclass  # used to define data collator
from math import ceil  # used to round up decimals

import evaluate  # used to import and compute evaluation metrics
import torch  # used to know if a GPU with CUDA is available
import wandb  # used for experiment tracking
from datasets import DatasetDict, IterableDatasetDict, load_dataset  # used to load the dataset in streaming mode
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

```

Then, we will load processor, model configuration, architecture and weights, and the dataset (in streaming mode). The Swedish split of Fleurs is not a massive dataset, thus we could easily download it and store it in memory, but it is good to learn how to use the streaming mode if you were to fine-tune your model on larger datasets. 


```python
model_id = "openai/whisper-tiny"
processor = AutoProcessor.from_pretrained(model_id)
config = AutoConfig.from_pretrained(model_id)
model = AutoModelForSpeechSeq2Seq.from_pretrained(model_id)
```


```python
dataset_id = "google/fleurs"
dataset_language_code = "sv_se"
dataset = load_dataset(dataset_id, dataset_language_code, streaming=True)
```

The first time you run this code, make sure everything works fine using a small sample and low number of training steps. Just uncomment the next cell and run it. One note: since the dataset is loaded in streaming mode, the instruction will not be executed immediately. Instead, the dataset will be subsampled only when data will be needed during training.


```python
# test_script = True
test_script = False
```


```python
## Sample dataset for testing
if test_script is True:
    if isinstance(dataset, IterableDatasetDict):
        dataset["train"] = dataset["train"].shuffle(seed=42).take(8)
        dataset["validation"] = dataset["validation"].shuffle(seed=42).take(4)
        dataset["test"] = dataset["test"].shuffle(seed=42).take(4)
    elif isinstance(dataset, DatasetDict):
        dataset["train"] = dataset["train"].shuffle(seed=42).select(range(8))
        dataset["validation"] = dataset["validation"].shuffle(seed=42).select(range(4))
        dataset["test"] = dataset["test"].shuffle(seed=42).select(range(4))
```

The raw dataset is not yet ready for training. As described in my first article about Whisper, the input audio waveform needs to be transformed into a Log-mel Spectrogram. I recommend you to read the [Audio Preprocessing section](https://marinone94.github.io/Whisper-paper/#audio-preprocessing) to understand the process. For the scope of this article, you should just know that the audio is translated from the time domain to its frequency representation using a sliding window, and adjusted to simulate human hearing. The Whisper Feature Extractor included in the Whisper Processor will take care of the rest.

Furthermore, the reference transcripts need to be tokenized since the model outputs one token at the time and they are used to compute the loss during training. Again, the Tokenizer will take care of that, but the task needs to be included in the preprocessing step.

When we introduced the WER metric, we learned about the importance of normalizing the texts. But should we do that also before training? That is up to you, but you should remember that Whisper models have been pretrained to predict Capitalization, digits, and punctuation. So if you normalize the reference teanscripts before fine-tuning, you will teach your model not to predict capital letters, digits, and punctuations. This does not mean that the model will never predict them, since it has been extensively pretrained to do so. To wrap up, your choice should depend on the final application and on the dataset size, but in general I recommend not to normalize the references before training.

Finally, by storing the input features in the default model input name, the trainer will automatically pick the correct ones during training. So, don't hard-code it!


```python
dataset_id = "google/fleurs"
dataset_language_code = "sv_se"
dataset = load_dataset(dataset_id, dataset_language_code, streaming=True)

###

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

###

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

```

We will use the `.map` method to apply our preprocessing function to the whole dataset. At the same time, we will drop all the columns which are not strictly needed during training. Since `input_features`, `ìnput_length` and `labels` are not features of the raw dataset, we can remove all the original ones. Finally, we will convert the dataset features to `torch` type since the dataset has no `__len__` property (again, we are in streaming mode). 


```python
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
```

Since we want to evaluate our model on the validation set during training, we also need to provide a method that computes the metrics given the model predictions. It looks very similar to the function we introduced above, but since it will receive a single prediction object, we need to extract the predicted tokens and the corresponding labels. Furthermore, we replace the label ids equal to -100 with the padding token. A couple of minutes of patience and you will understand why.

When decoding the prediction and the labels, we need to discard the special tokens. Those are used to force the model to perform specific tasks. You can read more [here](https://marinone94.github.io/Whisper-paper/#tasks).


```python
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
```

Alright, we are almost done preparing our dataset. Quite a lot of work, I know, but that is most of the job.

The last step is to define a data collator, which will build data btaches from the datasets during training using the Whisper Processor. It will also pad input features and labels.

Also, in the metrics computation method we replaced the labels with id equal to -100. It was done because the data collator "**must**" set the padding tokens to -100 so that the trainer will automatically ignore them when computing the loss. That was the reverse step.


```python
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

```


```python
data_collator = DataCollatorSpeechSeq2SeqWithPadding(
    processor=processor,
    decoder_start_token_id=model.config.decoder_start_token_id,
)
```

The next step was something I would have definitely missed had I not attended the 🤗 Whisper Fine-Tuning Event. Thanks, guys, I learned a ton!

Still, there is something misterious to me, so I would love if someone explained it to me. Streaming datasets are not automatically shuffled after each epoch, therefore we define a Callback to do so. However, if we set the number of epochs in the Training Arguments (which we will see shortly), the Trainer complains that the datset has no length, and it asks us to define the maximum number of training steps. So, will this Callback ever be used? Or the Trainer will not be aware of having completed an epoch? Thanks in advance to whoever will clarify this to me! 


```python
# Trainer callback to reinitialise and reshuffle the streamable datasets at the beginning of each epoch
# Only required for streaming: Trainer automatically shuffles non-streaming datasets
class ShuffleCallback(TrainerCallback):
    def on_epoch_begin(self, args, state, control, train_dataloader, **kwargs):
        if isinstance(train_dataloader.dataset, IterableDatasetShard):
            pass  # set_epoch() is handled by the Trainer
        elif isinstance(train_dataloader.dataset, IterableDataset):
            train_dataloader.dataset.set_epoch(train_dataloader.dataset._epoch + 1)

```

We are finally done preparing our data! But do you remember that Whisper is a multi-task Speech Recognition model? And that the task is simply induced using special prefix tokens? Good, now it is time to instruct the model. To do so, we can set those special tokens using the Tokenizer embedded in the Processor.



```python
processor.tokenizer.set_prefix_tokens(language="sv", task="transcribe")

## If you wanted to get an English transcription from Swedish audio
# processor.tokenizer.set_prefix_tokens(language="sv", task="translate")
```

(Here you can see what happens if we define only the number of epochs. Scroll down a bit to see explanation and working implementation of Training Arguments and Trainer).


```python
# Set output dir to 🤗 repo
output_dir = f"./{repo_name}"
# os.makedirs(output_dir, exist_ok=True)
```


```python
training_args = Seq2SeqTrainingArguments(
    output_dir=output_dir,
    num_train_epochs=2,
    do_train=True,
    do_eval=True,
    evaluation_strategy="steps",
    eval_steps=1,
    logging_strategy="steps",
    logging_steps=1,
    report_to="none",
    per_device_train_batch_size=4,
    per_device_eval_batch_size=2
)
```


```python
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
```


<pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace"><span style="color: #800000; text-decoration-color: #800000">╭─────────────────────────────── </span><span style="color: #800000; text-decoration-color: #800000; font-weight: bold">Traceback </span><span style="color: #bf7f7f; text-decoration-color: #bf7f7f; font-weight: bold">(most recent call last)</span><span style="color: #800000; text-decoration-color: #800000"> ────────────────────────────────╮</span>
<span style="color: #800000; text-decoration-color: #800000">│</span> in <span style="color: #00ff00; text-decoration-color: #00ff00">&lt;cell line: 1&gt;</span>:<span style="color: #0000ff; text-decoration-color: #0000ff">1</span>                                                                              <span style="color: #800000; text-decoration-color: #800000">│</span>
<span style="color: #800000; text-decoration-color: #800000">│</span>                                                                                                  <span style="color: #800000; text-decoration-color: #800000">│</span>
<span style="color: #800000; text-decoration-color: #800000">│</span> <span style="color: #bfbf7f; text-decoration-color: #bfbf7f">/usr/local/lib/python3.10/dist-packages/transformers/</span><span style="color: #808000; text-decoration-color: #808000; font-weight: bold">trainer_seq2seq.py</span>:<span style="color: #0000ff; text-decoration-color: #0000ff">56</span> in <span style="color: #00ff00; text-decoration-color: #00ff00">__init__</span>           <span style="color: #800000; text-decoration-color: #800000">│</span>
<span style="color: #800000; text-decoration-color: #800000">│</span>                                                                                                  <span style="color: #800000; text-decoration-color: #800000">│</span>
<span style="color: #800000; text-decoration-color: #800000">│</span>   <span style="color: #7f7f7f; text-decoration-color: #7f7f7f"> 53 │   │   </span>optimizers: Tuple[torch.optim.Optimizer, torch.optim.lr_scheduler.LambdaLR] = (<span style="color: #0000ff; text-decoration-color: #0000ff">N</span>   <span style="color: #800000; text-decoration-color: #800000">│</span>
<span style="color: #800000; text-decoration-color: #800000">│</span>   <span style="color: #7f7f7f; text-decoration-color: #7f7f7f"> 54 │   │   </span>preprocess_logits_for_metrics: Optional[Callable[[torch.Tensor, torch.Tensor], t   <span style="color: #800000; text-decoration-color: #800000">│</span>
<span style="color: #800000; text-decoration-color: #800000">│</span>   <span style="color: #7f7f7f; text-decoration-color: #7f7f7f"> 55 │   </span>):                                                                                     <span style="color: #800000; text-decoration-color: #800000">│</span>
<span style="color: #800000; text-decoration-color: #800000">│</span> <span style="color: #800000; text-decoration-color: #800000">❱ </span> 56 <span style="color: #7f7f7f; text-decoration-color: #7f7f7f">│   │   </span><span style="color: #00ffff; text-decoration-color: #00ffff">super</span>().<span style="color: #00ff00; text-decoration-color: #00ff00">__init__</span>(                                                                  <span style="color: #800000; text-decoration-color: #800000">│</span>
<span style="color: #800000; text-decoration-color: #800000">│</span>   <span style="color: #7f7f7f; text-decoration-color: #7f7f7f"> 57 │   │   │   </span>model=model,                                                                   <span style="color: #800000; text-decoration-color: #800000">│</span>
<span style="color: #800000; text-decoration-color: #800000">│</span>   <span style="color: #7f7f7f; text-decoration-color: #7f7f7f"> 58 │   │   │   </span>args=args,                                                                     <span style="color: #800000; text-decoration-color: #800000">│</span>
<span style="color: #800000; text-decoration-color: #800000">│</span>   <span style="color: #7f7f7f; text-decoration-color: #7f7f7f"> 59 │   │   │   </span>data_collator=data_collator,                                                   <span style="color: #800000; text-decoration-color: #800000">│</span>
<span style="color: #800000; text-decoration-color: #800000">│</span>                                                                                                  <span style="color: #800000; text-decoration-color: #800000">│</span>
<span style="color: #800000; text-decoration-color: #800000">│</span> <span style="color: #bfbf7f; text-decoration-color: #bfbf7f">/usr/local/lib/python3.10/dist-packages/transformers/</span><span style="color: #808000; text-decoration-color: #808000; font-weight: bold">trainer.py</span>:<span style="color: #0000ff; text-decoration-color: #0000ff">568</span> in <span style="color: #00ff00; text-decoration-color: #00ff00">__init__</span>                  <span style="color: #800000; text-decoration-color: #800000">│</span>
<span style="color: #800000; text-decoration-color: #800000">│</span>                                                                                                  <span style="color: #800000; text-decoration-color: #800000">│</span>
<span style="color: #800000; text-decoration-color: #800000">│</span>   <span style="color: #7f7f7f; text-decoration-color: #7f7f7f"> 565 │   │   │   </span>logger.info(<span style="color: #808000; text-decoration-color: #808000">"max_steps is given, it will override any value given in num_tra</span>  <span style="color: #800000; text-decoration-color: #800000">│</span>
<span style="color: #800000; text-decoration-color: #800000">│</span>   <span style="color: #7f7f7f; text-decoration-color: #7f7f7f"> 566 │   │   </span>                                                                                  <span style="color: #800000; text-decoration-color: #800000">│</span>
<span style="color: #800000; text-decoration-color: #800000">│</span>   <span style="color: #7f7f7f; text-decoration-color: #7f7f7f"> 567 │   │   </span><span style="color: #0000ff; text-decoration-color: #0000ff">if</span> train_dataset <span style="color: #ff00ff; text-decoration-color: #ff00ff">is</span> <span style="color: #ff00ff; text-decoration-color: #ff00ff">not</span> <span style="color: #0000ff; text-decoration-color: #0000ff">None</span> <span style="color: #ff00ff; text-decoration-color: #ff00ff">and</span> <span style="color: #ff00ff; text-decoration-color: #ff00ff">not</span> has_length(train_dataset) <span style="color: #ff00ff; text-decoration-color: #ff00ff">and</span> args.max_step  <span style="color: #800000; text-decoration-color: #800000">│</span>
<span style="color: #800000; text-decoration-color: #800000">│</span> <span style="color: #800000; text-decoration-color: #800000">❱ </span> 568 <span style="color: #7f7f7f; text-decoration-color: #7f7f7f">│   │   │   </span><span style="color: #0000ff; text-decoration-color: #0000ff">raise</span> <span style="color: #00ffff; text-decoration-color: #00ffff">ValueError</span>(                                                             <span style="color: #800000; text-decoration-color: #800000">│</span>
<span style="color: #800000; text-decoration-color: #800000">│</span>   <span style="color: #7f7f7f; text-decoration-color: #7f7f7f"> 569 │   │   │   │   </span><span style="color: #808000; text-decoration-color: #808000">"The train_dataset does not implement __len__, max_steps has to be speci</span>  <span style="color: #800000; text-decoration-color: #800000">│</span>
<span style="color: #800000; text-decoration-color: #800000">│</span>   <span style="color: #7f7f7f; text-decoration-color: #7f7f7f"> 570 │   │   │   │   </span><span style="color: #808000; text-decoration-color: #808000">"The number of steps needs to be known in advance for the learning rate </span>  <span style="color: #800000; text-decoration-color: #800000">│</span>
<span style="color: #800000; text-decoration-color: #800000">│</span>   <span style="color: #7f7f7f; text-decoration-color: #7f7f7f"> 571 │   │   │   </span>)                                                                             <span style="color: #800000; text-decoration-color: #800000">│</span>
<span style="color: #800000; text-decoration-color: #800000">╰──────────────────────────────────────────────────────────────────────────────────────────────────╯</span>
<span style="color: #ff0000; text-decoration-color: #ff0000; font-weight: bold">ValueError: </span>The train_dataset does not implement __len__, max_steps has to be specified. The number of steps needs 
to be known in advance for the learning rate scheduler.
</pre>



Cool, we are almost ready for training! Let's define (and create, if missing) the output directory and define some Training Arguments. You can read about all the parameterse on the [🤗 docs](https://huggingface.co/docs/transformers/main_classes/trainer#transformers.Seq2SeqTrainingArguments).

Here, we will instruct the trainer to both train and evaluate the model, define how often metrics should be logged, evaluation should be performed on the evaluation set, model saved, and what batch size to use. The model - in this configuration - **will not be** pushed to the 🤗 hub since it is quite slow. Make sure to authenticate, create a repo and push your model if you train a large model, or use a large dataset!

We will also use mixed precision (16-bit floating point, or fp16) if we are training on a GPU.

We will also instruct the model to use the `generate` method for evaluation. That method is used for inference, and it applies a decoding technique to the predicted logits. In this case, it will use greedy search, since we set the number of beams to 1. I briefly introduced decoding algorithgms in the [Decoder paragraph](https://marinone94.github.io/Whisper-paper/#decoder) of my first article, but for now you can simply think of it as selecting the next token as the highest probability, after applying a softmax to the logits. I am considering writing a post about the impact of decoding algorithms on Whisper performance, so let me know you are interested!

Last, we can track our training using several experiment tracking tools. I use Weights and Biases - great tool, you should definitely have a look - but 🤗 supports also "azure_ml", "comet_ml", "mlflow", "neptune" and "tensorboard". You can use "all" (default) to report to all integrations installed, "none" for no integrations. Since WandB is installed in this environment, you should explicitely set it to "none" if you don't have an account.


```python
## If you don't want to track your experiment with WandB, run this!
report_to = "none"
```


```python
# If you have a wandb account, login!
# Otherwise, edit this cell to loging with your favourite experiment tracker(s)
wandb.login()
wandb.init(project="whisper-training-post")
report_to = "wandb"
```


```python
# Check if we have a GPU.
# In case, we will use mixed precision
# to reduce memory footprint with
# with minimal to no harm to performance
device = "cuda" if torch.cuda.is_available() else "cpu"
use_fp16 = (device == "cuda")

# Let's first define the batch sizes
# Increase it if you have more than 16GB GPU
train_bs = 4 if test_script is True else 16
eval_bs = 2 if test_script is True else 8

# Then we infer the number of steps
# TODO: how did I find it?
num_training_samples = 2385
num_epochs = 3
max_steps_full_training = ceil(num_training_samples * num_epochs / train_bs)
max_steps = 2 if test_script is True else max_steps_full_training

# We don't want to evaluate too often since it slows down training a lot
eval_steps = 1 if test_script is True else int(max_steps / 10)
logging_steps = 1 if test_script is True else int(max_steps / 100)

# Init training arguments
training_args = Seq2SeqTrainingArguments(
    output_dir=output_dir,
    do_train=True,
    do_eval=True,
    max_steps=max_steps,  
    evaluation_strategy="steps",
    eval_steps=eval_steps,
    logging_strategy="steps",
    logging_steps=logging_steps,
    save_strategy="steps",
    save_steps=eval_steps,
    save_total_limit=3,
    learning_rate=7.5e-6,
    load_best_model_at_end=True,
    metric_for_best_model="wer",
    greater_is_better=False,
	  warmup_ratio=0.5 if test_script is True else 0.3,
    per_device_train_batch_size=train_bs,
    per_device_eval_batch_size=eval_bs,
    # important
    fp16=use_fp16,
    predict_with_generate=True,
    generation_num_beams=1,
    # track experiment
    report_to=report_to,
    # push model to hf hub recommended during training if
    # - training large model / on large datasets (i.e. training takes long)
    # - training on spot instances
    # if the instance crashes, you can resume your training from the checkpoint
    # see 🤗 docs for detailed instructions
    push_to_hub=False
)
```

Now we can provide the trainer with the model, tokenizer (important: use the one you assign language and task to! In this example, it is `processor.tokenizer`), training arguments, datasets, data collator, callback, and the method to compute metrics during evaluation.

Note that we don't need to place the model to the accelerator device, nor we had to do it in the data collator with the dataset! The trainer will take care of it, if a GPU is available.


```python
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
```

I hope you haven't left yet. If you have, bad for you, as we are ready for training our model! 🍾

As Whisper is a pretrained model ready to be used off-the-shelf, it is advisable to evaluate it before training on both the validation and test sets. Let's make sure we make no harm to it.


```python
eval_metrics = trainer.evaluate(
    eval_dataset=preprocessed_dataset["validation"],
    metric_key_prefix="eval_pretrained",
    max_length=448,
    num_beams=1,
    # gen_kwargs={"key": value}  to provide additional generation specific arguments by keyword
)

trainer.log_metrics("eval_pretrained", eval_metrics)
trainer.save_metrics("eval_pretrained", eval_metrics)
print(eval_metrics)
```

    ***** eval_pretrained metrics *****
      eval_pretrained_loss               =     1.7157
      eval_pretrained_runtime            = 0:03:17.02
      eval_pretrained_samples_per_second =      1.675
      eval_pretrained_steps_per_second   =      0.213
      eval_pretrained_wer                =    264.426
    {'eval_pretrained_loss': 1.71565842628479, 'eval_pretrained_wer': 264.42599393262014, 'eval_pretrained_runtime': 197.0203, 'eval_pretrained_samples_per_second': 1.675, 'eval_pretrained_steps_per_second': 0.213}



```python
test_metrics = trainer.evaluate(
    eval_dataset=preprocessed_dataset["test"],
    metric_key_prefix="test_pretrained",
    max_length=448,
    num_beams=1,
    # gen_kwargs={"key": value}  to provide additional generation specific arguments by keyword
)

trainer.log_metrics("test_pretrained", test_metrics)
trainer.save_metrics("test_pretrained", test_metrics)
print(test_metrics)
```

    ***** test_pretrained metrics *****
      test_pretrained_loss               =      1.725
      test_pretrained_runtime            = 0:04:33.25
      test_pretrained_samples_per_second =      2.778
      test_pretrained_steps_per_second   =      0.348
      test_pretrained_wer                =   261.9067
    {'test_pretrained_loss': 1.7249696254730225, 'test_pretrained_wer': 261.9066587001262, 'test_pretrained_runtime': 273.2544, 'test_pretrained_samples_per_second': 2.778, 'test_pretrained_steps_per_second': 0.348}



```python
train_result = trainer.train()
trainer.save_model()

metrics = train_result.metrics
trainer.log_metrics("train", metrics)
trainer.save_metrics("train", metrics)
trainer.save_state()
print(metrics)

```

    /usr/local/lib/python3.10/dist-packages/transformers/optimization.py:407: FutureWarning: This implementation of AdamW is deprecated and will be removed in a future version. Use the PyTorch implementation torch.optim.AdamW instead, or set `no_deprecation_warning=True` to disable this warning
      warnings.warn(




    <div>

      <progress value='448' max='448' style='width:300px; height:20px; vertical-align: middle;'></progress>
      [448/448 37:35, Epoch 2/9223372036854775807]
    </div>
    <table border="1" class="dataframe">
  <thead>
 <tr style="text-align: left;">
      <th>Step</th>
      <th>Training Loss</th>
      <th>Validation Loss</th>
      <th>Wer</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>44</td>
      <td>1.411100</td>
      <td>1.491944</td>
      <td>245.345681</td>
    </tr>
    <tr>
      <td>88</td>
      <td>1.050100</td>
      <td>1.225460</td>
      <td>225.882165</td>
    </tr>
    <tr>
      <td>132</td>
      <td>0.903200</td>
      <td>1.120284</td>
      <td>211.655756</td>
    </tr>
    <tr>
      <td>176</td>
      <td>0.814100</td>
      <td>1.067498</td>
      <td>184.623982</td>
    </tr>
    <tr>
      <td>220</td>
      <td>0.802900</td>
      <td>1.039361</td>
      <td>178.412901</td>
    </tr>
    <tr>
      <td>264</td>
      <td>0.632500</td>
      <td>1.030140</td>
      <td>216.637394</td>
    </tr>
    <tr>
      <td>308</td>
      <td>0.697100</td>
      <td>1.013514</td>
      <td>184.400447</td>
    </tr>
    <tr>
      <td>352</td>
      <td>0.605100</td>
      <td>1.006531</td>
      <td>194.714993</td>
    </tr>
    <tr>
      <td>396</td>
      <td>0.604700</td>
      <td>1.002950</td>
      <td>166.932780</td>
    </tr>
    <tr>
      <td>440</td>
      <td>0.585000</td>
      <td>1.004960</td>
      <td>191.234233</td>
    </tr>
  </tbody>
</table><p>


    ***** train metrics *****
      epoch                    =        2.33
      total_flos               = 163660945GF
      train_loss               =      0.8752
      train_runtime            =  0:37:40.84
      train_samples_per_second =        3.17
      train_steps_per_second   =       0.198
    {'train_runtime': 2260.8457, 'train_samples_per_second': 3.17, 'train_steps_per_second': 0.198, 'total_flos': 1.7572960198656e+17, 'train_loss': 0.8751586728862354, 'epoch': 2.33}


As we can see from the training logs, both the training and evaluation losses have plateaud at a high value, and the evaluation WER has decreased but was fluctuating. There are several reasons for this: however, they go beyond the scope of the article, which is already way too long. Today we focus on HOW to fine-tune the model, and WHAT happens inside the engine. In one of the next ones, we will train the best Whisper model we can. One thing at the time.

Okay, if you really can't wait, I'll spoiler you a couple of things: the model should have been trained longer, with a higher learning rate. And maybe the batch size was too large, considering the dataset size. Hyperparameters should be searched, if compute is available. Or look at others' work, and start from there. That's why open science advances humankind's knowledge at breakneck speed, isn't it?

Alright, let's evaluate it again on the held-out test set.


```python
final_metrics = trainer.evaluate(
    eval_dataset=preprocessed_dataset["test"],
    metric_key_prefix="test_finetuned",
    max_length=448,
    num_beams=1,
    # gen_kwargs={"key": value}  to provide additional generation specific arguments by keyword
)

trainer.log_metrics("test_finetuned", final_metrics)
trainer.save_metrics("test_finetuned", final_metrics)
print(final_metrics)
```

    ***** test_finetuned metrics *****
      epoch                             =       2.33
      test_finetuned_loss               =     1.0019
      test_finetuned_runtime            = 0:03:57.99
      test_finetuned_samples_per_second =      3.189
      test_finetuned_steps_per_second   =      0.399
      test_finetuned_wer                =   172.9602
    {'test_finetuned_loss': 1.0018519163131714, 'test_finetuned_wer': 172.96023368518888, 'test_finetuned_runtime': 237.997, 'test_finetuned_samples_per_second': 3.189, 'test_finetuned_steps_per_second': 0.399, 'epoch': 2.33}


As we can see, the test WER has dropped 34%, so the fine-tuning is definitely improving the model's performance. But what has happened during training? I mean, we did it all with a single line of code: `trainer.train()`. To improve your models, fix problems, and come out with innovative solutions, is essential to deeply understand what happens inside the engine. That is the subject of the next section. Before that, let's clean up our working environment and try out our new model.

As said, during training checkpoints are created. Let's clean up our directory. Don't worry, your best model is saved separately, so we can still push it to the 🤗 hub :)


```python
# Pushing to hub during training slows down training
# so we push it only in the end.
# Since training is completed and best model has been saved,
# we first delete the checkpoints
for filename in os.listdir(output_dir):
    if filename.startswith("checkpoint-"):
        shutil.rmtree(f"{output_dir}/{filename}")
trainer.push_to_hub()
```

Now we can load the fine-tuned model from the hub and use it for inference. Note that had we done some relevant changes to the processor, we should have had pushed it as well and loaded it hereafter like the model.


```python
model_id = f"{hf_user}/{repo_name}"
model = AutoModelForSpeechSeq2Seq.from_pretrained(model_id).to(device)
processor = AutoProcessor.from_pretrained("openai/whisper-tiny")
```

The following code block is copied from the [Training dataset](##training-dataset) chapter. You can skip it, if you have run it above.




```python
model_def_max_length = model.config.max_length

def transcribe(sample, return_pred_and_ref=False, print_results=False):
    if print_results is False and return_pred_and_ref is False:
        raise ValueError("Results are not printed nor returned.\n" +
                         "Set return_pred_and_ref to return results.\n" +
                         "Set print_results to print results to console.")

    # Load audio file (see previous cells)
    data = sample["audio"]["array"]
    sampling_rate = sample["audio"]["sampling_rate"]
    print("Audio loaded. Sample rate: ", sampling_rate)

    # Convert input audio to log-mel spectrogram
    input_features = processor(
        data, sampling_rate=sampling_rate, return_tensors="pt").input_features.to(device)

    # Get tokens to initiate transcription
    # and store them in the model config.
    # Under the hood, this will force the model to predict
    # these tokens in the beginning of the transcription.
    init_tokens = processor.get_decoder_prompt_ids(
        language="sv", task="transcribe", no_timestamps=True)
    model.config.forced_decoder_ids = init_tokens

    # Generate transcription tokens
    print("Generating transcription...")
    st = time.perf_counter_ns()
    transcription_token_ids = model.generate(
        input_features, max_new_tokens=model_def_max_length)
    et = time.perf_counter_ns()
    dt = (et - st) / 1e9
    print(f"Transcription generated in {dt} [s]")

    # Decode transcription
    # NOTE: input_features is a batch of one element,
    # and the returned token ids are batched analogously
    transcription_token_ids = transcription_token_ids[0]
    whisper_transcript = processor.decode(
        transcription_token_ids, skip_special_tokens=True).strip()

    # Print results and return if required
    reference_transcript = sample["raw_transcription"]
    if print_results is True:
        print("Whisper transcript:\n", whisper_transcript)
        print("Reference transcript:\n", reference_transcript)
    if return_pred_and_ref is True:
        return {
            "prediction": whisper_transcript,
            "label": reference_transcript
        }

```

The following code block is copied from the [Word Error Rate](##word-error-rate) chapter. You can skip it, if you have run it above.


```python
def wer(pred, label):
    # wer <= 1, we make it as %
    # and it wants list of predictions and references, not strings
    norm_pred = normalizer(pred)
    norm_label = normalizer(label)
    wer = 100 * metric.compute(predictions=[pred], references=[label])
    wer = round(wer, 2)

    print("-"*45)
    print("Prediction:", pred)
    print("Reference:", label)
    print("WER", wer)
    print("-"*45)

    norm_wer = 100 * metric.compute(predictions=[norm_pred], references=[norm_label])
    norm_wer = round(norm_wer, 2)

    print("Normalised prediction:", norm_pred)
    print("Normalised reference:", norm_label)
    print("WER", norm_wer)

    return wer, norm_wer
```


```python
response = transcribe(sample, return_pred_and_ref=True)
pred, label = response["prediction"], response["label"]
raw_wer, norm_wer = wer(pred, label)
```

    Audio loaded. Sample rate:  16000
    Generating transcription...
    Transcription generated in 0.222633967 [s]
    ---------------------------------------------
    Prediction: Liipströmmer är åtvänderna flödet från våger från Butravistan, ofta vid trev eller liknande.
    Reference: Ripströmmar är det återvändande flödet från vågor som bryter av vid stranden, ofta vid ett rev eller liknande.
    WER 61.11
    ---------------------------------------------
    Normalised prediction: liipströmmer är åtvänderna flödet från våger från butravistan ofta vid trev eller liknande 
    Normalised reference: ripströmmar är det återvändande flödet från vågor som bryter av vid stranden ofta vid ett rev eller liknande 
    WER 61.11


This inspection confirms that the WER has dropped on our random sample. 

### Inside the engine

As promised, we will now look inside the engine, and understand what happens exactly when we execute `trainer.train()`. To do so, we will re-implement and analyze the main steps required to fine-tune your Whisper model.


#### Step 1: a lot of checks and setup

If you inspect the `trainer` instantiation and the `train()` method, you will see a lot of steps that Trainer performs to validate setup, input data and parameters, and enable specific behaviours. This makes the Trainer suitable to be used for hyperparameter tuning, distributed and accelerated training, and many of the techniques recently invented. This article does not want to be an exaustive introduction to 🤗 Trainer API, and since we are training the model on a single GPU without using any special algortihm nor device, we don't need to go any further.

#### Step 2: place the model to device

In step 1, the Trainer will also automatically place the model on the most suitable device (i.e. a GPU, if it is available). This will be skipped when specific training configuration like Model Parallelism or DeepSpeed are set. We will do it manually here, since we are not going to use the Trainer API. Let's reload the pretrained model once again.


```python
from copy import copy, deepcopy
from dataclasses import dataclass

import torch
from datasets import IterableDatasetDict, load_dataset
from transformers import (
    AutoModelForSpeechSeq2Seq,
    AutoProcessor,
    WhisperProcessor
)
from transformers.models.whisper.english_normalizer import BasicTextNormalizer
```


```python
device = "cuda" if torch.cuda.is_available() else "cpu"
model = AutoModelForSpeechSeq2Seq.from_pretrained("openai/whisper-tiny").to(device)
processor = AutoProcessor.from_pretrained("openai/whisper-tiny")
```

#### Step 3: btach generation

As all Deep Learning models, Whisper is trained and fine-tuned in batches. This means that a bunch of examples are used together to compute the loss, and then the model parameters are updated through backpropagation. When writing a complete training loop, you will iterate over the training data and train your model one batch at the time. Here we will extract one batch and see what happens during a single iteration.

To do so, we will instantiate a Dataloader, which will take care of loading the next batch for us. It is important to understand what happens step by step, but we cannot go into the details of each line of code and rewrite everything from scratch.

The code blocks before the Dataloader instantiation are copied from the [Fine-tuning Whisper with 🤗](##fine-tuning-whisper-with-hug) chapter. You can skip it, if you have run it above.


```python
dataset_id = "google/fleurs"
dataset_language_code = "sv_se"
dataset = load_dataset(dataset_id, dataset_language_code, streaming=True)
```


```python
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
```


```python
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
```


```python
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
```


```python
data_collator = DataCollatorSpeechSeq2SeqWithPadding(
    processor=processor,
    decoder_start_token_id=model.config.decoder_start_token_id,
)
```


```python
dataloader = torch.utils.data.DataLoader(
    dataset=preprocessed_dataset["train"],
    batch_size=2,
    collate_fn=data_collator
)
```


```python
batch = next(iter(dataloader))
```


```python
batch["input_features"].shape
```




    torch.Size([2, 80, 3000])




```python
batch["labels"].shape
```




    torch.Size([2, 57])



Here we go! The dataloader has fetched the first two examples of the dataset and built a tensor with dimension 2, as it was the desired batch size specified in the dataloader. Since the dataset is loaded in streaming mode, the iteration over the dataset has triggered two main actions: it downloaded the data, and it applied the preprocessing defined in the `prepare_dataset()`method. Last, the dataloader has used the data collator to pad the input and prepare our first batch for training the model. If you are wondering why the other dimensions are 80 and 3000, you should definiteley check the [Audio Preprocessing section](https://marinone94.github.io/Whisper-paper/#audio-preprocessing) of my previous article to learn everything about the log-mel spectrogram. It is the standard audio preprocessing in ASR.

Last, we will upload the training data to the device.


```python
batch.to(device)
```




    {'input_features': tensor([[[-0.3325,  0.1385, -0.0556,  ..., -0.8577, -0.8577, -0.8577],
             [-0.5058,  0.0516, -0.2305,  ..., -0.8577, -0.8577, -0.8577],
             [-0.3280, -0.0614, -0.1021,  ..., -0.8577, -0.8577, -0.8577],
             ...,
             [-0.8577, -0.4861, -0.5043,  ..., -0.8577, -0.8577, -0.8577],
             [-0.8577, -0.5606, -0.3973,  ..., -0.8577, -0.8577, -0.8577],
             [-0.8577, -0.6364, -0.5126,  ..., -0.8577, -0.8577, -0.8577]],
    
            [[-0.4111, -0.4111, -0.4111,  ..., -0.4111, -0.4111, -0.4111],
             [-0.4111, -0.4111, -0.4111,  ..., -0.4111, -0.4111, -0.4111],
             [-0.4111, -0.4111, -0.4111,  ..., -0.4111, -0.4111, -0.4111],
             ...,
             [-0.4111, -0.4111, -0.4111,  ..., -0.4111, -0.4111, -0.4111],
             [-0.4111, -0.4111, -0.4111,  ..., -0.4111, -0.4111, -0.4111],
             [-0.4111, -0.4111, -0.4111,  ..., -0.4111, -0.4111, -0.4111]]],
           device='cuda:0'), 'labels': tensor([[50363,    46,    76,  1581,   364,    85, 16082,   465,  1224,  3239,
               543,    11, 47107,  1663,  4792,   317, 16770,   741,  4816, 17165,
                70, 16684,  3307,  3775,   220, 47189, 45845,   302,   951,  1387,
              1696,  3795, 15349,  2330,   220, 47189,   372,  1330,   273,  3307,
               350, 11397, 36959,    13, 50257,  -100,  -100,  -100,  -100,  -100,
              -100,  -100,  -100,  -100,  -100,  -100,  -100],
            [50363, 16257, 16770,   342,  9559, 23137, 37332,  1305,  6691, 18669,
               479,    45,    12, 22882,    70,  1505,  2830,  4792,   317, 16770,
              8124,   267,    11, 15349,  5758, 35368,  5657,   391,   811,   951,
               220,  1328,  4556, 25614,   741,   272,   351,    75, 33482,   220,
             47189,  5735,   897,   266,   642,    12,  3423, 33482,    11,   465,
              1305, 25752,   271, 42554,  9140,    13, 50257]], device='cuda:0')}



#### Step 4: forward pass

Now that we have loaded our training batch, we will execute a forward pass through the model. The output will then be compared with the labelled output to compute the loss.

Now, I don't know if you recall this detail, but in the data collator we cut the `<bos>` token in the beginning since the 🤗 Trainer appends it later. Since we are not using the 🤗 Trainer implementation, let's not forget it. I will copy-paste their implementation here.


```python
def shift_tokens_right(input_ids: torch.Tensor, pad_token_id: int, decoder_start_token_id: int):
    """
    Shift input ids one token to the right.
    """
    shifted_input_ids = input_ids.new_zeros(input_ids.shape)
    shifted_input_ids[:, 1:] = input_ids[:, :-1].clone()
    shifted_input_ids[:, 0] = decoder_start_token_id

    if pad_token_id is None:
        raise ValueError("self.model.config.pad_token_id has to be defined.")
    # replace possible -100 values in labels by `pad_token_id`
    shifted_input_ids.masked_fill_(shifted_input_ids == -100, pad_token_id)

    return shifted_input_ids
```


```python
decoder_input_ids = shift_tokens_right(
    batch.labels, model.config.pad_token_id, model.config.decoder_start_token_id
)
```


```python
decoder_input_ids
```




    tensor([[50258, 50363,    46,    76,  1581,   364,    85, 16082,   465,  1224,
              3239,   543,    11, 47107,  1663,  4792,   317, 16770,   741,  4816,
             17165,    70, 16684,  3307,  3775,   220, 47189, 45845,   302,   951,
              1387,  1696,  3795, 15349,  2330,   220, 47189,   372,  1330,   273,
              3307,   350, 11397, 36959,    13, 50257, 50257, 50257, 50257, 50257,
             50257, 50257, 50257, 50257, 50257, 50257, 50257],
            [50258, 50363, 16257, 16770,   342,  9559, 23137, 37332,  1305,  6691,
             18669,   479,    45,    12, 22882,    70,  1505,  2830,  4792,   317,
             16770,  8124,   267,    11, 15349,  5758, 35368,  5657,   391,   811,
               951,   220,  1328,  4556, 25614,   741,   272,   351,    75, 33482,
               220, 47189,  5735,   897,   266,   642,    12,  3423, 33482,    11,
               465,  1305, 25752,   271, 42554,  9140,    13]], device='cuda:0')



A quick verification will show us that we have prepended the `<bos>` token correctly.


```python
def batch_decode(input_ids):
  return processor.tokenizer.batch_decode(decoder_input_ids)
```


```python
batch_decode(decoder_input_ids)
```




    ['<|startoftranscript|><|notimestamps|>Om du använder en drönare, kontrollera ordentligt i förväg vad som är tillåtet att filma och vilka tillstånd som krävs.<|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|>',
     '<|startoftranscript|><|notimestamps|>Enligt stämningen blev avfall från FN-lägret inte ordentligt renat, vilket fick bakterier att ta sig ned i bifloden till Artibonite-floden, en av Haitis största.']




```python
outputs = model.model(batch.input_features, decoder_input_ids=decoder_input_ids)
```

Outputs is an instance of Seq2SeqModelOutput class implemented in 🤗 transformers library and is used in all Sequence to Sequence models. It contains the last encoder and decoder hidden states, and the accumulated key values. This must not be confused with the Seq2SeqLMOutput, which contains loss, logits, and past key values.


```python
type(outputs)
```




    transformers.modeling_outputs.Seq2SeqModelOutput




```python
outputs.keys()
```




    odict_keys(['last_hidden_state', 'past_key_values', 'encoder_last_hidden_state'])




```python
outputs.last_hidden_state.shape
```




    torch.Size([2, 57, 384])



We then use the model's head to project the last hidden decoder layer's state to the vocabulary dimension.


```python
logits = model.proj_out(outputs.last_hidden_state)
```

#### Step 5: backpropagation


```python
logits.shape
```




    torch.Size([2, 57, 51865])



As you can see, the logits have dimensions [`batch_size`, `sequence_length`, `vocab_size`]. Now we can (let PyTorch) compute the loss. To do so, we need to concatenate the outputs.


```python
from torch.nn import CrossEntropyLoss

cross_entropy_loss = CrossEntropyLoss()
```


```python
logits_reshaped = logits.view(-1, model.config.vocab_size)
labels_reshaped = batch.labels.reshape(-1)
print(logits_reshaped.shape)
print(labels_reshaped.shape)
```

    torch.Size([114, 51865])
    torch.Size([114])



```python
loss = cross_entropy_loss(logits_reshaped, labels_reshaped)
print(loss)
```

    tensor(3.2089, device='cuda:0', grad_fn=<NllLossBackward0>)


Finally, we go to the core of deep learning. Unfortunately you cannot easily debug the gradient computation step in Pytorch, as it invokes a C++ implementation for performance reasons. Fortunatley, tones of books and blogs have been written by real experts, and they explain it much better than I ever will in this limited time.


```python
loss.backward()
print(loss)
```

    tensor(3.2089, device='cuda:0', grad_fn=<NllLossBackward0>)



```python
loss = loss.detach()
print(loss)
```

    tensor(3.2089, device='cuda:0')


As you can see, the loss has been detached by the computational graph and will not be used any longer in the backpropagation, as the gradients have already been computed. Let's optimize the model, without forgetting to clip the gradient norm. This is done to prevent the gradients from becoming too large during training and making the training process unstable. In simple words, all the gradients are downscaled proportionally to a maximum value.


```python
from functools import partial

from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR
from torch.nn.utils import clip_grad_norm_
```


```python
clip_grad_norm_(model.parameters(), max_norm=1.0)
```




    tensor(63.5967, device='cuda:0')




```python
optimizer = AdamW(model.parameters(), lr=7.5e-3)
```


```python
# from huggingface implementation
def get_linear_schedule_with_warmup_lr_lambda(current_step: int, *, num_warmup_steps: int, num_training_steps: int):
    if current_step < num_warmup_steps:
        return float(current_step) / float(max(1, num_warmup_steps))
    return max(0.0, float(num_training_steps - current_step) / float(max(1, num_training_steps - num_warmup_steps)))
```


```python
num_training_steps = 100  # fictious number, this is training_samples * num_epochs / batch_size
warmup_ratio = 0.3
num_warmup_steps = num_training_steps * warmup_ratio
```


```python
# from huggingface implementation
lr_lambda = partial(
    get_linear_schedule_with_warmup_lr_lambda,
    num_warmup_steps=num_warmup_steps,
    num_training_steps=num_training_steps,
)
scheduler = LambdaLR(optimizer, lr_lambda, -1)
scheduler.step()
```

I hope this structure will not confuse you. In a training loop, the optimizer and the scheduler are initialized before the forward pass, but I preferred to keep them close to where they are used as we are navigating a single training step. Let's not forget to bring the scheduler a step forward, or the learning rate will be zero at the first step, meaning the paramters will not be updated.


```python
optimizer.step()
```


```python
scheduler.step()
```

Now the parameters have been updated based on the loss we just computed and the optimization algorithm. Last, let's not forget to reset the gradients, ready for the next batch!


```python
optimizer.zero_grad()
```

Cool, we have just inspected the basic steps required to fine-tune a Whisper model, or actually any deep learning model via backpropagation. Just repeat this over enough, good examples with the right hyperparameters and your model will "magically" become much better.

This might look trivial for experts, but many people today are relying on the 🤗 Trainer, and are not fully aware of the basic optimization steps.

## What's next

There are a bunch of things I am curious to investigate. Since training large models is expensive, I will investigate different training approaches and platforms. We will compare time, costs, final results and - when feasible - also CO2 equivalent emissions, so you will know in advance what approach suites best.

One of the platforms I will use is Lambda Cloud, which provides GPU instances with pre-installed Deep Learning packages. At the time of writing, however, no Python client is provided, and you need to ssh into the machine to run commands. The good news is that they expose some APIs, so before the next post you will most likely get an unofficial Python client for Lambda Cloud (and maybe I will get some free credits 😅).



## Thank you!

If a single person - excluding my closest friends, who really love me - is reading this chapter, then this post will be a success. I hope you found at least one concept you were not aware of, or you didn't think of.

The AI world is running at neckbreak speed, and sometimes I feel I am too slow to produce those articles. But it helps me first, to learn and challenge my knowledge. I will get faster and better, and your feedback is precious. Don't hesitate to share it.
