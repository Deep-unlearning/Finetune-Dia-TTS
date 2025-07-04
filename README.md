# Dia TTS Fine-Tuning on DailyTalk Conversations with Transformers 🤗

This project fine-tunes the [nari-labs/Dia-1.6B-0626](https://huggingface.co/nari-labs/Dia-1.6B-0626) text-to-speech model on the [eustlb/dailytalk-conversations-grouped](https://huggingface.co/datasets/eustlb/dailytalk-conversations-grouped) multi-turn conversational dataset using the Hugging Face `transformers` and `datasets` libraries.

## Installation

### Step 1: Clone the repository

```bash
git clone https://github.com/Deep-unlearning/dia-tts-dailytalk.git
cd dia-tts-dailytalk
```

### Step 2: Set up environment

Choose your preferred package manager:

<details>
<summary>📦 Using UV (recommended)</summary>

[Install `uv`](https://docs.astral.sh/uv/getting-started/installation/)


```bash
uv venv .venv --python 3.10 && source .venv/bin/activate
uv pip install -r requirements.txt
```
</details>

<details>
<summary>🐍 Using pip</summary>

```bash
python -m venv .venv --python 3.10 && source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```
</details>

### Step 3: (Optional) Install flash-attn for faster training

You can also install flash-attn for faster training:

```bash
uv pip install flash-attn --no-cache-dir
```

and then you can use the model with flash-attention:

```python
model = DiaForConditionalGeneration.from_pretrained("nari-labs/Dia-1.6B-0626", attn_implementation="flash_attention_2")
```

### Step 4: Prepare the dataset

If you're using the default `eustlb/dailytalk-conversations-grouped` dataset, you're good to go.

If you're using your own dataset, ensure it's preprocessed into grouped multi-turn conversations with speaker tags (`[S1]`, `[S2]`, etc.) like so:


### Step 5: Run the training script

```bash
uv run train.py
```

**Happy fine-tuning!** 🚀 