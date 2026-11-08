# paper-implementations

PyTorch implementations of deep learning papers, written from scratch to learn how they work. No pretrained weights unless I say otherwise.

## Papers

| Paper | Year | Folder | Notes |
|---|---|---|---|
| Attention Is All You Need (Vaswani et al.) | 2017 | `transformer/` | Full encoder-decoder, trained on a toy copy task |
| LoRA (Hu et al.) | 2021 | `lora/` | Added to a small GPT-style model that I pretrain myself, then tuned on a toy sentiment task |
| Neural Machine Translation by Jointly Learning to Align and Translate (Bahdanau et al.) | 2015 | `bahdanau_attention/` | Seq2seq with additive attention, toy reversal task |

## About

I wrote these to learn. I try not to copy from the official repos, so some of it is messy and slower than it should be.

All the tasks are synthetic so everything runs on a laptop CPU in a few minutes. Nothing needs a download.

## Running

```bash
git clone https://github.com/AAAAAXfda/paper-implementations
cd paper-implementations
pip install -r requirements.txt

# transformer
python transformer/train.py

# LoRA demo
python lora/lora_demo.py

# Bahdanau attention
python bahdanau_attention/train.py
```

## Results

Fill these in after you run each script.

| Model | Task | Metric | Score |
|---|---|---|---|
| Transformer (from scratch) | Copy task | Exact match | TBD |
| GPT-style + LoRA | Toy sentiment | Accuracy | TBD |
| Seq2seq + additive attention | Reversal task | Exact match | TBD |
