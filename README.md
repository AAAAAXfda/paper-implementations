# paper-implementations

PyTorch implementations of deep learning papers, written from scratch to learn how they work. No pretrained weights unless I say otherwise.

## Papers

| Paper | Year | Folder | Notes |
|---|---|---|---|
| Attention Is All You Need (Vaswani et al.) | 2017 | `transformer/` | Full encoder-decoder, trained on a toy translation task |
| LoRA (Hu et al.) | 2021 | `lora/` | Plugged into a small GPT-2 style model |
| Neural Machine Translation by Jointly Learning to Align and Translate (Bahdanau et al.) | 2015 | `bahdanau_attention/` | Seq2seq with additive attention |

## About

I wrote these to learn. I try not to copy from the official repos, so some of it is messy and slower than it should be.

## Running

```bash
git clone https://github.com/AAAAAXfda/paper-implementations
cd paper-implementations
pip install -r requirements.txt

# transformer
python transformer/train.py

# LoRA demo
python lora/lora_demo.py
```

## Results

| Model | Task | Metric | Score |
|---|---|---|---|
| Transformer (from scratch) | Copy task | Accuracy | 99.1% |
| LoRA on GPT-2 small | Sentiment | Accuracy | 91.3% |
