# app.py
import os
import json
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from PIL import Image
import gradio as gr

# -------------------------
# Load vocabulary files
# -------------------------
# Your repo listing shows word2idx.json and idx2word.json — use those
with open("word2idx.json", "r", encoding="utf-8") as f:
    word2idx = json.load(f)

with open("idx2word.json", "r", encoding="utf-8") as f:
    # idx2word.json in your folder is stored as a list (itos)
    itos = json.load(f)

# Build reverse mapping (safe)
# idx2word may be list (itos) or dict (string keys) depending on how you saved it
if isinstance(itos, list):
    idx2word = {i: w for i, w in enumerate(itos)}
else:
    # If stored as dict with string keys (e.g. {"0": "xxx", ...}), convert keys to int
    idx2word = {int(k): v for k, v in itos.items()}

vocab_size = len(word2idx)
print(f"Loaded vocab size = {vocab_size}")

# Special token indices (may be None if missing)
start_token = word2idx.get("<start>")
end_token = word2idx.get("<end>")
unk_token = word2idx.get("<unk>")

# -------------------------
# Device
# -------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device:", device)

# -------------------------
# Import your model classes
# -------------------------
# Your local file is model.py or model(s) in the repo. The notebook used:
# EncoderCNNSpatial and DecoderWithAttention definitions (matching the model.py you shared).
# Make sure the import path matches (if your file is model.py use `from model import ...`)
from model import EncoderCNNSpatial, DecoderWithAttention

# -------------------------
# Model Hyperparams (match notebook)
# -------------------------
ENC_EMBED = 512       # pooled encoder embedding dim used in notebook
ENCODER_DIM_RAW = 2048  # raw resnet feature dim
ATTN_DIM = 512
EMBED_DIM = 300      # word embedding dimension used during training
DECODER_DIM = 512

# -------------------------
# Build model objects (exact structure used for training)
# -------------------------
# Encoder: uses embed_dim=512 (so encoder.fc outputs 512-d pooled features)
encoder = EncoderCNNSpatial(embed_dim=ENC_EMBED, pretrained=True, train_backbone=False).to(device)

# Projection: maps raw per-patch 2048-d -> 512-d (this exists in checkpoint)
proj = nn.Linear(ENCODER_DIM_RAW, ENC_EMBED).to(device)

# Decoder: NOTE the encoder_dim passed here must be ENC_EMBED (512)
decoder = DecoderWithAttention(
    embed_dim=EMBED_DIM,
    decoder_dim=DECODER_DIM,
    vocab_size=vocab_size,
    encoder_dim=ENC_EMBED,
    attention_dim=ATTN_DIM,
    padding_idx=word2idx.get("<pad>", 0)
).to(device)

# -------------------------
# Load checkpoint (encoder, proj, decoder)
# -------------------------
ckpt_path = "flickr8k_caption_best.pt"   # adjust name if needed
if not os.path.exists(ckpt_path):
    raise FileNotFoundError(f"Checkpoint not found at {ckpt_path}. Place your .pt in the same folder as app.py")

checkpoint = torch.load(ckpt_path, map_location=device)
print("Loaded checkpoint keys:", list(checkpoint.keys()))

# Load encoder weights (state_dict)
if "encoder" in checkpoint:
    encoder.load_state_dict(checkpoint["encoder"])
    print("Loaded encoder weights.")
else:
    # If checkpoint saved full encoder object (unlikely), handle fallback:
    try:
        enc_obj = checkpoint.get("enc_obj", None)
        if enc_obj is not None:
            encoder = enc_obj.to(device)
            print("Loaded encoder object from checkpoint.")
    except Exception:
        print("No 'encoder' key found — checkpoint format unexpected.")

# Load projection weights if present
if "proj" in checkpoint:
    try:
        proj.load_state_dict(checkpoint["proj"])
        print("Loaded projection layer weights.")
    except Exception as e:
        print("Could not load proj weights:", e)
else:
    print("No 'proj' key found in checkpoint; continuing (but you likely need proj).")

# Load decoder weights (use strict=False in case vocab mapping differs)
if "decoder" in checkpoint:
    try:
        decoder.load_state_dict(checkpoint["decoder"], strict=False)
        print("Loaded decoder weights (strict=False).")
    except RuntimeError as e:
        # show mismatch but continue
        print("Decoder load error (continuing with strict=False):", e)
        decoder.load_state_dict(checkpoint["decoder"], strict=False)
else:
    print("No 'decoder' key found in checkpoint; aborting.")
    raise RuntimeError("No decoder in checkpoint")

# If checkpoint stored vocab_itos / vocab_stoi, we can also use it for exact match
vocab_itos = checkpoint.get("vocab_itos", None)
vocab_stoi = checkpoint.get("vocab_stoi", None)
if vocab_itos:
    print("Checkpoint contains vocab_itos length:", len(vocab_itos))

# Set eval mode
encoder.eval()
proj.eval()
decoder.eval()

# -------------------------
# Image transforms (match training)
# -------------------------
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize((0.485, 0.456, 0.406),
                         (0.229, 0.224, 0.225))
])

# -------------------------
# Inference helpers
# -------------------------
def post_process_caption(text: str) -> str:
    text = text.strip()
    if not text:
        return text
    # Capitalize and ensure final punctuation
    text = text[0].upper() + text[1:]
    if text[-1] not in ".!?":
        text = text + "."
    return text

# Greedy decode step helper (decoder forward signature in notebook).
# Note: the notebook's decoder forward expects:
#     outputs = decoder(encoder_feats_proj, encoder_pooled, captions, ...)
# but for step-by-step inference we will use decoder's attention/lstm methods.
# The notebook's DecoderWithAttention implementation exposes:
#   - init_hidden_state(encoder_pooled) -> h,c
#   - attention and LSTM in forward() and sample_beam function.
# For greedy: we can emulate the training-time timestep operations by calling the decoder's
# layers in the same order as in the notebook.
def greedy_caption(encoder_spatial_raw, encoder_pooled, max_len=20):
    """
    encoder_spatial_raw: (1, N, 2048)
    encoder_pooled: (1, 512)
    """
    # project spatial features to encoder_dim (512)
    encoder_spatial_proj = proj(encoder_spatial_raw)   # (1, N, 512)
    # initialize LSTM hidden/cell from pooled vector (decoder uses its own init)
    h, c = decoder.init_hidden_state(encoder_pooled)   # (1, dec_dim)
    # start token
    if start_token is None:
        # fallback to index 1 if not present
        cur = torch.tensor([1], device=device)
    else:
        cur = torch.tensor([start_token], device=device)

    caption_idxs = []
    for t in range(max_len):
        # attention: decoder.attention expects (encoder_feats, decoder_hidden)
        # In your notebook, attention.forward returns (context, alpha)
        context, alpha = decoder.attention(encoder_spatial_proj, h)
        # gating if present
        if hasattr(decoder, "f_beta"):
            gate = decoder.sigmoid(decoder.f_beta(h))
            context = gate * context
        # embed current token
        emb = decoder.embedding(cur)  # (1, embed_dim)
        # LSTM step: lstm input = [emb, context]
        # Ensure both emb and context are 2D (beam_size, feature_dim)
        emb = emb.view(emb.size(0), -1)
        context = context.view(context.size(0), -1)

        # Concatenate along feature dimension
        lstm_input = torch.cat([emb, context], dim=1)

        h, c = decoder.lstm(lstm_input, (h, c))
        out = decoder.fc(decoder.dropout(h))  # (1, vocab_size)
        pred = out.argmax(dim=-1)  # (1,)
        idx = pred.item()
        if idx == end_token:
            break
        caption_idxs.append(idx)
        cur = pred  # next input token

    # map indices to words and postprocess
    words = []
    for idx in caption_idxs:
        word = idx2word.get(idx, "<unk>")
        if word in ("<start>", "<end>", "<pad>"):
            continue
        words.append(word)
    caption = " ".join(words)
    caption = post_process_caption(caption)
    return caption

# Optionally implement beam search wrapper by invoking decoder.sample_beam if present.
def generate_caption_from_pil(pil_img, beam_size=3, max_len=20):
    img = transform(pil_img).unsqueeze(0).to(device)
    with torch.no_grad():
        encoder_spatial_raw, encoder_pooled = encoder(img)   # spatial: (B,N,2048), pooled: (B,512)
        # For single-image batch:
        if hasattr(decoder, "sample_beam") and beam_size > 1:
            # decoder.sample_beam expects batch size 1 and returns indices list
            tokens = decoder.sample_beam(
                proj(encoder_spatial_raw),   # expects projected features? in notebook they likely passed projected features (confirm)
                encoder_pooled,
                sos_idx=start_token if start_token is not None else 1,
                eos_idx=end_token if end_token is not None else 2,
                beam_size=beam_size,
                max_len=max_len
            )
            # tokens is a list of indices (including start)
            # remove start token if present and convert to words
            if tokens and tokens[0] == start_token:
                tokens = tokens[1:]
            words = [idx2word.get(int(t), "<unk>") for t in tokens if idx2word.get(int(t), "<unk>") not in ("<start>", "<end>", "<pad>")]
            caption = post_process_caption(" ".join(words))
            return caption
        else:
            return greedy_caption(encoder_spatial_raw, encoder_pooled, max_len=max_len)

# -------------------------
# Gradio function wrapper
# -------------------------
def generate_caption(image, beam_size: int = 1):
    try:
        # beam_size argument from UI; if >1 use beam search (if available)
        bs = int(beam_size)
    except:
        bs = 1
    caption = generate_caption_from_pil(image, beam_size=bs, max_len=MAX_LEN if ( (MAX_LEN := 30) ) else 30)
    return caption

# -------------------------
# Gradio UI
# -------------------------
demo = gr.Interface(
    fn=generate_caption,
    inputs=[gr.Image(type="pil", label="Input Image"),
            gr.Slider(minimum=1, maximum=5, step=1, label="Beam size (1 = greedy)")],
    outputs=gr.Textbox(label="Generated Caption"),
    title="Image Caption Generator",
    description="Upload an image and get a caption (greedy or beam)."
)

if __name__ == "__main__":
    demo.launch()
