import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as T
import torchvision.models as models

class EncoderCNNSpatial(nn.Module):
    def __init__(self, embed_dim=512, pretrained=True, train_backbone=False):
        super().__init__()
        resnet = models.resnet101(pretrained=pretrained)
        # remove last fc & avgpool so we have spatial map after layer4
        modules = list(resnet.children())[:-2]  # up to conv5_x -> outputs (B, 2048, H/32, W/32)
        self.backbone = nn.Sequential(*modules)
        self.avgpool = nn.AdaptiveAvgPool2d((1,1))
        self.fc = nn.Linear(2048, embed_dim)
        self.bn = nn.BatchNorm1d(embed_dim, momentum=0.01)

        # freeze backbone optionally
        if not train_backbone:
            for p in self.backbone.parameters():
                p.requires_grad = False

    def forward(self, images):
        # images: (B,3,224,224)
        feat_map = self.backbone(images)  # (B, 2048, Hf, Wf)
        b, c, h, w = feat_map.shape
        # pooled vector
        pooled = self.avgpool(feat_map).view(b, c)
        embed = self.fc(pooled)  # (B, embed_dim)
        embed = self.bn(embed)
        # flatten spatial features for attention: (B, H*W, C)
        spatial = feat_map.view(b, c, -1).permute(0,2,1)  # (B, num_patches, C)
        return spatial, embed  # spatial used by attention, embed as initial hidden
    

# Cell 7: Attention + Decoder
class BahdanauAttention(nn.Module):
    def __init__(self, encoder_dim, decoder_dim, attention_dim):
        super().__init__()
        self.enc_att = nn.Linear(encoder_dim, attention_dim)  # transform encoder
        self.dec_att = nn.Linear(decoder_dim, attention_dim)  # transform decoder hidden
        self.full_att = nn.Linear(attention_dim, 1)

    def forward(self, encoder_feats, decoder_hidden, mask=None):
        # encoder_feats: (B, num_patches, encoder_dim)
        # decoder_hidden: (B, decoder_dim)
        enc_proj = self.enc_att(encoder_feats)  # (B, N, attn_dim)
        dec_proj = self.dec_att(decoder_hidden).unsqueeze(1)  # (B,1,attn_dim)
        e = torch.tanh(enc_proj + dec_proj)  # (B,N,attn_dim)
        scores = self.full_att(e).squeeze(-1)  # (B,N)
        if mask is not None:
            # mask shape for spatial features - usually None (images always full)
            scores = scores.masked_fill(~mask, -1e9)
        alpha = F.softmax(scores, dim=1)  # (B,N)
        context = (encoder_feats * alpha.unsqueeze(-1)).sum(dim=1)  # (B, encoder_dim)
        return context, alpha

class DecoderWithAttention(nn.Module):
    def __init__(self, embed_dim, decoder_dim, vocab_size, encoder_dim=512,
                 attention_dim=512, dropout=0.5, padding_idx=0):
        super().__init__()
        self.encoder_dim = encoder_dim
        self.decoder_dim = decoder_dim
        self.vocab_size = vocab_size

        self.attention = BahdanauAttention(encoder_dim, decoder_dim, attention_dim)
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=padding_idx)
        self.dropout = nn.Dropout(p=dropout)
        self.lstm = nn.LSTMCell(embed_dim + encoder_dim, decoder_dim)

        # Initialize hidden state from encoder pooled features
        self.init_h = nn.Linear(encoder_dim, decoder_dim)
        self.init_c = nn.Linear(encoder_dim, decoder_dim)

        # Output layer
        self.fc = nn.Linear(decoder_dim, vocab_size)

        # Optional: attention gating
        self.f_beta = nn.Linear(decoder_dim, encoder_dim)
        self.sigmoid = nn.Sigmoid()

    def init_hidden_state(self, encoder_pooled):
        """Initialize LSTM hidden state from encoder output"""
        h = torch.tanh(self.init_h(encoder_pooled))
        c = torch.tanh(self.init_c(encoder_pooled))
        return h, c

    def forward(self, encoder_feats, encoder_pooled, captions, teacher_forcing_ratio=1.0):
        """
        Forward pass with proper teacher forcing
        Args:
            encoder_feats: (B, num_pixels, encoder_dim)
            encoder_pooled: (B, encoder_dim)
            captions: (B, max_len) - includes <start> at position 0
            teacher_forcing_ratio: probability of using ground truth
        Returns:
            outputs: (B, max_len, vocab_size) - predictions for positions 1 to max_len
        """
        batch_size = encoder_feats.size(0)
        seq_length = captions.size(1)

        # Initialize hidden state
        h, c = self.init_hidden_state(encoder_pooled)

        # Prepare outputs tensor
        outputs = torch.zeros(batch_size, seq_length, self.vocab_size,
                             device=encoder_feats.device)

        # Get embeddings for all caption tokens
        embeddings = self.embedding(captions)  # (B, seq_len, embed_dim)

        # Start with <start> token
        current_input = embeddings[:, 0, :]  # (B, embed_dim)

        # Generate predictions for positions 1 to seq_length
        for t in range(1, seq_length):
            # Attention
            context, alpha = self.attention(encoder_feats, h)

            # Optional gating
            if hasattr(self, 'f_beta'):
                gate = self.sigmoid(self.f_beta(h))
                context = gate * context

            # LSTM step
            lstm_input = torch.cat([current_input, context], dim=1)
            h, c = self.lstm(lstm_input, (h, c))

            # Predict next word
            output = self.fc(self.dropout(h))  # (B, vocab_size)
            outputs[:, t, :] = output

            # Teacher forcing decision
            use_teacher = random.random() < teacher_forcing_ratio
            if use_teacher and t < seq_length - 1:
                # Use ground truth
                current_input = embeddings[:, t, :]
            else:
                # Use prediction
                predicted_token = output.argmax(dim=-1)
                current_input = self.embedding(predicted_token)

        return outputs

    def sample_beam(self, encoder_feats, encoder_pooled, sos_idx, eos_idx, beam_size=3, max_len=20):
        # Beam search implementation (batch size = 1 only for simplicity)
        assert encoder_feats.size(0) == 1, "Beam search currently supports batch_size=1"
        device = encoder_feats.device
        # Flatten input for convenience
        enc = encoder_feats  # (1,N,enc_dim)
        pooled = encoder_pooled  # (1,enc_dim)
        h, c = self.init_hidden_state(pooled)  # (1, dec_dim)
        # Each beam: (tokens, logprob, h, c)
        beams = [([sos_idx], 0.0, h, c)]
        completed = []
        for _ in range(max_len):
            new_beams = []
            for tokens, score, h_b, c_b in beams:
                if tokens[-1] == eos_idx:
                    completed.append((tokens, score))
                    continue
                # get last token embedding
                last_idx = torch.tensor([tokens[-1]], device=device)
                emb = self.embedding(last_idx)  # (1, emb)
                context, _ = self.attention(enc, h_b)
                lstm_input = torch.cat([emb.squeeze(0), context], dim=0).unsqueeze(0)  # (1, emb+enc)
                h_new, c_new = self.lstm(lstm_input, (h_b, c_b))
                out = F.log_softmax(self.fc(h_new), dim=-1).squeeze(0)  # (vocab,)
                topk_logprobs, topk_idx = torch.topk(out, beam_size)
                for k_logp, k_idx in zip(topk_logprobs.tolist(), topk_idx.tolist()):
                    new_tokens = tokens + [int(k_idx)]
                    new_score = score + float(k_logp)
                    new_beams.append((new_tokens, new_score, h_new, c_new))
            # keep top beams
            new_beams = sorted(new_beams, key=lambda x: x[1], reverse=True)[:beam_size]
            beams = new_beams
            if len(beams) == 0:
                break
        completed.extend([(b[0], b[1]) for b in beams])
        completed = sorted(completed, key=lambda x: x[1], reverse=True)
        best_tokens = completed[0][0]
        return best_tokens