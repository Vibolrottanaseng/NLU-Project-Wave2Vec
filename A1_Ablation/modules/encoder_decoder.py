import torch
import torch.nn as nn


class EncoderDecoder(nn.Module):
    """
    Clean Transformer Encoder-Decoder for report generation
    """

    def __init__(self, args, tokenizer):
        super().__init__()

        self.tokenizer = tokenizer
        self.vocab_size = tokenizer.get_vocab_size()
        self.pad_idx = tokenizer.pad_idx

        d_model = args.d_model

        # =========================
        # IMAGE ENCODER
        # =========================
        self.att_embed = nn.Linear(d_model, d_model)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=8,
            batch_first=True
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=2)

        # =========================
        # TEXT EMBEDDING
        # =========================
        self.embed = nn.Embedding(
            self.vocab_size,
            d_model,
            padding_idx=self.pad_idx
        )

        # =========================
        # DECODER (CAUSAL)
        # =========================
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=d_model,
            nhead=8,
            batch_first=True
        )
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers=2)

        self.fc = nn.Linear(d_model, self.vocab_size)

    # ======================================================
    # FORWARD
    # ======================================================
    def forward(self, memory, seq=None, mode='train'):

        memory = self.att_embed(memory)
        memory = self.encoder(memory)

        if mode == 'train':
            return self._forward(memory, seq)
        else:
            return self._sample(memory)

    # ======================================================
    # TRAINING
    # ======================================================
    def _forward(self, memory, seq):

        embeddings = self.embed(seq)

        seq_len = seq.size(1)
        device = seq.device

        tgt_mask = nn.Transformer.generate_square_subsequent_mask(seq_len).to(device)

        outputs = self.decoder(
            embeddings,
            memory,
            tgt_mask=tgt_mask
        )

        logits = self.fc(outputs)

        return logits

    # ======================================================
    # SAMPLING
    # ======================================================
    def _sample(self, memory, max_len=50):

        batch_size = memory.size(0)
        device = memory.device

        generated = torch.full(
            (batch_size, max_len),
            self.pad_idx,
            dtype=torch.long,
            device=device
        )

        for t in range(max_len):

            seq = generated[:, :t+1]
            embeddings = self.embed(seq)

            tgt_mask = nn.Transformer.generate_square_subsequent_mask(
                seq.size(1)
            ).to(device)

            outputs = self.decoder(
                embeddings,
                memory,
                tgt_mask=tgt_mask
            )

            logits = self.fc(outputs)

            probs = torch.softmax(logits[:, -1], dim=-1)
            next_token = torch.multinomial(probs, 1).squeeze(1)

            generated[:, t] = next_token

        return generated