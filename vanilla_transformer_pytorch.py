# https://towardsdatascience.com/build-your-own-transformer-from-scratch-using-pytorch-84c850470dcb

import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data
import math
import copy


class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, num_heads):
        super().__init__()
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"

        self.d_model = d_model  # 512
        self.num_heads = num_heads  # 8
        self.d_k = d_model // num_heads  # 512 // 8 = 64

        self.W_q = nn.Linear(d_model, d_model)  # (d_model, num_heads*d_k): (512, 512)
        self.W_k = nn.Linear(d_model, d_model)  # (d_model, num_heads*d_k): (512, 512)
        self.W_v = nn.Linear(d_model, d_model)  # (d_model, num_heads*d_k): (512, 512)
        self.W_o = nn.Linear(d_model, d_model)  # (num_heads*d_k, d_model): (512, 512)

    def scaled_dot_product_attention(self, Q, K, V, mask=None):
        attn_scores = torch.matmul(Q, K.transpose(-2, -1))  # (4, 8, 197, 197)
        attn_scores /= math.sqrt(self.d_k)  # (4, 8, 197, 197)
        if mask is not None:
            attn_scores = attn_scores.masked_fill(mask == 0, -1e9)
        attn_probs = torch.softmax(attn_scores, dim=-1)  # (4, 8, 197, 197)
        output = torch.matmul(attn_probs, V)  # (4, 8, 197, 64)
        return output

    def split_heads(self, x):
        batch_size, seq_length, d_model = x.shape
        x = x.view(batch_size, seq_length, self.num_heads, self.d_k)  # (4, 197, 8, 64)
        x = x.transpose(1, 2)  # (4, 8, 197, 64)
        return x

    def combine_heads(self, x):
        batch_size, num_heads, seq_length, d_k = x.shape
        x = x.transpose(1, 2)  # (4, 197, 8, 64)
        x = x.contiguous()  # (4, 197, 8, 64)  TODO not sure what it is doing?
        x = x.view(batch_size, seq_length, self.d_model)  # (4, 197, 512)
        return x

    def forward(self, q, k, v, mask=None):
        Q = self.split_heads(self.W_q(q))  # (batch, head, seq, d_k) (4, 8, 197, 64)
        K = self.split_heads(self.W_k(k))  # (batch, head, seq, d_k) (4, 8, 197, 64)
        V = self.split_heads(self.W_v(v))  # (batch, head, seq, d_k) (4, 8, 197, 64)
        attn_output = self.scaled_dot_product_attention(Q, K, V, mask)  # (4, 8, 197, 64)
        output = self.W_o(self.combine_heads(attn_output))  # (4, 197, 512)
        return output


class PositionWiseFeedForward(nn.Module):
    def __init__(self, d_model, d_ff):
        super().__init__()
        self.fc1 = nn.Linear(d_model, d_ff)  # (512, 2048)
        self.fc2 = nn.Linear(d_ff, d_model)  # (2048, 512)
        self.relu = nn.ReLU()

    def forward(self, x):
        return self.fc2(self.relu(self.fc1(x)))


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_seq_length):
        super().__init__()

        pe = torch.zeros(max_seq_length, d_model)
        position = torch.arange(0, max_seq_length, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * -(math.log(10000.0) / d_model))

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        self.register_buffer('pe', pe.unsqueeze(0))

    def forward(self, x):
        return x + self.pe[:, :x.size(1)]


class EncoderLayer(nn.Module):
    def __init__(self, d_model, num_heads, d_ff, dropout):
        super().__init__()
        self.self_attn = MultiHeadAttention(d_model, num_heads)
        self.feed_forward = PositionWiseFeedForward(d_model, d_ff)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask):
        attn_output = self.self_attn(x, x, x, mask)  # (4, 197, 512)
        x = self.norm1(x + self.dropout(attn_output))  # (4, 197, 512)
        ff_output = self.feed_forward(x)  # (4, 197, 512)
        x = self.norm2(x + self.dropout(ff_output))  # (4, 197, 512)
        return x


class DecoderLayer(nn.Module):
    def __init__(self, d_model, num_heads, d_ff, dropout):
        super(DecoderLayer, self).__init__()
        self.self_attn = MultiHeadAttention(d_model, num_heads)
        self.cross_attn = MultiHeadAttention(d_model, num_heads)
        self.feed_forward = PositionWiseFeedForward(d_model, d_ff)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, enc_output, src_mask, tgt_mask):
        attn_output = self.self_attn(x, x, x, tgt_mask)  # (4, 196, 512)  # TODO which axis is being masked?
        x = self.norm1(x + self.dropout(attn_output))  # (4, 196, 512)
        attn_output = self.cross_attn(x, enc_output, enc_output, src_mask)  # (4, 196, 512)  # TODO which axis is being masked?
        x = self.norm2(x + self.dropout(attn_output))  # (4, 196, 512)
        ff_output = self.feed_forward(x)  # (4, 196, 512)
        x = self.norm3(x + self.dropout(ff_output))  # (4, 196, 512)
        return x


class Transformer(nn.Module):
    def __init__(self, src_vocab_size, tgt_vocab_size, d_model, num_heads, num_layers, d_ff, max_seq_length, dropout):
        super().__init__()
        self.encoder_embedding = nn.Embedding(src_vocab_size, d_model)  # (5000, 512)
        self.decoder_embedding = nn.Embedding(tgt_vocab_size, d_model)  # (5000, 512)
        self.positional_encoding = PositionalEncoding(d_model, max_seq_length)

        self.encoder_layers = nn.ModuleList([EncoderLayer(d_model, num_heads, d_ff, dropout) for _ in range(num_layers)])
        self.decoder_layers = nn.ModuleList([DecoderLayer(d_model, num_heads, d_ff, dropout) for _ in range(num_layers)])

        self.fc = nn.Linear(d_model, tgt_vocab_size)  # (512, 5000)
        self.dropout = nn.Dropout(dropout)

    def generate_mask(self, src, tgt):  # TODO what is this doing?
        # src: (4, 197)
        # tgt: (4, 196)
        src_mask = (src != 0).unsqueeze(1).unsqueeze(2)  # all Trues. (4, 1, 1, 197)
        tgt_mask = (tgt != 0).unsqueeze(1).unsqueeze(3)  # (4, 1, 196, 1)
        seq_length = tgt.size(1)  # 196
        nopeak_mask = (1 - torch.triu(torch.ones(1, seq_length, seq_length), diagonal=1)).bool()  # (1, 196, 196)
        tgt_mask = tgt_mask & nopeak_mask  # (4, 1, 196, 196)
        return src_mask, tgt_mask

    def forward(self, src, tgt):
        # src: (4, 197)
        # tgt: (4, 196)
        src_mask, tgt_mask = self.generate_mask(src, tgt)
        # src_mask: (4, 1, 1, 197) # TODO Does it make sense?
        # tgt_mask: (4, 1, 196, 196)
        src_embedded = self.dropout(self.positional_encoding(self.encoder_embedding(src)))  # (4, 197, 512)
        tgt_embedded = self.dropout(self.positional_encoding(self.decoder_embedding(tgt)))  # (4, 196, 512)

        enc_output = src_embedded  # (4, 197, 512)
        for enc_layer in self.encoder_layers:
            enc_output = enc_layer(enc_output, src_mask)  # (4, 197, 512)

        dec_output = tgt_embedded  # (4, 196, 512)
        for dec_layer in self.decoder_layers:
            dec_output = dec_layer(dec_output, enc_output, src_mask, tgt_mask)  # (4, 196, 512)

        output = self.fc(dec_output)  # (4, 196, 5000)
        return output




if __name__ == "__main__":

    src_vocab_size = 5000
    tgt_vocab_size = 5000
    d_model = 512
    num_heads = 8
    num_layers = 6
    d_ff = 2048
    max_seq_length = 197
    dropout = 0.1
    batch_size = 4

    # Generate random sample data
    src_data = torch.randint(1, src_vocab_size, (batch_size, max_seq_length))  # (batch_size, seq_length): (4, 197)
    tgt_data = torch.randint(1, tgt_vocab_size, (batch_size, max_seq_length))  # (batch_size, seq_length): (4, 197)

    model = Transformer(src_vocab_size, tgt_vocab_size, d_model, num_heads, num_layers, d_ff, max_seq_length, dropout)
    criterion = nn.CrossEntropyLoss(ignore_index=0)
    optimizer = optim.Adam(model.parameters(), lr=0.0001, betas=(0.9, 0.98), eps=1e-9)

    model.train()

    for epoch in range(1):
        optimizer.zero_grad()

        # src_data: (4, 197)
        # tgt_data[:, :-1]: (4, 196)
        output = model(src_data, tgt_data[:, :-1])  # (4, 196, 5000)

        # print(output.shape)

        print(output.contiguous().view(-1, tgt_vocab_size).shape)
        print(output.contiguous().view(-1, tgt_vocab_size))
        print(tgt_data[:, 1:].contiguous().view(-1).shape)
        print(tgt_data[:, 1:].contiguous().view(-1))
        
        loss = criterion(output.contiguous().view(-1, tgt_vocab_size), tgt_data[:, 1:].contiguous().view(-1))
        loss.backward()
        optimizer.step()
        print(f"Epoch: {epoch+1}, Loss: {loss.item()}")
