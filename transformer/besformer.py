#
#
import torch


#
#
class Attention(torch.nn.Module):
    def __init__(self, is_causal=False, d=32):
        super().__init__()
        self.is_causal = is_causal
        self.W_Q = torch.nn.Linear(d, d)
        self.W_K = torch.nn.Linear(d, d)
        self.W_V = torch.nn.Linear(d, d)
        self.prj = torch.nn.Linear(d, d)
        self.sft = torch.nn.Softmax(dim=-1)

    def forward(self, q, k, v):
        Q = self.W_Q(q)
        K = self.W_K(k)
        V = self.W_V(v)

        # Ensure K is transposed correctly
        K = K.transpose(-2, -1)  # Transpose the last two dimensions

        # Check shapes
        # print("Attention.forward.01.Q.shape", Q.shape)
        # print("Attention.forward.01.K.shape", K.shape)
        # print("Attention.forward.01.V.shape", V.shape)

        # Perform scaled dot-product attention
        s = (Q @ K) / (Q.size(-1) ** 0.5)

        if self.is_causal:
            s = s + torch.tril(torch.full_like(s, float("-inf")), diagonal=-1)

        o = self.sft(s) @ V
        o = self.prj(o)
        return o


#
#
class Forward(torch.nn.Module):
    def __init__(self, d=32):
        super().__init__()
        self.ins = torch.nn.Linear(d, d)
        self.rlu = torch.nn.ReLU()
        self.out = torch.nn.Linear(d, d)
        self.drp = torch.nn.Dropout(0.1)

    def forward(self, x):
        x = self.ins(x)
        x = self.rlu(x)
        x = self.drp(x)
        x = self.out(x)
        return x


#
#
class EncBlock(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.att = Attention(d=64)
        self.nr1 = torch.nn.LayerNorm(64)
        self.nr2 = torch.nn.LayerNorm(64)
        self.drp = torch.nn.Dropout(0.1)
        self.ffw = Forward(d=64)

    def forward(self, x):
        # print("EncBlock.forward.00.x.shape", x.shape)
        a = self.att(x, x, x)
        # print("EncBlock.forward.01.a.shape", a.shape)
        x = self.nr1(a + x)
        o = self.ffw(x)
        o = self.nr2(o + x)
        # print("EncBlock.forward.02.o.shape", o.shape)
        return o


#
#
class Encoder(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.emb = torch.nn.Linear(3 * 256 * 256, 64)
        self.pos = torch.nn.Embedding(16, 64)
        self.drp = torch.nn.Dropout(0.1)
        layers = [EncBlock() for _ in range(12)]
        self.bks = torch.nn.ModuleList(layers)
        self.prj = torch.nn.Linear(64, 32)

    def forward(self, x):
        batch_size = x.size(0)
        x = x.view(batch_size, -1)
        x = self.emb(x)

        # Calculate the correct sequence length
        seq_length = x.size(1) // 64  # Assuming 64 is the embedding dimension

        # Adjust the shape of x to match the positional embeddings
        x = x.view(batch_size, seq_length, 64)

        n = (
            torch.arange(seq_length, device=x.device)
            .unsqueeze(0)
            .expand(batch_size, -1)
        )
        x = x + self.pos(n)

        x = self.drp(x)
        for b in self.bks:
            x = b(x)
        x = self.prj(x)
        return x


#
#
class DecBlock(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.c_a = Attention(is_causal=True)
        self.x_a = Attention()
        self.nr1 = torch.nn.LayerNorm(32)
        self.nr2 = torch.nn.LayerNorm(32)
        self.nr3 = torch.nn.LayerNorm(32)
        self.drp = torch.nn.Dropout(0.1)
        self.ffw = Forward()

    def forward(self, e, x):
        # print("DecBlock.forward.00.e.shape", e.shape)
        # print("DecBlock.forward.00.x.shape", x.shape)
        c = self.c_a(x, x, x)
        # print("DecBlock.forward.01.c.shape", c.shape)
        c = self.nr1(c + x)
        a = self.x_a(c, e, e)
        a = self.nr2(a + c)
        o = self.ffw(x)
        o = self.nr3(o + x)
        # print("DecBlock.forward.02.o.shape", o.shape)
        return o


#
#
class Decoder(torch.nn.Module):
    def __init__(self, vocab_size):
        super().__init__()
        self.emb = torch.nn.Embedding(vocab_size, 32)
        self.pos = torch.nn.Embedding(100, 32)
        self.drp = torch.nn.Dropout(0.1)
        layers = [DecBlock() for _ in range(6)]
        self.bks = torch.nn.ModuleList(layers)
        self.prj = torch.nn.Linear(32, vocab_size)

    def forward(self, e, x):
        # Debug print statements in Decoder.forward()
        # print("Decoder.forward.00.x.shape:", x.shape)
        # print("Decoder.forward.00.x max index", x.max().item())
        # print("Decoder.forward.00.x dtype:", x.dtype)
        # print("Decoder.forward.00.x device:", x.device)

        # Positional Embeddings
        # print("Decoder.forward.01.emb.num_embeddings:", self.emb.num_embeddings)
        # print("Decoder.forward.01.pos.num_embeddings:", self.pos.num_embeddings)

        n = torch.arange(x.shape[-1], device=x.device)
        if n.shape[0] > self.pos.num_embeddings:
            raise ValueError(
                f"Positional embedding length {self.pos.num_embeddings} is not enough for sequence length {n.shape[0]}"
            )

        x = self.emb(x) + self.pos(n)
        # print("Decoder.forward.02.x.shape", x.shape)
        x = self.drp(x)

        # Ensure batch size of encoder output and decoder input match
        if e.shape[0] != x.shape[0]:
            e = e.repeat(x.shape[0] // e.shape[0], 1, 1)  # Repeat to match batch size
            # print(f"Encoder output repeated to match batch size: {e.shape}")

        for b in self.bks:
            x = b(e, x)
        x = self.prj(x)
        return x


#
#
class TransformerB(torch.nn.Module):
    def __init__(self, vocab_size):
        super().__init__()
        self.enc = Encoder()
        self.dec = Decoder(vocab_size)

    def forward(self, i, x):
        # print("Transformer.forward.00.i.shape", i.shape)
        # print("Transformer.forward.00.x.shape", x.shape)
        e = self.enc(i)
        # print("Transformer.forward.01.e.shape", e.shape)
        return self.dec(e, x)


#
#
if __name__ == "__main__":
    vocab_size = 16000
    t = TransformerB(vocab_size)
    params = sum(p.numel() for p in t.parameters())
    print("Parameters:", params)
    print("Architecture:", t)
