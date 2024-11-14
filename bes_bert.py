import torch


torch.manual_seed(31)
torch.set_printoptions(linewidth=200)


#
#
class Layer(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.proj = torch.nn.Linear(4, 4)
        self.soft = torch.nn.Softmax(dim=-1)
        self.relu = torch.nn.ReLU()
        self.forw = torch.nn.Linear(4, 4)

    def forward(self, x):
        x = self.proj(x)
        attn = x @ x.T
        attn = self.soft(attn)
        x = attn @ x
        x = self.relu(x)
        x = self.forw(x)
        return x


#
#
class Bert(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.emb = torch.nn.Embedding(9, 4)
        self.lys = torch.nn.ModuleList([Layer() for _ in range(1)])
        self.prj = torch.nn.Linear(4, 9)
        self.sft = torch.nn.Softmax(dim=-1)

    def forward(self, x):
        x = self.emb(x)
        x = x.squeeze(0)
        for lyr in self.lys:
            x = lyr(x)
        x = self.prj(x)
        x = self.sft(x)
        return x


#
#
if __name__ == "__main__":
    model = Bert()
    print(model)
    seq = torch.randint(0, 9, (1, 5))
    print(seq.shape)
    print(seq.squeeze(0).tolist())
    out = model(seq)
    print(out)
