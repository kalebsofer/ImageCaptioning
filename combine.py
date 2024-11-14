import torch
import torchvision
import random
import PIL


class Combine(torch.utils.data.Dataset):
    def init(self):
        super().init()
        self.tf = torchvision.transforms.ToTensor()
        self.ds = torchvision.datasets.MNIST(root=".", download=True)
        self.ln = len(self.ds)

    def len(self):
        return len(self.ds)

    def getitem(self, idx):
        idx = random.sample(range(self.ln), 4)
        store = []
        label = []

        for i in idx:
            x, y = self.ds[i]
            store.append(x)
            label.append(y)

        img = PIL.Image.new("L", (56, 56))
        img.paste(store[0], (0, 0))
        img.paste(store[1], (28, 0))
        img.paste(store[2], (0, 28))
        img.paste(store[3], (28, 28))
        return img, label


ds = Combine()
img, label = ds[0]
print(label)
img.show()
