import os
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
from torchvision import transforms
from PIL import Image
import matplotlib.pyplot as plt
from collections import Counter


import sys
import os
sys.path.append(os.path.abspath('../deployment_app'))

from generator import Generator

# =====================
#  Build Vocabulary
# =====================
def build_vocab(csv_path, min_freq=2, max_size=5000):
    data = pd.read_csv(csv_path)
    counter = Counter()
    for caption in data.iloc[:, 1]:
        tokens = caption.lower().split()
        counter.update(tokens)

    vocab = {"<pad>": 0, "<unk>": 1}
    idx = 2
    for word, freq in counter.items():
        if freq >= min_freq and idx < max_size:
            vocab[word] = idx
            idx += 1
    return vocab


# =====================
#  Dataset
# =====================
class TextImageDataset(Dataset):
    def _init_(self, img_dir, csv_path, vocab, transform=None, max_len=15):
        self.img_dir = img_dir
        self.data = pd.read_csv(csv_path)
        self.vocab = vocab
        self.transform = transform
        self.max_len = max_len

    def _len_(self):
        return len(self.data)

    def _getitem_(self, idx):
        img_name = str(self.data.iloc[idx, 0]).zfill(6) + ".jpg"
        caption = self.data.iloc[idx, 1]
        img_path = os.path.join(self.img_dir, img_name)

        image = Image.open(img_path).convert("RGB")
        if self.transform:
            image = self.transform(image)

        tokens = caption.lower().split()[:self.max_len]
        token_ids = [self.vocab.get(w, self.vocab["<unk>"]) for w in tokens]
        token_tensor = torch.tensor(token_ids, dtype=torch.long)

        return image, token_tensor


def collate_fn(batch):
    images, captions = zip(*batch)
    images = torch.stack(images, dim=0)
    captions_padded = pad_sequence(captions, batch_first=True, padding_value=0)
    return images, captions_padded


# =====================
#  Discriminator
# =====================
class Discriminator(nn.Module):
    def _init_(self, img_channels=3, embed_dim=128, vocab_size=5000, features=64):
        super()._init_()
        self.embed = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.text_fc = nn.Linear(embed_dim, 256)

        self.cnn = nn.Sequential(
            nn.Conv2d(img_channels, features, 4, 2, 1),   # 64→32
            nn.LeakyReLU(0.2),
            nn.Conv2d(features, features*2, 4, 2, 1),     # 32→16
            nn.BatchNorm2d(features*2),
            nn.LeakyReLU(0.2),
            nn.Conv2d(features*2, features*4, 4, 2, 1),   # 16→8
            nn.BatchNorm2d(features*4),
            nn.LeakyReLU(0.2),
            nn.Conv2d(features*4, features*8, 4, 2, 1),   # 8→4
            nn.BatchNorm2d(features*8),
            nn.LeakyReLU(0.2),
        )

        self.img_fc = nn.Linear(features*8*4*4, 256)
        self.fc = nn.Linear(512, 1)

    def forward(self, img, text):
        img_feat = self.cnn(img).view(img.size(0), -1)
        img_feat = self.img_fc(img_feat)

        text_emb = self.embed(text).mean(dim=1)
        text_feat = self.text_fc(text_emb)

        combined = torch.cat([img_feat, text_feat], dim=1)
        return torch.sigmoid(self.fc(combined))


# =====================
#  Training Prep
# =====================
device = "cuda" if torch.cuda.is_available() else "cpu"
z_dim = 256
csv_path = r"C:\Users\SATHVIKA\OneDrive\GAN_PROJECT_FILES\Projects\DFGAN_project\data\celeba\preprocessed_text_500.csv"
img_dir = r"C:\Users\SATHVIKA\OneDrive\GAN_PROJECT_FILES\Projects\DFGAN_project\data\celeba\preprocessed_images"

# Build vocab
vocab = build_vocab(csv_path, min_freq=2, max_size=5000)

transform = transforms.Compose([
    transforms.Resize((64,64)),
    transforms.ToTensor(),
    transforms.Normalize([0.5]*3, [0.5]*3),
])

dataset = TextImageDataset(img_dir, csv_path, vocab, transform=transform)
dataloader = DataLoader(dataset, batch_size=32, shuffle=True, collate_fn=collate_fn)

gen = Generator(z_dim=z_dim, vocab_size=len(vocab)).to(device)
disc = Discriminator(vocab_size=len(vocab)).to(device)

criterion = nn.BCELoss()
opt_g = torch.optim.Adam(gen.parameters(), lr=0.0002, betas=(0.5,0.999))
opt_d = torch.optim.Adam(disc.parameters(), lr=0.0002, betas=(0.5,0.999))


# =====================
#  Training Loop
# =====================
epochs = 10
for epoch in range(epochs):
    for real, captions in dataloader:
        real, captions = real.to(device), captions.to(device)
        batch_size = real.size(0)

        valid = torch.ones(batch_size, 1, device=device)
        fake = torch.zeros(batch_size, 1, device=device)

        # Train Discriminator
        z = torch.randn(batch_size, z_dim, device=device)
        fake_imgs = gen(z, captions)

        real_loss = criterion(disc(real, captions), valid)
        fake_loss = criterion(disc(fake_imgs.detach(), captions), fake)
        d_loss = (real_loss + fake_loss) / 2

        opt_d.zero_grad()
        d_loss.backward()
        opt_d.step()

        # Train Generator
        z = torch.randn(batch_size, z_dim, device=device)
        fake_imgs = gen(z, captions)
        g_loss = criterion(disc(fake_imgs, captions), valid)

        opt_g.zero_grad()
        g_loss.backward()
        opt_g.step()

    print(f"Epoch [{epoch+1}/{epochs}] D_loss: {d_loss.item():.4f} G_loss: {g_loss.item():.4f}")

# =====================
#  Save the Generator Model
# =====================
torch.save(gen.state_dict(), "generator.pth")
print("✅ Model saved as generator.pth")