#!/usr/bin/env python
# coding: utf-8

# In[43]:


import os
from PIL import Image
from torchvision import transforms
from tqdm import tqdm

# ------------------------
# Paths
# ------------------------
RAW_DATA_DIR = r"C:\Users\SATHVIKA\OneDrive\GAN_PROJECT_FILES\Projects\DFGAN_project\data\celeba\img_align_celeba\img_align_celeba"
SAVE_DIR = r"C:\Users\SATHVIKA\OneDrive\GAN_PROJECT_FILES\Projects\DFGAN_project\data\celeba\preprocessed_images"

os.makedirs(SAVE_DIR, exist_ok=True)

# ------------------------
# Transformations (CenterCrop + Resize)
# ------------------------
image_size = 128 # change to 128 if you want higher resolution
transform = transforms.Compose([
    transforms.CenterCrop(178),   # crop to square
    transforms.Resize(image_size),
    transforms.ToTensor(),
    transforms.Normalize([0.5]*3, [0.5]*3)   # [-1,1]
])

# ------------------------
# Process first 500 images
# ------------------------
img_names = sorted(os.listdir(RAW_DATA_DIR))[:10000]

count = 0
for img_name in tqdm(img_names):
    img_path = os.path.join(RAW_DATA_DIR, img_name)

    try:
        img = Image.open(img_path).convert("RGB")
        tensor_img = transform(img)

        # Denormalize for saving [0,1]
        unnormalize = transforms.Compose([
            transforms.Normalize([-1, -1, -1], [2, 2, 2]),
            transforms.Lambda(lambda x: x.clamp(0,1)),
            transforms.ToPILImage()
        ])

        processed_img = unnormalize(tensor_img)
        processed_img.save(os.path.join(SAVE_DIR, img_name))

        count += 1
    except Exception as e:
        print(f"Error processing {img_name}: {e}")

print(f"✅ Preprocessing complete! {count} images saved in {SAVE_DIR}")


# In[28]:


import matplotlib.pyplot as plt

# Pick first 10 images
raw_files = sorted([f for f in os.listdir(RAW_DATA_DIR) if f.endswith((".jpg", ".png"))])[:10]
processed_files = sorted([f for f in os.listdir(SAVE_DIR) if f.endswith((".jpg", ".png"))])[:10]

plt.figure(figsize=(20, 5))

for i, (raw_name, proc_name) in enumerate(zip(raw_files, processed_files)):
    # Load raw and processed images
    raw_img = Image.open(os.path.join(RAW_DATA_DIR, raw_name)).convert("RGB")
    proc_img = Image.open(os.path.join(SAVE_DIR, proc_name)).convert("RGB")

    # Show original
    plt.subplot(2, 10, i+1)
    plt.imshow(raw_img)
    plt.title("Original")
    plt.axis("off")

    # Show preprocessed
    plt.subplot(2, 10, i+11)
    plt.imshow(proc_img)
    plt.title("Preprocessed")
    plt.axis("off")

plt.suptitle("Original vs Preprocessed Images", fontsize=16)
plt.show()


# In[44]:


import os
import pandas as pd

# ------------------------
# Paths
# ------------------------
CAPTION_FILE = r"C:\Users\SATHVIKA\OneDrive\GAN_PROJECT_FILES\Projects\DFGAN_project\data\celeba\celeba_text_descriptions.csv"
SAVE_CAPTION_FILE = r"C:\Users\SATHVIKA\OneDrive\GAN_PROJECT_FILES\Projects\DFGAN_project\data\celeba\preprocessed_text.csv"

# ------------------------
# Load CSV
# ------------------------
df = pd.read_csv(CAPTION_FILE)

# ------------------------
# Fix columns
# ------------------------
# Rename columns
df = df.rename(columns={"Unnamed: 0": "image_id", "description": "caption"})

# Ensure image_id is string before replacing
df["image_id"] = df["image_id"].astype(str)

# Remove any '.jpg' extension if present
df["image_id"] = df["image_id"].str.replace(".jpg", "", regex=False)

# Zero-pad image IDs to 6 digits to match filenames like '000001.jpg'
df["image_id"] = df["image_id"].apply(lambda x: f"{int(x):06d}")

# ------------------------
# Save back CSV
# ------------------------
df.to_csv(SAVE_CAPTION_FILE, index=False)

# ------------------------
# Print confirmation
# ------------------------
print("✅ Fixed CSV saved at:", SAVE_CAPTION_FILE)
print("Columns:", df.columns.tolist())
print(df.head())


# In[45]:


import pandas as pd
import os

# ------------------------
# Paths
# ------------------------
RAW_CSV = r"C:\Users\SATHVIKA\OneDrive\GAN_PROJECT_FILES\Projects\DFGAN_project\data\celeba\celeba_text_descriptions.csv"
OUTPUT_CSV = r"C:\Users\SATHVIKA\OneDrive\GAN_PROJECT_FILES\Projects\DFGAN_project\data\celeba\preprocessed_text_500.csv"

# ------------------------
# Load raw captions CSV
# ------------------------
df = pd.read_csv(RAW_CSV)

# Rename columns if necessary
df = df.rename(columns={"Unnamed: 0": "image_id", "description": "caption"})

# ------------------------
# Clean image_id column
# ------------------------
df["image_id"] = df["image_id"].astype(str)

# Remove ".jpg" if present
df["image_id"] = df["image_id"].str.replace(".jpg", "", regex=False)

# Convert to int → str → zfill(6)
df["image_id"] = df["image_id"].astype(int).astype(str).str.zfill(6)

# ------------------------
# Keep only first 500 rows
# ------------------------
df_500 = df.head(10000).copy()

# ------------------------
# Save preprocessed CSV
# -----------------------
df_500.to_csv(OUTPUT_CSV, index=False)

print("✅ Preprocessed captions CSV saved at:", OUTPUT_CSV)
print("Total rows:", len(df_500))
print(df_500.head())


# In[46]:


import os
import pandas as pd
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

# ------------------------
# Paths
# ------------------------
IMAGE_DIR = r"C:\Users\SATHVIKA\OneDrive\GAN_PROJECT_FILES\Projects\DFGAN_project\data\celeba\preprocessed_images"
CAPTION_FILE = r"C:\Users\SATHVIKA\OneDrive\GAN_PROJECT_FILES\Projects\DFGAN_project\data\celeba\preprocessed_text_500.csv"

# ------------------------
# Load captions CSV
# ------------------------
df = pd.read_csv(CAPTION_FILE)

# Only keep first 500 (to match your preprocessed images)
df = df.iloc[:10000]

# ------------------------
# Define Dataset
# ------------------------
class CelebADataset(Dataset):
    def __init__(self, dataframe, image_dir, transform=None):
        self.dataframe = dataframe
        self.image_dir = image_dir
        self.transform = transform

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, idx):
        row = self.dataframe.iloc[idx]
        image_id = str(row["image_id"]).zfill(6) + ".jpg"   # match filenames
        caption = row["caption"]

        # Load image
        img_path = os.path.join(self.image_dir, image_id)
        image = Image.open(img_path).convert("RGB")

        if self.transform:
            image = self.transform(image)

        return image, caption

# ------------------------
# Image Transform
# ------------------------
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize([0.5]*3, [0.5]*3)   # normalize to [-1,1]
])

# ------------------------
# Create Dataset & Dataloader
# ------------------------
dataset = CelebADataset(df, IMAGE_DIR, transform=transform)
dataloader = DataLoader(dataset, batch_size=8, shuffle=True)

# ------------------------
# Test one batch
# ------------------------
images, captions = next(iter(dataloader))
print("✅ Dataset ready!")
print("Batch images shape:", images.shape)
print("Batch captions:", captions[:3])


# In[47]:


import matplotlib.pyplot as plt

# ------------------------
# Visualize one batch
# ------------------------
images, captions = next(iter(dataloader))

# Undo normalization for visualization
unnormalize = transforms.Compose([
    transforms.Normalize([-1, -1, -1], [2, 2, 2]),  # convert back to [0,1]
    transforms.Lambda(lambda x: x.clamp(0,1))
])

plt.figure(figsize=(16, 8))

for i in range(len(images)):
    img = unnormalize(images[i])  # denormalize
    img = transforms.ToPILImage()(img)

    plt.subplot(2, 4, i+1)
    plt.imshow(img)
    plt.title(captions[i][:40] + "..." if len(captions[i]) > 40 else captions[i], fontsize=8)
    plt.axis("off")

plt.suptitle("Sample Preprocessed Images with Captions", fontsize=14)
plt.show()


# In[48]:


import os
import pandas as pd

# Paths
IMG_DIR = r"C:\Users\SATHVIKA\OneDrive\GAN_PROJECT_FILES\Projects\DFGAN_project\data\celeba\preprocessed_images"
CSV_PATH = r"C:\Users\SATHVIKA\OneDrive\GAN_PROJECT_FILES\Projects\DFGAN_project\data\celeba\preprocessed_text_500.csv"

# Load CSV
df = pd.read_csv(CSV_PATH)

# Ensure image_id is padded to 6 digits
df["image_id"] = df["image_id"].astype(str).str.zfill(6)

# List of available image files (remove ".jpg")
img_files = set([os.path.splitext(f)[0] for f in os.listdir(IMG_DIR) if f.endswith(".jpg")])

# Check which IDs are missing
missing = [img_id for img_id in df["image_id"] if img_id not in img_files]

print("Total captions:", len(df))
print("Total images:", len(img_files))
print("Missing matches:", len(missing))
if missing:
    print("Example missing IDs:", missing[:10])
else:
    print("✅ All captions match images perfectly!")


# ##### 500 Images

# In[15]:


#  User Input Function
# =====================
def generate_from_text(gen, vocab, device, z_dim=100, num_samples=1):
    caption = input("Enter your caption: ").lower()
    tokens = caption.split()[:15]
    token_ids = [vocab.get(w, vocab["<unk>"]) for w in tokens]
    token_tensor = torch.tensor([token_ids], dtype=torch.long).to(device)

    z = torch.randn(num_samples, z_dim).to(device)

    gen.eval()
    with torch.no_grad():
        fake_img = gen(z, token_tensor)
    gen.train()

    img = (fake_img[0].cpu().permute(1,2,0).numpy() + 1) / 2.0
    plt.imshow(img)
    plt.axis("off")
    plt.show()


# Try it
generate_from_text(gen, vocab, device)


# In[1]:


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
    def __init__(self, img_dir, csv_path, vocab, transform=None, max_len=15):
        self.img_dir = img_dir
        self.data = pd.read_csv(csv_path)
        self.vocab = vocab
        self.transform = transform
        self.max_len = max_len

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
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
#  Generator
# =====================
class Generator(nn.Module):
    def __init__(self, z_dim=256, embed_dim=128, vocab_size=5000, features=64):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.text_fc = nn.Linear(embed_dim, z_dim)

        self.net = nn.Sequential(
            nn.ConvTranspose2d(z_dim*2, features*8, 4, 1, 0),  # 1x1 → 4x4
            nn.BatchNorm2d(features*8),
            nn.ReLU(True),

            nn.ConvTranspose2d(features*8, features*4, 4, 2, 1),  # 4x4 → 8x8
            nn.BatchNorm2d(features*4),
            nn.ReLU(True),

            nn.ConvTranspose2d(features*4, features*2, 4, 2, 1),  # 8x8 → 16x16
            nn.BatchNorm2d(features*2),
            nn.ReLU(True),

            nn.ConvTranspose2d(features*2, features, 4, 2, 1),    # 16x16 → 32x32
            nn.BatchNorm2d(features),
            nn.ReLU(True),

            nn.ConvTranspose2d(features, 3, 4, 2, 1),             # 32x32 → 64x64
            nn.Tanh(),
        )

    def forward(self, z, text):
        text_emb = self.embed(text).mean(dim=1)
        text_feat = self.text_fc(text_emb)
        combined = torch.cat([z, text_feat], dim=1).unsqueeze(2).unsqueeze(3)
        return self.net(combined)


# =====================
#  Discriminator
# =====================
class Discriminator(nn.Module):
    def __init__(self, img_channels=3, embed_dim=128, vocab_size=5000, features=64):
        super().__init__()
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

# Build vocab from CSV
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
#  Training Loop (mini)
# =====================
epochs = 10
for epoch in range(epochs):
    for real, captions in dataloader:
        real, captions = real.to(device), captions.to(device)
        batch_size = real.size(0)

        valid = torch.ones(batch_size, 1, device=device)
        fake = torch.zeros(batch_size, 1, device=device)

        # ---- Train Discriminator ----
        z = torch.randn(batch_size, z_dim, device=device)
        fake_imgs = gen(z, captions)

        real_loss = criterion(disc(real, captions), valid)
        fake_loss = criterion(disc(fake_imgs.detach(), captions), fake)
        d_loss = (real_loss + fake_loss) / 2

        opt_d.zero_grad()
        d_loss.backward()
        opt_d.step()

        # ---- Train Generator ----
        z = torch.randn(batch_size, z_dim, device=device)
        fake_imgs = gen(z, captions)
        g_loss = criterion(disc(fake_imgs, captions), valid)

        opt_g.zero_grad()
        g_loss.backward()
        opt_g.step()

    print(f"Epoch [{epoch+1}/{epochs}]  D_loss: {d_loss.item():.4f}  G_loss: {g_loss.item():.4f}")


# =====================
#  User Input Function
# =====================
def generate_from_text(gen, vocab, device, z_dim=256, num_samples=1):
    caption = input("Enter your caption: ").lower()
    tokens = caption.split()[:15]
    token_ids = [vocab.get(w, vocab["<unk>"]) for w in tokens]
    token_tensor = torch.tensor([token_ids], dtype=torch.long).to(device)

    z = torch.randn(num_samples, z_dim).to(device)

    gen.eval()
    with torch.no_grad():
        fake_img = gen(z, token_tensor)
    gen.train()

    img = (fake_img[0].cpu().permute(1,2,0).numpy() + 1) / 2.0
    plt.imshow(img)
    plt.axis("off")
    plt.show()


# Try generation after training
generate_from_text(gen, vocab, device, z_dim=z_dim)


# In[ ]:


#5000 images


# In[38]:


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
    def __init__(self, img_dir, csv_path, vocab, transform=None, max_len=15):
        self.img_dir = img_dir
        self.data = pd.read_csv(csv_path)
        self.vocab = vocab
        self.transform = transform
        self.max_len = max_len

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
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
#  Generator
# =====================
class Generator(nn.Module):
    def __init__(self, z_dim=256, embed_dim=128, vocab_size=5000, features=64):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.text_fc = nn.Linear(embed_dim, z_dim)

        self.net = nn.Sequential(
            nn.ConvTranspose2d(z_dim*2, features*8, 4, 1, 0),  # 1x1 → 4x4
            nn.BatchNorm2d(features*8),
            nn.ReLU(True),

            nn.ConvTranspose2d(features*8, features*4, 4, 2, 1),  # 4x4 → 8x8
            nn.BatchNorm2d(features*4),
            nn.ReLU(True),

            nn.ConvTranspose2d(features*4, features*2, 4, 2, 1),  # 8x8 → 16x16
            nn.BatchNorm2d(features*2),
            nn.ReLU(True),

            nn.ConvTranspose2d(features*2, features, 4, 2, 1),    # 16x16 → 32x32
            nn.BatchNorm2d(features),
            nn.ReLU(True),

            nn.ConvTranspose2d(features, 3, 4, 2, 1),             # 32x32 → 64x64
            nn.Tanh(),
        )

    def forward(self, z, text):
        text_emb = self.embed(text).mean(dim=1)
        text_feat = self.text_fc(text_emb)
        combined = torch.cat([z, text_feat], dim=1).unsqueeze(2).unsqueeze(3)
        return self.net(combined)


# =====================
#  Discriminator
# =====================
class Discriminator(nn.Module):
    def __init__(self, img_channels=3, embed_dim=128, vocab_size=5000, features=64):
        super().__init__()
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

# Build vocab from CSV
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
#  Training Loop (mini)
# =====================
epochs = 10
for epoch in range(epochs):
    for real, captions in dataloader:
        real, captions = real.to(device), captions.to(device)
        batch_size = real.size(0)

        valid = torch.ones(batch_size, 1, device=device)
        fake = torch.zeros(batch_size, 1, device=device)

        # ---- Train Discriminator ----
        z = torch.randn(batch_size, z_dim, device=device)
        fake_imgs = gen(z, captions)

        real_loss = criterion(disc(real, captions), valid)
        fake_loss = criterion(disc(fake_imgs.detach(), captions), fake)
        d_loss = (real_loss + fake_loss) / 2

        opt_d.zero_grad()
        d_loss.backward()
        opt_d.step()

        # ---- Train Generator ----
        z = torch.randn(batch_size, z_dim, device=device)



        fake_imgs = gen(z, captions)
        g_loss = criterion(disc(fake_imgs, captions), valid)

        opt_g.zero_grad()
        g_loss.backward()
        opt_g.step()

    print(f"Epoch [{epoch+1}/{epochs}]  D_loss: {d_loss.item():.4f}  G_loss: {g_loss.item():.4f}")


# In[42]:


#  User Input Function
# =====================


def generate_from_text(gen, vocab, device, z_dim=256, num_samples=1):
    caption = input("Enter your caption: ").lower()
    tokens = caption.split()[:15]
    token_ids = [vocab.get(w, vocab["<unk>"]) for w in tokens]
    token_tensor = torch.tensor([token_ids], dtype=torch.long).to(device)

    z = torch.randn(num_samples, z_dim).to(device)


    gen.eval()
    with torch.no_grad():
        fake_img = gen(z, token_tensor)
    gen.train()

    img = (fake_img[0].cpu().permute(1,2,0).numpy() + 1) / 2.0
    plt.imshow(img)
    plt.axis("off")
    plt.show()


# Try it
generate_from_text(gen, vocab, device)


# #10000 Images

# In[49]:


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
    def __init__(self, img_dir, csv_path, vocab, transform=None, max_len=15):
        self.img_dir = img_dir
        self.data = pd.read_csv(csv_path)
        self.vocab = vocab
        self.transform = transform
        self.max_len = max_len

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
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
#  Generator
# =====================
class Generator(nn.Module):
    def __init__(self, z_dim=256, embed_dim=128, vocab_size=5000, features=64):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.text_fc = nn.Linear(embed_dim, z_dim)

        self.net = nn.Sequential(
            nn.ConvTranspose2d(z_dim*2, features*8, 4, 1, 0),  # 1x1 → 4x4
            nn.BatchNorm2d(features*8),
            nn.ReLU(True),

            nn.ConvTranspose2d(features*8, features*4, 4, 2, 1),  # 4x4 → 8x8
            nn.BatchNorm2d(features*4),
            nn.ReLU(True),

            nn.ConvTranspose2d(features*4, features*2, 4, 2, 1),  # 8x8 → 16x16
            nn.BatchNorm2d(features*2),
            nn.ReLU(True),

            nn.ConvTranspose2d(features*2, features, 4, 2, 1),    # 16x16 → 32x32
            nn.BatchNorm2d(features),
            nn.ReLU(True),

            nn.ConvTranspose2d(features, 3, 4, 2, 1),             # 32x32 → 64x64
            nn.Tanh(),
        )

    def forward(self, z, text):
        text_emb = self.embed(text).mean(dim=1)
        text_feat = self.text_fc(text_emb)
        combined = torch.cat([z, text_feat], dim=1).unsqueeze(2).unsqueeze(3)
        return self.net(combined)


# =====================
#  Discriminator
# =====================
class Discriminator(nn.Module):
    def __init__(self, img_channels=3, embed_dim=128, vocab_size=5000, features=64):
        super().__init__()
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

# Build vocab from CSV
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
#  Training Loop (mini)
# =====================
epochs = 10
for epoch in range(epochs):
    for real, captions in dataloader:
        real, captions = real.to(device), captions.to(device)
        batch_size = real.size(0)

        valid = torch.ones(batch_size, 1, device=device)
        fake = torch.zeros(batch_size, 1, device=device)

        # ---- Train Discriminator ----
        z = torch.randn(batch_size, z_dim, device=device)
        fake_imgs = gen(z, captions)

        real_loss = criterion(disc(real, captions), valid)
        fake_loss = criterion(disc(fake_imgs.detach(), captions), fake)
        d_loss = (real_loss + fake_loss) / 2

        opt_d.zero_grad()
        d_loss.backward()
        opt_d.step()

        # ---- Train Generator ----
        z = torch.randn(batch_size, z_dim, device=device)



        fake_imgs = gen(z, captions)
        g_loss = criterion(disc(fake_imgs, captions), valid)

        opt_g.zero_grad()
        g_loss.backward()
        opt_g.step()

    print(f"Epoch [{epoch+1}/{epochs}]  D_loss: {d_loss.item():.4f}  G_loss: {g_loss.item():.4f}")


# In[ ]:





# In[53]:


#  User Input Function
# =====================


def generate_from_text(gen, vocab, device, z_dim=256, num_samples=1):
    caption = input("Enter your caption: ").lower()
    tokens = caption.split()[:15]
    token_ids = [vocab.get(w, vocab["<unk>"]) for w in tokens]
    token_tensor = torch.tensor([token_ids], dtype=torch.long).to(device)

    z = torch.randn(num_samples, z_dim).to(device)


    gen.eval()
    with torch.no_grad():
        fake_img = gen(z, token_tensor)
    gen.train()

    img = (fake_img[0].cpu().permute(1,2,0).numpy() + 1) / 2.0
    plt.imshow(img)
    plt.axis("off")
    plt.show()


# Try it
generate_from_text(gen, vocab, device)


# In[1]:


get_ipython().system('jupyter nbconvert --to script your_notebook.ipynb')


# In[ ]:




