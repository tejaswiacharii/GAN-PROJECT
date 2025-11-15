import streamlit as st
import torch
import torch.nn as nn
import numpy as np
from PIL import Image, ImageEnhance
from io import BytesIO
from generator import Generator

# ===========================
#   Streamlit Page Config
# ===========================
st.set_page_config(
    page_title="Text-to-Image GAN Generator",
    page_icon="🎨",
    layout="centered"
)

device = "cpu"
z_dim = 256
vocab_size = 64  # should match your trained vocab size

# ===========================
#   Load Model
# ===========================
@st.cache_resource
def load_model():
    model = Generator(z_dim=z_dim, vocab_size=vocab_size)
    model.load_state_dict(torch.load("generator.pth", map_location=device))
    model.eval()
    return model

model = load_model()

# ===========================
#   Streamlit UI
# ===========================
st.title("🎨 Text-to-Image Generator (GAN Demo)")
st.write("Generate realistic face images from text descriptions using your trained GAN model!")

st.divider()

caption = st.text_input("📝 Enter a text description:", placeholder="Example: a smiling woman with brown hair")

# Image customization controls
img_size = st.slider("🖼️ Output image size (in pixels)", 64, 512, 256, step=32)
frame_thickness = st.slider("🟦 Frame border thickness", 0, 20, 5, step=1)
frame_color = st.color_picker("🎨 Choose frame color", "#000000")

# Generate button
generate = st.button("🚀 Generate Image")

if generate and caption:
    with st.spinner("Generating image... please wait ⏳"):

        # ---------------------------
        #   Preprocess Text
        # ---------------------------
        tokens = caption.lower().split()[:15]
        vocab = {"<pad>": 0, "<unk>": 1}  # placeholder vocab
        token_ids = [vocab.get(w, 1) for w in tokens]
        token_tensor = torch.tensor([token_ids], dtype=torch.long)

        # ---------------------------
        #   Generate Image
        # ---------------------------
        z = torch.randn(1, z_dim)
        with torch.no_grad():
            fake_img = model(z, token_tensor)

        # Convert GAN output to image
        img = (fake_img[0].cpu().permute(1, 2, 0).numpy() + 1) / 2.0
        img = np.clip(img, 0, 1)
        img = (img * 255).astype(np.uint8)

        # Resize with LANCZOS for high-quality scaling
        image_pil = Image.fromarray(img).resize((img_size, img_size), Image.Resampling.LANCZOS)

        # Enhance clarity
        image_pil = ImageEnhance.Sharpness(image_pil).enhance(1.8)
        image_pil = ImageEnhance.Contrast(image_pil).enhance(1.2)

        # Convert to bytes for display/download
        buf = BytesIO()
        image_pil.save(buf, format="PNG")
        byte_im = buf.getvalue()

        # Display with frame styling
        st.markdown(
            f"""
            <div style="
                display: flex;
                justify-content: center;
                align-items: center;
                padding: {frame_thickness}px;
                border-radius: 12px;
                border: {frame_thickness}px solid {frame_color};
                background-color: #f8f9fa;
                width: fit-content;
                margin: auto;
            ">
                <img src="data:image/png;base64,{buf.getvalue().hex()}" width="{img_size}px">
            </div>
            """,
            unsafe_allow_html=True
        )

        # Display and download options
        st.image(image_pil, caption="🖼️ Sharpened & Enhanced Image", output_format="PNG", width=img_size)
        st.download_button("💾 Download Image", data=byte_im, file_name="generated_image.png", mime="image/png")

st.info("💡 Tip: Use short descriptive captions (e.g., 'a man with glasses', 'a smiling woman with curly hair').")
