import streamlit as st
import torch
import torch.nn as nn
import numpy as np
from PIL import Image

# =========================
# Konfigurasi Halaman
# =========================
st.set_page_config(
    page_title="SRGAN + Transfer Learning",
    page_icon="🖼️",
    layout="centered"
)

st.title("🖼️ SRGAN dengan Transfer Learning")
st.write(
    "Website sederhana untuk meningkatkan resolusi citra menggunakan **SRGAN** "
    "dengan model hasil *transfer learning*."
)

# =========================
# Model SRGAN (SESUAI CHECKPOINT)
# =========================
class ResidualBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(channels, channels, 3, 1, 1),
            nn.BatchNorm2d(channels),
            nn.PReLU(),
            nn.Conv2d(channels, channels, 3, 1, 1),
            nn.BatchNorm2d(channels),
        )

    def forward(self, x):
        return x + self.block(x)


class GeneratorSRGAN(nn.Module):
    def __init__(self, num_res_blocks=8):
        super().__init__()

        # block1
        self.block1 = nn.Sequential(
            nn.Conv2d(3, 64, 9, 1, 4),
            nn.PReLU()
        )

        # residual blocks
        self.res_blocks = nn.Sequential(
            *[ResidualBlock(64) for _ in range(num_res_blocks)]
        )

        # block2
        self.block2 = nn.Sequential(
            nn.Conv2d(64, 64, 3, 1, 1),
            nn.BatchNorm2d(64)
        )

        # upsampling (4×)
        self.upsampling = nn.Sequential(
            nn.Conv2d(64, 256, 3, 1, 1),
            nn.PixelShuffle(2),
            nn.PReLU(),
            nn.Conv2d(64, 256, 3, 1, 1),
            nn.PixelShuffle(2),
            nn.PReLU(),
        )

        # block3
        self.block3 = nn.Conv2d(64, 3, 9, 1, 4)

    def forward(self, x):
        x1 = self.block1(x)
        x2 = self.res_blocks(x1)
        x3 = self.block2(x2)
        x = x1 + x3
        x = self.upsampling(x)
        return self.block3(x)

# =========================
# Load Model
# =========================
@st.cache_resource
def load_model():
    model = GeneratorSRGAN(num_res_blocks=8)
    state_dict = torch.load("generator.pth", map_location="cpu")
    model.load_state_dict(state_dict)
    model.eval()
    return model

model = load_model()

# =========================
# Upload Citra
# =========================
st.subheader("📤 Upload Citra Resolusi Rendah")
uploaded_file = st.file_uploader(
    "Pilih gambar (PNG / JPG)",
    type=["png", "jpg", "jpeg"]
)

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Citra Resolusi Rendah", use_container_width=True)

    # Preprocessing
    img_np = np.array(image).astype(np.float32) / 255.0
    img_tensor = torch.from_numpy(img_np).permute(2, 0, 1).unsqueeze(0)

    if st.button("🚀 Proses SRGAN"):
        with st.spinner("Sedang meningkatkan resolusi..."):
            with torch.no_grad():
                sr = model(img_tensor)
                sr = sr.clamp(0, 1)

            sr_img = sr.squeeze(0).permute(1, 2, 0).numpy()
            sr_img = (sr_img * 255).astype(np.uint8)
            sr_image = Image.fromarray(sr_img)

        st.subheader("✨ Hasil Super-Resolution (4×)")
        st.image(sr_image, use_container_width=True)

        st.download_button(
            "⬇️ Download Hasil",
            data=sr_image.tobytes(),
            file_name="hasil_srgan.png",
            mime="image/png"
        )

# =========================
# Sidebar
# =========================
st.sidebar.title("ℹ️ Informasi Model")
st.sidebar.markdown("""
- **Model** : SRGAN Generator  
- **Residual Block** : 8  
- **Channel** : 64  
- **Upsampling** : PixelShuffle (4×)  
- **Pendekatan** : Transfer Learning  
""")
