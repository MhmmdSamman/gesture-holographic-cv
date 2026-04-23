# 🖐 Gesture Holographic Interface

<div align="center">

![Python](https://img.shields.io/badge/Python-3.10+-3776AB?style=for-the-badge&logo=python&logoColor=white)
![OpenCV](https://img.shields.io/badge/OpenCV-5C3EE8?style=for-the-badge&logo=opencv&logoColor=white)
![MediaPipe](https://img.shields.io/badge/MediaPipe-0097A7?style=for-the-badge&logo=google&logoColor=white)

**Sistem interaksi gestur tangan realtime berbasis MediaPipe + OpenCV. Render kubus 3D holografik yang dikontrol murni menggunakan gestur tangan — tanpa keyboard, tanpa mouse.**

[Demo](#demo) · [Instalasi](#instalasi) · [Kontrol Gestur](#kontrol-gestur) · [Pengembangan](#pengembangan)

</div>

---

## ✨ Fitur Utama

- 🖐 **Deteksi tangan realtime** menggunakan MediaPipe (21 landmark)
- 🎲 **Render kubus 3D** dengan perspektif proyeksi + depth sorting
- 💡 **Diffuse lighting** — kecerahan muka kubus berdasarkan normal vektor
- ✨ **Particle effect** saat reset & ganti mode
- 🎨 **6 mode interaksi** — semuanya via gestur
- 📸 **Screenshot** dengan flash effect (tekan `S`)
- 🔄 **Auto-rotate** mode

---

## 🗂 Struktur Proyek

```
jarvis-holographic-cv/
│
├── jarvis_advanced.py      # Versi lengkap — 6 mode, particles, lighting
├── jarvis_cube.py          # Versi dasar — rotate, zoom, reset
│
├── steps/                  # Tutorial step-by-step
│   ├── step1_deteksi_tangan.py    # Deteksi & visualisasi landmark
│   ├── step2_menggambar_2d.py     # Air drawing di atas kamera
│   └── step3_warna_tebal.py       # Drawing + toolbar warna & ketebalan
│
├── docs/
│   └── gesture_guide.md    # Panduan lengkap semua gestur
│
├── assets/
│   └── demo.gif
│
├── requirements.txt
└── .gitignore
```

---

## ⚙️ Instalasi

```bash
# Clone repo
git clone https://github.com/MuhammadSamman/jarvis-holographic-cv.git
cd jarvis-holographic-cv

# Buat virtual environment
python -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Jalankan versi lengkap
python jarvis_advanced.py

# Atau mulai dari step 1 (tutorial)
python steps/step1_deteksi_tangan.py
```

---

## 📦 Requirements

```
opencv-python>=4.8.0
mediapipe>=0.10.0
numpy>=1.24.0
```

---

## 🖐 Kontrol Gestur

### `jarvis_advanced.py` — 6 Mode

| Gestur | Aksi |
|--------|------|
| 👌 **Pinch** (telunjuk + jempol) | Ganti mode (ROTATE → DRAW → COLOR → EXPLODE → WIRE → AXES) |
| ✊ **Mengepal** | Reset semua (posisi, skala, warna) |
| 👍 **Jempol** | Toggle auto-rotate ON/OFF |
| 🤲 **2 tangan** | Zoom in/out + atur brightness |

### Per Mode

| Mode | Gestur | Aksi |
|------|--------|------|
| **ROTATE** | ☝ Telunjuk | Rotate X/Y |
| **ROTATE** | ✌ Telunjuk + Tengah | Rotate Z (roll) |
| **DRAW** | ☝ Telunjuk | Gambar stroke di atas scene |
| **DRAW** | ✌ Telunjuk + Tengah | Angkat pena |
| **COLOR** | ☝ Telunjuk | Pilih muka kubus → pilih warna |
| **EXPLODE** | ☝ Telunjuk (gerak atas/bawah) | Kontrol level explode |
| **WIRE** | ☝ Telunjuk | Toggle wireframe/solid |
| **AXES** | ☝ Telunjuk | Toggle sumbu XYZ & grid |

### Keyboard Shortcut

| Key | Aksi |
|-----|------|
| `Q` | Keluar |
| `S` | Screenshot |
| `C` | Hapus semua stroke & custom warna |
| `R` | Reset posisi |

---

## 📐 Teknis: 3D Projection

Sistem menggunakan **perspektif proyeksi** sederhana:

```python
x2d = CX + (x3d * scale) / (z + depth_offset)
y2d = CY + (y3d * scale) / (z + depth_offset)
```

Urutan render menggunakan **Painter's Algorithm** — face diurutkan berdasarkan rata-rata Z sebelum digambar.

---

## 🔢 Pengembangan Bertahap

| File | Fitur yang Dipelajari |
|------|-----------------------|
| `step1_deteksi_tangan.py` | MediaPipe setup, landmark extraction |
| `step2_menggambar_2d.py` | Canvas overlay, gesture state machine |
| `step3_warna_tebal.py` | UI toolbar, color picking via gestur |
| `jarvis_cube.py` | 3D projection, rotation matrix, zoom |
| `jarvis_advanced.py` | Mode system, particles, lighting, full polish |

---

## 📄 Lisensi

MIT License — lihat [LICENSE](LICENSE)

---

<div align="center">
Made with ❤️ by <a href="https://github.com/MuhammadSamman">Muhammad Samman</a>
</div>
