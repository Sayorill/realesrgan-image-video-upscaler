# AI Image & Video Upscaler (Real-ESRGAN)

Program Python untuk upscaling gambar **dan video** jadi lebih HD menggunakan
model AI Real-ESRGAN. Sudah dioptimalkan otomatis untuk GPU dengan VRAM kecil
seperti **RTX 3050 4GB**.

## 1. Instalasi

### a. Install PyTorch versi CUDA (WAJIB dilakukan lebih dulu agar GPU terpakai)

RTX 3050 mendukung CUDA 12.x. Jalankan:

```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
```

Cek apakah PyTorch sudah mendeteksi GPU:
```bash
python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0))"
```
Kalau outputnya `True RTX 3050 Laptop GPU` (atau serupa), berarti sudah siap.

### b. Install dependency lainnya

```bash
cd image-upscaler
pip install -r requirements.txt
```

### c. (Untuk fitur video) Install ffmpeg

Supaya audio video asli tetap terjaga saat diupscale:
- **Windows**: download dari https://ffmpeg.org/download.html, extract, tambahkan
  folder `bin` ke PATH. Atau lebih gampang: `choco install ffmpeg` atau `winget install ffmpeg`.
- **Cek instalasi**: `ffmpeg -version`

Kalau ffmpeg tidak ada, program tetap jalan tapi hasil video tanpa audio.

Model AI (`RealESRGAN_x4plus.pth` / `RealESRGAN_x2plus.pth`) otomatis
terdownload ke folder `weights/` saat pertama kali dijalankan (±64MB).

## 2. Cara Pakai — GUI (paling gampang, direkomendasikan untuk pemula)

Tinggal double-click file **`buka_gui.bat`** di folder ini — otomatis mengaktifkan
venv dan membuka aplikasi.

Atau lewat terminal (venv aktif):
```bash
python gui.py
```

Di dalam GUI, tinggal:
1. Klik **"Pilih Gambar / Video"** (atau **"Pilih Folder"** untuk banyak gambar sekaligus)
2. Atur opsi yang diinginkan — semua fitur dari versi command line ada di sini:
   scale 2x/4x, custom outscale, GPU/CPU, half precision, tile size, sharpening,
   dan face enhancement
3. Klik **"🚀 Mulai Upscale"**
4. Progress & log ditampilkan langsung di layar, tombol **"📂 Buka Folder Hasil"**
   aktif otomatis kalau sudah selesai

> Catatan: opsi **Face Enhancement** di GUI tetap butuh `pip install gfpgan`
> terlebih dulu (lihat bagian 6c) — kalau belum ada, GUI akan kasih tahu lewat
> pop-up saat tombol Mulai Upscale ditekan, bukan lewat command line.

---

## 3. Cara Pakai — Command Line (Gambar)

```bash
python upscale.py -i foto.jpg
python upscale.py -i foto.jpg -o hasil_hd.png
python upscale.py -i folder_foto/ --batch
```

## 4. Cara Pakai — Video (BARU)

Tinggal arahkan ke file video, otomatis terdeteksi:

```bash
python upscale.py -i video.mp4
```
Hasil disimpan sebagai `video_hd.mp4` (audio asli tetap ada kalau ffmpeg terpasang).

```bash
python upscale.py -i video.mp4 -o hasil.mp4 --scale 2
```

> ⚠️ **Catatan penting soal video**: proses berjalan frame-per-frame, jadi
> video 1 menit @30fps = 1800 frame yang harus diupscale satu-satu. Di RTX
> 3050, ini bisa memakan waktu **belasan menit sampai berjam-jam** tergantung
> resolusi & durasi. Disarankan:
> - Coba dulu dengan video pendek (5-10 detik) untuk tes.
> - Pakai `--scale 2` kalau video sudah lumayan besar resolusinya (lebih cepat
>   dari `--scale 4`).
> - Kalau video sangat panjang, potong dulu jadi beberapa bagian.

## 5. Optimasi Khusus untuk RTX 3050 4GB

Program ini **otomatis** mengatur `tile-size` berdasarkan VRAM yang terdeteksi
supaya tidak kehabisan memori (CUDA out of memory), dan otomatis memakai half
precision (FP16) di GPU untuk menghemat VRAM + mempercepat proses. Tidak perlu
setting manual, tapi kalau masih kena OOM, program otomatis coba ulang dengan
tile lebih kecil. Bisa juga diatur manual:

```bash
python upscale.py -i foto.jpg --tile 128
```

Semakin kecil nilai `--tile`, semakin hemat VRAM tapi sedikit lebih lambat.

Untuk menonaktifkan GPU (paksa pakai CPU):
```bash
python upscale.py -i foto.jpg --no-gpu
```

## 6. Bikin Hasil Lebih "Ultra HD"

Tiga opsi bisa dikombinasikan sesuai kebutuhan:

**a. Outscale lebih besar dari model scale**
Model tetap 4x tapi hasil akhir di-resize lebih jauh lagi (mis. dari foto 720p
ke setara 4K/8K):
```bash
python upscale.py -i foto.jpg --scale 4 --outscale 8
```
Catatan: detail yang ditambahkan tetap berasal dari model 4x, outscale 8
hanya membesarkan resolusi akhirnya — jadi tidak akan "lebih tajam" dari
outscale 4, tapi resolusinya lebih besar untuk keperluan cetak/layar besar.

**b. Sharpening pasca-proses (detail lebih tajam/kontras)**
```bash
python upscale.py -i foto.jpg --sharpen
python upscale.py -i foto.jpg --sharpen --sharpen-amount 1.5   # lebih kuat
```

**c. Face enhancement (khusus foto orang/wajah) — biasanya paling terasa**
```bash
pip install gfpgan
python upscale.py -i foto.jpg --face-enhance
```
GFPGAN akan merekonstruksi detail wajah (mata, kulit, rambut) yang biasanya
tetap agak blur walau sudah di-upscale Real-ESRGAN biasa.

**d. Kualitas maksimal (lebih lambat, tanpa half precision)**
```bash
python upscale.py -i foto.jpg --no-half
```

**Kombinasi lengkap untuk hasil terbaik pada foto orang:**
```bash
python upscale.py -i foto.jpg --scale 4 --face-enhance --sharpen --no-half
```

## 7. Tips

- Format gambar yang didukung: `.jpg .jpeg .png .bmp .webp`
- Format video yang didukung: `.mp4 .mov .avi .mkv .webm`
- Untuk foto wajah/orang yang ingin restorasi lebih detail, bisa upgrade ke
  varian **GFPGAN** — bilang saja kalau mau saya tambahkan.
- Gambar/video dengan resolusi sumber sangat besar sebaiknya pakai `--scale 2`.

## 8. Struktur Proyek

```
image-upscaler/
├── upscale.py         # logic utama (gambar + video) + bisa dipakai command line
├── gui.py              # aplikasi GUI (pakai upscale.py di belakang layar)
├── buka_gui.bat        # double-click untuk buka GUI langsung
├── requirements.txt   # daftar dependency
└── README.md          # panduan ini
```
