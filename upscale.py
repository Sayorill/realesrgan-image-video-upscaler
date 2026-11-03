#!/usr/bin/env python3
"""
AI Image & Video Upscaler menggunakan Real-ESRGAN
Upscale gambar maupun video menjadi HD (2x atau 4x) dengan kualitas AI.
Dioptimalkan otomatis untuk GPU VRAM kecil (mis. RTX 3050 4GB).

Penggunaan:
    python upscale.py -i gambar.jpg
    python upscale.py -i gambar.jpg -o hasil.png --scale 4
    python upscale.py -i folder_input/ --batch
    python upscale.py -i video.mp4                # otomatis terdeteksi sebagai video
    python upscale.py -i video.mp4 --scale 2 --tile 200
"""

import argparse
import shutil
import subprocess
import sys
import tempfile
import warnings
from pathlib import Path

import cv2
import torch
from tqdm import tqdm

# Warning ini murni informasi soal perubahan default PyTorch di masa depan,
# tidak memengaruhi hasil upscaling — disembunyikan supaya tidak membingungkan.
warnings.filterwarnings("ignore", category=FutureWarning)

MODEL_INFO = {
    2: {
        "name": "RealESRGAN_x2plus",
        "url": "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.1/RealESRGAN_x2plus.pth",
        "netscale": 2,
    },
    4: {
        "name": "RealESRGAN_x4plus",
        "url": "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.0/RealESRGAN_x4plus.pth",
        "netscale": 4,
    },
}

SUPPORTED_IMAGE_EXT = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
SUPPORTED_VIDEO_EXT = {".mp4", ".mov", ".avi", ".mkv", ".webm"}

# VRAM kecil (4GB seperti RTX 3050 laptop) gampang OOM kalau tile terlalu besar
# di scale 4x. Nilai ini dipilih supaya aman by default, bisa dioverride manual.
AUTO_TILE_BY_SCALE = {2: 512, 4: 256}


class MissingDependencyError(Exception):
    """Dependency opsional (mis. gfpgan) belum terinstall. Dipakai supaya GUI bisa
    menampilkan pesan error yang rapi, bukan langsung mematikan aplikasi."""
    pass


def has_ffmpeg() -> bool:
    return shutil.which("ffmpeg") is not None


def get_gpu_info():
    if not torch.cuda.is_available():
        return None
    props = torch.cuda.get_device_properties(0)
    return {"name": props.name, "vram_gb": props.total_memory / (1024 ** 3)}


def build_upsampler(scale: int, use_gpu: bool, tile: int, half: bool):
    """Menyiapkan model Real-ESRGAN. Model akan otomatis didownload saat pertama kali dipakai."""
    from basicsr.archs.rrdbnet_arch import RRDBNet
    from realesrgan import RealESRGANer

    info = MODEL_INFO[scale]
    model = RRDBNet(
        num_in_ch=3,
        num_out_ch=3,
        num_feat=64,
        num_block=23,
        num_grow_ch=32,
        scale=info["netscale"],
    )

    gpu_info = get_gpu_info()
    device = "cuda" if (use_gpu and gpu_info is not None) else "cpu"
    if use_gpu and device == "cpu":
        print("[!] GPU diminta tapi tidak terdeteksi, fallback ke CPU (akan jauh lebih lambat).")

    # Auto-tune tile & half precision kalau user tidak menentukan manual (tile=0 -> auto)
    if tile == 0 and device == "cuda":
        tile = AUTO_TILE_BY_SCALE.get(scale, 256)
        print(f"[i] Auto tile-size = {tile} (disesuaikan untuk VRAM {gpu_info['vram_gb']:.1f}GB pada {gpu_info['name']})")

    use_half = half and device == "cuda"

    upsampler = RealESRGANer(
        scale=info["netscale"],
        model_path=info["url"],
        model=model,
        tile=tile,          # 0 = tanpa tiling
        tile_pad=10,
        pre_pad=0,
        half=use_half,      # half precision menghemat VRAM signifikan di GPU
        device=device,
    )
    print(f"[i] Device: {device.upper()} | Model: {info['name']} | tile={tile} | half={use_half}")
    return upsampler


def build_face_enhancer():
    """Menyiapkan GFPGAN untuk restorasi wajah pasca-upscale. Model auto-download saat pertama dipakai."""
    try:
        from gfpgan import GFPGANer
    except ImportError:
        raise MissingDependencyError(
            "Package 'gfpgan' belum terinstall. Jalankan dulu: pip install gfpgan"
        )

    model_path = "https://github.com/TencentARC/GFPGAN/releases/download/v1.3.0/GFPGANv1.4.pth"
    # upscale=1 karena background sudah di-upscale duluan oleh Real-ESRGAN;
    # GFPGAN di sini cuma fokus memperbaiki detail wajah, bukan resize lagi.
    enhancer = GFPGANer(
        model_path=model_path,
        upscale=1,
        arch="clean",
        channel_multiplier=2,
        bg_upsampler=None,
    )
    print("[i] Face enhancement (GFPGAN) aktif.")
    return enhancer


def sharpen_image(img, amount: float = 1.0, radius: float = 1.5):
    """Unsharp masking sederhana untuk mempertajam detail hasil upscale."""
    blurred = cv2.GaussianBlur(img, (0, 0), radius)
    return cv2.addWeighted(img, 1 + amount, blurred, -amount, 0)


def upscale_frame(upsampler, img, outscale, face_enhancer=None, sharpen=False, sharpen_amount=1.0, retry_tiles=(128, 64)):
    """Upscale satu frame/gambar, otomatis mengecilkan tile kalau OOM."""
    try:
        output, _ = upsampler.enhance(img, outscale=outscale)
    except RuntimeError as e:
        if "out of memory" in str(e).lower():
            torch.cuda.empty_cache()
            output = None
            for t in retry_tiles:
                print(f"[!] CUDA out of memory, mencoba ulang dengan tile={t}...")
                upsampler.tile = t
                try:
                    output, _ = upsampler.enhance(img, outscale=outscale)
                    break
                except RuntimeError:
                    torch.cuda.empty_cache()
                    continue
            if output is None:
                raise
        else:
            raise

    if face_enhancer is not None:
        torch.cuda.empty_cache()  # bebaskan memori sisa dari tahap Real-ESRGAN dulu
        try:
            _, _, output = face_enhancer.enhance(output, has_aligned=False, only_center_face=False, paste_back=True)
        except RuntimeError as e:
            if "out of memory" in str(e).lower():
                torch.cuda.empty_cache()
                print("[!] GFPGAN kehabisan VRAM, dilewati untuk gambar ini (hasil upscale dasar tetap dipakai).")
            else:
                raise

    if sharpen:
        output = sharpen_image(output, amount=sharpen_amount)

    return output


def ensure_image_extension(out_path: Path, fallback_suffix: str = ".png") -> Path:
    """Pastikan out_path punya ekstensi gambar yang didukung OpenCV, kalau tidak
    (kosong / typo / ekstensi asing), otomatis diganti supaya penyimpanan tidak gagal."""
    if out_path.suffix.lower() in SUPPORTED_IMAGE_EXT:
        return out_path
    fixed = out_path.with_suffix(fallback_suffix)
    shown = out_path.suffix if out_path.suffix else "(tidak ada)"
    print(f"[!] Ekstensi output '{shown}' tidak didukung, disimpan sebagai '{fixed.name}' sebagai gantinya.")
    return fixed


def upscale_one(upsampler, in_path: Path, out_path: Path, outscale: float, face_enhancer=None, sharpen=False, sharpen_amount=1.0):
    img = cv2.imread(str(in_path), cv2.IMREAD_UNCHANGED)
    if img is None:
        print(f"[x] Gagal membaca gambar: {in_path}")
        return False

    try:
        output = upscale_frame(upsampler, img, outscale, face_enhancer, sharpen, sharpen_amount)
    except RuntimeError as e:
        print(f"[x] Gagal memproses {in_path.name}: {e}")
        print("    Coba turunkan --tile manual (mis. --tile 128) atau pakai --scale 2.")
        return False

    fallback_ext = in_path.suffix if in_path.suffix.lower() in SUPPORTED_IMAGE_EXT else ".png"
    out_path = ensure_image_extension(Path(out_path), fallback_suffix=fallback_ext)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    ok = cv2.imwrite(str(out_path), output)
    if not ok:
        print(f"[x] cv2.imwrite gagal menyimpan ke: {out_path}")
        return False
    return True


def upscale_video(upsampler, in_path: Path, out_path: Path, outscale: float, face_enhancer=None,
                   sharpen=False, sharpen_amount=1.0, progress_callback=None):
    """progress_callback(frame_idx, total_frames), kalau diisi dipakai sebagai pengganti tqdm
    (dipakai GUI supaya progress bisa ditampilkan di progress bar, bukan spam teks).

    Coba dulu stream frame ke ffmpeg (codec libx264, mendukung audio + resolusi besar).
    Kalau ffmpeg tidak bisa dijalankan (tidak ada / diblokir kebijakan sistem seperti Smart App
    Control), otomatis fallback ke cv2.VideoWriter (tanpa audio) supaya proses tetap selesai."""
    cap = cv2.VideoCapture(str(in_path))
    if not cap.isOpened():
        print(f"[x] Gagal membuka video: {in_path}")
        return False

    fps = cap.get(cv2.CAP_PROP_FPS) or 25
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    out_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"[i] Video: {total_frames} frame @ {fps:.2f}fps. Proses ini bisa memakan waktu cukup lama.")

    frame_idx = 0
    ffmpeg_proc = None
    ffmpeg_stderr_file = None
    cv2_writer = None
    use_ffmpeg = has_ffmpeg()
    out_w = out_h = None

    pbar = None if progress_callback else tqdm(total=total_frames if total_frames > 0 else None, desc="Upscaling video")
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            try:
                out_frame = upscale_frame(upsampler, frame, outscale, face_enhancer, sharpen, sharpen_amount)
            except RuntimeError as e:
                print(f"[x] Gagal di frame {frame_idx}: {e}")
                return False

            if ffmpeg_proc is None and cv2_writer is None:
                out_h, out_w = out_frame.shape[:2]
                print(f"[i] Resolusi output: {out_w}x{out_h}")

                if use_ffmpeg:
                    cmd = [
                        "ffmpeg", "-y",
                        "-f", "rawvideo", "-pix_fmt", "bgr24",
                        "-s", f"{out_w}x{out_h}", "-r", str(fps),
                        "-i", "-",                      # input 1: frame mentah dari stdin (video, tanpa audio)
                        "-i", str(in_path),              # input 2: file asli, cuma diambil audionya
                        "-map", "0:v:0", "-map", "1:a:0?",
                        "-c:v", "libx264", "-preset", "medium", "-crf", "18", "-pix_fmt", "yuv420p",
                        "-c:a", "aac", "-shortest",
                        str(out_path),
                    ]
                    try:
                        # stderr ditampung ke FILE (bukan subprocess.PIPE) supaya tidak deadlock:
                        # ffmpeg terus menulis log progress ke stderr, kalau ditampung pipe biasa
                        # dan cuma dibaca di akhir, buffernya bisa penuh dan bikin ffmpeg (lalu
                        # Python juga) macet total saling menunggu.
                        ffmpeg_stderr_file = tempfile.TemporaryFile(mode="w+b")
                        ffmpeg_proc = subprocess.Popen(
                            cmd, stdin=subprocess.PIPE, stdout=subprocess.DEVNULL, stderr=ffmpeg_stderr_file
                        )
                    except OSError as e:
                        print(f"[!] Tidak bisa menjalankan ffmpeg ({e}).")
                        print("    Kemungkinan diblokir kebijakan sistem (mis. Smart App Control di Windows).")
                        print("    Fallback: video akan disimpan TANPA audio memakai encoder bawaan OpenCV.")
                        use_ffmpeg = False

                if not use_ffmpeg:
                    fourcc = cv2.VideoWriter_fourcc(*"avc1")
                    cv2_writer = cv2.VideoWriter(str(out_path), fourcc, fps, (out_w, out_h))
                    if not cv2_writer.isOpened():
                        print("[!] Codec 'avc1' tidak tersedia, mencoba 'mp4v'...")
                        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
                        cv2_writer = cv2.VideoWriter(str(out_path), fourcc, fps, (out_w, out_h))
                    if not cv2_writer.isOpened():
                        print("[x] Gagal membuat video writer. Resolusi mungkin terlalu besar untuk encoder yang tersedia.")
                        print("    Coba turunkan --scale ke 2 atau proses video yang resolusinya lebih kecil.")
                        return False

            if ffmpeg_proc is not None:
                try:
                    ffmpeg_proc.stdin.write(out_frame.tobytes())
                except BrokenPipeError:
                    print("[x] ffmpeg berhenti tak terduga di tengah proses.")
                    break
            else:
                cv2_writer.write(out_frame)

            frame_idx += 1
            if progress_callback:
                progress_callback(frame_idx, total_frames)
            else:
                pbar.update(1)
    finally:
        cap.release()
        if cv2_writer:
            cv2_writer.release()
        if pbar:
            pbar.close()

    if frame_idx == 0:
        print("[x] Tidak ada frame yang berhasil diproses.")
        if ffmpeg_proc:
            ffmpeg_proc.stdin.close()
            ffmpeg_proc.wait()
        if ffmpeg_stderr_file:
            ffmpeg_stderr_file.close()
        return False

    if cv2_writer is not None:
        print(f"[✓] Video berhasil disimpan (TANPA audio) ke: {out_path}")
        return True

    ffmpeg_proc.stdin.close()
    ffmpeg_proc.wait()

    ok = ffmpeg_proc.returncode == 0
    if not ok:
        ffmpeg_stderr_file.seek(0)
        stderr_output = ffmpeg_stderr_file.read().decode(errors="ignore")
        print("[x] ffmpeg gagal encode video.")
        print(stderr_output[-800:])
    ffmpeg_stderr_file.close()

    if not ok:
        return False

    print(f"[✓] Video berhasil di-encode (dengan audio) ke: {out_path}")
    return True


def main():
    parser = argparse.ArgumentParser(description="AI Image & Video Upscaler (Real-ESRGAN)")
    parser.add_argument("-i", "--input", required=True, help="Path gambar, video, atau folder input")
    parser.add_argument("-o", "--output", help="Path output (file atau folder)")
    parser.add_argument("--scale", type=int, choices=[2, 4], default=4, help="Faktor model (2x/4x), default 4")
    parser.add_argument("--outscale", type=float, default=None, help="Faktor upscale akhir (default = sama dengan --scale)")
    parser.add_argument("--batch", action="store_true", help="Proses seluruh folder gambar input")
    parser.add_argument("--no-gpu", action="store_true", help="Paksa pakai CPU walau GPU tersedia")
    parser.add_argument("--tile", type=int, default=0, help="Ukuran tile manual untuk hemat VRAM (0 = auto)")
    parser.add_argument("--no-half", action="store_true", help="Nonaktifkan half precision (default aktif di GPU, aktifkan untuk kualitas maksimal)")
    parser.add_argument("--sharpen", action="store_true", help="Aktifkan sharpening pasca-proses untuk detail lebih tajam")
    parser.add_argument("--sharpen-amount", type=float, default=1.0, help="Kekuatan sharpening (default 1.0, coba 0.5-2.0)")
    parser.add_argument("--face-enhance", action="store_true", help="Restorasi wajah dengan GFPGAN (butuh: pip install gfpgan)")
    args = parser.parse_args()

    in_path = Path(args.input)
    outscale = args.outscale or args.scale

    if args.outscale and args.outscale > args.scale:
        print(f"[i] outscale={args.outscale} lebih besar dari model scale={args.scale}: hasil model di-resize lagi ke resolusi lebih tinggi.")

    upsampler = build_upsampler(
        scale=args.scale,
        use_gpu=not args.no_gpu,
        tile=args.tile,
        half=not args.no_half,
    )

    face_enhancer = None
    if args.face_enhance:
        try:
            face_enhancer = build_face_enhancer()
        except MissingDependencyError as e:
            print(f"[x] {e}")
            sys.exit(1)
    if args.sharpen:
        print(f"[i] Sharpening aktif (amount={args.sharpen_amount})")

    is_video = in_path.is_file() and in_path.suffix.lower() in SUPPORTED_VIDEO_EXT

    if is_video:
        out_path = Path(args.output) if args.output else in_path.with_name(f"{in_path.stem}_hd.mp4")
        print(f"[i] Mode VIDEO: {in_path} -> {out_path}")
        if upscale_video(upsampler, in_path, out_path, outscale, face_enhancer, args.sharpen, args.sharpen_amount):
            print(f"[✓] Selesai! Video hasil disimpan di: {out_path}")
        else:
            sys.exit(1)

    elif args.batch or in_path.is_dir():
        if not in_path.is_dir():
            print(f"[x] --batch dipakai tapi {in_path} bukan folder.")
            sys.exit(1)

        out_dir = Path(args.output) if args.output else in_path.parent / f"{in_path.name}_upscaled"
        files = [f for f in in_path.iterdir() if f.suffix.lower() in SUPPORTED_IMAGE_EXT]

        if not files:
            print(f"[x] Tidak ada gambar yang didukung di {in_path}")
            sys.exit(1)

        print(f"[i] Memproses {len(files)} gambar dari {in_path} -> {out_dir}")
        success = 0
        for f in tqdm(files, desc="Upscaling"):
            out_file = out_dir / f"{f.stem}_hd{f.suffix}"
            if upscale_one(upsampler, f, out_file, outscale, face_enhancer, args.sharpen, args.sharpen_amount):
                success += 1
        print(f"[✓] Selesai: {success}/{len(files)} gambar berhasil diupscale ke {out_dir}")

    else:
        if not in_path.exists():
            print(f"[x] File tidak ditemukan: {in_path}")
            sys.exit(1)
        out_path = Path(args.output) if args.output else in_path.with_name(f"{in_path.stem}_hd{in_path.suffix}")
        print(f"[i] Memproses {in_path} -> {out_path}")
        if upscale_one(upsampler, in_path, out_path, outscale, face_enhancer, args.sharpen, args.sharpen_amount):
            print(f"[✓] Selesai! Hasil disimpan di: {out_path}")
        else:
            sys.exit(1)


if __name__ == "__main__":
    main()
