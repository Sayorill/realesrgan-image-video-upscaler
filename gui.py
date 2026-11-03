#!/usr/bin/env python3
"""
GUI untuk AI Image & Video Upscaler (Real-ESRGAN + GFPGAN)
Membungkus semua fitur di upscale.py dalam tampilan point-and-click.

Jalankan dengan:
    python gui.py
"""

import os
import subprocess
import sys
import threading
import queue
import traceback
from pathlib import Path

import tkinter as tk
from tkinter import ttk, filedialog, messagebox

import cv2
from PIL import Image, ImageTk

import upscale  # file upscale.py harus ada di folder yang sama


class TextRedirector:
    """Menangkap print() dari thread proses lalu dikirim ke GUI lewat queue (thread-safe)."""

    def __init__(self, q: queue.Queue):
        self.q = q

    def write(self, msg):
        if msg.strip():
            self.q.put(("log", msg if msg.endswith("\n") else msg + "\n"))

    def flush(self):
        pass


class UpscalerGUI:
    def __init__(self, root: tk.Tk):
        self.root = root
        root.title("AI Image & Video Upscaler")
        root.geometry("820x920")
        root.minsize(760, 820)

        self.input_path = None
        self.is_folder = False
        self.is_video = False
        self.processing = False
        self.last_output_dir = None
        self._preview_imgtk = None

        self.log_queue = queue.Queue()

        self._build_ui()
        self._poll_queue()

    # ---------------------------------------------------------------- UI ----
    def _build_ui(self):
        pad = {"padx": 10, "pady": 6}

        # ---- 1. File ----
        frame_file = ttk.LabelFrame(self.root, text="1. Pilih File")
        frame_file.pack(fill="x", **pad)

        btn_row = ttk.Frame(frame_file)
        btn_row.pack(fill="x", padx=8, pady=6)
        ttk.Button(btn_row, text="🖼 Pilih Gambar / Video", command=self.pick_file).pack(side="left", padx=(0, 6))
        ttk.Button(btn_row, text="📁 Pilih Folder (Batch)", command=self.pick_folder).pack(side="left")

        self.file_label = ttk.Label(frame_file, text="Belum ada file dipilih", foreground="#555")
        self.file_label.pack(anchor="w", padx=8, pady=(0, 6))

        self.preview_label = ttk.Label(frame_file)
        self.preview_label.pack(padx=8, pady=(0, 8))

        # ---- 2. Output ----
        frame_out = ttk.LabelFrame(self.root, text="2. Lokasi Output (opsional)")
        frame_out.pack(fill="x", **pad)

        out_row = ttk.Frame(frame_out)
        out_row.pack(fill="x", padx=8, pady=6)
        self.output_path_var = tk.StringVar()
        ttk.Entry(out_row, textvariable=self.output_path_var).pack(side="left", fill="x", expand=True, padx=(0, 6))
        ttk.Button(out_row, text="Pilih...", command=self.pick_output).pack(side="left")

        ttk.Label(frame_out, text="Kosongkan untuk simpan otomatis (nama_hd) di lokasi yang sama",
                  foreground="#777").pack(anchor="w", padx=8, pady=(0, 6))

        # ---- 3. Opsi Upscale ----
        frame_opts = ttk.LabelFrame(self.root, text="3. Opsi Upscale")
        frame_opts.pack(fill="x", **pad)

        row1 = ttk.Frame(frame_opts); row1.pack(fill="x", padx=8, pady=4)
        ttk.Label(row1, text="Model scale:").pack(side="left")
        self.scale_var = tk.IntVar(value=4)
        ttk.Radiobutton(row1, text="2x (lebih cepat)", variable=self.scale_var, value=2).pack(side="left", padx=6)
        ttk.Radiobutton(row1, text="4x (lebih detail)", variable=self.scale_var, value=4).pack(side="left")

        row2 = ttk.Frame(frame_opts); row2.pack(fill="x", padx=8, pady=4)
        self.custom_outscale_var = tk.BooleanVar(value=False)
        ttk.Checkbutton(row2, text="Custom outscale (resolusi akhir lebih besar):",
                         variable=self.custom_outscale_var).pack(side="left")
        self.outscale_var = tk.DoubleVar(value=4.0)
        ttk.Spinbox(row2, from_=1, to=16, increment=0.5, textvariable=self.outscale_var, width=6).pack(side="left", padx=6)

        row3 = ttk.Frame(frame_opts); row3.pack(fill="x", padx=8, pady=4)
        self.use_gpu_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(row3, text="Gunakan GPU jika tersedia", variable=self.use_gpu_var).pack(side="left")
        self.half_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(row3, text="Half precision (hemat VRAM)", variable=self.half_var).pack(side="left", padx=12)

        row4 = ttk.Frame(frame_opts); row4.pack(fill="x", padx=8, pady=4)
        ttk.Label(row4, text="Tile size (0 = auto):").pack(side="left")
        self.tile_var = tk.IntVar(value=0)
        ttk.Spinbox(row4, from_=0, to=1024, increment=32, textvariable=self.tile_var, width=6).pack(side="left", padx=6)
        ttk.Label(row4, text="Kecilkan (mis. 128) kalau muncul error 'out of memory'",
                  foreground="#777").pack(side="left", padx=(6, 0))

        # ---- 4. Enhancement ----
        frame_enh = ttk.LabelFrame(self.root, text="4. Enhancement Tambahan")
        frame_enh.pack(fill="x", **pad)

        row5 = ttk.Frame(frame_enh); row5.pack(fill="x", padx=8, pady=4)
        self.sharpen_var = tk.BooleanVar(value=False)
        ttk.Checkbutton(row5, text="Sharpening", variable=self.sharpen_var).pack(side="left")
        ttk.Label(row5, text="Kekuatan:").pack(side="left", padx=(12, 0))
        self.sharpen_amount_var = tk.DoubleVar(value=1.0)
        ttk.Spinbox(row5, from_=0.1, to=3.0, increment=0.1, textvariable=self.sharpen_amount_var, width=6).pack(side="left", padx=6)

        row6 = ttk.Frame(frame_enh); row6.pack(fill="x", padx=8, pady=4)
        self.face_enhance_var = tk.BooleanVar(value=False)
        ttk.Checkbutton(row6, text="Face Enhancement / GFPGAN (untuk foto wajah — perlu: pip install gfpgan)",
                         variable=self.face_enhance_var).pack(side="left")

        # ---- 5. Aksi ----
        frame_action = ttk.Frame(self.root)
        frame_action.pack(fill="x", **pad)

        self.run_btn = ttk.Button(frame_action, text="🚀  Mulai Upscale", command=self.start_processing)
        self.run_btn.pack(side="left")
        self.open_folder_btn = ttk.Button(frame_action, text="📂  Buka Folder Hasil",
                                           command=self.open_output_folder, state="disabled")
        self.open_folder_btn.pack(side="left", padx=8)

        self.status_var = tk.StringVar(value="Siap.")
        ttk.Label(frame_action, textvariable=self.status_var, foreground="#0a5",
                  font=("Segoe UI", 9, "bold")).pack(side="right")

        self.progress = ttk.Progressbar(self.root, mode="determinate")
        self.progress.pack(fill="x", padx=10, pady=(0, 4))

        # ---- Log ----
        frame_log = ttk.LabelFrame(self.root, text="Log Proses")
        frame_log.pack(fill="both", expand=True, padx=10, pady=(0, 10))

        log_scroll = ttk.Scrollbar(frame_log)
        log_scroll.pack(side="right", fill="y")
        self.log_text = tk.Text(frame_log, height=10, state="disabled", bg="#111318", fg="#7CFC7C",
                                 font=("Consolas", 9), yscrollcommand=log_scroll.set, wrap="word")
        self.log_text.pack(side="left", fill="both", expand=True, padx=(6, 0), pady=6)
        log_scroll.config(command=self.log_text.yview)

    # ---------------------------------------------------------- file pickers ----
    def pick_file(self):
        exts = sorted(upscale.SUPPORTED_IMAGE_EXT | upscale.SUPPORTED_VIDEO_EXT)
        filetypes = [("Gambar & Video", " ".join(f"*{e}" for e in exts)), ("Semua file", "*.*")]
        path = filedialog.askopenfilename(title="Pilih gambar atau video", filetypes=filetypes)
        if not path:
            return
        self.input_path = Path(path)
        self.is_folder = False
        self.is_video = self.input_path.suffix.lower() in upscale.SUPPORTED_VIDEO_EXT
        kind = "Video" if self.is_video else "Gambar"
        self.file_label.config(text=f"{kind}: {self.input_path.name}")
        self.output_path_var.set("")
        self._update_preview()

    def pick_folder(self):
        path = filedialog.askdirectory(title="Pilih folder berisi gambar (batch)")
        if not path:
            return
        self.input_path = Path(path)
        self.is_folder = True
        self.is_video = False
        files = [f for f in self.input_path.iterdir() if f.suffix.lower() in upscale.SUPPORTED_IMAGE_EXT]
        self.file_label.config(text=f"Folder: {self.input_path.name} ({len(files)} gambar ditemukan)")
        self.output_path_var.set("")
        if files:
            self._show_preview_image(files[0])
        else:
            self.preview_label.config(image="", text="(tidak ada gambar yang didukung di folder ini)")

    def pick_output(self):
        if not self.input_path:
            messagebox.showinfo("Info", "Pilih file/folder input dulu ya.")
            return
        if self.is_folder:
            path = filedialog.askdirectory(title="Pilih folder output")
        else:
            default_ext = ".mp4" if self.is_video else (self.input_path.suffix or ".png")
            default_name = f"{self.input_path.stem}_hd{default_ext}"
            if self.is_video:
                filetypes = [("Video MP4", "*.mp4"), ("Semua file", "*.*")]
            else:
                filetypes = [("PNG", "*.png"), ("JPEG", "*.jpg *.jpeg"), ("Semua file", "*.*")]
            path = filedialog.asksaveasfilename(
                title="Simpan sebagai",
                initialfile=default_name,
                defaultextension=default_ext,
                filetypes=filetypes,
            )
        if path:
            self.output_path_var.set(path)

    def _update_preview(self):
        if self.is_video:
            try:
                cap = cv2.VideoCapture(str(self.input_path))
                ret, frame = cap.read()
                cap.release()
                if not ret:
                    self.preview_label.config(image="", text="(preview video tidak tersedia)")
                    return
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                pil_img = Image.fromarray(rgb)
                self._set_preview(pil_img)
            except Exception:
                self.preview_label.config(image="", text="(gagal memuat preview video)")
        else:
            self._show_preview_image(self.input_path)

    def _show_preview_image(self, path: Path):
        try:
            pil_img = Image.open(path).convert("RGB")
            self._set_preview(pil_img)
        except Exception:
            self.preview_label.config(image="", text="(gagal memuat preview)")

    def _set_preview(self, pil_img):
        pil_img.thumbnail((300, 170))
        self._preview_imgtk = ImageTk.PhotoImage(pil_img)
        self.preview_label.config(image=self._preview_imgtk, text="")

    def open_output_folder(self):
        if not self.last_output_dir or not self.last_output_dir.exists():
            return
        if sys.platform == "win32":
            os.startfile(self.last_output_dir)
        elif sys.platform == "darwin":
            subprocess.run(["open", str(self.last_output_dir)])
        else:
            subprocess.run(["xdg-open", str(self.last_output_dir)])

    # ---------------------------------------------------------- processing ----
    def start_processing(self):
        if self.processing:
            return
        if not self.input_path:
            messagebox.showwarning("Belum ada file", "Pilih gambar, video, atau folder dulu.")
            return
        if self.face_enhance_var.get():
            try:
                import gfpgan  # noqa: F401
            except ImportError:
                messagebox.showerror(
                    "Dependency belum ada",
                    "Face Enhancement butuh package 'gfpgan'.\n\nJalankan di terminal (venv aktif):\npip install gfpgan",
                )
                return

        self.processing = True
        self.run_btn.config(state="disabled")
        self.open_folder_btn.config(state="disabled")
        self.status_var.set("Memproses...")
        self._clear_log()
        self.progress.config(mode="determinate")
        self.progress["value"] = 0

        params = dict(
            input_path=self.input_path,
            is_folder=self.is_folder,
            is_video=self.is_video,
            output=self.output_path_var.get().strip() or None,
            scale=self.scale_var.get(),
            outscale=(self.outscale_var.get() if self.custom_outscale_var.get() else None),
            use_gpu=self.use_gpu_var.get(),
            half=self.half_var.get(),
            tile=self.tile_var.get(),
            sharpen=self.sharpen_var.get(),
            sharpen_amount=self.sharpen_amount_var.get(),
            face_enhance=self.face_enhance_var.get(),
        )

        threading.Thread(target=self._run_worker, args=(params,), daemon=True).start()

    def _run_worker(self, p):
        old_stdout = sys.stdout
        old_stderr = sys.stderr
        redirector = TextRedirector(self.log_queue)
        sys.stdout = redirector
        sys.stderr = redirector
        ok = False
        try:
            outscale = p["outscale"] or p["scale"]
            print(f"[i] Mulai proses... scale={p['scale']}x, outscale={outscale}")

            upsampler = upscale.build_upsampler(
                scale=p["scale"], use_gpu=p["use_gpu"], tile=p["tile"], half=p["half"]
            )

            face_enhancer = None
            if p["face_enhance"]:
                try:
                    face_enhancer = upscale.build_face_enhancer()
                except upscale.MissingDependencyError as e:
                    print(f"[x] {e}")
                    self.log_queue.put(("done", False))
                    return

            in_path = p["input_path"]

            if p["is_video"]:
                out_path = Path(p["output"]) if p["output"] else in_path.with_name(f"{in_path.stem}_hd.mp4")

                def _cb(idx, total):
                    if total > 0:
                        self.log_queue.put(("progress", int(idx / total * 100)))
                    if idx % 10 == 0 or idx == total:
                        print(f"[i] Frame {idx}/{total or '?'}")

                ok = upscale.upscale_video(
                    upsampler, in_path, out_path, outscale, face_enhancer,
                    p["sharpen"], p["sharpen_amount"], progress_callback=_cb,
                )
                self.last_output_dir = out_path.parent

            elif p["is_folder"]:
                out_dir = Path(p["output"]) if p["output"] else in_path.parent / f"{in_path.name}_upscaled"
                files = [f for f in in_path.iterdir() if f.suffix.lower() in upscale.SUPPORTED_IMAGE_EXT]
                total = len(files)
                success = 0
                for idx, f in enumerate(files, 1):
                    out_file = out_dir / f"{f.stem}_hd{f.suffix}"
                    print(f"[i] ({idx}/{total}) {f.name}")
                    if upscale.upscale_one(upsampler, f, out_file, outscale, face_enhancer,
                                            p["sharpen"], p["sharpen_amount"]):
                        success += 1
                    self.log_queue.put(("progress", int(idx / total * 100)))
                print(f"[✓] Selesai: {success}/{total} gambar berhasil.")
                ok = success > 0
                self.last_output_dir = out_dir

            else:
                out_path = Path(p["output"]) if p["output"] else in_path.with_name(f"{in_path.stem}_hd{in_path.suffix}")
                ok = upscale.upscale_one(upsampler, in_path, out_path, outscale, face_enhancer,
                                          p["sharpen"], p["sharpen_amount"])
                self.last_output_dir = out_path.parent

            if ok:
                print("[✓] Selesai!")

        except SystemExit as e:
            print(f"[x] Proses dihentikan paksa oleh salah satu library (kode: {e.code}).")
            ok = False
        except Exception as e:
            print(f"[x] Error tak terduga: {e}")
            traceback.print_exc()
            ok = False
        except BaseException as e:
            print(f"[x] Error fatal tak terduga: {type(e).__name__}: {e}")
            traceback.print_exc()
            ok = False
        finally:
            sys.stdout = old_stdout
            sys.stderr = old_stderr
            self.log_queue.put(("done", ok))

    # ---------------------------------------------------------- queue polling ----
    def _poll_queue(self):
        try:
            while True:
                kind, payload = self.log_queue.get_nowait()
                if kind == "log":
                    self._append_log(payload)
                elif kind == "progress":
                    self.progress["value"] = payload
                elif kind == "done":
                    self._on_done(payload)
        except queue.Empty:
            pass
        self.root.after(100, self._poll_queue)

    def _append_log(self, msg):
        self.log_text.config(state="normal")
        self.log_text.insert("end", msg)
        self.log_text.see("end")
        self.log_text.config(state="disabled")

    def _clear_log(self):
        self.log_text.config(state="normal")
        self.log_text.delete("1.0", "end")
        self.log_text.config(state="disabled")

    def _on_done(self, ok):
        self.processing = False
        self.run_btn.config(state="normal")
        if ok:
            self.status_var.set("✓ Selesai!")
            self.progress["value"] = 100
            self.open_folder_btn.config(state="normal")
        else:
            self.status_var.set("✗ Gagal, cek log di bawah.")


def main():
    root = tk.Tk()
    try:
        style = ttk.Style()
        if "vista" in style.theme_names():
            style.theme_use("vista")
        elif "clam" in style.theme_names():
            style.theme_use("clam")
    except Exception:
        pass
    UpscalerGUI(root)
    root.mainloop()


if __name__ == "__main__":
    main()
