import json
import os

CELLS = []


def md(text):
    CELLS.append({
        "cell_type": "markdown",
        "metadata": {},
        "source": text.strip("\n").splitlines(keepends=True),
    })


def code(text):
    CELLS.append({
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": text.strip("\n").splitlines(keepends=True),
    })


md(r"""
# OpenShorts — Free Mode di Google Colab

Notebook ini menjalankan **OpenShorts versi free mode** di Google Colab (disarankan GPU **T4** gratis). Semua model yang sebelumnya memakai layanan berbayar diganti dengan model lokal atau layanan gratis:

| Fitur | Pengganti gratis |
|---|---|
| fal.ai Flux (aktor & b-roll) | SDXL **RealVisXL V4.0** lokal di GPU |
| fal.ai Kling/VEED lipsync | **Wav2Lip** lokal (fallback: still image + audio) |
| ElevenLabs TTS | **edge-tts** (Microsoft neural voices) |
| ElevenLabs dubbing | **faster-whisper + Gemini/Google Translate + edge-tts** |
| Gemini image thumbnail | frame asli video + teks (PIL) |
| Gemini moment picker | **LFM3-1.7B** lokal via llama.cpp (default, tanpa key) |
| AWS S3 | folder `output/` lokal |
| Upload-Post | opsional (free tier) atau download manual |

**Yang perlu disiapkan**

1. Runtime Colab dengan GPU: `Runtime` → `Change runtime type` → `T4 GPU`.
2. **Tidak perlu upload file.** Cell 4 otomatis menjalankan `git clone` dari <https://github.com/ardyhim/openshorts.git> ke `/content/openshorts`. Jalankan ulang cell 4 kapan pun untuk mengambil update terbaru; `.env`, `output/`, dan cache tetap aman. Opsi Upload ZIP / Google Drive hanya cadangan offline.
3. **Opsional tetapi sangat disarankan:** API key Gemini gratis dari [Google AI Studio](https://aistudio.google.com/apikey). Tanpa key, AI Shorts dan YouTube Studio tetap jalan dengan fallback lokal. **Clip Generator** otomatis memakai **LFM3 lokal** (cell 6b) atau Gemini; **manual AI selection** juga tersedia tanpa Gemini sama sekali: transkrip diekspor, kamu minta AI apa pun memilih timestamp, lalu import JSON-nya. Lihat bagian **Manual AI selection** di bawah.

**LFM3 (otomatis, tanpa Gemini):**

Cell 2 menyalakan `LFM3_ENABLED=True`. Cell 6b membangun `llama-server` dari source llama.cpp (commit yang sudah mendukung arsitektur LFM3), mengunduh GGUF `LFM3-1.7B-Q4_K_M` (~1.1 GB), menjalankan server di `127.0.0.1:8089`, lalu mengisi `.env`. Build pertama 5–20 menit; sesi berikutnya memakai build yang sama selama runtime hidup. Di dashboard, buka **advanced options → clip analysis engine** dan pilih **Local only (lfm3-1.7b)** kalau ingin memaksa LFM3. LFM3 hanya teks: layout picker dan video tanpa suara tetap butuh Gemini. Kalau build gagal, set `LFM3_ENABLED=False` lalu jalankan ulang cell 6b; Clip Generator memakai Gemini/manual.

**Manual AI selection (tanpa Gemini):**

1. Buka **advanced options** di dashboard dan centang **manual AI selection (no Gemini)**.
2. Proses video seperti biasa. Whisper membuat transkrip, lalu job berhenti di status **MANUAL AI** (belum render, tanpa panggilan Gemini).
3. Klik **download transcript** di panel kanan. File JSON berisi transkrip bertimestamp, kandidat window, durasi target, dan prompt siap pakai.
4. Berikan file itu ke ChatGPT/Claude/Gemini web/Ollama/LM Studio. Minta jawaban berupa JSON `{"clips":[{"start":..., "end":..., "title":..., "hook":...}]}`.
5. Klik **import ai result**, pilih jawaban AI. App memvalidasi, merapikan cut ke batas kata, membuang overlap, lalu merender.
6. Kalau job otomatis gagal karena kuota tetapi transkrip sudah ada, panel manual tetap muncul di status error.

> `start`/`end` memakai detik absolut. Markdown fence diperbolehkan. Import juga menerima bentuk metadata internal (`shorts`, `video_title_for_youtube_short`, `predicted_score`, dst.).

> **Python 3.13:** MediaPipe 0.10.21 tidak punya wheel untuk Python 3.13, jadi cell 5 otomatis melewatinya dan app memakai **OpenCV Haar** (`face_detection_compat.py`) untuk face tracking. Di Python <= 3.12, MediaPipe tetap dipakai. Tidak ada langkah manual.

> **Peringatan privasi:** opsi tunnel **Cloudflared** membuat dashboard bisa diakses publik tanpa login. Gunakan **Colab proxy** (default) kalau tidak ingin orang lain membuka GPU dan hasil video Anda.

> **Lisensi model:** **Wav2Lip hanya untuk riset/personal/non-komersial** (lihat README upstream). Untuk konten komersial, set `WAV2LIP_DISABLE=1` di `.env` agar pipeline memakai still image + audio, atau ganti dengan model berlisensi komersial. RealVisXL V4.0 memakai lisensi OpenRAIL++-M. **LFM3-1.7B** memakai **LFM Open License v1.0** (`lfm1.0`), bukan MIT; baca `LICENSE` di repo Hugging Face.

> **Batas Colab gratis:** sesi bisa terputus kapan saja. Simpan hasil ke Google Drive sebelum runtime mati (ada cell backup di bagian akhir).
""")

code(r"""
#@title 1. Cek runtime Colab (GPU, RAM, disk)
import os, platform, shutil, subprocess, sys

print("Python   :", sys.version.split()[0])
print("Platform :", platform.platform())
try:
    import psutil
    print(f"RAM      : {psutil.virtual_memory().total / 1e9:.1f} GB")
except Exception:
    pass
try:
    total, used, free = shutil.disk_usage("/content" if os.path.isdir("/content") else ".")
    print(f"Disk     : {free / 1e9:.1f} GB free / {total / 1e9:.1f} GB")
    if free < 35e9:
        print("⚠️  Ruang disk < 35 GB. Model SDXL + Wav2Lip butuh ~10 GB; ruang bisa habis.")
except Exception:
    pass

if shutil.which("nvidia-smi"):
    out = subprocess.run(
        ["nvidia-smi", "--query-gpu=name,memory.total,driver_version", "--format=csv,noheader"],
        capture_output=True, text=True)
    print("GPU      :", (out.stdout or out.stderr).strip())
    if out.returncode != 0:
        print("⚠️  nvidia-smi gagal; cek Runtime > Change runtime type > T4 GPU.")
else:
    print("⚠️  GPU tidak terdeteksi. Runtime masih bisa jalan, tetapi SDXL akan memakai placeholder dan Whisper memakai CPU (lambat).")
    print("    Pilih Runtime > Change runtime type > Hardware accelerator > T4 GPU, lalu jalankan ulang dari awal.")
""")

code(r'''
#@title 2. Konfigurasi utama { display-mode: "form" }
#@markdown ### Sumber project
#@markdown **Git clone** (default): ambil/update langsung dari GitHub, tanpa upload file.
#@markdown Upload ZIP / Google Drive hanya cadangan kalau GitHub tidak bisa diakses.
PROJECT_SOURCE = "Git clone" #@param ["Git clone", "Upload ZIP", "Google Drive"]
PROJECT_DIR = "/content/openshorts" #@param {type:"string"}
UPDATE_PROJECT = True #@param {type:"boolean"}
RESET_PROJECT = False #@param {type:"boolean"}

#@markdown ### Lokasi ZIP (abaikan kalau memilih Git clone)
ZIP_PATH = "/content/openshorts-free-colab.zip" #@param {type:"string"}
DRIVE_ZIP_PATH = "/content/drive/MyDrive/openshorts-free-colab.zip" #@param {type:"string"}

#@markdown ### Git clone
REPO_URL = "https://github.com/ardyhim/openshorts.git" #@param {type:"string"}
REPO_BRANCH = "main" #@param {type:"string"}

#@markdown ### API key (opsional, gratis)
#@markdown Gemini: https://aistudio.google.com/apikey — tanpa kartu kredit.
#@markdown Upload-Post: free tier ±10 upload/bulan. Kalau diisi di sini, dashboard otomatis memakai key server — cukup pilih profil yang terhubung.
GEMINI_API_KEY = "" #@param {type:"string"}
HF_TOKEN = "" #@param {type:"string"}
UPLOAD_POST_API_KEY = "" #@param {type:"string"}
ASK_GEMINI_KEY = True #@param {type:"boolean"}

#@markdown ### Model
GEMINI_MODEL = "auto" #@param {type:"string"}
FREE_IMAGE_MODEL = "SG161222/RealVisXL_V4.0" #@param {type:"string"}
IMAGE_STEPS = 20 #@param {type:"integer"}
WHISPER_MODEL = "small" #@param ["tiny", "base", "small", "medium", "large-v3", "large-v3-turbo"]

#@markdown ### Pemilihan momen dengan LFM3 (lokal, tanpa API)
#@markdown LFM3-1.7B berjalan lewat llama.cpp di GPU/CPU Colab. Build pertama
#@markdown 5–20 menit; sesi berikutnya memakai build yang sama selama runtime hidup.
#@markdown Pilih **false** untuk tetap memakai Gemini / manual AI selection saja.
LFM3_ENABLED = True #@param {type:"boolean"}
LFM3_QUANT = "Q4_K_M" #@param ["Q4_K_M", "Q5_K_M", "Q8_0", "QAD-Q4_0"]
LFM3_CONTEXT = 16384 #@param {type:"integer"}
LFM3_FORCE_CPU = False #@param {type:"boolean"}

#@markdown ### Akses dashboard
PUBLIC_TUNNEL = "Colab proxy (privat)" #@param ["Colab proxy (privat)", "Cloudflared (publik)", "Tidak perlu"]

import getpass, os

GEMINI_API_KEY = (GEMINI_API_KEY or "").strip()
HF_TOKEN = (HF_TOKEN or "").strip()
UPLOAD_POST_API_KEY = (UPLOAD_POST_API_KEY or "").strip()
GEMINI_MODEL = (GEMINI_MODEL or "auto").strip()
FREE_IMAGE_MODEL = (FREE_IMAGE_MODEL or "SG161222/RealVisXL_V4.0").strip()
IMAGE_STEPS = max(8, min(50, int(IMAGE_STEPS)))
LFM3_QUANT = (LFM3_QUANT or "Q4_K_M").strip()
if LFM3_QUANT not in ("Q4_K_M", "Q5_K_M", "Q8_0", "QAD-Q4_0"):
    LFM3_QUANT = "Q4_K_M"
LFM3_CONTEXT = max(4096, min(32768, int(LFM3_CONTEXT)))

if ASK_GEMINI_KEY and not GEMINI_API_KEY:
    GEMINI_API_KEY = getpass.getpass(
        "Gemini API key (Enter = lewati; Clip Generator bisa manual tanpa Gemini): "
    ).strip()

PROJECT_DIR = os.path.abspath(os.path.expanduser(PROJECT_DIR.strip()))
if PROJECT_DIR in ("/", "/content"):
    raise ValueError("PROJECT_DIR terlalu umum; pakai misalnya /content/openshorts")
os.makedirs(os.path.dirname(PROJECT_DIR), exist_ok=True)
ZIP_PATH = os.path.abspath(os.path.expanduser(ZIP_PATH.strip()))
DRIVE_ZIP_PATH = os.path.abspath(os.path.expanduser(DRIVE_ZIP_PATH.strip()))


def _mask(value):
    if not value:
        return "(kosong)"
    if len(value) <= 10:
        return "terisi"
    return value[:4] + "..." + value[-4:]


print("Sumber project :", PROJECT_SOURCE)
print("Project dir    :", PROJECT_DIR)
print("Gemini key     :", _mask(GEMINI_API_KEY))
print("HF token       :", _mask(HF_TOKEN))
print("Upload-Post key:", _mask(UPLOAD_POST_API_KEY))
print("Gemini model   :", GEMINI_MODEL)
print("Model gambar   :", FREE_IMAGE_MODEL, f"({IMAGE_STEPS} steps)")
print("Whisper        :", WHISPER_MODEL)
print("LFM3 lokal     :", "aktif" if LFM3_ENABLED else "nonaktif",
      f"({LFM3_QUANT}, ctx {LFM3_CONTEXT})" if LFM3_ENABLED else "")
print("Akses          :", PUBLIC_TUNNEL)
''')

code(r'''
#@title 3. Install ffmpeg + font + utilitas sistem
import os, shutil, subprocess, sys

def run(cmd, **kwargs):
    print("+", " ".join(cmd) if isinstance(cmd, list) else cmd)
    return subprocess.run(cmd, **kwargs)

APT_PACKAGES = [
    "ffmpeg", "git", "curl", "wget", "unzip", "fontconfig",
    "libgl1", "libglib2.0-0", "libsm6", "libxext6", "libxrender1", "libsndfile1",
    "fonts-liberation", "fonts-noto-core", "fonts-noto-color-emoji", "fonts-dejavu-core",
    # LFM3 runs through llama.cpp, which the notebook builds from source when
    # no release asset supports the new architecture yet.
    "cmake", "build-essential",
]

if shutil.which("apt-get"):
    env = dict(os.environ, DEBIAN_FRONTEND="noninteractive")
    run(["apt-get", "update", "-qq", "-o", "Acquire::Retries=3"], check=True)
    run(["apt-get", "install", "-y", "-qq", "--no-install-recommends", *APT_PACKAGES], check=True, env=env)
else:
    print("apt-get tidak tersedia; pastikan ffmpeg/font sudah terpasang manual.")

ffmpeg = shutil.which("ffmpeg")
if not ffmpeg:
    raise RuntimeError("ffmpeg tidak ditemukan. Jalankan ulang cell ini atau install ffmpeg manual.")
version = subprocess.run([ffmpeg, "-version"], capture_output=True, text=True).stdout.splitlines()
print("ffmpeg :", version[0] if version else ffmpeg)
print("git    :", shutil.which("git") or "TIDAK ADA")
print("curl   :", shutil.which("curl") or "TIDAK ADA")
print("cmake  :", shutil.which("cmake") or "TIDAK ADA")
_nvcc = shutil.which("nvcc") or ("/usr/local/cuda/bin/nvcc" if os.path.exists("/usr/local/cuda/bin/nvcc") else None)
print("nvcc   :", _nvcc or "TIDAK ADA (LFM3 akan build CPU; tetap jalan, lebih lambat)")


def _node_major():
    node = shutil.which("node")
    if not node:
        return 0
    try:
        out = subprocess.run([node, "--version"], capture_output=True, text=True).stdout.strip()
        return int(out.lstrip("v").split(".")[0])
    except Exception:
        return 0


node_ok = _node_major() >= 18 and shutil.which("npm")
if not node_ok:
    print("Node.js >= 18 / npm belum ada; memasang Node.js 22 dari NodeSource...")
    run("curl -fsSL https://deb.nodesource.com/setup_22.x | bash -", shell=True, check=True)
    run(["apt-get", "install", "-y", "-qq", "nodejs"], check=True)
    node_ok = _node_major() >= 18 and shutil.which("npm")
if not node_ok:
    raise RuntimeError("Node.js/npm tidak siap; dashboard tidak bisa dibuild.")
print("node   :", subprocess.run(["node", "--version"], capture_output=True, text=True).stdout.strip())
print("npm    :", subprocess.run(["npm", "--version"], capture_output=True, text=True).stdout.strip())
''')

code(r'''
#@title 4. Ambil / update project OpenShorts
import os, shutil, subprocess, zipfile
from pathlib import Path

PROJECT_DIR = os.path.abspath(PROJECT_DIR)


def looks_like_project(path):
    p = Path(path)
    return (
        (p / "app.py").is_file()
        and (p / "free_backends.py").is_file()
        and (p / "colab" / "requirements-colab.txt").is_file()
    )


def locate_root(base):
    base = Path(base)
    if looks_like_project(base):
        return base
    for app in base.rglob("app.py"):
        if looks_like_project(app.parent):
            return app.parent
    return None


def install_from_root(root):
    root = Path(root)
    if os.path.exists(PROJECT_DIR) and RESET_PROJECT:
        print("Menghapus project lama:", PROJECT_DIR)
        shutil.rmtree(PROJECT_DIR)
    if os.path.exists(PROJECT_DIR):
        if looks_like_project(PROJECT_DIR):
            print("Project sudah ada dan valid; dilewati. Set RESET_PROJECT=True untuk menimpa.")
            return
        raise RuntimeError(
            f"{PROJECT_DIR} sudah ada tetapi bukan project free mode. "
            "Set RESET_PROJECT=True atau hapus folder tersebut."
        )
    shutil.copytree(root, PROJECT_DIR)
    print("Project disalin ke:", PROJECT_DIR)


def git(*args, check=True):
    return subprocess.run(["git", "-C", PROJECT_DIR, *args], check=check)


def update_git_project():
    # Folder hasil upload ZIP lama tidak punya .git. Jadikan repo git di tempat
    # supaya update berikutnya mudah, tanpa menghapus .env/output/cache:
    # reset --hard hanya menimpa file yang dilacak git.
    if not (Path(PROJECT_DIR) / ".git").is_dir():
        print("Folder lama bukan git repo; mengubahnya menjadi clone git...")
        git("init", "-q")
    remotes = subprocess.run(
        ["git", "-C", PROJECT_DIR, "remote"],
        capture_output=True, text=True, check=True,
    ).stdout.split()
    if "origin" in remotes:
        git("remote", "set-url", "origin", REPO_URL)
    else:
        git("remote", "add", "origin", REPO_URL)

    print("Memperbarui project dari GitHub...")
    git("fetch", "--depth", "1", "origin", REPO_BRANCH)
    git("reset", "--hard", "FETCH_HEAD")
    commit = subprocess.run(
        ["git", "-C", PROJECT_DIR, "log", "-1", "--format=%h %s"],
        capture_output=True, text=True, check=True,
    ).stdout.strip()
    print("Versi terbaru:", commit)


if os.path.exists(PROJECT_DIR) and RESET_PROJECT:
    print("Menghapus project lama:", PROJECT_DIR)
    shutil.rmtree(PROJECT_DIR)

if looks_like_project(PROJECT_DIR):
    print("Project free mode sudah ada:", PROJECT_DIR)
    if PROJECT_SOURCE == "Git clone" and UPDATE_PROJECT:
        update_git_project()
    elif PROJECT_SOURCE == "Git clone":
        print("UPDATE_PROJECT=False; memakai versi yang ada saat ini.")
else:
    if os.path.exists(PROJECT_DIR):
        raise RuntimeError(
            f"{PROJECT_DIR} sudah ada tetapi bukan project free mode. "
            "Set RESET_PROJECT=True atau hapus folder tersebut."
        )

    zip_path = None
    if PROJECT_SOURCE == "Upload ZIP":
        if not os.path.exists(ZIP_PATH):
            from google.colab import files
            print("Pilih file openshorts-free-colab.zip dari komputer Anda...")
            uploaded = files.upload()
            if not uploaded:
                raise RuntimeError("Tidak ada file yang di-upload.")
            name, data = next(iter(uploaded.items()))
            print(f"Upload: {name} ({len(data) / 1e6:.1f} MB)")
            with open(ZIP_PATH, "wb") as fh:
                fh.write(data)
        zip_path = ZIP_PATH
    elif PROJECT_SOURCE == "Google Drive":
        if not os.path.isdir("/content/drive/MyDrive"):
            from google.colab import drive
            drive.mount("/content/drive")
        if not os.path.exists(DRIVE_ZIP_PATH):
            raise FileNotFoundError(
                f"ZIP tidak ditemukan di Drive: {DRIVE_ZIP_PATH}\n"
                "Upload openshorts-free-colab.zip ke My Drive, atau ganti DRIVE_ZIP_PATH."
            )
        zip_path = DRIVE_ZIP_PATH
    elif PROJECT_SOURCE == "Git clone":
        if not REPO_URL.strip():
            raise ValueError(
                "REPO_URL kosong. Isi URL repo GitHub (misalnya "
                "https://github.com/ardyhim/openshorts.git), atau pilih Upload ZIP."
            )
        subprocess.run(
            ["git", "clone", "--depth", "1", "--branch", REPO_BRANCH, REPO_URL, PROJECT_DIR],
            check=True,
        )
    else:
        raise ValueError("PROJECT_SOURCE tidak dikenal.")

    if zip_path:
        if not os.path.exists(zip_path):
            raise FileNotFoundError(f"ZIP tidak ditemukan: {zip_path}")
        tmp = "/content/_openshorts_unzip"
        shutil.rmtree(tmp, ignore_errors=True)
        print("Mengekstrak:", zip_path)
        with zipfile.ZipFile(zip_path) as zf:
            zf.extractall(tmp)
        root = locate_root(tmp)
        if root is None:
            raise RuntimeError(
                "ZIP tidak berisi app.py + free_backends.py + colab/requirements-colab.txt. "
                "Pastikan memakai openshorts-free-colab.zip yang benar."
            )
        install_from_root(root)
        shutil.rmtree(tmp, ignore_errors=True)

if not looks_like_project(PROJECT_DIR):
    raise RuntimeError(
        "Project tidak valid. Pastikan repo berisi free_backends.py dan "
        "colab/requirements-colab.txt (ardyhim/openshorts sudah berisi keduanya)."
    )

os.chdir(PROJECT_DIR)
print("\nProject siap:", PROJECT_DIR)
print("free_backends.py:", os.path.getsize(os.path.join(PROJECT_DIR, "free_backends.py")) // 1024, "KB")
print("Isi utama:", ", ".join(sorted(os.listdir(PROJECT_DIR))[:12]), "...")
''')

code(r'''
#@title 5. Install dependensi Python (torch bawaan Colab tidak diganti)
import os, subprocess, sys

os.chdir(PROJECT_DIR)
pip_env = dict(os.environ, PIP_DISABLE_PIP_VERSION_CHECK="1", PYTHONUNBUFFERED="1")


def pip(*args):
    print("pip", " ".join(args))
    subprocess.run([sys.executable, "-m", "pip", *args], check=True, env=pip_env)


# Marker di requirements memilih NumPy/OpenCV sesuai versi Python:
# <= 3.12 -> NumPy 1.26 + OpenCV 4.10 (untuk mediapipe);
# 3.13    -> NumPy 2.x + OpenCV 4.12 (Haar cascade tetap tersedia).
try:
    pip("install", "-q", "--no-warn-script-location", "-r", "colab/requirements-colab.txt")
except subprocess.CalledProcessError:
    print("\n❌ Install dependensi gagal. Lihat baris ERROR dari pip di atas.")
    print("   Pastikan cell 4 memakai kode terbaru (UPDATE_PROJECT=True), lalu ulangi cell ini.")
    print("   Konflik yang umum: py3langid versi lama butuh NumPy 2, sedangkan requirements memakai NumPy 1.x.")
    raise

# MediaPipe 0.10.21 adalah rilis terakhir yang masih punya mp.solutions
# (dipakai face tracking). Wheel-nya hanya tersedia untuk Python <= 3.12;
# Colab sekarang memakai Python 3.13, jadi di sana app otomatis memakai
# fallback OpenCV Haar cascade di face_detection_compat.py.
if sys.version_info < (3, 13):
    # Deklarasi dependensi mediapipe menarik NumPy 2 dan JAX. Kita hanya butuh
    # graph vision bawaan, jadi pasang tanpa dependensi; sisanya sudah ada.
    pip("install", "-q", "--no-warn-script-location", "--no-deps", "mediapipe==0.10.21")
    print("mediapipe  : dipasang (BlazeFace)")
else:
    print(
        "mediapipe  : dilewati di Python "
        f"{sys.version_info.major}.{sys.version_info.minor} "
        "(belum ada wheel dengan mp.solutions); memakai OpenCV Haar."
    )

# CTranslate2 (dipakai faster-whisper) memanggil dlopen("libcublas.so.12").
# Colab bisa memakai torch CUDA 13, dan library pip tidak otomatis ada di
# loader path. Pasang runtime CUDA 12 kalau belum ada; cell 10 menambahkan
# folder lib-nya ke LD_LIBRARY_PATH. Tanpa ini Whisper jatuh ke CPU.
if sys.platform.startswith("linux"):
    import ctypes, glob, site

    def _cublas12_available():
        try:
            ctypes.CDLL("libcublas.so.12")
            return True
        except OSError:
            pass
        for sp in site.getsitepackages():
            if glob.glob(os.path.join(sp, "nvidia", "cublas", "lib", "libcublas.so.12*")):
                return True
        return bool(glob.glob("/usr/local/cuda*/lib64/libcublas.so.12*"))

    if _cublas12_available():
        print("libcublas   : tersedia")
    else:
        print("libcublas   : belum ada; memasang nvidia-cublas-cu12...")
        pip("install", "-q", "--no-warn-script-location", "nvidia-cublas-cu12")

print("\nSelesai. torch/torchvision tidak diubah:")
try:
    import torch
    print("torch", torch.__version__, "| CUDA:", torch.cuda.is_available())
    if torch.cuda.is_available():
        print("GPU  ", torch.cuda.get_device_name(0))
except Exception as exc:
    print("⚠️  import torch gagal:", exc)
''')

code(r'''
#@title 6. Tulis konfigurasi free mode + deteksi model Gemini
import json, os, re, stat
from pathlib import Path

PROJECT_DIR = os.path.abspath(PROJECT_DIR)
GEMINI_API_KEY = (GEMINI_API_KEY or "").strip()
fallback_model = "gemini-2.5-flash"
chosen_model = fallback_model
thumb_model = fallback_model
candidates = []
auth_invalid = False

if GEMINI_API_KEY:
    try:
        from google import genai
        from google.genai import types

        client = genai.Client(api_key=GEMINI_API_KEY)

        def _is_chat_model(name):
            low = name.lower()
            bad = (
                "image", "imagen", "embedding", "embed", "tts", "audio", "live",
                "native", "robotics", "computer-use", "deep-research", "veo", "gemma",
            )
            return "gemini" in low and not any(b in low for b in bad)

        def _score(name):
            low = name.lower()
            score = 0.0
            if "flash-lite" in low:
                score += 20
            elif "flash" in low:
                score += 40
            elif "pro" in low:
                score += 10
            if "latest" in low:
                score += 5
            if any(x in low for x in ("preview", "experimental", "-exp")):
                score -= 4
            m = re.search(r"gemini-(\d+)(?:\.(\d+))?", low)
            if m:
                score += int(m.group(1)) * 3 + float(m.group(2) or 0) / 100
            return score

        for model in client.models.list():
            name = (getattr(model, "name", "") or "").split("/")[-1]
            if not name or not _is_chat_model(name):
                continue
            actions = getattr(model, "supported_actions", None)
            if actions and "generateContent" not in actions:
                continue
            candidates.append(name)
        candidates = sorted(set(candidates), key=lambda n: (-_score(n), n))

        requested = (GEMINI_MODEL or "auto").strip()
        if requested.lower() != "auto" and requested:
            candidates = [requested] + [c for c in candidates if c != requested]

        def _smoke(model):
            global auth_invalid
            try:
                response = client.models.generate_content(
                    model=model,
                    contents="Reply with the single word OK.",
                    config=types.GenerateContentConfig(temperature=0, max_output_tokens=16),
                )
                return True, (response.text or "").strip()[:40]
            except Exception as exc:
                msg = str(exc)
                low = msg.lower()
                if any(x in low for x in ("api key not valid", "api_key_invalid", "unauthenticated")):
                    auth_invalid = True
                return False, msg[:300]

        selected = None
        if candidates:
            print("Model Gemini yang tersedia untuk key ini:")
            for name in candidates[:12]:
                print("  -", name)
            for name in candidates[:8]:
                ok, detail = _smoke(name)
                print(f"Uji {name}: {'OK' if ok else 'gagal'} {detail}")
                if ok:
                    selected = name
                    break
                low = detail.lower()
                if any(x in low for x in ("429", "resource_exhausted", "quota", "rate limit")):
                    print("   Kuota sedang habis, tetapi model diterima. Memakai model ini.")
                    selected = name
                    break
                if auth_invalid:
                    break
        if selected:
            chosen_model = selected
        else:
            chosen_model = candidates[0] if candidates else fallback_model
            print("⚠️  Tidak ada model yang lolos uji; memakai:", chosen_model)

        # Judul/thumbnail lebih baik dengan flash penuh; pipeline momen bisa
        # memakai model pilihan pertama (biasanya flash-lite bila tersedia).
        full_flash = next(
            (n for n in candidates if "flash" in n.lower() and "lite" not in n.lower()),
            chosen_model,
        )
        thumb_model = full_flash

        if auth_invalid:
            print("❌ API key Gemini ditolak. Key dikosongkan; fitur tanpa Gemini tetap berjalan.")
            GEMINI_API_KEY = ""
    except Exception as exc:
        print("⚠️  Deteksi model Gemini gagal:", exc)
        print("   Memakai fallback:", fallback_model)
        chosen_model = fallback_model
        thumb_model = fallback_model
else:
    print("Gemini key kosong.")
    print("  - AI Shorts, YouTube Studio, thumbnail, dubbing, dan Clip Generator tanpa AI")
    print("    tetap bisa dijalankan dengan fallback lokal.")
    print("  - Clip Generator (pemilihan momen viral) otomatis memakai LFM3 lokal dari cell 6b;")
    print("    Gemini tetap opsional untuk layout/video tanpa suara. Manual AI selection juga")
    print("    tersedia: transkrip → AI eksternal → import JSON.")
    print("  - Tambahkan key lalu jalankan ulang cell ini untuk memilih model yang tersedia.")

os.environ.update({
    "FREE_MODE": "1",
    "SERVE_DASHBOARD": "1",
    "MAX_CONCURRENT_JOBS": "1",
    "JOB_RETENTION_SECONDS": "86400",
    "SOURCE_RETENTION_SECONDS": "86400",
    "OUTPUT_MAX_GB": "60",
    "UPLOADS_MAX_GB": "40",
    "QUALITY_GATE_MIN_HEIGHT": "480",
    "WHISPER_MODEL": WHISPER_MODEL,
    "WHISPER_DEVICE": "cuda",
    "WHISPER_COMPUTE": "float16",
    "TRANSCRIBE_BACKEND": "whisper",
    "ASR_GPU_CONCURRENCY": "1",
    "ASR_HOST_SLOTS": "1",
    "FREE_IMAGE_MODEL": FREE_IMAGE_MODEL,
    "FREE_IMAGE_STEPS": str(IMAGE_STEPS),
    "FREE_IMAGE_GUIDANCE": "5.0",
    "WAV2LIP_TIMEOUT": "1800",
    "FFMPEG_ENCODER": "auto",
    "TOKENIZERS_PARALLELISM": "false",
    "PYTHONUNBUFFERED": "1",
    "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True",
    "GEMINI_MODEL": chosen_model,
    "GEMINI_MODEL_THUMBNAIL": thumb_model,
    "GEMINI_MODEL_SAAS": thumb_model,
    "FREE_TOOLS_GEMINI_MODEL": thumb_model,
    "HF_HOME": os.path.join(PROJECT_DIR, ".cache", "huggingface"),
    "HUGGINGFACE_HUB_CACHE": os.path.join(PROJECT_DIR, ".cache", "huggingface", "hub"),
    "TORCH_HOME": os.path.join(PROJECT_DIR, ".cache", "torch"),
})
for folder in (
    os.environ["HF_HOME"],
    os.environ["HUGGINGFACE_HUB_CACHE"],
    os.environ["TORCH_HOME"],
):
    os.makedirs(folder, exist_ok=True)

if GEMINI_API_KEY:
    os.environ["GEMINI_API_KEY"] = GEMINI_API_KEY
else:
    os.environ.pop("GEMINI_API_KEY", None)
if HF_TOKEN:
    os.environ["HF_TOKEN"] = HF_TOKEN
if UPLOAD_POST_API_KEY:
    os.environ["UPLOAD_POST_API_KEY"] = UPLOAD_POST_API_KEY

# Nilai yang ditulis ke .env agar server bisa dijalankan ulang tanpa mengulang
# seluruh notebook. File ini gitignored dan hanya ada di runtime Colab.
managed = {
    "FREE_MODE": "1",
    "SERVE_DASHBOARD": "1",
    "MAX_CONCURRENT_JOBS": "1",
    "JOB_RETENTION_SECONDS": "86400",
    "SOURCE_RETENTION_SECONDS": "86400",
    "OUTPUT_MAX_GB": "60",
    "UPLOADS_MAX_GB": "40",
    "QUALITY_GATE_MIN_HEIGHT": "480",
    "WHISPER_MODEL": WHISPER_MODEL,
    "WHISPER_DEVICE": "cuda",
    "WHISPER_COMPUTE": "float16",
    "TRANSCRIBE_BACKEND": "whisper",
    "ASR_GPU_CONCURRENCY": "1",
    "ASR_HOST_SLOTS": "1",
    "FREE_IMAGE_MODEL": FREE_IMAGE_MODEL,
    "FREE_IMAGE_STEPS": str(IMAGE_STEPS),
    "FREE_IMAGE_GUIDANCE": "5.0",
    "WAV2LIP_TIMEOUT": "1800",
    "FFMPEG_ENCODER": "auto",
    "TOKENIZERS_PARALLELISM": "false",
    "PYTHONUNBUFFERED": "1",
    "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True",
    "GEMINI_MODEL": chosen_model,
    "GEMINI_MODEL_THUMBNAIL": thumb_model,
    "GEMINI_MODEL_SAAS": thumb_model,
    "FREE_TOOLS_GEMINI_MODEL": thumb_model,
    "HF_HOME": os.environ["HF_HOME"],
    "HUGGINGFACE_HUB_CACHE": os.environ["HUGGINGFACE_HUB_CACHE"],
    "TORCH_HOME": os.environ["TORCH_HOME"],
    "GEMINI_API_KEY": GEMINI_API_KEY,
    "HF_TOKEN": HF_TOKEN,
    "UPLOAD_POST_API_KEY": UPLOAD_POST_API_KEY,
    # Cell 6b writes these only after the LFM3 server is confirmed healthy.
    # Listing them here makes a rerun of this cell drop a stale local-LLM
    # config instead of leaving the dashboard pointed at a dead port.
    "LLM_PROVIDER": "",
    "LLM_BASE_URL": "",
    "LLM_MODEL": "",
    "LLM_API_KEY": "",
    "LLM_TIMEOUT": "",
    "LLM_TEMPERATURE": "",
    "LLM_TOP_K": "",
    "LLM_REPETITION_PENALTY": "",
    "LLM_MAX_TOKENS": "",
    "LLM_DISABLE_THINKING": "",
    "LLM_CHAT_TEMPLATE_KWARGS": "",
    "LLM_SCORE_BATCH": "",
}


def _dotenv_value(value):
    value = str(value)
    if re.search(r"[\s#\"']", value):
        return json.dumps(value)
    return value


env_path = Path(PROJECT_DIR) / ".env"
lines = []
if env_path.exists():
    for raw in env_path.read_text(encoding="utf-8", errors="replace").splitlines():
        stripped = raw.strip()
        if stripped and not stripped.startswith("#") and "=" in stripped:
            key = stripped.split("=", 1)[0].strip()
            if key in managed:
                continue
        lines.append(raw.rstrip())
if not lines:
    lines.append("# Generated by colab/OpenShorts_Free_Colab.ipynb")
for key, value in managed.items():
    if value is not None and str(value) != "":
        lines.append(f"{key}={_dotenv_value(value)}")
env_path.write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")
try:
    env_path.chmod(0o600)
except Exception:
    pass

print("\nKonfigurasi ditulis ke:", env_path)
print("GEMINI_MODEL           :", chosen_model, "(thumbnails/SaaS:", thumb_model + ")")
print("FREE_MODE              :", os.environ.get("FREE_MODE"))
print("WHISPER                :", WHISPER_MODEL, "cuda/float16")
print("HF_HOME                :", os.environ.get("HF_HOME"))
''')

code(r'''
#@title 6b. Siapkan LFM3 lokal (llama.cpp + GGUF)
#@markdown Build pertama bisa 5–20 menit. Server LFM3 tetap hidup di background;
#@markdown dashboard otomatis menampilkan pilihan **Local only (LFM3)** di advanced options.
import glob, json, os, shutil, signal, site, subprocess, time, urllib.error, urllib.request
from pathlib import Path

PROJECT_DIR = os.path.abspath(PROJECT_DIR)

# The CUDA build links against cuBLAS. Colab may keep it in a pip nvidia
# package instead of the loader's default path; add every candidate before
# building/starting llama-server (cell 10 repeats this for the app process).
cuda_lib_dirs = []
for _sp in site.getsitepackages():
    cuda_lib_dirs += glob.glob(os.path.join(_sp, "nvidia", "*", "lib"))
cuda_lib_dirs += glob.glob("/usr/local/cuda*/lib64")
cuda_lib_dirs = [d for d in cuda_lib_dirs if os.path.isdir(d)]
if cuda_lib_dirs:
    _old_ld = os.environ.get("LD_LIBRARY_PATH", "")
    os.environ["LD_LIBRARY_PATH"] = ":".join(
        cuda_lib_dirs + ([_old_ld] if _old_ld else []))
    print("CUDA libs:", ", ".join(cuda_lib_dirs))
LFM3_PORT = 8089
LFM3_ALIAS = "lfm3-1.7b"
# Commit setelah arsitektur LFM3 masuk ke llama.cpp (PR #31252). Belum ada
# release binary yang lebih baru dari b11936, jadi build dari source adalah
# jalur yang paling pasti. Ganti SHA ini kalau ingin versi llama.cpp lain.
LFM3_LLAMA_REF = "1b4554eff124a8f0033bc09b3f708a5e51cbda15"
LFM3_MODELS = {
    "Q4_K_M": ("LFM3-1.7B-Q4_K_M.gguf", 1099627808),
    "Q5_K_M": ("LFM3-1.7B-Q5_K_M.gguf", 1258663200),
    "Q8_0": ("LFM3-1.7B-Q8_0.gguf", 1846238496),
    "QAD-Q4_0": ("LFM3-1.7B-QAD-Q4_0.gguf", 1049615680),
}
LFM3_KEYS = (
    "LLM_PROVIDER", "LLM_BASE_URL", "LLM_MODEL", "LLM_API_KEY",
    "LLM_TIMEOUT", "LLM_TEMPERATURE", "LLM_TOP_K",
    "LLM_REPETITION_PENALTY", "LLM_MAX_TOKENS", "LLM_DISABLE_THINKING",
    "LLM_CHAT_TEMPLATE_KWARGS", "LLM_SCORE_BATCH",
)
ENV_PATH = Path(PROJECT_DIR) / ".env"
PID_PATH = Path(PROJECT_DIR) / ".lfm3.pid"
MODEL_MARKER = Path(PROJECT_DIR) / ".lfm3-model"
SERVER_LOG = Path("/content/lfm3-server.log")
BUILD_LOG = Path("/content/llama-build.log")
SRC_DIR = Path("/content/llama.cpp-src")
BUILD_DIR = Path("/content/llama.cpp-build")
MODEL_DIR = Path(PROJECT_DIR).parent / "openshorts-models"
BUILD_MARKER = BUILD_DIR / ".lfm3-build"


def write_env(updates):
    """Update .env without disturbing keys managed by other cells."""
    keys = set(updates)
    lines = []
    if ENV_PATH.exists():
        for raw in ENV_PATH.read_text(encoding="utf-8", errors="replace").splitlines():
            stripped = raw.strip()
            if stripped and not stripped.startswith("#") and "=" in stripped:
                key = stripped.split("=", 1)[0].strip()
                if key in keys:
                    continue
            lines.append(raw.rstrip())
    for key, value in updates.items():
        if value is not None and str(value) != "":
            lines.append(f"{key}={value}")
    ENV_PATH.write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")
    try:
        ENV_PATH.chmod(0o600)
    except Exception:
        pass


def clear_llm_env():
    for key in LFM3_KEYS:
        os.environ.pop(key, None)
    write_env({key: "" for key in LFM3_KEYS})


def stop_lfm3():
    if not PID_PATH.exists():
        return
    try:
        pid = int(PID_PATH.read_text().strip())
    except Exception:
        PID_PATH.unlink(missing_ok=True)
        return
    try:
        os.killpg(pid, signal.SIGTERM)
        print(f"Menhentikan LFM3 lama (PID {pid})...")
    except ProcessLookupError:
        pass
    except Exception:
        try:
            os.kill(pid, signal.SIGTERM)
        except Exception:
            pass
    for _ in range(20):
        try:
            os.kill(pid, 0)
        except ProcessLookupError:
            break
        time.sleep(0.5)
    else:
        try:
            os.killpg(pid, signal.SIGKILL)
        except Exception:
            try:
                os.kill(pid, signal.SIGKILL)
            except Exception:
                pass
    PID_PATH.unlink(missing_ok=True)


def lfm3_models():
    try:
        with urllib.request.urlopen(
                f"http://127.0.0.1:{LFM3_PORT}/v1/models", timeout=2) as resp:
            data = json.load(resp)
        return [m.get("id") for m in data.get("data") or []]
    except Exception:
        return []


def tail(path, lines=50):
    try:
        text = Path(path).read_text(encoding="utf-8", errors="replace")
    except Exception:
        return "(log belum ada)"
    return "\n".join(text.splitlines()[-lines:])


def download(url, dest, resume=False):
    dest = Path(dest)
    dest.parent.mkdir(parents=True, exist_ok=True)
    cmd = ["curl", "-L", "--fail", "--retry", "3", "--retry-delay", "2",
           "-o", str(dest)]
    if resume:
        cmd[2:2] = ["-C", "-"]
    # Never send the Hugging Face token to GitHub (the llama.cpp tarball uses
    # this same helper).
    if HF_TOKEN and "huggingface.co" in url:
        cmd += ["-H", f"Authorization: Bearer {HF_TOKEN}"]
    cmd.append(url)
    print("+", " ".join(cmd[:9]), "...")
    subprocess.run(cmd, check=True)


def valid_tarball(path):
    """True only when tar can read the archive end to end.

    A download interrupted after the size threshold would otherwise be reused
    forever and fail extraction on every rerun."""
    try:
        result = subprocess.run(
            ["tar", "-tzf", str(path)],
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, timeout=120)
        return result.returncode == 0
    except Exception:
        return False


def nvcc_path():
    if os.path.exists("/usr/local/cuda/bin/nvcc"):
        return "/usr/local/cuda/bin/nvcc"
    return shutil.which("nvcc")


def gpu_arch():
    try:
        import torch
        if torch.cuda.is_available():
            major, minor = torch.cuda.get_device_capability(0)
            return f"{major}{minor}"
    except Exception:
        pass
    try:
        out = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=compute_cap", "--format=csv,noheader"],
            text=True, stderr=subprocess.DEVNULL).strip().splitlines()[0]
        return out.replace(".", "").strip()
    except Exception:
        return ""


def build_llama(use_cuda):
    mode = "cuda" if use_cuda else "cpu"
    wanted = f"{LFM3_LLAMA_REF}:{mode}"
    if BUILD_MARKER.exists() and BUILD_MARKER.read_text(
            encoding="utf-8", errors="replace").strip() == wanted:
        for candidate in (BUILD_DIR / "bin" / "llama-server",
                          BUILD_DIR / "bin" / "Release" / "llama-server"):
            if candidate.exists():
                return candidate
    if BUILD_DIR.exists():
        current = ""
        try:
            current = BUILD_MARKER.read_text(encoding="utf-8", errors="replace").strip()
        except Exception:
            pass
        if current != wanted:
            shutil.rmtree(BUILD_DIR, ignore_errors=True)
    BUILD_DIR.mkdir(parents=True, exist_ok=True)
    nvcc = nvcc_path()
    args = [
        "cmake", "-S", str(SRC_DIR), "-B", str(BUILD_DIR),
        "-DCMAKE_BUILD_TYPE=Release",
        "-DLLAMA_BUILD_TESTS=OFF",
        "-DLLAMA_BUILD_EXAMPLES=OFF",
        "-DLLAMA_BUILD_TOOLS=ON",
        "-DLLAMA_BUILD_SERVER=ON",
        f"-DGGML_CUDA={'ON' if use_cuda else 'OFF'}",
    ]
    if use_cuda:
        arch = gpu_arch() or "75"
        args.append(f"-DCMAKE_CUDA_ARCHITECTURES={arch}")
        if nvcc:
            args.append(f"-DCMAKE_CUDA_COMPILER={nvcc}")
        print(f"Build llama.cpp CUDA (sm_{arch})...")
    else:
        print("Build llama.cpp CPU (LFM3 tetap jalan, lebih lambat)...")
    jobs = max(1, os.cpu_count() or 2)
    with open(BUILD_LOG, "a", encoding="utf-8") as log:
        log.write("\n" + "=" * 70 + "\n")
        log.write(" ".join(args) + "\n")
        log.flush()
        subprocess.run(args, stdout=log, stderr=subprocess.STDOUT, check=True)
        subprocess.run(
            ["cmake", "--build", str(BUILD_DIR), "--target", "llama-server",
             "--config", "Release", "-j", str(jobs)],
            stdout=log, stderr=subprocess.STDOUT, check=True)
    for candidate in (BUILD_DIR / "bin" / "llama-server",
                      BUILD_DIR / "bin" / "Release" / "llama-server"):
        if candidate.exists():
            candidate.chmod(0o755)
            BUILD_MARKER.write_text(wanted, encoding="utf-8")
            return candidate
    raise RuntimeError("llama-server tidak ditemukan setelah build")


def post_chat(extra, use_format=True):
    body = {
        "model": LFM3_ALIAS,
        "messages": [{"role": "user",
                      "content": 'Reply with exactly {"ok": true} and nothing else.'}],
        "temperature": 0.1,
        "max_tokens": 64,
        "stream": False,
    }
    if use_format:
        body["response_format"] = {"type": "json_object"}
    body.update(extra)
    req = urllib.request.Request(
        f"http://127.0.0.1:{LFM3_PORT}/v1/chat/completions",
        data=json.dumps(body).encode("utf-8"),
        headers={"Content-Type": "application/json",
                 "Authorization": "Bearer ollama"},
    )
    try:
        with urllib.request.urlopen(req, timeout=180) as resp:
            data = json.load(resp)
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")[:600]
        raise RuntimeError(f"HTTP {exc.code}: {detail}") from exc
    return (data.get("choices") or [{}])[0].get("message") or {}


def setup_lfm3():
    if not LFM3_ENABLED:
        stop_lfm3()
        clear_llm_env()
        print("LFM3 nonaktif: Clip Generator memakai Gemini atau manual AI selection.")
        return False

    clear_llm_env()
    running = lfm3_models()
    # Include the source ref and the CPU preference: changing either must
    # restart the server, not silently keep the old binary/model.
    build_pref = "cpu" if LFM3_FORCE_CPU else "auto"
    wanted_model = (
        f"{LFM3_LLAMA_REF[:12]}:{LFM3_QUANT}:{LFM3_CONTEXT}:{build_pref}")
    same_model = False
    try:
        same_model = MODEL_MARKER.read_text(encoding="utf-8").strip() == wanted_model
    except Exception:
        pass
    if running and not same_model:
        print("Server LFM3 yang lama memakai model/context berbeda; restart...")
        stop_lfm3()
        running = []
    if not running:
        stop_lfm3()

        tar_path = Path(f"/content/llama.cpp-src-{LFM3_LLAMA_REF[:12]}.tar.gz")
        if (not tar_path.exists() or tar_path.stat().st_size < 1_000_000
                or not valid_tarball(tar_path)):
            if tar_path.exists():
                tar_path.unlink()
            print("Mengunduh source llama.cpp...")
            download(
                f"https://github.com/ggml-org/llama.cpp/archive/{LFM3_LLAMA_REF}.tar.gz",
                tar_path)
            if not valid_tarball(tar_path):
                raise RuntimeError(
                    f"Source llama.cpp tidak lengkap: {tar_path}. "
                    "Hapus file itu lalu jalankan ulang cell ini.")
        source_marker = SRC_DIR / ".lfm3-ref"
        try:
            source_ref = source_marker.read_text(encoding="utf-8").strip()
        except Exception:
            source_ref = ""
        if (source_ref != LFM3_LLAMA_REF
                or not (SRC_DIR / "CMakeLists.txt").exists()
                or not (SRC_DIR / "src" / "models" / "lfm3.cpp").exists()):
            print("Mengekstrak source llama.cpp...")
            shutil.rmtree(SRC_DIR, ignore_errors=True)
            SRC_DIR.mkdir(parents=True, exist_ok=True)
            subprocess.run(
                ["tar", "-xzf", str(tar_path), "-C", str(SRC_DIR), "--strip-components=1"],
                check=True)
            if not (SRC_DIR / "src" / "models" / "lfm3.cpp").exists():
                raise RuntimeError(
                    "Source llama.cpp tidak berisi dukungan LFM3; "
                    "periksa LFM3_LLAMA_REF di cell 2.")
            source_marker.write_text(LFM3_LLAMA_REF, encoding="utf-8")

        model_name, model_size = LFM3_MODELS[LFM3_QUANT]
        model_path = MODEL_DIR / model_name
        if not model_path.exists() or model_path.stat().st_size != model_size:
            if model_path.exists() and model_path.stat().st_size > model_size:
                # A stale/oversized file cannot be resumed; curl would reject
                # the range request. Start over instead of failing forever.
                model_path.unlink()
            print(f"Mengunduh {model_name} (~{model_size / 1e9:.2f} GB)...")
            download(
                f"https://huggingface.co/LiquidAI/LFM3-1.7B-GGUF/resolve/main/{model_name}",
                model_path, resume=True)
        actual = model_path.stat().st_size if model_path.exists() else 0
        if actual != model_size:
            raise RuntimeError(
                f"Ukuran model tidak cocok: {actual} != {model_size}. "
                f"Hapus {model_path} lalu jalankan ulang cell ini.")

        use_cuda = bool(not LFM3_FORCE_CPU and gpu_arch() and nvcc_path())
        if not LFM3_FORCE_CPU and gpu_arch() and not nvcc_path():
            print("nvcc tidak ditemukan; memakai build CPU (lebih lambat).")
        try:
            server = build_llama(use_cuda)
        except Exception:
            if not use_cuda:
                raise
            print("Build CUDA gagal; fallback ke build CPU...")
            print(tail(BUILD_LOG))
            shutil.rmtree(BUILD_DIR, ignore_errors=True)
            server = build_llama(False)
            use_cuda = False

        # CMake usually bakes a build-tree RUNPATH into the binary, but keep
        # the shared libs findable even if the build was configured without it.
        server_lib_dir = str(Path(server).parent)
        _ld = os.environ.get("LD_LIBRARY_PATH", "")
        if server_lib_dir not in _ld.split(":"):
            os.environ["LD_LIBRARY_PATH"] = ":".join(
                [server_lib_dir] + ([_ld] if _ld else []))

        cmd = [
            str(server), "-m", str(model_path),
            "-c", str(LFM3_CONTEXT),
            "--host", "127.0.0.1", "--port", str(LFM3_PORT),
            "--jinja", "--alias", LFM3_ALIAS,
            "--threads", str(max(1, os.cpu_count() or 2)),
        ]
        if use_cuda:
            cmd += ["-ngl", "99"]
        print("Menjalankan:", " ".join(cmd))
        SERVER_LOG.write_text("", encoding="utf-8")
        log_handle = open(SERVER_LOG, "ab", buffering=0)
        proc = subprocess.Popen(
            cmd, stdout=log_handle, stderr=subprocess.STDOUT,
            start_new_session=True, env=os.environ.copy())
        PID_PATH.write_text(str(proc.pid), encoding="utf-8")
        deadline = time.time() + 300
        while time.time() < deadline:
            if proc.poll() is not None:
                raise RuntimeError(
                    f"llama-server berhenti (kode {proc.returncode}).\n"
                    + tail(SERVER_LOG, 60))
            if LFM3_ALIAS in lfm3_models():
                break
            time.sleep(2)
        else:
            raise RuntimeError(
                "llama-server tidak siap dalam 5 menit.\n" + tail(SERVER_LOG, 60))
        MODEL_MARKER.write_text(wanted_model, encoding="utf-8")

    # LFM3's template reads enable_thinking. Verify the server accepts the
    # per-request kwarg before writing LLM_DISABLE_THINKING to .env: a server
    # that rejects it would fail every moment-picker call.
    disable_thinking = True
    try:
        post_chat({"chat_template_kwargs": {"enable_thinking": False}},
                  use_format=False)
    except Exception as exc:
        detail = str(exc).lower()
        if "chat_template_kwargs" in detail or "enable_thinking" in detail:
            print("⚠️ Server menolak chat_template_kwargs; thinking akan di-strip dari output.")
            disable_thinking = False
        else:
            print("⚠️ Uji chat LFM3 gagal (server tetap dipakai):", exc)

    try:
        extra = ({"chat_template_kwargs": {"enable_thinking": False}}
                 if disable_thinking else {})
        message = post_chat(extra, use_format=True)
        answer = (message.get("content") or "").strip()
        print("Self-test LFM3:", answer[:180] or "(kosong)")
    except Exception as exc:
        print("⚠️ Uji JSON LFM3 gagal (app punya fallback response_format):", exc)

    updates = {
        "LLM_PROVIDER": "openai",
        "LLM_BASE_URL": f"http://127.0.0.1:{LFM3_PORT}/v1",
        "LLM_MODEL": LFM3_ALIAS,
        "LLM_API_KEY": "ollama",
        "LLM_TIMEOUT": "900",
        "LLM_TEMPERATURE": "0.1",
        "LLM_TOP_K": "50",
        "LLM_REPETITION_PENALTY": "1.1",
        "LLM_MAX_TOKENS": "4096",
        "LLM_SCORE_BATCH": "3",
    }
    if disable_thinking:
        updates["LLM_DISABLE_THINKING"] = "1"
    else:
        os.environ.pop("LLM_DISABLE_THINKING", None)
    write_env(updates)
    os.environ.update(updates)
    print("\n✅ LFM3 siap:", updates["LLM_BASE_URL"], "| model:", LFM3_ALIAS)
    print("   Dashboard → advanced options → clip analysis engine → Local only (LFM3).")
    print("   LFM3 hanya untuk pemilihan momen; layout/silent-video tetap butuh Gemini.")
    return True


try:
    LFM3_READY = setup_lfm3()
except Exception as exc:
    print("\n❌ LFM3 gagal disiapkan:", exc)
    print("\n--- build log ---")
    print(tail(BUILD_LOG))
    print("\n--- server log ---")
    print(tail(SERVER_LOG))
    stop_lfm3()
    clear_llm_env()
    LFM3_READY = False
    print("\nClip Generator tetap bisa memakai Gemini atau manual AI selection.")
    print("Perbaiki error di atas lalu jalankan ulang cell ini.")
''')

code(r'''
#@title 7. Build dashboard (Vite)
import os, shutil, subprocess

dash = os.path.join(PROJECT_DIR, "dashboard")
if not os.path.isdir(dash):
    raise RuntimeError("Folder dashboard/ tidak ditemukan.")

build_env = dict(os.environ, NODE_OPTIONS="--max-old-space-size=4096", CI="1")

if os.path.exists(os.path.join(dash, "package-lock.json")):
    print("npm ci...")
    subprocess.run(["npm", "ci", "--no-audit", "--no-fund"], cwd=dash, check=True, env=build_env)
else:
    print("package-lock.json tidak ada; npm install...")
    subprocess.run(["npm", "install", "--no-audit", "--no-fund"], cwd=dash, check=True, env=build_env)

print("npm run build...")
subprocess.run(["npm", "run", "build"], cwd=dash, check=True, env=build_env)

index = os.path.join(dash, "dist", "index.html")
if not os.path.exists(index):
    raise RuntimeError("Build selesai tetapi dashboard/dist/index.html tidak ada.")
size_mb = sum(
    os.path.getsize(os.path.join(root, f))
    for root, _, files in os.walk(os.path.join(dash, "dist"))
    for f in files
) / 1e6
print(f"Dashboard siap: {index} ({size_mb:.1f} MB)")
''')

code(r'''
#@title 8. (Opsional) Unduh model lokal sekarang
PRELOAD_WAV2LIP = True #@param {type:"boolean"}
PRELOAD_SDXL = False #@param {type:"boolean"}

import os, shutil

if not PRELOAD_WAV2LIP and not PRELOAD_SDXL:
    print("Dilewati. Model akan diunduh otomatis saat fitur pertama kali dipakai.")
else:
    os.chdir(PROJECT_DIR)
    import free_backends

    if PRELOAD_WAV2LIP:
        print("Menyiapkan Wav2Lip (repo + checkpoint ~520 MB)...")
        ok = free_backends.ensure_wav2lip()
        print("Wav2Lip siap:", ok)
        if not ok:
            print("  (Tidak masalah: AI Shorts akan memakai fallback still-image + audio.)")

    if PRELOAD_SDXL:
        print("Mengunduh + memuat RealVisXL V4.0 (~7 GB). Ini bisa 5-15 menit...")
        try:
            pipe = free_backends._load_image_pipeline()
            print("SDXL siap:", bool(pipe))
        finally:
            free_backends.release_image_pipeline()

    free = shutil.disk_usage(PROJECT_DIR).free / 1e9
    print(f"Sisa disk: {free:.1f} GB")
''')

code(r'''
#@title 9. Uji cepat instalasi
import json, os, subprocess, sys, urllib.request

check_code = r"""
import sys
import numpy as np
print("numpy      :", np.__version__)
print("python     :", sys.version.split()[0])
if sys.version_info < (3, 13):
    assert np.__version__.startswith("1."), "NumPy harus 1.x untuk mediapipe 0.10.21"

import cv2
print("opencv     :", cv2.__version__)

import torch
print("torch      :", torch.__version__, "| CUDA:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("gpu        :", torch.cuda.get_device_name(0))

from face_detection_compat import FaceDetection
detector = FaceDetection(model_selection=1, min_detection_confidence=0.5)
print("face detect:", detector.backend)
result = detector.process(np.zeros((240, 320, 3), dtype=np.uint8))
print("face detect: OK (deteksi pada frame kosong:", len(result.detections or []), "wajah)")

import edge_tts, diffusers, transformers, faster_whisper
print("free stack : OK (edge-tts, diffusers, transformers, faster-whisper)")

import free_backends, thumbnail, saasshorts
print("project    : OK (free_backends, thumbnail, saasshorts)")
"""
print("== Uji library ==")
subprocess.run([sys.executable, "-c", check_code], cwd=PROJECT_DIR, env=os.environ.copy(), check=True)

print("\n== Uji import server ==")
subprocess.run(
    [sys.executable, "-c", "import app; print('app import: OK')"],
    cwd=PROJECT_DIR, env=os.environ.copy(), check=True,
)

print("\n== Uji LFM3 ==")
if globals().get("LFM3_READY"):
    base = os.environ.get("LLM_BASE_URL", "")
    try:
        with urllib.request.urlopen(base.rstrip("/") + "/models", timeout=10) as resp:
            models = json.load(resp).get("data") or []
        print("LFM3       : OK", [m.get("id") for m in models])
        print("LLM model  :", os.environ.get("LLM_MODEL"))
        print("LLM timeout:", os.environ.get("LLM_TIMEOUT"), "detik")
    except Exception as exc:
        print("LFM3       : GAGAL", exc)
        print("Jalankan ulang cell 6b, atau pakai Gemini / manual AI selection.")
else:
    print("LFM3       : tidak aktif (Gemini / manual AI selection)")
''')

code(r'''
#@title 10. Jalankan server OpenShorts
import glob, json, os, signal, site, subprocess, sys, time, urllib.request
from pathlib import Path

PROJECT_DIR = os.path.abspath(PROJECT_DIR)

# CTranslate2 mencari libcublas.so.12 lewat loader, bukan lewat sys.path.
# Tambahkan folder lib pip NVIDIA ke LD_LIBRARY_PATH; env ini diwariskan ke
# server di bawah dan ke subprocess job (main.py).
cuda_lib_dirs = []
for sp in site.getsitepackages():
    cuda_lib_dirs += glob.glob(os.path.join(sp, "nvidia", "*", "lib"))
cuda_lib_dirs += glob.glob("/usr/local/cuda*/lib64")
cuda_lib_dirs = [d for d in cuda_lib_dirs if os.path.isdir(d)]
if cuda_lib_dirs:
    old_ld = os.environ.get("LD_LIBRARY_PATH", "")
    os.environ["LD_LIBRARY_PATH"] = ":".join(
        cuda_lib_dirs + ([old_ld] if old_ld else []))
    print("CUDA libs:", ", ".join(cuda_lib_dirs))

pid_path = Path(PROJECT_DIR) / ".server.pid"
log_path = Path(PROJECT_DIR) / "server.log"


def stop_pidfile(path, label):
    path = Path(path)
    if not path.exists():
        return
    try:
        pid = int(path.read_text().strip())
    except Exception:
        path.unlink(missing_ok=True)
        return
    try:
        os.killpg(pid, signal.SIGTERM)
        print(f"Menhentikan {label} lama (PID {pid})...")
    except ProcessLookupError:
        pass
    except Exception:
        try:
            os.kill(pid, signal.SIGTERM)
        except Exception:
            pass
    for _ in range(10):
        try:
            os.kill(pid, 0)
        except ProcessLookupError:
            break
        time.sleep(0.5)
    else:
        try:
            os.killpg(pid, signal.SIGKILL)
        except Exception:
            try:
                os.kill(pid, signal.SIGKILL)
            except Exception:
                pass
    path.unlink(missing_ok=True)


stop_pidfile(pid_path, "server")

cmd = [
    sys.executable, "-m", "uvicorn", "app:app",
    "--host", "0.0.0.0", "--port", "8000",
    "--proxy-headers", "--forwarded-allow-ips", "*",
    "--timeout-graceful-shutdown", "15",
    "--log-level", "info",
]
print("Menjalankan:", " ".join(cmd))
server_log = open(log_path, "w", encoding="utf-8", buffering=1)
server_proc = subprocess.Popen(
    cmd,
    cwd=PROJECT_DIR,
    env=os.environ.copy(),
    stdout=server_log,
    stderr=subprocess.STDOUT,
    start_new_session=True,
)
pid_path.write_text(str(server_proc.pid), encoding="utf-8")
print("PID:", server_proc.pid, "| log:", log_path)


def tail_log(limit=20000):
    if not log_path.exists():
        return "(log belum ada)"
    with open(log_path, "rb") as fh:
        fh.seek(max(0, log_path.stat().st_size - limit))
        return fh.read().decode("utf-8", errors="replace")


ready = False
last_error = ""
for i in range(180):
    if server_proc.poll() is not None:
        last_error = f"Server berhenti dengan kode {server_proc.returncode}."
        break
    try:
        with urllib.request.urlopen("http://127.0.0.1:8000/health/ready", timeout=2) as resp:
            if 200 <= resp.status < 300:
                ready = True
                break
    except Exception as exc:
        last_error = str(exc)
    time.sleep(1)

if not ready:
    print("\n--- 80 baris terakhir server.log ---")
    print("\n".join(tail_log().splitlines()[-80:]))
    raise RuntimeError(f"Server tidak siap. Error terakhir: {last_error}")

print("Server siap.\n")
with urllib.request.urlopen("http://127.0.0.1:8000/api/config", timeout=10) as resp:
    cfg = json.load(resp)
for key in ("billingEnabled", "freeMode", "freeTts", "freeImages", "freeDubbing",
            "serverGemini", "manualAnalysis", "localLlm"):
    print(f"  {key:15}: {cfg.get(key)}")
if not cfg.get("freeMode"):
    raise RuntimeError("freeMode tidak aktif. Periksa .env / cell 6.")
try:
    with urllib.request.urlopen("http://127.0.0.1:8000/", timeout=10) as resp:
        html = resp.read(200).decode("utf-8", errors="replace")
    print("  dashboard      :", "OK" if ("<html" in html.lower() or "<!doctype" in html.lower()) else "tidak terdeteksi")
except Exception as exc:
    print("  dashboard      : gagal dimuat:", exc)
''')

code(r'''
#@title 11. Buka dashboard
import os, re, subprocess, time
from pathlib import Path
from IPython.display import display, HTML

PROJECT_DIR = os.path.abspath(PROJECT_DIR)
url = "http://127.0.0.1:8000"


def colab_proxy_url():
    from google.colab.output import eval_js
    return str(eval_js("google.colab.kernel.proxyPort(8000)")).rstrip("/")


def show_url(target, note=""):
    print("URL dashboard:", target)
    if note:
        print(note)
    display(HTML(
        f'<div style="font-size:16px;padding:14px 18px;border:1px solid #d0d7de;'
        f'border-radius:10px;background:#f6f8fa;margin:8px 0">'
        f'<a href="{target}" target="_blank" rel="noopener" '
        f'style="font-weight:700;font-size:18px">Buka OpenShorts →</a>'
        f'<div style="margin-top:6px;font-size:12px;color:#57606a;word-break:break-all">{target}</div>'
        f'</div>'
    ))


if PUBLIC_TUNNEL.startswith("Colab proxy"):
    url = colab_proxy_url()
    show_url(url, "URL ini memakai proxy Colab dan hanya hidup selama runtime aktif.")
    try:
        from IPython.display import IFrame
        display(IFrame(url, width="100%", height=820))
    except Exception:
        pass
elif PUBLIC_TUNNEL.startswith("Cloudflared"):
    cf = "/usr/local/bin/cloudflared"
    if not os.path.exists(cf) or os.path.getsize(cf) < 1_000_000:
        print("Mengunduh cloudflared...")
        subprocess.run([
            "curl", "-fsSL", "-o", cf,
            "https://github.com/cloudflare/cloudflared/releases/latest/download/cloudflared-linux-amd64",
        ], check=True)
        os.chmod(cf, 0o755)

    cf_pid_path = Path(PROJECT_DIR) / ".cloudflared.pid"
    if cf_pid_path.exists():
        try:
            import signal
            old = int(cf_pid_path.read_text().strip())
            os.killpg(old, signal.SIGTERM)
        except Exception:
            pass
        cf_pid_path.unlink(missing_ok=True)

    cf_log_path = Path(PROJECT_DIR) / "cloudflared.log"
    cf_log = open(cf_log_path, "w", encoding="utf-8", buffering=1)
    cf_proc = subprocess.Popen(
        [cf, "tunnel", "--no-autoupdate", "--protocol", "http2", "--url", "http://127.0.0.1:8000"],
        stdout=cf_log, stderr=subprocess.STDOUT, start_new_session=True,
    )
    cf_pid_path.write_text(str(cf_proc.pid), encoding="utf-8")

    public_url = None
    for _ in range(60):
        text = cf_log_path.read_text(encoding="utf-8", errors="replace") if cf_log_path.exists() else ""
        match = re.search(r"https://[a-z0-9-]+\.trycloudflare\.com", text)
        if match:
            public_url = match.group(0)
            break
        if cf_proc.poll() is not None:
            break
        time.sleep(1)

    if public_url:
        show_url(public_url)
        display(HTML(
            '<div style="padding:12px 16px;border:1px solid #d1242f;border-radius:10px;'
            'background:#ffebe9;color:#82071e;margin:8px 0">'
            '<b>URL publik, tanpa login.</b> Siapa pun yang punya link ini bisa memakai '
            'GPU dan melihat file di folder ini. Jangan bagikan link. Hentikan runtime '
            'setelah selesai (cell 13).</div>'
        ))
    else:
        print("Tunnel gagal dibuat. 30 baris terakhir cloudflared.log:")
        if cf_log_path.exists():
            print("\n".join(cf_log_path.read_text(encoding="utf-8", errors="replace").splitlines()[-30:]))
        print("\nFallback ke Colab proxy:")
        url = colab_proxy_url()
        show_url(url)
else:
    print("Dashboard hanya tersedia di dalam VM Colab:", url)
    print("Gunakan opsi Colab proxy atau Cloudflared jika ingin membukanya dari browser.")
''')

code(r'''
#@title 12. Lihat log server + LFM3 (opsional)
from pathlib import Path

for log_path in (Path(PROJECT_DIR) / "server.log", Path("/content/lfm3-server.log")):
    if not log_path.exists():
        print(f"{log_path} belum ada.")
        continue
    size = log_path.stat().st_size
    with open(log_path, "rb") as fh:
        fh.seek(max(0, size - 25000))
        text = fh.read().decode("utf-8", errors="replace")
    print(f"\n--- {log_path} ({size / 1e6:.2f} MB), 25 KB terakhir ---")
    print("\n".join(text.splitlines()[-150:]))
''')

code(r'''
#@title 13. Hentikan server dan tunnel
import os, signal
from pathlib import Path


def stop_pidfile(path, label):
    path = Path(path)
    if not path.exists():
        print(label, ": tidak ada PID file.")
        return
    try:
        pid = int(path.read_text().strip())
    except Exception:
        path.unlink(missing_ok=True)
        return
    try:
        os.killpg(pid, signal.SIGTERM)
        print(f"{label}: SIGTERM ke PID {pid}")
    except ProcessLookupError:
        print(label, ": proses sudah berhenti.")
    except Exception as exc:
        print(label, ": gagal menghentikan:", exc)
    path.unlink(missing_ok=True)


stop_pidfile(Path(PROJECT_DIR) / ".server.pid", "Server")
stop_pidfile(Path(PROJECT_DIR) / ".lfm3.pid", "LFM3")
stop_pidfile(Path(PROJECT_DIR) / ".cloudflared.pid", "Cloudflared")
print("Runtime Colab masih hidup. Untuk mematikan sepenuhnya: Runtime > Disconnect and delete runtime.")
''')

code(r'''
#@title 14. (Opsional) Backup hasil ke Google Drive
import os, shutil
from pathlib import Path

src = Path(PROJECT_DIR) / "output"
dst = Path("/content/drive/MyDrive/OpenShorts_output")

if not src.exists():
    print("Folder output belum ada; belum ada hasil untuk dibackup.")
else:
    if not Path("/content/drive/MyDrive").is_dir():
        from google.colab import drive
        drive.mount("/content/drive")
    dst.mkdir(parents=True, exist_ok=True)
    count = 0
    total = 0
    for root, _, files in os.walk(src):
        rel = os.path.relpath(root, src)
        target = dst / rel
        target.mkdir(parents=True, exist_ok=True)
        for name in files:
            s = Path(root) / name
            t = target / name
            shutil.copy2(s, t)
            count += 1
            total += s.stat().st_size
    print(f"Backup {count} file ({total / 1e6:.1f} MB) ke {dst}")
''')

md(r"""
## Troubleshooting

| Gejala | Solusi |
|---|---|
| `GPU tidak terdeteksi` | `Runtime` → `Change runtime type` → `T4 GPU`, lalu jalankan ulang dari cell 1. |
| `numpy` bukan 1.x / `mediapipe` gagal import | Jalankan ulang cell 5. MediaPipe 0.10.21 hanya dipasang di Python <= 3.12; di Python 3.13 app otomatis memakai OpenCV Haar dan tidak butuh MediaPipe. Jangan install `mediapipe` versi terbaru: `mp.solutions` sudah dihapus. |
| `open() got an unexpected keyword argument 'metadata_errors'` | `av` 19 menghapus argumen yang masih dipakai `faster-whisper 1.2.1`. Jalankan `pip install "av==18.1.0"`, lalu restart server: cell 13 (stop) → cell 10 (start). |
| `RuntimeError: Library libcublas.so.12 is not found` | Cell 5 memasang `nvidia-cublas-cu12` bila belum ada, dan cell 10 menambahkan folder lib-nya ke `LD_LIBRARY_PATH`. Jalankan ulang cell 5 lalu cell 10. Kalau GPU tetap tidak bisa, Whisper otomatis retry di CPU dan job tetap selesai (lebih lambat). |
| `CUDA out of memory` | Hentikan server (cell 13), restart runtime, turunkan `IMAGE_STEPS` ke 12–16, dan jangan jalankan dua video bersamaan. Kalau AI Shorts (SDXL) berebut VRAM dengan LFM3, set `LFM3_FORCE_CPU=True` atau `LFM3_ENABLED=False` di cell 2 lalu jalankan ulang cell 6b. |
| Model SDXL terasa lambat / tidak muncul | Unduh dulu lewat cell 8 (`PRELOAD_SDXL=True`). Generation pertama memakan 5–15 menit. |
| `Wav2Lip` gagal | Normal: pipeline otomatis memakai still image + audio. Cek log; kualitas lipsync butuh checkpoint ~520 MB. |
| Gemini `404 model not found` | Jalankan ulang cell 6 dengan key terisi; cell itu memilih model flash yang benar-benar tersedia untuk key Anda. |
| Gemini `429` / `quota exceeded` | Kuota gratis habis. Tunggu beberapa menit/hari, ganti key, atau centang **manual AI selection** di advanced options lalu ikuti langkah di bagian atas notebook (tanpa Gemini). |
| LFM3 gagal disiapkan / build llama.cpp gagal | Lihat output cell 6b dan `/content/llama-build.log`. Pastikan `cmake` + `build-essential` terpasang (cell 3). Kalau CUDA bermasalah, set `LFM3_FORCE_CPU=True` di cell 2 lalu jalankan ulang cell 6b. Kalau tetap gagal, set `LFM3_ENABLED=False`; Clip Generator memakai Gemini/manual. |
| `unknown model architecture: 'lfm3'` | Binary llama.cpp lama (release b11936) belum mendukung LFM3. Cell 6b sudah build dari source commit yang mendukung; jangan mengganti `LFM3_LLAMA_REF` ke release lama. |
| LFM3 lambat / JSON sering retry | Wajar untuk model 1.7B. Pakai `Q4_K_M`, `LFM3_CONTEXT=16384`, dan satu video sekaligus. Untuk video panjang/nuansa sulit, pilih **Gemini only** atau manual AI selection. |
| Dashboard masih memakai LFM3 yang sudah mati | Server OpenShorts membaca `.env` hanya saat start. Jalankan cell 13 (stop) lalu cell 10 (start). Kalau LFM3 memang gagal, jalankan ulang cell 6b untuk membersihkan konfigurasi `LLM_*`. |
| Panel **MANUAL AI** tidak muncul | Pastikan server memakai `FREE_MODE=1`/`BILLING_ENABLED=0` dan job dijalankan dengan opsi manual. Video tanpa suara tidak punya transkrip untuk diekspor. |
| Import JSON ditolak | `start` harus lebih kecil dari `end`, memakai detik absolut, dan tidak melewati durasi video. Markdown fence diperbolehkan; baca pesan validasi di panel. |
| `yt-dlp` gagal download YouTube | Colab memakai IP datacenter yang sering diblokir YouTube. Download video di komputer lalu upload file ke dashboard. |
| Dashboard 403 di Colab proxy | Buka ulang link, pastikan masih login akun Google yang sama, atau pakai Cloudflared. |
| `npm ci` gagal | Pastikan Node.js >= 18 (`node -v`). Cell 3 akan memasang Node 22 bila perlu. |
| Hasil hilang setelah runtime mati | Jalankan cell 14 sebelum runtime berhenti, atau unduh video dari dashboard. |
| Upload-Post menolak | Free tier Upload-Post dibatasi (sekitar 10 upload/bulan). Unduh video lalu upload manual. |

**Catatan kualitas**

- **Clip Generator** otomatis memakai **LFM3 lokal** (cell 6b) atau Gemini. Tanpa keduanya, aktifkan **manual AI selection**: transkrip diekspor, AI eksternal memilih timestamp, lalu JSON-nya di-import. Tidak ada panggilan Gemini pada alur ini. LFM3 1.7B lebih lemah dari Gemini/7B dan hanya memproses teks; layout/video tanpa suara tetap butuh Gemini.
- **AI Shorts** tanpa Gemini memakai template analisis/script lokal; hasilnya lebih generik.
- **Thumbnail** free mode dibuat dari frame asli + teks, bukan gambar generatif.
- **Dubbing** free mode memakai terjemahan per segmen + edge-tts; bukan voice cloning seperti ElevenLabs.
- **Wav2Lip** memberi lip-sync pada foto aktor statis; hasilnya lebih sederhana daripada model avatar berbayar.
- Semua model berjalan di T4 16 GB. Satu video AI Shorts biasanya 5–20 menit tergantung panjang script dan jumlah b-roll.

**Privasi & keamanan**

- Mode self-host tidak punya login. Siapa pun yang bisa membuka URL dashboard bisa memakai server dan melihat file.
- **Colab proxy** lebih aman karena terkait sesi Google Anda; **Cloudflared** membuat URL publik. Jangan bagikan URL Cloudflared.
- API key disimpan di `.env` runtime dan hanya ada selama sesi Colab. Jangan commit `.env` atau membagikan notebook yang sudah dijalankan.
- Video, audio, dan hasil generate tersimpan di `/content/openshorts/output` (ephemeral). Backup ke Drive secara berkala.
""")

notebook = {
    "cells": CELLS,
    "metadata": {
        "accelerator": "GPU",
        "colab": {
            "provenance": [],
            "gpuType": "T4",
            "toc_visible": True,
        },
        "kernelspec": {
            "display_name": "Python 3",
            "name": "python3",
        },
        "language_info": {
            "name": "python",
        },
    },
    "nbformat": 4,
    "nbformat_minor": 0,
}

out = os.path.join(os.path.dirname(os.path.abspath(__file__)), "OpenShorts_Free_Colab.ipynb")

# Compile every plain-Python cell before writing. ast.parse() alone misses
# symbol-table errors such as `nonlocal x` at module scope.
for idx, cell in enumerate(CELLS):
    if cell["cell_type"] != "code":
        continue
    source = "".join(cell["source"])
    if source.lstrip().startswith(("!", "%")):
        continue
    try:
        compile(source, f"cell-{idx}", "exec")
    except SyntaxError as exc:
        raise SystemExit(f"Notebook cell {idx} has a syntax error: {exc}")

with open(out, "w", encoding="utf-8", newline="\n") as fh:
    json.dump(notebook, fh, ensure_ascii=False, indent=1)
print("wrote", out)
