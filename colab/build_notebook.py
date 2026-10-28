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

Jalankan **OpenShorts free mode** di Colab (disarankan **T4 GPU**). Fitur berbayar diganti model lokal/gratis:

| Fitur | Pengganti gratis |
|---|---|
| Flux (aktor & b-roll) | SDXL **RealVisXL V4.0** lokal |
| Kling/VEED lipsync | **Wav2Lip** lokal (fallback still image) |
| ElevenLabs TTS/dubbing | **edge-tts** + **faster-whisper** |
| Gemini moment picker | **LFM3-1.7B** lokal (default) / Gemini / manual |
| Thumbnail Gemini | frame asli + teks |
| S3 / Upload-Post | folder `output/` / upload manual |

**Cara pakai**

1. `Runtime` → `Change runtime type` → **T4 GPU**.
2. Jalankan cell **1–6** berurutan (atau `Runtime` → `Run all`). Cell 2 otomatis `git clone`/update repo; tidak perlu upload file.
3. Buka link dashboard dari output cell **6**.

**Opsional**
- Gemini key gratis dari [Google AI Studio](https://aistudio.google.com/apikey). Tanpa key, AI Shorts/YouTube Studio memakai fallback lokal; Clip Generator memakai LFM3 atau manual AI selection.
- **Manual AI selection:** aktifkan di advanced options, download transkrip, minta AI eksternal memilih timestamp, lalu import JSON. Tanpa Gemini.
- **LFM3** diunduh otomatis (~3.4 GB) oleh cell **5**. Kalau gagal, set `LFM3_ENABLED=False` di cell 1 lalu jalankan ulang cell 5.
- Cell **7** opsional untuk backup ke Drive dan/atau menghentikan server.

> **Privasi:** Cloudflared membuat URL publik tanpa login. Gunakan Colab proxy (default). **Lisensi:** Wav2Lip hanya riset/non-komersial; LFM3 memakai LFM Open License v1.0.
""")

code(r'''
#@title 1. Konfigurasi + cek runtime { display-mode: "form" }
#@markdown ### Project
#@markdown Notebook selalu memakai **Git clone/update** dari GitHub; tidak perlu upload file.
PROJECT_DIR = "/content/openshorts" #@param {type:"string"}
UPDATE_PROJECT = True #@param {type:"boolean"}
RESET_PROJECT = False #@param {type:"boolean"}

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
#@markdown LFM3-1.7B dimuat langsung dengan Transformers di GPU/CPU Colab.
#@markdown Unduhan pertama ~3.4 GB; sesi berikutnya memakai cache selama runtime hidup.
#@markdown Pilih **false** untuk tetap memakai Gemini / manual AI selection saja.
LFM3_ENABLED = True #@param {type:"boolean"}
LFM3_CONTEXT = 16384 #@param {type:"integer"}
LFM3_FORCE_CPU = False #@param {type:"boolean"}

#@markdown ### Akses dashboard
PUBLIC_TUNNEL = "Colab proxy (privat)" #@param ["Colab proxy (privat)", "Cloudflared (publik)", "Tidak perlu"]

import getpass, os, platform, shutil, subprocess, sys

GEMINI_API_KEY = (GEMINI_API_KEY or "").strip()
HF_TOKEN = (HF_TOKEN or "").strip()
UPLOAD_POST_API_KEY = (UPLOAD_POST_API_KEY or "").strip()
GEMINI_MODEL = (GEMINI_MODEL or "auto").strip()
FREE_IMAGE_MODEL = (FREE_IMAGE_MODEL or "SG161222/RealVisXL_V4.0").strip()
IMAGE_STEPS = max(8, min(50, int(IMAGE_STEPS)))
LFM3_CONTEXT = max(4096, min(32768, int(LFM3_CONTEXT)))

if ASK_GEMINI_KEY and not GEMINI_API_KEY:
    GEMINI_API_KEY = getpass.getpass(
        "Gemini API key (Enter = lewati; Clip Generator bisa manual tanpa Gemini): "
    ).strip()

PROJECT_DIR = os.path.abspath(os.path.expanduser(PROJECT_DIR.strip()))
if PROJECT_DIR in ("/", "/content"):
    raise ValueError("PROJECT_DIR terlalu umum; pakai misalnya /content/openshorts")
os.makedirs(os.path.dirname(PROJECT_DIR), exist_ok=True)


def _mask(value):
    if not value:
        return "(kosong)"
    if len(value) <= 10:
        return "terisi"
    return value[:4] + "..." + value[-4:]


print("Sumber project : Git clone")
print("Project dir    :", PROJECT_DIR)
print("Gemini key     :", _mask(GEMINI_API_KEY))
print("HF token       :", _mask(HF_TOKEN))
print("Upload-Post key:", _mask(UPLOAD_POST_API_KEY))
print("Gemini model   :", GEMINI_MODEL)
print("Model gambar   :", FREE_IMAGE_MODEL, f"({IMAGE_STEPS} steps)")
print("Whisper        :", WHISPER_MODEL)
print("LFM3 lokal     :", "aktif" if LFM3_ENABLED else "nonaktif",
      f"(Transformers, ctx {LFM3_CONTEXT})" if LFM3_ENABLED else "")
print("Akses          :", PUBLIC_TUNNEL)

print("\n== Cek runtime ==")
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
''')

code(r'''
#@title 2. Install sistem, ambil project, build dashboard
import os, shutil, subprocess, sys
from pathlib import Path

def run(cmd, **kwargs):
    print("+", " ".join(cmd) if isinstance(cmd, list) else cmd)
    return subprocess.run(cmd, **kwargs)

APT_PACKAGES = [
    "ffmpeg", "git", "curl", "wget", "unzip", "fontconfig",
    "libgl1", "libglib2.0-0", "libsm6", "libxext6", "libxrender1", "libsndfile1",
    "fonts-liberation", "fonts-noto-core", "fonts-noto-color-emoji", "fonts-dejavu-core",
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
_nvcc = shutil.which("nvcc") or ("/usr/local/cuda/bin/nvcc" if os.path.exists("/usr/local/cuda/bin/nvcc") else None)
print("nvcc   :", _nvcc or "TIDAK ADA (tidak diperlukan; LFM3 memakai CUDA dari torch)")


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

print("\n== Ambil / update project ==")
PROJECT_DIR = os.path.abspath(PROJECT_DIR)


def looks_like_project(path):
    p = Path(path)
    return (
        (p / "app.py").is_file()
        and (p / "free_backends.py").is_file()
        and (p / "colab" / "requirements-colab.txt").is_file()
    )


def git(*args, check=True):
    return subprocess.run(["git", "-C", PROJECT_DIR, *args], check=check)


def update_git_project():
    # Folder lama yang belum punya .git dijadikan repo git di tempat supaya
    # update berikutnya mudah, tanpa menghapus .env/output/cache:
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
    if UPDATE_PROJECT:
        update_git_project()
    else:
        print("UPDATE_PROJECT=False; memakai versi yang ada saat ini.")
else:
    if os.path.exists(PROJECT_DIR):
        raise RuntimeError(
            f"{PROJECT_DIR} sudah ada tetapi bukan project free mode. "
            "Set RESET_PROJECT=True atau hapus folder tersebut."
        )
    if not REPO_URL.strip():
        raise ValueError(
            "REPO_URL kosong. Isi URL repo GitHub, misalnya "
            "https://github.com/ardyhim/openshorts.git."
        )
    subprocess.run(
        ["git", "clone", "--depth", "1", "--branch", REPO_BRANCH, REPO_URL, PROJECT_DIR],
        check=True,
    )

if not looks_like_project(PROJECT_DIR):
    raise RuntimeError(
        "Project tidak valid. Pastikan repo berisi free_backends.py dan "
        "colab/requirements-colab.txt (ardyhim/openshorts sudah berisi keduanya)."
    )

os.chdir(PROJECT_DIR)
print("\nProject siap:", PROJECT_DIR)
print("free_backends.py:", os.path.getsize(os.path.join(PROJECT_DIR, "free_backends.py")) // 1024, "KB")
print("Isi utama:", ", ".join(sorted(os.listdir(PROJECT_DIR))[:12]), "...")

print("\n== Build dashboard ==")
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
#@title 3. Install dependensi Python
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
    print("   Pastikan cell 2 memakai kode terbaru (UPDATE_PROJECT=True), lalu ulangi cell ini.")
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
# loader path. Pasang runtime CUDA 12 kalau belum ada; cell 6 menambahkan
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
#@title 4. Tulis konfigurasi free mode + Gemini
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
    print("  - Clip Generator (pemilihan momen viral) otomatis memakai LFM3 lokal dari cell 5;")
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
    # cell 5 writes these only after the LFM3 server is confirmed healthy.
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
#@title 5. Siapkan LFM3 lokal (Transformers, tanpa llama.cpp)
#@markdown Unduhan pertama ~3.4 GB (model BF16) + install runtime di venv terpisah.
#@markdown Biasanya 1–3 menit; setelah itu server LFM3 tetap hidup di background.
#@markdown Kalau `python -m venv` gagal, cell otomatis mencoba `venv --without-pip`, `virtualenv`, lalu `uv`.
#@markdown Dashboard otomatis menampilkan pilihan **Local only (LFM3)** di advanced options.
import json, os, shutil, signal, subprocess, sys, time, urllib.error, urllib.request
from pathlib import Path

PROJECT_DIR = os.path.abspath(PROJECT_DIR)
LFM3_PORT = 8089
LFM3_ALIAS = "lfm3-1.7b"
LFM3_MODEL = "LiquidAI/LFM3-1.7B"
# The checkpoint's remote code uses the Transformers 5.x API
# (use_kernel_func_from_hub_with_fallback), so 5.21.0 is the tested runtime.
# The main app keeps transformers 4.57.1 for diffusers/SDXL, so LFM3 lives in
# its own venv with --system-site-packages (it reuses Colab's CUDA torch).
LFM3_RUNTIME = "transformers==5.21.0|accelerate==1.15.0"
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
VENV_DIR = Path("/content/lfm3-venv")
VENV_PY = VENV_DIR / "bin" / "python"
VENV_MARKER = VENV_DIR / ".lfm3-runtime"
MODEL_DIR = Path("/content/openshorts-models/LFM3-1.7B")
SERVER_SCRIPT = Path(PROJECT_DIR) / "colab" / "lfm3_server.py"


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


def run(cmd, **kwargs):
    print("+", " ".join(str(part) for part in cmd))
    result = subprocess.run([str(part) for part in cmd], **kwargs)
    if result.returncode != 0:
        detail = "\n".join(
            text.strip() for text in (result.stdout, result.stderr)
            if isinstance(text, str) and text.strip())
        if detail:
            print(detail[-4000:])
        raise RuntimeError(
            f"perintah gagal (kode {result.returncode}): "
            + " ".join(str(part) for part in cmd))
    return result


def attempt(name, cmd):
    """Run a venv creation command, show its output, and report success."""
    print(f"Mencoba {name}:")
    print("+", " ".join(str(part) for part in cmd))
    result = subprocess.run([str(part) for part in cmd],
                            capture_output=True, text=True)
    output = "\n".join(
        text.strip() for text in (result.stdout, result.stderr)
        if text and text.strip())
    if output:
        print(output[-4000:])
    return result.returncode == 0, output


def pip_works():
    if not VENV_PY.exists():
        return False
    result = subprocess.run([str(VENV_PY), "-m", "pip", "--version"],
                            capture_output=True, text=True)
    return result.returncode == 0


def system_pip_install(*packages):
    """Install into Colab's Python, tolerating PEP 668 protection."""
    cmd = [sys.executable, "-m", "pip", "install", "-q", *packages]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode == 0:
        return True
    output = "\n".join(
        text.strip() for text in (result.stdout, result.stderr)
        if text and text.strip())
    if "externally-managed-environment" in output or "externally managed" in output:
        result = subprocess.run(cmd + ["--break-system-packages"],
                                capture_output=True, text=True)
        if result.returncode == 0:
            return True
        output = "\n".join(
            text.strip() for text in (result.stdout, result.stderr)
            if text and text.strip())
    print("(install pendukung gagal:", output[-1500:] or "tanpa pesan", ")")
    return False


def bootstrap_pip():
    """Give a --without-pip venv a working pip."""
    if pip_works():
        return True
    # pip >= 23.1 can target another interpreter directly.
    result = subprocess.run(
        [sys.executable, "-m", "pip", "--python", str(VENV_PY),
         "install", "-q", "--upgrade", "pip"],
        capture_output=True, text=True)
    if result.returncode == 0 and pip_works():
        return True
    # Last resort: the official bootstrap script inside the venv.
    try:
        get_pip = Path("/tmp/get-pip.py")
        urllib.request.urlretrieve("https://bootstrap.pypa.io/get-pip.py", get_pip)
        result = subprocess.run([str(VENV_PY), str(get_pip)],
                                capture_output=True, text=True)
        if result.returncode == 0 and pip_works():
            return True
        print("get-pip gagal:", (result.stderr or result.stdout or "").strip()[-1500:])
    except Exception as exc:
        print("get-pip gagal:", exc)
    return False


def reset_venv():
    if not VENV_DIR.exists():
        return
    shutil.rmtree(VENV_DIR, ignore_errors=True)
    if VENV_DIR.exists():
        subprocess.run(["rm", "-rf", str(VENV_DIR)],
                       capture_output=True, text=True)


def create_venv():
    """Create the LFM3 venv, with fallbacks for Colab images without ensurepip."""
    print("Python LFM3  :", sys.version.split()[0], sys.executable)
    failures = []
    plans = [
        ("venv", [sys.executable, "-m", "venv",
                  "--system-site-packages", str(VENV_DIR)]),
        # Some Colab images ship Python without ensurepip, which makes plain
        # "python -m venv" fail before it writes anything useful.
        ("venv --without-pip", [sys.executable, "-m", "venv",
                                "--system-site-packages", "--without-pip",
                                str(VENV_DIR)]),
    ]
    for name, cmd in plans:
        reset_venv()
        ok, output = attempt(name, cmd)
        if not ok:
            failures.append(f"{name}:\n{output[-1500:] or '(tanpa pesan)'}")
            continue
        if name.endswith("--without-pip") and not bootstrap_pip():
            failures.append(f"{name}: pip tidak bisa dipasang")
            continue
        if pip_works():
            print("Venv LFM3    : siap")
            return
        failures.append(f"{name}: python/pip venv tidak bisa dipakai")

    # virtualenv bundles its own pip/setuptools wheels, so it works even when
    # the stdlib venv module or ensurepip is broken.
    reset_venv()
    if system_pip_install("virtualenv"):
        ok, output = attempt("virtualenv", [
            sys.executable, "-m", "virtualenv",
            "--system-site-packages", str(VENV_DIR)])
        if ok and pip_works():
            print("Venv LFM3    : siap (virtualenv)")
            return
        failures.append(f"virtualenv:\n{output[-1500:] or '(tanpa pesan)'}")

    # uv is preinstalled on some Colab images and also works without ensurepip.
    uv = shutil.which("uv")
    if uv:
        reset_venv()
        ok, output = attempt("uv venv", [
            uv, "venv", "--system-site-packages",
            "--python", sys.executable, str(VENV_DIR)])
        if ok:
            subprocess.run([uv, "pip", "install", "--python", str(VENV_PY),
                            "-q", "pip"], capture_output=True, text=True)
            if pip_works():
                print("Venv LFM3    : siap (uv)")
                return
        failures.append(f"uv venv:\n{output[-1500:] or '(tanpa pesan)'}")

    raise RuntimeError(
        "Gagal membuat venv LFM3. Coba Runtime → Restart session lalu jalankan "
        "ulang cell ini.\n\n" + "\n\n".join(failures))


def ensure_runtime():
    """Create an isolated transformers 5.21.0 venv that reuses Colab's torch."""
    if VENV_PY.exists() and VENV_MARKER.exists():
        try:
            if VENV_MARKER.read_text(encoding="utf-8").strip() == LFM3_RUNTIME:
                check = subprocess.run(
                    [str(VENV_PY), "-c",
                     "import torch, transformers, accelerate; "
                     "print(torch.__version__, transformers.__version__, accelerate.__version__)"],
                    capture_output=True, text=True, timeout=180)
                if check.returncode == 0 and "5.21.0" in check.stdout:
                    print("Runtime LFM3 :", check.stdout.strip())
                    return
        except Exception:
            pass
    if VENV_DIR.exists():
        print("Membuat ulang venv LFM3...")
    create_venv()
    run([str(VENV_PY), "-m", "pip", "install", "--upgrade", "pip", "-q"],
        check=True)
    run([str(VENV_PY), "-m", "pip", "install", "-q",
         "transformers==5.21.0", "accelerate==1.15.0"], check=True)
    check = subprocess.run(
        [str(VENV_PY), "-c",
         "import torch, transformers, accelerate; "
         "print(torch.__version__, transformers.__version__, accelerate.__version__)"],
        capture_output=True, text=True, timeout=180)
    if check.returncode != 0:
        raise RuntimeError(
            "Runtime LFM3 tidak bisa di-import:\n" + (check.stderr or check.stdout))
    print("Runtime LFM3 :", check.stdout.strip())
    VENV_MARKER.write_text(LFM3_RUNTIME, encoding="utf-8")


def ensure_model():
    weights = sorted(MODEL_DIR.glob("*.safetensors"))
    total = sum(path.stat().st_size for path in weights)
    if (MODEL_DIR / "config.json").exists() and total > 1_000_000_000:
        print(f"Model LFM3   : {MODEL_DIR} ({total / 1e9:.2f} GB)")
        return
    print("Mengunduh LFM3-1.7B (~3.4 GB, sekali per runtime)...")
    env = os.environ.copy()
    if HF_TOKEN:
        env["HF_TOKEN"] = HF_TOKEN
    run([str(VENV_PY), str(SERVER_SCRIPT),
         "--model", LFM3_MODEL, "--download-only",
         "--local-dir", str(MODEL_DIR)], check=True, env=env)
    weights = sorted(MODEL_DIR.glob("*.safetensors"))
    total = sum(path.stat().st_size for path in weights)
    if total < 1_000_000_000:
        raise RuntimeError(f"Model LFM3 tidak lengkap di {MODEL_DIR}")
    print(f"Model LFM3   : {total / 1e9:.2f} GB")


def post_chat(extra, use_format=True, timeout=600):
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
        with urllib.request.urlopen(req, timeout=timeout) as resp:
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

    if not SERVER_SCRIPT.exists():
        raise RuntimeError(
            f"{SERVER_SCRIPT} tidak ditemukan; jalankan git pull di folder project.")

    clear_llm_env()
    device_pref = "cpu" if LFM3_FORCE_CPU else "auto"
    try:
        server_mtime = SERVER_SCRIPT.stat().st_mtime_ns
    except Exception:
        server_mtime = 0
    # Include the runtime, context, device preference and server-script mtime:
    # changing any of them must restart the server, not silently keep the old
    # process (or the old transformers version).
    wanted_model = (
        f"{LFM3_MODEL}:{LFM3_CONTEXT}:{device_pref}:"
        f"{LFM3_RUNTIME}:{server_mtime}")
    running = lfm3_models()
    same_model = False
    try:
        same_model = MODEL_MARKER.read_text(
            encoding="utf-8").strip() == wanted_model
    except Exception:
        pass
    if running and not same_model:
        print("Server LFM3 lama memakai model/context/runtime berbeda; restart...")
        stop_lfm3()
        running = []
    if not running:
        stop_lfm3()
        ensure_runtime()
        ensure_model()

        cmd = [
            str(VENV_PY), str(SERVER_SCRIPT),
            "--model", str(MODEL_DIR),
            "--served-model-name", LFM3_ALIAS,
            "--host", "127.0.0.1", "--port", str(LFM3_PORT),
            "--max-context", str(LFM3_CONTEXT),
            "--max-new-tokens", "4096",
            "--device", device_pref,
            "--dtype", "auto",
            "--triton", "auto",
        ]
        print("Menjalankan:", " ".join(cmd))
        SERVER_LOG.write_text("", encoding="utf-8")
        log_handle = open(SERVER_LOG, "ab", buffering=0)
        env = os.environ.copy()
        env["PYTHONUNBUFFERED"] = "1"
        env["TOKENIZERS_PARALLELISM"] = "false"
        if HF_TOKEN:
            env["HF_TOKEN"] = HF_TOKEN
        proc = subprocess.Popen(
            cmd, stdout=log_handle, stderr=subprocess.STDOUT,
            start_new_session=True, env=env)
        # The child inherited the descriptor; the parent no longer needs it.
        log_handle.close()
        PID_PATH.write_text(str(proc.pid), encoding="utf-8")
        deadline = time.time() + 600
        while time.time() < deadline:
            if proc.poll() is not None:
                raise RuntimeError(
                    f"lfm3_server.py berhenti (kode {proc.returncode}).\n"
                    + tail(SERVER_LOG, 80))
            if LFM3_ALIAS in lfm3_models():
                break
            time.sleep(2)
        else:
            raise RuntimeError(
                "Server LFM3 tidak siap dalam 10 menit.\n" + tail(SERVER_LOG, 80))
        MODEL_MARKER.write_text(wanted_model, encoding="utf-8")
        print("Server LFM3 : siap")

    # The template's enable_thinking switch is passed per request; the server
    # also defaults to False, so a bare request stays fast.
    try:
        post_chat({"chat_template_kwargs": {"enable_thinking": False}},
                  use_format=False, timeout=300)
    except Exception as exc:
        print("⚠️ Uji chat LFM3 gagal (server tetap dipakai):", exc)

    try:
        message = post_chat(
            {"chat_template_kwargs": {"enable_thinking": False}},
            use_format=True, timeout=600)
        answer = (message.get("content") or "").strip()
        print("Self-test LFM3:", answer[:180] or "(kosong)")
    except Exception as exc:
        print("⚠️ Uji JSON LFM3 gagal (app punya fallback response_format):", exc)

    updates = {
        "LLM_PROVIDER": "openai",
        "LLM_BASE_URL": f"http://127.0.0.1:{LFM3_PORT}/v1",
        "LLM_MODEL": LFM3_ALIAS,
        "LLM_API_KEY": "ollama",
        # CPU generation is slow; the app's httpx client must not give up
        # halfway through a 4096-token answer.
        "LLM_TIMEOUT": "1800",
        "LLM_TEMPERATURE": "0.1",
        "LLM_TOP_K": "50",
        "LLM_REPETITION_PENALTY": "1.1",
        "LLM_MAX_TOKENS": "4096",
        "LLM_DISABLE_THINKING": "1",
        "LLM_SCORE_BATCH": "3",
    }
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
    print("\n--- server log ---")
    print(tail(SERVER_LOG, 80))
    stop_lfm3()
    clear_llm_env()
    LFM3_READY = False
    print("\nClip Generator tetap bisa memakai Gemini atau manual AI selection.")
    print("Perbaiki error di atas lalu jalankan ulang cell ini.")
''')

code(r'''
#@title 6. Jalankan server + buka dashboard
import glob, json, os, re, signal, site, subprocess, sys, time, urllib.request
from pathlib import Path
from IPython.display import display, HTML

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
    raise RuntimeError("freeMode tidak aktif. Periksa .env / cell 4.")
try:
    with urllib.request.urlopen("http://127.0.0.1:8000/", timeout=10) as resp:
        html = resp.read(200).decode("utf-8", errors="replace")
    print("  dashboard      :", "OK" if ("<html" in html.lower() or "<!doctype" in html.lower()) else "tidak terdeteksi")
except Exception as exc:
    print("  dashboard      : gagal dimuat:", exc)

print("\n== Buka dashboard ==")
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
            'setelah selesai (cell 7).</div>'
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
#@title 7. (Opsional) Backup hasil & hentikan server
ACTION = "Tidak ada" #@param ["Tidak ada", "Backup ke Drive", "Hentikan server", "Backup + hentikan"]

import os, shutil, signal
from pathlib import Path

PROJECT_DIR = os.path.abspath(PROJECT_DIR)

if ACTION in ("Backup ke Drive", "Backup + hentikan"):
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
                source = Path(root) / name
                target_file = target / name
                shutil.copy2(source, target_file)
                count += 1
                total += source.stat().st_size
        print(f"Backup {count} file ({total / 1e6:.1f} MB) ke {dst}")

if ACTION in ("Hentikan server", "Backup + hentikan"):
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

if ACTION == "Tidak ada":
    print("Tidak ada aksi. Pilih opsi di dropdown untuk backup dan/atau menghentikan server.")
''')

md(r"""
## Troubleshooting singkat

| Gejala | Solusi |
|---|---|
| GPU tidak terdeteksi | `Runtime` → `Change runtime type` → `T4 GPU`, lalu jalankan ulang cell 1. |
| `npm ci` / build dashboard gagal | Pastikan Node.js >= 18; jalankan ulang cell 2. |
| Install dependensi gagal | Jalankan ulang cell 2 lalu cell 3. Konflik umum: NumPy vs MediaPipe (lihat pesan pip). |
| LFM3 gagal disiapkan | Lihat output cell 5 dan `/content/lfm3-server.log`. Set `LFM3_ENABLED=False` di cell 1 kalau tetap gagal; Clip Generator bisa memakai Gemini/manual. |
| Gemini `429` / kuota habis | Tunggu, ganti key, atau pakai **manual AI selection**. |
| Server tidak siap | Lihat `/content/openshorts/server.log`, lalu jalankan ulang cell 6. |
| Dashboard tidak terbuka | Pastikan cell 6 sukses; coba opsi **Colab proxy** di cell 1. |
| Hasil hilang setelah runtime mati | Jalankan cell 7 → **Backup ke Drive** sebelum runtime berhenti. |

Panduan lengkap: [`colab/README.md`](https://github.com/ardyhim/openshorts/blob/main/colab/README.md).

**Catatan:** LFM3 1.7B hanya untuk pemilihan momen teks; layout/video tanpa suara tetap butuh Gemini. Wav2Lip hanya riset/non-komersial.
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
