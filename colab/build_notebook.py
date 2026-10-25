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
| AWS S3 | folder `output/` lokal |
| Upload-Post | opsional (free tier) atau download manual |

**Yang perlu disiapkan**

1. Runtime Colab dengan GPU: `Runtime` → `Change runtime type` → `T4 GPU`.
2. File ZIP project `openshorts-free-colab.zip` (ada di folder `colab/` repo ini). Kalau sudah punya fork GitHub yang berisi free mode, bisa juga memakai opsi **Git clone**.
3. **Opsional tetapi sangat disarankan:** API key Gemini gratis dari [Google AI Studio](https://aistudio.google.com/apikey). Tanpa key, AI Shorts dan YouTube Studio tetap jalan dengan fallback lokal, tetapi **Clip Generator butuh Gemini** (atau server LLM lokal via `LLM_BASE_URL`) untuk memilih momen viral.

> **Peringatan privasi:** opsi tunnel **Cloudflared** membuat dashboard bisa diakses publik tanpa login. Gunakan **Colab proxy** (default) kalau tidak ingin orang lain membuka GPU dan hasil video Anda.

> **Lisensi model:** **Wav2Lip hanya untuk riset/personal/non-komersial** (lihat README upstream). Untuk konten komersial, set `WAV2LIP_DISABLE=1` di `.env` agar pipeline memakai still image + audio, atau ganti dengan model berlisensi komersial. RealVisXL V4.0 memakai lisensi OpenRAIL++-M.

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

#@markdown ### Akses dashboard
PUBLIC_TUNNEL = "Colab proxy (privat)" #@param ["Colab proxy (privat)", "Cloudflared (publik)", "Tidak perlu"]

import getpass, os

GEMINI_API_KEY = (GEMINI_API_KEY or "").strip()
HF_TOKEN = (HF_TOKEN or "").strip()
UPLOAD_POST_API_KEY = (UPLOAD_POST_API_KEY or "").strip()
GEMINI_MODEL = (GEMINI_MODEL or "auto").strip()
FREE_IMAGE_MODEL = (FREE_IMAGE_MODEL or "SG161222/RealVisXL_V4.0").strip()
IMAGE_STEPS = max(8, min(50, int(IMAGE_STEPS)))

if ASK_GEMINI_KEY and not GEMINI_API_KEY:
    GEMINI_API_KEY = getpass.getpass(
        "Gemini API key (Enter = lewati; Clip Generator nanti butuh key/LLM lokal): "
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


# NumPy 1.26 + OpenCV 4.10 + protobuf 4.x ikut dipasang dari requirements.
try:
    pip("install", "-q", "--no-warn-script-location", "-r", "colab/requirements-colab.txt")
except subprocess.CalledProcessError:
    print("\n❌ Install dependensi gagal. Lihat baris ERROR dari pip di atas.")
    print("   Pastikan cell 4 memakai kode terbaru (UPDATE_PROJECT=True), lalu ulangi cell ini.")
    print("   Konflik yang umum: py3langid versi lama butuh NumPy 2, sedangkan mediapipe butuh NumPy 1.x.")
    raise

# MediaPipe 0.10.21 masih punya mp.solutions (dipakai face tracking), tetapi
# deklarasi dependensinya menarik NumPy 2 dan JAX. Kita hanya butuh graph
# vision bawaan, jadi pasang tanpa dependensi; dependency lain sudah ada di atas.
pip("install", "-q", "--no-warn-script-location", "--no-deps", "mediapipe==0.10.21")

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
            nonlocal auth_invalid
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
    print("  - Clip Generator (pemilihan momen viral) membutuhkan Gemini atau LLM lokal.")
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
import os, subprocess, sys

check_code = r"""
import numpy as np
print("numpy      :", np.__version__)
assert np.__version__.startswith("1."), "NumPy harus 1.x untuk mediapipe 0.10.21"

import cv2
print("opencv     :", cv2.__version__)

import torch
print("torch      :", torch.__version__, "| CUDA:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("gpu        :", torch.cuda.get_device_name(0))

import mediapipe as mp
print("mediapipe  :", mp.__version__)
mp.solutions.face_detection.FaceDetection(model_selection=1, min_detection_confidence=0.5)
print("face detect: OK")

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
''')

code(r'''
#@title 10. Jalankan server OpenShorts
import json, os, signal, subprocess, sys, time, urllib.request
from pathlib import Path

PROJECT_DIR = os.path.abspath(PROJECT_DIR)
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
for key in ("billingEnabled", "freeMode", "freeTts", "freeImages", "freeDubbing", "serverGemini"):
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
#@title 12. Lihat log server (opsional)
from pathlib import Path

log_path = Path(PROJECT_DIR) / "server.log"
if not log_path.exists():
    print("server.log belum ada.")
else:
    size = log_path.stat().st_size
    with open(log_path, "rb") as fh:
        fh.seek(max(0, size - 25000))
        text = fh.read().decode("utf-8", errors="replace")
    print(f"--- {log_path} ({size / 1e6:.2f} MB), 25 KB terakhir ---")
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
| `numpy` bukan 1.x / `mediapipe` gagal import | Jalankan ulang cell 5. Jangan install `mediapipe` versi terbaru; versi 0.10.21 dibutuhkan karena masih punya `mp.solutions`. |
| `CUDA out of memory` | Hentikan server (cell 13), restart runtime, turunkan `IMAGE_STEPS` ke 12–16, dan jangan jalankan dua video bersamaan. |
| Model SDXL terasa lambat / tidak muncul | Unduh dulu lewat cell 8 (`PRELOAD_SDXL=True`). Generation pertama memakan 5–15 menit. |
| `Wav2Lip` gagal | Normal: pipeline otomatis memakai still image + audio. Cek log; kualitas lipsync butuh checkpoint ~520 MB. |
| Gemini `404 model not found` | Jalankan ulang cell 6 dengan key terisi; cell itu memilih model flash yang benar-benar tersedia untuk key Anda. |
| Gemini `429` / `quota exceeded` | Kuota gratis habis. Tunggu beberapa menit/hari, atau ganti key. |
| `yt-dlp` gagal download YouTube | Colab memakai IP datacenter yang sering diblokir YouTube. Download video di komputer lalu upload file ke dashboard. |
| Dashboard 403 di Colab proxy | Buka ulang link, pastikan masih login akun Google yang sama, atau pakai Cloudflared. |
| `npm ci` gagal | Pastikan Node.js >= 18 (`node -v`). Cell 3 akan memasang Node 22 bila perlu. |
| Hasil hilang setelah runtime mati | Jalankan cell 14 sebelum runtime berhenti, atau unduh video dari dashboard. |
| Upload-Post menolak | Free tier Upload-Post dibatasi (sekitar 10 upload/bulan). Unduh video lalu upload manual. |

**Catatan kualitas**

- **Clip Generator** membutuhkan Gemini (gratis) atau server LLM lokal (`LLM_BASE_URL`, misalnya Ollama). Tanpa itu, pipeline tidak bisa memilih momen viral.
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
with open(out, "w", encoding="utf-8", newline="\n") as fh:
    json.dump(notebook, fh, ensure_ascii=False, indent=1)
print("wrote", out)
