# OpenShorts Free Mode — Google Colab

Folder ini berisi setup untuk menjalankan **OpenShorts** di Google Colab **tanpa API berbayar**. Semua fitur inti tetap ada, tetapi provider mahal diganti dengan model lokal di GPU T4 atau layanan gratis.

**Buka langsung di Colab (tanpa download/upload apa pun):**

<https://colab.research.google.com/github/ardyhim/openshorts/blob/main/colab/OpenShorts_Free_Colab.ipynb>

| File | Fungsi |
|---|---|
| `OpenShorts_Free_Colab.ipynb` | Notebook utama: cek GPU, install, build dashboard, jalankan server, buka URL. |
| `requirements-colab.txt` | Dependensi Python yang aman untuk Colab (tanpa mengganti torch bawaan). |
| `openshorts-free-colab.zip` | Cadangan offline; tidak dipakai secara default karena notebook memakai Git clone. |
| `build_zip.py` | Membuat ulang ZIP dari repo: `python colab/build_zip.py`. |
| `build_notebook.py` | Generator notebook (untuk maintainer): `python colab/build_notebook.py`. |

> **Penting:** notebook ini **tidak** memakai Docker. Colab tidak menyediakan Docker, jadi server FastAPI, dashboard, ffmpeg, dan model berjalan langsung di VM Colab.

---

## 1. Yang diganti

| Sebelumnya (berbayar) | Free mode |
|---|---|
| fal.ai Flux 2 Pro — foto aktor | **RealVisXL V4.0** (SDXL) lokal via `diffusers` |
| fal.ai Kling/VEED — talking head | **Wav2Lip** lokal; fallback still image + audio |
| fal.ai Kling — b-roll | still SDXL + efek Ken Burns |
| ElevenLabs TTS | **edge-tts** (Microsoft neural voices, tanpa key) |
| ElevenLabs dubbing | **faster-whisper → Gemini/Google Translate → edge-tts → ffmpeg** |
| Gemini image — thumbnail | frame asli video + teks (PIL) |
| AWS S3 | folder `output/` lokal |
| Upload-Post | opsional (free tier) atau download manual |

Yang **masih memakai internet gratis**:

- Hugging Face untuk mengunduh model (RealVisXL, Wav2Lip, Whisper).
- edge-tts untuk suara (Microsoft Edge online).
- Gemini free tier untuk pemilihan momen/script/title (opsional, tanpa kartu kredit).
- Google Translate web untuk dubbing tanpa Gemini.
- YouTube/website sumber video (kalau memakai URL).

---

## 2. Mulai cepat

### 2.1 Buka notebook

Tidak perlu download repo atau upload ZIP.

1. Buka **langsung** notebook ini di Colab: <https://colab.research.google.com/github/ardyhim/openshorts/blob/main/colab/OpenShorts_Free_Colab.ipynb>
2. `Runtime` → `Change runtime type` → **T4 GPU** → `Save`.
3. Cell 4 otomatis menjalankan `git clone` dari <https://github.com/ardyhim/openshorts.git> ke `/content/openshorts`.

Kalau ingin memakai notebook dari file lokal: `File` → `Upload notebook`. ZIP project tetap tersedia sebagai cadangan offline, tetapi tidak diperlukan.

### 2.2 API key Gemini (gratis, opsional tetapi disarankan)

1. Buka <https://aistudio.google.com/apikey>.
2. Login dengan akun Google, klik **Create API key**. Tidak perlu kartu kredit.
3. Copy key (`AIza...`).
4. Di notebook, cell **2** punya kolom `GEMINI_API_KEY`; paste di sana. Kalau dikosongkan, notebook akan meminta lewat prompt (tekan Enter untuk melewati).

**Kapan key dibutuhkan?**

| Fitur | Tanpa Gemini key | Dengan Gemini key gratis |
|---|---|---|
| Clip Generator | Tidak bisa memilih momen viral¹ | Bisa |
| AI Shorts — analisis/script | Template lokal generik | Analisis produk + script lebih relevan |
| YouTube Studio — judul/deskripsi | Fallback dari transkrip | Judul/deskripsi lebih kreatif |
| Thumbnail | Frame + teks | Tetap frame + teks (free mode) |
| Dubbing | Google Translate web | Terjemahan Gemini (lebih rapi) |

¹ Bisa juga memakai LLM lokal lewat `LLM_BASE_URL` (Ollama/LM Studio/vLLM), tetapi tidak dijalankan otomatis oleh notebook ini karena butuh RAM/VRAM tambahan.

### 2.3 Jalankan

1. Jalankan cell **1–14** berurutan, atau `Runtime` → `Run all`.
2. Cell **4** otomatis clone repo GitHub ke `/content/openshorts` — tidak ada file yang perlu di-upload.
3. Tunggu install + build dashboard (sekitar 3–6 menit, tergantung Colab).
4. Cell **11** menampilkan tombol **Buka OpenShorts**.
5. Selesai. Video dan thumbnail muncul di folder `output/`.

> Kalau menjalankan ulang cell **4**, `UPDATE_PROJECT=True` akan mengambil versi terbaru dari GitHub. File `.env`, `output/`, dan cache model tetap aman karena tidak dilacak git.

> Pertama kali memakai AI Shorts, model SDXL (~7 GB) dan Wav2Lip (~520 MB) diunduh otomatis. Ini bisa memakan 5–15 menit. Untuk mengunduh lebih awal, jalankan cell **8** dengan `PRELOAD_SDXL=True`.

---

## 3. Pilihan akses dashboard

### Colab proxy (default, lebih aman)

```python
PUBLIC_TUNNEL = "Colab proxy (privat)"
```

URL yang muncul terkait sesi Google Anda. Hanya hidup selama runtime Colab aktif. Ini pilihan yang disarankan.

### Cloudflared (publik)

```python
PUBLIC_TUNNEL = "Cloudflared (publik)"
```

Notebook mengunduh `cloudflared` dan membuat quick tunnel `https://....trycloudflare.com`.

> **Peringatan:** mode self-host **tidak punya login**. Siapa pun yang memiliki URL Cloudflared bisa membuka dashboard, memakai GPU Anda, dan melihat file di folder project. Jangan bagikan URL ini. Hentikan runtime setelah selesai.

### Tanpa tunnel

`PUBLIC_TUNNEL = "Tidak perlu"` hanya menampilkan `http://127.0.0.1:8000`, yang tidak bisa dibuka dari browser luar. Gunakan hanya kalau Anda mengakses VM dengan cara lain.

---

## 4. Upload-Post (publishing, opsional)

OpenShorts bisa memposting ke TikTok, Instagram Reels, dan YouTube Shorts lewat [Upload-Post](https://www.upload-post.com/). Free tier mereka (sekitar 10 upload/bulan, tanpa kartu kredit) cukup untuk mencoba.

- Isi `UPLOAD_POST_API_KEY` di cell 2. Dashboard otomatis mendeteksi key server ini; kamu tinggal memilih profil yang terhubung di `Settings` (tidak perlu mengetik ulang key).
- Bisa juga mengosongkannya di cell 2 lalu memasukkan key langsung di `Settings` dashboard. Key browser tidak disimpan di server.
- Kalau tidak punya key: tetap bisa membuat video, lalu **download manual** dari dashboard.
- Tidak ada API resmi gratis dari TikTok/Instagram untuk akun pribadi. Upload-Post adalah jalan termudah; alternatifnya upload manual.
- YouTube juga bisa upload manual; kualitas dan fitur tidak berkurang.

---

## 5. Batas dan kualitas free mode

| Aspek | Kenyataan |
|---|---|
| GPU | T4 16 GB. Satu video AI Shorts biasanya 5–20 menit. Clip Generator lebih cepat. |
| Sesi | Colab gratis bisa terputus kapan saja; tidak ada SLA. Backup hasil ke Drive. |
| SDXL | 2–3 foto aktor + b-roll; tiap gambar 20–60 detik di T4. |
| Wav2Lip | Lip-sync pada foto statis. Kualitas di bawah avatar berbayar, tetapi gratis dan lokal. |
| Dubbing | Terjemahan per segmen, bukan voice cloning. Panjang audio disesuaikan agar sinkron. |
| Thumbnail | Komposisi frame asli + teks, bukan gambar AI generatif. Hasilnya konsisten dengan video. |
| Clip Generator | Butuh Gemini free tier atau LLM lokal. Tanpa itu, tahap pemilihan momen tidak bisa dijalankan. |
| YouTube URL | IP datacenter Colab sering ditolak YouTube. Kalau gagal, download video di komputer lalu upload ke dashboard. |
| Kuota Gemini | Free tier punya batas RPM/RPD. Kalau kena `429`, tunggu atau ganti key. |
| Publishing | Upload-Post free tier terbatas; tidak ada auto-post gratis tanpa layanan pihak ketiga. |

---

## 6. Lisensi model (penting)

| Komponen | Lisensi / catatan |
|---|---|
| **Wav2Lip** (repo + checkpoint) | **Hanya untuk riset, akademik, dan personal/non-komersial.** README upstream menyatakan penggunaan komersial dilarang. Untuk penggunaan komersial, set `WAV2LIP_DISABLE=1` (pipeline memakai still image + audio) atau ganti dengan model/layanan yang berlisensi komersial. |
| **RealVisXL V4.0** (SDXL) | CreativeML **OpenRAIL++-M**. Penggunaan komersial diizinkan dengan batasan penggunaan (baca kartu model di Hugging Face). |
| **edge-tts** | Menggunakan layanan suara Microsoft Edge, bukan API resmi. Gunakan secara wajar; jangan membuat layanan TTS publik berbasis volume besar. |
| **faster-whisper / Whisper** | Kode MIT; bobot model mengikuti lisensi OpenAI Whisper. |
| **OpenShorts** | MIT (lihat `LICENSE` di root repo). |

> Free mode ini cocok untuk eksperimen pribadi, konten sendiri, dan riset. Kalau hasilnya dipakai untuk klien atau iklan berbayar, periksa ulang lisensi Wav2Lip dan aktifkan fallback non-Wav2Lip.

---

## 7. Struktur project setelah dijalankan

```
/content/openshorts/
├── app.py                  # FastAPI + static dashboard
├── free_backends.py        # semua provider gratis/lokal
├── face_detection_compat.py # MediaPipe/OpenCV face detection
├── saasshorts.py           # pipeline AI Shorts
├── thumbnail.py            # YouTube Studio
├── dashboard/dist/         # hasil build Vite
├── output/                 # video, thumbnail, audio (ephemeral)
├── uploads/                # sumber upload (ephemeral)
├── third_party/Wav2Lip/    # repo + checkpoint Wav2Lip
├── .cache/huggingface/     # cache model SDXL/Whisper
├── .env                    # konfigurasi runtime (gitignored)
├── server.log              # log uvicorn
└── .server.pid
```

---

## 8. Update project / memakai fork sendiri

Default notebook memakai repo <https://github.com/ardyhim/openshorts.git> dan **tidak perlu upload ZIP**.

- Untuk mengambil update terbaru: jalankan ulang cell **4** dengan `UPDATE_PROJECT=True`.
- Untuk clone ulang dari nol (menghapus folder project): set `RESET_PROJECT=True`, lalu jalankan cell **4**.
- Untuk fork sendiri: ganti `REPO_URL` di cell **2** dengan URL fork Anda. Pastikan fork berisi `free_backends.py` dan `colab/requirements-colab.txt`.
- `REPO_BRANCH` default `main`.

`git reset --hard` saat update hanya membuang perubahan pada file yang dilacak git. `.env`, `output/`, `uploads/`, dan cache model tidak dihapus. Folder lama dari ZIP yang belum punya `.git` otomatis diubah menjadi git repo oleh cell **4**, jadi tidak perlu clone ulang dari nol.

---

## 9. Troubleshooting

| Masalah | Penyebab / solusi |
|---|---|
| `GPU tidak terdeteksi` | Runtime belum T4. `Runtime` → `Change runtime type` → `T4 GPU`, lalu ulang dari cell 1. |
| `mediapipe` / `numpy` error | MediaPipe hanya dipasang di Python <= 3.12. Di Colab Python 3.13, app otomatis memakai OpenCV Haar (`face_detection_compat.py`), jadi `mediapipe` tidak diperlukan. Jangan `pip install mediapipe` versi terbaru: `mp.solutions` sudah dihapus. |
| `pip install` gagal / `ResolutionImpossible` | Jalankan ulang cell 4 dengan `UPDATE_PROJECT=True` lalu cell 5. `requirements-colab.txt` memakai `py3langid==0.2.2` dan marker NumPy/OpenCV per versi Python. Versi 0.3.0 memaksa NumPy 2 dan bentrok dengan Python <= 3.12. |
| `CUDA out of memory` | Restart runtime, turunkan `IMAGE_STEPS` ke 12–16, jalankan satu video sekaligus. |
| SDXL lambat / download macet | Jalankan cell 8 `PRELOAD_SDXL=True`; cek koneksi dan sisa disk. |
| Wav2Lip gagal | Pipeline otomatis memakai still image + audio. Cek `server.log`. |
| Gemini `404` | Jalankan ulang cell 6 dengan key; model dipilih dari daftar yang tersedia. |
| Gemini `429` | Kuota free tier habis. Tunggu beberapa menit/hari. |
| `yt-dlp` gagal | Upload file video langsung, bukan URL YouTube. |
| Dashboard tidak muncul | Cek cell 12 (`server.log`), pastikan cell 10 sukses dan port 8000 hidup. |
| `npm ci` gagal | Pastikan Node.js >= 18. Cell 3 memasang Node 22 bila perlu. |
| Hasil hilang | Jalankan cell 14 (backup ke Drive) sebelum runtime berhenti. |

---

## 10. Keamanan dan privasi

- **Jangan commit `.env`.** File itu berisi API key dan hanya dibuat di runtime.
- **Jangan bagikan URL Cloudflared.** Dashboard self-host tidak punya autentikasi.
- **Jangan memproses video sensitif di tunnel publik.** Gunakan Colab proxy.
- Video/audio tersimpan sementara di `/content`; Google dapat menghapus runtime kapan saja sesuai kebijakan Colab.
- API key Gemini/Upload-Post hanya dikirim ke layanan masing-masing; tidak ada server OpenShorts pusat dalam mode ini.

---

## 11. Menjalankan tanpa Colab (ringkas)

Free mode juga bisa jalan di Linux dengan GPU/CPU:

```bash
# Python 3.11–3.13, ffmpeg, Node.js >= 18
# 1) Pasang torch/torchvision DULU sesuai CUDA Anda (jangan pakai pin requirements.txt
#    kalau versi tersebut tidak cocok). Contoh untuk CUDA 12.4:
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124

# 2) Dependensi aplikasi + free mode. File ini sengaja tidak menyentuh torch.
pip install -r colab/requirements-colab.txt
# MediaPipe hanya untuk Python <= 3.12; Python 3.13 memakai OpenCV Haar.
python - <<'PY'
import subprocess, sys
if sys.version_info < (3, 13):
    subprocess.run([sys.executable, "-m", "pip", "install", "--no-deps", "mediapipe==0.10.21"], check=True)
else:
    print("Python 3.13: memakai OpenCV Haar, mediapipe dilewati.")
PY

# 3) Dashboard + server
cd dashboard && npm ci && npm run build && cd ..
FREE_MODE=1 SERVE_DASHBOARD=1 \
  WHISPER_MODEL=small WHISPER_DEVICE=cuda WHISPER_COMPUTE=float16 \
  uvicorn app:app --host 0.0.0.0 --port 8000
```

Tanpa GPU, ganti `WHISPER_DEVICE=cuda`/`WHISPER_COMPUTE=float16` dengan
`WHISPER_DEVICE=cpu`/`WHISPER_COMPUTE=int8`; SDXL akan sangat lambat atau bisa
dimatikan dengan `FREE_IMAGE_MODEL=""` (AI Shorts memakai fallback gambar polos).
Di Windows/Linux, `pip install -r colab/requirements-colab.txt` memilih NumPy/OpenCV
otomatis: Python <= 3.12 memakai NumPy 1.x + OpenCV 4.10, Python 3.13 memakai
NumPy 2.x + OpenCV 4.12. Pastikan torch/torchvision yang sudah ada kompatibel.
Cara paling aman tetap Colab (semua dependensi diuji untuk Linux/T4).
