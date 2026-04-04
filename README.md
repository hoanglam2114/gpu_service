# gpu_service

Flask API server chạy trên GPU — fine-tuning, evaluation, inference, clustering.

## Cấu trúc

```
gpu_service/
├── main.py                    # Entrypoint Flask
├── start.sh                   # One-command launcher
├── Dockerfile
├── docker-compose.yml
├── requirements.txt
├── .env.example
└── app/
    ├── config.py              # Đọc biến môi trường
    ├── state.py               # Global in-memory state
    ├── callbacks.py           # Training callbacks
    ├── routers/               # Flask Blueprints
    │   ├── train.py           # POST /api/train/start, GET /status, POST /stop
    │   ├── eval.py            # POST /api/eval/start, GET /status, GET /result
    │   ├── infer.py           # POST /api/model/load, POST /api/infer/stream
    │   ├── system.py          # GET /api/system/resources, POST /api/cluster
    │   └── worker.py          # GET /api/worker/status
    └── services/
        ├── gpu.py             # release_gpu_memory()
        ├── training.py        # LoRA fine-tuning (unsloth + SFTTrainer)
        ├── evaluation.py      # BLEU/ROUGE + Claude LLM judge
        ├── inference.py       # SSE streaming + think-tag filter
        ├── rag.py             # DuckDuckGo web search + RAG
        └── clustering.py      # DBSCAN + KMeans conversation clustering
```

---

## Yêu cầu

| Thứ | Yêu cầu |
|-----|---------|
| GPU | NVIDIA, VRAM ≥ 16 GB (T4 / RTX 3090 trở lên) |
| OS  | Windows 10/11 (WSL2) hoặc Linux Ubuntu 22.04 |
| Docker Desktop | ≥ 4.x (Windows) **hoặc** Docker Engine (Linux) |
| NVIDIA Driver | ≥ 525 (Windows) hoặc ≥ 525 (Linux) |

---

## Cài đặt — Windows (Docker Desktop)

Docker Desktop trên Windows hỗ trợ GPU qua WSL2 backend với NVIDIA GPU Paravirtualization, không cần cài nvidia-container-toolkit riêng.

### Bước 1 — Cài NVIDIA Driver trên Windows (host)

Tải driver tại: https://www.nvidia.com/Download/index.aspx

> ⚠️ Chỉ cài driver trên Windows — **không** cài CUDA trong WSL hay trong container.

Kiểm tra driver sau khi cài:
```powershell
nvidia-smi
```

### Bước 2 — Cài Docker Desktop

Tải tại: https://www.docker.com/products/docker-desktop

Sau khi cài, vào **Settings → General** → bật **"Use the WSL 2 based engine"**.

### Bước 3 — Bật WSL2 Integration

Vào **Settings → Resources → WSL Integration** → bật toggle cho distro Ubuntu của bạn.

### Bước 4 — Xác nhận GPU hoạt động trong Docker

Mở terminal (PowerShell hoặc WSL), chạy:
```bash
docker run --rm --gpus all nvidia/cuda:12.1.0-base-ubuntu22.04 nvidia-smi
```
Nếu thấy output `nvidia-smi` là OK.

### Bước 5 — Chạy gpu_service

```bash
# Trong WSL hoặc PowerShell tại thư mục gpu_service/
cp .env.example .env
# Mở .env, điền NGROK_TOKEN và ANTHROPIC_API_KEY

bash start.sh
# hoặc:
docker compose up --build
```

---

## Cài đặt — Linux (Docker Engine)

```bash
# 1. Cài nvidia-container-toolkit
curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey | sudo gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg
curl -s -L https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list | \
    sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' | \
    sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list
sudo apt-get update && sudo apt-get install -y nvidia-container-toolkit
sudo nvidia-ctk runtime configure --runtime=docker
sudo systemctl restart docker

# 2. Chạy service
cp .env.example .env && nano .env
bash start.sh
```

---

## Khởi động không dùng Docker (local Python)

```bash
# Yêu cầu: Python 3.11, CUDA driver đã cài sẵn
bash start.sh --local
```

---

## API endpoints

### Training
| Method | Path | Mô tả |
|--------|------|--------|
| POST | `/api/train/start` | Bắt đầu fine-tuning job (multipart/form-data) |
| GET  | `/api/train/status/:job_id` | Lấy trạng thái + logs |
| POST | `/api/train/stop/:job_id` | Gửi tín hiệu dừng |

### Evaluation
| Method | Path | Mô tả |
|--------|------|--------|
| POST | `/api/eval/start` | Bắt đầu eval job (multipart/form-data) |
| GET  | `/api/eval/status/:eval_job_id` | Progress + stage detail |
| GET  | `/api/eval/result/:eval_job_id` | Kết quả đầy đủ sau COMPLETED |

### Inference
| Method | Path | Mô tả |
|--------|------|--------|
| POST | `/api/model/load` | Load model vào GPU |
| POST | `/api/infer/stream` | SSE streaming inference |
| GET  | `/api/infer/logs` | Danh sách inference logs |
| GET  | `/api/infer/logs/:id` | Log của một inference cụ thể |

### System
| Method | Path | Mô tả |
|--------|------|--------|
| GET  | `/api/system/resources` | VRAM, GPU util, eval slots |
| GET  | `/api/worker/status` | Trạng thái train + eval workers |
| POST | `/api/cluster` | Phân cụm conversations |
| POST | `/api/cluster/filter` | Lọc mẫu theo centroid threshold |
| DELETE | `/api/cluster/cache` | Xoá clustering cache |

---

## Biến môi trường (.env)

| Biến | Bắt buộc | Mô tả |
|------|----------|--------|
| `NGROK_TOKEN` | Có | Token ngrok để tạo public URL |
| `ANTHROPIC_API_KEY` | Có | API key cho Claude LLM judge |
| `BACKEND_URL` | Không | URL Node.js backend (mặc định localhost:3000) |
| `FLASK_PORT` | Không | Port Flask (mặc định 5000) |
| `GPU_EVAL_SLOTS` | Không | Số eval job chạy song song (mặc định 3) |
| `NUM_CLUSTERS` | Không | Số cluster KMeans (mặc định 7) |

---

## Logs và debug

```bash
# Xem logs container đang chạy
docker compose logs -f

# Vào shell trong container
docker compose exec gpu_service bash

# Kiểm tra GPU trong container
docker compose exec gpu_service nvidia-smi
```
