# 💰 Financial Education LLM Assistant (Fin-LLM Platform)

### 🧠 Project Overview  
A **cloud-based financial education assistant** powered by a **fine-tuned Chinese financial LLM**, designed to provide domain-specific question answering, tutoring, and data interpretation in real time.  

The system integrates **Dify (as the frontend orchestration framework)**, **FastAPI-based broker service**, and **a local Ollama/vLLM instance** serving the **ssc-FinLLM (7B, FP16)** model.

It is fully **deployed on AWS EC2**, accessible from web and mobile browsers.

> A complete cloud-to-local LLM integration pipeline for financial education — enabling low-latency, real-time Chinese financial query responses.

---
<!-- #### User Interface -->
<!-- ![alt text](./doc_image/UI.png) -->
<!-- <img src="./doc_image/UI.png" alt="User Interface" width="1000"> -->

### ▶️ Real-time Demo Video (Click below image to see the demo video❗)
Video URL: https://reurl.cc/EQx22R

[![Real-time Demo Video ](./doc_image/UI.png)](https://drive.google.com/file/d/13gUPVPvrnMjAFcjswvMSbEEROo9Srip6/view?usp=sharing)
<!-- <video width="1000" controls>
  <source src="./doc_image/demo_video.mp4" type="video/mp4">
  你的瀏覽器不支援影片播放。
</video> -->


---

### ⚙️ System Architecture  

```text
[User: Web / Mobile Browser]
        │  HTTPS (80/443, WebSocket/SSE)
        ▼
[EC2 Nginx (Dify/Docker)]
        │  Reverse proxy → /api /console
        ▼
[Dify Web / Dify API Service]
        │  1) Workflow orchestration & UI logic
        │  2) Optional: Knowledge base recall (RAG)
        │       ├─> [VectorDB: pgvector / Weaviate]
        │       └─> Retrieve Top-K snippets → Inject into prompt
        │
        │  3) Calls OpenAI-compatible API via custom API Base
        ▼  HTTP(S)/SSE (OpenAI-compatible)
[Broker (EC2:8080)]
        │  A) Forwarding & Protocol Alignment:
        │     - /v1/models
        │     - /v1/chat/completions  (stream/non-stream)
        │  B) Optional: 繁↔簡 translation pipeline
        │  C) Retry / Timeout / Error wrapping
        ▼  HTTP(S)
[Tailscale / Intranet]
        ▼
[Local PC vLLM Server (port 8000)]
        │  - Loads ssc-FinLLM (7B, FP16, RTX 3090)
        │  - Handles Tokenizer / KV Cache / Inference
        │  - Streams tokens (SSE)
        ▼
<────────── token chunks（SSE）return via Broker ──────────
        ▲
[Broker translates back (if needed) and sends to Dify]
        ▲
[Dify API → Frontend (Web/App)]
        ▲
[EC2 Nginx → HTTPS Response]
        ▲
[User sees real-time LLM output]
```

*(Diagram above illustrates full cloud-to-local streaming pipeline.)*

---

### 💡 Key Features  
- Integrated **Dify**, **FastAPI Broker**, and **local vLLM/Ollama** into a seamless workflow  
- **Deployed full stack on AWS EC2**, including Docker-based Nginx reverse proxy  
- Implemented **streaming token inference (SSE)** for real-time LLM responses  
- Supported **繁體↔簡體 translation** pipeline for Traditional Chinese user experience  
- Designed a **modular backend** supporting scalable RAG extensions  
- Delivered a **production-ready service** accessible via mobile and desktop browsers  

---

### 🧩 Tech Stack  

| Category | Tools & Frameworks |
|-----------|--------------------|
| LLM Engine | vLLM / Ollama (ssc-FinLLM 7B FP16) |
| Frontend / Orchestration | Dify |
| Backend | FastAPI, Nginx, Broker |
| Cloud | AWS EC2 (Ubuntu 22.04) |
| Networking | Tailscale (intranet tunneling) |
| Database | pgvector / Weaviate (for RAG) |
| Communication | WebSocket, Server-Sent Events (SSE) |

---

### 📊 Results & Demo  
- **Deployment:** AWS EC2 + Local GPU (RTX 3090)  
- **Token Speed:** ~62 tokens/sec  
- **Concurrent Users:** 20-50 simulated  
- **Latency:** Sub-second streaming response per token  

---

### 🧭 What I Learned  
- Familiar with **Dify framework** for low-code LLM app orchestration  
- Built full-stack **LLM deployment pipeline (frontend → backend → inference)**  
- Learned **FastAPI + Broker architecture** for OpenAI-compatible API forwarding  
- Integrated **local GPU inference** (vLLM / Ollama) with **cloud services**  
- Gained hands-on experience with **RAG, vector DBs, and translation pipelines**  

---

### 📂 Repository Structure 

#### Repository Structure on AWS
```bash
financial-education-helper/
│
├── dify/                        # Main orchestration & frontend (Docker)
│   ├── web/                     # Web UI & workflow definition
│   ├── api/                     # REST API service
│   ├── sdks/                    # SDKs for model access
│   ├── docker/                  # Docker environment configs
│   ├── images/                  # Docker images & setup scripts
│   ├── LICENSE / Makefile       # Base project setup
│   └── README_*.md              # Dify multi-language guides
│
├── ssc-broker/                  # Custom FastAPI broker for model bridging
│   ├── broker.py                # Core logic: /v1/models & /v1/chat/completions
│   ├── requirements.txt         # Python dependencies
│   ├── docker-compose.yml       # Container orchestration
│   └── Dockerfile               # Broker container build file
│
└── test_dir/                    # Experimental or validation scripts
```

#### Repository Structure on Local PC (Files downloaded form ssc-FinLLM github website) 
```bash
~/models/ssc-finllm-model/
│
├── config.json                       # Model configuration (hidden_size, layers, etc.)
├── generation_config.json             # Sampling, temperature, top_p, max_length
├── modeling_internlm2.py              # InternLM-based model architecture
├── configuration_internlm.py          # Config class registration
│
├── pytorch_model-00001-of-00008.bin   # Model weights (split shards)
├── pytorch_model-00002-of-00008.bin
├── pytorch_model-00003-of-00008.bin
├── pytorch_model-00004-of-00008.bin
├── pytorch_model-00005-of-00008.bin
├── pytorch_model-00006-of-00008.bin
├── pytorch_model-00007-of-00008.bin
├── pytorch_model-00008-of-00008.bin
├── pytorch_model.bin.index.json       # Index mapping across shards
│
├── tokenizer.model                    # SentencePiece tokenizer
├── tokenizer_config.json              # Tokenizer settings
├── tokenization_internlm.py           # Tokenizer logic
├── special_tokens_map.json            # Special token mapping (<bos>, <eos>, etc.)
│
└── README.md                          # Model training or usage notes
```


---

### 📄 License & References  
- [ssc-FinLLM (7B) – Fine-tuned Chinese financial model](https://github.com/haidizym/ssc-FinLLM?tab=readme-ov-file)  
- [Dify Documentation](https://docs.dify.ai/)  
- [FastAPI](https://fastapi.tiangolo.com/)  
- [vLLM GitHub](https://github.com/vllm-project/vllm)  
- [Ollama](https://ollama.ai/)  

---

### ✨ Summary  
> A fully deployed financial education LLM system integrating Dify, FastAPI Broker, and local vLLM inference — enabling low-latency, real-time financial education in Traditional Chinese.
