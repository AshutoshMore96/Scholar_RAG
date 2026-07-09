"""
Modal GPU Ollama server for ScholarRAG "Deep Retrieval using GPU".

Serves an Ollama-compatible API (/api/chat, /api/generate) on a GPU, exposed as
a public Modal web endpoint. Point the local ScholarRAG server at it:

    OLLAMA_DEEP_URL=https://<workspace>--scholar-ollama-serve.modal.run
    OLLAMA_DEEP_MODEL=llama3.1:8b

Deploy:
    pip install modal && modal setup            # one-time auth
    modal deploy scholar-rag/modal/ollama_gpu.py
    # -> prints the web endpoint URL to use as OLLAMA_DEEP_URL

The model is baked into the image at build time, so containers start fast.
"""
import modal

# Model to serve on the GPU (bigger than the CPU default — the GPU can handle it).
MODEL = "llama3.1:8b"          # or "qwen2.5:7b", "mistral:7b", "llama3.1:70b" (needs a bigger GPU)
GPU = "A10G"                    # "T4" (cheapest), "A10G" (good for 7-8B), "A100" (for 70B)

image = (
    modal.Image.debian_slim()
    .apt_install("curl")
    .run_commands(
        "curl -fsSL https://ollama.com/install.sh | sh",
        # start ollama, pull the model, and bake it into the image layer
        f"bash -c 'ollama serve & sleep 6 && ollama pull {MODEL} && sleep 1'",
    )
    .env({"OLLAMA_KEEP_ALIVE": "-1", "OLLAMA_HOST": "0.0.0.0:11434"})
)

app = modal.App("scholar-ollama", image=image)


@app.function(
    gpu=GPU,
    timeout=600,
    scaledown_window=300,       # keep warm 5 min after last request, then scale to zero
    min_containers=0,           # set to 1 to keep one always-warm (costs while idle)
)
@modal.concurrent(max_inputs=8)  # handle several requests per container
@modal.web_server(port=11434, startup_timeout=180)
def serve():
    import subprocess
    subprocess.Popen(["ollama", "serve"])
