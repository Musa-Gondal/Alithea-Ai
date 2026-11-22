"""Helper for running the pipeline on Modal GPUs (development/benchmark only; not used by graders).

    modal run modal_download.py                 # fetch the 50 FineVideo parquet files into the Volume
    modal run modal_app.py::sh --cmd "python run.py --input /data/finevideo/data --output /data/out/results.jsonl"
"""
import subprocess

import modal

app = modal.App("video-indexing")
vol = modal.Volume.from_name("finevideo", create_if_missing=True)

image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("libgl1", "libglib2.0-0", "procps", "util-linux")
    .pip_install_from_requirements("requirements.txt")
    .pip_install("huggingface_hub[hf_transfer]")
    .run_commands(
        "mkdir -p /weights && python -c \"import urllib.request;"
        "urllib.request.urlretrieve('https://github.com/ultralytics/assets/releases/download/v8.3.0/yolo11l.pt',"
        "'/weights/yolo11l.pt')\""
    )
    .env({"YOLO_WEIGHTS": "/weights/yolo11l.pt", "HF_HUB_ENABLE_HF_TRANSFER": "1", "YOLO_OFFLINE": "1"})
    .add_local_dir(".", "/repo", ignore=["weights", ".git", "*.jsonl", "*.gz", "__pycache__", "ai_chats"])
)


@app.function(image=image, volumes={"/data": vol}, gpu="L40S", cpu=16, memory=65536, timeout=6 * 3600)
def sh(cmd: str):
    subprocess.run("mkdir -p /data/out", shell=True)
    p = subprocess.run(f"cd /repo && {cmd}", shell=True)
    vol.commit()
    return p.returncode


@app.local_entrypoint()
def main(cmd: str = "nvidia-smi; nproc; free -g; lscpu | head -20"):
    print(sh.remote(cmd))
