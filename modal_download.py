"""Download the first 50 FineVideo parquet files into the Modal Volume (needs Modal secret 'huggingface').

    modal run modal_download.py::download
"""
import subprocess

import modal

app = modal.App("video-indexing-download")
vol = modal.Volume.from_name("finevideo", create_if_missing=True)
image = (modal.Image.debian_slim(python_version="3.11")
         .pip_install("huggingface_hub[hf_transfer]")
         .env({"HF_HUB_ENABLE_HF_TRANSFER": "1"}))


@app.function(image=image, volumes={"/data": vol}, secrets=[modal.Secret.from_name("huggingface")],
              timeout=3 * 3600, cpu=8)
def download():
    subprocess.run(
        ["hf", "download", "HuggingFaceFV/finevideo", "--repo-type", "dataset",
         "--include", "data/train-000[0-4]*", "--local-dir", "/data/finevideo"],
        check=True,
    )
    vol.commit()
    subprocess.run("ls /data/finevideo/data | wc -l; du -sh /data/finevideo", shell=True)
