"""Download the first 50 FineVideo parquet files into the Modal Volume (needs Modal secret 'huggingface').

    modal run modal_download.py::download
"""
import subprocess

import modal

from modal_app import app, image, vol

@app.function(image=image, volumes={"/data": vol}, secrets=[modal.Secret.from_name("huggingface")],
              timeout=3 * 3600, cpu=8)
def download():
    subprocess.run(
        ["hf", "download", "HuggingFaceFV/finevideo", "--repo-type", "dataset",
         "--include", "data/train-000[0-4]*", "--local-dir", "/data/finevideo"],
        check=True,
    )
    vol.commit()
    subprocess.run("ls -la /data/finevideo/data | head; du -sh /data/finevideo", shell=True)

