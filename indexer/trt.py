"""Optional TensorRT backend for YOLO11l.

`build_engine` (called from prepare.py, untimed) exports the fused PyTorch model to ONNX with
dynamic batch/height/width and builds an FP16 TensorRT engine for this GPU. `TRTModel` runs it
on torch CUDA tensors and returns the same (B, 84, N) tensor as the PyTorch model.
"""
from __future__ import annotations

from pathlib import Path

import torch

MAX_BATCH = 32
OPT_SHAPE = (MAX_BATCH, 3, 384, 640)
MIN_SHAPE = (1, 3, 64, 64)
MAX_SHAPE = (MAX_BATCH, 3, 640, 640)


def export_onnx(weights: str, onnx_path: str):
    from ultralytics import YOLO

    model = YOLO(weights).model.fuse(verbose=False).eval().float()
    for m in model.modules():  # same export switches ultralytics uses
        if m.__class__.__name__ == "Detect":
            m.export = True
            m.format = "onnx"
            m.dynamic = True
    x = torch.zeros(1, 3, 384, 640)
    torch.onnx.export(
        model, x, onnx_path, opset_version=17, input_names=["images"], output_names=["output0"],
        dynamic_axes={"images": {0: "batch", 2: "height", 3: "width"}, "output0": {0: "batch", 2: "anchors"}},
        do_constant_folding=True, dynamo=False,
    )


def build_engine(weights: str, engine_path: str, workspace_gb: int = 8):
    import tensorrt as trt

    onnx_path = str(Path(engine_path).with_suffix(".onnx"))
    export_onnx(weights, onnx_path)
    logger = trt.Logger(trt.Logger.WARNING)
    builder = trt.Builder(logger)
    network = builder.create_network(0)
    parser = trt.OnnxParser(network, logger)
    with open(onnx_path, "rb") as f:
        if not parser.parse(f.read()):
            raise RuntimeError("ONNX parse failed: " + "; ".join(str(parser.get_error(i)) for i in range(parser.num_errors)))
    config = builder.create_builder_config()
    config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, workspace_gb << 30)
    config.set_flag(trt.BuilderFlag.FP16)
    profile = builder.create_optimization_profile()
    profile.set_shape("images", MIN_SHAPE, OPT_SHAPE, MAX_SHAPE)
    config.add_optimization_profile(profile)
    blob = builder.build_serialized_network(network, config)
    if blob is None:
        raise RuntimeError("TensorRT engine build failed")
    with open(engine_path, "wb") as f:
        f.write(blob)


class TRTModel:
    def __init__(self, engine_path: str, device: torch.device):
        import tensorrt as trt

        self.device = device
        self.logger = trt.Logger(trt.Logger.ERROR)
        with open(engine_path, "rb") as f:
            self.engine = trt.Runtime(self.logger).deserialize_cuda_engine(f.read())
        if self.engine is None:
            raise RuntimeError("could not deserialize engine")
        self.context = self.engine.create_execution_context()
        self.in_name, self.out_name = "images", "output0"
        self.in_dtype = torch.float16 if self.engine.get_tensor_dtype(self.in_name) == trt.DataType.HALF else torch.float32
        self.out_dtype = torch.float16 if self.engine.get_tensor_dtype(self.out_name) == trt.DataType.HALF else torch.float32

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        x = x.to(self.in_dtype).contiguous()
        self.context.set_input_shape(self.in_name, tuple(x.shape))
        out_shape = tuple(self.context.get_tensor_shape(self.out_name))
        out = torch.empty(out_shape, dtype=self.out_dtype, device=self.device)
        self.context.set_tensor_address(self.in_name, x.data_ptr())
        self.context.set_tensor_address(self.out_name, out.data_ptr())
        if not self.context.execute_async_v3(torch.cuda.current_stream(self.device).cuda_stream):
            raise RuntimeError("TensorRT execution failed")
        return out
