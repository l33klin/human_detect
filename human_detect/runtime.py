"""Explicit Edge TPU delegate and graph audit; CPU neural inference is prohibited."""

import hashlib
import json
from pathlib import Path

import cv2
import numpy as np
import tflite

from human_detect.identity import normalize
from human_detect.schemas import Box


class EngineUnavailable(RuntimeError):
    pass


def audit_model(path):
    data = Path(path).read_bytes()
    model = tflite.Model.GetRootAsModel(data, 0)
    names = {
        value: key for key, value in vars(tflite.BuiltinOperator).items() if type(value) is int
    }
    # Only tensor plumbing and SSD box decoding/NMS may execute outside the TPU.
    allowed = {"RESHAPE", "QUANTIZE", "DEQUANTIZE", "L2_NORMALIZATION"}
    found_tpu = False
    host_ops = []
    for graph_index in range(model.SubgraphsLength()):
        graph = model.Subgraphs(graph_index)
        for index in range(graph.OperatorsLength()):
            code = model.OperatorCodes(graph.Operators(index).OpcodeIndex())
            builtin = max(code.BuiltinCode(), code.DeprecatedBuiltinCode())
            if builtin == tflite.BuiltinOperator.CUSTOM:
                name = code.CustomCode().decode()
            else:
                name = names.get(builtin, f"unknown:{builtin}")
            if name == "edgetpu-custom-op":
                found_tpu = True
            elif name in allowed or name == "TFLite_Detection_PostProcess":
                host_ops.append(name)
            else:
                raise EngineUnavailable(f"{Path(path).name}: prohibited CPU operation {name}")
    if not found_tpu:
        raise EngineUnavailable(f"{Path(path).name}: no compiled Edge TPU operation")
    return {"sha256": hashlib.sha256(data).hexdigest(), "host_ops": sorted(set(host_ops))}


class TpuModel:
    def __init__(self, path, device, kind):
        self.audit = audit_model(path)
        self.kind = kind
        try:
            from tflite_runtime.interpreter import Interpreter, load_delegate

            self.delegate = load_delegate("libedgetpu.so.1", {"device": device})
            self.interpreter = Interpreter(
                model_path=str(path), experimental_delegates=[self.delegate], num_threads=1
            )
            self.interpreter.allocate_tensors()
            inputs = self.interpreter.get_input_details()
            if len(inputs) != 1:
                raise ValueError("expected one input tensor")
            self.input = inputs[0]
            shape = self.input["shape"].tolist()
            if len(shape) != 4 or shape[0] != 1 or shape[3] != 3:
                raise ValueError(f"expected NHWC RGB input, got {shape}")
            if self.input["dtype"] not in (np.uint8, np.int8):
                raise ValueError("only quantized int8/uint8 models are supported")
            self.height, self.width = shape[1:3]
            self.outputs = self.interpreter.get_output_details()
            # Fail at startup if the delegate/device cannot actually invoke the model.
            self.run(np.zeros((self.height, self.width, 3), dtype=np.uint8))
            if kind == "embedding":
                if len(self.outputs) != 1:
                    raise ValueError("expected a single embedding output")
                self.dimension = int(np.prod(self.outputs[0]["shape"]))
            elif len(self.outputs) != 4:
                raise ValueError("detectors require four SSD postprocess outputs")
        except Exception as exc:
            raise EngineUnavailable(f"failed to initialize {kind} on {device}: {exc}") from exc

    def run(self, rgb):
        image = cv2.resize(rgb, (self.width, self.height), interpolation=cv2.INTER_LINEAR)
        if self.input["dtype"] == np.int8:
            image = (image.astype(np.int16) - 128).astype(np.int8)
        self.interpreter.set_tensor(self.input["index"], image[np.newaxis])
        self.interpreter.invoke()
        values = []
        for detail in self.outputs:
            value = self.interpreter.get_tensor(detail["index"])
            scale, zero = detail["quantization"]
            if scale > 0:
                value = (value.astype(np.float32) - zero) * scale
            values.append(value)
        return values

    def detect(self, rgb, threshold, class_id=None):
        values = self.run(rgb)
        # SSD MobileNet/MobileDet output contracts used by the pinned manifest.
        if values[3].size == 1:
            boxes, classes, scores, count = values
        else:
            scores, boxes, count, classes = values
        boxes, classes, scores = boxes[0], classes[0], scores[0]
        count = min(int(count.reshape(-1)[0]), len(scores))
        height, width = rgb.shape[:2]
        result = []
        for index in range(count):
            score = float(scores[index])
            if not np.isfinite(score) or score < threshold:
                continue
            if class_id is not None and int(classes[index]) != class_id:
                continue
            raw = boxes[index]
            if not np.all(np.isfinite(raw)):
                continue
            y1, x1, y2, x2 = raw
            box = Box(
                x1=max(0, min(width, int(x1 * width))),
                y1=max(0, min(height, int(y1 * height))),
                x2=max(0, min(width, int(np.ceil(x2 * width)))),
                y2=max(0, min(height, int(np.ceil(y2 * height)))),
            )
            if box.x2 > box.x1 and box.y2 > box.y1:
                result.append((box, score))
        return sorted(result, key=lambda entry: entry[1], reverse=True)

    def embed(self, rgb):
        return normalize(self.run(rgb)[0])

    def close(self):
        # The interpreter must release its graph before unloading the delegate.
        self.interpreter = None
        self.delegate = None


class ModelBundle:
    def __init__(self, settings):
        manifest = json.loads((settings.models_dir / "manifest.json").read_text())
        models = {}
        try:
            for role in ("person", "face", "embedding"):
                item = manifest[role]
                path = settings.models_dir / item["file"]
                checksum = hashlib.sha256(path.read_bytes()).hexdigest()
                if checksum != item["sha256"]:
                    raise EngineUnavailable(f"checksum mismatch for {path.name}")
                models[role] = TpuModel(path, settings.device, role)
        except Exception:
            for model in models.values():
                model.close()
            raise
        self.person, self.face, self.embedding = (
            models[r] for r in ("person", "face", "embedding")
        )
        self.fingerprint = self.embedding.audit["sha256"] + ":rgb-raw-resize-margin0.1-v1"
        self.dimension = self.embedding.dimension
        self.report = {role: model.audit for role, model in models.items()}

    def close(self):
        for role in ("person", "face", "embedding"):
            getattr(self, role).close()
