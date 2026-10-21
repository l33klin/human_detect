import json
import sys
from pathlib import Path
from types import ModuleType

import flatbuffers
import numpy as np
import pytest
import tflite

from human_detect.runtime import EngineUnavailable, ModelBundle, TpuModel, audit_model


def minimal_model(path, builtin):
    builder = flatbuffers.Builder(256)
    tflite.OperatorCodeStart(builder)
    tflite.OperatorCodeAddBuiltinCode(builder, builtin)
    tflite.OperatorCodeAddDeprecatedBuiltinCode(builder, min(builtin, 127))
    opcode = tflite.OperatorCodeEnd(builder)
    tflite.OperatorStart(builder)
    tflite.OperatorAddOpcodeIndex(builder, 0)
    operation = tflite.OperatorEnd(builder)
    tflite.SubGraphStartOperatorsVector(builder, 1)
    builder.PrependUOffsetTRelative(operation)
    operations = builder.EndVector()
    tflite.SubGraphStart(builder)
    tflite.SubGraphAddOperators(builder, operations)
    graph = tflite.SubGraphEnd(builder)
    tflite.ModelStartSubgraphsVector(builder, 1)
    builder.PrependUOffsetTRelative(graph)
    graphs = builder.EndVector()
    tflite.ModelStartOperatorCodesVector(builder, 1)
    builder.PrependUOffsetTRelative(opcode)
    codes = builder.EndVector()
    tflite.ModelStart(builder)
    tflite.ModelAddVersion(builder, 3)
    tflite.ModelAddOperatorCodes(builder, codes)
    tflite.ModelAddSubgraphs(builder, graphs)
    root = tflite.ModelEnd(builder)
    builder.Finish(root, file_identifier=b"TFL3")
    path.write_bytes(builder.Output())


def test_reject_cpu_neural_operations_and_uncompiled_models(tmp_path):
    path = tmp_path / "model.tflite"
    minimal_model(path, tflite.BuiltinOperator.CONV_2D)
    with pytest.raises(EngineUnavailable, match="prohibited CPU operation CONV_2D"):
        audit_model(path)
    minimal_model(path, tflite.BuiltinOperator.RESHAPE)
    with pytest.raises(EngineUnavailable, match="no compiled Edge TPU"):
        audit_model(path)


def test_actual_pinned_models_are_tpu_compiled():
    root = Path(__file__).resolve().parents[1] / "models"
    manifest = json.loads((root / "manifest.json").read_text())
    for role, item in manifest.items():
        path = root / item["file"]
        if not path.exists():
            pytest.skip("run python scripts/download_models.py for artifact validation")
        report = audit_model(path)
        assert report["sha256"] == item["sha256"]
        assert report["host_ops"] == (
            [] if role == "embedding" else ["DEQUANTIZE", "TFLite_Detection_PostProcess"]
        )


def install_fake_runtime(monkeypatch, load_delegate, interpreter):
    package = ModuleType("tflite_runtime")
    module = ModuleType("tflite_runtime.interpreter")
    module.load_delegate = load_delegate
    module.Interpreter = interpreter
    monkeypatch.setitem(sys.modules, "tflite_runtime", package)
    monkeypatch.setitem(sys.modules, "tflite_runtime.interpreter", module)
    monkeypatch.setattr("human_detect.runtime.audit_model", lambda path: {"sha256": "fake"})


def test_delegate_load_failure_never_constructs_cpu_interpreter(monkeypatch):
    calls = []

    def delegate(name, options):
        calls.append((name, options))
        raise RuntimeError("missing device")

    def interpreter(**kwargs):
        pytest.fail("CPU interpreter must not be constructed after delegate failure")

    install_fake_runtime(monkeypatch, delegate, interpreter)
    with pytest.raises(EngineUnavailable, match="missing device"):
        TpuModel("unused", "pci:0", "embedding")
    assert calls == [("libedgetpu.so.1", {"device": "pci:0"})]


def test_quantized_embedding_input_and_output_contract(monkeypatch):
    delegate = object()

    class Interpreter:
        def __init__(self, **kwargs):
            assert kwargs["experimental_delegates"] == [delegate]
            self.invocations = 0

        def allocate_tensors(self):
            pass

        def get_input_details(self):
            return [dict(index=0, shape=np.array([1, 112, 96, 3]), dtype=np.uint8)]

        def get_output_details(self):
            return [dict(index=1, shape=np.array([1, 3]), quantization=(0.1, 121))]

        def set_tensor(self, index, value):
            self.input = value

        def invoke(self):
            self.invocations += 1

        def get_tensor(self, index):
            return np.array([[131, 121, 121]], dtype=np.uint8)

    install_fake_runtime(monkeypatch, lambda *a: delegate, Interpreter)
    model = TpuModel("unused", "pci:0", "embedding")
    image = np.full((300, 300, 3), 200, dtype=np.uint8)
    np.testing.assert_allclose(model.embed(image), [1, 0, 0])
    assert model.dimension == 3
    assert model.interpreter.input.shape == (1, 112, 96, 3)
    assert np.all(model.interpreter.input == 200)  # no accidental double-normalization
    assert model.interpreter.invocations == 2  # startup warmup + embedding


def test_ssd_coordinates_class_filter_and_invalid_boxes():
    model = object.__new__(TpuModel)
    model.run = lambda rgb: [
        np.array([[[0.1, -0.1, 1.1, 0.5], [0, 0, 1, 1], [0.8, 0.8, 0.1, 0.1]]]),
        np.array([[0, 1, 0]]),
        np.array([[0.9, 0.99, 0.8]]),
        np.array([3]),
    ]
    detected = model.detect(np.zeros((100, 200, 3), dtype=np.uint8), 0.5, class_id=0)
    assert len(detected) == 1
    assert detected[0][0].model_dump() == {"x1": 0, "y1": 10, "x2": 100, "y2": 100}


def test_bundle_initialization_failure_releases_already_loaded_models(settings, monkeypatch):
    import hashlib

    settings.models_dir = settings.database.parent / "models"
    settings.models_dir.mkdir()
    manifest = {}
    for role in ("person", "face", "embedding"):
        path = settings.models_dir / f"{role}.tflite"
        path.write_bytes(role.encode())
        manifest[role] = dict(file=path.name, sha256=hashlib.sha256(path.read_bytes()).hexdigest())
    (settings.models_dir / "manifest.json").write_text(json.dumps(manifest))
    closed = []

    class Model:
        def __init__(self, path, device, role):
            if role == "face":
                raise EngineUnavailable("missing TPU")

        def close(self):
            closed.append("person")

    monkeypatch.setattr("human_detect.runtime.TpuModel", Model)
    with pytest.raises(EngineUnavailable, match="missing TPU"):
        ModelBundle(settings)
    assert closed == ["person"]
