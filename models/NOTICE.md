# Model sources

Artifacts are downloaded without modifications from the pinned URLs in `manifest.json`.

- Person detector and face detector: Google Coral `test_data`, commit `104342d2d3480b3e66203073dac24f4e2dbb4c41`. Upstream repository license: [Apache 2.0](https://github.com/google-coral/test_data/blob/104342d2d3480b3e66203073dac24f4e2dbb4c41/LICENSE).
- Face embedding model: `NikolasChorianopoulos/face-recognition-on-the-google-coral-dev-board`, commit `03bd43ff94e9c83b604eb802bb191beb7f1fb185`. Upstream repository license: [MIT](https://github.com/NikolasChorianopoulos/face-recognition-on-the-google-coral-dev-board/blob/03bd43ff94e9c83b604eb802bb191beb7f1fb185/LICENSE). The upstream example supplies the compiled MobileFaceNet artifact and raw RGB uint8 preprocessing. The repository license is identified here as provenance; the artifact has not been independently audited for training-data provenance or recognition accuracy.

No upstream application source is included in this service. The service implements its own inference adapters and pipeline.

Coral documentation:

- [Edge TPU model requirements](https://www.coral.withgoogle.com/docs/edgetpu/models-intro/)
- [Python delegate loading](https://www.coral.withgoogle.com/docs/edgetpu/tflite-python/)
- [M.2 / Mini PCIe setup](https://www.coral.withgoogle.com/docs/m2/get-started/)
