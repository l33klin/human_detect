# Model sources

Artifacts are downloaded without modifications from the pinned URLs in `manifest.json`.

- Person detector and face detector: Google Coral `test_data`, commit `104342d2d3480b3e66203073dac24f4e2dbb4c41`. Upstream repository license: [Apache 2.0](https://github.com/google-coral/test_data/blob/104342d2d3480b3e66203073dac24f4e2dbb4c41/LICENSE).
- Face embedding model: `NikolasChorianopoulos/face-recognition-on-the-google-coral-dev-board`, commit `03bd43ff94e9c83b604eb802bb191beb7f1fb185`. Upstream repository license: [MIT](https://github.com/NikolasChorianopoulos/face-recognition-on-the-google-coral-dev-board/blob/03bd43ff94e9c83b604eb802bb191beb7f1fb185/LICENSE). The upstream example supplies the compiled MobileFaceNet artifact and raw RGB uint8 preprocessing. The repository license is identified here as provenance; the artifact has not been independently audited for training-data provenance or recognition accuracy.

No upstream application source is included in this service. The service implements its own inference adapters and pipeline.

The optional `models/yolov9/manifest.json` profile pins the 320×320, 17-class
YOLOv9-s ReLU6 INT8 Edge TPU artifact from
[dbro/frigate-detector-edgetpu-yolo9 v1.0](https://github.com/dbro/frigate-detector-edgetpu-yolo9/releases/tag/v1.0).
Its SHA-256 is `49a38e77439b475d0d259a6d9fe7ea32fcaa58de3748499dd120aceebad9c591`.
The source repository declares MIT; its README identifies the underlying YOLOv9
model as GPL-licensed. The model binary is downloaded separately and is not
redistributed in this repository. Our decoder implements the published output
contract; it does not import or vendor Frigate/plugin code. Face and embedding
artifacts remain identical to the default profile.

The Docker image downloads the official [Coral runtime 20221024 archive](https://github.com/google-coral/libedgetpu/releases/tag/release-grouper), verifies SHA-256 `7f17d85e6f21ec861af3f9168043a7ec351b80e08c9f168013ab68e99410aee2`, and installs its standard (`throttled`) Linux library. The archive's binary LICENSE is preserved at `/usr/local/share/edgetpu/LICENSE`; it is distinct from the source repository license. The x86_64 library SHA-256 is `686c861ec1601a1dcb81bba698ec919f1ebe07720c189baf26a8f952aa261ced`, identical to the standard runtime validated on pn41. No runtime binary is stored in this repository.

Coral documentation:

- [Edge TPU model requirements](https://www.coral.withgoogle.com/docs/edgetpu/models-intro/)
- [Python delegate loading](https://www.coral.withgoogle.com/docs/edgetpu/tflite-python/)
- [M.2 / Mini PCIe setup](https://www.coral.withgoogle.com/docs/m2/get-started/)
