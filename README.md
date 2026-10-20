# Human Detect

本分支为检测模型联合编译实验。见 [实验说明与复现命令](docs/experiments/cocompile.md)、[对照协议](docs/experiments/protocol.md)和[实测结果](docs/experiments/results.md)。默认仍使用原模型。

在 Ubuntu Server 上使用 **Coral M.2 / PCIe Edge TPU**，提供人体检测、人脸检测和已录入人员的身份匹配。支持 RTSP、HTTP(S) 视频流、USB/V4L2 摄像头和静态图片，输出 JSON。

视频处理链路：

```text
视频解码 → 最新帧 → TPU 人体检测 → 人体区域内 TPU 人脸检测
                                          ↓
                               TPU 人脸特征 → 身份库匹配 → JSON
```

- 三个模型都显式加载 Edge TPU delegate。缺失设备、加载失败或推理失败时返回错误，**没有 CPU 推理回退**。
- 启动前校验模型 SHA-256、检查计算图并进行真实 TPU 预热；拒绝未编译模型和遗留在 CPU 上的卷积等计算。
- CPU 负责视频解码、图像缩放/裁剪、SSD 框解码/NMS、反量化和小规模向量比较。当前三个模型只有两个检测模型保留反量化和 SSD 后处理算子。
- 原始图片、视频帧只在内存中处理。SQLite 仅保存 ID、姓名、特征向量和模型指纹；结果不含图像或特征向量。
- 单个进程串行访问 TPU。每路视频保留至多一个待处理帧，处理不及时丢弃旧帧；所有输入共享有界推理队列。
- 视频断线自动重连；采集进程卡住时重建；删除视频源会回收进程。重启服务后需重新添加视频源。

## Ubuntu 部署

推荐 Docker，以隔离 Coral 旧版运行时与 Ubuntu 系统 Python。镜像使用 Python 3.9 和 Coral 发布的 `tflite-runtime 2.5.0.post1`；Python 3.9 已停止上游维护，这里是运行时兼容性约束。宿主机只需要 PCIe 驱动和 Docker。不要直接将系统 Python 的 `tflite-runtime` 升到最新版本替代该运行时。

先检查设备：

```bash
lspci -nn | grep -i '1ac1:089a'
lsmod | grep -E 'apex|gasket'
ls -l /dev/apex_0
```

若驱动尚未安装，按 [Coral PCIe 官方说明](https://www.coral.withgoogle.com/docs/m2/get-started/)安装；不要在已正常运行的服务器重复安装驱动。

```bash
cp .env.example .env

# 必须是设备所属组的数字 GID；容器以普通用户运行。
export HD_APEX_GID=$(stat -c '%g' /dev/apex_0)

# 只构建，不启动，构建时下载固定版本模型。
docker compose build

# 三个模型分别在真实 TPU 上完成一次调用；成功后才启动。
docker compose run --rm human-detect human-detect doctor --invoke
docker compose up -d
curl http://127.0.0.1:8000/health/ready
```

运行期间不需要联网。身份库保存在 `identities` Docker volume。容器根文件系统只读，临时目录是 tmpfs。

局域网访问：在 `.env` 中设置 `HD_BIND_ADDRESS=0.0.0.0`、`HD_APEX_GID=<设备组 GID>` 和 `HD_API_KEY=<自定义口令>`，再启动 Compose。携带 `Authorization: Bearer <口令>` 调用所有接口；`/health/live` 无需口令。默认仅监听本机，口令为空。

USB 摄像头需在 `compose.yaml` 的 `devices` 中增加 `/dev/video0:/dev/video0`，并在 `group_add` 中增加该设备所属组的数字 GID。RTSP/HTTP 视频流不需要映射摄像头设备。服务只使用一个 Uvicorn worker，请勿增加 worker 数量共享同一块 TPU。

宿主机原生运行时，请先准备兼容的 Python 3.9：

```bash
python3.9 -m venv .venv
.venv/bin/pip install .
.venv/bin/pip install --extra-index-url https://google-coral.github.io/py-repo/ \
  'tflite-runtime==2.5.0.post1'
python3 scripts/download_models.py
.venv/bin/human-detect doctor --invoke
.venv/bin/human-detect serve
# 局域网：.venv/bin/human-detect serve --host 0.0.0.0
```

宿主机需已安装 `libedgetpu1-std`，运行用户需能读写 `/dev/apex_0`。

## API

图片使用**原始二进制请求体**，不要使用 multipart。这样上传内容不会被框架暂存到磁盘。

### 录入身份

```bash
curl -X POST 'http://127.0.0.1:8000/v1/identities/alice/samples?name=Alice' \
  -H 'Content-Type: image/jpeg' --data-binary @alice.jpg
```

样本中必须恰好有一张足够大的人脸。可给同一 ID 多次录入不同角度、光照的样本；新样本追加，姓名更新。注册照可仅包含人脸，不要求检测到全身。

```json
{"person_id":"alice","name":"Alice","samples":1}
```

```bash
curl http://127.0.0.1:8000/v1/identities
curl -X DELETE http://127.0.0.1:8000/v1/identities/alice
```

### 静态图片

```bash
curl -X POST http://127.0.0.1:8000/v1/images/analyze \
  -H 'Content-Type: image/jpeg' --data-binary @photo.jpg
```

示例结构，数值仅作格式说明：

```json
{
  "captured_at": "2026-10-20T10:00:00Z",
  "width": 1920,
  "height": 1080,
  "persons": [{
    "box": {"x1": 200, "y1": 100, "x2": 700, "y2": 1050},
    "confidence": 0.94,
    "faces": [{
      "box": {"x1": 350, "y1": 130, "x2": 480, "y2": 290},
      "confidence": 0.91,
      "identity": {
        "status": "matched",
        "person_id": "alice",
        "name": "Alice",
        "similarity": 0.82
      }
    }]
  }],
  "processing_ms": 40.5
}
```

坐标为原图像素坐标，右下角 `x2/y2` 不包含边界像素；图片先应用 EXIF 旋转，坐标和 `width/height` 对应旋转后的图像。人体未找到人脸时 `faces=[]`；未发现人体时 `persons=[]`，不继续检测全图人脸。

身份状态：`matched`、`unknown`、`face_too_small`。`similarity` 是余弦相似度，不是概率；没有身份库或未提取特征时为 `null`。未知人员的 `person_id/name` 为 `null`。

### 视频流

```bash
curl -X POST http://127.0.0.1:8000/v1/streams \
  -H 'Content-Type: application/json' \
  -d '{"source":"rtsp://user:password@192.168.1.50:554/live","fps":5}'

# USB 摄像头索引使用整数，例如 {"source":0,"fps":5}
```

返回 `stream_id`，随后查询或订阅：

```bash
curl http://127.0.0.1:8000/v1/streams
curl http://127.0.0.1:8000/v1/streams/<stream_id>

# 每行一个 JSON，包含状态、sequence、processed、skipped 和 latest 分析结果。
curl -N http://127.0.0.1:8000/v1/streams/<stream_id>/events

curl -X DELETE http://127.0.0.1:8000/v1/streams/<stream_id>
```

`events` 使用 NDJSON；没有新结果时约每 10 秒返回一次状态快照。订阅慢时跳过旧结果，不无限积压。用 `sequence` 判断是否为新分析结果。`skipped` 仅统计推理队列拒绝的帧，不包含采集侧主动丢弃的帧。源 URL 和认证信息不会出现在响应或应用日志中。

`fps` 是分析速率上限，不是视频解码帧率，也不是性能保证。多人画面需要逐人体检测人脸并提取特征；一块 TPU 轮换三个模型会有开销。初始建议单路、5 FPS，再根据服务器实测调整。当前不跨帧追踪，也不凭衣着或背影延续某个人的身份；每次身份结果都来自当帧人脸匹配。

状态包括 `connecting`、`running`、`reconnecting`、`busy`、`failed`、`stopped`。断线时 `latest` 保留最后成功结果，请同时检查状态和 `captured_at`，不要当作当前画面。

### 错误与健康状态

| HTTP 状态 | 含义 |
| --- | --- |
| 401 | API 口令不正确 |
| 413 | 图片字节数或像素数超限 |
| 422 | 无效图片、参数或注册样本 |
| 429 | 推理队列或视频源数量已满 |
| 503 | TPU 推理失败或不可用 |
| 504 | 推理任务超时 |

`/health/live` 检查服务进程；`/health/ready` 检查初始化和已记录的推理故障，并返回模型校验信息。就绪检查不会额外执行推理。TPU 推理异常或正在执行的推理超时后服务锁定为不可用，检查硬件后需重启；不会静默恢复到 CPU。已排队但尚未开始的请求超时会尝试取消。已经开始执行的注册请求无法强制撤销，超时后应查询身份库确认是否完成。

默认上传上限 10 MiB、像素上限 2000 万；视频帧也受像素上限约束。其他配置见 [.env.example](.env.example)。

## 模型与识别效果

固定版本、下载地址和 SHA-256 见 [models/manifest.json](models/manifest.json)。来源说明见 [models/NOTICE.md](models/NOTICE.md)。

| 用途 | 模型 | 输入 | CPU 图算子 |
| --- | --- | --- | --- |
| 人体 | SSDLite MobileDet COCO，类别 0 | RGB uint8，320×320 | 反量化、检测后处理 |
| 人脸 | SSD MobileNet V2 Face | RGB uint8，320×320 | 反量化、检测后处理 |
| 身份特征 | MobileFaceNet，128 维 | RGB uint8，宽 96×高 112 | 无 |

推理输入按模型上游示例使用 RGB 原始 uint8 像素；输出特征先反量化，再 L2 归一化。人脸裁剪增加 10% 边距。匹配时按身份取最佳样本的余弦相似度，同时要求与次佳身份有足够差距。没有人脸关键点对齐，因此较大的侧脸、遮挡、模糊和小脸可能明显降低效果。

默认相似度阈值 `0.65`、身份间隔 `0.08` 只是初始值，需使用你的真实摄像头、已知人和陌生人样本校准。更换特征模型或预处理时必须重建身份库；服务会阻止继续使用模型指纹不一致的旧数据库。更换检测模型还需要确认输出契约；当前适配的是 manifest 中的 SSD 四输出模型。

## 开发与测试

普通开发机无需 TPU 即可运行逻辑测试；真实推理只能在有 Coral 的机器运行：

```bash
python3.11 -m venv .venv
.venv/bin/pip install -e '.[dev]'
python3 scripts/download_models.py
.venv/bin/pytest -q
.venv/bin/ruff check human_detect scripts tests
.venv/bin/ruff format --check human_detect scripts tests
```

测试替身仅用于测试，不作为生产服务的 CPU 回退。真实硬件验证用 `human-detect doctor --invoke`。

可选完整链路烟测（需要 dev 依赖和真实 TPU）：

```bash
.venv/bin/python scripts/validate_hardware.py --seconds 8 --fps 5
# 或用自己的单人、单脸样例；图片只进入内存：
.venv/bin/python scripts/validate_hardware.py --image /path/to/sample.jpg
```

不提供 `--image` 时，烟测从 Coral 官方仓库读取公开示例图，临时开启本机 HTTP MJPEG 流，测试结束后自动关闭；不会启动常驻服务。`pn41` 的驱动配置和实测性能见 [验证记录](docs/pn41-validation.md)。
