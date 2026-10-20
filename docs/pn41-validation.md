# pn41 硬件与服务验证

验证日期：2026-10-20。通过用户授权的 `ssh pn41` 完成。未修改驱动、系统 Python、现有容器或系统服务，未部署常驻服务。

## 系统与设备

| 项目 | 实测 |
| --- | --- |
| 系统 | Ubuntu Server 22.04.5 LTS，x86_64 |
| 内核 | 5.15.0-181-generic |
| PCIe TPU | `01:00.0`，`1ac1:089a` |
| 驱动 | `apex`、`gasket` 已加载 |
| 驱动包 | `gasket-dkms 1.0-18` |
| 设备节点 | `/dev/apex_0`，`root:apex`，权限 `0660` |
| 设备组 | 数字 GID `1004`；SSH 用户已在组内 |
| 运行时 | `libedgetpu1-std 16.0`，`libedgetpu.so.1` 可发现 |
| Python | 系统默认 3.10.12；另有 3.9.25 |
| Docker / Compose | 28.1.1 / v2.35.1 |
| 摄像头设备 | 未发现 `/dev/video*` |
| 验证前资源 | 可用内存约 1 GB，Swap 已使用 2 GB；磁盘可用约 106 GB |

验证前未发现 TPU 被占用。使用临时目录中的 Python 3.9 venv 安装项目和 Coral `tflite-runtime==2.5.0.post1`，没有安装系统包。

## 模型与测试

`human-detect doctor --invoke` 成功，三个固定 SHA-256 的模型都实际调用了 PCIe TPU：

- 人体模型：主计算在 TPU；主机仅保留 `DEQUANTIZE` 和 `TFLite_Detection_PostProcess`。
- 人脸模型：主计算在 TPU；主机仅保留相同两种后处理算子。
- 身份特征模型：没有额外主机算子。

服务器上 27 项自动化测试全部通过。本地 macOS/Python 3.11 上同一组 27 项测试也全部通过。

额外临时启动真实 Uvicorn HTTP 进程，仅绑定回环地址，验证了启动、存活检查、Bearer 鉴权、TPU 就绪检查；完成后进程已退出。Compose 配置在服务器上解析检查通过，尚未构建或启动正式容器。

## 端到端测试

执行 `scripts/validate_hardware.py --seconds 8 --fps 5`。使用 Google Coral 官方测试数据中的 `grace_hopper.bmp`，在内存中转换成连续 MJPEG 视频。经过真实 HTTP 解码、人体检测、人脸检测、特征提取和身份匹配，输出 JSON。

原图和视频帧尺寸均为 **512×600 像素（宽×高）**。人体检测输入为 320×320；人体区域裁剪后的人脸检测输入为 320×320；人脸特征模型输入为 96×112。

| 指标 | 实测结果 |
| --- | --- |
| 图片分析次数 | 20 |
| 示例图检测结果 | 1 人、1 张脸 |
| 图片处理 P50 | 61.32 ms |
| 图片处理 P95 | 63.88 ms |
| 同图录入后匹配 | 成功 |
| 删除身份后返回 unknown | 成功 |
| 视频目标分析速率 | 5 FPS |
| 视频观测时间 | 8.08 秒 |
| 视频成功处理帧数 | 36 |
| 含启动时间的平均速率 | 4.46 FPS |
| 推理队列丢帧 | 0 |
| 测试后服务就绪状态 | 正常 |

图片耗时是 pipeline 内部耗时，不含 HTTP 上传和图片解码。同图身份匹配仅证明链路连通，不代表真实识别准确率。视频是循环示例图的 HTTP MJPEG 流，没有实际测试 RTSP 摄像头、运动画面、多人吞吐或长期稳定性。

视频测试主动将分析速率上限设置为 5 FPS；4.46 FPS 包含启动时间，不代表最大吞吐。

## 部署依据

- 宿主驱动已正常，无需重新安装。
- Compose 设备组可用 `HD_APEX_GID=1004`。
- 当前 SSH 用户不在 Docker 组内，使用 Docker 需 `sudo`；例如在项目目录执行 `sudo env HD_APEX_GID=1004 docker compose build`，再按 README 做 `doctor --invoke` 和启动。也可将 `HD_APEX_GID=1004` 写进项目 `.env`，使用 `sudo docker compose ...`。
- 建议先单路、5 FPS；服务器现有内存压力较大。
- 真实摄像头接入后需校准人脸最小尺寸、身份相似度阈值和身份间隔，并观测实际延迟。
- 若直接在宿主机运行，使用隔离的 Python 3.9 环境；不要向系统默认 Python 3.10 安装不匹配的旧 wheel。
