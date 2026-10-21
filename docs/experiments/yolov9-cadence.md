# YOLOv9 与完整检测频率对比

2026-10-21，在 `experiment/yolov9-detection-cadence` 分支、pn41 上验证。结论：**提高完整检测频率不能解决持续的桌面误检；YOLOv9 的分数需要单独校准。** 在本片段上，YOLOv9、5 Hz、人体阈值 0.3 减少了桌面误检，同时接近 25 FPS，但仍漏掉右侧人物的少量帧，因此没有替换默认模型或全局阈值。

## 方法与硬件

- Intel N6005、PCIe Coral、Ubuntu 22.04，沿用已有驱动。临时只读容器以 UID 10001 运行，Python 3.9、PyAV 12.3、FFmpeg 5.1.9、iHD 23.1.1，MediaMTX 1.21.2。服务器其他服务保持原状。
- 单路强制 VAAPI 解码与 H.264 CQP 24 编码。卷积等模型计算只在 TPU 上运行，没有 CPU 神经网络回退。
- 人体分别使用原 MobileDet 和固定的 YOLOv9-s ReLU6 INT8 320×320、17 类版本；person 均为类别 0。人脸、身份模型及其阈值保持不变。两者均采用 RGB 原图直接 resize 至 320×320，和 YOLO 模型作者公开的 benchmark 预处理一致。
- 六组主实验保持人体阈值 0.5，比较目标频率 3、5、8 Hz；另外三组诊断比较 YOLO 的 0.4、0.3 阈值及关闭空场景降频。不同模型相同数值的阈值不代表相同误检率或召回率。
- YOLO 编译文件 SHA-256：`49a38e77439b475d0d259a6d9fe7ea32fcaa58de3748499dd120aceebad9c591`。图审计的 `host_ops` 为 `[]`，实际 TPU 调用通过。输出为 `[1,2100,17]` 类别 logits 和 `[1,2100,64]` 框分布，主机执行反量化、sigmoid、DFL 框解码与 NMS；NMS IoU 阈值 0.4。这些是显式后处理，不是将未编译的卷积等算子放到 CPU。
- 模型来源及权重：[v1.0 发布页](https://github.com/dbro/frigate-detector-edgetpu-yolo9/releases/tag/v1.0)。本项目实现输出适配，不导入旧 Frigate 插件；当前 Frigate 已原生支持该模型类型。来源及许可说明见 [NOTICE](../../models/NOTICE.md)。

输入为 `videoplayback (1).mp4`，1280×720、25 FPS、376 帧、15.04 秒、H.264、无音轨。SHA-256：`be41233b75eebf9a7432ecd75e772ddf0ef2c1fae596720a44ff850172ce185a`。

分两种验证：完整视频离线导出保留全部帧，按源视频时间轴驱动检测和跟踪，用于画面比较；实时测试循环原片、以原速复制到回环 RTSP，预热后测量约 35 秒，按输出计数差计算 FPS。**离线导出的处理速度不作为实时容量。** 实时组顺序执行，每组独占 TPU；本轮没有多次随机交错重复，细小 FPS 差异不能判定为显著提升。各组测量快照均为 running、单一 source_epoch，没有观测到中途重连。

## 实时性能

| 人体模型 | 人体阈值 | 目标完整检测 Hz | 实测完整检测 Hz | 平均输出 FPS | 测量期间主动丢帧 |
| --- | --- | --- | --- | --- | --- |
| MobileDet | 0.5 | 3 | 3.16 | 24.74 | 0 |
| MobileDet | 0.5 | 5 | 5.41 | 24.58 | 0 |
| MobileDet | 0.5 | 8 | 6.61 | 24.38 | 9 |
| YOLOv9 | 0.5 | 3 | 3.36 | 25.05 | 0 |
| YOLOv9 | 0.5 | 5 | 5.23 | 25.00 | 0 |
| YOLOv9 | 0.5 | 8 | 5.77 | 25.05 | 0 |
| YOLOv9 | 0.4 | 5 | 5.08 | 24.96 | 0 |
| YOLOv9，关闭空场景降频 | 0.4 | 5 | 6.64 | 24.99 | 0 |
| YOLOv9 | 0.3 | 5 | 5.16 | 24.91 | 0 |

目标频率不是硬性上限：现有策略每 `ceil(源帧率/目标频率)` 帧定期完整检测，25 FPS 对应间隔 9、5、4 帧，即定期约 2.78、5、6.25 Hz；最大刷新间隔仍为 0.5 秒。跟踪失败、空结果等情况会提前完整检测，确认空场景则可能降频至 2 秒复查。因此用人体检测器实际调用次数计算实测 Hz，而不以配置值或快照模式占比代替。

关闭空场景策略后，原有跟踪会对空结果尽快重新检测，实际调用频率可能高于 5 Hz。略高于 25 的短窗口输出也不代表制造了额外独立视频帧。图像传输、跟踪、绘制、模型切换与编码都会影响整个链路，不能将公开的单模型推理毫秒数直接换算成本项目的视频 FPS。

## 完整视频的目标覆盖与误检观察

人工查看原片每秒抽样及输出画面后，选取两个固定区域进行可复查的输出统计：右侧持续可见的人物区域 `[1010,245,1195,595]`，桌面物品区域 `[580,420,825,705]`。某帧有任意人体框与区域的 IoU ≥ 0.3 即记为命中。它衡量两个固定区域的输出覆盖，**不是全场景的准确率或召回率**，不评价其他走动人物及玻璃反射。

| 配置／文件前缀 | 右侧人物区域有框／376 帧 | 桌面区域有框／376 帧 | 完整导出的实际检测次数 |
| --- | --- | --- | --- |
| `mobile-3` | 376 | 205 | 45 |
| `mobile-5` | 376 | 199 | 82 |
| `mobile-8` | 376 | 198 | 101 |
| `yolo-3`，阈值 0.5 | 227 | 0 | 54 |
| `yolo-5`，阈值 0.5 | 238 | 0 | 78 |
| `yolo-8`，阈值 0.5 | 237 | 0 | 87 |
| `yolo-5-t04`，阈值 0.4 | 313 | 0 | 76 |
| `yolo-5-t04-noidle`，阈值 0.4、关闭空场景降频 | 335 | 0 | 106 |
| `yolo-5-t03`，阈值 0.3 | 365 | 0 | 77 |

MobileDet 提高频率后，桌面误检只小幅减少，说明主要问题是持续的模型误判。YOLO 在这些区域上减少了误检，但阈值 0.5 时人物漏检较多；提高频率并未显著改善。误检率与置信分数的关系不能跨模型直接套用。

YOLO 将人物漏检为空场景后，现有策略会复用空结果并等待运动或周期唤醒，延长漏检。关闭该策略在 0.4 阈值下恢复了 22 帧，但还不能消除所有漏检。阈值 0.3 的组仅剩 11 帧未覆盖右侧人物区域，合计约 0.44 秒；桌面区域保持无框，不过尚未证明整个画面都没有其他误检。

全部九组完整视频的人脸输出均为 0，人体模型替换未解决本片段的人脸漏检，也没有验证人物身份识别的提升。

## 交付与复现

本地 `data/review/yolo-comparison/` 保存九组原尺寸、25 FPS、376 帧的 MP4 及逐帧 JSON。所有单路成片使用真实 TPU 推理和 VAAPI 编解码，标签包含检测分数。`comparison-4up.mp4` 同步排列 MobileDet 3 Hz、MobileDet 5 Hz、YOLO 5 Hz/0.5、YOLO 5 Hz/0.3；每格保留原尺寸，顶部注明配置。拼图在本机转码，仅用于查看，不计入硬件性能测试。

仓库保留 [统计摘要](data/yolov9-cadence/summary.json) 和九组 `*-live.json` 原始实时记录；完整逐帧记录及其 SHA-256 留在本地，视频与图片不提交到仓库。离线 JSON 的 `source_frame.seconds` 对应原片时间；`captured_at` 使用固定时间轴锚点，不是原摄像头的实际拍摄日期。

在已准备 Coral、VAAPI 和字体的服务 Linux/Python 环境中：

```bash
python scripts/download_models.py --models models/yolov9

PYTHONPATH=. python scripts/export_annotated_video.py \
  --clip '/path/videoplayback (1).mp4' --models models/yolov9 \
  --inference-fps 5 --person-threshold 0.3 \
  --output data/review/yolo-5-t03.mp4

PYTHONPATH=. python scripts/validate_media.py \
  --mediamtx /path/to/mediamtx --sample /path/to/grace_hopper.bmp \
  --clip '/path/videoplayback (1).mp4' --models models/yolov9 \
  --inference-fps 5 --person-threshold 0.3 \
  --work /tmp/unique-validation-dir --seconds 35 --benchmark-only \
  --output data/review/yolo-5-t03-live.json
```

切换到 `--models models` 即为默认 MobileDet；调整 `--inference-fps` 为 3、5、8，阈值控制组用 `--person-threshold 0.5`，空场景消融追加 `--disable-empty-scene`。离线导出工具目前只支持无音轨视频；不覆盖已有输出文件。实时工具需要独立测试端口和单人脸参考录入样本，不能与生产实例争用同一块 TPU。

本地 Python 3.11、服务器 Python 3.9 均通过 63 项测试，包含 YOLO logits/DFL 坐标解码、类别过滤、NMS、无效数值和输出契约检查；ruff 与格式检查通过。最后的 YOLO 5 Hz/0.3 组还执行完整 RTSP 采样、停止／启动、服务重启、采集进程强杀恢复及删除验证，均通过；其余组使用 benchmark-only，没有为每组重复恢复流程。

建议将 YOLO 5 Hz/0.3 作为减少该桌面误检的候选交给用户查验。若优先保留右侧人物的连续检测，当前 MobileDet 在本片段上仍更完整；5 Hz 比 3 Hz 更新更频繁，而 8 Hz 已出现少量主动丢帧。所有结论局限于此 15 秒片段，暂不据此更换默认模型、阈值或现有部署。
