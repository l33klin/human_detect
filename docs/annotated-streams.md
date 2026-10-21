# 带标注 RTSP 与网页观看

## 部署

第一阶段优先支持单路 H.264、1920×1080、25 FPS。输出保留输入尺寸和时间戳，尽量保持原帧率；实际帧率取决于人员数量、跟踪稳定性和解码、绘制、编码开销。神经网络只运行在 Coral；核显执行 VAAPI 解码与 H.264 编码，CPU 执行跟踪、绘制和音频转换。硬件路径失败不会自动切换软件路径。

在原有 Docker 部署基础上，配置 `.env`：

```dotenv
HD_BIND_ADDRESS=0.0.0.0
HD_APEX_GID=1004
HD_RENDER_GID=109
HD_MEDIA_PUBLIC_HOST=192.168.1.10
```

这里的 GID 必须分别通过 `stat -c '%g' /dev/apex_0` 和 `stat -c '%g' /dev/dri/renderD128` 获取；地址替换为服务器的局域网 IP 或可解析主机名。仅访问本机时可用 `127.0.0.1`。

```bash
docker compose -f compose.yaml -f compose.vaapi.yaml -f compose.media.yaml up --build -d
```

MediaMTX 固定为 `1.21.2`。应用通过容器网络发布 RTSP、代理 WebRTC 的 WHEP 信令；浏览器从同一个应用地址获取页面和 API。局域网需能访问服务器 TCP 8000、TCP 8554 和 UDP/TCP 8189；无需对外开放 MediaMTX HTTP 8889。网页访问 `http://<服务器地址>:8000/`，所有静态资源随包提供，无 CDN 依赖。

媒体 Compose 默认限制一路，以优先保证帧率。后续可通过 `HD_MEDIA_MAX_STREAMS` 扩容，实际容量仍需测量。

如需已验证的联合编译模型，再追加 `-f compose.stable.yaml`；先按 README 准备 `models/joint`。不要在没有准备模型时直接叠加该配置。

`HD_API_KEY` 启用时在网页“连接设置”填写，密钥只保存在当前标签页的 sessionStorage 中。该密钥保护 API 和 WHEP 信令；RTSP 权限由 MediaMTX 独立控制，当前配置适用于受信任的局域网。

## 使用

网页支持添加、启动、停止、删除 RTSP 视频流，观看带标注画面，查看输出帧率、音频状态和目标数，复制输出 RTSP 地址。播放器默认静音，可通过其音量控件打开声音。身份页支持上传样本、查看身份列表和删除身份；图片通过原始请求体上传，只提取特征，不保存原图。相同身份 ID 可追加样本。

通过 API 添加：

```bash
curl -X POST http://127.0.0.1:8000/v1/streams \
  -H 'Content-Type: application/json' \
  -d '{"name":"门口","source":"rtsp://user:password@camera/live","publish":true}'
```

如果启用口令，增加 `Authorization: Bearer <口令>`。响应立即返回 `stream_id`、`connecting` 状态和输出地址；需要等视频状态进入 `running` 后播放。连接断开时状态会转为 `reconnecting` 并重试。示例输出字段：

```json
{
  "stream_id": "<32位ID>",
  "publish": true,
  "output": {
    "path": "annotated/<32位ID>",
    "rtsp_url": "rtsp://192.168.1.10:8554/annotated/<32位ID>",
    "whep_url": "/v1/streams/<32位ID>/whep",
    "video_codec": "h264",
    "audio_codec": "opus"
  }
}
```

用 VLC 等客户端打开 `rtsp_url`，标注直接位于视频像素中。人体框为绿色、人脸框为黄色，身份显示姓名和 ID；未匹配显示“未知”，没有找到脸时显示“未检测到人脸”。网页播放同一份编码视频，不叠加另外一份 JSON 结果。

人体标签显示检测分数，例如“人体 0.605 | 未检测到人脸”；人脸框附近另显示“人脸 0.823”。分数保留三位小数，表示相应检测模型的置信分数，不是身份匹配相似度或校准后的概率。跟踪帧沿用最近一次完整检测的分数，不会重新计算检测置信分数。

源音频统一解码为 48 kHz 双声道 PCM，保留时间戳后编码为 Opus，兼容网页 WebRTC 和 MediaMTX RTSP。保留的是音频内容和时间关系，不是原编码字节；无音轨的输入不会生成虚构音轨。

```bash
curl http://127.0.0.1:8000/v1/streams/<ID>
curl -X POST http://127.0.0.1:8000/v1/streams/<ID>/stop
curl -X POST http://127.0.0.1:8000/v1/streams/<ID>/start
curl -X DELETE http://127.0.0.1:8000/v1/streams/<ID>
```

停止保留连接配置；删除移除配置并回收进程。服务重启后自动恢复此前运行的流，ID 和 RTSP 输出路径保持不变。配置保存在身份库同目录下的 `streams.sqlite3`，包括必要的摄像头凭据；API、前端和应用日志不返回原始源地址。Docker 数据卷同时保存身份库和连接配置，不保存图片或录像。

## 帧率、同步与结果语义

`publish:false` 延续原 JSON 模式，`fps` 是分析帧率上限，允许在 RGB 转换前丢帧。`publish:true` 不按这个值降低输出视频帧率：每个输出帧都有对应的跟踪或检测结果；`fps` 控制定期 TPU 检测的目标频率，跟踪失败、身份库更新、强制刷新期限到达时允许提前完整检测。标注流默认使用 `HD_MEDIA_INFERENCE_FPS=3`，25 FPS 视频约每 9 帧刷新一次；最长间隔仍受 `HD_TEMPORAL_DETECTION_SECONDS` 控制。可在创建流时传入 `fps:5` 提高定期检测频率，但可能降低输出帧率。JSON 模式仍默认 5 FPS。关闭 temporal 会逐帧运行完整 TPU 推理，通常明显降低可输出帧率。

原视频从一个连接读取音频与视频，通过 NUT 在 FFmpeg 与 Python 之间传递共同归零后的 PTS。每路只有一个共享 RGB 帧等待识别；对应结果返回后才绘制该帧，不把任意最新 JSON 画到另一帧上。单独的写入线程通过最多 4 个音视频包的队列执行封装，与下一帧识别重叠；队列持有各帧独立像素，不复用正在等待识别的共享缓冲。保留输入分辨率，不能通过降低输出尺寸自动改善小脸识别效果。

JSON 的 `latest.frame` 和 `media.frame` 提供：

| 字段 | 含义 |
| --- | --- |
| `source_epoch` | 本次输入连接的唯一标识，重连后变化 |
| `frame_id` | 当前连接内的视频帧序号；主动丢帧时有缺口 |
| `pts`、`time_base` | 解码后、共同归零时间轴上的视频 PTS 及其单位 |
| `seconds` | `pts × time_base`，不是摄像头的 UTC 时钟 |

`captured_at` 仍是服务端接收该帧的 UTC 时间。`latest` 是最新完成的分析结果，可能是跟踪或空场景复用，查看 `temporal` 的模式和年龄字段；断线时保留的结果不能当作实时画面。

`media.source_fps` 是输入名义帧率；`media.output_fps` 是最近至多 100 帧进入媒体写入队列的速率，不能代表浏览器实际接收帧率。`media.output_frames` 是当前连接中完成 NUT 封装、提交给编码器的帧数，不表示下游已收到这些帧。`media.frame` 对应刚完成分析的帧，可能尚在等待编码。`media.previous_frame_ms` 用于诊断前一处理帧的读取、推理等待、绘制和写入队列等待时间。`media.dropped_frames` 统计该连接因处理延迟主动丢弃的帧。重连后这些计数重新开始。

处理滞后超过 `HD_MEDIA_MAX_LAG_SECONDS`（默认 0.75 秒）时，丢弃落后的解码视频帧，优先恢复实时性；音频继续按原时间戳传递，避免随视频丢帧而断续。这个阈值不是端到端延迟保证；摄像头缓冲、关键帧等待、网络和浏览器缓冲也会影响延迟。繁忙时可以输出未标注的对应帧，JSON 标记 `busy`，避免误用旧身份。

## 编码与故障定位

pn41 的 iHD 驱动实测只提供 CQP 码率控制，故 VAAPI 使用 `-rc_mode CQP -qp 24`，关闭 B 帧。可通过 `HD_MEDIA_QP` 调整质量，数值越小画质越高、带宽通常越大；CQP 不承诺固定码率。`HD_MEDIA_BITRATE` 仅用于明确选择的 `HD_MEDIA_ENCODER=libx264` 软件编码模式，不参与自动回退。Compose 核显配置默认强制 VAAPI。

原生部署额外安装 `av==12.3.0`、FFmpeg、Intel VAAPI 驱动和中文字体；默认字体路径为 `/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc`，可以用 `HD_MEDIA_FONT` 指定替代字体。Docker 已包含依赖。缺少字体或解码、编码、发布失败会导致媒体重连；检查设备权限、编码支持、字体路径、MediaMTX 连接和输入地址。底层 FFmpeg 日志被抑制，以避免输出源凭据。

无 `HD_MEDIA_PUBLIC_HOST` 时，API 用请求主机名生成 RTSP 地址；Compose 必须显式设置该值，同时用于 WebRTC ICE。网页无法播放但 RTSP 正常时，优先检查这个地址及 8189 端口。API 能访问并不意味着 WebRTC 媒体端口也能访问。

## 验证

单路 1080p/25 FPS 有声样本、用户提供视频、前端与故障恢复的实测结果见 [验证记录](annotated-streams-validation.md)。

`tests/test_published.py` 覆盖帧结果对应、配置恢复、停启、共享内存清理、WHEP 信令代理及鉴权。真实硬件测试脚本使用独立 MediaMTX、公开样本生成的视频和临时数据库，默认单路，可用 `--streams 2` 扩展；`--clip` 可指定已有视频：

```bash
PYTHONPATH=. python scripts/validate_media.py \
  --mediamtx /path/to/mediamtx --sample /path/to/public-single-face.jpg \
  --models models --work /tmp/hd-media-check --output /tmp/hd-media-result.json
```

该脚本需要 dev 依赖、TPU 和核显设备访问权限，使用回环端口 18554、18889、18189。测试源码中不提供生产用的模拟识别或 CPU 神经网络回退。
