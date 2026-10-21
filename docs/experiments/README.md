# 实验归档

以下报告与原始数据记录合并前的受控实验，配置名、默认值和命令均对应各自实验分支。main 已删除用于无运动唤醒降频、直接 NV12→RGB、核显 VPP 缩放、反向联合编译等对照的实验代码。重现历史对照请检出对应分支；当前支持的配置见项目 README。

| 范围 | 报告 | 历史分支 |
| --- | --- | --- |
| 跟踪、身份特征复用 | [结果](results.md) | `experiment/temporal-tracking` |
| 两检测模型联合编译 | [结果](cocompile-results.md) | `experiment/cocompile` |
| 跟踪与联合编译组合 | [结果](combined-results.md) | `experiment/combined-optimizations` |
| 运动、空场景及视频解码 | [结果](motion-vaapi-results.md) | `experiment/motion-vaapi` |

所有实验使用 pn41 的 N6005 与单块 PCIe Coral。素材主要是公开图片合成视频，报告中的一致性对比不能代表真实摄像头准确率。main 的最终合并验证单独记录在 `docs/final-validation.md`。
