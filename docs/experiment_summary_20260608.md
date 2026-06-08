# CarbonBench + HLS 图像实验阶段性总结

日期：2026-06-08

## 目标

当前目标不是简单把遥感 CNN 加到 CarbonBench 上，而是判断 HLS spatial context 是否能改善 tower-level GPP prediction，尤其是 lower-tail / difficult towers。

对标参考是 CarbonBench Transformer 的 GPP per-site R2：

- p25 / median / p75：0.311 / 0.709 / 0.804

## 当前统一实验设置

- 数据范围：global all-IGBP
- Split：Koppen tower-level split
- 时间序列：30-day Transformer
- 任务：multi-task GPP / RECO / NEE
- 图像：monthly HLS patch manifest
- 图像输入：6-band HLS patch, 67 x 67 pixels
- 图像 context：site-level multi-patch pooling, K=8
- 评估：tower-level per-site R2，重点看 GPP p25 / median / p75

图像分支不是 daily image sequence。当前实现是对每个 tower 取多个 HLS patches，用 ResNet18 编码后做 masked mean pooling，形成 site-level image context，再与 tabular + temporal Transformer feature 融合。

## 关键结果

| 模型 | GPP overall R2 | GPP site R2 p25 / median / p75 | 解释 |
| --- | ---: | --- | --- |
| no-image Transformer | 0.6554 | 0.2800 / 0.6128 / 0.7659 | patch-covered subset 上的公平 no-image control |
| CNN site-pool K=8 concat | 0.6836 | 0.4772 / 0.5831 / 0.7898 | 显著提高 p25 和 overall，但 median 下降 |
| CNN gated residual fusion | 0.6780 | 0.3932 / 0.6108 / 0.8112 | 最均衡，p25 提高且 median 基本接近 no-image |
| CNN hierFiLM no-MoE | 0.6809 | 0.1883 / 0.5825 / 0.7974 | 不推荐作为主线 |
| CNN hierFiLM + MoE | 0.6730 | 0.1892 / 0.6530 / 0.7878 | median 最高，但 p25 明显下降 |
| CarbonBench Transformer reference | - | 0.311 / 0.709 / 0.804 | 原论文参考结果 |

## 主要发现

1. HLS 图像确实带来信息，但不是简单全面提升。CNN concat 把 GPP p25 从 0.2800 提高到 0.4772，说明它明显帮助了一部分困难站点；但 median 从 0.6128 降到 0.5831，说明图像也会干扰一部分中等站点。

2. Gated residual fusion 是目前最稳的结构。它保留了 no-image Transformer 的稳定性，同时允许 CNN 学一个较小的 image correction。结果 p25 提高到 0.3932，median 维持在 0.6108，p75 达到 0.8112。

3. MoE 有提升 median 的潜力，但会损害 lower-tail robustness。MoE 的 test median 为 0.6530，是当前 CNN 模型中最高的；但 p25 只有 0.1892，说明 routing/expert 机制可能偏向主流或容易站点，不适合作为当前主模型。

4. 当前还没有超过 CarbonBench Transformer 的 median 0.709。更准确的阶段性结论是：HLS spatial context 目前最有价值的信号在 lower-tail tower improvement，而不是整体超越 CarbonBench median。

## 建议下一步

优先方向：

1. 保留 no-image Transformer 作为强 baseline。
2. 以 gated residual fusion 作为主线，而不是继续盲目加更复杂 MoE。
3. 导出 per-site metrics，分析哪些 tower、IGBP 类型、Koppen 气候区被 CNN 改善或变差。
4. 加入 site-balanced 或 lower-tail-aware objective，让训练目标更照顾困难 tower。
5. 用多个 seeds/splits 检查 worst towers 是否稳定，再做 landscape structure / fragmentation / SHDI / edge density 分析。

一句话总结：

HLS 图像目前没有直接把 median 推过 CarbonBench Transformer，但已经显示出改善困难站点的信号；后续研究重点应从“继续堆模型结构”转向“稳定识别 difficult towers，并解释哪些生态系统/气候区从 spatial context 中受益”。
