# 自动调制识别（Automatic Modulation Recognition）

## 项目简介
自动调制识别（AMR）是无线通信中的关键任务，旨在通过分析无线电信号来识别其调制方式。随着无线通信技术的快速发展，传统的信号处理方法已无法满足高效识别的需求，因此深度学习技术逐渐成为AMR领域的研究热点。本项目实现了一个基于深度学习的自动调制识别模型，通过对输入的无线信号进行多层次特征提取、时间序列建模和重要特征聚焦，准确地对不同的调制方式进行分类。

## 数据集

RML2016.10a数据集，包含11种不同调制类型的信号样本。信号以复数形式存储，包含不同信噪比（SNR）条件下的样本。

## 模型架构

本项目的模型结构包括：
- **特征提取器**：特征提取部分使用多个残差块，通过卷积操作提取信号的局部特征，同时通过 SE 模块提升通道间的特征表示能力。
- **序列建模器**：BILSTM用于捕获信号序列中的时间依赖性。
- **多头注意力机制**：多头注意力机制用于进一步增强模型对输入序列中重要特征的聚焦能力，选择性地关注对分类任务最重要的时刻特征。

## 结果分析

结果表明，在高信噪比条件下，模型的分类准确率可稳定90%以上，而在低信噪比下，尽管准确率较低，仍然实现了一个小幅度提升。实验数据已保存在results.json文件，您可以直接运行data_processor.py文件得到可视化结果。当然，本项目由于个人精力有限，还未进行严格的对照实验，所以请**谨慎参考**。
