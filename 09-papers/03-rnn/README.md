# 03-rnn - 循环神经网络

本目录收录循环神经网络(RNN)及其变体的经典论文，理解序列建模的基本原理。

## 论文清单

| 序号 | 论文 | 年份 | 核心贡献 |
|------|------|------|----------|
| 1 | Backpropagation Through Time: What It Does and How to Do It | 1990 | RNN训练算法BPTT |
| 2 | Long Short-Term Memory (LSTM) | 1997 | 解决长程依赖问题 |
| 3 | Learning Phrase Representations using RNN Encoder-Decoder | 2014 | Seq2Seq架构 |
| 4 | Neural Machine Translation by Jointly Learning to Align and Translate | 2015 | Attention机制 |

## 学习路径

### 入门顺序
1. **BPTT** → 理解RNN如何训练
2. **LSTM** → 解决梯度消失问题
3. **Seq2Seq** → 序列到序列学习
4. **Attention** → 注意力机制（Transformer的前置）

### 关键概念
- 循环连接和隐藏状态
- 梯度消失/爆炸
- 长短期记忆
- 编码器-解码器架构

## 为什么需要RNN？

### 序列数据的挑战

**传统神经网络的局限**:
- 输入输出维度固定
- 无法处理变长序列
- 没有记忆能力

**序列数据例子**:
- 文本（单词序列）
- 语音（音频帧序列）
- 时间序列（股票、天气）
- DNA序列

### RNN的优势

- **变长处理**: 可处理任意长度序列
- **参数共享**: 每个时间步使用相同参数
- **记忆能力**: 隐藏状态保存历史信息

## RNN的演进

### 第一代: 简单RNN (1980s-1990s)
- 基础循环结构
- BPTT训练算法
- 梯度消失问题严重

### 第二代: LSTM/GRU (1997-2014)
- 门控机制
- 解决长程依赖
- 成为序列建模标准

### 第三代: Seq2Seq + Attention (2014-2017)
- 编码器-解码器架构
- 注意力机制
- 机器翻译突破

### 第四代: Transformer (2017-至今)
- 自注意力机制
- 完全并行
- 取代RNN成为主流

## 阅读建议

1. **理解原理**: RNN的核心是循环和隐藏状态
2. **掌握问题**: 梯度消失是RNN的根本挑战
3. **学习解决**: LSTM通过门控机制解决
4. **了解演进**: Attention和Transformer的发展

## 现代意义

虽然Transformer已成为主流，但RNN仍有价值：
- **轻量级应用**: 移动端、嵌入式
- **流式处理**: 实时序列处理
- **特定领域**: 某些时序任务
- **理解基础**: 理解序列建模的演变

---

**RNN是理解序列建模的必经之路！**
