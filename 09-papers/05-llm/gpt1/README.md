# GPT: Improving Language Understanding by Generative Pre-Training

## 论文信息

- **标题**: Improving Language Understanding by Generative Pre-Training
- **作者**: Alec Radford, Karthik Narasimhan, Tim Salimans, Ilya Sutskever (OpenAI)
- **发表**: 2018 (技术报告)
- **链接**: [Paper](https://s3-us-west-2.amazonaws.com/openai-assets/research-covers/language-unsupervised/language_understanding_paper.pdf) | [Code](https://github.com/openai/finetune-transformer-lm) | [Blog](https://openai.com/research/language-unsupervised)

## 一句话总结

GPT首次提出"生成式预训练+判别式微调"的两阶段范式,证明了Transformer在无监督预训练后,通过简单微调即可在多种NLP任务上取得SOTA,开启了大语言模型预训练时代。

## 历史背景

### 2018年NLP的困境

**标注数据稀缺**:
- 有监督学习需要大量标注数据
- 标注成本高,耗时长
- 领域迁移困难

**无监督学习的挑战**:
- 词嵌入(Word2Vec, GloVe)是静态的
- 无法捕捉上下文信息
- ELMo使用RNN,难以并行

**半监督学习的探索**:
- 如何有效利用大量无标注文本?
- 预训练+微调成为研究方向

### Transformer的成熟

**Attention Is All You Need (2017)**:
- Transformer架构发布
- 强大的特征提取能力
- 可并行训练

**契机**:
- 用Transformer做预训练?
- 生成式预训练的可行性?

## 核心创新

### 1. 两阶段训练范式

**阶段一: 无监督预训练 (Unsupervised Pre-training)**

**目标**: 在大规模无标注文本上学习语言模型

**方法**:
```
给定文本序列 U = {u₁, u₂, ..., uₙ}
语言模型目标: L₁(U) = Σ log P(uᵢ | uᵢ₋ₖ, ..., uᵢ₋₁)
```

**特点**:
- 使用Transformer Decoder
- 单向语言模型(只看左边)
- 在大规模语料上训练

**预训练数据**:
- BooksCorpus: 7000本未发表书籍
- 约8亿词
- 长程依赖结构丰富

**阶段二: 有监督微调 (Supervised Fine-tuning)**

**目标**: 在特定任务的标注数据上微调

**方法**:
```
给定标注数据集 C = {(x₁, y₁), ..., (xₙ, yₙ)}
微调目标: L₂(C) = Σ log P(y | x₁, ..., xₙ)
```

**辅助目标**:
- 加入语言模型目标作为辅助
- 提高泛化能力,加速收敛
- 最终目标: L(C) = L₂(C) + λ × L₁(C)

### 2. 任务特定的输入变换

**挑战**:
- 不同NLP任务输入格式不同
- 分类、蕴含、相似度、问答
- 如何统一处理?

**解决方案**:

**分类任务**:
```
输入: [Start] 文本 [Extract]
输出: 线性层 → 类别
```

**文本蕴含**:
```
输入: [Start] 前提 [Delim] 假设 [Extract]
输出: 线性层 → 蕴含/矛盾/中性
```

**文本相似度**:
```
输入1: [Start] 文本1 [Delim] 文本2 [Extract]
输入2: [Start] 文本2 [Delim] 文本1 [Extract]
输出: 两个输出逐元素相加 → 线性层 → 类别
```

**问答/常识推理**:
```
输入: [Start] 上下文 [Delim] 问题 [Delim] 答案 [Extract]
输出: 线性层 → 正确/错误
```

**关键设计**:
- 使用特殊token标记结构
- 将各种任务转化为统一格式
- 最小化架构修改

### 3. 模型架构

**Transformer Decoder**:
```
输入嵌入 + 位置编码
    ↓
[Masked Multi-Head Attention]
    ↓
[Layer Norm + Residual]
    ↓
[Feed Forward]
    ↓
[Layer Norm + Residual]
    ↓
× 12层
    ↓
输出
```

**配置**:
- 层数: 12
- 注意力头数: 12
- 维度: 768
- FFN维度: 3072
- 参数量: 1.17亿

**与BERT的区别**:
- GPT: Decoder, 单向, 生成式
- BERT: Encoder, 双向, 判别式

## 实验结果

### GLUE基准测试

| 任务 | 之前SOTA | GPT微调 | 提升 |
|------|----------|---------|------|
| MNLI | 80.6% | 82.1% | +1.5% |
| QQP | 66.1% | 70.3% | +4.2% |
| QNLI | 82.3% | 88.1% | +5.8% |
| SST-2 | 93.2% | 91.3% | -1.9% |
| CoLA | 35.0% | 45.4% | +10.4% |
| STS-B | 81.0% | 82.0% | +1.0% |
| MRPC | 86.0% | 82.3% | -3.7% |
| RTE | 61.7% | 56.0% | -5.7% |

**总体**: 9/12任务超过SOTA

### 其他任务

**问答 (SQuAD)**:
- F1: 75.4% (之前SOTA: 71.4%)

**常识推理 (Story Cloze Test)**:
- 准确率: 86.5% (之前SOTA: 77.6%)
- 提升近9个百分点!

### 消融实验

**辅助目标的重要性**:
| 配置 | MNLI | QQP | QNLI |
|------|------|-----|------|
| 仅微调 | 81.1% | 69.0% | 87.4% |
| 辅助LM目标 | 82.1% | 70.3% | 88.1% |

**预训练的作用**:
- 随机初始化 vs 预训练权重
- 预训练在所有任务上都有显著提升
- 证明无监督预训练的有效性

## 关键发现

### 1. 零样本能力

**无微调表现**:
- 某些任务上,预训练模型已有一定能力
- 无需任何标注数据就能工作
- 暗示了更大模型的潜力

### 2. 迁移学习能力

**层特征分析**:
- 底层: 词法信息 (词性、语法)
- 中层: 句法信息 (句法结构)
- 高层: 语义信息 (语义角色、指代)

**可视化**:
- 注意力权重显示学到的语言结构
- 不同头关注不同方面

### 3. 预训练规模效应

**数据量影响**:
- 更多预训练数据 → 更好的微调性能
- 证明大规模预训练的价值

**模型大小**:
- 1.17亿参数在当时已算大模型
- 为后续GPT-2、GPT-3的规模扩展铺路

## 与BERT的对比

| 特性 | GPT | BERT |
|------|-----|------|
| 架构 | Transformer Decoder | Transformer Encoder |
| 方向 | 单向 (左→右) | 双向 |
| 预训练目标 | 语言模型 | Masked LM + NSP |
| 适用任务 | 生成任务 | 理解任务 |
| 发布时间 | 2018年6月 | 2018年10月 |
| 参数 | 1.17亿 | 3.4亿 (large) |

**互补性**:
- GPT擅长生成
- BERT擅长理解
- 两者共同推动了预训练范式

## 后续发展

### GPT系列演进

**GPT-2 (2019)**:
- 15亿参数
- WebText数据集 (40GB)
- 展示零样本能力
- 因"太危险"延迟发布

**GPT-3 (2020)**:
- 1750亿参数
- 少样本学习突破
- 上下文学习
- 通用人工智能曙光

**GPT-4 (2023)**:
- 多模态
- 更强推理能力
- ChatGPT基础

### 影响与意义

**范式转变**:
- 从"从头训练"到"预训练+微调"
- 从"任务特定"到"通用模型"
- 从"有监督"到"无监督预训练"

**工业界应用**:
- 搜索引擎
- 智能写作
- 对话系统
- 代码生成

## 局限性和问题

### 1. 单向限制

**问题**:
- 只能看到左边上下文
- 无法利用未来信息
- 某些任务(如完形填空)劣势

**对比**:
- BERT双向,理解能力更强
- ELMo双向LSTM

### 2. 生成质量

**问题**:
- 长文本生成连贯性差
- 容易重复
- 事实准确性不高

### 3. 偏见和毒性

**问题**:
- 继承训练数据偏见
- 可能生成不当内容
- 缺乏价值观对齐

## 代码实现要点

```python
import torch
import torch.nn as nn
from transformers import GPT2Config, GPT2LMHeadModel

# GPT使用Transformer Decoder
class GPT(nn.Module):
    def __init__(self, vocab_size, d_model=768, nhead=12, 
                 num_layers=12, dim_feedforward=3072):
        super().__init__()
        
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.pos_encoding = PositionalEncoding(d_model)
        
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            batch_first=True
        )
        self.transformer_decoder = nn.TransformerDecoder(
            decoder_layer,
            num_layers=num_layers
        )
        
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)
    
    def forward(self, input_ids):
        # 嵌入 + 位置编码
        x = self.embedding(input_ids)
        x = self.pos_encoding(x)
        
        # 生成因果mask (上三角为-inf)
        seq_len = input_ids.size(1)
        mask = torch.triu(torch.ones(seq_len, seq_len), diagonal=1).bool()
        
        # Transformer Decoder
        output = self.transformer_decoder(x, x, tgt_mask=mask)
        
        # 语言模型头
        logits = self.lm_head(output)
        
        return logits

# 预训练
# 目标: 预测下一个词
# loss = CrossEntropyLoss(logits[:, :-1], input_ids[:, 1:])

# 微调
# 添加任务特定的分类头
# classifier_head = nn.Linear(d_model, num_classes)
```

## 关键引用

```bibtex
@article{radford2018improving,
  title={Improving language understanding by generative pre-training},
  author={Radford, Alec and Narasimhan, Karthik and Salimans, Tim and Sutskever, Ilya},
  journal={OpenAI Technical Report},
  year={2018}
}
```

## 相关论文

- **Transformer (2017)**: GPT的基础架构
- **ELMo (2018)**: 另一个预训练方向
- **BERT (2018)**: 双向预训练
- **GPT-2 (2019)**: 规模扩展

## 个人思考

### GPT的启示

1. **预训练的价值**:
   - 无监督预训练蕴含丰富知识
   - 微调可以高效迁移
   - 两阶段范式成为标准

2. **生成式预训练**:
   - 语言模型作为通用目标
   - 生成能力可迁移到理解
   - 为后续GPT系列奠基

3. **规模的重要性**:
   - 1.17亿参数在当时已算大
   - 为后续规模扩展指明方向

### 历史地位

**开创性**:
- 首个成功的Transformer预训练模型
- 证明了生成式预训练的可行性
- 开启了LLM时代

**与BERT的关系**:
- 两者几乎同时期
- 不同方向但都成功
- 共同推动NLP革命

### 现代意义

**虽然被后续模型超越,但GPT**:
- 奠定了预训练范式
- 证明了生成式方法的价值
- 是理解GPT-2/3/4的基础

---

**阅读时间**: 建议2-3小时
**难度**: ⭐⭐⭐ (中等)
**重要性**: ⭐⭐⭐⭐⭐ (必读,预训练时代的开端)

**个人评价**: GPT是预训练语言模型的开山之作,虽然简单,但奠定了"预训练+微调"的范式,是理解现代LLM的必读论文。
