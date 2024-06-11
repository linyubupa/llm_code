---
license: cc-by-nc-4.0
language:
- zh
- en
size_categories:
- 10K<n<100K
---
* 2023.12.20更新：增加来自[skypile](https://huggingface.co/datasets/Skywork/SkyPile-150B)数据集的长数据


# Long text dataset for pretraining

* This dataset contains samples with the length greater than 16k, which can be used for pretraining models with extremely long context lengths.

* The dataset is continuously updating.



# 长文本模型预训练数据集
* 此数据集包含长度大于16k的预训练数据，可用于对极长上下文长度的模型进行预训练。
* 数据正在持续增加中

## 中文数据
* 筛选自 [悟道200G开源数据](https://github.com/BAAI-WuDao/Data)、[书生万卷数据集](https://opendatalab.org.cn/OpenDataLab/WanJuan1_dot_0)、
[CCI中文互联网语料库](https://huggingface.co/datasets/BAAI/CCI-Data)
、中文维基百科等，
每条数据长度在16000字以上

## 英文数据
* 筛选自 [SlimPajama-dc]

(https://huggingface.co/datasets/MBZUAI-LLM/SlimPajama-627B-DC)， 每条数据长度在16000个word以上

## Sharegpt长对话
* 包含筛选自sharegpt的长度大于8k字的中文和英文对话。[Sharegpt长对话](https://huggingface.co/datasets/yuyijiong/Sharegpt-long-conversation)

## 图书、小说
中文小说由于数据量太大，已经上传至[云盘](https://cloud.tsinghua.edu.cn/d/0670fcb14d294c97b5cf/)。
英文图书来自RedPajamaBook，筛选了长度大于100k words的图书。

## 注意
注意，有些长文本数据中含有大量的重复字符串，推荐在使用以下代码进行去重：
```python
import re
#删除连续出现4次以上的非数字和字母的字符，只保留4次
content = re.sub(r'([^a-zA-Z0-9])\1{4,}', r'\1\1\1\1', content)
#删除连续出现3次以上的数字和字母组成的子串，例如“121212”变为“12”
content = re.sub(r'([a-zA-Z0-9]{3,}?)\1+', r'\1', content)
```

