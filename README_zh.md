# EFSA

[English](README.md) | 简体中文

ACL 2024 论文《EFSA: Towards Event-Level Financial Sentiment Analysis》的 PyTorch 开源实现。

[ACL Anthology 链接](https://aclanthology.org/2024.acl-long.402/)
[PDF 链接](https://aclanthology.org/2024.acl-long.402.pdf)
[DOI 链接](https://doi.org/10.18653/v1/2024.acl-long.402)

如有任何问题，欢迎通过邮件或 GitHub issues 与我们联系。

## 引用

如果本工作对你的研究或工作有帮助，请引用我们的论文。

```bibtex
@inproceedings{chen-etal-2024-efsa,
    title = "{EFSA}: Towards Event-Level Financial Sentiment Analysis",
    author = "Chen, Tianyu  and
      Zhang, Yiming  and
      Yu, Guoxin  and
      Zhang, Dapeng  and
      Zeng, Li  and
      He, Qing  and
      Ao, Xiang",
    editor = "Ku, Lun-Wei  and
      Martins, Andre  and
      Srikumar, Vivek",
    booktitle = "Proceedings of the 62nd Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers)",
    month = aug,
    year = "2024",
    address = "Bangkok, Thailand",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2024.acl-long.402/",
    doi = "10.18653/v1/2024.acl-long.402",
    pages = "7455--7467"
}
```

## 摘要

本文将金融情感分析（Financial Sentiment Analysis, FSA）扩展到事件级别，因为事件通常是金融文本中情感表达的主体。尽管从金融文本中抽取事件有助于提升情感预测的准确性，但金融文本中的事件往往较长且存在不连续性，因此这一任务具有专门挑战。为此，我们将事件抽取重新建模为分类任务，并设计了由粗粒度事件类别和细粒度事件类别组成的事件分类体系。在这一设定下，我们提出事件级金融情感分析任务（Event-Level Financial Sentiment Analysis，简称 EFSA），该任务从金融文本中输出由（公司、行业、粗粒度事件、细粒度事件、情感）构成的五元组。我们公开了一个大规模中文数据集，包含 12,160 篇新闻文章和 13,725 个五元组，为该任务提供新的测试平台。本文还设计了一种基于大语言模型的四跳 Chain-of-Thought 方法。我们在数据集上进行了系统实验，结果展示了已有方法和本文方法的基准性能，其中本文方法达到了当前最优水平。

## 数据集

数据集文件位于 `data/data.json`。

公司与行业的对应关系位于 `knowledge/company2industry.xlsx`。

## 4-hop CoT 框架

代码细节见 `code/direct_prompt.py` 和 `code/reasoning_prompt.py`。

## 环境依赖

运行代码前请安装所需依赖：

```bash
pip install torch transformers
```

Prompt 脚本通过 `transformers` 加载本地模型或 Hugging Face 兼容模型。运行前请将脚本中的 `model-name` 替换为实际模型路径或模型标识符。

## 更新

[2026-07-01] 我们更新了仓库 README，补充 ACL Anthology 元数据、引用信息以及更清晰的数据集和代码说明。

[2024-08] EFSA 论文发表于第 62 届计算语言学协会年会（ACL 2024）。
