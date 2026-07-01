# EFSA

English | [简体中文](README_zh.md)

The PyTorch open source implementation for the ACL 2024 paper "EFSA: Towards Event-Level Financial Sentiment Analysis".

[ACL Anthology link](https://aclanthology.org/2024.acl-long.402/)
[arXiv link](https://arxiv.org/abs/2404.08681)
[PDF link](https://aclanthology.org/2024.acl-long.402.pdf)
[DOI link](https://doi.org/10.18653/v1/2024.acl-long.402)

Please reach us via emails or via GitHub issues for any enquiries.

## Citation

Please cite our work if you find it useful for your research and work.

```bibtex
@inproceedings{chen2024efsa,
  title={EFSA: Towards event-level financial sentiment analysis},
  author={Chen, Tianyu and Zhang, Yiming and Yu, Guoxin and Zhang, Dapeng and Zeng, Li and He, Qing and Ao, Xiang},
  booktitle={Proceedings of the 62nd Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers)},
  pages={7455--7467},
  year={2024}
}
```

## Abstract

In this paper, we extend financial sentiment analysis (FSA) to event-level since events usually serve as the subject of the sentiment in financial text. Though extracting events from the financial text may be conducive to accurate sentiment predictions, it has specialized challenges due to the lengthy and discontinuity of events in a financial text. To this end, we reconceptualize the event extraction as a classification task by designing a categorization comprising coarse-grained and fine-grained event categories. Under this setting, we formulate the Event-Level Financial Sentiment Analysis (EFSA for short) task that outputs quintuples consisting of (company, industry, coarse-grained event, fine-grained event, sentiment) from financial text. A large-scale Chinese dataset containing 12,160 news articles and 13,725 quintuples is publicized as a brand new testbed for our task. A four-hop Chain-of-Thought LLM-based approach is devised for this task. Systematic investigations are conducted on our dataset, and the empirical results demonstrate the benchmarking scores of existing methods and our proposed method can reach the current state-of-the-art.

## Dataset

All the data in our dataset can be found in `data/data.json`.

The correspondence between companies and industries can be found in `knowledge/company2industry.xlsx`.

## 4-hop CoT Framework

See `code/direct_prompt.py` and `code/reasoning_prompt.py` for details.

## Requirements

Install the required dependencies before running the code:

```bash
pip install torch transformers
```

The prompt scripts load a local or Hugging Face-compatible model through `transformers`. Replace `model-name` in the scripts with the actual model path or model identifier before running.

## Update

[2026-07-01] We updated the repository README with ACL Anthology metadata, citation information, and clearer dataset/code instructions.

[2024-08] EFSA was published in the Proceedings of the 62nd Annual Meeting of the Association for Computational Linguistics (ACL 2024).
