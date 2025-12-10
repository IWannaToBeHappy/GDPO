eval用于评估方法性能

数据集设置：无标签数据集starcodedata
        https://github.com/xing-hu/TL-CodeSum JAVA有标签数据集  《Summarizing source code with transferred API knowledge》
        https://drive.google.com/drive/folders/1NMRfcC1VgxjGGfVPrlRUrNSx2SGdtWeW C有标签数据集 《Retrieval-augmented  generation for code summarization via hybrid GNN》
        
语言设置：C、python、java
指标设置：BLUE、METEOR、ROUGE-L
Baseline：同规模大模型（<3B）
        Llama-3.2-1B-instruct
        gemma-2-2b-it
        GLM-edge-1.5b-chat
        deepseek-coder-1.3b-instruct
        Hunyuan-1.8B-Instruct
        qwen3-1b
实验设置：
    各指标各数据集各模型
    可读性Reward
    几何平均/算数平均
    summary长度与code长度限制对比（要求summary长度小于code的几分之一）

