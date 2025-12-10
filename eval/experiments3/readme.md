实验3针对不同训练手段进行比较

baseline1：A Prompt Learning Framework for Source Code Summarization
baseline2: No More Fine-Tuning? An Experimental Evaluation of Prompt Tuning in Code Intelligence

数据集：
    CodeXGLUE java

步骤：
    首先修改两篇论文的基模型为deepseek-coder-1_3b，以使得比较基准相同
    运行各自文件夹下的run.sh获得基线模型
    修改训练文件plugin.py中，指定目标语言为java
    运行data.py将java数据集转化为GAPO训练格式
    运行train_script/geometric/deepseek-code-1_5b-r2-java.sh训练GDPO模型
    运行model_transform_promptcs.py和model_pt4code.py，将保存模型转化为checkpoints
    因为两个基线只对摘要生成进行了训练，且包含prompt embedding等拆分手段，需要将summary model 和 code model分开。使用生成的模型对val.jsonl进行推理，获得摘要。
    使用deepseek-coder基模型根据摘要生成代码
    评估三个摘要数据集对于deepseek-coder基模型来说的spmi得分。

