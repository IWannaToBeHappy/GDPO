# 将C语言数据集转化为支持ms-swift训练的C函数数据集
from tree_sitter import Language,Parser# version 0.24.0
import tree_sitter_c # version 0.23.2
import tree_sitter_python # version 0.23.4
import tree_sitter_java # version 0.23.4
import pandas as pd
from tqdm import tqdm
import json

def parse_file(content: str,parser:Parser) -> list:
    """
    解析C语言文件和python语言文件,输出每个函数的内容，并去除注释和空行
    结果转换为标准预训练数据集格式：
        {"messages": [{"role": "assistant", "content": "function content"}]}
    """
    # 解析输入的C语言代码
    tree = parser.parse(bytes(content, "utf8"))

    # 提取函数节点
    functions = []
    for node in tree.root_node.children:
        if node.type == "function_definition":
            # 过滤类成员函数
            cur,add = node,True
            while cur.parent:
                if cur.parent.type in ["class_specifier","struct_specifier", "union_specifier"]:
                    add = False
                    break
                cur = cur.parent
            if add:
                functions.append(node)

    # 生成标准预训练数据集格式
    result = []
    for func in functions:
        # 过滤掉代码行数过低的函数
        if func.end_point[0] - func.start_point[0] < 10:
            continue
        func_content = func.text.decode("utf8").strip()
        result.append({"messages": [{"role": "user", "content": func_content}]})

    return result

def parse_java_file(content: str,parser:Parser) -> list:
    """
    解析C语言文件和python语言文件,输出每个类定义的内容，并去除注释和空行
    结果转换为标准预训练数据集格式：
        {"messages": [{"role": "assistant", "content": "function content"}]}
    """
    # 解析输入的Java代码
    tree = parser.parse(bytes(content, "utf8"))

    # 提取类节点
    classes = []
    for node in tree.root_node.children:
        if node.type == "class_declaration":
            classes.append(node)

    # 生成标准预训练数据集格式
    result = []
    for cls in classes:
        cls_content = cls.text.decode("utf8").strip()
        result.append({"messages": [{"role": "user", "content": cls_content}]})

    return result

def convert_dataset_c(dataset, output_file: str):
    """
    将C语言数据集转换为标准预训练数据集格式
    :param input_file: 输入数据集
    :param output_file: 输出的JSONL文件路径
    :param dataset_type: 数据集类型，默认为"file"
    """
    # starcoderdata 数据集使用parquet格式存储
    # 初始化Tree-sitter解析器
    c_lang = Language(tree_sitter_c.language())
    parser = Parser(c_lang)
    df = pd.read_parquet(dataset)
    with open(output_file, "a", encoding="utf8") as f:
        for index, row in tqdm(df.iterrows()):
            content = row['content']
            parsed_data = parse_file(content,parser)
            for item in parsed_data:
                json.dump(item, f, ensure_ascii=False)
                f.write("\n")

def convert_dataset_python(dataset, output_file: str):
    """
    将Python语言数据集转换为标准预训练数据集格式
    :param input_file: 输入数据集
    :param output_file: 输出的JSONL文件路径
    :param dataset_type: 数据集类型，默认为"file"
    """
    # starcoderdata 数据集使用parquet格式存储
    # 初始化Tree-sitter解析器
    python_lang = Language(tree_sitter_python.language())
    parser = Parser(python_lang)
    df = pd.read_parquet(dataset)
    with open(output_file, "a", encoding="utf8") as f:
        for index, row in tqdm(df.iterrows()):
            content = row['content']
            parsed_data = parse_file(content,parser)
            for item in parsed_data:
                json.dump(item, f, ensure_ascii=False)
                f.write("\n")

def convert_dataset_java(dataset, output_file: str):
    """
    将Java语言数据集转换为标准预训练数据集格式
    :param input_file: 输入数据集
    :param output_file: 输出的JSONL文件路径
    :param dataset_type: 数据集类型，默认为"file"
    """
    # starcoderdata 数据集使用parquet格式存储
    # 初始化Tree-sitter解析器
    java_lang = Language(tree_sitter_java.language())
    parser = Parser(java_lang)
    df = pd.read_parquet(dataset)
    with open(output_file, "a", encoding="utf8") as f:
        for index, row in tqdm(df.iterrows()):
            content = row['content']
            parsed_data = parse_java_file(content,parser)
            for item in parsed_data:
                json.dump(item, f, ensure_ascii=False)
                f.write("\n")

if __name__ == "__main__":
    input_file = "/workspace/dataset/starcoderdata/c.parquet"  # 替换为你的C语言文件路径
    output_file = "/workspace/dataset/train/starcoderdata_c_0.jsonl"  # 替换为输出的JSONL文件路径
    # convert_dataset_c(input_file, output_file)
    input_file = "/workspace/dataset/starcoderdata/python.parquet"  # 替换为你的Python语言文件路径
    output_file = "/workspace/dataset/train/starcoderdata_python_0.jsonl"  # 替换为输出的JSONL文件路径
    # convert_dataset_python(input_file, output_file)
    input_file = "/workspace/dataset/starcoderdata/java.parquet"  # 替换为你的Java语言文件路径
    output_file = "/workspace/dataset/train/starcoderdata_java_0.jsonl"  # 替换为输出的JSONL文件路径
    convert_dataset_java(input_file, output_file)

    print(f"Converted {input_file} to {output_file}")