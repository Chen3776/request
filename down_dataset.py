import os
import argparse
# 不再需要 import yaml
from datasets import load_dataset

def get_value_from_yaml_text(file_path, key):
    """
    以文本方式逐行读取 YAML 文件，安全地提取指定键的值。
    避免因自定义标签（如 !function）导致的标准解析器错误。
    """
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                # 移除行首的空格并检查是否以 key 开头
                if line.strip().startswith(key + ":"):
                    # 分割键和值，并清理值中的空格和引号
                    value = line.split(":", 1)[1].strip()
                    # 移除可能存在的引号
                    if value.startswith("'") and value.endswith("'"):
                        value = value[1:-1]
                    if value.startswith('"') and value.endswith('"'):
                        value = value[1:-1]
                    return value
    except Exception:
        return None
    return None


def download_dataset_from_yaml(yaml_file, save_path):
    """
    从单个 YAML 文件中提取数据集信息并下载到指定本地路径。
    """
    print(f"--- Starting dataset download process ---")
    print(f"Processing YAML file: {yaml_file}")
    print(f"Dataset will be saved to (cache): {save_path}\n")

    # 确保保存路径存在
    os.makedirs(save_path, exist_ok=True)

    if not os.path.exists(yaml_file):
        print(f"ERROR: The file '{yaml_file}' was not found.")
        return

    try:
        # --- 修改部分：使用新的文本解析方式 ---
        dataset_path = get_value_from_yaml_text(yaml_file, "dataset_path")
        dataset_name = get_value_from_yaml_text(yaml_file, "dataset_name") # 可能为 None

        if not dataset_path:
            print(f"ERROR: 'dataset_path' not found in {yaml_file}")
            return

        print(f"Found dataset info: path='{dataset_path}', name='{dataset_name or 'Not specified'}'")

        # 使用 datasets 库下载数据集，并指定 cache_dir
        load_dataset(
            path=dataset_path,
            name=dataset_name,
            cache_dir=save_path,
        )

        print(f"SUCCESS: Successfully downloaded '{dataset_path}' to '{save_path}'\n")

    except Exception as e:
        print(f"ERROR: Failed to download dataset from {yaml_file}. Details: {e}\n")

    print("--- Task completed. ---")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Download a Hugging Face dataset based on a single YAML configuration file to a specific local directory."
    )
    parser.add_argument(
        "--yaml_file",
        type=str,
        required=True,
        help="The .yaml dataset configuration file."
    )
    parser.add_argument(
        "--save_path",
        type=str,
        required=True,
        help="The local path where the dataset will be downloaded and cached."
    )
    
    args = parser.parse_args()
    
    download_dataset_from_yaml(args.yaml_file, args.save_path)