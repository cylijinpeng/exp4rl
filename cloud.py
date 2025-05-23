import requests
import json
import re

def select_topk(api_key, json_file_path, content_num, model="deepseek-chat"):
    """
    从JSON文件读取数据并发送给 DeepSeek API 进行分析

    Args:
        api_key (str): DeepSeek API Key
        json_file_path (str): JSON文件的路径
        content_num (int): 内容总数（ID 从 0 到 content_num-1）
        model (str): 使用的模型，默认是 deepseek-chat

    Returns:
        list: 长度为 content_num 的概率数组
    """
    # 新的 Prompt：只返回一个数组
    _prompt = '''# Task
You are an expert in the EXP4 multi-armed bandit setting. Given user access history, output a single JSON array of length N, where N is the total number of contents (arms). The element at index n must be the probability (float between 0 and 1) that arm/content with id = n is selected. The array must sum to exactly 1.0.

# Workflow
1. Analyze the provided access history.
2. Compute a probability for each content id from 1 to N.
3. Output exactly one JSON array of length N.

# Output Notes
- Only output the array, e.g. [0.05, 0.10, ..., 0.07]
- The sum of all elements must equal 1.0.
- Do NOT wrap the array in any additional object or markdown.
- Probabilities must be floats (you may use up to 4 decimal places).
    '''

    # 1. 读取 JSON 文件
    try:
        with open(json_file_path, 'r', encoding='utf-8') as f:
            json_data = json.load(f)
    except Exception as e:
        print(f"读取文件错误: {e}")
        return None

    # 2. 准备 API 请求
    url = "https://api.deepseek.com/v1/chat/completions"
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": _prompt},
            {"role": "user", "content": (
                f"Total contents: {content_num} (content_id from 1 to {content_num}).\n"
                "User access records (timestamp, content_id):\n" +
                json.dumps(json_data, ensure_ascii=False, indent=2)
            )}
        ],
    }

    # 3. 发送请求
    try:
        resp = requests.post(url, headers=headers, json=payload)
        resp.raise_for_status()
        text = resp.json()["choices"][0]["message"]["content"]
    except Exception as e:
        print(f"API 请求失败: {e}")
        return None

    # 4. 清洗并提取 JSON 数组
    # 去掉 markdown ``` 或额外空白
    clean = re.sub(r"```json|```", "", text).strip()
    # 匹配第一个 [...] 数组
    m = re.search(r"(\[.*\])", clean, flags=re.S)
    if not m:
        raise ValueError(f"未能匹配到 JSON 数组，模型返回：\n{text}")
    arr_str = m.group(1)

    # 5. 解析并返回
    try:
        probs = json.loads(arr_str)
    except json.JSONDecodeError as e:
        raise ValueError(f"解析 JSON 数组失败: {e}\nContent: {arr_str}")
    if not isinstance(probs, list) or len(probs) != content_num:
        raise ValueError(f"返回数组长度应为 {content_num}, 实际为 {len(probs)}")
    # 使其总和为1
    total = sum(probs)
    normalized = [x / total for x in probs]
    return normalized


# 示例使用
if __name__ == "__main__":
    DEEPSEEK_API_KEY = "sk-8d309cf3a82c44838eccbf8630f92982"
    JSON_FILE_PATH = "record.json"
    TOTAL_CONTENT = 15

    probabilities = select_topk(DEEPSEEK_API_KEY, JSON_FILE_PATH, TOTAL_CONTENT)
    if probabilities:
        print("Returned probability array:")
        print(probabilities)
        print(sum(probabilities))
