from transformers import pipeline
import os
import re

# 核心替换：使用可正常下载的中文NER模型，aggregation_strategy合并连续实体
ner = pipeline(
    task="ner",
    model="dslim/bert-base-chinese-ner",  # 可用的中文NER模型
    aggregation_strategy="max"  # 关键：把连续的影片名字符合并成一个实体，避免拆分
)

# 路径简单清洗：先去掉网址、后缀、特殊符号，减少模型干扰（提升识别精度）
def simple_clean(path):
    path = os.path.splitext(path)[0]  # 去掉文件后缀（.mp4/.mkv等）
    path = re.sub(r"https?://[^\s]+", "", path)  # 去掉网盘链接/网址
    path = re.sub(r"[【】\[\]_-]+", " ", path)  # 把特殊符号换成空格，让模型更容易识别实体
    return path.strip()

# 从NER结果中提取影片名（模型会把影片名标为WORK_OF_ART/作品名）
def extract_movie_name_ner(file_path):
    cleaned_path = simple_clean(file_path)
    if not cleaned_path:
        return ""
    # 模型推理识别实体
    ner_result = ner(cleaned_path)
    # 筛选出「作品名」实体（核心：WORK_OF_ART是影片名的标准标签）
    movie_parts = [r["word"] for r in ner_result if r["entity_group"] == "WORK_OF_ART"]
    # 拼接结果，去重
    movie_name = "".join(list(dict.fromkeys(movie_parts)))
    return movie_name if movie_name else "未识别到影片名"

# 测试：适配网上混乱的网盘路径
if __name__ == "__main__":
    test_paths = [
        "https://pan.baidu.com/s/123xxx_【高清4K】肖申克的救赎_1994_豆瓣9.7.mp4",
        "阿里云盘-星际穿越.Interstellar.2014.BD蓝光.mkv",
        "【B站】你的名字。.2016.日语中字.avi",
        "电影分享-霸王别姬_1993_1080P.x265.mp4-提取码：6666"
    ]
    # 批量识别
    for path in test_paths:
        name = extract_movie_name_ner(path)
        print(f"原路径：{path}\n提取影片名：{name}\n{'-'*50}")