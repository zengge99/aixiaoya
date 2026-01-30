from transformers import pipeline

# 优化1：选中文专用NER模型，不要用默认英文模型
ner = pipeline("ner", model="uer/roberta-base-finetuned-ner-chinese", aggregation_strategy="max")

# 优化2：先对路径做简单清洗，去掉明显的网址/后缀，减少干扰
import os
def simple_clean(path):
    path = os.path.splitext(path)[0]  # 去后缀
    path = re.sub(r"https?://[^\s]+", "", path)  # 去网址
    return path

# 推理
messy_path = "https://pan.baidu.com/s/123_肖申克的救赎_1994.mp4"
cleaned_path = simple_clean(messy_path)
ner_result = ner(cleaned_path)
# 提取NER识别的“作品名”（中文NER模型一般会把影片名标为“TITLE”/“WORK_OF_ART”）
movie_name = "".join([r["word"] for r in ner_result if r["entity_group"] in ["TITLE", "WORK_OF_ART"]])
print(movie_name)