from transformers import pipeline

# 核心：指定中文专用的问答模型，解决原英文模型不支持中文的问题
# model选用hfl/chinese-roberta-wwm-ext-large-squad2，中文问答效果优、轻量易下载
question_answerer = pipeline(
    task="question-answering",
    model="uer/roberta-base-chinese-extractive-qa",
    tokenizer="uer/roberta-base-chinese-extractive-qa"
)

# 中文问答测试：问题+上下文（核心规则：答案必须在上下文中，问答任务是“从上下文提取答案”）
answer = question_answerer(
    question="请提取路径中的影片名",
    context="/电影/原盘/合集/06 高码率视频(电影为主，少量纪录片剧集演唱会)MKV REMUX格式为主，多为单片20G以上，初整理，不重，多版本除外) 437.92T 15937个文件(若干外挂字幕)/00画质控蓝光原盘REMUX 53.45T 2694/3. 电影-蓝光原盘REMUX/蓝光原盘REMUX [国粤双语] [中英双字 国配中字 官译中字]/Twilight.Of.The.Warriors.Walled.In.2024.1080p.BluRay.REMUX.AVC.DTS-HD.MA.TrueHD.7.1.Atmos.mkv"
)
print(answer)