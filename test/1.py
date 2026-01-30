from transformers import pipeline

# 核心：指定中文专用的问答模型，解决原英文模型不支持中文的问题
# model选用hfl/chinese-roberta-wwm-ext-large-squad2，中文问答效果优、轻量易下载
question_answerer = pipeline(
    task="question-answering",
    model="hfl/chinese-roberta-wwm-ext-large-squad2",
    tokenizer="hfl/chinese-roberta-wwm-ext-large-squad2"
)

# 中文问答测试：问题+上下文（核心规则：答案必须在上下文中，问答任务是“从上下文提取答案”）
answer = question_answerer(
    question="我在哪里工作？",
    context="我叫小明，我在上海的字节跳动公司从事人工智能相关的工作"
)
print(answer)