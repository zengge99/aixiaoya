from transformers import pipeline

# 构建QA pipeline，核心配置model_kwargs/tokenizer_kwargs
qa_pipeline = pipeline(
    task="question-answering",
    model="uer/roberta-base-chinese-extractive-qa",  # 推荐QA微调模型，解决权重缺失
    tokenizer="uer/roberta-base-chinese-extractive-qa",
    # 模型相关配置：离线+屏蔽讨论区
    model_kwargs={
        "local_files_only": True
    }
)

# 测试使用
result = qa_pipeline(
    question="这家公司是做什么的？",
    context="字节跳动公司从事人工智能领域的研发和应用，总部位于北京。"
)
print(result)