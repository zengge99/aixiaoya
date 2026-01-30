from transformers import pipeline

classifier = pipeline("zero-shot-classification")
result = classifier(
"九龙城寨之围城 蓝光原盘[国粤双语] [中英双字 国配中字 官译中字]",
candidate_labels=["movie", "education", "politics", "business"],
)
print(result)