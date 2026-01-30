from transformers import pipeline

classifier = pipeline("zero-shot-classification")
result = classifier(
"九龙城寨之围城",
candidate_labels=["movie", "education", "politics", "business"],
)
print(result)