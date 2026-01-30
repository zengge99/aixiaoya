from transformers import pipeline

generator = pipeline("text-generation")
results = generator("我是")
print(results)
results = generator(
    "我是",
    num_return_sequences=2,
    max_length=50
) 
print(results)