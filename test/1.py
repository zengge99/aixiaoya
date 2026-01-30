from transformers import pipeline

generator = pipeline("text-generation")
results = generator(
    "[CLS]我是",
    num_return_sequences=2,
    max_length=50
) 
print(results)