from transformers import pipeline

MyModel = pipeline("text-generation", model="meta-llama/Meta-Llama-3-8B")

print(MyModel("hello how are you doing?"))