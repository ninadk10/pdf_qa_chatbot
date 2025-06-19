from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline

class LocalLLM:
    def __init__(self, model_name="distilgpt2"):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(model_name)
        self.pipe = pipeline("text-generation", model=self.model, tokenizer=self.tokenizer)

    def answer(self, context, question):
        prompt = f"Context: {context}\nQuestion: {question}\nAnswer:"
        return self.pipe(prompt, max_new_tokens=100)[0]["generated_text"]
