from transformers import AutoModelForSequenceClassification, AutoTokenizer

MODEL_NAME = "bansalsi/467ArxivClassification
TOKENIZER_NAME = "bert-base-uncased"

if __name__ == "__main__":
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME)
    tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_NAME)
    model.save_pretrained("ArxivClassificationModel/")
    tokenizer.save_pretrained("ArxivClassificationTokenizer/")
    print("Model and Tokenizer downloaded from the hugging face hub")