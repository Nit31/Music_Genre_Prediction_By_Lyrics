import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from clean_text import text_preprocessing_pipeline
import torch

nltk.download('punkt')
nltk.download('stopwords')
stop_words = set(stopwords.words('english'))
stemmer = PorterStemmer()

def vectorise(text, vectorizer):
    return vectorizer.encode_plus(
        text,
        truncation=True,
        add_special_tokens=True,
        max_length=200,
        padding='max_length',
        return_token_type_ids=True
    )


def predict(text, model, tokenizer):
    text = text_preprocessing_pipeline(text)
    text = vectorise(text, tokenizer)
    input_ids = torch.tensor(text['input_ids']).unsqueeze(0)
    attention_mask = torch.tensor(text['attention_mask']).unsqueeze(0)
    token_type_ids = torch.tensor(text['token_type_ids']).unsqueeze(0)
    output = model(input_ids, attention_mask, token_type_ids).detach().numpy()
    print(output)
    print(output.argmax().item())
    return output.argmax().item()

