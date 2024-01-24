import torch
from transformers import MLukeTokenizer, LukeModel
import pickle


# 日本語Sentence-BERTの定義クラス
class SentenceLukeJapanese:
    def __init__(self, model_name_or_path, device=None):
        self.tokenizer = MLukeTokenizer.from_pretrained(model_name_or_path)
        self.model = LukeModel.from_pretrained(model_name_or_path)
        self.model.eval()
        # deviceの準備
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = torch.device(device)
        self.model.to(device)

    def _mean_pooling(self, model_output, attention_mask):
        token_embeddings = model_output[0]
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)

    def encode(self, sentences, batch_size=8):
        all_embeddings = []
        iterator = range(0, len(sentences), batch_size)
        for batch_idx in iterator:
            batch = sentences[batch_idx:batch_idx + batch_size]

            encoded_input = self.tokenizer.batch_encode_plus(batch, padding="longest", truncation=True, return_tensors="pt").to(self.device)
            model_output = self.model(**encoded_input)
            sentence_embeddings = self._mean_pooling(model_output, encoded_input["attention_mask"]).to('cpu')

            all_embeddings.extend(sentence_embeddings)

        return torch.stack(all_embeddings)


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)
MODEL_NAME = "sonoisa/sentence-luke-japanese-base-lite"
sbert_model = SentenceLukeJapanese(MODEL_NAME)

number = 0

with open(f"./purposes_by_t5/purpose{number}_abstracted_cleasing.txt", "r") as f:
    lines = [line.replace("\n", "") for line in f.readlines()]
purposes = []
ids = []
for line in lines:
    purpose, id = line.split(",")
    purposes.append(purpose)
    ids.append(id)

# ベクトルに変換
purposes_cos = []
for i, purpose in enumerate(purposes):
    if i % 1000 == 0:
        print(i)
    output_vector = sbert_model.encode(purpose, batch_size=8)
    o_vec = output_vector[0].to('cpu').detach().numpy().copy()
    purposes_cos.append(o_vec)

with open(f"./purposes_cos/purposes{number}_cos.pickle", "wb") as f:
    pickle.dump(purposes_cos, f)
