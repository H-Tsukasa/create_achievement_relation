import torch
from transformers import MLukeTokenizer, LukeModel
from sklearn.metrics.pairwise import cosine_similarity 
import numpy as np
import pickle
import MeCab
mecab = MeCab.Tagger('-d /usr/lib/x86_64-linux-gnu/mecab/dic/mecab-ipadic-neologd')


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

with open("./grouping/query.txt", "r") as f:
    actions = [line.replace("\n", "") for line in f.readlines()]

# ベクトル読み込み    
with open(f"./purposes_cos/purposes{number}_cos.pickle", "rb") as f:
    purposes_cos = pickle.load(f)


# 類似度比較
simirality_base = 0.7
simirality_same = 0.4
purposes_len = len(purposes)
ex_verb = ["する", "使う"]
for action in actions:
    parsed_lines = mecab.parse(action).split("\n")[:-2]
    print(parsed_lines)
    pos = [line.split('\t')[1].split(",")[0] for line in parsed_lines]
    base = [line.split('\t')[1].split(",")[6] for line in parsed_lines]
    with open(f"./grouping/{action}.txt", "w") as f:
        input_vector = sbert_model.encode(action, batch_size=8)
        i_vec = input_vector[0].to('cpu').detach().numpy().copy()
        for i in range(len(purposes)):     
            same_flag = False       
            if i % 10000 == 0:
                print("progress:", i, "/", purposes_len)
            parsed_lines = mecab.parse(purposes[i]).split("\n")[:-2]
            pos2 = [line.split('\t')[1].split(",")[0] for line in parsed_lines]
            base2 = [line.split('\t')[1].split(",")[6] for line in parsed_lines]
            for j, p in enumerate(pos2):
                if p == "名詞":
                    if base2[j] in base:
                        same_flag = True
                        break
                if p == "動詞":
                    if base2[j] in base and base2[j] not in ex_verb:
                        same_flag = True
                        break

            output = cosine_similarity(np.array([i_vec]), np.array([purposes_cos[i]]))
            if same_flag:
                simirality = simirality_same
            else:
                simirality = simirality_base
            if simirality <= output:
                f.write(purposes[i] + ",")
                f.write(str(i)+",")
                f.write(str(ids[i])+"\n")
