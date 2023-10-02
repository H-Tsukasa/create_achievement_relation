import os
import re
import sys
from pathlib import Path

import demoji
import mojimoji
import torch
from transformers import T5ForConditionalGeneration, T5Tokenizer

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
max_seq_length = 256

model_dir_path = Path("model")
tokenizer = T5Tokenizer.from_pretrained(model_dir_path)
trained_model = T5ForConditionalGeneration.from_pretrained(model_dir_path)
trained_model = trained_model.to(device)

load_file_dir = "./use_data"


def text_revise(text):
    text = text.replace("\n", "").replace("\r", "")  # 改行削除
    text = text.replace(" ", "")  # スペース削除
    text = text.replace("　", "")
    text = demoji.replace(string=text, repl="")  # 絵文字削除
    text = re.sub(
        r'[!"#$%&\'\\\\()*+,-./:;<=>?@[\\]^_`{|}~「」〔〕“”〈〉『』【】＆＊・（）＄＃＠。、？！｀＋￥％]',
        "",
        text,
    )
    text = re.sub(
        "[\uFF01-\uFF0F\uFF1A-\uFF20\uFF3B-\uFF40\uFF5B-\uFF65\u3000-\u303F]", "", text
    )
    text = re.sub(r"\b\d{1,3}(,\d{3})*\b", "0", text)  # 0に変換
    text = re.sub(r"\d+", "0", text)  # 0に変換
    text = text.lower()  # 大文字を小文字に変換
    text = mojimoji.han_to_zen(text)  # 半角から全角
    return text


# ファイル名取得
if len(sys.argv) < 2:
    test_file_name = "purposes.txt"
else:
    test_file_name = sys.argv[1]
    if not os.path.exists(f"{load_file_dir}/{test_file_name}"):
        print("ファイルが存在しません. use_dataフォルダにtxtまたはcsvファイルを入れてください.")
        exit()

# 前処理
purpose_texts = []
purposes = []
with open(f"{load_file_dir}/{test_file_name}", "r") as f:
    for line in enumerate(f.readlines()):
        line = line.replace("\n", "")
        split_line = line.split(",")
        purpose_texts.append(split_line[0])
        purposes.append(split_line[1])

purposes = [text_revise(purpose) for purpose in purposes]
purpose_texts = [text_revise(purpose_text) for purpose_text in purpose_texts]

# 検証
purposes_len = len(purposes)
with open("./test_results/result.txt", "w") as f:
    for i, line in enumerate(purpose_texts):
        if i % 20 == 0 and not i == 0:
            print(f"Progress:{i}/{purposes_len}")
        f.write(line)
        f.write(",")
        token = tokenizer(
            line, truncation=True, max_length=max_seq_length, padding="max_length"
        )
        output = trained_model.generate(
            input_ids=torch.tensor(token["input_ids"]).to("cuda").unsqueeze(0),
            attention_mask=torch.tensor(token["attention_mask"])
            .to("cuda")
            .unsqueeze(0),
        )
        output_decode = tokenizer.decode(output[0], skip_special_tokens=True)
        f.write(output_decode)
        f.write("\n")
