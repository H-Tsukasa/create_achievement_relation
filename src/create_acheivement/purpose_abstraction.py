import re

import demoji
import mojimoji
import torch
from transformers import T5ForConditionalGeneration, T5Tokenizer

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
max_seq_length = 256

# model_dir_path = Path("../t5_purpose_abstraction/model_prefix")
tokenizer = T5Tokenizer.from_pretrained("../../t5_purpose_abstraction/model_prefix")
trained_model = T5ForConditionalGeneration.from_pretrained("../../t5_purpose_abstraction/model_prefix")
trained_model = trained_model.to(device)

load_file_dir = "./purposes_by_bert"


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

for number in range(1, 3):

    test_file_name = f"purpose{number}_cleasing.txt"
    # if len(sys.argv) < 2:
    #     test_file_name = "purposes.txt"
    # else:
    #     test_file_name = sys.argv[1]
    #     if not os.path.exists(f"{load_file_dir}/{test_file_name}"):
    #         print("ファイルが存在しません. use_dataフォルダにtxtまたはcsvファイルを入れてください.")
    #         exit()

    # 前処理
    purpose_texts = []
    purpose_ids = []
    with open(f"{load_file_dir}/{test_file_name}", "r") as f:
        for line in f.readlines():
            line = line.replace("\n", "")
            split_line = line.split(",")
            purpose_texts.append(split_line[0])
            purpose_ids.append(split_line[1])

    purpose_texts = [text_revise(purpose_text) for purpose_text in purpose_texts]

    purposes_len = len(purpose_texts)

    with open(f"./purposes_by_t5/purpose{number}_abstracted.txt", "w") as f:
        for i, line in enumerate(purpose_texts):
            if i % 20 == 0 and not i == 0:
                print(f"Progress:{i}/{purposes_len}")
            input_data = "要約:「" + line + "」の目的は"
            token = tokenizer(
                input_data, truncation=True, max_length=max_seq_length, padding="max_length"
            )
            output = trained_model.generate(
                input_ids=torch.tensor(token["input_ids"]).to("cuda").unsqueeze(0),
                attention_mask=torch.tensor(token["attention_mask"])
                .to("cuda")
                .unsqueeze(0),
            )
            output_decode = tokenizer.decode(output[0], skip_special_tokens=True)
            search_text = re.search('(?<=「).+?(?=\」)', output_decode)
            if not search_text:
                f.write(line)
            else:
                f.write(search_text.group())
            f.write(f",{purpose_ids[i]}")
            f.write("\n")
