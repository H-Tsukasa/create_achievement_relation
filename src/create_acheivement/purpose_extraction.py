import os
import random

import numpy as np
import pandas as pd
import torch
import transformers
from lib.data_processing import load_and_cache_examples
from lib.evaluation import model_load_checkpoint, test_prediction
from lib.SequenceClassification import SequenceClassification
from torch.utils.data import DataLoader

torch.manual_seed(42)
np.random.seed(42)
random.seed(42)

# 入力データに関する変数
text_column_name = "texts"
pair_text_column_name = None
label_names = ["目的", "非目的"]
label_column_name = "labels"
max_seq_length = 51

# データの読み込みに関する変数
load_file_dir = "../use_data"

# 学習に関する変数
pretrained_path = "cl-tohoku/bert-base-japanese-v3"
tokenizer = transformers.BertJapaneseTokenizer.from_pretrained(pretrained_path)
batch_size = 2

# checkpointに関する変数
save_checkpoint_dir = "../../bert_purpose_classification/checkpoints"
checkpoint_name = "bert_fine_tuning_checkpoint_for_binary_classification"
checkpoint_path = os.path.join(
    save_checkpoint_dir, "{}".format(checkpoint_name + ".pt")
)

# BERTモデルの定義(num_clabelsはn値分類の場合はn)
model = SequenceClassification.from_pretrained(pretrained_path, num_clabels=2)

# チェックポイントから学習したパラメータを読み込む
model = model_load_checkpoint(model, checkpoint_path)


for number in range(4, 10):
    for sep_number in range(0, 5):
        print(number)
        # ファイル名の取得
        load_test_file_name = f"review{number}_{sep_number}.csv"
        # load_test_file_name = f"review_test.csv"

        # テストデータ読み込み＆変換
        test_tds = load_and_cache_examples(
            data_dir=load_file_dir,
            mode=load_test_file_name,
            max_seq_length=max_seq_length,
            tokenizer=tokenizer,
            pad_token_label_id=0,
            label_names=label_names,
            text_column_name=text_column_name,
            pair_text_column_name=pair_text_column_name,
            label_column_name=label_column_name,
        )

        test_dataloader = DataLoader(test_tds, batch_size=batch_size, shuffle=False)

        # 検証の実行
        outputs = test_prediction(
            model=model,
            input_dataloader=test_dataloader,
            tokenizer=tokenizer,
            use_vector="cls",
            label_names=label_names,
        )

        output_df = pd.DataFrame(outputs)
        purpose_df = output_df.query('予測ラベル=="目的"')
        non_purpose_df = output_df.query('予測ラベル=="非目的"')
        purposes = list(purpose_df["入力文"])
        purposes = [p.replace("[CLS]", "").replace("[SEP]", "") for p in purposes]
        non_purposes = list(non_purpose_df["入力文"])
        non_purposes = [p.replace("[CLS]", "").replace("[SEP]", "") for p in non_purposes]

        purposes_id = list(zip(purposes, list(purpose_df.index)))
        non_purposes_id = list(zip(non_purposes, list(non_purpose_df.index)))

        print(purpose_df)

        with open(f"./purposes_by_bert/purpose{number}.txt", "a") as f:
            for p in purposes_id:
                f.write(f"{p[0]},{p[1]}")
                f.write("\n")

        with open(f"./purposes_by_bert/non_purpose{number}.txt", "a") as f:
            for p in non_purposes_id:
                f.write(f"{p[0]},{p[1]}")
                f.write("\n")

