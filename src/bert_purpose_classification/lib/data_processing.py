import os

import mojimoji
import pandas as pd
import torch
from lib.Input import InputExample, InputFeatures
from torch.utils.data import TensorDataset


# データに対する一連の処理をする際に最初に使う関数
def load_and_cache_examples(
    data_dir=None,
    mode=None,
    max_seq_length=None,
    tokenizer=None,
    pad_token_label_id=None,
    label_names=None,
    text_column_name=None,
    pair_text_column_name=None,
    label_column_name=None,
):

    """
    引数の説明：
    data_dir: tsvやtxtファイルがあるフォルダのディレクトリ
    mode: data_dirで指定したファイル内の自分が読み込みたいファイルの名前を指定。
    max_seq_length: BERTモデルへ入力する時の最大入力長（東北大学のモデルの場合=512）
    tokenizer: 使用するtokenizer
    pad_token_label_id: PADトークンに対応したIDを指定（東北大学モデルの場合は0で良い）
    text_column_name: 上記で指定したcsvにおいて、テキストデータが記載されているカラム名
    label_column_name: 上記で指定したcsvにおいて、ラベルデータが記載されているカラム名
    """
    # ファイル内のデータをexamplesに格納
    examples = read_examples_from_file(
        data_dir, mode, text_column_name, pair_text_column_name, label_column_name
    )
    # examples内のテキストデータをBERTへの入力へ必要な各種id列に変換しfeatursに格納
    features = convert_examples_to_features(
        examples, label_names, max_seq_length, tokenizer
    )
    # 各種入力id列を一つのtensorにまとめる（型はすべてlong型に揃える）
    all_input_token_id = torch.tensor([f.token_ids for f in features], dtype=torch.long)
    all_input_mask_id = torch.tensor([f.mask_ids for f in features], dtype=torch.long)
    all_input_segment_id = torch.tensor(
        [f.segment_ids for f in features], dtype=torch.long
    )
    all_input_label = torch.tensor([f.label for f in features], dtype=torch.long)
    # 上記のデータを全て、TensorDatasetにまとめる
    dataset = TensorDataset(
        all_input_token_id, all_input_mask_id, all_input_segment_id, all_input_label
    )
    return dataset


# ファイルからデータを読み込みexampleに格納する関数
def read_examples_from_file(
    data_dir, mode, text_column_name, pair_text_column_name, label_column_name
):
    """
    2-3-2_c-1，pd.DataFrameでcsv等のファイルからデータを読み込む工程--------------
    """
    # 入力ファイルのpathを定義
    file_path = os.path.join(data_dir, "{}".format(mode))
    # ファイルの読み込み
    input_df = pd.read_csv(file_path)
    # テキスト部分
    texts_a = list(input_df[text_column_name])
    # ペアとなるテキストがある場合
    texts_b = [None] * len(texts_a)
    if pair_text_column_name is not None:
        texts_b = list(input_df[pair_text_column_name])
    # ラベル部分
    labels = list(input_df[label_column_name])
    """
    examplesを定義し、データを格納していく工程-------------------------
    """
    examples = []
    guid = 0
    for text_a, text_b, label in zip(texts_a, texts_b, labels):
        # examplesへの格納
        examples.append(
            InputExample(guid=guid, text_a=text_a, text_b=text_b, label=label)
        )
        guid += 1
    """
    ---------------------------------------------------------------------------
    """
    # 帰り値
    return examples


# ファイルから読み込まれたデータ（exanmple）をBERTへの入力形式に変換する関数
def convert_examples_to_features(examples, label_names, max_seq_length, tokenizer):
    features = []
    """
    前準備-------------------------------------------------------------
    """
    # 入力ラベルをkey、ラベルを示すid（int）をvalueとするdictを定義
    label2id_dict = {}
    for (i, label) in enumerate(label_names):
        label2id_dict[label] = i
    """
    exampleに格納されたデータを取り出してBERTへの入力形式に整えていく-----
    """
    for example in examples:
        # 入力テキストを全て全角にする
        text_a = mojimoji.han_to_zen(example.text_a)
        # 入力テキストをサブワードに分割
        tokens_a = tokenizer.tokenize(text_a)
        """
        2-3-3_c-3，入力文に対応したtoken_id列、mask_id列、segment_id列を作成する---
        """
        tokens = []
        mask_ids = []
        segment_ids = []
        # 最初に[CLS]トークンを付与する
        tokens.append("[CLS]")
        # 対応したmask_id, segment_idをそれぞれ付与
        mask_ids.append(1)
        segment_ids.append(0)
        # 各入力トークンをtokensに追加する
        # この時、それぞれに対応したmask_id, segment_idも付与する。
        for token in tokens_a:
            tokens.append(token)
            mask_ids.append(1)
            segment_ids.append(0)
        # テキストの最後（あるいは文の最後）には[SEP]トークンを置く
        tokens.append("[SEP]")
        mask_ids.append(1)
        segment_ids.append(0)
        """
        入力文のペアとなるテキストに対する処理---------------------------
        """
        if example.text_b is not None:
            # 入力テキストを全て全角にする
            text_b = mojimoji.han_to_zen(example.text_b)
            # 入力テキストをサブワードに分割
            tokens_b = tokenizer.tokenize(text_b)
            # ペアテキストにおけるトークンをtokensリスト等に格納する
            for token in tokens_b:
                # ペアテキストの語彙
                tokens.append(token)
                # マスクid
                mask_ids.append(1)
                # セグメントid(注：ペア文なので「１」)
                segment_ids.append(1)
            # 末尾に[SEP]を付与
            tokens.append("[SEP]")
            mask_ids.append(1)
            segment_ids.append(1)
        # 分割された各トークンをidに変換（例：[CLS]→1, [SEP]→2）
        token_ids = tokenizer.convert_tokens_to_ids(tokens)
        """
        各id列の入力長を[PAD]トークンで揃える----------------------------
        """
        while len(token_ids) < max_seq_length:
            token_ids.append(0)
            mask_ids.append(0)
            segment_ids.append(0)
        # 入力長が最大入力長を超えていた場合、超えている部分でぶつ切りをする
        token_ids = token_ids[:max_seq_length]
        mask_ids = mask_ids[:max_seq_length]
        segment_ids = segment_ids[:max_seq_length]
        # 各id列の長さが適切かどうかを確認
        assert len(token_ids) == max_seq_length
        assert len(mask_ids) == max_seq_length
        assert len(segment_ids) == max_seq_length
        """
        分類ラベルに対する処理----------------------------------------
        """
        # ラベルをidに変換
        label = label2id_dict[example.label]
        """
        作成された各id列をInputFeaturesに揃えた後、featuresに格納する--
        """
        features.append(
            InputFeatures(
                token_ids=token_ids,
                mask_ids=mask_ids,
                segment_ids=segment_ids,
                label=label,
            )
        )
    """
    ----------------------------------------------------------------------------
    """
    # 返り値
    return features
