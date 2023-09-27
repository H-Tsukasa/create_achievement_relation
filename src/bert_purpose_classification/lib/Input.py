# csv等から読み込んだデータを格納するためのクラス
class InputExample(object):
    """A single training/test example for token classification."""
    def __init__(self, guid, text_a, text_b, label):
        """Constructs a InputExample.
        Args:
            guid: Unique id for the example.
            words: list. The words of the sequence.
            labels: (Optional) list. The labels for each word of the sequence. This should be
            specified for train and dev examples, but not for test examples.
        """
        # 入力データのインデックス番号
        self.guid = guid
        # 入力テキスト
        self.text_a = text_a
        # 入力テキストに対するペア
        self.text_b = text_b
        # 分類ラベル
        self.label = label

# BERTへの入力形式（id列）に変換されたデータを格納するためのクラス
class InputFeatures(object):
    """A single set of features of data."""
    def __init__(self, token_ids, mask_ids, segment_ids, label):
        self.token_ids = token_ids
        self.mask_ids = mask_ids
        self.segment_ids = segment_ids
        self.label = label