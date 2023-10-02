import torch
import torch.nn as nn
import transformers


class SequenceClassification(transformers.BertPreTrainedModel):

    # init関数
    def __init__(self, config, num_clabels):
        super().__init__(config)
        self.bert = transformers.BertModel(config)
        self.num_clabels = num_clabels
        self.init_weights()
        # 平均プーリング層の定義
        self.adaptive_avg_layer = nn.AdaptiveAvgPool2d((1, 768))
        # ドロップアウト層の定義(transformersライブラリのデフォルト)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        print("drop_out率", config.hidden_dropout_prob)
        # 追加の全結合層（分類層）
        self.classifier = nn.Linear(config.hidden_size, self.num_clabels)

    # forward関数
    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        output_attentions=None,
        output_hidden_states=None,
        loss_weight=None,
        labels=None,
        output_vec="cls",
        return_dict=None,
    ):
        return_dict = (
            return_dict if return_dict is not None else self.config.use_return_dict
        )
        """
        引数で指定されたデータをBERTに入力する処理（返り値はoutoutsに格納）----------
        """
        outputs = self.bert(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        """
        BERTから出力されたベクトルに対する処理----------------------------------------
        """
        sequence_output = None
        # CLSベクトルのみを使う場合
        if output_vec == "cls":
            sequence_output = outputs[1]
        # 平均ベクトルを使う場合
        else:
            # outputs[0]にはBERTから出力された全てのベクトルが格納
            pooled_output = outputs[0]
            # 平均ベクトルを得るための関数に入力
            sequence_output = self.get_average_pooling(input_ids, pooled_output)
        # ドロップアウト層に入力
        sequence_output = self.dropout(sequence_output)
        # 分類層における予測ラベルの推論
        logits = self.classifier(sequence_output)
        """
        損失値の計算---------------------------------------------------------------
        """
        loss = None
        if labels is not None:
            # クラスウェイト有りの場合
            if loss_weight is not None:
                loss_weight = torch.tensor(loss_weight, dtype=torch.float32)
                loss_fct = nn.CrossEntropyLoss(weight=loss_weight)
                loss = loss_fct(logits.view(-1, self.num_clabels), labels.view(-1))
            # 無の場合
            else:
                loss_fct = nn.CrossEntropyLoss()
                loss = loss_fct(logits.view(-1, self.num_clabels), labels.view(-1))
        """
        ---------------------------------------------------------------------------------------
        """
        # 返り値はdict型でlossが損失値で，logitsが分類層（全結合層）からの出力である。
        return {"loss": loss, "logits": logits}

    # 平均ベクトルを計算するための関数
    def get_average_pooling(self, input_ids, pooled_output):
        average_vectors = []
        nums_ctos_token = []
        batch_size = 0
        # ミニバッチからデータを取り出し、CLSからSEPまでのトークンの数を数える
        for input_id in input_ids:
            input_id = input_id[torch.where(input_id != 0)]
            nums_ctos_token.append(len(input_id))
        # ミニバッチからデータを一つずつ取り出す
        for output_matrix, num_ctos_token in zip(pooled_output, nums_ctos_token):
            # CLS, SEP, PADトークンに対応したベクトルの削除
            purpose_matrix = output_matrix[1 : num_ctos_token - 1]
            purpose_matrix = purpose_matrix.view(-1, num_ctos_token - 2, 768)
            # 平均プーリング層へ入力
            average_vector = self.adaptive_avg_layer(purpose_matrix)
            average_vectors.append(average_vector)
            batch_size += 1
        # 各ベクトルを持つリストをtensorに変換
        average_vectors = torch.cat(average_vectors, dim=1)
        average_vectors = average_vectors.view(-1)
        # batchに整える。
        average_vectors = average_vectors.view(batch_size, 768)
        return average_vectors
