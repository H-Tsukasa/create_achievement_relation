import torch

# PADトークンを削除するための関数
def remove_pad(vocabs):
    remove_pad_vocabs = []
    for vocab in vocabs:
        if vocab == '[PAD]':
            break
        else:
            remove_pad_vocabs.append(vocab)
    return remove_pad_vocabs

# モデルの性能を検証するための関数
def test_prediction(model=None,
                    input_dataloader=None,
                    tokenizer=None,
                    use_vector=None,
                    label_names=None,
                    device=None,
                    batch_size=2,
                    test_dataloader=None
                  ):
    '''
    前準備--------------------------------------------------------------
    '''
    # ラベルidをkey、元々のラベルをvalueとするdictを作成
    id2label_dict = {}
    for (i, label) in enumerate(label_names):
        id2label_dict[i] = label
    # モデルをGPUへ
    model.to(device)
    torch.backends.cudnn.benchmark = True
    # モデルを検証モードへ
    model.eval()
    # BERTに入力した元々の文章を格納するリスト
    input_sentences = []
    # 正解のラベルidを格納するリスト
    true_label_ids = []
    # 予測されたラベルidを格納するリスト
    pred_label_ids = []
    # 正解したデータの数を数える
    epoch_corrects = 0
    # 経過したミニバッチを記録
    batch_processed_num = 0
    '''
    モデル性能の検証開始-----------------------------------------------
    '''
    # データローダーからミニバッチを取り出す
    for batch in input_dataloader:
            '''
            各入力idを定義する-----------------------------------------
            '''
            # トークンid
            input_token_ids = batch[0].to(device)
            # マスクid
            input_mask_ids = batch[1].to(device)
            # セグメントid
            input_segment_ids = batch[2].to(device)
            # ラベルid
            input_labels = batch[3].to(device)
            '''
            token_id列を基のサブワード列に戻す処理----------------------
            '''
            # 入力テキストをid列からトークン列へ
            input_tokens_in_batch =[tokenizer.convert_ids_to_tokens(token_ids) \
                                    for token_ids in batch[0]]
            # [PAD]トークン削除
            input_tokens_in_batch = [remove_pad(tokens) \
                                     for tokens in input_tokens_in_batch]
            # トークン列をセンテンスに変換
            input_sentences_in_batch = [''.join(tokens) \
                                        for tokens in input_tokens_in_batch]
            #「##」の除去
            input_sentences_in_batch = [sentence.replace('##', '') \
                                        for sentence in input_sentences_in_batch]
            '''
            正解ラベルを保持する処理------------------------------------
            '''
            # 入力センテンスをリストに格納
            for sentence in input_sentences_in_batch:
                input_sentences.append(sentence)

            # 以下は正解ラベルをリストに格納
            for true_label in input_labels:
                true_label_ids.append(true_label.item())
            '''
            各id列をBERTに入力、ラベル推論-------------------------------
            '''
            # BERTモデルへの入力
            results = model(input_ids=input_token_ids,
                            attention_mask=input_mask_ids,
                            token_type_ids=input_segment_ids,
                            output_vec=use_vector)
            # 予測ラベルの取得
            _, preds = torch.max(results['logits'], 1)
            preds = preds.view(-1)
            # 予測ラベルをリストに格納
            for pred_label in preds:
                pred_label_ids.append(pred_label.item())
            # それぞれの予測ラベルにおいて正解した数を蓄積
            epoch_corrects += torch.sum(preds==input_labels.data)
            batch_processed_num += 1
            if batch_processed_num % 10 == 0 and batch_processed_num != 0:
                print('Processed : ', batch_processed_num * batch_size)
    # 正解率の計算
    # epoch_acc = epoch_corrects.double() / (len(test_dataloader.dataset))
    # print('正解率', epoch_acc.item())
    '''
    帰値をまとめる-----------------------------------------------------
    '''
    # 正解ラベルのidを基のラベルに変換
    true_labels = [id2label_dict[label_id] for label_id in true_label_ids]
    pred_labels = [id2label_dict[label_id] for label_id in pred_label_ids]
    # 各推論結果をdict型にまとめて返す
    return{'入力文': input_sentences,
           '正解ラベル': true_labels,
           '予測ラベル': pred_labels,
           'true_label_id': true_label_ids,
           'pred_label_id': pred_label_ids}
    
# モデルにチェックポイントをロードする関数
def model_load_checkpoint(model, load_path):
    """
        Args:
            model: 使用するBERTモデル（元となる事前学習モデル）
            load_path: 読み込むチェックポイントのpath
    """
    # checkpointの読み込み
    checkpoint = torch.load(load_path)
    # モデルのパラメータ部分の読み込み
    model.load_state_dict(checkpoint['model_state_dict'])
    return model
  
