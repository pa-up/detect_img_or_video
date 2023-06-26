import numpy as np

def get_form_name_list(choice):
    """ formに表示する文字列を取得する関数 """
    choice_name_list = choice[: , 1]
    return choice_name_list

def form_name_to_variable_data(choice, form_name):
    """ formで得た文字列を変数のデータに変換する関数 """
    form_name_row = np.where(choice[:,1] == form_name)[0][0]
    variable_data = choice[form_name_row , 0]
    return variable_data


inside_or_background = np.array([
    ['inside', '内側'],
    ['background', '外側'],
])


convert_choice= np.array([
    ['normal', '無加工'],
    ['mosaic', 'モザイクをかける'],
    ['mask', '白黒画像にする'],
    ['light', '明るくする'],
    ['dark', '暗くする'],
    ['gray', 'グレースケール画像にする'],
])


person_cascade_path =  "./CascadeClassifier/person/haarcascade_fullbody.xml"
car_cascade_path =  "./CascadeClassifier/vehicle/cars.xml"
face_cascade_path =  "./CascadeClassifier/person/haarcascade_frontalface_default.xml"

all_img_label = np.array([
    ['all', '画像全体'],
])

# カスケード分類器ラベル
cascade_label = np.array([
    [face_cascade_path, '人の顔'],
])

# yolov5のCOCOラベル
yolov5_coco_label = np.array([
    ['person', '人間'],
    ['bicycle', '自転車'],
    ['car', '車'],
    ['motorcycle', 'バイク'],
    ['airplane', '飛行機'],
    ['bus', 'バス'],
    ['train', '電車'],
    ['truck', 'トラック'],
    ['boat', '船'],
    ['traffic light', '信号機'],
    ['fire hydrant', '消火栓'],
    ['stop sign', 'ストップサイン'],
    ['parking meter', 'パーキングメーター'],
    ['bench', 'ベンチ'],
    ['bird', '鳥'],
    ['cat', '猫'],
    ['dog', '犬'],
    ['horse', '馬'],
    ['sheep', '羊'],
    ['cow', '牛'],
    ['elephant', '象'],
    ['bear', 'クマ'],
    ['zebra', 'シマウマ'],
    ['giraffe', 'キリン'],
    ['backpack', 'リュック'],
    ['umbrella', '傘'],
    ['handbag', '手持ち鞄'],
    ['tie', 'ネクタイ'],
    ['suitcase', 'スーツケース'],
    ['frisbee', 'フリスビー'],
    ['skis', 'スキー'],
    ['snowboard', 'スノーボード'],
    ['sports ball', 'ボール'],
    ['kite', '凧'],
    ['baseball bat', '野球バット'],
    ['baseball glove', '野球のグローブ'],
    ['skateboard', 'スケートボード'],
    ['surfboard', 'サーフボード'],
    ['tennis racket', 'テニスラケット'],
    ['bottle', 'ボトル'],
    ['wine glass', 'ワイングラス'],
    ['cup', 'カップ'],
    ['fork', 'フォーク'],
    ['knife', 'ナイフ'],
    ['spoon', 'スプーン'],
    ['bowl', 'ボウル'],
    ['banana', 'バナナ'],
    ['apple', 'リンゴ'],
    ['sandwich', 'サンドイッチ'],
    ['orange', 'オレンジ'],
    ['broccoli', 'ブロッコリー'],
    ['carrot', 'にんじん'],
    ['hot dog', 'ホットドッグ'],
    ['pizza', 'ピザ'],
    ['donut', 'ドーナツ'],
    ['cake', 'ケーキ'],
    ['chair', '椅子'],
    ['sofa', 'ソファ'],
    ['potted plant', '観葉植物'],
    ['bed', 'ベッド'],
    ['dining table', 'ダイニングテーブル'],
    ['toilet', 'トイレ'],
    ['tv', 'テレビ'],
    ['laptop', 'ノートパソコン'],
    ['mouse', 'マウス'],
    ['remote', 'リモコン'],
    ['keyboard', 'キーボード'],
    ['cell phone', '携帯電話'],
    ['microwave', '電子レンジ'],
    ['oven', 'オーブン'],
    ['toaster', 'トースター'],
    ['sink', '流し台'],
    ['refrigerator', '冷蔵庫'],
    ['blender', 'ミキサー'],
    ['book', '本'],
    ['clock', '時計'],
    ['vase', '花瓶'],
    ['scissors', 'はさみ'],
    ['teddy bear', 'テディベア'],
    ['hair drier', 'ヘアドライヤー'],
    ['toothbrush', '歯ブラシ'],
])


# フォームのラベルの選択肢
label_choice = np.concatenate((all_img_label, yolov5_coco_label), axis=0)

