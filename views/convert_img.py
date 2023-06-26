import cv2
import moviepy.editor as mp
import numpy as np
from PIL import Image
import random
import time
from typing import Callable, List, Optional, TypeVar, Any
ReturnType = TypeVar('ReturnType')
import streamlit as st
import views.useful_st as us


def no_change(cv_img, func2):
    """ 無修正 """
    return cv_img



def gray(cv_img, func2):
    """ グレースケール化 """
    cv_calc_img = cv2.cvtColor(cv_img, cv2.COLOR_RGB2BGR)
    cv_calc_img = cv2.cvtColor(cv_img, cv2.COLOR_BGR2GRAY)
    cv_calc_img = cv2.cvtColor(cv_calc_img, cv2.COLOR_GRAY2BGR)
    return cv_calc_img



def binar2pil(binary_img):
    """ バイナリ画像PIL画像に変換 """
    pil_img = Image.open(binary_img)
    return pil_img



def binar2opencv(binary_img):
    """ バイナリ画像をOpenCV画像に変換 """
    pil_img = Image.open(binary_img)
    cv_img = pil2opencv(pil_img)
    return cv_img



def pil2opencv(pil_img):
    """ PIL画像をOpenCV画像に変換 """
    cv_img = np.array(pil_img, dtype=np.uint8)

    if cv_img.ndim == 2:  # モノクロ
        pass
    elif cv_img.shape[2] == 3:  # カラー
        cv_img = cv2.cvtColor(cv_img, cv2.COLOR_RGB2BGR)
    elif cv_img.shape[2] == 4:  # 透過
        cv_img = cv2.cvtColor(cv_img, cv2.COLOR_RGBA2BGRA)
    return cv_img



def opencv2pil(cv_calc_img):
    """ OpenCV画像をPIL画像に変換 """
    pil_img = cv_calc_img.copy()
    
    if pil_img.ndim == 2:  # モノクロ
        pass
    elif pil_img.shape[2] == 3:  # カラー
        pil_img = cv2.cvtColor(pil_img, cv2.COLOR_BGR2RGB)
    elif pil_img.shape[2] == 4:  # 透過
        pil_img = cv2.cvtColor(pil_img, cv2.COLOR_BGRA2RGBA)
    pil_img = Image.fromarray(pil_img)
    return pil_img



def max_size(cv_img , max_img_size):
    """ 
    縦横の倍率を保ちながら、画像の辺の長さの最大値を定義
    例）max_img_size = 1500 : 画像の縦または横サイズの最大値を1500に制限
    """
    rows = cv_img.shape[0]
    cols = cv_img.shape[1]
    new_row = rows
    new_col = cols
    if (rows >= cols)  and (rows > max_img_size) :
        new_row = max_img_size
        new_col = int( cols / (rows/max_img_size) )
    #
    if (cols > rows)  and (cols > max_img_size) :
        new_col = max_img_size
        new_row = int( rows / (cols/max_img_size) )
    #
    cv_img = cv2.resize( cv_img , dsize=(new_col, new_row) )
    return cv_img


def generate_input_img_path():
    # 現在時刻をシード値として使用
    random.seed(time.time())
    digits = [str(random.randint(0, 9)) for _ in range(7)]
    input_img_path = "".join(digits) + ".jpg"
    return input_img_path



def brightness(input_image , gamma):
  """ 
  画像の明るさ（輝度）を変える関数
  gamma > 1  =>  明るくなる
  gamma < 1  =>  暗くなる 
  """
  img2gamma = np.zeros((256,1),dtype=np.uint8)  # ガンマ変換初期値

  for i in range(256):
    # ガンマ補正の公式 : Y = 255(X/255)**(1/γ)
    img2gamma[i][0] = 255 * (float(i)/255) ** (1.0 /gamma)
  
  # 読込画像をガンマ変換
  gamma_img = cv2.LUT(input_image , img2gamma)
  return gamma_img


def mosaic(input_image , parameter):
  """ 
  モザイク処理（画像を縮小してモザイク効果を適用）
  parameter : リサイズにおける 縦＝横 サイズ（小さいほどモザイクが強くなる）
  例）一般的には parameter = 25 ~ 50 など
  """
  mosaic_img = cv2.resize(input_image, (parameter , parameter), interpolation=cv2.INTER_NEAREST)
  mosaic_img = cv2.resize(mosaic_img, input_image.shape[:2][::-1], interpolation=cv2.INTER_NEAREST)
  return mosaic_img



def mask(input_image , threshold):
  """ 
   2値化（マスク）処理 
   threshold : しきい値（ 0 ~ 255 の 整数値）
  """
  # グレースケール化
  input_image = cv2.cvtColor(input_image, cv2.COLOR_RGB2BGR)
  img_gray = cv2.cvtColor(input_image, cv2.COLOR_BGR2GRAY)
  # 2値化
  ret, mask_img = cv2.threshold(img_gray, threshold, 255, cv2.THRESH_BINARY)
  # 2値画像を3チャンネルに拡張する
  mask_img_3ch = cv2.cvtColor(mask_img, cv2.COLOR_GRAY2BGR)
  mask_img_3ch = cv2.cvtColor(mask_img_3ch, cv2.COLOR_BGR2RGB)
  return mask_img_3ch


def form_name_to_convert_func(img_conversion):
    """ 入力した文字列に応じて、任意の画像処理関数とパラメータを返す関数 """
    """ WEB上でのユーザーからのform入力に対して、コマンドを返すことにも利用可 """
    convert_img_func = no_change
    convert_object_parameter = ""
    if img_conversion == "mosaic":
        convert_img_func = mosaic
        convert_object_parameter = 25
    if img_conversion == "mask":
        convert_img_func = mask
        convert_object_parameter = 120
    if img_conversion == "light":
        convert_img_func = brightness
        convert_object_parameter = 2
    if img_conversion == "dark":
        convert_img_func = brightness
        convert_object_parameter = 0.5
    if img_conversion == "gray":
        convert_img_func = gray
        convert_object_parameter = ""
    
    return convert_img_func , convert_object_parameter


def resize_to_square(input_img , resized_length):
  """ 
  入力画像を正方形に収まるようにリサイズし、余白を黒で塗りつぶす関数
  （リサイズ後の画像を左 or 上に敷き詰め、画像を縦横の短い方向を黒で塗りつぶす）
  """
  input_height = input_img.shape[0]
  input_width = input_img.shape[1]
  
  # 入力画像が正方形の場合
  if input_width == input_height:
    resized_height , resized_width = resized_length , resized_length
    resized_input_img = cv2.resize( input_img, (resized_height , resized_width) )
    resized_square_img = resized_input_img

  # 入力画像が縦長の場合
  if  input_width < input_height:
    resized_height , resized_width = resized_length , int( input_width * resized_length / input_height )
    resized_input_img = cv2.resize( input_img, (resized_width , resized_height) )
    # 画像を正方形の左に敷き詰め、右の余白を黒で埋め尽くす
    resized_square_img = np.zeros( (resized_length , resized_length , 3) )
    resized_square_img[ : ,  : resized_width] = resized_input_img
  
  # 入力画像が横長の場合
  if  input_width > input_height:
    resized_height , resized_width = int( input_height * resized_length / input_width ) , resized_length
    resized_input_img = cv2.resize( input_img, (resized_width , resized_height) )
    # 画像を正方形の上に敷き詰め、下の余白を黒で埋め尽くす
    resized_square_img = np.zeros( (resized_length , resized_length , 3) )
    resized_square_img[ : resized_height ,  : ] = resized_input_img

  resized_square_img = resized_square_img.astype(np.float32)
  return resized_square_img , resized_input_img



def convert_choiced_area(input_img , argu2):
    """
    特定の検出領域に任意の画像処理を実行する関数
    argu2 = [ label_information , detect_results ]
        label_information = np.array([choiced_object_label , inside_or_background , convert_object_func , convert_object_parameter])
            inside_or_background : 文字列（"inside" or "background"）
            convert_object_func : 画像処理を実行する関数
            convert_object_parameter : 画像処理関数のパラメータ（ = 第二引数）（第一引数が入力画像numpy）
        detect_results = np.array([ x_min_list , y_min_list , x_max_list , y_max_list , classes, confidences ])
            x_min_list ~ y_max_list : List[int, int, ...]
            classes : N行0列（ラベル名の文字列）
            confidences : N行0列（ラベルの確信度）
    """
    output_img = input_img.copy()
    label_information , detect_results = argu2[0] , argu2[1]

    choiced_object_label = label_information[0]
    inside_or_background = label_information[1]
    convert_object_func = label_information[2]
    convert_object_parameter = label_information[3]

    # 検出領域毎に場合わけ
    if detect_results.shape[1] != 0:
        background_initial_all_convert = True    #「inside_or_background == "background"」の場合、最初のループ時に全体を画像処理する
        for area_number in range( detect_results.shape[1] ):
            detected_label = detect_results[4][area_number]

            if detected_label == choiced_object_label:
                x_min, y_min = detect_results[0][area_number] , detect_results[1][area_number]
                x_max, y_max = detect_results[2][area_number] , detect_results[3][area_number]
                input_detection_area = input_img[y_min : y_max , x_min : x_max]
                
                # 検出領域の切り抜き画像が空であるか否か
                img_empty_valid = input_detection_area.shape[0] == 0 or input_detection_area.shape[1] == 0

                if inside_or_background == "inside" and img_empty_valid == False :
                    # 検出領域に画像加工を実施
                    converted_area = convert_object_func(input_detection_area , convert_object_parameter)
                    output_img[y_min : y_max , x_min : x_max] = converted_area
            
                if inside_or_background == "background" and img_empty_valid == False :
                    # まず全体に画像加工を行った後に、各検出領域に加工前の入力画像を適用
                    if background_initial_all_convert:
                        output_img = convert_object_func(input_img , convert_object_parameter)
                        background_initial_all_convert = False
                    output_img[y_min : y_max , x_min : x_max] = input_detection_area
    return output_img


class Yolov5OnnxDetector:
    def __init__(self, onnx_model_path: str , resize_length: int = 640):
        # ONNXモデルを読み込む
        self.onnx_model_path = onnx_model_path
        self.cv_dnn_torch = cv2.dnn.readNetFromONNX(onnx_model_path)
        self.resize_length = resize_length
        self.yolov5_coco_label = np.array([
            ['person'],['bicycle'],['car'],['motorcycle'],['airplane'],['bus'],
            ['train'],['truck'],['boat'],['traffic light'],['fire hydrant'], 
            ['stop sign'], ['parking meter'],['bench'],['bird'],['cat'],['dog'],
            ['horse'],['sheep'],['cow'],['elephant'],['bear'],['zebra'],['giraffe'],            
            ['backpack'],['umbrella'],['handbag'],['tie'],['suitcase'],['frisbee'],
            ['skis'],['snowboard'],['sports ball'],['kite'],['baseball bat'], 
            ['baseball glove'],['skateboard'],['surfboard'],['tennis racket'],
            ['bottle'],['wine glass'],['cup'],['fork'],['knife'],['spoon'],['bowl'],
            ['banana'],['apple'],['sandwich'],['orange'],['broccoli'],['carrot'],
            ['hot dog'],['pizza'],['donut'],['cake'],['chair'],['sofa'],['potted plant'],
            ['bed'],['dining table'],['toilet'],['tv'],['laptop'],['mouse'],['remote'],
            ['keyboard'],['cell phone'],['microwave'],['oven'],['toaster'], ['sink'],
            ['refrigerator'],['blender'],['book'],['clock'],['vase'],['scissors'],
            ['teddy bear'],['hair drier'],['toothbrush'],
        ])

    def resize_to_square(self, input_img , resized_length):
        """ 
        入力画像を正方形に収まるようにリサイズし、余白を黒で塗りつぶす関数
        （リサイズ後の画像を左 or 上に敷き詰め、画像を縦横の短い方向を黒で塗りつぶす）
        """
        input_height = input_img.shape[0]
        input_width = input_img.shape[1]
        
        # 入力画像が正方形の場合
        if input_width == input_height:
            resized_height , resized_width = resized_length , resized_length
            resized_input_img = cv2.resize( input_img, (resized_height , resized_width) )
            resized_square_img = resized_input_img

        # 入力画像が縦長の場合
        if  input_width < input_height:
            resized_height , resized_width = resized_length , int( input_width * resized_length / input_height )
            resized_input_img = cv2.resize( input_img, (resized_width , resized_height) )
            # 画像を正方形の左に敷き詰め、右の余白を黒で埋め尽くす
            resized_square_img = np.zeros( (resized_length , resized_length , 3) )
            resized_square_img[ : ,  : resized_width] = resized_input_img
        
        # 入力画像が横長の場合
        if  input_width > input_height:
            resized_height , resized_width = int( input_height * resized_length / input_width ) , resized_length
            resized_input_img = cv2.resize( input_img, (resized_width , resized_height) )
            # 画像を正方形の上に敷き詰め、下の余白を黒で埋め尽くす
            resized_square_img = np.zeros( (resized_length , resized_length , 3) )
            resized_square_img[ : resized_height ,  : ] = resized_input_img

        resized_square_img = resized_square_img.astype(np.float32)
        return resized_square_img , resized_input_img

    def detect_by_yolov5s_onnx(self, input_img: np.ndarray):
        # 入力画像の前処理
        model_input_shape = (3, self.resize_length, self.resize_length)  # モデルの入力サイズに合わせる
        resized_square_img, resized_input_img = self.resize_to_square(input_img, model_input_shape[1])
        preprocessing = resized_square_img.copy()
        # モデルに入力し、出力を取得する
        input_cv2_dnn = cv2.dnn.blobFromImage(preprocessing, 1/255.0, (model_input_shape[1], model_input_shape[2]), swapRB=True, crop=False)
        self.cv_dnn_torch.setInput(input_cv2_dnn)
        output_cv_torch = self.cv_dnn_torch.forward()
        print(f"resized_input_img : {resized_input_img.shape}")
        return output_cv_torch, resized_input_img

    def analyse_detection_results(self, output_cv_torch, confidence_threshold: int = 0.98):
        """
        推論結果から検出領域の座標、クラス、確信度を取得する後処理の関数
        returns:
            detect_results = np.array([ x_min_list , y_min_list , x_max_list , y_max_list , classes, confidences ])
                x_min_list ~ y_max_list : List[int, int, ...]
                classes : N行0列（ラベル名の文字列）
                confidences : N行0列（ラベルの確信度）
        """
        boxes, classes, confidences = [], [], []
        for output in output_cv_torch:
            for detection in output:
                class_id = np.argmax(detection[5:])
                confidence = detection[5 + class_id]
                box = detection[:4]
                x1, y1, x2, y2 = box.astype("int")
                if confidence >= confidence_threshold and x2 > 0 and y2 > 0:
                    classes.append(class_id)
                    confidences.append(confidence)
                    x_min, y_min, x_max, y_max = int(x1 - x2 / 2), int(y1 - y2 / 2), int(x1 + x2 / 2), int(y1 + y2 / 2)
                    boxes.append([x_min, y_min, x_max, y_max])
        boxes, labels, confidences = np.array(boxes), np.array(classes), np.array(confidences)
        boxes = boxes.reshape((-1, 4))
        labels = np.empty(len(classes), dtype='object')
        for n in range(len(classes)):
            labels[n] = self.yolov5_coco_label[classes[n]]
        detect_results = np.array([
            boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3],
            labels, confidences,
        ])
        return detect_results

    def convert_detected_area(self, input_img: np.ndarray, label_information: np.ndarray, onnx_model_path:str = "" , confidence_threshold: int = 0.98):
        """
        特定の検出領域に任意の画像処理を実行する関数
        returns:
            input_img (np.ndarray) : RGBの入力画像
            label_information = np.array([choiced_object_label , inside_or_background , convert_object_func , convert_object_parameter])
                inside_or_background : 文字列（"inside" or "background"）
                convert_object_func : 画像処理を実行する関数
                convert_object_parameter : 画像処理関数のパラメータ（ = 第二引数）（第一引数が入力画像numpy）
            confidence_threshold (int) : 物体検出の確信度の閾値
        """
        # 物体検出の実行
        output_cv_torch, input_img = self.detect_by_yolov5s_onnx(input_img)
        print(f"input_img : {input_img.shape}")
        # 検出結果から、検出領域の座標・ラベル名・確信度のリストを取得
        detect_results = self.analyse_detection_results(output_cv_torch, confidence_threshold)

        output_img = input_img.copy()
        choiced_object_label = label_information[0]
        inside_or_background = label_information[1]
        convert_object_func = label_information[2]
        convert_object_parameter = label_information[3]

        if detect_results.shape[1] != 0:
            background_initial_all_convert = True
            for area_number in range(detect_results.shape[1]):
                detected_label = detect_results[4][area_number]
                if detected_label == choiced_object_label:
                    x_min, y_min = detect_results[0][area_number], detect_results[1][area_number]
                    x_max, y_max = detect_results[2][area_number], detect_results[3][area_number]
                    input_detection_area = input_img[y_min: y_max, x_min: x_max]
                    img_empty_valid = input_detection_area.shape[0] == 0 or input_detection_area.shape[1] == 0

                    if inside_or_background == "inside" and not img_empty_valid:
                        converted_area = convert_object_func(input_detection_area, convert_object_parameter)
                        output_img[y_min: y_max, x_min: x_max] = converted_area

                    if inside_or_background == "background" and not img_empty_valid:
                        if background_initial_all_convert:
                            output_img = convert_object_func(input_img, convert_object_parameter)
                            background_initial_all_convert = False
                        output_img[y_min: y_max, x_min: x_max] = input_detection_area
        return output_img




def video_information(input_video_file , output_video_file:str = ""):
  """
  総再生時間、総フレーム数の表示 : frame_number , total_time
  動画の書き込み形式の取得 : fmt , writer
  """

  # 動画をフレームに分割
  cap = cv2.VideoCapture(input_video_file)

  #動画サイズ取得
  width = int( cap.get(cv2.CAP_PROP_FRAME_WIDTH) )
  height = int( cap.get(cv2.CAP_PROP_FRAME_HEIGHT) )

  #フレームレート取得
  fps = int(cap.get(cv2.CAP_PROP_FPS))

  #フォーマット指定（動画の書き込み形式）
  fmt = cv2.VideoWriter_fourcc('m', 'p', '4', 'v')
  writer = cv2.VideoWriter( output_video_file , fmt, fps, (width, height) )

  # 表示
  st.write("合計フレーム数：")
  frame_number = int( cap.get(cv2.CAP_PROP_FRAME_COUNT) )
  st.write(f"{frame_number} 枚 \n")

  st.write("合計再生時間（総フレーム数 / FPS）：")
  total_time = cap.get(cv2.CAP_PROP_FRAME_COUNT) / cap.get(cv2.CAP_PROP_FPS)
  total_time = round(total_time , 3)
  st.write(f"{total_time} 秒  \n \n")

  return cap , writer , fmt , fps , width , height , frame_number , total_time



def gifs_to_mp4(output_gif_file: str, output_video_file: str, fps: int):
    # gif動画ファイルの読み込み
    gif_movie = mp.VideoFileClip(output_gif_file)
    #mp4動画ファイルの保存
    gif_movie.write_videofile(output_video_file, fps, )
    gif_movie.close()


def st_video_convert(
    input_video_file: str ,
    output_gif_file: str ,
    conversion_func: Callable[..., Optional[ReturnType]] ,
    arguments_list: list ,
    fps_divide: int = 4 ,
):
    """
    streamlitで動画に画像加工を施し、gifファイルとして保存する関数
        streamlitでは、おそらく OpenCVの「writer.writer」が利用不可
    parameters :
        video_input_file : 入力動画のパス
        output_gif_file : gif動画の出力先のパス
        conversion_func : 画像加工の関数（第一引数は必ず入力画像にする必要がある）
        arguments_list : conversion_func()に代入する引数のリスト
            例1）conversion_func()の引数が4つ → arguments_list = [argument1,argument2,argumen3,argumen4]
            例2）conversion_func()の引数が1つ → arguments_list = [argument]
        fps_divide : フレーム数の縮小比 = 処理速度の倍率
    """

    def function_of_any_arguments(function , arguments_list):
        return function(*arguments_list)
    
    def pils_to_gifs(frames_pil_list: list, output_gif_file: str, fps: int):
        # GIF動画を作成して保存
        frames_pil_list[0].save(output_gif_file, save_all=True, append_images=frames_pil_list[1:], optimize=False, duration=1000//fps, loop=0)    
  
    get_movie_info = us.AddOrDeleteStMessage(message = "アップロードされた動画の情報を取得中")
    # 入力動画をフレームに分割、動画出力の形式・総再生時間などを取得・表示
    cap , writer , fmt , fps , width , height , frame_number , total_time \
    = video_information(input_video_file)
    get_movie_info.delete_st_item()
    st.write("<p><br></p>" , unsafe_allow_html=True)

    frame_number = int( cap.get(cv2.CAP_PROP_FRAME_COUNT) )

    if cap.isOpened():
        frames = []
        for count in range(frame_number):
            # process_loop_frame = us.AddOrDeleteStMessage(message = f"{count + 1}枚目/{frame_number} 枚中 のフレームを処理")
            ret, frame = cap.read()
            if ret:
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                _ , frame = resize_to_square(frame , resized_length=640)
                if ( count % fps_divide == 0 ) :
                    # フレーム画像に画像処理を実行
                    output_frame_name = "media/img/output_frame" + str(count) +".jpg"
                    output_frame = function_of_any_arguments(conversion_func , list([frame , *arguments_list]))
                    frames.append(output_frame)
                    if (fps_divide > 1):
                        cv2.imwrite(output_frame_name , output_frame)
                else:
                    # 間引いたフレームは検出が行われたフレームの出力を引き継ぐ
                    output_frame = cv2.imread(output_frame_name)
                    frames.append(output_frame)
            # process_loop_frame.delete_st_item()

        # 画像をpilに変換してリストに格納
        output_video_wait = us.AddOrDeleteStMessage(message = "処理後の動画を書き出し中")
        frames_pil_list = []
        for loop, frame in enumerate(frames):
            if loop % (fps_divide * 2) < fps_divide:
                frame = Image.fromarray(frame)
                frames_pil_list.append(frame)
        del frames
        
        # gifファイルに変換して保存
        pils_to_gifs(frames_pil_list, output_gif_file, fps)
        del frames_pil_list
        output_video_wait.delete_st_item()

        # 動画処理を閉じる
        cap.release()
        cv2.destroyAllWindows()

        return fps


def main():
    """ 静止画像・動画の加工や画像認識を実行するクラス・関数を含むモジュール """


if __name__ == "__main__":
    main()