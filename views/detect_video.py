import streamlit as st
import views.useful_st as us
import views.convert_img as ci
import forms
import numpy as np
import os
import moviepy.editor as mp
from googleapiclient.discovery import build
from googleapiclient.http import MediaIoBaseDownload
# Googleドライブ APIの認証
from google.oauth2 import service_account


class ManipulateGoogleDriveAPI:
    """ GoogleDriveのAPIを用いて操作するクラス """
    def __init__(self , json_file_path: str = 'static/json/google_drive_api.json') -> None:
        self.json_file_path = json_file_path
        pass

    def download_drive_file(self, file_id: str, destination: str):
        """
        Googleドライブ上のファイルを取得する関数
        parameters:
            file_id : 「リンクを共有」で取得したURL「https://drive.google.com/file/d/{file_id}/view」から取得可能
            destination : ダウンロードしたファイルの保存先のパス
            json_file_path : GCPのAPIを利用可能にするjsonファイルのパス
        """
        # jsonファイルの定義
        credentials = service_account.Credentials.from_service_account_file(
            self.json_file_path, scopes=['https://www.googleapis.com/auth/drive']
        )

        # Googleドライブ APIのビルド
        service = build('drive', 'v3', credentials=credentials)

        # ファイルIDを指定してダウンロード
        request = service.files().get_media(fileId=file_id)
        fh = open(destination, 'wb')
        downloader = MediaIoBaseDownload(fh, request)

        done = False
        while done is False:
            status, done = downloader.next_chunk()
        fh.close()


def download_file(file_path , download_message: str = "ファイルをダウンロード"):
    with open(file_path, "rb") as file:
        file_data = file.read()
    st.download_button(download_message, file_data, file_path)

def gifs_to_mp4(output_gif_file: str, output_video_file: str, fps: int):
    # gif動画ファイルの読み込み
    gif_movie = mp.VideoFileClip(output_gif_file)
    #mp4動画ファイルの保存
    gif_movie.write_videofile(output_video_file, fps, )
    gif_movie.close()


def result_download_button(page_subtitle: str = "<h1>動画の加工・物体検出</h1><p></p>" , video_folder: str = "media/video/"):
    st.write(page_subtitle, unsafe_allow_html=True)
    output_video_file = video_folder + "output_video.mp4"
    download_file(output_video_file , download_message = "処理後の動画をダウンロード")



def main(page_subtitle: str = "<h1>動画の加工・物体検出</h1><p></p>" , video_folder: str = "media/video/"):
    st.write(page_subtitle, unsafe_allow_html=True)
    # formに入力する文字列を定義
    choiced_object_form_name = st.selectbox("検出したい物体を選択", forms.label_choice[:, 1].tolist() , index=0 )
    inside_or_background_form_name = st.selectbox("内側と外側のどちらかを選択", forms.inside_or_background[:, 1].tolist() , index=0 )
    choiced_convert_form_name = st.selectbox("実行したい画像加工の内容を選択", forms.convert_choice[:, 1].tolist() , index=0 )

    st.write("<p></p>", unsafe_allow_html=True)
    display_html = """
        '<span style="font-size:75%;">検出したい物体によって、検出精度が異なります。
        <br>「人の顔」は基本的に画像内に締める割合が最も大きい顔のみ検出されます。
        <br>「人の顔」以外は画像内のあらゆる物体を検出することができます。 </span> """
    st.write(display_html, unsafe_allow_html=True)

    # formの情報を変数や関数として変換
    choiced_object_label = forms.label_choice[forms.label_choice[:, 1] == choiced_object_form_name][0, 0]
    inside_or_background = forms.inside_or_background[forms.inside_or_background[:, 1] == inside_or_background_form_name][0, 0]
    choiced_convert_func = forms.convert_choice[forms.convert_choice[:, 1] == choiced_convert_form_name][0, 0]
    # 検出物体に行う画像処理の関数と引数を決定
    convert_object_func, convert_object_parameter = ci.form_name_to_convert_func(choiced_convert_func)

    st.write("<p></p>", unsafe_allow_html=True)
    st.write("<p></p>", unsafe_allow_html=True)

    uploaded_file = st.file_uploader("動画ファイルをアップロードしてください", type=["mp4"])
    st.write("")
    input_video_file = video_folder + "input_video.mp4"
    output_gif_file = video_folder + "output_video.gif"
    output_video_file = video_folder + "output_video.mp4"

    if uploaded_file:
        # アップロードされた動画をローカルディレクトリに保存
        with open(input_video_file, "wb") as f:
            f.write(uploaded_file.getbuffer())

        # 動画のフレーム全体に画像加工
        if choiced_object_label == "all":
            frame_func = convert_object_func
            frame_func_parameter = convert_object_parameter
            fps_divide = 4
        
        # 動画のフレーム内の検出物体のみに画像加工
        else:
            # ONNX変換された学習済みYOLOv5モデルのファイルパス
            model_folder = "models/"
            onnx_model_path = model_folder + "yolov5x" + ".onnx"
            # ONNXモデルのダウンロード
            if not os.path.exists(onnx_model_path):
                st.write(f"AIを呼び出し中")
                google_drive = ManipulateGoogleDriveAPI()
                google_drive.download_drive_file('1ZEilvzHUSJXdABuIrev4vNHaaUPgGGml', onnx_model_path)
            
            # 推論器の準備
            yolov5_onnx_detector = ci.Yolov5OnnxDetector(onnx_model_path)
            frame_func = yolov5_onnx_detector.convert_detected_area
            # 画像処理の内容と検出物体の情報をnumpyに格納
            frame_func_parameter = np.array([
                choiced_object_label, inside_or_background, convert_object_func, convert_object_parameter,
            ])
            fps_divide = 4

        # 動画の物体検出を実行し、gif画像を取得して書き出し
        fps = ci.st_video_convert(
            input_video_file ,output_gif_file  ,
            frame_func ,[frame_func_parameter] ,fps_divide=fps_divide
        )
        # gif動画をmp4動画に変換
        gif_to_mp4 = us.AddOrDeleteStMessage(message="動画を書き出し中")
        ci.gifs_to_mp4(output_gif_file , output_video_file , fps)
        gif_to_mp4.delete_st_item()
        return output_video_file


if __name__ == '__main__':
    main()