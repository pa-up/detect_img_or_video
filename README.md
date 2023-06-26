# detect_img_or_video
静止画 or 動画を加工・物体検出するWEBアプリ

<h2>アプリ起動URL</h2>
https://detect-img-or-video.streamlit.app/
※ streamlit cloudにデプロイしております。

<h2>アプリ機能説明</h2>
静止画像または動画（フレーム毎）に対して、画像加工および物体検出ができます。
物体検出に関しては、
<ul>
<li>検出したい物体</li>
<li>検出した物体の内側と外側のどちらを加工するか</li>
<li>画像加工の内容（モザイク化、白黒化など...）</li>
</ul>
を入力できるフォームを用いて、特定の検出物体にのみ画像加工を行えます。<br>
例）「物体「車」以外の部分をモザイク処理」など...

<h2>ソースコード・環境</h2>
本WEBアプリの物体の検出は、学習済みYolov5xモデルを事前にローカル環境内でONNX変換したファイルを用いて推論を行なっています。<br>
また、アプリのフレームワークはStreamlitを用いており、Streamlit Cloudにデプロイしております。
