import streamlit as st
from views import detect_img , detect_video

st.title('画像加工アプリ')
st.write('\n')

def change_page(page):
    st.session_state["page"] = page

def detect_img_page():
    # 別ページへの遷移
    st.button("動画の加工・物体検出はこちら >", on_click=change_page, args=["detect_video_page"])
    # 画像処理の実行
    page_subtitle = "<p></p><h3>静止画の加工・物体検出</h3>"
    detect_img.main(page_subtitle)

def detect_video_page():
    # 別ページへの遷移
    st.button("静止画の加工・物体検出はこちら >", on_click=change_page, args=["detect_img_page"])
    # 画像処理の実行
    page_subtitle = "<p></p><h3>動画の加工・物体検出</h3>"
    output_video_file = detect_video.main(page_subtitle)
    return output_video_file

def result_video_page():
    # 別ページへの遷移
    st.button("再度動画の加工・物体検出はこちら >", on_click=change_page, args=["detect_video_page"])
    # 画像処理の実行
    page_subtitle = "<p></p><h3>動画の加工・物体検出</h3>"
    detect_video.result_download_button(page_subtitle)




# メイン
def main():
    # セッション状態を取得
    session_state = st.session_state

    # セッション状態によってページを表示
    if "page" not in session_state:
        session_state["page"] = "detect_img_page"

    if session_state["page"] == "detect_img_page":
        detect_img_page()
    if session_state["page"] == "detect_video_page":
        output_video_file = detect_video_page()
        if output_video_file:
            session_state["page"] = "result_video_page"
            st.experimental_rerun()
    if session_state["page"] == "result_video_page":
        result_video_page()

if __name__ == "__main__":
    main()
