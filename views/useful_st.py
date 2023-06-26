import streamlit as st

class AddOrDeleteStMessage:
    """
    streamlitアプリでメッセージを追加or後で削除するクラス 
    使用例 :
        add_or_delete = AddOrDeleteStMessage(message = "処理中")
        time.sleep(5)
        add_or_delete.delete_st_item()
    挙動 :
        上記の使用例の場合、5秒間メッセージ「処理中」が表示されて5秒後に消失する
    """
    def __init__(self , message: str = "Hello World") -> None:
        """ メッセージを表示する関数 """
        self.my_element = st.empty()
        self.my_element.write(message)

    def delete_st_item(self) -> None:
        """ メッセージを削除する関数 """
        self.my_element.empty()


def main():
    """ streamlitアプリで利用できる便利なクラス・関数を含むモジュール """

if __name__ == "__main__":
    main()