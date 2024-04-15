import streamlit as st
import time

class Utils:
    @staticmethod
    def show_progress_bar():

        progress_text = "Operation in progress. Please wait."
        progress_text2 = "Almost done ...."
        my_bar = st.progress(0, text=progress_text)

        for percent_complete in range(100):
            time.sleep(0.01)
            if percent_complete < 75:
                my_bar.progress(percent_complete + 1, text=progress_text)
            else:
                my_bar.progress(percent_complete + 1, text=progress_text2)
        time.sleep(1)
        my_bar.empty()

    @staticmethod
    def split_list(lst, n):
        """
        Split a list into n approximately equal parts.
        """
        k, m = divmod(len(lst), n)
        return [lst[i * k + min(i, m):(i + 1) * k + min(i + 1, m)] for i in range(n)]