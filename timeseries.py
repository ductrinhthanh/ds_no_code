import streamlit as st
from statsmodels.tsa.holtwinters import ExponentialSmoothing
import plotly.express as px

class TimeSeriesForecast:
    @staticmethod
    def forecast(df, forecast_col):
        st.write(f"Your model is <b style='color: green'>ExponentialSmoothing</b>.", unsafe_allow_html=True)
        return ExponentialSmoothing(endog=df[forecast_col], trend="add", seasonal="add", seasonal_periods=12).fit()

    @staticmethod
    def get_timeseries_line_chart(df, x, y):
        df = df.sort_values(by=x)
        fig = px.line(df, x=x, y=y, title="Preview")
        st.plotly_chart(fig, theme="streamlit")