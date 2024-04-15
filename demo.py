import streamlit as st
import pandas as pd
import plotly.express as px
from timeseries import TimeSeriesForecast
from datetime import date, timedelta
from sklearn.preprocessing import StandardScaler
from classification import CustomerChurnPredict, CustomerPlanPredict
from clustering import CustomerSegmentation
from utils import Utils

st.set_page_config(page_title="DS No Code", page_icon="static/Teko-logo.png")

if "customer_churn_predict" not in st.session_state:
    st.session_state.customer_churn_predict = False
if "select_churn_col" not in st.session_state:
    st.session_state.customer_churn_predict = False
if "do_segmentation" not in st.session_state:
    st.session_state.do_segmentation = False
if "predict_segmentation" not in st.session_state:
    st.session_state.predict_segmentation = False
if "load_progress_bar" not in st.session_state:
    st.session_state.load_progress_bar = 0
if "predict_customer_plan" not in st.session_state:
    st.session_state.predict_customer_plan = False
if "predict_customer_churn" not in st.session_state:
    st.session_state.predict_customer_churn = False


def timeseries_forecast():
    upload_df = st.file_uploader(
        "Upload dataset", accept_multiple_files=False, type='csv', key='timeseries_forecast_upload')

    if upload_df is not None:
        df = pd.read_csv(upload_df)
        st.write(df)
        # Select columns:
        time_col = st.selectbox("Select time column:",
                                df.columns.values.tolist())
        metric_col = st.selectbox("Select metric to forecast:", [
                                  i for i in df.columns.values.tolist() if i != time_col])
        # Preview
        if st.session_state.load_progress_bar != 1:
            Utils.show_progress_bar()
            st.session_state.load_progress_bar = 1
        TimeSeriesForecast.get_timeseries_line_chart(
            df, x=time_col, y=metric_col)
        # select numbers of periods
        number_of_period = st.number_input(
            "Insert numbers of periods to forecast", value=10, key='timeseries_periods')
        is_forecast = st.button("Forecast")
        if is_forecast:
            #progressbar
            Utils.show_progress_bar()
            # Forecast
            model = TimeSeriesForecast.forecast(df=df, forecast_col=metric_col)
            # Viz
            x_preds = pd.date_range(start=df[time_col].max(
            ), periods=number_of_period, freq=pd.offsets.MonthBegin(1)).strftime('%Y-%m-%d')
            y_preds = model.forecast(steps=number_of_period)
            forecast_df = pd.DataFrame({
                time_col: x_preds,
                metric_col: y_preds,
            })
            forecast_df['type'] = 'forecast'
            df['type'] = 'actual'
            final_df = df.append(forecast_df).sort_values(by=time_col)
            fig = px.line(final_df, x=time_col, y=metric_col, title='Forecast Values Over Time',
                          color='type', color_discrete_sequence=["#83C9FF", "yellow"])
            st.plotly_chart(fig, theme="streamlit")
            st.write(forecast_df[[time_col, metric_col]])


def customer_churn_predict():
    upload_df = st.file_uploader(
        "Upload dataset", accept_multiple_files=False, type='csv', key='customer_churn_predict_upload')
    if upload_df is not None:
        df = pd.read_csv(upload_df)
        df = df.dropna()
        st.write(df)
        # Select columns:
        customer_col = st.selectbox("Select Customer column:", df.columns.values.tolist(
        ), index=None, placeholder="Select customer column...")
        label_col = st.selectbox("Churn column:", [i for i in df.columns.values.tolist(
        ) if i != customer_col], index=None, placeholder="Select churn column...")
        
        if (customer_col and label_col):
            # progress bar
            if st.session_state.load_progress_bar != 2:
                Utils.show_progress_bar()
                st.session_state.load_progress_bar = 2
            st.write("<b style='color:green; font-size:24px'>Your model is ready !!!</b>", unsafe_allow_html=True)
            
            train_df = df[[i for i in df.columns.values.tolist(
            ) if i not in [customer_col, label_col]]]

            X = CustomerChurnPredict.clean(train_df)
            y = df[label_col]
            

            is_predict = st.button("Predict")
            if is_predict or st.session_state.predict_customer_plan:
                #progressbar
                Utils.show_progress_bar()
                st.session_state.predict_customer_plan = True
            
                model = CustomerChurnPredict.predict(X=X, y=y)
                st.write("Change values to predict outcome")
                test_df = CustomerChurnPredict.clean(
                pd.concat([train_df, CustomerChurnPredict.use_model(df=train_df)])).iloc[-1:]
                probs = model.predict_proba(test_df)
                st.write(
                    f'This model predicts that this customer has <br> <b style="color:red; font-size:24px">{probs[0][1]*100:.2f}% to churn</b> or <b style="color:green; font-size:24px">{probs[0][0]*100:.2f}% to retain</b>', unsafe_allow_html=True)
                CustomerChurnPredict.draw_pie(probs=probs)


def do_customer_segmentation():
    upload_df = st.file_uploader(
        "Upload dataset", accept_multiple_files=False, type='csv', key='do_customer_segmentation_upload')
    if upload_df is not None:
        df = pd.read_csv(upload_df)
        st.write(df)
        customer_col = st.selectbox("Select Customer column:", df.columns.values.tolist(
        ), index=None, placeholder="Select customer column...")

        if customer_col:
            train_df = df[[i for i in df.columns.values.tolist(
            ) if i not in [customer_col]]]
            # progress bar
            if st.session_state.load_progress_bar !=3:
                Utils.show_progress_bar()
                st.session_state.load_progress_bar = 3

            test_K = 12
            CustomerSegmentation.try_different_clusters(K=test_K, data=train_df)

            K = st.number_input(
            "Insert numbers of segments to divide into", value=2, key="segments")

            is_do_segmentation = st.button("Do segmentation")
            if is_do_segmentation or st.session_state.do_segmentation:
                Utils.show_progress_bar()
                model = CustomerSegmentation.do_segment(K=K, df=train_df)
                st.session_state.do_segmentation = True

                df_new = df.copy()
                NEW_COLUMN = "segmentation_group"
                df_new[NEW_COLUMN] = model.labels_
                st.write("Here comes your customer data after segmentation")
                st.write(df_new)
                df_cols = [i for i in df.columns.values.tolist() if i not in [customer_col, NEW_COLUMN]]
                CustomerSegmentation.visualize_cluster_scatter3d(df=df_new, segmentation_col=NEW_COLUMN, df_cols=df_cols)
                for col in df_cols:
                    if df_new[col].dtypes == 'object':
                        CustomerSegmentation.visualize_cluster_bar(df=df_new, x=col, segmentation_col=NEW_COLUMN)
                    else:
                        CustomerSegmentation.visualize_cluster_hist(df=df_new, x=col, k=K, segmentation_col=NEW_COLUMN)

                st.write("Change values to predict outcome")
                test_df = pd.concat([train_df, CustomerSegmentation.use_model(df=train_df)]).iloc[-1:]
                predict_group = model.predict(test_df)
                st.write(
                    f'This model predicts that this customer would be in group <b style="color:green">{predict_group[0]}</b>', unsafe_allow_html=True)
                
def customer_plan_predict():
    upload_df = st.file_uploader(
        "Upload dataset", accept_multiple_files=False, type='csv', key='customer_plan_predict_upload')
    if upload_df is not None:
        df = pd.read_csv(upload_df)
        st.write(df)
        customer_col = st.selectbox("Select Customer column:", df.columns.values.tolist(
        ), index=None, placeholder="Select customer column...")
        label_col = st.selectbox("Plans column:", [i for i in df.columns.values.tolist(
        ) if i != customer_col], index=None, placeholder="Select customer plan column...")

        if (customer_col and label_col):
            train_df = df[[i for i in df.columns.values.tolist(
            ) if i not in [customer_col]]]

            df_clean = CustomerPlanPredict.clean(train_df)
            X = df_clean.drop(label_col, axis=1)
            y = df_clean[[label_col]]

            # st.write(X)
            # st.write(y)

            is_predict = st.button("Predict")
            if is_predict or st.session_state.predict_customer_plan:
                #progressbar
                Utils.show_progress_bar()
                st.session_state.predict_customer_plan = True

                X_for_test = train_df.drop(label_col, axis=1)
                # st.write(X_for_test["page_count"].dtypes)

                (encode_label_model, original_labels) = CustomerPlanPredict.encode_label(train_df, label_col)

                model = CustomerPlanPredict.predict(X=X, y=y)
                test_df = CustomerPlanPredict.clean(
                pd.concat([X_for_test, CustomerPlanPredict.use_model(df=X_for_test)])).iloc[-1:]
                predicted_plan = encode_label_model.inverse_transform(model.predict(test_df))
                st.write(f"This model predicts that this customer may choose plan <b style='color:green'>{predicted_plan[0]}</b>", unsafe_allow_html=True)
                st.write(model.predict_proba(test_df))



page_names_to_funcs = {
    "Timeseries Forecast": timeseries_forecast,
    "Customer Churn Prediction": customer_churn_predict,
    "Customer Segmentation": do_customer_segmentation,
    "Customer Plans Prediction": customer_plan_predict,
}

demo_name = st.sidebar.selectbox("Choose a model", page_names_to_funcs.keys())
page_names_to_funcs[demo_name]()
