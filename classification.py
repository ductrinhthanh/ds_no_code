import pandas as pd
import sklearn as sk
from sklearn import metrics
import plotly.express as px
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn import svm
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
import streamlit as st
import numpy as np
from sklearn.datasets import make_multilabel_classification
from sklearn.multioutput import MultiOutputClassifier
from sklearn.linear_model import LogisticRegressionCV
from sklearn.preprocessing import LabelEncoder
from sklearn.multiclass import OneVsRestClassifier
from utils import Utils
from pandas.api.types import is_numeric_dtype


class CustomerChurnPredict:
    @staticmethod
    def clean(df):
        return pd.get_dummies(df, drop_first=True)

    @staticmethod
    def predict(X, y):
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.25, random_state=16)
        logreg = LogisticRegression(random_state=16)
        logreg.fit(X_train, y_train)
        y_pred = logreg.predict(X_test)
        accuracy = metrics.accuracy_score(y_test, y_pred)*100
        st.write(
            f"Your model is <b style='color: green'>Logistic Regression</b> with accuracy of <b style='color: green'>{accuracy:.2f}%</b>. This is <b style='color: green'>great</b> performance.", unsafe_allow_html=True)
        return logreg

    @staticmethod
    def use_model(df):
        # Split the columns into 3 equal parts
        list_columns = df.columns.values.tolist()
        df_cols = Utils.split_list(list_columns, 3)

        # Create 3 columns for input widgets
        col1, col2, col3 = st.columns(3)
        values = []
        # Iterate over the columns in each part
        for i, cols in enumerate([df_cols[0], df_cols[1], df_cols[2]]):
            with [col1, col2, col3][i]:
                for col in cols:
                    if df[col].dtypes in [object, bool]:
                        value = st.selectbox(col, options=df[col].unique())
                    else:
                        value = st.number_input(col, value=df[col].mean(), key=col)
                    values.append(value)

        df_test = pd.DataFrame(data=[values], columns=list_columns)
        return df_test

    @staticmethod
    def draw_pie(probs):
        data = {
            'type': ['Retain', 'Churn'],
            'probability': probs[0]
        }
        # st.write(probs)
        df = pd.DataFrame(data)
        fig = px.pie(df, values='probability', names='type', color='type',
                     title='Predict Probability', color_discrete_map={'Churn': 'red', 'Retain': 'green'})
        st.plotly_chart(fig, theme='streamlit')
        # st.write(df)


class CustomerPlanPredict:
    # Multi class classification
    @staticmethod
    def clean(df_train):
        x = df_train.select_dtypes(exclude=['number']).apply(LabelEncoder().fit_transform)
        x = x.join(df_train.select_dtypes(include=['number']))
        return x

    
    @staticmethod
    def encode_label(df_train, label_col):
        # st.write(df_train[label_col].unique())
        original_labels = df_train[label_col].unique()
        label_encoder = LabelEncoder()
        encoded_labels = label_encoder.fit_transform(original_labels)
        # st.write(encoded_labels)
        return label_encoder, original_labels
    
    @staticmethod
    def predict(X, y):
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.25, random_state=16)
        
        # Initialize a logistic regression classifier
        base_classifier = LogisticRegression(max_iter=1000, random_state=42)

        # Create the One-vs-Rest classifier
        ovr_classifier = OneVsRestClassifier(base_classifier)

        # Fit the model to the training data
        ovr_classifier.fit(X_train, y_train)

        y_pred = ovr_classifier.predict(X_test)
        accuracy = metrics.accuracy_score(y_test, y_pred)*100

        if accuracy > 70:
            st.write(
                f"Your model is <b style='color: green'>Logistic Regression</b> with accuracy of <b style='color: green'>{accuracy:.2f}%</b>. This is <b style='color: green'>great</b> performance.", unsafe_allow_html=True)
        else:
            st.write(
                f"Your model is <b style='color: green'>Logistic Regression</b> with accuracy of <b style='color: green'>{accuracy:.2f}%</b>. This is <b style='color: yellow'>not so good</b> performance.", unsafe_allow_html=True)
        return ovr_classifier
    
    @staticmethod
    def use_model(df):
        # Split the columns into 3 equal parts
        list_columns = df.columns.values.tolist()
        df_cols = Utils.split_list(list_columns, 3)

        # Create 3 columns for input widgets
        col1, col2, col3 = st.columns(3)
        values = []
        # Iterate over the columns in each part
        for i, cols in enumerate([df_cols[0], df_cols[1], df_cols[2]]):
            with [col1, col2, col3][i]:
                for col in cols:
                    if df[col].dtypes in [object, bool]:
                        value = st.selectbox(col, options=df[col].unique())
                    else:
                        value = st.number_input(col, value=df[col].mean())
                    values.append(value)

        df_test = pd.DataFrame(data=[values], columns=list_columns)
        return df_test
    
