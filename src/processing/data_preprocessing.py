import pandas as pd
import numpy as np
from sklearn.impute import KNNImputer
import numpy as np
import matplotlib.pyplot as plt


def clean_data(df: pd.DataFrame):

    ### Remove useless column
    df = df.drop(columns=['Unnamed: 0'])


    ### Handle missing values
    cols_with_zeros = ['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI']
    df[cols_with_zeros] = df[cols_with_zeros].replace(0, np.nan)


    ### KNN Imputation for Missing Values
    numeric_cols = [ 'Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI']
    numeric_df = df[numeric_cols]
    imputer = KNNImputer(n_neighbors=5)
    imputed_data = imputer.fit_transform(numeric_df)
    df[numeric_cols] = pd.DataFrame(imputed_data, columns=numeric_cols)


    ### Detect outliers IQR
    insulin_max = 650
    bmi_max = 65
    bloodP_max = 120
    bloodP_min = 40
    skinthick_max = 65

    df = df[
        (df['Insulin'] <= insulin_max) &
        (df['BMI'] <= bmi_max) &
        (df['BloodPressure'] <= bloodP_max) &
        (df['BloodPressure'] >= bloodP_min) &
        (df['SkinThickness'] <= skinthick_max)
    ]


    ### Standardize numerical features.
    from sklearn.preprocessing import StandardScaler
    numeric_cols = df.select_dtypes(include=['number']).columns
    scaler = StandardScaler()
    df[numeric_cols] = scaler.fit_transform(df[numeric_cols])


    return df