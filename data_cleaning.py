import pandas as pd
from sklearn.preprocessing import OrdinalEncoder
import joblib


df = pd.read_csv("data/rawdata.csv")

newdf = df.drop("customerID", axis="columns")

newdf['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')

categorical_cols = newdf.select_dtypes(include=['object', 'category']).columns.to_list()

encoder = OrdinalEncoder()
encooded_categories = encoder.fit_transform(newdf[categorical_cols])

newdf[categorical_cols] = encooded_categories

print(newdf.head())

newdf.to_csv("data/processed_data.csv", index=False)

