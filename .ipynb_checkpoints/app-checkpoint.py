import pandas as pd
import streamlit as st
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt


data = {
    "year": [2015, 2016, 2017, 2018, 2019, 2020],
    "mileage": [80000, 70000, 60000, 50000, 40000, 30000],
    "price": [300000, 350000, 400000, 500000, 600000, 700000]
}


df = pd.DataFrame(data)


X = df[["year", "mileage"]]
y = df["price"]

model = LinearRegression()
model.fit(X, y)

st.title("Car Price Prediction Dashboard")

st.subheader("Car Dataset")
st.write(df)

year = st.number_input("Enter Car Year", 2010, 2025, 2018)
mileage = st.number_input("Enter Mileage", 10000, 200000, 50000)


if st.button("Predict Price"):
    prediction = model.predict([[year, mileage]])
    st.success(f"Predicted Car Price: ₹{prediction[0]:,.0f}")

st.subheader("Feature Importance")

importance = pd.DataFrame({
    "Feature": ["Year", "Mileage"],
    "Importance": model.coef_
})

fig, ax = plt.subplots()
ax.bar(importance["Feature"], importance["Importance"])
ax.set_title("Feature Importance")
st.pyplot(fig)