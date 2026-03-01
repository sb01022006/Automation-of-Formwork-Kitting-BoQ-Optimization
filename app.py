import streamlit as st
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from scipy.optimize import linprog

st.title("🏗 Formwork Kitting & BoQ Optimization System")

# Upload Data
file = st.file_uploader("Upload Project CSV", type=["csv"])

if file:
    data = pd.read_csv(file)
    st.subheader("📊 Project Data")
    st.write(data)

    # ----------------------------
    # 1️⃣ Repetition Detection
    # ----------------------------
    st.subheader("🔍 Repetition Analysis")

    features = data[['columns', 'walls', 'area']]
    scaler = StandardScaler()
    scaled = scaler.fit_transform(features)

    kmeans = KMeans(n_clusters=2, random_state=42)
    data['Repetition Group'] = kmeans.fit_predict(scaled)

    st.write(data[['floor', 'Repetition Group']])

    # ----------------------------
    # 2️⃣ BoQ Prediction
    # ----------------------------
    st.subheader("📈 BoQ Prediction")

    X = data[['area', 'walls']]
    y = data['columns'] * 2  # Example formwork calculation

    model = LinearRegression()
    model.fit(X, y)

    area = st.number_input("Enter Area")
    walls = st.number_input("Enter Number of Walls")

    if st.button("Predict Formwork Requirement"):
        prediction = model.predict([[area, walls]])
        st.success(f"Predicted Formwork Required: {prediction[0]:.2f} units")

    # ----------------------------
    # 3️⃣ Inventory Optimization
    # ----------------------------
    st.subheader("📦 Inventory Cost Optimization")

    demand = st.number_input("Total Required Units", value=100)

    cost = [500, 700]  # Cost of 2 types of formwork

    c = cost
    A = [[1, 1]]
    b = [demand]

    bounds = [(0, demand), (0, demand)]

    result = linprog(c, A_ub=A, b_ub=b, bounds=bounds)

    if st.button("Optimize Inventory"):
        st.success(f"Optimal Allocation: Type1 = {result.x[0]:.2f}, Type2 = {result.x[1]:.2f}")

    # ----------------------------
    # 4️⃣ Kit Generator
    # ----------------------------
    st.subheader("🛠 Automated Kit Generator")

    col = st.number_input("Columns for Kit", value=20)
    wall = st.number_input("Walls for Kit", value=50)

    if st.button("Generate Kit"):
        kit = {
            "Column Panels": col * 4,
            "Wall Panels": wall * 2,
            "Clamps": (col + wall) * 3
        }
        st.write(kit)
