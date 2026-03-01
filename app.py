import streamlit as st
import pandas as pd
import os
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from scipy.optimize import linprog

st.set_page_config(page_title="Formwork Optimization", layout="wide")

st.title("🏗 Automation of Formwork Kitting & BoQ Optimization")

# ---------------------------------------------------
# 1️⃣ Load Default CSV
# ---------------------------------------------------
DEFAULT_FILE = "project_data.csv"

def load_default_data():
    if os.path.exists(DEFAULT_FILE):
        return pd.read_csv(DEFAULT_FILE)
    else:
        return None

# ---------------------------------------------------
# 2️⃣ Drag & Drop File Upload
# ---------------------------------------------------
uploaded_file = st.file_uploader(
    "📂 Drag & Drop your CSV file here",
    type=["csv"]
)

# If user uploads file → use it
if uploaded_file is not None:
    data = pd.read_csv(uploaded_file)
    st.success("Using Uploaded CSV File ✅")
else:
    data = load_default_data()
    if data is not None:
        st.info("Using Default Project CSV File 📊")
    else:
        st.error("No CSV file found.")
        st.stop()

st.subheader("📊 Project Data")
st.dataframe(data)

# ---------------------------------------------------
# 3️⃣ Repetition Detection
# ---------------------------------------------------
st.subheader("🔍 Repetition Analysis")

features = data[['columns', 'walls', 'area']]
scaler = StandardScaler()
scaled = scaler.fit_transform(features)

kmeans = KMeans(n_clusters=2, random_state=42)
data['Repetition Group'] = kmeans.fit_predict(scaled)

st.dataframe(data[['floor', 'Repetition Group']])

# ---------------------------------------------------
# 4️⃣ BoQ Prediction
# ---------------------------------------------------
st.subheader("📈 BoQ Prediction")

X = data[['area', 'walls']]
y = data['columns'] * 2

model = LinearRegression()
model.fit(X, y)

col1, col2 = st.columns(2)

with col1:
    area = st.number_input("Enter Area", value=1200)

with col2:
    walls = st.number_input("Enter Walls", value=50)

if st.button("Predict Formwork Requirement"):
    prediction = model.predict([[area, walls]])
    st.success(f"Predicted Formwork Required: {prediction[0]:.2f} units")

# ---------------------------------------------------
# 5️⃣ Inventory Optimization
# ---------------------------------------------------
st.subheader("📦 Inventory Optimization")

demand = st.number_input("Total Required Units", value=100)

cost = [500, 700]
c = cost
A = [[1, 1]]
b = [demand]

bounds = [(0, demand), (0, demand)]

result = linprog(c, A_ub=A, b_ub=b, bounds=bounds)

if st.button("Optimize Inventory"):
    st.success(
        f"Optimal Allocation → Type1: {result.x[0]:.2f} | Type2: {result.x[1]:.2f}"
    )

# ---------------------------------------------------
# 6️⃣ Kit Generator
# ---------------------------------------------------
st.subheader("🛠 Automated Kit Generator")

col = st.number_input("Columns for Kit", value=20)
wall = st.number_input("Walls for Kit", value=50)

if st.button("Generate Kit"):
    kit = {
        "Column Panels": col * 4,
        "Wall Panels": wall * 2,
        "Clamps": (col + wall) * 3
    }
    st.json(kit)
