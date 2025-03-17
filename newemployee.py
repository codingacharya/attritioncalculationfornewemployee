import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
import tensorflow as tf
import torch
import torch.nn as nn
import torch.optim as optim

# Streamlit app title
st.title("🤖 AI-Driven HR Analytics - Employee Attrition Prediction")

# Upload CSV file
uploaded_file = st.file_uploader("Upload Employee Data (CSV)", type=["csv"])

if uploaded_file:
    # Load dataset
    df = pd.read_csv(uploaded_file)
    
    # Display dataset preview
    st.subheader("📂 Dataset Preview")
    st.dataframe(df.head())

    # Ensure 'Attrition' column exists
    if "Attrition" not in df.columns:
        st.error("Dataset must contain an 'Attrition' column (Yes/No).")
    else:
        # Encode categorical variable
        le = LabelEncoder()
        df["Attrition"] = le.fit_transform(df["Attrition"])  # Yes -> 1, No -> 0

        # Select relevant features
        features = ["Age", "MonthlyIncome", "JobSatisfaction", "YearsAtCompany", "WorkLifeBalance"]
        df = df.dropna()  # Handle missing values

        # Split dataset
        X = df[features]
        y = df["Attrition"]
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        # Standardize features
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)

        ### ---- MODEL 1: MACHINE LEARNING (RANDOM FOREST) ---- ###
        rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
        rf_model.fit(X_train, y_train)
        rf_pred = rf_model.predict(X_test)
        rf_accuracy = accuracy_score(y_test, rf_pred)

        st.subheader("🌲 Random Forest Model (Scikit-Learn)")
        st.write(f"**Accuracy:** {rf_accuracy:.2%}")
        st.text(classification_report(y_test, rf_pred))

        ### ---- MODEL 2: DEEP LEARNING (TENSORFLOW) ---- ###
        st.subheader("🧠 Deep Learning Model (TensorFlow)")

        tf_model = tf.keras.Sequential([
            tf.keras.layers.Dense(16, activation='relu', input_shape=(X_train.shape[1],)),
            tf.keras.layers.Dense(8, activation='relu'),
            tf.keras.layers.Dense(1, activation='sigmoid')
        ])
        tf_model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
        tf_model.fit(X_train, y_train, epochs=10, batch_size=8, verbose=0)

        tf_loss, tf_accuracy = tf_model.evaluate(X_test, y_test, verbose=0)
        st.write(f"**Accuracy:** {tf_accuracy:.2%}")

        ### ---- MODEL 3: NEURAL NETWORK (PYTORCH) ---- ###
        st.subheader("🔥 Neural Network (PyTorch)")

        class HRNeuralNet(nn.Module):
            def __init__(self, input_dim):
                super(HRNeuralNet, self).__init__()
                self.fc1 = nn.Linear(input_dim, 16)
                self.relu = nn.ReLU()
                self.fc2 = nn.Linear(16, 8)
                self.fc3 = nn.Linear(8, 1)
                self.sigmoid = nn.Sigmoid()

            def forward(self, x):
                x = self.relu(self.fc1(x))
                x = self.relu(self.fc2(x))
                x = self.sigmoid(self.fc3(x))
                return x

        # Convert data to PyTorch tensors
        X_train_torch = torch.tensor(X_train, dtype=torch.float32)
        y_train_torch = torch.tensor(y_train.values, dtype=torch.float32).view(-1, 1)
        X_test_torch = torch.tensor(X_test, dtype=torch.float32)
        y_test_torch = torch.tensor(y_test.values, dtype=torch.float32).view(-1, 1)

        # Initialize model
        pytorch_model = HRNeuralNet(X_train.shape[1])
        criterion = nn.BCELoss()
        optimizer = optim.Adam(pytorch_model.parameters(), lr=0.01)

        # Train model
        for epoch in range(50):
            optimizer.zero_grad()
            output = pytorch_model(X_train_torch)
            loss = criterion(output, y_train_torch)
            loss.backward()
            optimizer.step()

        # Evaluate model
        with torch.no_grad():
            test_preds = pytorch_model(X_test_torch)
            test_preds = (test_preds.numpy() > 0.5).astype(int)
            pytorch_accuracy = accuracy_score(y_test, test_preds)

        st.write(f"**Accuracy:** {pytorch_accuracy:.2%}")

        ### ---- FEATURE IMPORTANCE ---- ###
        st.subheader("📊 Feature Importance (Random Forest)")
        feature_importance = pd.DataFrame({"Feature": features, "Importance": rf_model.feature_importances_})
        feature_importance = feature_importance.sort_values(by="Importance", ascending=False)
        st.dataframe(feature_importance)

        ### ---- VISUALIZATIONS ---- ###
        st.subheader("📉 Attrition Distribution")
        fig, ax = plt.subplots()
        sns.countplot(x="Attrition", data=df, ax=ax, palette="Set2")
        st.pyplot(fig)

        st.subheader("💰 Salary vs Attrition")
        fig, ax = plt.subplots()
        sns.boxplot(x="Attrition", y="MonthlyIncome", data=df, ax=ax, palette="coolwarm")
        st.pyplot(fig)

        ### ---- PREDICTION ON NEW EMPLOYEE ---- ###
        st.subheader("🔮 Predict Attrition for a New Employee")
        age = st.number_input("Age", min_value=18, max_value=65, value=30)
        monthly_income = st.number_input("Monthly Income", min_value=1000, max_value=50000, value=5000)
        job_satisfaction = st.slider("Job Satisfaction (1-4)", min_value=1, max_value=4, value=3)
        years_at_company = st.slider("Years at Company", min_value=0, max_value=40, value=5)
        work_life_balance = st.slider("Work-Life Balance (1-4)", min_value=1, max_value=4, value=3)

        if st.button("Predict"):
            user_data = np.array([[age, monthly_income, job_satisfaction, years_at_company, work_life_balance]])
            user_data_scaled = scaler.transform(user_data)

            # Get predictions from models
            rf_prediction = rf_model.predict(user_data_scaled)[0]
            tf_prediction = (tf_model.predict(user_data_scaled) > 0.5).astype(int)[0][0]
            pytorch_prediction = (pytorch_model(torch.tensor(user_data_scaled, dtype=torch.float32)).detach().numpy() > 0.5).astype(int)[0][0]

            # Display results
            st.write(f"🌲 **Random Forest Prediction:** {'Yes' if rf_prediction == 1 else 'No'}")
            st.write(f"🧠 **TensorFlow Prediction:** {'Yes' if tf_prediction == 1 else 'No'}")
            st.write(f"🔥 **PyTorch Prediction:** {'Yes' if pytorch_prediction == 1 else 'No'}")

else:
    st.info("Please upload a CSV file to analyze attrition risk.")

