import streamlit as st
import pandas as pd
import datetime
import os
import pickle
import numpy as np
import warnings

warnings.filterwarnings("ignore", category=UserWarning)

# External modules (make sure these are in your PYTHONPATH)
import RFM_Analysis
import KMeans_Model
import RandomForest_Model

# Path to your ZIP container and internal CSV name
CSV_FILE = "Online_Retail_Train.zip"
ARCHIVE_NAME = "Online_Retail_Train.csv"

def add_entry_to_csv(invoice_no, stock_code, description, quantity, unit_price, customer_id, country):
    """
    Appends a new transaction to the CSV inside a ZIP file.
    """
    current_datetime = datetime.datetime.now().strftime("%d-%m-%Y %H:%M")
    new_row = {
        "InvoiceNo": invoice_no,
        "StockCode": stock_code,
        "Description": description,
        "Quantity": quantity,
        "InvoiceDate": current_datetime,
        "UnitPrice": unit_price,
        "CustomerID": customer_id,
        "Country": country
    }
    df_new = pd.DataFrame([new_row])

    # Read existing data from the ZIP (if present)
    if os.path.exists(CSV_FILE):
        df_existing = pd.read_csv(CSV_FILE, compression="zip")
        df_updated = pd.concat([df_existing, df_new], ignore_index=True)
    else:
        df_updated = df_new

    # Write back into the ZIP, preserving archive structure
    df_updated.to_csv(
        CSV_FILE,
        index=False,
        compression={
            "method": "zip",
            "archive_name": ARCHIVE_NAME
        }
    )
    return df_updated

def show_csv_visualizations(df):
    """
    Renders Streamlit charts for transaction date, quantity, and unit price.
    """
    st.subheader("Transactions by Date")
    df["InvoiceDate"] = pd.to_datetime(df["InvoiceDate"], errors="coerce")
    by_date = df.groupby(df["InvoiceDate"].dt.date).size()
    st.line_chart(by_date)

    st.subheader("Quantity Distribution")
    if not df["Quantity"].empty:
        qc = df["Quantity"].value_counts().sort_index()
        st.bar_chart(qc)

    st.subheader("Unit Price Distribution")
    if not df["UnitPrice"].empty:
        upc = df["UnitPrice"].value_counts().sort_index()
        st.bar_chart(upc)

def show_rfm_visualizations(recency, frequency, monetary, r, f, m, cluster_name):
    """
    Renders comparison of actual vs. predicted RFM, and shows cluster.
    """
    df_cmp = pd.DataFrame({
        "Actual":    [recency, frequency, monetary],
        "Predicted": [r,       f,         m      ]
    }, index=["Recency", "Frequency", "Monetary"])
    st.subheader("Comparison of Calculated and Predicted RFM Values")
    st.bar_chart(df_cmp)

    st.subheader("Predicted Customer Cluster")
    st.write(f"The customer is predicted to belong to cluster: **{cluster_name}**")

class RFMProcessor:
    def perform_rfm_analysis(self, invoice_no, quantity, unit_price, customer_id):
        """
        Computes raw RFM scores for the incoming transaction.
        """
        try:
            existing_ids = RFM_Analysis.RFMScores["CustomerID"].tolist()
            if customer_id not in existing_ids:
                df_in = pd.DataFrame([{
                    "InvoiceNo": invoice_no,
                    "Quantity": quantity,
                    "UnitPrice": unit_price,
                    "InvoiceDate": RFM_Analysis.Latest_Date,
                    "CustomerID": customer_id
                }])
            else:
                cust_df = RFM_Analysis.RFMScores[
                    RFM_Analysis.RFMScores["CustomerID"] == customer_id
                ]
                total_qty = cust_df["Quantity"].sum() + quantity
                df_in = pd.DataFrame([{
                    "InvoiceNo": invoice_no,
                    "Quantity": total_qty,
                    "UnitPrice": unit_price,
                    "InvoiceDate": RFM_Analysis.Latest_Date,
                    "CustomerID": customer_id
                }])

            analyzer = RFM_Analysis.RFMAnalyzer(df_in)
            analyzer.preprocess_data()
            analyzer.calculate_RFMScores()

            recency  = analyzer.RFMScores["Recency"].iloc[0]
            frequency = analyzer.RFMScores["Frequency"].iloc[0]
            monetary  = analyzer.RFMScores["Monetary"].iloc[0]
            return recency, frequency, monetary

        except Exception as e:
            st.error(f"Error during RFM analysis: {e}")
            return None, None, None

    def predict_rfm_values(self, recency, frequency, monetary):
        """
        Uses a RandomForest to predict refined R, F, M values.
        """
        feature_cols = ["Recency", "Frequency", "Monetary"]
        target_cols  = ["R", "F", "M"]

        trainer = RandomForest_Model.RandomForestModelTrainer(
            KMeans_Model.RFMScores, feature_cols, target_cols
        )
        trainer.prepare_data()
        trainer.scale_data()
        trainer.train_model()
        trainer.save_model()

        X_new = np.array([recency, frequency, monetary]).reshape(1, -1)
        # find and load the latest .pkl
        model_file = self.find_pkl_files(".")
        model = self.load_model_from_file(model_file)
        X_scaled = trainer.scaler.transform(X_new)
        pred = self.make_rf_prediction(model, X_scaled)
        if pred is not None:
            return pred[0][0], pred[0][1], pred[0][2]
        return None, None, None

    def predict_clusters(self, r, f, m):
        """
        Uses a RandomForest to predict the KMeans cluster name.
        """
        feature_cols = ["R", "F", "M"]
        target_cols  = ["Cluster"]

        trainer = RandomForest_Model.RandomForestModelTrainer(
            KMeans_Model.RFMScores, feature_cols, target_cols
        )
        trainer.prepare_data()
        trainer.scale_data()
        trainer.train_model()
        trainer.save_model()

        X_new = np.array([r, f, m]).reshape(1, -1)
        model_file = self.find_pkl_files(".")
        model = self.load_model_from_file(model_file)
        X_scaled = trainer.scaler.transform(X_new)
        pred = self.make_rf_prediction(model, X_scaled)
        if pred is not None:
            cluster_id = int(pred[0])
            df_clusters = KMeans_Model.RFMScores[
                KMeans_Model.RFMScores["Cluster"].astype(int) == cluster_id
            ]
            if not df_clusters.empty:
                return df_clusters["Cluster_Name"].iloc[0]
        return "Unknown"

    def find_pkl_files(self, directory):
        for fn in os.listdir(directory):
            if fn.endswith(".pkl"):
                return fn
        return None

    def load_model_from_file(self, filename):
        try:
            with open(filename, "rb") as f:
                return pickle.load(f)
        except Exception as e:
            st.error(f"Error loading model {filename}: {e}")
            return None

    def make_rf_prediction(self, model, X):
        try:
            return model.predict(X)
        except Exception as e:
            st.error(f"Prediction error: {e}")
            return None

def main():
    st.title("RFM Analysis with RandomForest & KMeans Models")

    st.header("Enter Transaction Data")
    with st.form("transaction_form"):
        col1, col2 = st.columns(2)
        with col1:
            stock_code  = st.text_input("Stock Code")
            quantity    = st.number_input("Quantity", min_value=1, value=1, step=1)
            customer_id = st.number_input("Customer ID", min_value=1, value=1, step=1)
        with col2:
            description = st.text_input("Description")
            unit_price  = st.number_input("Unit Price", min_value=0.0, value=0.0, step=0.1)
            country     = st.text_input("Country")
        submitted = st.form_submit_button("Submit Transaction")

    if submitted:
        invoice_no = np.random.randint(100000, 1000000)
        df_updated = add_entry_to_csv(
            invoice_no, stock_code, description,
            quantity, unit_price, customer_id, country
        )

        st.success("Transaction added successfully!")
        st.subheader("Updated CSV Data")
        st.dataframe(df_updated)

        st.header("CSV Data Visualizations")
        show_csv_visualizations(df_updated)

        st.header("RFM Analysis & Predictions")
        proc = RFMProcessor()
        rec, freq, mon = proc.perform_rfm_analysis(
            invoice_no, quantity, unit_price, customer_id
        )
        if rec is not None:
            st.write(f"**Calculated RFM:** R={rec}, F={freq}, M={mon}")
            pr, pf, pm = proc.predict_rfm_values(rec, freq, mon)
            if pr is not None:
                st.write(f"**Predicted RFM:** R={pr}, F={pf}, M={pm}")
                cluster_name = proc.predict_clusters(pr, pf, pm)
                st.write(f"**Predicted Cluster:** {cluster_name}")
                show_rfm_visualizations(rec, freq, mon, pr, pf, pm, cluster_name)
            else:
                st.error("Failed to predict RFM values.")
        else:
            st.error("RFM analysis failed.")

    # Always show the full dashboard if the ZIP exists
    if os.path.exists(CSV_FILE):
        st.header("CSV Data Dashboard")
        df_dash = pd.read_csv(CSV_FILE, compression="zip")
        st.dataframe(df_dash)

if __name__ == "__main__":
    main()
