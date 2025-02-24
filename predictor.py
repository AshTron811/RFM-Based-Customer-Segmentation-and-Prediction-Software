import streamlit as st
import pandas as pd
import datetime
import os
import pickle
import numpy as np
import warnings

warnings.filterwarnings("ignore", category=UserWarning)

# Import your external modules (ensure these are in your Python path)
import RFM_Analysis
import KMeans_Model
import RandomForest_Model

# Define the CSV file path (an already existing CSV file)
CSV_FILE = "Online_Retail_Train.zip"

def add_entry_to_csv(invoice_no, stock_code, description, quantity, unit_price, customer_id, country):
    """
    Appends a new transaction record to the existing Online_Retail_Train.csv file.
    The record includes additional fields: StockCode, Description, and Country.
    InvoiceDate now includes both date and time (only hours and minutes).
    """
    current_datetime = datetime.datetime.now().strftime("%d-%m-%Y %H:%M")
    data = {
        "InvoiceNo": invoice_no,
        "StockCode": stock_code,
        "Description": description,
        "Quantity": quantity,
        "InvoiceDate": current_datetime,
        "UnitPrice": unit_price,
        "CustomerID": customer_id,
        "Country": country
    }
    df_new = pd.DataFrame([data])
    
    if os.path.exists(CSV_FILE):
        df_existing = pd.read_csv(CSV_FILE)
        df_updated = pd.concat([df_existing, df_new], ignore_index=True)
    else:
        df_updated = df_new
        
    df_updated.to_csv(CSV_FILE, index=False)
    return df_updated

def show_csv_visualizations(df):
    """
    Displays visualizations for the CSV data.
    """
    st.subheader("Transactions by Date")
    df["InvoiceDate"] = pd.to_datetime(df["InvoiceDate"], errors='coerce')
    transactions_by_date = df.groupby(df["InvoiceDate"].dt.date).size()
    st.line_chart(transactions_by_date)
    
    st.subheader("Quantity Distribution")
    if not df["Quantity"].empty:
        quantity_counts = df["Quantity"].value_counts().sort_index()
        st.bar_chart(quantity_counts)
    
    st.subheader("Unit Price Distribution")
    if not df["UnitPrice"].empty:
        unitprice_counts = df["UnitPrice"].value_counts().sort_index()
        st.bar_chart(unitprice_counts)

def show_rfm_visualizations(recency, frequency, monetary, r, f, m, cluster_name):
    """
    Creates visualizations comparing calculated vs. predicted RFM values and displays the predicted cluster.
    """
    df_compare = pd.DataFrame({
        "Actual": [recency, frequency, monetary],
        "Predicted": [r, f, m]
    }, index=["Recency", "Frequency", "Monetary"])
    
    st.subheader("Comparison of Calculated and Predicted RFM Values")
    st.bar_chart(df_compare)
    
    st.subheader("Predicted Customer Cluster")
    st.write(f"The customer is predicted to belong to cluster: **{cluster_name}**")

# Adapted RFMProcessor that uses passed parameters (instead of console input)
class RFMProcessor:
    def perform_rfm_analysis(self, invoice_no, quantity, unit_price, customer_id):
        """
        Computes the RFM scores based on transaction data.
        If the customer already exists, aggregates the Quantity but retains the new InvoiceNo as is.
        """
        try:
            existing_customer_ids = RFM_Analysis.RFMScores['CustomerID'].tolist()
            if customer_id not in existing_customer_ids:
                input_data = [{
                    'InvoiceNo': invoice_no,
                    'Quantity': quantity,
                    'UnitPrice': unit_price,
                    'InvoiceDate': RFM_Analysis.Latest_Date,
                    'CustomerID': customer_id
                }]
                input_data = pd.DataFrame(input_data)
            else:
                customer_data = RFM_Analysis.RFMScores[
                    RFM_Analysis.RFMScores['CustomerID'] == customer_id
                ]
                total_quantity = customer_data['Quantity'].sum() + quantity
                input_data = [{
                    'InvoiceNo': invoice_no,  # Retain new invoice number as is.
                    'Quantity': total_quantity,
                    'UnitPrice': unit_price,
                    'InvoiceDate': RFM_Analysis.Latest_Date,
                    'CustomerID': customer_id
                }]
                input_data = pd.DataFrame(input_data)

            analyzer = RFM_Analysis.RFMAnalyzer(input_data)
            analyzer.preprocess_data()
            analyzer.calculate_RFMScores()
            
            recency = analyzer.RFMScores['Recency'].values[0]
            frequency = analyzer.RFMScores['Frequency'].values[0]
            monetary = analyzer.RFMScores['Monetary'].values[0]
            return recency, frequency, monetary
        except Exception as e:
            st.error(f"Error during RFM analysis: {e}")
            return None, None, None

    def predict_rfm_values(self, recency, frequency, monetary):
        """
        Uses a Random Forest model to predict refined RFM values.
        """
        feature_columns = ['Recency', 'Frequency', 'Monetary']
        target_columns = ['R', 'F', 'M']
        
        trainer = RandomForest_Model.RandomForestModelTrainer(KMeans_Model.RFMScores, feature_columns, target_columns)
        trainer.prepare_data()
        trainer.scale_data()
        trainer.train_model()
        trainer.save_model()

        input_data = np.array([recency, frequency, monetary]).reshape(1, -1)
        
        directory = '.'
        model_filename = self.find_pkl_files(directory)
        loaded_model = self.load_model_from_file(model_filename)
        scaled_new_data = trainer.scaler.transform(input_data)
        prediction = self.make_rf_prediction(loaded_model, scaled_new_data)
        if prediction is not None:
            return prediction[0][0], prediction[0][1], prediction[0][2]
        return None, None, None

    def predict_clusters(self, r, f, m):
        """
        Uses a Random Forest model to predict the customer cluster.
        """
        feature_columns = ['R', 'F', 'M']
        target_columns = ['Cluster']
        
        trainer = RandomForest_Model.RandomForestModelTrainer(KMeans_Model.RFMScores, feature_columns, target_columns)
        trainer.prepare_data()
        trainer.scale_data()
        trainer.train_model()
        trainer.save_model()
        
        input_data = np.array([r, f, m]).reshape(1, -1)
        
        directory = '.'
        model_filename = self.find_pkl_files(directory)
        loaded_model = self.load_model_from_file(model_filename)
        scaled_new_data = trainer.scaler.transform(input_data)
        if loaded_model is not None:
            prediction = self.make_rf_prediction(loaded_model, scaled_new_data)
            if prediction is not None:
                filtered_df = KMeans_Model.RFMScores[KMeans_Model.RFMScores['Cluster'].astype(int) == prediction[0]]
                if not filtered_df.empty:
                    return filtered_df['Cluster_Name'].iloc[0]
        return None

    def find_pkl_files(self, directory):
        pkl_file = None
        for filename in os.listdir(directory):
            if filename.endswith('.pkl'):
                pkl_file = filename
        return pkl_file
    
    def load_model_from_file(self, filename):
        try:
            if isinstance(filename, str):
                with open(filename, 'rb') as file:
                    loaded_model = pickle.load(file)
                return loaded_model
            else:
                raise ValueError("Invalid filename format.")
        except FileNotFoundError:
            st.error(f"Error: File {filename} not found.")
            return None
        except Exception as e:
            st.error(f"Error loading model from {filename}: {e}")
            return None
    
    def make_rf_prediction(self, model, scaled_new_data):
        try:
            prediction = model.predict(scaled_new_data)
            return prediction
        except Exception as e:
            st.error(f"Prediction error: {e}")
            return None

def main():
    st.title("RFM Analysis with RandomForest & KMeans Models - CSV & Visualizations")
    
    st.header("Enter Transaction Data")
    with st.form("transaction_form"):
        col1, col2 = st.columns(2)
        # Left column: three fields
        with col1:
            stock_code = st.text_input("Stock Code")
            quantity = st.number_input("Quantity", min_value=1, value=1, step=1)
            customer_id = st.number_input("Customer ID", min_value=1, value=1, step=1)
        # Right column: three fields
        with col2:
            description = st.text_input("Description")
            unit_price = st.number_input("Unit Price", min_value=0.0, value=0.0, step=0.1)
            country = st.text_input("Country")
        submitted = st.form_submit_button("Submit Transaction")
    
    if submitted:
        # Generate a random 6-digit invoice number (range [100000, 1000000))
        invoice_no = np.random.randint(100000, 1000000)
        
        # Append transaction to CSV and show updated details
        df_updated = add_entry_to_csv(invoice_no, stock_code, description, quantity, unit_price, customer_id, country)
        st.success("Transaction added successfully!")
        st.write("Updated CSV Data:")
        st.dataframe(df_updated)
        
        st.header("CSV Data Visualizations")
        show_csv_visualizations(df_updated)
        
        st.header("RFM Analysis & Model Predictions")
        rfm_processor = RFMProcessor()
        recency, frequency, monetary = rfm_processor.perform_rfm_analysis(invoice_no, quantity, unit_price, customer_id)
        if recency is not None:
            st.write(f"**Calculated RFM Scores**: Recency = {recency}, Frequency = {frequency}, Monetary = {monetary}")
            r, f, m = rfm_processor.predict_rfm_values(recency, frequency, monetary)
            if r is not None:
                st.write(f"**Predicted RFM Values**: R = {r}, F = {f}, M = {m}")
                cluster_name = rfm_processor.predict_clusters(r, f, m)
                if cluster_name:
                    st.write(f"**Predicted Cluster:** {cluster_name}")
                else:
                    st.error("Unable to predict cluster.")
                show_rfm_visualizations(recency, frequency, monetary, r, f, m, cluster_name if cluster_name else "Unknown")
            else:
                st.error("RFM value prediction failed.")
        else:
            st.error("RFM analysis failed.")
    
    # Always show the CSV Dashboard below the "Enter Transaction Data" section.
    if os.path.exists(CSV_FILE):
        st.header("CSV Data Dashboard")
        df_dashboard = pd.read_csv(CSV_FILE)
        st.dataframe(df_dashboard)

if __name__ == "__main__":
    main()
