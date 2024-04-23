import os
import pickle
import pandas as pd
import RFM_Analysis
import KMeans_Model
import RandomForest_Model
import warnings
import numpy as np

warnings.filterwarnings("ignore", category=UserWarning)

class RFMProcessor:
    def __init__(self):
        self.temp = 0

    def perform_rfm_analysis(self):
        try:
            invoiceno = int(input('Enter Invoice No.: '))
            quantity = int(input('Enter Quantity: '))
            unitprice = float(input('Enter Unit Price: '))
            customerid = int(input('Enter Customer ID: '))
            
            existing_customer_ids = RFM_Analysis.RFMScores['CustomerID'].tolist()
            if customerid not in existing_customer_ids:
                input_data =  [{'InvoiceNo': invoiceno,'Quantity': quantity,'UnitPrice': unitprice, 'InvoiceDate': RFM_Analysis.Latest_Date, 'CustomerID': customerid}]
                input_data = pd.DataFrame(input_data)
            else:
                customer_data = RFM_Analysis.RFMScores[RFM_Analysis.RFMScores['CustomerID'] == customerid]
                quantity += customer_data['Quantity'].sum()
                invoiceno = customer_data['InvoiceNo'].sum()
                input_data = [{'InvoiceNo': invoiceno,'Quantity': quantity,'UnitPrice': unitprice, 'InvoiceDate': RFM_Analysis.Latest_Date, 'CustomerID': customerid}]
                input_data = pd.DataFrame(input_data)
                self.temp += 1
            analyzer = RFM_Analysis.RFMAnalyzer(input_data)
            print("Today's Date: ", analyzer.Latest_Date.strftime('%Y-%m-%d'))
            analyzer.preprocess_data()
            analyzer.calculate_RFMScores(self.temp)
            
            return analyzer.RFMScores['Recency'].values[0], analyzer.RFMScores['Frequency'].values[0], analyzer.RFMScores['Monetary'].values[0]
            
        except KeyboardInterrupt:
            print("\nUser interrupted input.")
            return None
        except EOFError:
            print("\nInput stream closed.")
            return None

    def predict_rfm_values(self, recency, frequency, monetary):
        
        feature_columns = ['Recency', 'Frequency', 'Monetary']
        target_columns = ['R', 'F', 'M']
        
        trainer = RandomForest_Model.RandomForestModelTrainer(KMeans_Model.RFMScores, feature_columns, target_columns)
        trainer.prepare_data()
        trainer.scale_data()
        trainer.train_model()
        trainer.save_model()

        input_data = [recency, frequency, monetary]
        input_data = np.array(input_data).reshape(1, -1)
        
        directory = '.'
        model_filename = self.find_pkl_files(directory)
        loaded_model = self.load_model_from_file(model_filename)
        scaled_new_data = trainer.scaler.transform(input_data)
        prediction = self.make_rf_prediction(loaded_model, scaled_new_data)
        return prediction[0][0], prediction[0][1], prediction[0][2]

    def predict_clusters(self, r, f, m):
        feature_columns = ['R', 'F', 'M']
        target_columns = ['Cluster']
        
        trainer = RandomForest_Model.RandomForestModelTrainer(KMeans_Model.RFMScores, feature_columns, target_columns)
        trainer.prepare_data()
        trainer.scale_data()
        trainer.train_model()
        trainer.save_model()
        
        input_data = [r, f, m]
        input_data = np.array(input_data).reshape(1, -1)
        
        directory = '.'
        model_filename = self.find_pkl_files(directory)
        loaded_model = self.load_model_from_file(model_filename)
        scaled_new_data = trainer.scaler.transform(input_data)
        if loaded_model is not None:
            prediction = self.make_rf_prediction(loaded_model, scaled_new_data)
            if prediction is not None:
                filtered_df = KMeans_Model.RFMScores[KMeans_Model.RFMScores['Cluster'].astype(int) == prediction[0]]
                if not filtered_df.empty:
                    print("Prediction: ", filtered_df['Cluster_Name'].iloc[0])
            else:
                print("Error: Unable to make prediction.")
                
    def find_pkl_files(self, directory):
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
            print(f"Error: File {filename} not found.")
            return None
        except Exception as e:
            print(f"Error loading model from {filename}: {e}")
            return None
    
    def make_rf_prediction(self, model, scaled_new_data):
        try:
            prediction = model.predict(scaled_new_data)
            return prediction
        except ValueError:
            print("Error: Invalid input. Please enter a valid number.")
            return None
        except KeyboardInterrupt:
            print("\nUser interrupted input.")
            return None
        except EOFError:
            print("\nInput stream closed.")
            return None

rfm_processor = RFMProcessor()
recency, frequency, monetary = rfm_processor.perform_rfm_analysis() 
r, f, m = rfm_processor.predict_rfm_values(recency, frequency, monetary)
rfm_processor.predict_clusters(r, f, m)