import pandas as pd
import datetime as dt

Latest_Date = dt.datetime(2011, 12, 11)

class RFMAnalyzer:
    def __init__(self, RFMScores):
        self.Latest_Date = Latest_Date
        self.quantiles = None
        self.RFMScores = RFMScores

    def preprocess_data(self):
        # Replace empty strings with NaN and drop any rows with missing values.
        self.RFMScores.replace(r'^\s*$', pd.NA, regex=True, inplace=True)
        self.RFMScores.dropna(inplace=True)
        
        self.RFMScores['TotalAmount'] = self.RFMScores['Quantity'] * self.RFMScores['UnitPrice']
        self.RFMScores['InvoiceDate'] = pd.to_datetime(self.RFMScores['InvoiceDate'], dayfirst=True)

    def calculate_RFMScores(self):
        self.RFMScores = self.RFMScores.groupby('CustomerID').agg({
            'CustomerID': lambda x: x.unique()[0],
            'InvoiceDate': lambda x: (self.Latest_Date - x.max()).days,
            'InvoiceNo': lambda x: len(x),
            'TotalAmount': lambda x: x.sum()
        })
        self.RFMScores['InvoiceDate'] = self.RFMScores['InvoiceDate'].astype(int)
        self.RFMScores.rename(columns={'InvoiceDate': 'Recency',
                                       'InvoiceNo': 'Frequency',
                                       'TotalAmount': 'Monetary'}, inplace=True)

    def calculate_quantiles(self):
        self.quantiles = self.RFMScores.quantile(q=[0.25, 0.5, 0.75]).to_dict()

    def RScoring(self, x, p):
        if x <= self.quantiles[p][0.25]:
            return 4
        elif x <= self.quantiles[p][0.50]:
            return 3
        elif x <= self.quantiles[p][0.75]:
            return 2
        else:
            return 1

    def FnMScoring(self, x, p):
        if x <= self.quantiles[p][0.25]:
            return 1
        elif x <= self.quantiles[p][0.50]:
            return 2
        elif x <= self.quantiles[p][0.75]:
            return 3
        else:
            return 4

    def calculate_RFM_scores(self):
        self.RFMScores['R'] = self.RFMScores['Recency'].apply(self.RScoring, args=('Recency',))
        self.RFMScores['F'] = self.RFMScores['Frequency'].apply(self.FnMScoring, args=('Frequency',))
        self.RFMScores['M'] = self.RFMScores['Monetary'].apply(self.FnMScoring, args=('Monetary',))
        
    def calculate_RFM_total(self):
        self.RFMScores['RFMScore'] = self.RFMScores['R'] + self.RFMScores['F'] + self.RFMScores['M']

    def assign_loyalty_levels(self):
        Loyalty_Level = ['Bronze', 'Silver', 'Gold', 'Platinum']
        Score_cuts = pd.qcut(self.RFMScores.RFMScore, q=4, labels=Loyalty_Level)
        self.RFMScores['RFM_Loyalty_Level'] = Score_cuts.values

    def save_results(self, output_path):
        self.RFMScores.to_csv(output_path, index=False)


# Example usage:
RFMScores = pd.read_csv('Online_Retail_Train.zip', encoding='unicode_escape')

analyzer = RFMAnalyzer(RFMScores)
analyzer.preprocess_data()
analyzer.calculate_RFMScores()
analyzer.calculate_quantiles()
analyzer.calculate_RFM_scores()
analyzer.calculate_RFM_total()
analyzer.assign_loyalty_levels()
analyzer.save_results('RFMScores.csv')
