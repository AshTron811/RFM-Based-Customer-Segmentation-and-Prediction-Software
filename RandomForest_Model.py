from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import pickle

class RandomForestModelTrainer:
    def __init__(self, RFMScores, feature_columns, target_columns, model_filename='rf.pkl', test_size=0.2, random_state=42):
        self.feature_columns = feature_columns
        self.target_columns = target_columns
        self.model_filename = model_filename
        self.test_size = test_size
        self.random_state = random_state
        self.RFMScores = RFMScores
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.scaler = None
        self.rf_classifier = None

    def prepare_data(self):
        X = self.RFMScores[self.feature_columns]
        y = self.RFMScores[self.target_columns]
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(X, y, test_size=self.test_size, random_state=self.random_state)

    def scale_data(self):
        self.scaler = StandardScaler()
        self.X_train_scaled = self.scaler.fit_transform(self.X_train)
        self.X_test_scaled = self.scaler.transform(self.X_test)

    def train_model(self):
        self.rf_classifier = RandomForestClassifier(n_estimators=100, random_state=self.random_state)
        self.rf_classifier.fit(self.X_train_scaled, self.y_train)

    def save_model(self):
        with open(self.model_filename, 'wb') as file:
            pickle.dump(self.rf_classifier, file)