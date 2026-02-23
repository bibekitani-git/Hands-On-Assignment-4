import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from pyod.models.auto_encoder import AutoEncoder
from sklearn.metrics import classification_report

def run_fraud_detection():
    # 1. Load Dataset
    try:
        df = pd.read_csv('creditcard.csv')
    except FileNotFoundError:
        print("Error: creditcard.csv not found in current directory.")
        return

    # 2. Preprocessing
    scaler = StandardScaler()
    # Scikit-learn expects 2D arrays, so we use double brackets
    df['Amount'] = scaler.fit_transform(df[['Amount']])
    
    X = df.drop(['Class'], axis=1) 
    y = df['Class']

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # 3. Initialize AutoEncoder with your specific parameters
    # Note the change to 'hidden_neuron_list' and 'epoch_num'
    model = AutoEncoder(
        hidden_neuron_list=[30, 15, 15, 30], 
        epoch_num=20, 
        contamination=0.01, 
        preprocessing=True,   # Set to True since it's a default in your version
        verbose=1,
        random_state=42
    )

    print("\n--- Training AutoEncoder Model ---")
    model.fit(X_train)

    # 4. Predict
    y_test_pred = model.predict(X_test) 

    # 5. Evaluation
    print("\n--- Model Evaluation ---")
    print(classification_report(y_test, y_test_pred))

if __name__ == "__main__":
    run_fraud_detection()