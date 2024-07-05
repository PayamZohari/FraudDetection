from airflow import DAG
from airflow.operators.python_operator import PythonOperator
from airflow.operators.email_operator import EmailOperator
from datetime import datetime
from clickhouse_driver import Client
import pandas as pd
from joblib import load
from sklearn.preprocessing import LabelEncoder
import smtplib

default_args = {
    'owner': 'admin',
    'start_date': datetime(2024, 6, 25),
    'retries': 1,
}

# Load your saved RandomForest model
model_path = '/home/payam/airflow/dags/fraud_detector.joblib'
# Load the model from the file
random_forest_model = load(model_path)


def send_email(content):
    """
    Function to send an email using the SMTP protocol.

    Parameters:
    content (str): The body content of the email.
    emailSubject (str): The subject of the email.
    senderEmailAddress (str): The sender's email address. Default is fetched from environment variable SENDER_EMAIL.
    senderPassword (str): The sender's email password. Default is fetched from environment variable SENDER_PASS.
    recieverEmailAddress (str): The receiver's email address. Default is fetched from environment variable RECIVER_EMAIL.
    """

    # Initialize the SMTP connection
    mail = smtplib.SMTP('smtp.gmail.com', 587)
    mail.ehlo()  # Identify ourselves to the SMTP server
    mail.starttls()  # Secure the SMTP connection
    mail.login("payamzohari80@gmail.com", 'cmbl wqdv bawl btvd')  # Log in to the SMTP server

    # Construct the email header and body
    header = 'To:' + "payamzohari80@gmail.com" + '\n' + 'From:' + "payamzohari80@gmail.com" + '\n' + 'subject:' + "fraud report" + '\n\n'
    body = header + content

    # Send the email
    mail.sendmail("payamzohari80@gmail.com", "payamzohari80@gmail.com", body)
    mail.close()  # Close the SMTP connection

# Function to preprocess the data
def preprocess_data(df):
    # Select relevant numeric columns
    numeric_columns = ['step', 'amount', 'oldbalanceOrig', 'newbalanceOrig', 
                       'oldbalanceDest', 'newbalanceDest']
                       
    # Encode categorical columns if needed
    label_encoder = LabelEncoder()
    df['step'] = label_encoder.fit_transform(df['step'])  # Example encoding for 'step'
    
    # Handle any other preprocessing steps, such as filling missing values, scaling data, etc.
    # Example: Fill missing values with mean
    df = df.fillna(df.mean())
    
    # Include additional features based on their availability in your dataset
    if 'type_CASH_IN' not in df.columns:
        df['type_CASH_IN'] = False  # Example: if not present, assume False
    
    if 'type_CASH_OUT' not in df.columns:
        df['type_CASH_OUT'] = False
    
    if 'type_DEBIT' not in df.columns:
        df['type_DEBIT'] = False
    
    if 'type_PAYMENT' not in df.columns:
        df['type_PAYMENT'] = False
    
    if 'type_TRANSFER' not in df.columns:
        df['type_TRANSFER'] = False
    
    if 'balance_diff' not in df.columns:
        df['balance_diff'] = 0.0  # Example: if not present, assume 0.0
    
    if 'balance_change' not in df.columns:
        df['balance_change'] = 0.0
    
    # Ensure columns match expected features
    df = df[numeric_columns + ['type_CASH_IN', 'type_CASH_OUT', 'type_DEBIT', 'type_PAYMENT', 'type_TRANSFER', 'balance_diff', 'balance_change']]
    
    
    return df
    
   


def read_from_clickhouse_and_apply_model():
    client = Client(host='localhost', port=9000, user='default', password='', database='datawarehouse')
    query = 'SELECT * FROM transactions_dest'
    result = client.execute(query)
    df = pd.DataFrame(result, columns=['step', 'amount', 'nameOrig' , 'oldbalanceOrig', 'newbalanceOrig', 'nameDest', 'oldbalanceDest', 'newbalanceDest', 'isFraud', 'isFlaggedFraud']) 
    
    df.drop(['isFraud', 'isFlaggedFraud', 'nameOrig', 'nameDest'], axis=1, inplace=True) 
    
    # Preprocess data
    df = preprocess_data(df)
    
    predictions = []
    for index, row in df.iterrows():
        features = row[['step', 'amount', 'oldbalanceOrig', 'newbalanceOrig', 'oldbalanceDest', 'newbalanceDest', 
                        'type_CASH_IN', 'type_CASH_OUT', 'type_DEBIT', 'type_PAYMENT', 'type_TRANSFER', 
                        'balance_diff', 'balance_change']].values.reshape(1, -1)
        prediction = random_forest_model.predict(features)
        print(f"Prediction for record {index}: {prediction}")
        if prediction > 0:
            predictions.append(prediction)
    
    # Send email if there are predictions greater than 0
    if not predictions:
        send_email(str(predictions))
        
        
with DAG('my_clickhouse_dag',
         default_args=default_args,
         schedule_interval='@daily',
         catchup=False) as dag:

    read_from_clickhouse_task = PythonOperator(
        task_id='read_from_clickhouse_and_apply_model',
        python_callable=read_from_clickhouse_and_apply_model,
    )

read_from_clickhouse_task
