'''

import json
from kafka import KafkaProducer
from kafka.errors import KafkaError

producer = KafkaProducer(
    bootstrap_servers = "localhost:19092",
    value_serializer=lambda m: json.dumps(m).encode('ascii')
)

topic = "transactions_topic"

def on_success(metadata):
  print(f"Message produced to topic '{metadata.topic}' at offset {metadata.offset}")

def on_error(e):
  print(f"Error sending message: {e}")

# Produce asynchronously with callbacks
for i in range(1, 11):
  msg = { "id": i, "content": "Some value"}
  future = producer.send(topic, msg)
  future.add_callback(on_success)
  future.add_errback(on_error)

producer.flush()
producer.close()
'''
import json
import csv
from kafka import KafkaProducer
from kafka.errors import KafkaError

producer = KafkaProducer(
    bootstrap_servers="localhost:19092",
    value_serializer=lambda m: json.dumps(m).encode('utf-8')  # Ensure proper encoding
)

topic = "transactions"

def on_success(metadata):
    print(f"Message produced to topic '{metadata.topic}' at offset {metadata.offset}")

def on_error(e):
    print(f"Error sending message: {e}")

# Function to read CSV and produce messages
def produce_from_csv(file_path, num_records):
    counter = 0
    with open(file_path, mode='r', encoding='utf-8') as file:
        reader = csv.DictReader(file)
        for row in reader:
            if (counter < 300 and counter > 200):
                break
            counter = counter+1
            # Map CSV columns to JSON keys
            msg = {
                "step": row["type"],
                "amount": float(row["amount"]),
                "nameOrig": row["nameOrig"],
                "oldbalanceOrig": float(row["oldbalanceOrg"]),
                "newbalanceOrig": float(row["newbalanceOrig"]),
                "nameDest": row["nameDest"],
                "oldbalanceDest": float(row["oldbalanceDest"]),
                "newbalanceDest": float(row["newbalanceDest"]),
                "isFraud": int(row["isFraud"]),
                "isFlaggedFraud": int(row["isFlaggedFraud"])
            }
            future = producer.send(topic, msg)
            future.add_callback(on_success)
            future.add_errback(on_error)

# Specify the path to your CSV file and the number of records to produce
file_path = './transactions.csv'  # Update this path
num_records = 100  # Number of records to read and produce

produce_from_csv(file_path, num_records)

producer.flush()
producer.close()
