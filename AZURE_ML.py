# ---
# jupyter:
#   jupytext:
#     formats: py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.17.0
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# %%
from azure.storage.blob import BlobServiceClient
from io import StringIO
import pandas as pd

BLOB_URL = "https://iotstorage001.blob.core.windows.net/equipment-data/equipment_data.csv"
ACCOUNT_KEY = "a3Dom8whPtzCsG+mL3ydjIK/********************************************+/WwSoBSn4ybs+AStWhPhrw=="

# %%
account_name = BLOB_URL.split("//")[1].split(".")[0]
account_url = f"https://{account_name}.blob.core.windows.net"
path_after_net = BLOB_URL.split(".net/")[1]
container_name, blob_name = path_after_net.split("/", 1)
blob_service = BlobServiceClient(account_url=account_url, credential=ACCOUNT_KEY)
blob_client = blob_service.get_blob_client(container=container_name, blob=blob_name)
download_stream = blob_client.download_blob()
csv_bytes = download_stream.readall()
csv_str = csv_bytes.decode('utf-8')


# %%
df = pd.read_csv(StringIO(csv_str))
print("Data loaded successfully. Here's a preview:")
display(df.head())


# %%
print("Data info:")
df.info()

# %%
print("\nMissing values per column:")
print(df.isnull().sum())

# %%
print("\nSummary statistics:")
print(df.describe())

# %%
# !pip install scikit-learn --quiet
# !pip install scikit-learn --quiet --disable-pip-version-check


# %%
print("Columns in the dataset:")
print(df.columns.tolist())


# %%
df_clean = df.dropna()

# %%
df_clean['device_id_enc'] = pd.factorize(df_clean['device_id'])[0]
df_clean['location_enc'] = pd.factorize(df_clean['location'])[0]

# %%
for col in ['temperature_c', 'vibration_level', 'pressure_psi']:
    mean = df_clean[col].mean()
    std = df_clean[col].std()
    df_clean[col] = (df_clean[col] - mean) / std

print("Preprocessing done. Here's a preview:")
display(df_clean.head())

# %%
X = df_clean[['device_id_enc', 'location_enc', 'temperature_c', 'vibration_level', 'pressure_psi']]
y = df_clean['status']


# %%
y_enc = pd.factorize(y)[0]


# %%
import numpy as np

# Shuffle the dataset
df_clean = df_clean.sample(frac=1, random_state=42).reset_index(drop=True)

# Calculate split index
split_index = int(0.8 * len(df_clean))

# Split
train_df = df_clean.iloc[:split_index]
test_df = df_clean.iloc[split_index:]

# Features and labels
X_train = train_df[['device_id_enc', 'location_enc', 'temperature_c', 'vibration_level', 'pressure_psi']].values
y_train = pd.factorize(train_df['status'])[0]

X_test = test_df[['device_id_enc', 'location_enc', 'temperature_c', 'vibration_level', 'pressure_psi']].values
y_test = pd.factorize(test_df['status'])[0]


# %%
# !pip install azureml-core azure-ai-ml scikit-learn --quiet


# %%
from azureml.train.automl import AutoMLConfig
from azureml.core import Workspace, Experiment

ws = Workspace.from_config()

automl_config = AutoMLConfig(
    task='classification',
    training_data=train_df,
    label_column_name='status',
    n_cross_validations=3,
    primary_metric='accuracy',
    max_concurrent_iterations=4,
    iterations=10,
)

experiment = Experiment(ws, 'equipment-status-classification')
run = experiment.submit(automl_config, show_output=True)


# %%
df_clean['device_id_enc'] = pd.factorize(df_clean['device_id'])[0]
df_clean['location_enc'] = pd.factorize(df_clean['location'])[0]
df_clean['status_enc'] = pd.factorize(df_clean['status'])[0] 

# %%
df_clean['status_enc'] = pd.factorize(df_clean['status'])[0]
def simple_classifier(row):
    if row['vibration_level'] > 5.0 or row['temperature_c'] > 80:
        return 1  # Predict FAULTY
    return 0  # Predict WORKING

df_clean['predicted_status'] = df_clean.apply(simple_classifier, axis=1)



# %%
print(df_clean.groupby('status')[['temperature_c', 'vibration_level', 'pressure_psi']].mean())


# %%
def improved_classifier(row):
    if (
        row['temperature_c'] > 75 and
        row['vibration_level'] > 4.5
    ) or (
        row['pressure_psi'] < 25
    ):
        return 1  # Predict FAULTY
    return 0  # Predict WORKING

df_clean['predicted_status'] = df_clean.apply(improved_classifier, axis=1)

# Accuracy
correct = (df_clean['predicted_status'] == df_clean['status_enc']).sum()
total = len(df_clean)
accuracy = correct / total
print(f"\n Accuracy: {accuracy:.2f}")



# %%
def classification_report_basic(y_true, y_pred):
    labels = sorted(set(y_true) | set(y_pred))
    print("\n Classification Report :")
    for label in labels:
        tp = sum((y_true == label) & (y_pred == label))
        fp = sum((y_true != label) & (y_pred == label))
        fn = sum((y_true == label) & (y_pred != label))
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
        print(f"Label {label} → Precision: {precision:.2f}, Recall: {recall:.2f}, F1-score: {f1:.2f}")

# Run the report
classification_report_basic(df_clean['status_enc'], df_clean['predicted_status'])

# %%
# Upload to Azure Blob Storage
blob_service_client = BlobServiceClient.from_connection_string("DefaultEndpointsProtocol=https;AccountName=iotstorage001;AccountKey=a3Dom8whPtzCsG+mL3ydjIK/xk7XRLjgPRLzGHb0hBT0pwNtHwUvs4ungaQvyQD+/WwSoBSn4ybs+AStWhPhrw==;EndpointSuffix=core.windows.net")
container_client = blob_service_client.get_container_client("equipment-data")
blob_client = container_client.get_blob_client("equipment_predictions.csv")

blob_client.upload_blob(output_csv.getvalue(), overwrite=True)
print("Predictions saved and uploaded to Blob as 'equipment_predictions.csv'")
