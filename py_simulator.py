import ssl
import time
import json
import random
from paho.mqtt import client as mqtt_client

# AWS IoT Core endpoint (replace with your own)
AWS_ENDPOINT = "*************************************************"
PORT = 8883
TOPIC = "factory/equipment"
CLIENT_ID = "equipment-sensor-1"

# Certificate paths
CA_PATH = "**************.pem"
CERT_PATH = "certificate.pem.crt"
KEY_PATH = "private.pem.key"

def on_connect(client, userdata, flags, rc):
    if rc == 0:
        print("Connected to AWS IoT Core")
    else:
        print(f"Connection failed with code {rc}")

def generate_sensor_data():
    temperature = round(random.uniform(35, 85), 1)
    vibration = round(random.uniform(0.1, 3.5), 2)
    pressure = round(random.uniform(90, 150), 2)

    # Maintenance score simulated: lower values indicate higher maintenance need
    maintenance_score = round(
        max(0, 100 - (temperature * 0.4 + vibration * 15 + (pressure - 100) * 0.3)), 2
    )

    return {
        "timestamp": time.strftime('%Y-%m-%d %H:%M:%S'),
        "temperature": temperature,
        "vibration": vibration,
        "pressure": pressure,
        "maintenance_score": maintenance_score
    }

def publish_data(client):
    while True:
        data = generate_sensor_data()
        payload = json.dumps(data)
        result = client.publish(TOPIC, payload)
        status = result[0]
        if status == 0:
            print(f"Published: {payload}")
        else:
            print("Failed to send message")
        time.sleep(3)

def connect_mqtt():
    client = mqtt_client.Client(CLIENT_ID)
    client.on_connect = on_connect
    client.tls_set(
        ca_certs=CA_PATH,
        certfile=CERT_PATH,
        keyfile=KEY_PATH,
        tls_version=ssl.PROTOCOL_TLSv1_2
    )
    client.connect(AWS_ENDPOINT, PORT)
    return client

if __name__ == '__main__':
    client = connect_mqtt()
    client.loop_start()
    publish_data(client)
