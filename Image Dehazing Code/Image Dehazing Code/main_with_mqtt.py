import os
import cv2
import pickle
import numpy as np
from datetime import datetime
import paho.mqtt.client as mqtt
import requests
import threading
import time

# MQTT Configuration
broker = "broker.emqx.io"
port = 1883
topic = "IMAGE_DEHAZING_KUCHBHI_ROBOT_SUB_TP"
publish_topic = "IMAGE_DEHAZING_KUCHBHI_ROBOT_PUB_TP"

# Global signal flag
signal_received = False


def ensure_output_folder(folder_name="images"):

    os.makedirs(folder_name, exist_ok=True)
    return folder_name


def load_dehazing_model(model_path="dehaze_model.pkl"):

    try:
        with open(model_path, "rb") as file:
            return pickle.load(file)
    except Exception as e:
        print(f"Error loading dehazing model: {e}")
        return None


def dehaze_image(image, model):

    try:
        dark_channel = model.get_dark_channel(image)
        atmosphere = model.get_atmosphere(image, dark_channel)
        transmission = model.get_transmission(image, atmosphere)
        return model.recover_image(image, transmission, atmosphere)
    except Exception as e:
        print(f"Error during dehazing: {e}")
        return None


def save_image(image, folder, prefix, timestamp):

    file_path = os.path.join(folder, f"{prefix}_{timestamp}.jpg")
    cv2.imwrite(file_path, image)
    return file_path


def handle_signal(frame, dehaze_model, output_folder, mqtt_client):

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Save original frame
    original_image_path = save_image(frame, output_folder, "current", timestamp)

    # Perform dehazing
    dehazed_image = dehaze_image(frame, dehaze_model)
    if dehazed_image is None:
        return

    # Save dehazed image
    dehazed_image_path = save_image(dehazed_image, output_folder, "dehazed", timestamp)

    # Send the saved image to the server using threading
    threading.Thread(
        target=send_image_to_server, args=(dehazed_image_path, mqtt_client)
    ).start()


# MQTT Callbacks
def on_connect(client, userdata, flags, rc):

    if rc == 0:
        print("Connected to MQTT Broker!")
        client.subscribe(topic)
    else:
        print(f"Failed to connect, return code {rc}")


def on_message(client, userdata, message):

    global signal_received
    try:
        if message.payload.decode() == "1":
            signal_received = True
            print("Signal received from MQTT!")
    except Exception as e:
        print(f"Error processing MQTT message: {e}")


def publish_data(client, topic, data):
    result = client.publish(topic, data)
    status = result[0]
    if status == 0:
        print(f"Sent {data} to topic {topic}")
    else:
        print(f"Failed to send message to topic {topic}")


def setup_mqtt():

    client = mqtt.Client()
    client.on_connect = on_connect
    client.on_message = on_message
    client.connect(broker, port)
    client.loop_start()
    return client


def send_image_to_server(file_path, client):

    url = "https://digigrow.ca/image-dehazing-robot/upload_image.php"

    with open(file_path, "rb") as image_file:

        files = {"image": (file_path, image_file, "image/jpeg")}

        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/58.0.3029.110 Safari/537.36",
            "Accept": "application/json",
        }

        response = requests.post(url, files=files, headers=headers)

    # Check the response status and content
    if response.status_code == 200:
        print("Image uploaded successfully!")
        print("Response:", response.json())
        publish_data(client, publish_topic, response.json()["image_url"])

    else:
        print("Failed to upload image. Status code:", response.status_code)
        print("Response:", response.text)


def live_video_feed(dehaze_model, output_folder="images", mqtt_client=None):

    global signal_received

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not open video feed.")
        return

    print("Press 'q' to quit.")
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error: Failed to capture frame.")
            break

        cv2.imshow("Live Video Feed", frame)

        # Process signal if received
        if signal_received:
            signal_received = False
            handle_signal(frame, dehaze_model, output_folder, mqtt_client)

        # Exit on 'q' key
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":

    output_folder = ensure_output_folder()

    # Load required models
    dehaze_model = load_dehazing_model()

    if dehaze_model:
        # Setup MQTT connection
        mqtt_client = setup_mqtt()

        # Start the live video feed
        live_video_feed(dehaze_model, output_folder, mqtt_client)

        # Stop MQTT loop after exiting
        mqtt_client.loop_stop()
