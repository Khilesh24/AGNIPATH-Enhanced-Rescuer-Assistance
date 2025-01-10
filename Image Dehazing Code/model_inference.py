# model_inference.py
import cv2
import pickle
import numpy as np

# Load the model from the pickle file
with open("dehaze_model.pkl", "rb") as file:
    loaded_model = pickle.load(file)

# Load the hazy image
hazy_image = cv2.imread("../road_hazed.jpg").astype(np.float32)

# Perform dehazing using the loaded model
dark_channel = loaded_model.get_dark_channel(hazy_image)
atmosphere = loaded_model.get_atmosphere(hazy_image, dark_channel)
transmission = loaded_model.get_transmission(hazy_image, atmosphere)
dehazed_image = loaded_model.recover_image(hazy_image, transmission, atmosphere)

# Save and display the dehazed image
cv2.imwrite("../dehazed_image_road.jpg", dehazed_image)
cv2.imshow("Dehazed Image", dehazed_image)
cv2.waitKey(0)
cv2.destroyAllWindows()
