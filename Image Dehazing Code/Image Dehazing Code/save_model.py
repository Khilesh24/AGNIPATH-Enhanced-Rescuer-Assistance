# save_model.py
import pickle
from Model.dehaze_model import DehazeModel

# Create an instance of DehazeModel
model = DehazeModel(omega=0.95, size=15, t0=0.1)

# Save the model using pickle
with open("dehaze_model.pkl", "wb") as file:
    pickle.dump(model, file)

print("Dehaze model exported as dehaze_model.pkl")
