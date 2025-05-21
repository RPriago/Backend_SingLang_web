import pickle
import numpy as np

with open("model.p", "rb") as f:
    model = pickle.load(f)

def predict_gesture(features):
    features = np.array(features).reshape(1, -1)
    return model.predict(features)[0]