from predict import Predictor
from load_model import Model
m = Model()
model = m.load_model()


p = Predictor()
print(p.predict(model, "0"))