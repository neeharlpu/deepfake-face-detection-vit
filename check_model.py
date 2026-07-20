from model import *

model = build_model()

count_parameters(model)

print(model.classifier)