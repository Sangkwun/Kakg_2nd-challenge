from dataset import Dataset
from model import Model

dataset = Dataset()
x_train, x_test, y_train = dataset.get_train_testset()
y_scaler = dataset.get_scaler()

model = Model(x_train, x_test, y_train, y_scaler, output='output_2.csv')
model.start()
model.submit()