import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
import pandas as pd

data={
        'age':[30,35,40,45,50,55,60,65,70,75],
        'income':[50000,55000,60000,65000,70000,75000,80000,85000,90000,95000],
        'buy_car':[0,0,1,1,1,1,1,1,1,1]
    }

df = pd.DataFrame(data)

x = df[['age','income']]
y = df[['buy_car']]
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.3)

model = LogisticRegression()
model.fit(x_train,y_train)
predictions = model.predict(x_test)
print(f"predictions: {predictions}")

user_age = int(input('enter age: '))
user_income = int(input('enter income: '))
answer = model.predict([[user_age,user_income]])
# print(f"PREDICTED: {answer}")
if answer == 1:
    print('will be buy')
else:
    print('No chance for buying')