
import pandas as pd
import numpy as np
from Traditional.Settings import *
from math import ceil, floor

train = pd.read_csv(train_cleaned_file_path)
test = pd.read_csv(test_cleaned_file_path)

trains = []

for i in range(0, 6):
    df = pd.concat([train.iloc[:, 0], train.iloc[:, 1], train.iloc[:, i+2]], axis=1)
    trains.append(df)

# print classes to file
for df in trains:
    class1 = df[(df.iloc[:, 2] == 1)]
    file_name = './Classes/' + class1.columns[2] + '.csv'
    class1.to_csv(file_name, index=False)
rowsums = train.iloc[:, 2:].sum(axis=1)
train['nontoxic'] = (rowsums == 0)
clean = train[train['nontoxic'] == True]
df_clean = clean.iloc[:, 1]
df_clean.to_csv('./Classes/clean.csv', index=False)



# # print last 10% training set to file
# validation = train.iloc[floor(train.__len__() * 0.9): floor(train.__len__() * 1.0), :]
# validation.to_csv("./Classes/validation.csv", index=False)
#
# class_x = validation[validation['threat'] == 1]
# print(class_x.__len__())
