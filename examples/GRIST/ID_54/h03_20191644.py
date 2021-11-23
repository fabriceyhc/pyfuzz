# @Time    : 11/11/21 9:09 PM
# @Author  : Fabrice Harel-Canada
# @File    : h03_20191644.py

import torch
import torch.nn.functional as F
import torch.nn as nn

def setup():

    global model

    # training data

    x_T = torch.FloatTensor([
        [1, 2, 1, 1], 
        [2, 1, 3, 2], 
        [3, 1, 3, 4], 
        [4, 1, 5, 5], 
        [1, 7, 5, 5], 
        [1, 2, 5, 6], 
        [1, 6, 6, 6], 
        [1, 7, 7, 7]
    ])
    y_T = torch.LongTensor([2, 2, 2, 1, 1, 1, 0, 0])

    # model + optimizer
    model = nn.Linear(4, 3)
    optimizer = torch.optim.Adam(model.parameters(), lr=1)

    # train
    for epoch in range(3001):
        z = model(x_T)
        loss = F.cross_entropy(z, y_T) 

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    return model

def predict(inputs):
    outputs = model(inputs)


    # apparently pytorch doesn't throw errors when the input to F.softmax or torch.argmax contains nans
    # the argmax has some default logic in it where it returns 0 when all values are the same (i.e. all nans) 
    # I don't think its fair to force an error here, but this is a sneaky issue...
    # if torch.isnan(outputs).any():
    #     raise ValueError("outputs contained nan!")
    
    class_probs = F.softmax(outputs, dim=1)
    class_preds = torch.argmax(class_probs, dim=1)
    return class_probs

if __name__ == '__main__':
    model = setup()

    test_data = torch.FloatTensor([
        [1, 11, 10, 9], 
        [1, 3, 4, 3], 
        [1, 1, 0, 1]
    ])
    class_preds = predict(test_data)
    print(class_preds)
