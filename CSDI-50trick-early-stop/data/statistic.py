import os
import pickle
import csv
import torch

def get_batch(label):
    batch_list = []
    length = len(label)
    i = 0
    while(i < length):
        if label[i]:
            j = i
            temp_length = 0
            while label[j] and j < length:
                temp_length += 1
                j += 1
            batch_list.append(temp_length)
            i = j
        else:
            i += 1
    return batch_list

result = csv.writer(open("result.csv","w"))
result.writerow(
    ['name','number','average','max_batch','ge_20_number','ge_50_number','ge_100_number']
)
for file in os.listdir("Machine"):
    if "test_label.pkl" in file:
        fr = open("Machine/" + file, "rb")
        label = pickle.load(fr)
        name = file.split("_")[0]

        batch_list = get_batch(label)

        number = len(batch_list)
        average = sum(batch_list) / number
        max_batch = max(batch_list)

        batch_list = torch.Tensor(batch_list)
        ge_20_number = sum(batch_list > 20).item()
        ge_50_number = sum(batch_list > 50).item()
        ge_100_number = sum(batch_list > 100).item()

        result.writerow([
            name,number,average,max_batch,ge_20_number,ge_50_number,ge_100_number
        ])



