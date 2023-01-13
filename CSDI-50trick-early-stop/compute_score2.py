import pickle
import torch
import csv
from tqdm import tqdm

def analyze_one_file(dict_file):

    csv_writer = csv.writer(open(f"result.csv" , "w"))
    best_f = -1
    for threshold in tqdm(range(0,200)):
        created_prediction = False
        one_tune_f = []
        for key in dict_file:

            result = dict_file[key]

            all_gen = result['all_generated_samples']
            all_target = result["all_target"]

            # print(torch.median(all_gen,1))
            # all_gen, _ = torch.median(all_gen,1)
            # all_gen = torch.Tensor(all_gen)
            # all_gen = torch.median(all_gen, 1).values

            eval_points = result["all_evalpoint"]

            all_gen = all_gen * eval_points
            all_target = all_target * eval_points

            residual = torch.sum((all_gen - all_target) ** 2, dim=-1).reshape(-1)
            # print("average residual is")
            # print(residual.sum() / torch.ones_like(residual).sum())
            label = pickle.load(
                open("data/Machine/machine-1-1_test_label.pkl", "rb")
            )[:len(residual)]

            label = torch.Tensor(label)

            true = torch.ones_like(residual)
            false = torch.zeros_like(residual)

            if not created_prediction:
                created_prediction = True
                prediction = torch.zeros_like(residual)
            # 每一次投一点票
            prediction += torch.where(residual > threshold, true, false) / len(dict_file.keys())

            # 计算一下每一次投票的f值
            one_turn_prediction = torch.where(residual > threshold, true, false)
            one_precise = torch.sum(one_turn_prediction == label).item() / torch.sum(one_turn_prediction == one_turn_prediction).item()
            one_recall = torch.sum((one_turn_prediction == label) * label).item() / torch.sum(label).item()

            one_f = 2 * one_precise * one_recall / (one_precise + one_recall)
            one_tune_f.append(one_f)


        origin_prediction = prediction
        for t in range(0,100):
            prediction = torch.where(origin_prediction > t/100, true, false)
            precise = torch.sum(prediction == label).item() / torch.sum(prediction == prediction).item()
            recall = torch.sum((prediction == label) * label).item() / torch.sum(label).item()

            f = 2 * precise * recall / (precise + recall)

            if f > best_f:
                best_f = f
            # csv_writer.writerow([precise,recall,f])
                print("best f is")
                print(best_f)
                print(one_tune_f)
    csv_writer.writerow(["best f",best_f])
    print(f"\nbest f is:")
    print(best_f)

import os


analyze_one_file(torch.load("ensemble_test.pkl"))