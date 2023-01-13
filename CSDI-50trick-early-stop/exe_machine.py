import argparse
import torch
import datetime
import json
import yaml
import os

from main_model import CSDI_Physio
from dataset import get_dataloader
from utils import train, evaluate

parser = argparse.ArgumentParser(description="CSDI")
parser.add_argument("--config", type=str, default="base.yaml")
parser.add_argument('--device', default='cuda:3', help='Device for Attack')
parser.add_argument("--seed", type=int, default=1)
parser.add_argument("--testmissingratio", type=float, default=0.1)
parser.add_argument(
    "--nfold", type=int, default=0, help="for 5fold test (valid value:[0-4])"
)
parser.add_argument("--unconditional", action="store_true")
parser.add_argument("--modelfolder", type=str, default="")
parser.add_argument("--nsample", type=int, default=30)
parser.add_argument("--ratio",type=float,default=0.7)
parser.add_argument("--epochs",type=int,default=100)
parser.add_argument("--diffusion_step",type=int,default=50)
parser.add_argument("--machine_number",type=int,default=1)
args = parser.parse_args()


path = "config/" + args.config
with open(path, "r") as f:
    config = yaml.safe_load(f)

config["model"]["is_unconditional"] = args.unconditional
config["model"]["test_missing_ratio"] = args.testmissingratio

config["diffusion"]["num_steps"] = args.diffusion_step
config["train"]["epochs"] = args.epochs
print(json.dumps(config, indent=4))

current_time = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")


# 由于是分开进行预测，
machine_number = args.machine_number

train_data_path_list = []
test_data_path_list = []
label_data_path_list = []


for file in os.listdir("data/Machine"):
    # if "_train.pkl" in file and f"machine-{str(machine_number)}" in file:
    if "_train.pkl" in file: # 一次性生成所有的内容
        train_data_path_list.append("data/Machine/" + file)
        test_data_path_list.append("data/Machine/" + file.replace("_train.pkl","_test.pkl"))
        label_data_path_list.append("data/Machine/" + file.replace("_train.pkl","_test_label.pkl"))

for i, train_data_path in enumerate(train_data_path_list):
    foldername = "./save3/" + f"{train_data_path.replace('_train.pkl','').replace('data/Machine/','')}" + "_unconditional:" + str(args.unconditional) + "_ratio:" + str(
        args.ratio) + "_diffusion_step:" + str(args.diffusion_step) + "/"
    print('model folder:', foldername)
    try:
        os.makedirs(foldername)
    except:
        # 如果当前已经存在这个文件，则说明已经有别的进程在处理它了，跳过之。
        continue
    with open(foldername + "config.json", "w") as f:
        json.dump(config, f, indent=4)

    test_data_path = test_data_path_list[i]
    label_data_path = label_data_path_list[i]


    train_loader, valid_loader, test_loader1, test_loader2 = get_dataloader(
        train_data_path,
        test_data_path,
        label_data_path,
        batch_size = 24
    )


    model = CSDI_Physio(config, args.device,target_dim=38,ratio = args.ratio).to(args.device)

    if args.modelfolder == "":
        train(
            model,
            config["train"],
            train_loader,
            valid_loader=valid_loader,
            foldername=foldername,
            test_loader1=test_loader1,
            test_loader2=test_loader2
        )


# evaluate(model, test_loader, nsample=args.nsample, scaler=1, foldername=foldername)
