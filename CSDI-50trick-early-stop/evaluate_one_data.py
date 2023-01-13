import argparse
import torch
import datetime
import json
import yaml
import os
from  torch.utils.data import DataLoader
from main_model import CSDI_Physio
from dataset import get_dataloader,TestData
from utils import evaluate,ddim_evaluate

parser = argparse.ArgumentParser(description="CSDI")
parser.add_argument("--config", type=str, default="base.yaml")
parser.add_argument('--device', default='cuda:3', help='Device for Attack')
parser.add_argument("--seed", type=int, default=1)
parser.add_argument("--testmissingratio", type=float, default=0.1)

parser.add_argument("--unconditional", action="store_true")
parser.add_argument("--path", type=str, default="")
parser.add_argument("--nsample", type=int, default=30)
parser.add_argument("--ratio",type=float,default=0.7)
parser.add_argument("--diffusion_step",type=int,default=50)
parser.add_argument("--name",type=str)
parser.add_argument("--times",type=int,default=1)
parser.add_argument("--foldername",type=str,default="")
parser.add_argument("--ddim",action="store_true")
parser.add_argument("--data_number",type=str)
parser.add_argument("--inference_time",type=int,default=5)
parser.add_argument("--ddim_eta",type=float,default=0)
parser.add_argument("--ddim_steps",type=int,default=100)
args = parser.parse_args()
print(args)

path = "config/" + args.config
with open(path, "r") as f:
    config = yaml.safe_load(f)

config["model"]["is_unconditional"] = args.unconditional
config["model"]["test_missing_ratio"] = args.testmissingratio
config["diffusion"]["num_steps"] = args.diffusion_step

train_loader, valid_loader, test_loader1, test_loader2 = get_dataloader(
    f"./data/Machine/machine-{args.data_number}_train.pkl",
    f"./data/Machine/machine-{args.data_number}_test.pkl",
    f"./data/Machine/machine-{args.data_number}_test_label.pkl",
    batch_size = 32,
)

model = CSDI_Physio(config, args.device,target_dim=38,ratio = args.ratio).to(args.device)

model.load_state_dict(torch.load(args.path))
model = model.to(args.device)

print("model load over")
try:
    os.mkdir(args.foldername)
except:
    pass
for i, inference_counter in enumerate(range(args.inference_time)):
    if args.ddim:
        print("using ddim")
        ddim_evaluate(model, test_loader1, test_loader2, nsample=1, scaler=1, name=inference_counter,foldername = args.foldername,ddim_eta=args.ddim_eta,ddim_steps=args.ddim_steps)
    else:
        evaluate(model, test_loader1,test_loader2, nsample=1, scaler=1, name= inference_counter,foldername = args.foldername)