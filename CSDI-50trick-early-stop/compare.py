import argparse
import torch
import datetime
import json
import yaml
import os

from main_model import CSDI_Physio
from dataset import get_dataloader
from utils import evaluate, ddim_evaluate

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
parser.add_argument("--foldername",type=str,default="")
parser.add_argument("--times",type=int,default=1)
parser.add_argument("--ddim",action="store_true")

args = parser.parse_args()
print(args)

path = "config/" + args.config
with open(path, "r") as f:
    config = yaml.safe_load(f)

config["model"]["is_unconditional"] = args.unconditional
config["model"]["test_missing_ratio"] = args.testmissingratio
config["diffusion"]["num_steps"] = args.diffusion_step

train_loader, valid_loader, test_loader = get_dataloader(
    "./data/Machine/machine-1-1_train.pkl",
    "./data/Machine/machine-1-1_test.pkl",
    "./data/Machine/machine-1-1_test_label.pkl",
    batch_size = 96,
    ratio = args.ratio
)

model = CSDI_Physio(config, args.device,target_dim=38,ratio = args.ratio).to(args.device)

model.load_state_dict(torch.load(args.path))
model = model.to(args.device)
print("model load over")
eta_list = [0,0.2,0.5,1]
ddim_step_list = [10,20,50,100]
evaluate(model,test_loader,nsample=1,foldername=".")

for eta in eta_list:
    for step in ddim_step_list:
        print(f"eta is {eta}")
        print(f"step is {step}")
        ddim_evaluate(model, test_loader, nsample=1, ddim_eta=eta, ddim_steps=step, foldername=".",name="eta0.0")
