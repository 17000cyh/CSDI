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
parser.add_argument("--model_path",type=str,default="save/machine_ratio0.95_diffusion_step50/19-model.pth")

args = parser.parse_args()
print(args)

path = "config/" + args.config
with open(path, "r") as f:
    config = yaml.safe_load(f)

config["model"]["is_unconditional"] = args.unconditional
config["model"]["test_missing_ratio"] = args.testmissingratio

config["diffusion"]["num_steps"] = args.diffusion_step
config["train"]["epochs"] = args.epochs
print(json.dumps(config, indent=4))

current_time = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

foldername = "./save2/machine" +"_unconditional:" + str(args.unconditional) + "_ratio:" + str(args.ratio) + "_diffusion_step:" + str(args.diffusion_step)  + "/"
print('model folder:', foldername)
os.makedirs(foldername, exist_ok=True)
with open(foldername + "config.json", "w") as f:
    json.dump(config, f, indent=4)


train_loader, valid_loader, test_loader = get_dataloader(
    "./data/Machine/machine-1-1_train.pkl",
    "./data/Machine/machine-1-1_test.pkl",
    "./data/Machine/machine-1-1_test_label.pkl",
    batch_size = 24,
    ratio = args.ratio
)


model = CSDI_Physio(config, args.device,target_dim=38,ratio = args.ratio).to(args.device)

model.load_state_dict(torch.load(args.model_path))

device = args.device

model = model.to(device)
for test_batch in test_loader:
    output = model.evaluate_middle_result(test_batch,args.nsample)
    break

samples, middle_samples = output[0], output[1]

print(samples.shape)
print(middle_samples.shape)
