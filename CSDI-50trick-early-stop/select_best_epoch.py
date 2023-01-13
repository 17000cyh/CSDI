import os
import csv
import shutil


try:
    os.mkdir("save_best_epoch")
except:
    pass

best_epoch_file = csv.reader(open("save3/selected_threshold0.004/target_result.csv"))
best_dict = {}
for line in best_epoch_file:
    if "machine" not  in line[0]:
        continue
    best_dict[line[0]] = line[-1]

for key in best_dict.keys():
    try:
        os.mkdir(f"save_best_epoch/{key}_unconditional:True_ratio:0.7_diffusion_step:100")
    except:
        pass
    src = f"save3/{key}_unconditional:True_ratio:0.7_diffusion_step:100/{best_dict[key]}-generated_outputs_nsample1.pk"
    tar = f"save_best_epoch/{key}_unconditional:True_ratio:0.7_diffusion_step:100/{best_dict[key]}-generated_outputs_nsample1.pk"
    shutil.copy(src,tar)

    src = f"save3/{key}_unconditional:True_ratio:0.7_diffusion_step:100/{best_dict[key]}-generated_outputs_nsample1train_error.pk"
    tar = f"save_best_epoch/{key}_unconditional:True_ratio:0.7_diffusion_step:100/{best_dict[key]}-generated_outputs_nsample1train_error.pk"
    shutil.copy(src, tar)