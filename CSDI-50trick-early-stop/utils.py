import numpy as np
import torch
from torch.optim import Adam
from tqdm import tqdm
import pickle


def train(
    model,
    config,
    train_loader,
    test_loader1=None,
    test_loader2=None,
    valid_loader=None,
    valid_epoch_interval=5,
    foldername="",
):
    optimizer = Adam(model.parameters(), lr=config["lr"], weight_decay=1e-6)

    p1 = int(0.75 * config["epochs"])
    p2 = int(0.9 * config["epochs"])
    lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(
        optimizer, milestones=[p1, p2], gamma=0.1
    )
    nsample_list = [
        1
    ]
    best_mse_score = 10000
    stop_counter = 0
    best_valid_loss = 1e10
    # for epoch_no in range(config["epochs"]):
    # !for test!
    for epoch_no in range(0,500):

        avg_loss = 0
        model.train()
        with tqdm(train_loader) as it:
            for batch_no, train_batch in enumerate(it, start=1):
                optimizer.zero_grad()
                #!for test
                # if batch_no == 100:
                #     break

                loss = model(train_batch)
                loss.backward()
                avg_loss += loss.item()
                optimizer.step()
                it.set_postfix(
                    ordered_dict={
                        "avg_epoch_loss": avg_loss / batch_no,
                        "epoch": epoch_no,
                    },
                    refresh=False,
                )
            lr_scheduler.step()

        # if valid_loader is not None and (epoch_no + 1) % valid_epoch_interval == 0:
        mse_score = validation(model, valid_loader ,nsample=1)
        if mse_score < best_mse_score:
            stop_counter = 0
            best_mse_score = mse_score
            print("best mse score update")
            print("now best is")
            print(best_mse_score)
            output_path = foldername + f"/best-model.pth"
            torch.save(model.state_dict(), output_path)
        else:
            stop_counter += 1

        # 如果
        if stop_counter > 10:
            break



        """
            model.eval()
            avg_loss_valid = 0
            with torch.no_grad():
                with tqdm(valid_loader) as it:
                    for batch_no, valid_batch in enumerate(it, start=1):
                        loss = model(valid_batch, is_train=0)
                        avg_loss_valid += loss.item()
                        it.set_postfix(
                            ordered_dict={
                                "valid_avg_epoch_loss": avg_loss_valid / batch_no,
                                "epoch": epoch_no,
                            },
                            refresh=False,
                        )
            if best_valid_loss > avg_loss_valid:
                best_valid_loss = avg_loss_valid
                print(
                    "\n best loss is updated to ",
                    avg_loss_valid / batch_no,
                    "at",
                    epoch_no,
                )
            """

        # !for test
        # if epoch_no:
        if (epoch_no + 1 ) % 5 == 0:
            if foldername != "":
                output_path = foldername + f"/{epoch_no}-model.pth"
                torch.save(model.state_dict(), output_path)

            for nsample in nsample_list:

                evaluate(model, test_loader1, test_loader2, nsample, foldername=foldername, epoch_number=str(epoch_no))

def quantile_loss(target, forecast, q: float, eval_points) -> float:
    return 2 * torch.sum(
        torch.abs((forecast - target) * eval_points * ((target <= forecast) * 1.0 - q))
    )


def calc_denominator(target, eval_points):
    return torch.sum(torch.abs(target * eval_points))


def calc_quantile_CRPS(target, forecast, eval_points, mean_scaler, scaler):
    target = target * scaler + mean_scaler
    forecast = forecast * scaler + mean_scaler

    quantiles = np.arange(0.05, 1.0, 0.05)
    denom = calc_denominator(target, eval_points)
    CRPS = 0
    for i in range(len(quantiles)):
        q_pred = []
        for j in range(len(forecast)):
            q_pred.append(torch.quantile(forecast[j : j + 1], quantiles[i], dim=1))
        q_pred = torch.cat(q_pred, 0)
        q_loss = quantile_loss(target, q_pred, quantiles[i], eval_points)
        CRPS += q_loss / denom
    return CRPS.item() / len(quantiles)


def validation(model, valid_loader, nsample=20, scaler=1):

    with torch.no_grad():
        model.eval()
        mse_total = 0
        mae_total = 0
        evalpoints_total = 0


        with tqdm(valid_loader, mininterval=5.0, maxinterval=50.0) as it:
            for batch_no, test_batch in enumerate(it, start=1):
                # 计算strategy1的结果
                output = model.evaluate(test_batch, nsample)
                samples, c_target, eval_points, observed_points, observed_time = output
                samples = samples.permute(0, 1, 3, 2)  # (B,nsample,L,K)
                c_target = c_target.permute(0, 2, 1)  # (B,L,K)
                eval_points = eval_points.permute(0, 2, 1)
                # observed_points = observed_points.permute(0, 2, 1)

                samples_median = samples.median(dim=1)

                mse_current = (
                    ((samples_median.values - c_target) * eval_points) ** 2
                ) * (scaler ** 2)
                mae_current = (
                    torch.abs((samples_median.values - c_target) * eval_points)
                ) * scaler

                mse_total += mse_current.sum().item()
                mae_total += mae_current.sum().item()
                evalpoints_total += eval_points.sum().item()

                it.set_postfix(
                    ordered_dict={
                        "rmse_total": np.sqrt(mse_total / evalpoints_total),
                        "mae_total": mae_total / evalpoints_total,
                        "batch_no": batch_no,
                    },
                    refresh=True,
                )
    return np.sqrt(mse_total / evalpoints_total)

# def evaluate(model, test_loader, nsample=100, scaler=1, mean_scaler=0, foldername=""):
def evaluate(model, test_loader1, test_loader2, nsample=20, scaler=1, mean_scaler=0, foldername="",epoch_number = "",name=""):

    with torch.no_grad():
        model.eval()
        mse_total = 0
        mae_total = 0
        evalpoints_total = 0

        all_target = []
        all_observed_point = []
        all_observed_time = []
        all_evalpoint = []
        all_generated_samples = []
        test_loader2 = iter(test_loader2)

        with tqdm(test_loader1, mininterval=5.0, maxinterval=50.0) as it:
            for batch_no, test_batch in enumerate(it, start=1):
                # 计算strategy1的结果
                output = model.evaluate(test_batch, nsample)
                samples, c_target, eval_points, observed_points, observed_time = output
                samples = samples.permute(0, 1, 3, 2)  # (B,nsample,L,K)
                c_target = c_target.permute(0, 2, 1)  # (B,L,K)
                eval_points = eval_points.permute(0, 2, 1)
                observed_points = observed_points.permute(0, 2, 1)

                # 计算strategy2的结果
                output2 = model.evaluate(next(test_loader2), nsample)
                samples2 = output2[0]

                samples2 = samples2.permute(0, 1, 3, 2)  # (B,nsample,L,K)
                samples_length = samples.shape[2]
                samples[:,:,samples_length // 4 : samples_length //2, :] = samples2[:,:,samples_length // 4 : samples_length //2, :]
                samples[:,:,samples_length - samples_length // 4:,:] = samples2[:,:,samples_length - samples_length // 4:,:]

                samples_median = samples.median(dim=1)
                all_target.append(c_target)
                all_evalpoint.append(eval_points)
                all_observed_point.append(observed_points)
                all_observed_time.append(observed_time)
                all_generated_samples.append(samples)

                mse_current = (
                    ((samples_median.values - c_target)) ** 2
                ) * (scaler ** 2)
                mae_current = (
                    torch.abs((samples_median.values - c_target))
                ) * scaler

                mse_total += mse_current.sum().item()
                mae_total += mae_current.sum().item()
                # evalpoints_total += eval_points.sum().item()
                evalpoints_total += torch.ones_like(mse_current).sum().item()

                it.set_postfix(
                    ordered_dict={
                        "rmse_total": np.sqrt(mse_total / evalpoints_total),
                        "mae_total": mae_total / evalpoints_total,
                        "batch_no": batch_no,
                    },
                    refresh=True,
                )

            with open(
                foldername + f"/{epoch_number}-generated_outputs_nsample" + str(nsample) + f"{name}.pk", "wb"
            ) as f:
                all_target = torch.cat(all_target, dim=0).to("cpu")
                all_evalpoint = torch.cat(all_evalpoint, dim=0).to("cpu")
                all_observed_point = torch.cat(all_observed_point, dim=0).to("cpu")
                all_observed_time = torch.cat(all_observed_time, dim=0).to("cpu")
                all_generated_samples = torch.cat(all_generated_samples, dim=0).to("cpu")

                pickle.dump(
                    [
                        all_generated_samples,
                        all_target,
                        all_evalpoint,
                        all_observed_point,
                        all_observed_time,
                        scaler,
                        mean_scaler,
                    ],
                    f,
                )

            CRPS = calc_quantile_CRPS(
                all_target, all_generated_samples, all_evalpoint, mean_scaler, scaler
            )

            with open(
                foldername + f"/{epoch_number}-result_nsample" + str(nsample) + ".pk", "wb"
            ) as f:
                pickle.dump(
                    [
                        np.sqrt(mse_total / evalpoints_total),
                        mae_total / evalpoints_total,
                        CRPS,
                    ],
                    f,
                )
                print("RMSE:", np.sqrt(mse_total / evalpoints_total))
                print("MAE:", mae_total / evalpoints_total)
                print("CRPS:", CRPS)


def ddim_evaluate(model, test_loader1, test_loader2, nsample=20, scaler=1, mean_scaler=0, foldername="",epoch_number = "",name="",ddim_eta=0,ddim_steps=10):

    with torch.no_grad():
        model.eval()
        mse_total = 0
        mae_total = 0
        evalpoints_total = 0

        all_target = []
        all_observed_point = []
        all_observed_time = []
        all_evalpoint = []
        all_generated_samples = []
        test_loader2 = iter(test_loader2)

        with tqdm(test_loader1, mininterval=5.0, maxinterval=50.0) as it:
            for batch_no, test_batch in enumerate(it, start=1):
                # 计算strategy1的结果
                # output = model.evaluate(test_batch, nsample)
                output = model.ddim_evaluate(test_batch, nsample, ddim_eta=ddim_eta, ddim_steps=ddim_steps)
                samples, c_target, eval_points, observed_points, observed_time = output
                samples = samples.permute(0, 1, 3, 2)  # (B,nsample,L,K)
                c_target = c_target.permute(0, 2, 1)  # (B,L,K)
                eval_points = eval_points.permute(0, 2, 1)
                observed_points = observed_points.permute(0, 2, 1)

                # 计算strategy2的结果
                # output2 = model.evaluate(next(test_loader2), nsample)
                output2 = model.ddim_evaluate(next(test_loader2), nsample, ddim_eta=ddim_eta, ddim_steps=ddim_steps)

                samples2 = output2[0]

                samples2 = samples2.permute(0, 1, 3, 2)  # (B,nsample,L,K)
                samples_length = samples.shape[2]
                samples[:,:,samples_length // 4 : samples_length //2, :] = samples2[:,:,samples_length // 4 : samples_length //2, :]
                samples[:,:,samples_length - samples_length // 4:,:] = samples2[:,:,samples_length - samples_length // 4:,:]

                samples_median = samples.median(dim=1)
                all_target.append(c_target)
                all_evalpoint.append(eval_points)
                all_observed_point.append(observed_points)
                all_observed_time.append(observed_time)
                all_generated_samples.append(samples)

                mse_current = (
                    ((samples_median.values - c_target)) ** 2
                ) * (scaler ** 2)
                mae_current = (
                    torch.abs((samples_median.values - c_target))
                ) * scaler

                mse_total += mse_current.sum().item()
                mae_total += mae_current.sum().item()
                # evalpoints_total += eval_points.sum().item()
                evalpoints_total += torch.ones_like(mse_current).sum().item()

                it.set_postfix(
                    ordered_dict={
                        "rmse_total": np.sqrt(mse_total / evalpoints_total),
                        "mae_total": mae_total / evalpoints_total,
                        "batch_no": batch_no,
                    },
                    refresh=True,
                )

            with open(
                foldername + f"/{epoch_number}-generated_outputs_nsample" + str(nsample) + f"{name}.pk", "wb"
            ) as f:
                all_target = torch.cat(all_target, dim=0).to("cpu")
                all_evalpoint = torch.cat(all_evalpoint, dim=0).to("cpu")
                all_observed_point = torch.cat(all_observed_point, dim=0).to("cpu")
                all_observed_time = torch.cat(all_observed_time, dim=0).to("cpu")
                all_generated_samples = torch.cat(all_generated_samples, dim=0).to("cpu")

                pickle.dump(
                    [
                        all_generated_samples,
                        all_target,
                        all_evalpoint,
                        all_observed_point,
                        all_observed_time,
                        scaler,
                        mean_scaler,
                    ],
                    f,
                )

            CRPS = calc_quantile_CRPS(
                all_target, all_generated_samples, all_evalpoint, mean_scaler, scaler
            )

            with open(
                foldername + f"/{epoch_number}-result_nsample" + str(nsample) + ".pk", "wb"
            ) as f:
                pickle.dump(
                    [
                        np.sqrt(mse_total / evalpoints_total),
                        mae_total / evalpoints_total,
                        CRPS,
                    ],
                    f,
                )
                print("RMSE:", np.sqrt(mse_total / evalpoints_total))
                print("MAE:", mae_total / evalpoints_total)
                print("CRPS:", CRPS)

"""
def ddim_evaluate(model, test_loader, nsample=20, scaler=1, mean_scaler=0, foldername="",epoch_number = "",name="",ddim_eta=1,ddim_steps =10):

    with torch.no_grad():
        model.eval()
        mse_total = 0
        mae_total = 0
        evalpoints_total = 0

        all_target = []
        all_observed_point = []
        all_observed_time = []
        all_evalpoint = []
        all_generated_samples = []

        with tqdm(test_loader, mininterval=5.0, maxinterval=50.0) as it:
            for batch_no, test_batch in enumerate(it, start=1):
                output = model.ddim_evaluate(test_batch, nsample,ddim_eta=ddim_eta,ddim_steps=ddim_steps)
                samples, c_target, eval_points, observed_points, observed_time = output
                samples = samples.permute(0, 1, 3, 2)  # (B,nsample,L,K)
                c_target = c_target.permute(0, 2, 1)  # (B,L,K)
                eval_points = eval_points.permute(0, 2, 1)
                observed_points = observed_points.permute(0, 2, 1)

                samples_median = samples.median(dim=1)
                all_target.append(c_target)
                all_evalpoint.append(eval_points)
                all_observed_point.append(observed_points)
                all_observed_time.append(observed_time)
                all_generated_samples.append(samples)

                mse_current = (
                    ((samples_median.values - c_target) * eval_points) ** 2
                ) * (scaler ** 2)
                mae_current = (
                    torch.abs((samples_median.values - c_target) * eval_points)
                ) * scaler

                mse_total += mse_current.sum().item()
                mae_total += mae_current.sum().item()
                evalpoints_total += eval_points.sum().item()

                it.set_postfix(
                    ordered_dict={
                        "rmse_total": np.sqrt(mse_total / evalpoints_total),
                        "mae_total": mae_total / evalpoints_total,
                        "batch_no": batch_no,
                    },
                    refresh=True,
                )

            with open(
                foldername + f"/{epoch_number}-generated_outputs_nsample" + str(nsample) + f"{name}.pk", "wb"
            ) as f:
                all_target = torch.cat(all_target, dim=0).to("cpu")
                all_evalpoint = torch.cat(all_evalpoint, dim=0).to("cpu")
                all_observed_point = torch.cat(all_observed_point, dim=0).to("cpu")
                all_observed_time = torch.cat(all_observed_time, dim=0).to("cpu")
                all_generated_samples = torch.cat(all_generated_samples, dim=0).to("cpu")

                pickle.dump(
                    [
                        all_generated_samples,
                        all_target,
                        all_evalpoint,
                        all_observed_point,
                        all_observed_time,
                        scaler,
                        mean_scaler,
                    ],
                    f,
                )

            CRPS = calc_quantile_CRPS(
                all_target, all_generated_samples, all_evalpoint, mean_scaler, scaler
            )

            with open(
                foldername + f"/{epoch_number}-result_nsample" + str(nsample) + ".pk", "wb"
            ) as f:
                pickle.dump(
                    [
                        np.sqrt(mse_total / evalpoints_total),
                        mae_total / evalpoints_total,
                        CRPS,
                    ],
                    f,
                )
                print("RMSE:", np.sqrt(mse_total / evalpoints_total))
                print("MAE:", mae_total / evalpoints_total)
                print("CRPS:", CRPS)
"""


def ensemble(model, test_loader, nsample=10, scaler=1, name = ""):

    with torch.no_grad():
        model.eval()
        mse_total = 0
        mae_total = 0
        evalpoints_total = 0

        impute_sample_dict = {

        }

        for i in tqdm(range(0,nsample)):
            impute_sample_dict[i] = {
            }
            all_target = []
            all_observed_point = []
            all_observed_time = []
            all_evalpoint = []
            all_generated_samples = []
            with tqdm(test_loader, mininterval=5.0, maxinterval=50.0) as it:
                for batch_no, test_batch in enumerate(it, start=1):
                    output = model.evaluate(test_batch, 1)
                    samples, c_target, eval_points, observed_points, observed_time = output
                    samples = samples.permute(0, 1, 3, 2).squeeze()  # (B,nsample,L,K)
                    c_target = c_target.permute(0, 2, 1)  # (B,L,K)
                    eval_points = eval_points.permute(0, 2, 1)
                    observed_points = observed_points.permute(0, 2, 1)

                    samples_median = samples
                    all_target.append(c_target)
                    all_evalpoint.append(eval_points)
                    all_observed_point.append(observed_points)
                    all_observed_time.append(observed_time)
                    all_generated_samples.append(samples)

                    mse_current = (
                        ((samples_median - c_target) * eval_points) ** 2
                    ) * (scaler ** 2)
                    mae_current = (
                        torch.abs((samples_median - c_target) * eval_points)
                    ) * scaler

                    mse_total += mse_current.sum().item()
                    mae_total += mae_current.sum().item()
                    evalpoints_total += eval_points.sum().item()

                    it.set_postfix(
                        ordered_dict={
                            "rmse_total": np.sqrt(mse_total / evalpoints_total),
                            "mae_total": mae_total / evalpoints_total,
                            "batch_no": batch_no,
                        },
                        refresh=True,
                    )

                if 1:
                    all_target = torch.cat(all_target, dim=0).to("cpu")
                    all_evalpoint = torch.cat(all_evalpoint, dim=0).to("cpu")
                    all_observed_point = torch.cat(all_observed_point, dim=0).to("cpu")
                    all_observed_time = torch.cat(all_observed_time, dim=0).to("cpu")
                    all_generated_samples = torch.cat(all_generated_samples, dim=0).to("cpu")
                    impute_sample_dict[i]["all_target"] = all_target
                    impute_sample_dict[i]["all_evalpoint"] = all_evalpoint
                    impute_sample_dict[i]["all_observed_point"] = all_observed_point
                    impute_sample_dict[i]["all_observed_time"] = all_observed_time
                    impute_sample_dict[i]["all_generated_samples"] = all_generated_samples
    torch.save(impute_sample_dict,name)

def ddim_ensemble(model, test_loader, nsample=10, scaler=1, name = "",ddim_eta=1,ddim_steps =10):

    with torch.no_grad():
        model.eval()
        mse_total = 0
        mae_total = 0
        evalpoints_total = 0

        impute_sample_dict = {

        }

        for i in tqdm(range(0,nsample)):
            impute_sample_dict[i] = {
            }
            all_target = []
            all_observed_point = []
            all_observed_time = []
            all_evalpoint = []
            all_generated_samples = []
            with tqdm(test_loader, mininterval=5.0, maxinterval=50.0) as it:
                for batch_no, test_batch in enumerate(it, start=1):
                    output = model.ddim_evaluate(test_batch, 1,ddim_eta=ddim_eta,ddim_steps = ddim_steps)
                    samples, c_target, eval_points, observed_points, observed_time = output
                    samples = samples.permute(0, 1, 3, 2).squeeze()  # (B,nsample,L,K)
                    c_target = c_target.permute(0, 2, 1)  # (B,L,K)
                    eval_points = eval_points.permute(0, 2, 1)
                    observed_points = observed_points.permute(0, 2, 1)

                    samples_median = samples
                    all_target.append(c_target)
                    all_evalpoint.append(eval_points)
                    all_observed_point.append(observed_points)
                    all_observed_time.append(observed_time)
                    all_generated_samples.append(samples)

                    mse_current = (
                        ((samples_median - c_target) * eval_points) ** 2
                    ) * (scaler ** 2)
                    mae_current = (
                        torch.abs((samples_median - c_target) * eval_points)
                    ) * scaler

                    mse_total += mse_current.sum().item()
                    mae_total += mae_current.sum().item()
                    evalpoints_total += eval_points.sum().item()

                    it.set_postfix(
                        ordered_dict={
                            "rmse_total": np.sqrt(mse_total / evalpoints_total),
                            "mae_total": mae_total / evalpoints_total,
                            "batch_no": batch_no,
                        },
                        refresh=True,
                    )

                if 1:
                    all_target = torch.cat(all_target, dim=0).to("cpu")
                    all_evalpoint = torch.cat(all_evalpoint, dim=0).to("cpu")
                    all_observed_point = torch.cat(all_observed_point, dim=0).to("cpu")
                    all_observed_time = torch.cat(all_observed_time, dim=0).to("cpu")
                    all_generated_samples = torch.cat(all_generated_samples, dim=0).to("cpu")
                    impute_sample_dict[i]["all_target"] = all_target
                    impute_sample_dict[i]["all_evalpoint"] = all_evalpoint
                    impute_sample_dict[i]["all_observed_point"] = all_observed_point
                    impute_sample_dict[i]["all_observed_time"] = all_observed_time
                    impute_sample_dict[i]["all_generated_samples"] = all_generated_samples
    torch.save(impute_sample_dict,name)