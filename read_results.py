# %%
import numpy as np
import pandas as pd
import json
import os


def read_jsonl(file_path):
    data = []
    with open(file_path, "r") as f:
        for line in f:
            data.append(json.loads(line))
    return data


prompt_type_map = {
    "openPangu-Embedded-7B-V1.1": "pangu",
    "sail/Qwen2.5-Math-7B-Oat-Zero": "qwen25-math-cot",
    "qihoo360/Light-R1-7B-DS": "deepseek-r1",
    "Skywork/Skywork-OR1-Math-7B": "deepseek-r1",
}

k = 16

root_path = f"score_pass_{k}"

datasets = ["amc23", "aime24", "aime25"]


def read_results(model, think_mode="slow") -> pd.DataFrame:
    results = []
    for dataset in datasets:
        if model == "openPangu-Embedded-7B-V1.1":
            filename = f"test_pangu_-1_seed0_t1.0_top-p0.8_{think_mode}_metrics.json"
        else:
            filename = (
                f"test_{prompt_type_map[model]}_-1_seed0_t1.0_top-p0.8_metrics.json"
            )

        metric_file = os.path.join(
            root_path,
            model,
            dataset,
            filename,
        )
        with open(metric_file, "r") as f:
            metrics = json.load(f)

        var_metric_file = os.path.join(
            root_path,
            model,
            f"var_{dataset}",
            filename,
        )

        with open(var_metric_file, "r") as f:
            var_metrics = json.load(f)

        drop = var_metrics["acc"] - metrics["acc"]
        drop_ratio = drop / metrics["acc"] if metrics["acc"] > 0 else 0

        results.append(
            {
                "dataset": dataset,
                "acc": round(metrics["acc"], 2),
                "var_acc": round(var_metrics["acc"], 2),
                "drop": round(drop, 2),
                "drop_ratio": f"{round(drop_ratio * 100, 2)}%",
                "token_len": round(metrics["avg_token_len"]),
                "var_token_len": round(var_metrics["avg_token_len"]),
            }
        )

    df = pd.DataFrame(results)
    return df


# %%
read_results("openPangu-Embedded-7B-V1.1", think_mode="slow")
#%%
read_results("openPangu-Embedded-7B-V1.1", think_mode="fast")
#%%
read_results("openPangu-Embedded-7B-V1.1", think_mode="auto")
# %%
read_results("sail/Qwen2.5-Math-7B-Oat-Zero")
# %%
read_results("qihoo360/Light-R1-7B-DS")
# %%
read_results("Skywork/Skywork-OR1-Math-7B")
# %%
