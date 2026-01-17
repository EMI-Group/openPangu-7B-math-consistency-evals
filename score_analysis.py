import numpy as np
import os
import pandas as pd
from typing import Iterable, Union, Any
from pathlib import Path
import json
from scipy import stats
import argparse

pd.set_option("display.max_rows", None)
pd.set_option("display.max_columns", None)
pd.set_option("display.width", None)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset", type=str, default="amc23", choices=["amc23", "aime24", "aime25"]
    )
    parser.add_argument(
        "--t_test", action="store_true", help="Whether to perform t-test"
    )
    args = parser.parse_args()

    id_ref_file = "./data/{dataset}/test.jsonl"

    dataset_list = [f"var_{args.dataset.lower()}", args.dataset.lower()]
    # dataset_list = [args.dataset.lower()]
    agg_mode_list = ["loose", "strict"]

    N_samples = 16
    t_test = args.t_test
    N_boostrap = N_samples if t_test else 1000

    model_name_list = [
        ("sail/Qwen2.5-Math-7B-Oat-Zero", "qwen25-math-cot"),
        ("Skywork/Skywork-OR1-Math-7B", "deepseek-r1"),
        ("qihoo360/Light-R1-7B-DS", "deepseek-r1"),
        ("openPangu-Embedded-7B-V1.1", "pangu"),
    ]
    pangu_think_mode_list = ["slow", "auto", "fast"]

    def load_jsonl(file: Union[str, Path]) -> Iterable[Any]:
        with open(file, "r", encoding="utf-8") as f:
            for line in f:
                try:
                    yield json.loads(line)
                except:
                    print("Error in loading:", line)
                    exit()

    def load_json(file):
        with open(file, "r", encoding="utf-8") as f:
            return json.load(f)

    def format_p_value(p):
        if isinstance(p, (int, float)) and p < 0.01:
            return "<0.01**"
        elif isinstance(p, (int, float)) and p < 0.05:
            return f"{p:.2f}*"
        elif isinstance(p, (int, float)):
            return f"{p:.2f}"
        else:
            return p

    def process_eval(
        dataset, model_name, prompt_type, agg_mode=None, pangu_think_mode=None
    ):
        pre_path = f"score_pass_16/{model_name}/{dataset}"
        sample_file = f"test_{prompt_type}_-1_seed0_t1.0_top-p0.8.jsonl"
        if prompt_type == "pangu":
            sample_file = (
                f"test_{prompt_type}_-1_seed0_t1.0_top-p0.8_{pangu_think_mode}.jsonl"
            )
        metric_file = sample_file.replace(".jsonl", "_metrics.json")

        sample_filepath = os.path.join(pre_path, sample_file)
        # TODO: do we need sort?
        sample_list = sorted(list(load_jsonl(sample_filepath)), key=lambda x: x["idx"])
        metrics = load_json(os.path.join(pre_path, metric_file))

        id_ref_path = id_ref_file.format(dataset=dataset.lower())
        id_ref_list = list(load_jsonl(id_ref_path))
        idx_2_id_map = {idx: item["id"] for idx, item in enumerate(id_ref_list)}

        data_scores = []
        num_refined_samples, num_samples = 0, 0
        for idx in range(N_boostrap):
            data = {}
            for sample in sample_list:
                q_id = str(idx_2_id_map[sample["idx"]]).split("_")[0]
                real_n_samples = min(len(sample["score"]), N_samples)
                if t_test:
                    random_idx = idx % real_n_samples
                else:
                    random_idx = np.random.randint(real_n_samples)
                data.setdefault(q_id, []).append(sample["score"][random_idx])

            data_score = {
                q: (int(np.all(sc)) if agg_mode == "strict" else np.mean(sc))
                for q, sc in data.items()
            }
            data_scores.append(np.mean(list(data_score.values())))
            if idx == 0:
                num_refined_samples, num_samples = (
                    sum(len(v) > 1 for v in data.values()),
                    len(data),
                )

        display_name = (
            f"{model_name} ({pangu_think_mode})" if pangu_think_mode is not None else model_name
        )

        if agg_mode is not None:
            display_name = f"{display_name}[{'L' if agg_mode == 'loose' else 'S'}]"

        row = [
            display_name,
            dataset,
            prompt_type,
            np.mean(data_scores),
            np.std(data_scores),
            num_refined_samples,
            num_samples,
            metrics["avg_token_len"],
        ]
        return row, data_scores

    all_models = []
    df_data = []
    t_test_info = {}

    for model_name, prompt_type in model_name_list:
        for think_mode in pangu_think_mode_list if prompt_type == "pangu" else [None]:
            for dataset in dataset_list:
                for agg_mode in agg_mode_list if dataset.startswith("var_") else [None]:
                    tmp_row, scores = process_eval(
                        dataset, model_name, prompt_type, agg_mode, think_mode
                    )
                    df_data.append(tmp_row)
                    if not dataset.startswith("var_"):
                        all_models.append(tmp_row[0])
                    t_test_info.setdefault(tmp_row[0], {})[dataset] = scores

    df = pd.DataFrame(
        df_data,
        columns=[
            "model_name",
            "dataset",
            "prompt_type",
            "score",
            "score_std",
            "num_refined_samples",
            "num_samples",
            "token_len",
        ],
    ).set_index("model_name")
    df["score"] = df["score"].apply(lambda x: round(x * 100, 1))
    df["score_std"] = df["score_std"].apply(lambda x: round(x * 100, 1))
    df["token_len"] = df["token_len"].apply(lambda x: int(round(x)))
    print("============ Detail infos ============")
    print(df)

    var_dataset = dataset_list[0]
    ori_dataset = dataset_list[1]

    new_data_list = []
    for model in all_models:
        for agg in ["L", "S"]:
            var_model = f"{model}[{agg}]"

            ori_score, ori_score_std = df.loc[model, ["score", "score_std"]]
            var_score, var_score_std = df.loc[var_model, ["score", "score_std"]]
            drop = round(ori_score - var_score, 1)
            drop_ratio = round(drop / ori_score * 100, 1) if ori_score > 0 else 0.0
            ori_tokens = int(round(df.loc[model, "token_len"]))
            var_tokens = int(round(df.loc[var_model, "token_len"]))

            new_data = {
                "Model": var_model,
                ("Dataset", ori_dataset): f"{ori_score} ({ori_score_std})",
                ("Dataset", var_dataset): f"{var_score} ({var_score_std})",
                "Drop": f"{drop} ({drop_ratio}%)",
                ("Tokens", ori_dataset): ori_tokens,
                ("Tokens", var_dataset): var_tokens,
            }

            if t_test:
                orig_scores = t_test_info[model][ori_dataset]
                var_scores = t_test_info[var_model][var_dataset]
                t_statistic, p_value = stats.ttest_ind(
                    var_scores, orig_scores, alternative="less"
                )
                new_data["p_value"] = format_p_value(p_value)

            new_data_list.append(new_data)

    df_new = pd.DataFrame(new_data_list)

    new_columns = []
    for col in df_new.columns:
        if isinstance(col, tuple):
            new_columns.append(col)
        else:
            new_columns.append((col, ""))

    # 3. 重新赋值为 MultiIndex
    df_new.columns = pd.MultiIndex.from_tuples(new_columns)
    print("============ Results Table ============")
    print(df_new)
