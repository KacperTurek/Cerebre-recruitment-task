import itertools
import os
import pandas as pd
from ultralytics import YOLO
from pathlib import Path

param_grid = {
    'lr0': [0.01, 0.001],
    'momentum': [0.9, 0.937],
    'weight_decay': [0.0005, 0.0001],
    'batch': [16, 32],
}

def train_with_params(run_id, params):
    model = YOLO("yolov8n.pt")
    run_name = f"grid_run_{run_id}"

    print(f"Training {run_name} with params: {params}")
    model.train(
        data=Path("../data/data.yaml").resolve(),
        epochs=20,
        imgsz=416,
        batch=params['batch'],
        lr0=params['lr0'],
        momentum=params['momentum'],
        weight_decay=params['weight_decay'],
        project="runs/grid_search",
        name=run_name,
        verbose=False
    )

    results_path = f"runs/grid_search/{run_name}/results.csv"
    if os.path.exists(results_path):
        df = pd.read_csv(results_path)
        final_map = df["metrics/mAP50(B)"].iloc[-1]
    else:
        final_map = None

    return {**params, "run_name": run_name, "mAP50": final_map}

def main():
    param_combos = list(itertools.product(*param_grid.values()))
    param_keys = list(param_grid.keys())

    results = []
    for i, combo in enumerate(param_combos):
        params = dict(zip(param_keys, combo))
        result = train_with_params(i, params)
        results.append(result)

    df = pd.DataFrame(results).sort_values(by="mAP50", ascending=False)
    df.to_csv("grid_search_results.csv", index=False)

    print("Grid Search Complete. Top Results:")
    print(df.head())

if __name__ == "__main__":
    main()
