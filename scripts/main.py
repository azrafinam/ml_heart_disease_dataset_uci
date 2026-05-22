"""
Command-line pipeline for the UCI Heart Disease project.

This script mirrors the notebook workflow for both target formulations:
1. multiclass severity on num (0-4),
2. binary presence on num (0=no disease, 1=disease with 1-4 collapsed to 1).
"""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

import joblib
import pandas as pd

from utils import (
    RANDOM_STATE,
    TARGET_COLUMN,
    TARGET_MODE_DESCRIPTIONS,
    TargetMode,
    build_model_zoo,
    create_model_comparison_report,
    ensure_output_dirs,
    evaluate_classifier,
    get_best_model,
    load_data,
    prepare_train_test_data,
    save_json,
    save_model,
)

DATA_PATH = Path("data/heart_disease_uci.csv")


def _artifact_suffix(target_mode: TargetMode) -> str:
    return "" if target_mode == "multiclass" else "_binary"


def run_pipeline(df: pd.DataFrame, target_mode: TargetMode) -> dict[str, str]:
    """Train, evaluate, and persist artifacts for one target formulation."""
    suffix = _artifact_suffix(target_mode)
    mode_label = target_mode.upper()
    target_description = TARGET_MODE_DESCRIPTIONS[target_mode]

    print(f"\n{'=' * 72}")
    print(f"TARGET MODE: {mode_label} ({target_description})")
    print(f"{'=' * 72}")

    print("\nPreparing leakage-safe train/test data")
    prepared = prepare_train_test_data(df, random_state=RANDOM_STATE, target_mode=target_mode)
    X_train = prepared["X_train"]
    X_test = prepared["X_test"]
    y_train = prepared["y_train"]
    y_test = prepared["y_test"]
    preprocessor = prepared["preprocessor"]

    X_train.to_csv(f"data/processed/X_train_processed{suffix}.csv", index=False)
    X_test.to_csv(f"data/processed/X_test_processed{suffix}.csv", index=False)
    y_train.to_csv(f"data/processed/y_train{suffix}.csv", index=False)
    y_test.to_csv(f"data/processed/y_test{suffix}.csv", index=False)
    pd.concat([X_train, X_test], ignore_index=True).to_csv(
        f"data/processed/X_processed{suffix}.csv",
        index=False,
    )
    pd.concat([y_train, y_test], ignore_index=True).to_csv(
        f"data/processed/y_processed{suffix}.csv",
        index=False,
    )
    save_model(preprocessor, f"outputs/models/preprocessor{suffix}.pkl")

    print(f"  Train shape: {X_train.shape}")
    print(f"  Test shape : {X_test.shape}")
    print(f"  Train target distribution:\n{y_train.value_counts().sort_index().to_string()}")

    print("\nDefining models")
    models = build_model_zoo(random_state=RANDOM_STATE, target_mode=target_mode)
    print(f"  Models defined: {len(models)}")

    print("\nTraining models")
    trained_models = {}
    training_log = []
    model_manifest = {}

    for model_name, model in models.items():
        try:
            print(f"  Training {model_name}...", end=" ")
            model.fit(X_train, y_train)
            trained_models[model_name] = model
            model_path = f"outputs/models/{model_name}_trained{suffix}.pkl"
            save_model(model, model_path)
            model_manifest[model_name] = model_path
            training_log.append(
                {
                    "model": model_name,
                    "status": "success",
                    "timestamp": datetime.now().isoformat(),
                }
            )
            print("done")
        except Exception as exc:
            training_log.append(
                {
                    "model": model_name,
                    "status": "failed",
                    "error": str(exc),
                    "timestamp": datetime.now().isoformat(),
                }
            )
            print(f"failed: {str(exc)[:80]}")

    save_json(model_manifest, f"outputs/reports/day4_model_manifest{suffix}.json")
    save_json(
        {
            "training_date": datetime.now().isoformat(),
            "target_mode": target_mode,
            "target_description": target_description,
            "total_models_defined": len(models),
            "total_models_trained": len(trained_models),
            "models_trained": list(trained_models.keys()),
            "training_log": training_log,
            "data_split": {
                "train_samples": int(X_train.shape[0]),
                "test_samples": int(X_test.shape[0]),
                "features": int(X_train.shape[1]),
                "random_state": RANDOM_STATE,
                "split_before_preprocessing": True,
            },
        },
        f"outputs/reports/day4_training_summary{suffix}.json",
    )

    print("\nEvaluating models")
    evaluation_results = {}
    for model_name, model in trained_models.items():
        try:
            print(f"  Evaluating {model_name}...", end=" ")
            evaluation_results[model_name] = evaluate_classifier(model, X_test, y_test)
            print("done")
        except Exception as exc:
            print(f"failed: {str(exc)[:80]}")

    comparison_df = create_model_comparison_report(evaluation_results)
    comparison_path = f"outputs/reports/model_comparison_table{suffix}.csv"
    comparison_df.to_csv(comparison_path)
    print("\nModel comparison")
    print(comparison_df.to_string())

    best_model_name, best_f1 = get_best_model(comparison_df, metric="f1_score")
    best_model = trained_models[best_model_name]
    best_metrics = comparison_df.loc[best_model_name].to_dict()
    print(f"\nBest model ({target_mode}): {best_model_name} | weighted F1: {best_f1:.4f}")

    evaluation_report = {
        "evaluation_date": datetime.now().isoformat(),
        "target_mode": target_mode,
        "target_description": target_description,
        "models_evaluated": len(evaluation_results),
        "model_list": list(evaluation_results.keys()),
        "best_model": best_model_name,
        "best_model_metrics": {k: float(v) for k, v in best_metrics.items()},
        "all_model_metrics": {
            model: {
                k: (None if pd.isna(v) else float(v))
                for k, v in metrics.items()
                if k != "confusion_matrix"
            }
            for model, metrics in evaluation_results.items()
        },
    }
    save_json(evaluation_report, f"outputs/reports/day5_evaluation_report{suffix}.json")

    final_bundle = {
        "model": best_model,
        "preprocessor": preprocessor,
        "model_name": best_model_name,
        "feature_names": prepared["feature_names"],
        "numeric_features": prepared["numeric_features"],
        "categorical_features": prepared["categorical_features"],
        "target_column": TARGET_COLUMN,
        "target_mode": target_mode,
        "target_description": target_description,
        "class_labels": sorted(y_train.unique().tolist()),
    }
    final_model_path = f"outputs/models/{best_model_name}_final_bundle{suffix}.pkl"
    joblib.dump(final_bundle, final_model_path)

    metadata = {
        "model_name": best_model_name,
        "model_version": "1.0",
        "creation_date": datetime.now().isoformat(),
        "performance_metrics": {k: float(v) for k, v in best_metrics.items()},
        "file_path": final_model_path,
        "dataset": "UCI Heart Disease",
        "target_mode": target_mode,
        "target": f"num ({target_description})",
        "preprocessing": "Saved preprocessor fitted on training data only",
        "models_compared": len(evaluation_results),
    }
    save_json(metadata, f"outputs/models/{best_model_name}_metadata{suffix}.json")

    usage_guide = f"""# Model Usage Guide ({target_mode})

Load the final bundle and pass raw feature rows with the original feature columns
except `id` and `num`.

```python
import joblib
import pandas as pd

bundle = joblib.load("{final_model_path}")
preprocessor = bundle["preprocessor"]
model = bundle["model"]

raw_rows = pd.DataFrame([...])
X_new_array = preprocessor.transform(raw_rows)
X_new = pd.DataFrame(X_new_array, columns=bundle["feature_names"])
predictions = model.predict(X_new)
```
"""
    Path(f"outputs/reports/MODEL_USAGE_GUIDE{suffix}.txt").write_text(usage_guide)

    save_json(
        {
            "project_information": {
                "project_name": "Heart Disease Prediction",
                "dataset": "UCI Heart Disease",
                "generation_date": datetime.now().isoformat(),
            },
            "target_mode": target_mode,
            "target_description": target_description,
            "best_model": metadata,
            "all_models_evaluation": evaluation_report["all_model_metrics"],
            "file_structure": {
                "final_model_bundle": final_model_path,
                "preprocessor": f"outputs/models/preprocessor{suffix}.pkl",
                "comparison_table": comparison_path,
                "evaluation_report": f"outputs/reports/day5_evaluation_report{suffix}.json",
            },
        },
        f"outputs/reports/day6_final_pipeline_report{suffix}.json",
    )

    return {
        "target_mode": target_mode,
        "best_model_name": best_model_name,
        "best_f1": f"{best_f1:.4f}",
        "final_model_path": final_model_path,
        "models_trained": str(len(trained_models)),
    }


def main() -> None:
    print("HEART DISEASE PREDICTION - FULL ML PIPELINE")
    ensure_output_dirs()

    print("\nLoading data")
    df = load_data(DATA_PATH)

    summaries = []
    for target_mode in ("multiclass", "binary"):
        summaries.append(run_pipeline(df, target_mode))

    print("\nPipeline completed successfully")
    for summary in summaries:
        print(
            f"  [{summary['target_mode']}] models trained: {summary['models_trained']} | "
            f"best: {summary['best_model_name']} | F1: {summary['best_f1']} | "
            f"bundle: {summary['final_model_path']}"
        )


if __name__ == "__main__":
    main()
