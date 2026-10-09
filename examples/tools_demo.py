"""Call every tool directly - no LLM involved. Shows exactly what the agent receives.

    python examples/tools_demo.py
"""

import json

from data_agent import (call_dataframe_method, evaluate_classification_dataset, evaluate_regression_dataset,
                        get_dataset_summaries, list_csv_files, preload_datasets)

print("list_csv_files →", list_csv_files.invoke({}))
print("preload_datasets →", preload_datasets.invoke({"paths": ["house-prices.csv", "customer-churn.csv"]}))

summary = get_dataset_summaries.invoke({"dataset_paths": ["customer-churn.csv"]})[0]
print(f"\nget_dataset_summaries → {summary['file_name']}: {summary['rows']} rows")
for col in summary["columns"]:
    print(f"   {col['name']:18} dtype={col['dtype']:8} n_unique={col['n_unique']:<4} missing={col['missing']}")

print("\ncall_dataframe_method(house-prices.csv, 'head') →")
print(call_dataframe_method.invoke({"file_name": "house-prices.csv", "method": "head"}))
print("\ncall_dataframe_method(house-prices.csv, 'to_csv') →",
      call_dataframe_method.invoke({"file_name": "house-prices.csv", "method": "to_csv"}))

print("\nevaluate_classification_dataset(customer-churn.csv, 'churned') →")
print(json.dumps(evaluate_classification_dataset.invoke({"file_name": "customer-churn.csv", "target_column": "churned"}), indent=2))
print("\nevaluate_regression_dataset(house-prices.csv, 'price_k') →")
print(json.dumps(evaluate_regression_dataset.invoke({"file_name": "house-prices.csv", "target_column": "price_k"}), indent=2))
print("\nevaluate_regression_dataset(house-prices.csv, 'colour') →",
      evaluate_regression_dataset.invoke({"file_name": "house-prices.csv", "target_column": "colour"}))
