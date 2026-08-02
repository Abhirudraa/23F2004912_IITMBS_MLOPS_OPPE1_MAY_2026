# AI Usage Documentation

## AI Tools Utilized and Conversation History
> List all GenAI / LLM tools used during the exam  
> Provide **public share links** to AI chats or attach conversation files if links are not available  

- Tool Name
    - Purpose : to get command and solve errors .. 
    - Shared Chat Link : https://chatgpt.com/share/6a6f4ee5-e2fc-83ee-88bd-8ee0cebacfbc 
                          https://chatgpt.com/share/6a6f4f35-6188-83e8-9f3e-c9bde702efcd
    - 
    - Notes (optional) :

*If a tool does not support link sharing, please specify and paste prompt-response logs in Section 3.*

---

## 3️⃣ Prompts and Responses Used
> Include **all prompts** that contributed to solving the exam tasks
>
> fix this large file error 

base) jupyter@instance-20260801-182812:~$ ls

23F2004912_IITMBS_MLOPS_OPPE1_MAY_2026  OPPE1_setup.ipynb

(base) jupyter@instance-20260801-182812:~$ cd 23F2004912_IITMBS_MLOPS_OPPE1_MAY_2026 

(base) jupyter@instance-20260801-182812:~/23F2004912_IITMBS_MLOPS_OPPE1_MAY_2026$ git status

On branch main

Your branch is up to date with 'origin/main'.



Untracked files:

  (use "git add <file>..." to include in what will be committed)

        data/

        models/

        scripts/



nothing added to commit but untracked files present (use "git add" to track)

(base) jupyter@instance-20260801-182812:~/23F2004912_IITMBS_MLOPS_OPPE1_MAY_2026$ git diff --stat

(base) jupyter@instance-20260801-182812:~/23F2004912_IITMBS_MLOPS_OPPE1_MAY_2026$ git ls-files | grep parquet

feature_repo/data/features.parquet

(base) jupyter@instance-20260801-182812:~/23F2004912_IITMBS_MLOPS_OPPE1_MAY_2026$ git ls-files | grep db

feature_repo/data/online_store.db

feature_repo/data/registry.db

(base) jupyter@instance-20260801-182812:~/23F2004912_IITMBS_MLOPS_OPPE1_MAY_2026$ du -sh .

du -sh feature_repo/data

du -sh models

du -sh data

8.8G    .

44K     feature_repo/data

7.8G    models

90M     data

(base) jupyter@instance-20260801-182812:~/23F2004912_IITMBS_MLOPS_OPPE1_MAY_2026$ ps -ef | grep git

jupyter    36231   19178 90 12:30 pts/0    00:03:15 git add .

jupyter    37688   37315  0 12:34 pts/2    00:00:00 grep git

(base) jupyter@instance-20260801-182812:~/23F2004912_IITMBS_MLOPS_OPPE1_MAY_2026$ du -sh models/*

2.3G    models/model_v0.joblib

5.5G    models/model_v1.joblib

(base) jupyter@instance-20260801-182812:~/23F2004912_IITMBS_MLOPS_OPPE1_MAY_2026$ find models -type f -exec ls -lh {} \; | sort -k5 -h | tail -20

-rw-r--r-- 1 jupyter jupyter 2.3G Aug  2 12:08 models/model_v0.joblib

-rw-r--r-- 1 jupyter jupyter 5.5G Aug  2 12:28 models/model_v1.joblib

(base) jupyter@instance-20260801-182812:~/23F2004912_IITMBS_MLOPS_OPPE1_MAY_2026$ python - <<'EOF'

import joblib



obj = joblib.load("models/model_v0.joblib")



print(type(obj))



if hasattr(obj, "__dict__"):

    print(obj.__dict__.keys())

EOF

<class 'sklearn.ensemble._forest.RandomForestClassifier'>

dict_keys(['estimator', 'n_estimators', 'estimator_params', 'bootstrap', 'oob_score', 'n_jobs', 'random_state', 'verbose', 'warm_start', 'class_weight', 'max_samples', 'criterion', 'max_depth', 'min_samples_split', 'min_samples_leaf', 'min_weight_fraction_leaf', 'max_features', 'max_leaf_nodes', 'min_impurity_decrease', 'monotonic_cst', 'ccp_alpha', 'feature_names_in_', 'n_features_in_', '_n_samples', 'n_outputs_', 'classes_', 'n_classes_', '_n_samples_bootstrap', 'estimator_', 'estimators_'])

(base) jupyter@instance-20260801-182812:~/23F2004912_IITMBS_MLOPS_OPPE1_MAY_2026$ grep -n "joblib.dump" scripts/train.py

86:joblib.dump(model, args.model_out)

(base) jupyter@instance-20260801-182812:~/23F2004912_IITMBS_MLOPS_OPPE1_MAY_2026$ grep -n "RandomForestClassifier" scripts/train.py

6:from sklearn.ensemble import RandomForestClassifier

69:model = RandomForestClassifier(

(base) jupyter@instance-20260801-182812:~/23F2004912_IITMBS_MLOPS_OPPE1_MAY_2026$ cat scripts/train.py

import os

import argparse

import joblib

import pandas as pd



from sklearn.ensemble import RandomForestClassifier

from sklearn.model_selection import train_test_split

from sklearn.metrics import (

    accuracy_score,

    precision_score,

    recall_score,

    f1_score,

)



parser = argparse.ArgumentParser()

parser.add_argument("--data_dir", required=True)

parser.add_argument("--model_out", required=True)

args = parser.parse_args()



all_data = []



for file in os.listdir(args.data_dir):

    if not file.endswith(".csv"):

        continue



    df = pd.read_csv(os.path.join(args.data_dir, file))



    df["rolling_avg_10"] = (

        df["close"]

        .rolling(window=10, min_periods=1)

        .mean()

    )



    df["volume_sum_10"] = (

        df["volume"]

        .rolling(window=10, min_periods=1)

        .sum()

    )



    df["target"] = (

        df["close"].shift(-5) > df["close"]

    ).astype(int)



    df = df.iloc[:-5]



    all_data.append(

        df[

            [

                "rolling_avg_10",

                "volume_sum_10",

                "target",

            ]

        ]

    )



dataset = pd.concat(all_data, ignore_index=True)



X = dataset[["rolling_avg_10", "volume_sum_10"]]

y = dataset["target"]



X_train, X_test, y_train, y_test = train_test_split(

    X,

    y,

    test_size=0.2,

    random_state=42,

    stratify=y,

)



model = RandomForestClassifier(

    n_estimators=100,

    random_state=42,

)



model.fit(X_train, y_train)



pred = model.predict(X_test)



print("\nEvaluation Metrics")

print("------------------")

print("Accuracy :", accuracy_score(y_test, pred))

print("Precision:", precision_score(y_test, pred))

print("Recall   :", recall_score(y_test, pred))

print("F1 Score :", f1_score(y_test, pred))



os.makedirs(os.path.dirname(args.model_out), exist_ok=True)

joblib.dump(model, args.model_out)



print(f"\nModel saved to {args.model_out}")

(base) jupyter@instance-20260801-182812:~/23F2004912_IITMBS_MLOPS_OPPE1_MAY_2026$ 



print("Total nodes:", total_nodes)

EOF

Trees: 100

Depth of first tree: 120

Nodes in first tree: 735121

Total nodes: 73560726 



any its timed exam .. what is the solution  ... now 

Deliverable 4: Training & Evaluation — 2 Incremental Iterations (2 Marks)

Execute training and evaluation scripts that produce valid predictions across two incremental data iterations, demonstrating how the model responds as more data becomes available.



Iteration 1: Train and evaluate on v0 data only.

Iteration 2: Retrain and evaluate on the merged v0 + v1 data.

Use a proper train/test split and report evaluation metrics for each iteration.

Deliverable 5: Hyperparameter Tuning & Experiment Tracking with MLflow (2 Marks)

Run a hyperparameter tuning sweep and track every run with MLflow, so runs can be compared and the best model promoted.



Log parameters, metrics, and artifacts for each tuning run to MLflow Tracking.

Select the best run based on your chosen evaluation metric.

Register the best model to the MLflow Model Registry so CI can fetch it by name.



should i oush it with model.joblib .. then we can do fast inference on subset or .. during task 5 then CI using that 

We will optimize it.



Instead of training 10–20 models on the full dataset, we'll do something like:



sample_frac = 0.2



or



sample(n=20000)



and only search



n_estimators = {50,100}

max_depth = {4,6}



That's 4 runs.



Each run should take around 20–40 seconds, so the whole MLflow task should finish in 2–3 minutes instead of 20+ minutes.



The deliverable only requires:



multiple tracked runs,

parameter logging,

metric logging,

selecting the best model,

registering it.



It does not require exhaustive tuning.

ok i am here right now 

(base) jupyter@instance-20260801-182812:~/23F2004912_IITMBS_MLOPS_OPPE1_MAY_2026$ ls

AI_USAGE_DOC_TEMPLATE.md  LICENSE  README.md  StockAnalyticaData  data  data_processing_reference.ipynb  feature_repo  models  reports  requirements.txt  scripts  tests

(base) jupyter@instance-20260801-182812:~/23F2004912_IITMBS_MLOPS_OPPE1_MAY_2026$ tree

.

├── AI_USAGE_DOC_TEMPLATE.md

├── LICENSE

├── README.md

├── StockAnalyticaData

│   ├── v0

│   │   ├── AARTIIND__EQ__NSE__NSE__MINUTE.csv

│   │   └── ABCAPITAL__EQ__NSE__NSE__MINUTE.csv

│   ├── v0.dvc

│   ├── v1

│   │   ├── ABFRL__EQ__NSE__NSE__MINUTE.csv

│   │   ├── ADANIENT__EQ__NSE__NSE__MINUTE.csv

│   │   └── ADANIGAS__EQ__NSE__NSE__MINUTE.csv

│   └── v1.dvc

├── data

│   └── merged

│       ├── AARTIIND__EQ__NSE__NSE__MINUTE.csv

│       ├── ABCAPITAL__EQ__NSE__NSE__MINUTE.csv

│       ├── ABFRL__EQ__NSE__NSE__MINUTE.csv

│       ├── ADANIENT__EQ__NSE__NSE__MINUTE.csv

│       └── ADANIGAS__EQ__NSE__NSE__MINUTE.csv

├── data_processing_reference.ipynb

├── feature_repo

│   ├── data

│   │   ├── features.parquet

│   │   ├── online_store.db

│   │   └── registry.db

│   ├── feature_definitions.py

│   ├── feature_store.yaml

│   ├── pit_training.py

│   ├── prepare_features.py

│   └── test_features.py

├── models

│   ├── model_v0.joblib

│   └── model_v1.joblib

├── reports

├── requirements.txt

├── scripts

│   └── train.py

└── tests



12 directories, 28 files

(base) jupyter@instance-20260801-182812:~/23F2004912_IITMBS_MLOPS_OPPE1_MAY_2026$  .. psh this 

ok  now task 5 and 6 are

Deliverable 5: Hyperparameter Tuning & Experiment Tracking with MLflow (2 Marks)

Run a hyperparameter tuning sweep and track every run with MLflow, so runs can be compared and the best model promoted.



Log parameters, metrics, and artifacts for each tuning run to MLflow Tracking.

Select the best run based on your chosen evaluation metric.

Register the best model to the MLflow Model Registry so CI can fetch it by name.

Deliverable 6: CI on Main Branch with CML Report Generation (2 Marks)

Configure a Continuous Integration (CI) workflow that runs on the main branch and executes the following pipeline on each trigger:



Fetch the best model from the MLflow Model Registry.

Pull the versioned test data from DVC (the Git-pinned snapshot, ensuring a reproducible evaluation).

Run predictions and compute evaluation metrics on the test data.

Include at least one sanity test per feature to validate each feature against the raw data.

Generate a CML report and post the metrics and plots as a comment on the GitHub PR.

(base) jupyter@instance-20260801-182812:~/23F2004912_IITMBS_MLOPS_OPPE1_MAY_2026$ git status

On branch main

Your branch is up to date with 'origin/main'.



Untracked files:

  (use "git add <file>..." to include in what will be committed)

        mlflow.db

        scripts/tune_mlflow.py



nothing added to commit but untracked files present (use "git add" to track)

(base) jupyter@instance-20260801-182812:~/23F2004912_IITMBS_MLOPS_OPPE1_MAY_2026$

2026/08/02 13:42:10 INFO:     Stopping parent process [55086]

(base) jupyter@instance-20260801-182812:~/23F2004912_IITMBS_MLOPS_OPPE1_MAY_2026$ python - <<EOF

import mlflow



client = mlflow.MlflowClient()



v = client.get_latest_versions("stock_rf")[0]



print("Source:", v.source)

print("Run ID:", v.run_id)

EOF

<stdin>:5: FutureWarning: ``mlflow.tracking.client.MlflowClient.get_latest_versions`` is deprecated since 2.9.0. Model registry stages will be removed in a future major release. To learn more about the deprecation of model registry stages, see our migration guide here: https://mlflow.org/docs/latest/model-registry.html#migrating-from-stages

Source: models:/m-5d5dccc6242f4a8e8a0b0e7ff34392a2

Run ID: 1b24e172ae7b42a5a5fccec448da6bb2

(base) jupyter@instance-20260801-182812:~/23F2004912_IITMBS_MLOPS_OPPE1_MAY_2026$  

> Include **all responses** in case of public share links are not available to share

### Tool Name #1: ___CHATGPT_________
- Prompt 1:
    - Response Log:
---

### Tool Name #2: ___________________
- Prompt 1:
    - Response Log:
