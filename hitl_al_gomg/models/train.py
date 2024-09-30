import warnings

warnings.filterwarnings("ignore")

import click
import json
import pickle
import pandas as pd
import numpy as np

from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import (
    r2_score,
    mean_squared_error,
    roc_auc_score,
    f1_score,
    matthews_corrcoef,
)
from sklearn.model_selection import GridSearchCV

from hitl_al_gomg.utils import ecfp_generator


fp_counter = ecfp_generator(radius=3, useCounts=True)


def load_oracle(path_to_simulator):
    print(f"\nLoad oracle")
    return pickle.load(open(f"{path_to_simulator}.pkl", "rb"))


def preprocess_init_data(
    path_to_train_data, path_to_test_data, regression, path_to_simulator
):
    print(f"\nLoad and preprocess data for predictor training")
    train_set = pd.read_csv(f"{path_to_train_data}.csv")
    test_set = pd.read_csv(f"{path_to_test_data}.csv")

    # drop SMILES duplicates
    train_set.drop_duplicates(subset=["SMILES"], inplace=True)
    test_set.drop_duplicates(subset=["SMILES"], inplace=True)

    smiles_train, smiles_test = train_set.SMILES.tolist(), test_set.SMILES.tolist()

    # calculate Morgan fingerprints
    fps_train = fp_counter.get_fingerprints(train_set.SMILES.tolist())
    fps_test = fp_counter.get_fingerprints(test_set.SMILES.tolist())

    print("\nDataset specifications")
    print("\nTrain set size:", train_set.shape)
    print("Test set size:", test_set.shape)
    if not regression:
        print(
            "\nInitial train 0/1 ratio:",
            np.unique(train_set.target, return_counts=True),
        )
        print("Initial test 0/1 ratio:", np.unique(test_set.target, return_counts=True))

    # get ground truth labels from oracle
    oracle = load_oracle(path_to_simulator)
    oracle_labels_train = oracle.predict(fps_train)
    oracle_labels_test = oracle.predict(fps_test)

    if not regression:
        print(
            "\nUpdated train 0/1 ratio:",
            np.unique(oracle_labels_train, return_counts=True),
        )
        print(
            "Updated test 0/1 ratio:", np.unique(oracle_labels_test, return_counts=True)
        )

    train_set["target"] = oracle_labels_train.tolist()
    test_set["target"] = oracle_labels_test.tolist()

    train_set = train_set[["SMILES", "target"]]
    test_set = test_set[["SMILES", "target"]]

    # save updated datasets
    print(f"\nSave datasets")
    train_set.to_csv(f"{path_to_train_data}.csv")
    test_set.to_csv(f"{path_to_test_data}.csv")

    return (
        fps_train,
        fps_test,
        smiles_train,
        smiles_test,
        oracle_labels_train,
        oracle_labels_test,
    )


def hyperparam_opt_train(
    x_train,
    y_train,
    sample_weights,
    save_to_path,
    regression=False,
    path_to_param_grid=None,
):
    if path_to_param_grid:
        print("\nGrid CV")
        with open(path_to_param_grid, "r") as file:
            param_grid = json.load(file)
            print(param_grid)

        if regression:
            rf = RandomForestRegressor(oob_score=True)
            grid_rf = GridSearchCV(
                rf, param_grid, scoring="neg_mean_squared_error", cv=5
            )
        else:
            rf = RandomForestClassifier(oob_score=True)
            grid_rf = GridSearchCV(rf, param_grid, scoring="f1_weighted", cv=5)
        grid_rf.fit(x_train, y_train, sample_weight=sample_weights)

        print("\nHyperparameter search completed")
        print(grid_rf.cv_results_)

        print("\nFinal predictor")
        print(grid_rf.best_estimator_)

        print(f"\nSave final predictor to {save_to_path}")
        pickle.dump(grid_rf.best_estimator_, open(save_to_path, "wb"))

        return grid_rf.best_estimator_
    else:
        print("\nTraining without hyperparameter grid search")
        if regression:
            rf = RandomForestRegressor(oob_score=True, n_estimators=300, max_depth=20)
        else:
            rf = RandomForestClassifier(oob_score=True, n_estimators=300, max_depth=20)
        rf.fit(x_train, y_train, sample_weight=sample_weights)

        print(f"\nSave final predictor to {save_to_path}")
        pickle.dump(rf, open(save_to_path, "wb"))

        return rf


def eval(model, x_test, y_test, regression=False):
    print(model)
    print("\nOOB score", model.oob_score_)
    if regression:
        print("\nR2", r2_score(y_test, model.predict(x_test)))
        print("\nMSE", mean_squared_error(y_test, model.predict(x_test)))
    else:
        print("\nAUC", roc_auc_score(y_test, model.predict(x_test)))
        print("\nF1", f1_score(y_test, model.predict(x_test)))
        print("\nMCC", matthews_corrcoef(y_test, model.predict(x_test)))


@click.command()
@click.option(
    "--regression",
    type=bool,
    default=False,
    help="Whether it is a regression or classification task",
)
@click.option(
    "--path_to_train_data",
    type=str,
    help="Name of csv file where to store the processed training set (without .csv extension)",
)
@click.option(
    "--path_to_test_data",
    type=str,
    help="Name of csv file where to store the processed test set (without .csv extension)",
)
@click.option(
    "--path_to_predictor",
    type=str,
    help="Name of pickle file where to store the trained predictor (without .pkl extension)",
)
@click.option(
    "--path_to_simulator",
    type=str,
    help="Path to oracle or assay simulator (without .pkl extension)",
)
@click.option(
    "--path_to_param_grid",
    type=str,
    help="Path to JSON file containing hyperparameter grid for performing grid search cross-validation",
)
@click.option(
    "--train",
    type=bool,
    default=False,
    help="Whether to train and evaluate or only to evaluate the model",
)
@click.option(
    "--demo",
    type=bool,
    default=False,
    help="Whether to run a quick demo with the expert model to test the initial level of agreement with the predictor",
)
def main(
    regression,
    path_to_train_data,
    path_to_test_data,
    path_to_predictor,
    path_to_simulator,
    path_to_param_grid,
    train,
    demo,
):
    print(f"\nSave trained model to {path_to_predictor}.pkl")
    save_to_path = f"{path_to_predictor}.pkl"
    oracle = load_oracle(path_to_simulator)

    (
        fps_train,
        fps_test,
        smiles_train,
        smiles_test,
        oracle_labels_train,
        oracle_labels_test,
    ) = preprocess_init_data(
        path_to_train_data, path_to_test_data, regression, path_to_simulator
    )

    if train:
        # search for optimal hyperparameters and pre-train ML property predictor
        print("\nHyperparameter grid search and training ML predictor")
        # grid example for classification task:
        # param_grid_clf = {
        #    'n_estimators': [10, 50, 100, 300],
        #    'max_depth': [None, 2, 5, 10],
        #    'criterion': ['gini', 'entropy'],
        #    'min_samples_split': [2, 5, 10]
        #    }
        # grid example for regression task:
        # param_grid_reg = {
        #    'n_estimators': [10, 50, 100, 300],
        #    'max_depth': [None, 2, 5, 10],
        #    'min_samples_split': [2, 5, 10]
        #    }
        ml_predictor = hyperparam_opt_train(
            fps_train,
            oracle_labels_train.ravel(),
            None,
            save_to_path,
            regression=regression,
            path_to_param_grid=path_to_param_grid,
        )
        eval(ml_predictor, fps_test, oracle_labels_test.ravel(), regression=regression)
    else:
        try:
            print("\nLoading pre-trained ML predictor")
            ml_predictor = pickle.load(open(save_to_path, "rb"))
            print(ml_predictor)
            eval(
                ml_predictor,
                fps_test,
                oracle_labels_test.ravel(),
                regression=regression,
            )
        except:
            ValueError("No existing model in the specified path.")

    if demo:

        print("\nRunning quick demo with expert model")
        sigma_noise = float(0.15)
        print("\nNoise term will be sampled from Normal(0, sigma=0.15)")

        def expert(smi, oracle, sigma_noise):
            noise = np.random.normal(0, sigma_noise, 1).item()
            y_oracle = oracle.predict_proba(fp_counter.get_fingerprints([smi]))[
                :, 1
            ].item()
            return np.clip(y_oracle + noise, 0, 1)

        print("\nExpert demo")
        for i in range(5):
            print(
                "Test labels:",
                oracle_labels_test[i],
                "ML predicted proba:",
                ml_predictor.predict(fps_test[i].reshape(1, -1)).item(),
                "Expert scores:",
                expert(smiles_test[i], oracle, sigma_noise),
            )


if __name__ == "__main__":
    main()
