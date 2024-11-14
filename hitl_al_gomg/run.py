# load dependencies
import click
import pickle
import os
import subprocess
import shutil
import json
import pandas as pd
import numpy as np

from hitl_al_gomg.synthitl.simulated_expert import EvaluationModel, utility
from hitl_al_gomg.scoring.write import write_REINVENT_config
from hitl_al_gomg.synthitl.acquisition import select_query

from hitl_al_gomg.models.RandomForest import RandomForestClf, RandomForestReg
from hitl_al_gomg.utils import ecfp_generator

from hitl_al_gomg.path import priors

fp_counter = ecfp_generator(radius=3, useCounts=True)


def load_training_set(path_to_train_data):
    # Load the data used to train the initial target property predictor
    train_set = pd.read_csv(f"{path_to_train_data}.csv")
    # Get samples
    smiles_train = train_set["SMILES"].values
    # Get labels
    labels = train_set["target"].values.astype(int)
    # Compute molecular features
    fingerprints = fp_counter.get_fingerprints(smiles_train)
    print("\nDataset specifications: ")
    print("\t# Compounds:", labels.shape[0])
    print("\t# Features:", fingerprints.shape[1])
    x_train, y_train = fingerprints, labels
    return smiles_train, x_train, y_train


def load_model(path_to_scoring_model):
    # Load the initial target property predictor
    print("\nLoading target property predictor")
    prop_predictor = pickle.load(open(path_to_scoring_model, "rb"))
    return prop_predictor


def copy_model(path_to_scoring_model, output_folder, iter):
    # Save the initial target property predictor to the root output folder
    scoring_model_name = path_to_scoring_model.split("/")[-1].split(".pkl")[0]
    model_load_path = f"{output_folder}/{scoring_model_name}_iteration_{iter}.pkl"
    if iter == 0 and not os.path.exists(model_load_path):
        shutil.copy(f"{path_to_scoring_model}.pkl", output_folder)
        os.rename(f"{output_folder}/{scoring_model_name}.pkl", model_load_path)
    return scoring_model_name, model_load_path


def update_reinvent_config_file(
    output_folder,
    num_opt_steps,
    path_to_scoring_model,
    smiles_train,
    model_type,
    task,
    scoring_component_name,
    train_similarity=False,
    pretrained_prior=False,
    multi_objectives=False,
    path_to_herg_simulator=None,
    iter=0,
):
    if iter == 0:
        configuration = json.load(open(os.path.join(output_folder, "config.json")))
    else:
        configuration = json.load(
            open(os.path.join(output_folder, f"config_iteration{iter}.json"))
        )
    # write specified number of REINVENT optimization steps in configuration file
    # (example: if R = 5 (rounds) and Reinvent opt_steps = 100, we will run 5*100 RL optimization steps)
    configuration["parameters"]["reinforcement_learning"]["n_steps"] = num_opt_steps
    # Write initial scoring predictor path in configuration file
    configuration_scoring_function = configuration["parameters"]["scoring_function"][
        "parameters"
    ]
    for i in range(len(configuration_scoring_function)):
        if configuration_scoring_function[i]["component_type"] == "predictive_property":
            configuration_scoring_function[i]["name"] = scoring_component_name
            configuration_scoring_function[i]["specific_parameters"][
                "scikit"
            ] = model_type
            if configuration_scoring_function[i]["name"] == scoring_component_name:
                # Only target property predictor is updated through AL, do not update other predictive components with the same path
                configuration_scoring_function[i]["specific_parameters"][
                    "model_path"
                ] = path_to_scoring_model
            if model_type == "classification":
                configuration_scoring_function[i]["specific_parameters"][
                    "transformation"
                ] = {"transformation_type": "no_transformation"}
            elif model_type == "regression" and task == "logp":
                configuration_scoring_function[i]["specific_parameters"][
                    "transformation"
                ] = {
                    "transformation_type": "double_sigmoid",
                    "high": 4,  # Highest desired LogP value according to the paper experiment (use case 1)
                    "low": 2,  # Lowest desired LogP value according to the paper experiment (use case 1)
                    "coef_div": 3.0,
                    "coef_si": 10,
                    "coef_se": 10,
                }

    if pretrained_prior:
        configuration["parameters"]["reinforcement_learning"][
            "agent"
        ] = f"{priors}/{task}_focused.agent"

    if train_similarity:
        tanimoto_config = {
            "component_type": "tanimoto_similarity",
            "name": "Tanimoto Similarity",
            "weight": 1,
            "specific_parameters": {"smiles": smiles_train.tolist()},
        }

        configuration["parameters"]["scoring_function"]["parameters"].append(
            tanimoto_config
        )

    if multi_objectives:  # According to the paper experiment (multi objective use case)
        qed_config = {
            "component_type": "qed_score",
            "name": "QED",
            "specific_parameters": {
                "transformation": {
                    "low": 0.35,
                    "transformation": True,
                    "transformation_type": "right_step",
                }
            },
            "weight": 0.5,
        }

        herg_config = {
            "component_type": "predictive_property",
            "name": "herg",
            "weight": 1,
            "specific_parameters": {
                "container_type": "scikit_container",
                "model_path": f"{path_to_herg_simulator}.pkl",
                "smiles": "",
                "scikit": "classification",
                "descriptor_type": "ecfp_counts",
                "size": 2048,
                "radius": 3,
                "transformation": {"transformation_type": "flip_probability"},
                "use_counts": False,
                "use_features": False,
            },
        }
        configuration["parameters"]["scoring_function"]["parameters"].append(qed_config)
        configuration["parameters"]["scoring_function"]["parameters"].append(
            herg_config
        )

    # Write the updated REINVENT configuration file to the disc
    if iter == 0:
        configuration_json_path = os.path.join(output_folder, "config.json")
    else:
        configuration_json_path = os.path.join(
            output_folder, f"config_iteration{iter+1}.json"
        )
    with open(configuration_json_path, "w") as f:
        json.dump(configuration, f, indent=4, sort_keys=True)


def run_reinvent(
    path_to_reinvent_env,
    path_to_reinvent_repo,
    acquisition,
    configuration_json_path,
    output_folder,
    iter,
):
    if iter == 0 and acquisition != "None":
        if os.path.exists(
            os.path.join(output_folder, "iteration_0/scaffold_memory.csv")
        ):
            # Load existing scaffold_memory
            print(f"\nLoad REINVENT output, round {iter}")
            data = pd.read_csv(
                os.path.join(output_folder, "iteration_0/scaffold_memory.csv")
            )
            data.reset_index(inplace=True)
        else:
            # Start from scratch
            print(f"\nRun REINVENT, round {iter}")
            command = [
                str(path_to_reinvent_env) + "/bin/python",
                str(path_to_reinvent_repo) + "/input.py",
                str(configuration_json_path),
            ]

            with open(os.path.join(output_folder, "run.err"), "w") as err_file:
                process = subprocess.Popen(
                    command, stdout=subprocess.PIPE, stderr=err_file
                )

                # Wait for the process to finish
                output, errors = process.communicate()

                # Check if there are any errors during execution
                if process.returncode != 0:
                    print(f"Error occurred while running REINVENT: {errors.decode()}")
                else:
                    print(output.decode())  # Print any standard output

            # Read the scaffold_memory CSV
            data = pd.read_csv(
                os.path.join(output_folder, "results/scaffold_memory.csv")
            )

    else:
        # Overwrite existing scaffold_memory
        print(f"\nRun REINVENT, round {iter}")
        command = [
            str(path_to_reinvent_env) + "/bin/python",
            str(path_to_reinvent_repo) + "/input.py",
            str(configuration_json_path),
        ]

        with open(os.path.join(output_folder, "run.err"), "w") as err_file:
            process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=err_file)

            # Wait for the process to finish
            output, errors = process.communicate()

            # Check if there are any errors during execution
            if process.returncode != 0:
                print(f"Error occurred while running REINVENT: {errors.decode()}")
            else:
                print(output.decode())  # Print any standard output

        # Read the scaffold_memory CSV
        data = pd.read_csv(os.path.join(output_folder, "results/scaffold_memory.csv"))

    return data


def prep_pool(data, threshold_value, scoring_component_name):
    bioactivity_score = data[scoring_component_name]
    # Save the indexes of high scoring molecules for satisfying the target property according to predictor
    high_scoring_idx = bioactivity_score > threshold_value
    smiles = data[high_scoring_idx].SMILES.tolist()
    print(f"\n{len(smiles)} high-scoring (> {threshold_value}) molecules")
    return smiles


def active_learning_selection(
    pool,
    smiles,
    selected_feedback,
    n_queries,
    acquisition,
    scoring_model,
    model_type,
    rng,
):
    # Create the fine-tuned predictor (RFC or RFR) object
    if model_type == "regression":
        model = RandomForestReg(scoring_model)
    elif model_type == "classification":
        model = RandomForestClf(scoring_model)
    # Select queries to show to expert
    if len(smiles) >= n_queries:
        new_query = select_query(
            pool,
            n_queries,
            smiles,
            model,
            selected_feedback,
            acquisition=acquisition,
            rng=rng,
        )
    else:
        # Select all high-scoring smiles if their number is less than n_queries
        new_query = select_query(
            pool,
            len(smiles),
            smiles,
            model,
            selected_feedback,
            acquisition=acquisition,
            rng=rng,
        )

    selected_smiles = [smiles[i] for i in new_query]

    # Append selected feedback
    selected_feedback = np.hstack((selected_feedback, new_query))

    return selected_smiles, selected_feedback


def get_expert_feedback(selected_smiles, task, model_type, path_to_simulator, noise):
    if task == "drd2":
        feedback_model = EvaluationModel(
            task, path_to_simulator=f"{path_to_simulator}.pkl"
        )
    else:
        feedback_model = EvaluationModel(task)
    raw_feedback = np.array(
        [feedback_model.human_score(s, noise) for s in selected_smiles]
    )
    if model_type == "regression":
        # Get normalized expert scores from raw feedback (i.e., positive probabilities if classification or continuous values if regression)
        # (On the GUI, the expert directly provides [0,1] scores)
        scores = [utility(f, low=2, high=4) for f in raw_feedback]
        feedback = raw_feedback
    elif model_type == "classification":
        scores = raw_feedback
        feedback = np.round(raw_feedback).astype(int)
    # Get confidence scores based on expert scores
    # (e.g., If the expert estimates a 10% chance that a molecule satisfies the target property, we apply a weight of 1−0.1=0.9 when retraining the predictor.
    # This suggests high confidence in the molecule's inactivity. If the expert estimates a 50% chance, we apply a weight of 0.5.
    # This indicates moderate certainty or balanced likelihood.)
    print(
        f"\nNumber of approved molecules by expert: {np.sum(np.round(scores)).astype(int)}"
    )
    confidences = [s if s > 0.5 else 1 - s for s in scores]
    return feedback, confidences


def augment_train_set(
    x_train,
    y_train,
    sample_weights,
    smiles_train,
    selected_smiles,
    feedback,
    confidences,
    output_folder,
    iter,
    t,
):
    x_new = fp_counter.get_fingerprints(selected_smiles)
    x_train = np.concatenate([x_train, x_new])
    y_train = np.concatenate([y_train, feedback])
    sample_weights = np.concatenate([sample_weights, confidences])
    smiles_train = np.concatenate([smiles_train, selected_smiles])
    print(
        f"\nAugmented train set size at iteration {iter}, {t}: {x_train.shape[0]} {y_train.shape[0]}"
    )
    # Save augmented training data
    D_r = pd.DataFrame(
        np.concatenate([smiles_train.reshape(-1, 1), y_train.reshape(-1, 1)], 1)
    )
    D_r.columns = ["SMILES", "target"]
    D_r.to_csv(os.path.join(output_folder, f"augmented_train_set_iter{iter}.csv"))
    return x_train, y_train, sample_weights, smiles_train


def retrain_model(
    x_train, y_train, sample_weights, prop_predictor, model_type, model_new_savefile
):
    print("\nRetrain model")
    if model_type == "regression":
        model = RandomForestReg(prop_predictor)
    elif model_type == "classification":
        model = RandomForestClf(prop_predictor)
    # Retrain and save the updated predictor
    model._retrain(x_train, y_train, sample_weights, save_to_path=model_new_savefile)


def save_configuration_file(
    output_folder,
    initial_dir,
    conf_filename,
    jobid,
    seed,
    scoring_component_name,
    iter,
    model_new_savefile=None,
):
    # Get initial configuration
    configuration = json.load(open(os.path.join(initial_dir, conf_filename)))
    conf_filename = f"iteration{iter}_config.json"

    # Modify predictor path in configuration using the updated predictor's path
    configuration_scoring_function = configuration["parameters"]["scoring_function"][
        "parameters"
    ]
    for i in range(len(configuration_scoring_function)):
        if (
            model_new_savefile
            and configuration_scoring_function[i]["component_type"]
            == "predictive_property"
            and configuration_scoring_function[i]["name"] == scoring_component_name
        ):
            configuration_scoring_function[i]["specific_parameters"][
                "model_path"
            ] = model_new_savefile

    # Define new directory for the next round
    root_output_dir = os.path.expanduser(f"{jobid}_seed{seed}")
    output_folder = os.path.join(root_output_dir, f"iteration{iter}")
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # Update REINVENT agent checkpoint
    if iter == 1:
        configuration["parameters"]["reinforcement_learning"]["agent"] = os.path.join(
            initial_dir, "results/Agent.ckpt"
        )
    else:
        configuration["parameters"]["reinforcement_learning"]["agent"] = os.path.join(
            os.path.join(root_output_dir, f"iteration{iter-1}"), "results/Agent.ckpt"
        )

    # Modify log and result paths in REINVENT configuration
    configuration["logging"]["logging_path"] = os.path.join(
        output_folder, "progress.log"
    )
    configuration["logging"]["result_folder"] = os.path.join(output_folder, "results")

    # Write the updated configuration file to the disc
    configuration_json_path = os.path.join(output_folder, conf_filename)
    with open(configuration_json_path, "w") as f:
        json.dump(configuration, f, indent=4, sort_keys=True)

    return output_folder, configuration_json_path


@click.command()
@click.option(
    "--path_to_output_dir",
    type=str,
    help="Path to the directory where you wish to store all results",
)
@click.option("--seed", "-s", default=42, type=int, help="Experiment seed")
@click.option("--rounds", "-R", default=4, type=int, help="Number of rounds")
@click.option(
    "--num_opt_steps",
    default=250,
    type=int,
    help="Number of REINVENT optimization steps",
)
@click.option(
    "--path_to_reinvent_env",
    type=str,
    help="Path to python virtual environment for REINVENT V3.2",
)
@click.option(
    "--path_to_reinvent_repo",
    type=str,
    help="Path to the cloned REINVENT V3.2 repository",
)
@click.option(
    "--path_to_scoring_model",
    type=str,
    help="Path to the pickled pretrained property predictor to use for molecule scoring (without the .pkl extension)",
)
@click.option(
    "--model_type",
    type=click.Choice(["regression", "classification"]),
    help="Whether the scoring model is a regressor or classifier",
)
@click.option(
    "--scoring_component_name",
    type=str,
    help="Name given to the predictor component in REINVENT output files",
)
@click.option(
    "--threshold_value",
    "-t",
    default=0.5,
    type=float,
    help="Score threshold value used to select high-scoring generated molecule for active learning",
)
@click.option(
    "--dirname", "-o", type=str, help="Name of output folder to store all results"
)
@click.option(
    "--path_to_train_data",
    type=str,
    help="Path to csv containing initial predictor training data (without .csv extension)",
)
@click.option(
    "--multi_objectives",
    type=bool,
    default=False,
    help="Whether to optimize multiple objectives",
)
@click.option(
    "--path_to_herg_simulator",
    type=str,
    default=None,
    help="Path to the pickled hERG bioactivity simulator used in the multi-objective scoring experiments",
)
@click.option(
    "--train_similarity",
    type=bool,
    default=False,
    help="Whether to optimize Tanimoto similarity in REINVENT with respect to predictor training set",
)
@click.option(
    "--pretrained_prior",
    type=bool,
    default=False,
    help="Whether to use a REINVENT prior agent pre-trained on the predictor training set",
)
@click.option(
    "--al_iterations", "-T", default=5, type=int, help="Number of AL iterations"
)
@click.option(
    "--acquisition",
    "-a",
    type=click.Choice(
        [
            "random",
            "greedy_classification",
            "greedy_regression",
            "epig",
            "uncertainty",
            "entropy",
            "margin",
            "None",
        ]
    ),
    help="Data acquisition method",
)
@click.option(
    "--n_queries",
    "-n",
    default=10,
    help="Number of selected queries to be evaluated by the expert",
)
@click.option(
    "--task",
    type=click.Choice(["logp", "drd2"]),
    help="Goal of the molecule generation",
)
@click.option(
    "--path_to_simulator",
    type=str,
    help="Path to oracle or assay simulator used to assess the predictor accuracy (without .pkl extension)",
)
@click.option(
    "--noise",
    default=0.0,
    type=float,
    help="Sigma value for the noise term used in the expert model (i.e., output from the oracle + noise term)",
)
def main(
    path_to_output_dir,
    seed,
    rounds,
    num_opt_steps,
    path_to_reinvent_env,
    path_to_reinvent_repo,
    path_to_scoring_model,
    model_type,
    scoring_component_name,
    threshold_value,
    dirname,
    path_to_train_data,
    multi_objectives,
    path_to_herg_simulator,
    train_similarity,
    pretrained_prior,
    al_iterations,
    acquisition,
    n_queries,
    task,
    path_to_simulator,
    noise,
):

    np.random.seed(seed)
    rng = np.random.default_rng(seed)

    if acquisition != "None":
        jobid = (
            f"{path_to_output_dir}/{dirname}_R{rounds}_step{num_opt_steps}_T{al_iterations}_n{n_queries}_{acquisition}_noise{noise}"
            if acquisition != "None"
            else f"{dirname}_R{rounds}_{acquisition}"
        )
    else:
        jobid = f"{path_to_output_dir}/{dirname}_R{rounds}_step{num_opt_steps}_None"
    output_folder = os.path.expanduser(f"{jobid}_seed{seed}")

    # General output directory where to store results from all REINVENT jobs
    if not os.path.exists(path_to_output_dir):
        os.makedirs(path_to_output_dir)

    # REINVENT job results
    if os.path.exists(output_folder):
        shutil.rmtree(output_folder)
    os.makedirs(output_folder)
    print(f"\nCreating output directory: {output_folder}.")

    initial_dir = (
        f"{path_to_output_dir}/{dirname}_R{rounds}_step{num_opt_steps}_None_seed{seed}"
    )
    if acquisition != "None":
        if os.path.exists(initial_dir):
            os.makedirs(os.path.join(output_folder, "iteration_0"))
            try:
                initial_unlabelled_pool = os.path.join(
                    initial_dir, "results/scaffold_memory.csv"
                )
                shutil.copy(
                    initial_unlabelled_pool, os.path.join(output_folder, "iteration_0")
                )
            except FileNotFoundError:
                pass

    print(
        f"\nRunning REINVENT with {num_opt_steps} optimization steps guided by predictive property ({task}) component"
    )
    if acquisition != "None":
        print(
            f"\nRunning HITL AL experiment (seed {seed}) with R={rounds}, T={al_iterations}, n_queries={n_queries}, acquisition={acquisition}. \n Results will be saved at {output_folder}"
        )
    else:
        print(
            f"\nRunning HITL AL experiment (seed {seed}) with R={rounds}, acquisition={acquisition}. \n Results will be saved at {output_folder}"
        )

    prop_predictor = load_model(f"{path_to_scoring_model}.pkl")
    scoring_model_name, path_to_scoring_model = copy_model(
        path_to_scoring_model, output_folder, iter=0
    )

    train_smiles, train_fps, train_labels = load_training_set(path_to_train_data)
    sample_weights = np.ones(len(train_smiles))

    conf_filename = "config.json"
    jobname = "fine-tune predictive component"
    configuration_json_path = write_REINVENT_config(
        output_folder, conf_filename, jobid, jobname
    )
    print(f"\nCreating config file: {configuration_json_path}")
    update_reinvent_config_file(
        output_folder,
        num_opt_steps,
        path_to_scoring_model,
        train_smiles,
        model_type,
        task,
        scoring_component_name,
        train_similarity=train_similarity,
        pretrained_prior=pretrained_prior,
        multi_objectives=multi_objectives,
        path_to_herg_simulator=path_to_herg_simulator,
        iter=0,
    )

    generated_molecules = run_reinvent(
        path_to_reinvent_env,
        path_to_reinvent_repo,
        acquisition,
        configuration_json_path,
        output_folder,
        iter=0,
    )

    for r in range(1, rounds + 1):
        highscore_molecules = prep_pool(
            generated_molecules, threshold_value, scoring_component_name
        )

        if acquisition != "None":
            # store molecule indexes selected for feedback
            selected_feedback = np.empty(0).astype(int)
            for t in range(1, al_iterations + 1):
                selected_smiles, selected_feedback = active_learning_selection(
                    generated_molecules,
                    highscore_molecules,
                    selected_feedback,
                    n_queries,
                    acquisition,
                    prop_predictor,
                    model_type,
                    rng,
                )
                # Remove any generated SMILES which is identical to train set SMILES
                selected_smiles = [s for s in selected_smiles if s not in train_smiles]
                feedback, confidences = get_expert_feedback(
                    selected_smiles, task, model_type, path_to_simulator, noise
                )
                train_fps, train_labels, sample_weights, train_smiles = (
                    augment_train_set(
                        train_fps,
                        train_labels,
                        sample_weights,
                        train_smiles,
                        selected_smiles,
                        feedback,
                        confidences,
                        output_folder,
                        r,
                        t,
                    )
                )
                model_new_savefile = os.path.join(
                    output_folder, f"{scoring_model_name}_iteration_{r}.pkl"
                )
                retrain_model(
                    train_fps,
                    train_labels,
                    sample_weights,
                    prop_predictor,
                    model_type,
                    model_new_savefile,
                )
                prop_predictor = load_model(model_new_savefile)
                highscore_molecules = [s for s in highscore_molecules if s not in selected_smiles]
            output_folder, configuration_json_path = save_configuration_file(
                output_folder,
                initial_dir,
                conf_filename,
                jobid,
                seed,
                scoring_component_name,
                r,
                model_new_savefile,
            )
        else:
            output_folder, configuration_json_path = save_configuration_file(
                output_folder,
                initial_dir,
                conf_filename,
                jobid,
                seed,
                scoring_component_name,
                r,
            )

        generated_molecules = run_reinvent(
            path_to_reinvent_env,
            path_to_reinvent_repo,
            acquisition,
            configuration_json_path,
            output_folder,
            r,
        )

    print(f"\nExit and save results")


if __name__ == "__main__":
    main()