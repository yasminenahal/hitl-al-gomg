# load dependencies
import click
import pickle
import os
import shutil
import json
import pandas as pd
import numpy as np
from numpy.random import default_rng

from synthitl.simulated_expert import EvaluationModel
from synthitl.write import write_REINVENT_config
from synthitl.acquisition import select_query

from models.RandomForest import RandomForestClf, RandomForestReg
from utils import ecfp_generator

from path import reinvent, reinventconda, training, predictors, simulators, priors

fp_counter = ecfp_generator(radius=3, useCounts=True)

def load_training_set(init_train_set):
    # Load the data used to train the initial target property predictor
    train_set = pd.read_csv(f"{training}/{init_train_set}.csv")
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

def load_model(scoring_model):
    # Load the initial target property predictor
    print("\nLoading target property predictor")
    fitted_model = pickle.load(open(f"{predictors}/{scoring_model}.pkl", "rb"))
    return fitted_model

def copy_model(scoring_model, output_folder, iter):
    # Save the initial target property predictor to the root output folder
    model_load_path = output_folder + f"/{scoring_model}_iteration_{iter}.pkl"
    if iter==0 and not os.path.exists(model_load_path):
        shutil.copy(scoring_model, output_folder)

def update_reinvent_config_file(
    output_folder, 
    num_opt_steps, 
    scoring_model, 
    model_type,
    task, 
    scoring_component_name, 
    multi_objectives,
    train_similarity, 
    pretrained_prior, 
    smiles_train, 
    iter
    ):
    if iter == 0:
        configuration = json.load(open(os.path.join(output_folder, "config.json")))
    else:
        configuration = json.load(open(os.path.join(output_folder, f"config_iteration{iter}.json")))
    # write specified number of REINVENT optimization steps in configuration file
    # (example: if R = 5 (rounds) and Reinvent opt_steps = 100, we will run 5*100 RL optimization steps)
    configuration["parameters"]["reinforcement_learning"]["n_steps"] = num_opt_steps
    # Write initial scoring predictor path in configuration file
    configuration_scoring_function = configuration["parameters"]["scoring_function"]["parameters"]
    for i in range(len(configuration_scoring_function)):
        if configuration_scoring_function[i]["component_type"] == "predictive_property" and configuration_scoring_function[i]["name"] == scoring_component_name:
            configuration_scoring_function[i]["specific_parameters"]["model_path"] = scoring_model
            configuration_scoring_function[i]["specific_parameters"]["scikit"] = model_type
            if model_type == "classification":
                configuration_scoring_function[i]["specific_parameters"]["transformation"] = {"transformation_type": "no_transformation"}
            elif model_type == "regression" and task == "logp":
                configuration_scoring_function[i]["specific_parameters"]["transformation"] = {
                    "transformation_type": "double_sigmoid",
                    "high": 4, # Highest desired LogP value according to the paper experiment (use case 1)
                    "low": 2, # Lowest desired LogP value according to the paper experiment (use case 1)
                    "coef_div": 3.0,
                    "coef_si": 10,
                    "coef_se": 10
                }
    
    if pretrained_prior:
        configuration["parameters"]["reinforcement_learning"]["agent"] = f"{priors}/{task}_focused.agent"
    
    if train_similarity:
        tanimoto_config = {
                "component_type": "tanimoto_similarity",
                "name": "Tanimoto Similarity",
                "weight": 1,
                "specific_parameters": {
                    "smiles": smiles_train
                    }
            }
        
        configuration["parameters"]["scoring_function"]["parameters"].append(tanimoto_config)

    if multi_objectives: # According to the paper experiment (multi objective use case)
        qed_config = {
            "component_type": "qed_score",
            "name": "QED",
            "specific_parameters": {
                "transformation": {
                    "low": 0.35,
                    "transformation": True,
                    "transformation_type": "right_step"
                }
            },
            "weight": 0.5
            }
        
        herg_config = {
            "component_type": "predictive_property",
            "name": "herg",
            "weight": 1,
            "specific_parameters": {
                "container_type": "scikit_container",
                "model_path": f"{simulators}/herg.pkl",
                "smiles": "",
                "scikit": "classification",
                "descriptor_type": "ecfp_counts",
                "size": 2048,
                "radius": 3,
                "selected_feat_idx": "none",
                "transformation": {
                    "transformation_type": "flip_probability"
                },
                "use_counts": False,
                "use_features": False
            }
        }
        configuration["parameters"]["scoring_function"]["parameters"].append(qed_config)
        configuration["parameters"]["scoring_function"]["parameters"].append(herg_config)
    
    # Write the updated REINVENT configuration file to the disc
    configuration_json_path = os.path.join(output_folder, f"config_iteration{iter+1}.json")
    with open(configuration_json_path, "w") as f:
        json.dump(configuration, f, indent=4, sort_keys=True)

def run_reinvent(acquisition, configuration_json_path, output_folder, iter):
    if iter == 1 and acquisition != "None":
        if os.path.exists(os.path.join(output_folder, "iteration_0/scaffold_memory.csv")):
            # Start from an existing scaffold_memory i.e., pool of unlabelled compounds
            with open(os.path.join(output_folder, "iteration_0/scaffold_memory.csv"), "r") as file:
                data = pd.read_csv(file)
                data.reset_index(inplace=True)
        else:
            # Start from scratch and generate a pool of unlabelled compounds with REINVENT
            print("\nRun REINVENT")
            os.system(str(reinventconda) + "/bin/python " + str(reinvent) + "/input.py " + str(configuration_json_path) + "&> " + str(output_folder) + "/run.err")
            
            with open(os.path.join(output_folder, "results/scaffold_memory.csv"), "r") as file:
                data = pd.read_csv(file)

    else:
        # Overwrite any existing scaffold_memory and run REINVENT from scratch
        print("\nRun REINVENT")
        os.system(str(reinventconda)  + "/bin/python " + str(reinvent) + "/input.py " + str(configuration_json_path) + "&> " + str(output_folder) + "/run.err")

        with open(os.path.join(output_folder, "results/scaffold_memory.csv"), "r") as file:
            data = pd.read_csv(file)
    return data

def prep_pool(data, threshold_value, scoring_component_name):
    smiles = data["SMILES"]
    bioactivity_score = data[scoring_component_name]
    # Save the indexes of high scoring molecules for satisfying the target property according to predictor
    high_scoring_idx = bioactivity_score > threshold_value
    smiles = smiles[high_scoring_idx]
    print(f"\n{len(smiles)} high-scoring (> {high_scoring_idx}) molecules")
    return smiles

def active_learning_selection(smiles, n_queries, acquisition, fitted_model, model_type, rng):
    # Create the fine-tuned predictor (RFC or RFR) object
    if model_type == "regression":
        model = RandomForestReg(fitted_model)
    elif model_type == "classification":
        model = RandomForestClf(fitted_model)
    # Select queries to show to expert
    if len(smiles) >= n_queries:
        new_query = select_query(smiles, n_queries, model, acquisition, rng)
    else:
        # Select all high-scoring smiles if their number is less than n_queries
        new_query = select_query(smiles, len(smiles), model, acquisition, rng)
    return new_query, smiles.drop(new_query, axis=0)

def get_expert_feedback(smiles, new_query, task, model_type, expert_model, noise):
    if task == "drd2":
        feedback_model = EvaluationModel(task, path_to_simulator=f"{simulators}/{expert_model}")
    else:
        feedback_model = EvaluationModel(task)
    selected_smiles = smiles.iloc[new_query].SMILES.values
    raw_feedback = np.array([feedback_model.human_score(s, noise) for s in selected_smiles])
    if model_type == "regression":
        # Get normalized expert scores from raw feedback (i.e., positive probabilities if classification or continuous values if regression)
        # (On the GUI, the expert directly provides [0,1] scores)
        scores = [feedback_model.utility(f, low=2, high=4) for f in raw_feedback]
        feedback = raw_feedback
    elif model_type == "classification":
        scores = [1 if f > 0.5 else 0 for f in feedback]
        feedback = np.round(raw_feedback).astype(int)
    # Get confidence scores based on expert scores 
    # (e.g., If the expert estimates a 10% chance that a molecule satisfies the target property, we apply a weight of 1−0.1=0.9 when retraining the predictor. 
    # This suggests high confidence in the molecule's inactivity. If the expert estimates a 50% chance, we apply a weight of 0.5. 
    # This indicates moderate certainty or balanced likelihood.)
    print(f"\nNumber of approved molecules by expert: {np.sum(np.round(scores)).astype(int)}")
    confidences = [s if s > 0.5 else 1-s for s in scores]
    return feedback, confidences

def concatenate(
        x_train, 
        y_train, 
        sample_weights, 
        smiles_train, 
        selected_smiles, 
        feedback, 
        confidences, 
        output_folder, 
        iter
    ):
    
    x_new = fp_counter.get_fingerprints(selected_smiles)
    x_train = np.concatenate([x_train, x_new])
    y_train = np.concatenate([y_train, feedback])
    sample_weights = np.concatenate([sample_weights, confidences])
    smiles_train = np.concatenate([smiles_train, selected_smiles])
    print(f"\nAugmented train set size at iteration {iter}: {x_train.shape[0]} {y_train.shape[0]}")
    # Save augmented training data
    D_r = pd.DataFrame(np.concatenate([smiles_train.reshape(-1,1), y_train.reshape(-1,1)], 1))
    D_r.columns = ["SMILES", "target"]
    D_r.to_csv(os.path.join(output_folder, f"augmented_train_set_iter{iter}.csv"))
    return x_train, y_train, sample_weights

def retrain_model(x_train, y_train, sample_weights, fitted_model, model_type, model_new_savefile):
    print("\nRetrain model")
    if model_type == "regression":
        model = RandomForestReg(fitted_model)
    elif model_type == "classification":
        model = RandomForestClf(fitted_model)
    # Retrain and save the updated predictor
    model._retrain(x_train, y_train, sample_weights, save_to_path = model_new_savefile)

def save_configuration_file(output_folder, initial_dir, jobid, replicate, scoring_component_name, model_new_savefile, iter): 
    # Get current REINVENT configuration
    configuration = json.load(open(os.path.join(output_folder, conf_filename)))
    conf_filename = f"iteration{iter}_config.json"   

    # Modify predictor path in configuration using the updated predictor's path
    configuration_scoring_function = configuration["parameters"]["scoring_function"]["parameters"]
    for i in range(len(configuration_scoring_function)):
        if configuration_scoring_function[i]["component_type"] == "predictive_property" and configuration_scoring_function[i]["name"] == scoring_component_name:
            configuration_scoring_function[i]["specific_parameters"]["model_path"] = model_new_savefile

    # Update REINVENT agent checkpoint
    if iter == 1:
        configuration["parameters"]["reinforcement_learning"]["agent"] = os.path.join(initial_dir, "results/Agent.ckpt")
    else:
        configuration["parameters"]["reinforcement_learning"]["agent"] = os.path.join(output_folder, "results/Agent.ckpt")

    root_output_dir = os.path.expanduser(f"{jobid}_seed{replicate}")

    # Define new directory for the next round
    output_folder = os.path.join(root_output_dir, f"iteration{iter}")
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    print(output_folder)

    # Modify log and result paths in REINVENT configuration
    configuration["logging"]["logging_path"] = os.path.join(output_folder, "progress.log")
    configuration["logging"]["result_folder"] = os.path.join(output_folder, "results")

    # Write the updated configuration file to the disc
    configuration_JSON_path = os.path.join(output_folder, conf_filename)
    with open(configuration_JSON_path, "w") as f:
        json.dump(configuration, f, indent=4, sort_keys=True)



@click.command()
@click.option("--replicate", "-r", default=42, type=int, help="Experiment replicate number")
@click.option("--rounds", "-R", default=4, type=int, help="Number of rounds")
@click.option("--num_opt_steps", default=250, type=int, help="Number of REINVENT optimization steps")
@click.option("--scoring_model", "-m", type=str, help="Path to the initial target property predictor")
@click.option("--model_type", type=click.Choice(["regression", "classification"]), help="Whether the scoring model is a regressor or classifier")
@click.option("--scoring_component_name", type=str, help="Name given to the predictor component in REINVENT output files")
@click.option("--threshold_value", "-t", default=0.5, type=float, help="Score threshold value used to select high-scoring generated molecule for active learning")
@click.option("--dirname", "-o", type=str, help="Name of output folder to store all results")
@click.option("--init_train_set", type=str, help="Path to initial predictor training data")
@click.option("--multi_objectives", type=bool, default=False, help="Whether to optimize multiple objectives")
@click.option("--train_similarity", type=bool, default=False, help="Whether to optimize Tanimoto similarity in REINVENT with respect to predictor training set")
@click.option("--pretrained_prior", type=bool, default=False, help="Whether to use a REINVENT prior agent pre-trained on the predictor training set")
@click.option("--al_iterations", "-T", default=5, type=int, help="Number of AL iterations")
@click.option("--acquisition", "-a", type=click.Choice(["random", "greedy", "epig", "qbc", "None"]), help="Data acquisition method")
@click.option("--n_queries", "-n", default=10, help="Number of selected queries to be evaluated by the expert")
@click.option("--task", type=click.Choice(["logp", "drd2"]), help="Goal of the molecule generation")
@click.option("--expert_model", type=str, help="Path to expert model used for assessing predictions with respect to the goal")
@click.option("--noise", default=0.0, type=float, help="Sigma value for the noise term in the expert model (if 0, expert model = Oracle)")
def main(
    replicate, 
    rounds, 
    num_opt_steps,
    scoring_model,
    model_type,
    scoring_component_name,
    threshold_value,
    dirname,
    init_train_set,
    multi_objectives,
    train_similarity,
    pretrained_prior,
    al_iterations,
    acquisition,
    n_queries,
    task,
    expert_model,
    noise
    ):

    np.random.seed(replicate)
    rng = default_rng(replicate)
    
    reinvent_dir = str(reinvent)
    reinvent_env = str(reinventconda)
    jobid = f"{dirname}_R{rounds}_T{al_iterations}_n{n_queries}_{acquisition}_noise{noise}" if acquisition != "None" else f"{dirname}_R{rounds}_{acquisition}"
    output_folder = os.path.expanduser(f"{jobid}_rep{replicate}")
    
    if os.path.exists(output_folder):
        shutil.rmtree(output_folder)
    os.makedirs(output_folder)
    print(f"\nCreating output directory: {output_folder}.")

    if acquisition != "None":
        initial_dir = f"{dirname}_R{rounds}_None_seed{replicate}"
        if os.path.exists(initial_dir):
            os.makedirs(os.path.join(output_folder, "iteration_0"))
            try:
                initial_unlabelled_pool = os.path.join(initial_dir, "results/scaffold_memory.csv")
                shutil.copy(initial_unlabelled_pool, os.path.join(output_folder, "iteration_0"))
            except FileNotFoundError:
                pass

    print(f"\nRunning REINVENT with {num_opt_steps} guided by predictive property ({task}) component")
    print(f"\nRunning HITL AL experiment {replicate} with R={rounds}, T={al_iterations}, n_queries={n_queries}, acquisition={acquisition}. \n Results will be saved at {output_folder}")
    
    fitted_model = load_model(scoring_model)
    copy_model(scoring_model, output_folder, iter=0)
    
    train_smiles, train_fps, train_labels = load_training_set(init_train_set)
    sample_weights = np.ones(len(train_smiles))
    
    conf_filename = "config.json"
    jobname = "fine-tune predictive component"
    configuration_json_path = write_REINVENT_config(reinvent_dir, reinvent_env, output_folder, conf_filename, jobid, jobname)
    print(f"\nCreating config file: {configuration_json_path}")
    update_reinvent_config_file(
        output_folder, 
        num_opt_steps, 
        scoring_model, 
        model_type, 
        scoring_component_name, 
        multi_objectives, 
        train_similarity, 
        pretrained_prior, 
        train_smiles, 
        iter=0
        )

    generated_molecules = run_reinvent(acquisition, configuration_json_path, output_folder, iter=0)

    for r in range(1, rounds + 1):
        highscore_molecules = prep_pool(generated_molecules, threshold_value, scoring_component_name)
        for t in range(1, al_iterations + 1):
            selected_molecules, remaining_highscore_molecules = active_learning_selection(highscore_molecules, n_queries, acquisition, fitted_model, model_type, rng)
            feedback, confidences = get_expert_feedback(highscore_molecules, selected_molecules, task, model_type, expert_model, noise)
            train_fps, train_labels, sample_weights = concatenate(train_fps, train_labels, sample_weights, train_smiles, selected_molecules, feedback, confidences, output_folder, r)
            model_new_savefile = os.path.join(output_folder, f"{scoring_component_name}_iteration_{r}.pkl")
            retrain_model(train_fps, train_labels, sample_weights, fitted_model, model_type, model_new_savefile)
            highscore_molecules = remaining_highscore_molecules
            print(highscore_molecules)
            fitted_model = load_model(model_new_savefile)

        save_configuration_file(output_folder, initial_dir, jobid, replicate, scoring_component_name, model_new_savefile, r)
        run_reinvent(acquisition, configuration_json_path, output_folder, r)

if __name__ == "__main__":
    main()
