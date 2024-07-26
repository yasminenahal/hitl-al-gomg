# load dependencies
import click
import joblib
import os
import shutil
import json
import pandas as pd
import numpy as np
from numpy.random import default_rng

#from FtF.synthitl.simulated_expert import ActivityEvaluationModel
from FtF.synthitl.write import write_REINVENT_config
from FtF.synthitl.updated_acquisition import select_query

from FtF.teacher.RandomForest import RandomForestClf
from FtF.teacher.train import _compute_morgan

from FtF.path import reinvent, reinventconda, training


def load_training_set(init_train_set):
    # load background training data used to pre-train the QSAR model
    train_set = pd.read_csv(str(training / f"{init_train_set}.csv"))
    train_set.rename(columns={"0": "SMILES", "1": "Outcome"}, inplace = True)
    smiles_train = train_set["SMILES"].values # get samples
    # Get labels.
    labels = train_set["Outcome"].values.astype(int)
    # Compute molecules descriptors.
    fingerprints = np.array([_compute_morgan(i, radius=3) for i in list(smiles_train)], dtype=float)
    print("\nDataset specs: ")
    print("\t# Compound:", labels.shape[0])
    print("\t# features:", fingerprints.shape[1])
    x_train, y_train = fingerprints, labels
    return smiles_train, x_train, y_train


def load_model(scoring_model):
    # load the predictive model
    print("Loading predictive model.")
    fitted_model = joblib.load(scoring_model)
    return fitted_model


def copy_model(scoring_model, output_folder, iter):
    # save the model to the root output folder
    model_name = scoring_model.split("/")[-1].split(".")[0]
    model_load_path = output_folder + '/{}_iteration_{}.pkl'.format(model_name, iter)
    if iter==0 and not os.path.exists(model_load_path):
        shutil.copy(scoring_model, output_folder)


def update_reinvent_config_file(output_folder, num_opt_steps, scoring_model, scoring_component_name, train_similarity, smiles_train, iter):
    if iter == 0:
        configuration = json.load(open(os.path.join(output_folder, "config.json")))
    else:
        configuration = json.load(open(os.path.join(output_folder, f"config_iteration{iter}.json")))
    # write specified number of REINVENT optimization steps in configuration
    # (example: if R = 5 (rounds) and Reinvent opt_steps = 100, we will run 5*100 RL optimization steps)
    configuration["parameters"]["reinforcement_learning"]["n_steps"] = num_opt_steps
    # write initial scoring_model path in configuration
    configuration_scoring_function = configuration["parameters"]["scoring_function"]["parameters"]
    for i in range(len(configuration_scoring_function)):
        if configuration_scoring_function[i]["component_type"] == "predictive_property" and configuration_scoring_function[i]["name"] == scoring_component_name:
            configuration_scoring_function[i]["specific_parameters"]["model_path"] = scoring_model
            configuration_scoring_function[i]["specific_parameters"]["transformation"] = {"transformation_type": "no_transformation"}
    
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
    
    # write the updated configuration file to the disc
    configuration_json_path = os.path.join(output_folder, f"config_iteration{iter+1}.json")
    with open(configuration_json_path, 'w') as f:
        json.dump(configuration, f, indent=4, sort_keys=True)


def run_reinvent(acquisition, configuration_json_path, output_folder, iter):
    if iter == 1 and acquisition != 'none':
        if os.path.exists(os.path.join(output_folder, "iteration_0/scaffold_memory.csv")):
            # start from the pre-existing scaffold_memory i.e., pool of unlabelled compounds
            with open(os.path.join(output_folder, "iteration_0/scaffold_memory.csv"), 'r') as file:
                data = pd.read_csv(file)
                data.reset_index(inplace=True)
        else:
            # generate a pool of unlabelled compounds with REINVENT
            print("Run REINVENT")
            os.system(str(reinventconda) + '/bin/python ' + str(reinvent) + '/input.py ' + str(configuration_json_path) + '&> ' + str(output_folder) + '/run.err')
            
            with open(os.path.join(output_folder, "results/scaffold_memory.csv"), 'r') as file:
                data = pd.read_csv(file)

    else:
        # we overwrite the existing scaffold_memory and run REINVENT from scratch
        print("Run REINVENT")
        os.system(str(reinventconda)  + '/bin/python ' + str(reinvent) + '/input.py ' + str(configuration_json_path) + '&> ' + str(output_folder) + '/run.err')

        with open(os.path.join(output_folder, "results/scaffold_memory.csv"), 'r') as file:
            data = pd.read_csv(file)
    return data

def pre_active_learning(data, threshold_value, scoring_component_name):
    smiles = data['SMILES']
    bioactivity_score = data[scoring_component_name]
    # save the indexes of high scoring molecules for bioactivity
    high_scoring_idx = bioactivity_score > threshold_value
    # only analyse highest scoring molecules
    smiles = smiles[high_scoring_idx]
    print(f'{len(smiles)} high-scoring (> {high_scoring_idx}) molecules')
    return smiles

def active_learning_selection(smiles, n_queries, acquisition, fitted_model, rng):
    # create the fine-tuned model (imbalanced RFC) object
    model = RandomForestClf(fitted_model)
    # query selection
    if len(smiles) >= n_queries:
        new_query = select_query(smiles, n_queries, model, acquisition, rng)
    else:
        # select all high-scoring smiles if their number is less than n_queries
        new_query = select_query(smiles, len(smiles), model, acquisition, rng)
    return new_query, smiles.drop(new_query, axis=0)

def get_feedback_labels(smiles, new_query, feedback_model, noise):
    selected_smiles = smiles.iloc[new_query].SMILES.values
    feedback_labels = np.array([feedback_model.human_score(s, noise) for s in selected_smiles])
    return feedback_labels

def concatenate(x_train, y_train, smiles_train, selected_smiles, feedback_labels, output_folder, iter):
    x_new = np.array([_compute_morgan(i, radius=3) for i in selected_smiles])
    x_train = np.concatenate([x_train, x_new])
    y_train = np.concatenate([y_train, feedback_labels])
    smiles_train = np.concatenate([smiles_train, selected_smiles])
    print(f"Augmented train set size at iteration {iter}: {x_train.shape[0]} {y_train.shape[0]}")
    # save augmented training data
    D_r = pd.DataFrame(np.concatenate([smiles_train.reshape(-1,1), y_train.reshape(-1,1)], 1))
    D_r.columns = ["SMILES", "target"]
    D_r.to_csv(os.path.join(output_folder, f"augmented_train_set_iter{iter}.csv"))
    return x_train, y_train

def retrain_model(x_train, y_train, fitted_model, model_new_savefile):
    print("Retrain model")
    model = RandomForestClf(fitted_model)
    # retrain and save the model
    model._retrain(x_train, y_train, save_to_path = model_new_savefile)

def save_configuration_file(output_folder, initial_dir, jobid, replicate, scoring_component_name, model_new_savefile, iter): 
    # get current configuration
    configuration = json.load(open(os.path.join(output_folder, conf_filename)))
    conf_filename = "iteration{}_config.json".format(iter)    

    # modify student model path in configuration using the updated model path
    configuration_scoring_function = configuration["parameters"]["scoring_function"]["parameters"]
    for i in range(len(configuration_scoring_function)):
        if configuration_scoring_function[i]["component_type"] == "predictive_property" and configuration_scoring_function[i]["name"] == scoring_component_name:
            configuration_scoring_function[i]["specific_parameters"]["model_path"] = model_new_savefile

    # update agent checkpoint
    if iter == 1:
        configuration["parameters"]["reinforcement_learning"]["agent"] = os.path.join(initial_dir, "results/Agent.ckpt")
    else:
        configuration["parameters"]["reinforcement_learning"]["agent"] = os.path.join(output_folder, "results/Agent.ckpt")

    root_output_dir = os.path.expanduser("{}_seed{}".format(jobid, replicate))

    # Define new directory for the next round
    output_folder = os.path.join(root_output_dir, f"iteration{iter}")
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    print(output_folder)

    # modify log and result paths in configuration
    configuration["logging"]["logging_path"] = os.path.join(output_folder, "progress.log")
    configuration["logging"]["result_folder"] = os.path.join(output_folder, "results")

    # write the updated configuration file to the disc
    configuration_JSON_path = os.path.join(output_folder, conf_filename)
    with open(configuration_JSON_path, 'w') as f:
        json.dump(configuration, f, indent=4, sort_keys=True)



@click.command()
@click.option("--replicate", "-r", default=42, type=int, help="Experiment replicate identifier")
@click.option("--rounds", "-R", default=3, type=int, help="Number of rounds")
@click.option("--al_iterations", "-T", default=3, type=int, help="Number of AL iterations")
@click.option("--n_queries", "-n", default=20, help="Number of selected queries")
@click.option("--num_opt_steps", default=250, type=int, help="Number of REINVENT optimization steps")
@click.option("--acquisition", "-a", type=click.Choice(["random", "greedy", "epig", "qbc", "none"]), help="Data acquisition method")
@click.option("--scoring_model", "-m", type=str, help="Path to the scoring model to be modified")
@click.option("--scoring_component_name", type=str, help="Name given to the scoring model component in REINVENT output files")
@click.option("--threshold_value", "-t", default=0.5, type=float, help="Score threshold value for generated molecule selection")
@click.option("--dirname", "-o", type=str, help="Name of output folder")
@click.option("--init_train_set", type=str, help="Path to initial training data")
@click.option("--train_similarity", type=bool, default=False, help="Whether to optimize Tanimoto similarity with initial training set")
@click.option("--noise", default=0.0, type=float, help="Noise term in the human/student model")
def main(
    replicate, 
    rounds, 
    al_iterations,
    n_queries,
    num_opt_steps,
    acquisition,
    scoring_model,
    scoring_component_name,
    threshold_value,
    dirname,
    init_train_set,
    train_similarity,
    noise
    ):
    np.random.seed(replicate)
    rng = default_rng(replicate)
    
    reinvent_dir = str(reinvent)
    reinvent_env = str(reinventconda)
    jobid = f"{dirname}_R{rounds}_T{al_iterations}_n{n_queries}_{acquisition}_noise{noise}" if acquisition != 'none' else f"{dirname}_R{rounds}_{acquisition}"
    output_folder = os.path.expanduser(f"{jobid}_rep{replicate}")
    
    if os.path.exists(output_folder):
        shutil.rmtree(output_folder)
    os.makedirs(output_folder)
    print(f"Creating output directory: {output_folder}.")

    if acquisition != 'none':
        initial_dir = f"{dirname}_R{rounds}_None_seed{replicate}"
        if os.path.exists(initial_dir):
            os.makedirs(os.path.join(output_folder, "iteration_0"))
            try:
                initial_unlabelled_pool = os.path.join(initial_dir, "results/scaffold_memory.csv")
                shutil.copy(initial_unlabelled_pool, os.path.join(output_folder, "iteration_0"))
            except FileNotFoundError:
                pass

    print(f"Running MPO experiment with R={rounds}, T={al_iterations}, n_queries={n_queries}, seed={replicate}. \n Results will be saved at {output_folder}")
    
    fitted_model = load_model(scoring_model)
    copy_model(scoring_model, output_folder, iter=0)
    train_smiles, train_fps, train_labels = load_training_set(init_train_set)
    
    conf_filename = "config.json"
    jobname = "fine-tune predictive component"
    configuration_json_path = write_REINVENT_config(reinvent_dir, reinvent_env, output_folder, conf_filename, jobid, jobname)
    print(f"Creating config file: {configuration_json_path}.")
    update_reinvent_config_file(output_folder, num_opt_steps, scoring_model, scoring_component_name, train_similarity, train_smiles, iter=0)

    generated_molecules = run_reinvent(acquisition, configuration_json_path, output_folder, iter=0)

    for r in range(1, rounds + 1):
        highscore_molecules = pre_active_learning(generated_molecules, threshold_value, scoring_component_name)
        for t in range(1, al_iterations + 1):
            selected_molecules, remaining_highscore_molecules = active_learning_selection(highscore_molecules, n_queries, acquisition, fitted_model, rng)
            feedback_labels = get_feedback_labels(highscore_molecules, selected_molecules, fitted_model, noise)
            train_fps, train_labels = concatenate(train_fps, train_labels, train_smiles, selected_molecules, feedback_labels, output_folder, r)
            model_new_savefile = os.path.join(output_folder, f'{scoring_component_name}_iteration_{r}.pkl')
            retrain_model(train_fps, train_labels, fitted_model, model_new_savefile)
            highscore_molecules = remaining_highscore_molecules
            print(highscore_molecules)
            fitted_model = load_model(model_new_savefile)

        save_configuration_file(output_folder, initial_dir, jobid, replicate, scoring_component_name, model_new_savefile, r)
        run_reinvent(acquisition, configuration_json_path, output_folder, r)

if __name__ == "__main__":
    main()

#Example usage
#python run_updated.py --acquisition "epig" --scoring_model teacher/trained_models/fluc.pkl --scoring_component_name "interference" --dirname "fluc" --init_train_set ../data/training/fluc.csv &