import os
import click
import pickle
import pandas as pd
import numpy as np

from hitl_al_gomg.utils import ecfp_generator
from hitl_al_gomg.scoring.metrics import (
    get_pairwise_structural_metrics,
    internal_diversity,
    QED,
    Weight,
    logP,
    SA,
    fraction_unique,
    novelty,
)
from hitl_al_gomg.scoring.write import write_sample_file
from hitl_al_gomg.synthitl.simulated_expert import EvaluationModel, utility
from hitl_al_gomg.path import priors, chemspace

from sklearn.metrics import mean_absolute_error, f1_score, roc_auc_score

from rdkit import Chem
from rdkit import RDLogger

# Disable RDKit error messages
RDLogger.DisableLog("rdApp.*")

fp_counter = ecfp_generator(radius=3, useCounts=True)


def sample_mols_from_agent(
    jobid, jobname, agent_dir, reinvent_env, reinvent_dir, agent="Agent.ckpt", N=1000
):
    print("Sampling from agent " + os.path.join(agent_dir, agent))
    conf_file = write_sample_file(jobid, jobname, agent_dir, agent, N)
    os.system(
        str(reinvent_env)
        + "/bin/python "
        + str(reinvent_dir)
        + "/input.py "
        + str(conf_file)
        + "&> "
        + str(agent_dir)
        + "/sampling.err"
    )


def keep_valid_smiles(list_smi):
    okay_mols = []
    wrong_mols = []
    for i, s in enumerate(list_smi):
        try:
            fp_counter.get_fingerprints([s])
            okay_mols.append(i)
        except:
            wrong_mols.append(i)
    return okay_mols


def get_result_data_for_figures(
    job_name,
    task,
    path_to_predictor,
    path_to_simulator,
    path_to_train_data,
    path_to_test_data,
    path_to_reinvent_env,
    path_to_reinvent_repo,
    path_to_output_dir,
    seed,
    R,
    T,
    n_opt_steps=250,
    acq=None,
    sigma_noise=None,
    n_queries=None,
    score_component_name="bioactivity",
    model_type="regression",
    by_step=True,
):

    print("START")

    jobname = "fine-tune predictive component"

    model0 = pickle.load(open(f"{path_to_predictor}.pkl", "rb"))

    print(f"\nGet sample ChEMBL oracle target values for {task}")
    chembl_data = pd.read_csv(f"{chemspace}/chembl.csv")
    chembl_target_values = chembl_data[f"target_{task}"].tolist()

    if task == "logp":
        feedback_model = EvaluationModel(task)
    else:
        feedback_model = EvaluationModel(
            task, path_to_simulator=f"{path_to_simulator}.pkl"
        )

    train_set = pd.read_csv(f"{path_to_train_data}.csv")
    test_set = pd.read_csv(f"{path_to_test_data}.csv")
    train_smiles = train_set.SMILES.tolist()

    # sample from prior
    print("Sample from prior, step 0")
    if not os.path.exists(os.path.join(f"{priors}", f"sampled_N_10000.csv")):
        sample_mols_from_agent(
            "prior",
            jobname,
            priors,
            path_to_reinvent_env,
            path_to_reinvent_repo,
            agent="random.prior.new",
            N=10000,
        )
    try:
        sampled_smiles_prior = pd.read_csv(
            os.path.join(f"{priors}", f"sampled_N_10000.csv"), header=None
        )
        sampled_smiles_prior.rename(columns={0: "SMILES"}, inplace=True)
        valid_mols = keep_valid_smiles(sampled_smiles_prior.SMILES.tolist())
        sampled_smiles_prior = sampled_smiles_prior[
            sampled_smiles_prior.index.isin(valid_mols)
        ]
    except:
        sampled_smiles_prior = pd.DataFrame({"SMILES": []})

    x_test = fp_counter.get_fingerprints(test_set.SMILES.tolist())
    y_test = test_set.target.values

    (
        all_oracle_values_by_step_top250,
        mean_oracle_value_by_step_top250,
        all_oracle_scores_by_step_top250,
        mean_oracle_score_by_step_top250,
    ) = ({}, {}, {}, {})
    (
        all_oracle_values_by_step_top500,
        mean_oracle_value_by_step_top500,
        all_oracle_scores_by_step_top500,
        mean_oracle_score_by_step_top500,
    ) = ({}, {}, {}, {})
    (
        all_oracle_values_by_step_top1000,
        mean_oracle_value_by_step_top1000,
        all_oracle_scores_by_step_top1000,
        mean_oracle_score_by_step_top1000,
    ) = ({}, {}, {}, {})
    (
        all_oracle_values_by_step_above05,
        mean_oracle_value_by_step_above05,
        all_oracle_scores_by_step_above05,
        mean_oracle_score_by_step_above05,
    ) = ({}, {}, {}, {})

    (
        all_predicted_values_by_step_top250,
        mean_predicted_value_by_step_top250,
        all_predicted_scores_by_step_top250,
        mean_predicted_score_by_step_top250,
    ) = ({}, {}, {}, {})
    (
        all_predicted_values_by_step_top500,
        mean_predicted_value_by_step_top500,
        all_predicted_scores_by_step_top500,
        mean_predicted_score_by_step_top500,
    ) = ({}, {}, {}, {})
    (
        all_predicted_values_by_step_top1000,
        mean_predicted_value_by_step_top1000,
        all_predicted_scores_by_step_top1000,
        mean_predicted_score_by_step_top1000,
    ) = ({}, {}, {}, {})
    (
        all_predicted_values_by_step_above05,
        mean_predicted_value_by_step_above05,
        all_predicted_scores_by_step_above05,
        mean_predicted_score_by_step_above05,
    ) = ({}, {}, {}, {})

    (
        acc_wrt_current_by_step_top250,
        acc_wrt_current_by_step_top500,
        acc_wrt_current_by_step_top1000,
        acc_wrt_current_by_step_above05,
    ) = (
        {"f1": [], "roc_auc": [], "mae": []},
        {"f1": [], "roc_auc": [], "mae": []},
        {"f1": [], "roc_auc": [], "mae": []},
        {"f1": [], "roc_auc": [], "mae": []},
    )
    (
        snn_wrt_queries_by_step_top250,
        snn_wrt_queries_by_step_top500,
        snn_wrt_queries_by_step_top1000,
        snn_wrt_queries_by_step_above05,
    ) = ([], [], [], [])
    (
        frag_wrt_queries_by_step_top250,
        frag_wrt_queries_by_step_top500,
        frag_wrt_queries_by_step_top1000,
        frag_wrt_queries_by_step_above05,
    ) = ([], [], [], [])
    (
        fcd_wrt_queries_by_step_top250,
        fcd_wrt_queries_by_step_top500,
        fcd_wrt_queries_by_step_top1000,
        fcd_wrt_queries_by_step_above05,
    ) = ([], [], [], [])
    (
        snn_wrt_train_by_step_top250,
        snn_wrt_train_by_step_top500,
        snn_wrt_train_by_step_top1000,
        snn_wrt_train_by_step_above05,
    ) = ([], [], [], [])
    (
        frag_wrt_train_by_step_top250,
        frag_wrt_train_by_step_top500,
        frag_wrt_train_by_step_top1000,
        frag_wrt_train_by_step_above05,
    ) = ([], [], [], [])
    (
        fcd_wrt_train_by_step_top250,
        fcd_wrt_train_by_step_top500,
        fcd_wrt_train_by_step_top1000,
        fcd_wrt_train_by_step_above05,
    ) = ([], [], [], [])
    (
        snn_wrt_previous_set_by_step_top250,
        snn_wrt_previous_set_by_step_top500,
        snn_wrt_previous_set_by_step_top1000,
        snn_wrt_previous_set_by_step_above05,
    ) = ([], [], [], [])
    (
        frag_wrt_previous_set_by_step_top250,
        frag_wrt_previous_set_by_step_top500,
        frag_wrt_previous_set_by_step_top1000,
        frag_wrt_previous_set_by_step_above05,
    ) = ([], [], [], [])
    (
        fcd_wrt_previous_set_by_step_top250,
        fcd_wrt_previous_set_by_step_top500,
        fcd_wrt_previous_set_by_step_top1000,
        fcd_wrt_previous_set_by_step_above05,
    ) = ([], [], [], [])
    (
        internal_diversity_by_step_top250,
        internal_diversity_by_step_top500,
        internal_diversity_by_step_top1000,
        internal_diversity_by_step_above05,
    ) = ([], [], [], [])
    (
        uniqueness_by_step_top250,
        uniqueness_by_step_top500,
        uniqueness_by_step_top1000,
        uniqueness_by_step_above05,
    ) = ([], [], [], [])
    (
        novelty_by_step_top250,
        novelty_by_step_top500,
        novelty_by_step_top1000,
        novelty_by_step_above05,
    ) = ([], [], [], [])
    qed_by_step_top250, qed_by_step_top500, qed_by_step_top1000, qed_by_step_above05 = (
        [],
        [],
        [],
        [],
    )
    mw_by_step_top250, mw_by_step_top500, mw_by_step_top1000, mw_by_step_above05 = (
        [],
        [],
        [],
        [],
    )
    sa_by_step_top250, sa_by_step_top500, sa_by_step_top1000, sa_by_step_above05 = (
        [],
        [],
        [],
        [],
    )
    (
        logp_by_step_top250,
        logp_by_step_top500,
        logp_by_step_top1000,
        logp_by_step_above05,
    ) = ([], [], [], [])

    sample_acc_wrt_current_by_step = {"f1": [], "roc_auc": [], "mae": []}
    (
        sample_all_oracle_values_by_step,
        sample_mean_oracle_value_by_step,
        sample_all_oracle_scores_by_step,
        sample_mean_oracle_score_by_step,
    ) = ({}, {}, {}, {})
    (
        sample_all_predicted_values_by_step,
        sample_mean_predicted_value_by_step,
        sample_all_predicted_scores_by_step,
        sample_mean_predicted_score_by_step,
    ) = ({}, {}, {}, {})
    sample_snn_wrt_queries_by_step, sample_snn_wrt_train_by_step = [], []
    sample_frag_wrt_queries_by_step, sample_frag_wrt_train_by_step = [], []
    sample_fcd_wrt_queries_by_step, sample_fcd_wrt_train_by_step = [], []
    (
        sample_snn_wrt_previous_set_by_step,
        sample_frag_wrt_previous_set_by_step,
        sample_fcd_wrt_previous_set_by_step,
    ) = ([], [], [])
    (
        sample_internal_diversity_by_step,
        sample_uniqueness_by_step,
        sample_novelty_by_step,
    ) = ([], [], [])
    sample_qed_by_step, sample_qed_by_step, sample_qed_by_step, sample_qed_by_step = (
        [],
        [],
        [],
        [],
    )
    sample_mw_by_step, sample_mw_by_step, sample_mw_by_step, sample_mw_by_step = (
        [],
        [],
        [],
        [],
    )
    sample_sa_by_step, sample_sa_by_step, sample_sa_by_step, sample_sa_by_step = (
        [],
        [],
        [],
        [],
    )
    (
        sample_logp_by_step,
        sample_logp_by_step,
        sample_logp_by_step,
        sample_logp_by_step,
    ) = ([], [], [], [])

    acc_wrt_d0 = {"f1": [], "roc_auc": [], "mae": []}
    acc_wrt_chembl = {"f1": [], "roc_auc": [], "mae": []}

    # add initial value
    if model_type == "classification":
        sample_prior_pred_vals = model0.predict_proba(
            fp_counter.get_fingerprints(sampled_smiles_prior.SMILES.tolist())
        )[:, 1].tolist()
        sample_prior_oracle_vals = [
            feedback_model.oracle_score(s) for s in sampled_smiles_prior.SMILES.tolist()
        ]
        sampled_smiles_prior["predicted"] = sample_prior_pred_vals
        sample_all_predicted_values_by_step["initial"] = sample_prior_pred_vals
        sample_mean_predicted_value_by_step["initial"] = np.mean(sample_prior_pred_vals)
        sample_all_predicted_scores_by_step["initial"] = sample_prior_pred_vals
        sample_mean_predicted_score_by_step["initial"] = np.mean(sample_prior_pred_vals)
        sample_all_oracle_values_by_step["initial"] = sample_prior_oracle_vals
        sample_mean_oracle_value_by_step["initial"] = np.mean(
            sample_all_oracle_values_by_step["initial"]
        )
        sample_all_oracle_scores_by_step["initial"] = sample_all_oracle_values_by_step[
            "initial"
        ]
        sample_mean_oracle_score_by_step["initial"] = sample_mean_oracle_value_by_step[
            "initial"
        ]
        sample_acc_wrt_current_by_step["mae"].append(
            mean_absolute_error(sample_prior_oracle_vals, sample_prior_pred_vals)
        )
        sample_acc_wrt_current_by_step["f1"].append(
            f1_score(
                np.round(sample_prior_oracle_vals), np.round(sample_prior_pred_vals)
            )
        )
        sample_acc_wrt_current_by_step["roc_auc"].append(
            roc_auc_score(
                np.round(sample_prior_oracle_vals), np.round(sample_prior_pred_vals)
            )
        )

    if model_type == "regression":
        sample_prior_pred_vals = model0.predict(
            fp_counter.get_fingerprints(sampled_smiles_prior.SMILES.tolist())
        )
        sample_prior_oracle_vals = [
            feedback_model.oracle_score(s) for s in sampled_smiles_prior.SMILES.tolist()
        ]
        sample_prior_pred_vals_utility = [
            utility(s, low=2, high=4) for s in sample_prior_pred_vals
        ]
        sample_prior_oracle_vals_utility = [
            utility(s, low=2, high=4) for s in sample_prior_oracle_vals
        ]
        sample_all_predicted_values_by_step["initial"] = sample_prior_pred_vals
        sample_mean_predicted_value_by_step["initial"] = np.mean(sample_prior_pred_vals)
        sample_all_predicted_scores_by_step["initial"] = sample_prior_pred_vals_utility
        sample_mean_predicted_score_by_step["initial"] = np.mean(
            sample_prior_pred_vals_utility
        )
        sample_all_oracle_values_by_step["initial"] = sample_prior_oracle_vals
        sample_mean_oracle_value_by_step["initial"] = np.mean(sample_prior_oracle_vals)
        sample_all_oracle_scores_by_step["initial"] = sample_prior_oracle_vals_utility
        sample_mean_oracle_score_by_step["initial"] = np.mean(
            sample_prior_oracle_vals_utility
        )
        sample_acc_wrt_current_by_step["mae"].append(
            mean_absolute_error(sample_prior_oracle_vals, sample_prior_pred_vals)
        )

    sample_snn_wrt_train_by_step.append(
        get_pairwise_structural_metrics(
            sampled_smiles_prior.SMILES.tolist(), train_smiles, "snn"
        )
    )
    sample_frag_wrt_train_by_step.append(
        get_pairwise_structural_metrics(
            sampled_smiles_prior.SMILES.tolist(), train_smiles, "frag"
        )
    )
    sample_fcd_wrt_train_by_step.append(
        get_pairwise_structural_metrics(
            sampled_smiles_prior.SMILES.tolist(), train_smiles, "fcd"
        )
    )

    sample_internal_diversity_by_step.append(
        internal_diversity(sampled_smiles_prior.SMILES.tolist())
    )

    sample_uniqueness_by_step.append(
        fraction_unique(sampled_smiles_prior.SMILES.tolist())
    )
    sample_novelty_by_step.append(
        novelty(sampled_smiles_prior.SMILES.tolist(), train_smiles)
    )

    sample_qed_by_step.append(
        np.mean(
            [QED(Chem.MolFromSmiles(s)) for s in sampled_smiles_prior.SMILES.tolist()]
        )
    )
    sample_mw_by_step.append(
        np.mean(
            [
                Weight(Chem.MolFromSmiles(s))
                for s in sampled_smiles_prior.SMILES.tolist()
            ]
        )
    )
    sample_logp_by_step.append(
        np.mean(
            [logP(Chem.MolFromSmiles(s)) for s in sampled_smiles_prior.SMILES.tolist()]
        )
    )
    sample_sa_by_step.append(
        np.mean(
            [SA(Chem.MolFromSmiles(s)) for s in sampled_smiles_prior.SMILES.tolist()]
        )
    )

    for i in range(R):

        print("\nREINVENT-HITL-AL cycle ", i)

        print("\n1- Get generated molecules from Scaffold Memory buffer")

        if acq != "None":

            dirname = f"{path_to_output_dir}/{job_name}_R{R}_step{n_opt_steps}_T{T}_n{n_queries}_{acq}_noise{sigma_noise}_seed{seed}"

            if i == 0:
                scaff = pd.read_csv(f"{dirname}/iteration_0/scaffold_memory.csv")
                jobid = f"{job_name}_R{R}_step{n_opt_steps}_None_seed{seed}"
                base_dirname = f"{path_to_output_dir}/{job_name}_R{R}_step{n_opt_steps}_None_seed{seed}"

                if not os.path.exists(
                    os.path.join(f"{base_dirname}/results", f"sampled_N_10000.csv")
                ):
                    sample_mols_from_agent(
                        jobid,
                        jobname,
                        f"{base_dirname}/results",
                        path_to_reinvent_env,
                        path_to_reinvent_repo,
                        N=1000,
                    )
                try:
                    sampled_smiles_agent = pd.read_csv(
                        os.path.join(f"{base_dirname}/results", f"sampled_N_10000.csv"),
                        header=None,
                    )
                except:
                    sampled_smiles_agent = pd.DataFrame({"SMILES": []})
                queried_smiles = pd.read_csv(
                    f"{dirname}/augmented_train_set_iter{i+1}.csv"
                ).SMILES.values[-n_queries:]
                model = model0
            else:
                scaff = pd.read_csv(
                    f"{dirname}/iteration{i}/results/scaffold_memory.csv"
                )
                if not os.path.exists(
                    os.path.join(
                        f"{dirname}/iteration{i}/results", f"sampled_N_10000.csv"
                    )
                ):
                    sample_mols_from_agent(
                        dirname,
                        jobname,
                        f"{dirname}/iteration{i}/results",
                        path_to_reinvent_env,
                        path_to_reinvent_repo,
                        N=1000,
                    )
                try:
                    sampled_smiles_agent = pd.read_csv(
                        os.path.join(
                            f"{dirname}/iteration{i}/results", f"sampled_N_10000.csv"
                        ),
                        header=None,
                    )
                except:
                    sampled_smiles_agent = pd.DataFrame({"SMILES": []})
                queried_smiles = pd.read_csv(
                    f"{dirname}/iteration{i}/augmented_train_set_iter{i+1}.csv"
                ).SMILES.values[-n_queries:]
                predictor_name = path_to_predictor.split("/")[-1].split(".pkl")[0]
                if i == 1:
                    model = pickle.load(
                        open(f"{dirname}/{predictor_name}_iteration_1.pkl", "rb")
                    )
                if i > 1:
                    model = pickle.load(
                        open(
                            f"{dirname}/iteration{i-1}/{predictor_name}_iteration_{i}.pkl",
                            "rb",
                        )
                    )

        else:
            dirname = f"{path_to_output_dir}/{job_name}_R{R}_step{n_opt_steps}_None_seed{seed}"
            jobid = f"{job_name}_R{R}_step{n_opt_steps}_None_seed{seed}"
            if i == 0:
                scaff = pd.read_csv(f"{dirname}/results/scaffold_memory.csv")
                if not os.path.exists(
                    os.path.join(f"{dirname}/results", f"sampled_N_10000.csv")
                ):
                    sample_mols_from_agent(
                        jobid,
                        jobname,
                        f"{dirname}/results",
                        path_to_reinvent_env,
                        path_to_reinvent_repo,
                        N=1000,
                    )
                try:
                    sampled_smiles_agent = pd.read_csv(
                        os.path.join(f"{dirname}/results", f"sampled_N_10000.csv"),
                        header=None,
                    )
                except:
                    sampled_smiles_agent = pd.DataFrame({"SMILES": []})
            else:
                scaff = pd.read_csv(
                    f"{dirname}/iteration{i}/results/scaffold_memory.csv"
                )
                if not os.path.exists(
                    os.path.join(
                        f"{dirname}/iteration{i}/results", f"sampled_N_10000.csv"
                    )
                ):
                    sample_mols_from_agent(
                        dirname,
                        jobname,
                        f"{dirname}/iteration{i}/results",
                        path_to_reinvent_env,
                        path_to_reinvent_repo,
                        N=1000,
                    )
                try:
                    sampled_smiles_agent = pd.read_csv(
                        os.path.join(
                            f"{dirname}/iteration{i}/results", f"sampled_N_10000.csv"
                        ),
                        header=None,
                    )
                except:
                    sampled_smiles_agent = pd.DataFrame({"SMILES": []})
            model = model0

        # get accuracy on d0 and ChEMBL by cycle
        print("\nMeasure predictive performance of current predictor on D0 and ChEMBL")
        if model_type == "regression":
            chembl_pred_values = model.predict(
                fp_counter.get_fingerprints(chembl_data.SMILES.tolist())
            )
        if model_type == "classification":
            chembl_pred_values = model.predict_proba(
                fp_counter.get_fingerprints(chembl_data.SMILES.tolist())
            )[:, 1]
            acc_wrt_d0["f1"].append(f1_score(y_test, model.predict(x_test)))
            acc_wrt_d0["mae"].append(
                mean_absolute_error(
                    [feedback_model.oracle_score(s) for s in test_set.SMILES.tolist()],
                    model.predict_proba(x_test)[:, 1],
                )
            )
            try:
                acc_wrt_d0["roc_auc"].append(
                    roc_auc_score(y_test, model.predict(x_test))
                )
            except:
                acc_wrt_d0["roc_auc"].append(0.0)
            acc_wrt_chembl["mae"].append(
                mean_absolute_error(chembl_target_values, chembl_pred_values)
            )
            acc_wrt_chembl["f1"].append(
                f1_score(np.round(chembl_target_values), np.round(chembl_pred_values))
            )
            try:
                acc_wrt_chembl["roc_auc"].append(
                    roc_auc_score(
                        np.round(chembl_target_values), np.round(chembl_pred_values)
                    )
                )
            except:
                acc_wrt_chembl["roc_auc"].append(0.0)

        sampled_smiles_agent.rename(columns={0: "SMILES"}, inplace=True)
        valid_mols = keep_valid_smiles(sampled_smiles_agent.SMILES.tolist())
        sampled_smiles_agent = sampled_smiles_agent[
            sampled_smiles_agent.index.isin(valid_mols)
        ]

        scaff.sort_values(by=[score_component_name], ascending=False, inplace=True)

        if model_type == "classification":
            try:
                sample_agent_pred_vals = model.predict_proba(
                    fp_counter.get_fingerprints(sampled_smiles_agent.SMILES.tolist())
                )[:, 1].tolist()
            except:
                sample_agent_pred_vals = [0]
            sampled_smiles_agent["predicted"] = sample_agent_pred_vals
        if model_type == "regression":
            try:
                sample_agent_pred_vals = model.predict(
                    fp_counter.get_fingerprints(sampled_smiles_agent.SMILES.tolist())
                )
            except:
                sample_agent_pred_vals = [0]
            sample_agent_pred_vals_utility = [
                utility(s, low=2, high=4) for s in sample_agent_pred_vals
            ]
            sampled_smiles_agent["predicted"] = sample_agent_pred_vals_utility

        sample_agent_oracle_vals = [
            feedback_model.oracle_score(s) for s in sampled_smiles_agent.SMILES.tolist()
        ]

        sample_all_oracle_values_by_step[i] = sample_agent_oracle_vals
        sample_all_predicted_values_by_step[i] = sample_agent_pred_vals
        sample_mean_oracle_value_by_step[i] = np.mean(sample_agent_oracle_vals)
        sample_mean_predicted_value_by_step[i] = np.mean(sample_agent_pred_vals)
        if model_type == "regression":
            sample_predicted_scores = [
                utility(s, low=2, high=4) for s in sample_agent_pred_vals
            ]
            sample_oracle_scores = [
                utility(s, low=2, high=4) for s in sample_agent_oracle_vals
            ]
            sample_all_oracle_scores_by_step[i] = sample_oracle_scores
            sample_all_predicted_scores_by_step[i] = sample_predicted_scores
            sample_mean_oracle_score_by_step[i] = np.mean(sample_oracle_scores)
            sample_mean_predicted_score_by_step[i] = np.mean(sample_predicted_scores)
        if model_type == "classification":
            sample_all_oracle_scores_by_step[i] = sample_agent_oracle_vals
            sample_all_predicted_scores_by_step[i] = sample_agent_pred_vals
            sample_mean_oracle_score_by_step[i] = np.mean(sample_agent_oracle_vals)
            sample_mean_predicted_score_by_step[i] = np.mean(sample_agent_pred_vals)

        if len(sampled_smiles_agent.SMILES) > 0:
            sample_acc_wrt_current_by_step["mae"].append(
                mean_absolute_error(sample_agent_oracle_vals, sample_agent_pred_vals)
            )
            if model_type == "classification":
                sample_acc_wrt_current_by_step["f1"].append(
                    f1_score(
                        np.round(sample_agent_oracle_vals),
                        np.round(sample_agent_pred_vals),
                    )
                )
                try:
                    sample_acc_wrt_current_by_step["roc_auc"].append(
                        roc_auc_score(
                            np.round(sample_agent_oracle_vals),
                            np.round(sample_agent_pred_vals),
                        )
                    )
                except:
                    sample_acc_wrt_current_by_step["roc_auc"].append(0.0)

        print("\nGet MOSES structural distance and diversity metrics")
        sampled_smiles_agent.dropna(inplace=True)
        if len(sampled_smiles_agent.SMILES.tolist()) > 0:
            if i == 0:
                sample_snn_wrt_previous_set_by_step.append(
                    get_pairwise_structural_metrics(
                        sampled_smiles_agent.SMILES.tolist(), train_smiles, "snn"
                    )
                )
                sample_frag_wrt_previous_set_by_step.append(
                    get_pairwise_structural_metrics(
                        sampled_smiles_agent.SMILES.tolist(), train_smiles, "frag"
                    )
                )
                sample_fcd_wrt_previous_set_by_step.append(
                    get_pairwise_structural_metrics(
                        sampled_smiles_agent.SMILES.tolist(), train_smiles, "fcd"
                    )
                )
            else:
                sample_snn_wrt_previous_set_by_step.append(
                    get_pairwise_structural_metrics(
                        sampled_smiles_agent.SMILES.tolist(), sample_prev_smiles, "snn"
                    )
                )
                sample_frag_wrt_previous_set_by_step.append(
                    get_pairwise_structural_metrics(
                        sampled_smiles_agent.SMILES.tolist(), sample_prev_smiles, "frag"
                    )
                )
                sample_fcd_wrt_previous_set_by_step.append(
                    get_pairwise_structural_metrics(
                        sampled_smiles_agent.SMILES.tolist(), sample_prev_smiles, "fcd"
                    )
                )
            if acq != "None":
                sample_snn_wrt_queries_by_step.append(
                    get_pairwise_structural_metrics(
                        sampled_smiles_agent.SMILES.tolist(), queried_smiles, "snn"
                    )
                )
                sample_frag_wrt_queries_by_step.append(
                    get_pairwise_structural_metrics(
                        sampled_smiles_agent.SMILES.tolist(), queried_smiles, "frag"
                    )
                )
                sample_fcd_wrt_queries_by_step.append(
                    get_pairwise_structural_metrics(
                        sampled_smiles_agent.SMILES.tolist(), queried_smiles, "fcd"
                    )
                )
            sample_internal_diversity_by_step.append(
                internal_diversity(sampled_smiles_agent.SMILES.tolist())
            )

            sample_snn_wrt_train_by_step.append(
                get_pairwise_structural_metrics(
                    sampled_smiles_agent.SMILES.tolist(), train_smiles, "snn"
                )
            )
            sample_frag_wrt_train_by_step.append(
                get_pairwise_structural_metrics(
                    sampled_smiles_agent.SMILES.tolist(), train_smiles, "frag"
                )
            )
            sample_fcd_wrt_train_by_step.append(
                get_pairwise_structural_metrics(
                    sampled_smiles_agent.SMILES.tolist(), train_smiles, "fcd"
                )
            )

            print(
                "\nGet MOSES distributions of key chemical properties for the generated molecules"
            )
            sample_qed_by_step.append(
                np.mean(
                    [
                        QED(Chem.MolFromSmiles(s))
                        for s in sampled_smiles_agent.SMILES.tolist()
                    ]
                )
            )
            sample_mw_by_step.append(
                np.mean(
                    [
                        Weight(Chem.MolFromSmiles(s))
                        for s in sampled_smiles_agent.SMILES.tolist()
                    ]
                )
            )
            sample_logp_by_step.append(
                np.mean(
                    [
                        logP(Chem.MolFromSmiles(s))
                        for s in sampled_smiles_agent.SMILES.tolist()
                    ]
                )
            )
            sample_sa_by_step.append(
                np.mean(
                    [
                        SA(Chem.MolFromSmiles(s))
                        for s in sampled_smiles_agent.SMILES.tolist()
                    ]
                )
            )

            print("\nGet MOSES uniqueness and novelty metrics")
            sample_uniqueness_by_step.append(
                fraction_unique(sampled_smiles_agent.SMILES.tolist())
            )
            sample_novelty_by_step.append(
                novelty(sampled_smiles_agent.SMILES.tolist(), train_smiles)
            )

            sample_prev_smiles = sampled_smiles_agent.SMILES.tolist()

        if by_step:

            for j in range(n_opt_steps + 1):
                if j % 50 == 0:
                    print("\nPolicy opt step ", j)
                    print("\nGet mean and distribution of predicted scores")
                    # NEW
                    if j == 0:
                        scaff_step = scaff[scaff["Step"] == j]
                    else:
                        scaff_step = scaff[scaff["Step"] == j - 1]
                    scaff_step["oracle_values"] = [
                        feedback_model.oracle_score(s)
                        for s in scaff_step.SMILES.tolist()
                    ]
                    scaff_step_above05 = scaff_step[
                        scaff_step[f"{score_component_name}"] > 0.5
                    ]

                    all_predicted_values_by_step_top250[f"cycle{i}_step{j}"] = (
                        scaff_step.head(250)[
                            f"raw_{score_component_name}"
                        ].values.tolist()
                    )
                    all_predicted_values_by_step_top500[f"cycle{i}_step{j}"] = (
                        scaff_step.head(500)[
                            f"raw_{score_component_name}"
                        ].values.tolist()
                    )
                    all_predicted_values_by_step_top1000[f"cycle{i}_step{j}"] = (
                        scaff_step.head(1000)[
                            f"raw_{score_component_name}"
                        ].values.tolist()
                    )
                    all_predicted_values_by_step_above05[f"cycle{i}_step{j}"] = (
                        scaff_step_above05[
                            f"raw_{score_component_name}"
                        ].values.tolist()
                    )

                    all_predicted_scores_by_step_top250[f"cycle{i}_step{j}"] = (
                        scaff_step.head(250)[f"{score_component_name}"].values.tolist()
                    )
                    all_predicted_scores_by_step_top500[f"cycle{i}_step{j}"] = (
                        scaff_step.head(500)[f"{score_component_name}"].values.tolist()
                    )
                    all_predicted_scores_by_step_top1000[f"cycle{i}_step{j}"] = (
                        scaff_step.head(1000)[f"{score_component_name}"].values.tolist()
                    )
                    all_predicted_scores_by_step_above05[f"cycle{i}_step{j}"] = (
                        scaff_step_above05[f"{score_component_name}"].values.tolist()
                    )

                    mean_predicted_value_by_step_top250[f"cycle{i}_step{j}"] = np.mean(
                        scaff_step.head(250)[
                            f"raw_{score_component_name}"
                        ].values.tolist()
                    )
                    mean_predicted_value_by_step_top500[f"cycle{i}_step{j}"] = np.mean(
                        scaff_step.head(500)[
                            f"raw_{score_component_name}"
                        ].values.tolist()
                    )
                    mean_predicted_value_by_step_top1000[f"cycle{i}_step{j}"] = np.mean(
                        scaff_step.head(1000)[
                            f"raw_{score_component_name}"
                        ].values.tolist()
                    )
                    mean_predicted_value_by_step_above05[f"cycle{i}_step{j}"] = np.mean(
                        scaff_step_above05[
                            f"raw_{score_component_name}"
                        ].values.tolist()
                    )

                    mean_predicted_score_by_step_top250[f"cycle{i}_step{j}"] = np.mean(
                        scaff_step.head(250)[f"{score_component_name}"].values.tolist()
                    )
                    mean_predicted_score_by_step_top500[f"cycle{i}_step{j}"] = np.mean(
                        scaff_step.head(500)[f"{score_component_name}"].values.tolist()
                    )
                    mean_predicted_score_by_step_top1000[f"cycle{i}_step{j}"] = np.mean(
                        scaff_step.head(1000)[f"{score_component_name}"].values.tolist()
                    )
                    mean_predicted_score_by_step_above05[f"cycle{i}_step{j}"] = np.mean(
                        scaff_step_above05[f"{score_component_name}"].values.tolist()
                    )

                    all_oracle_values_by_step_top250[f"cycle{i}_step{j}"] = (
                        scaff_step.head(250).oracle_values.tolist()
                    )
                    all_oracle_values_by_step_top500[f"cycle{i}_step{j}"] = (
                        scaff_step.head(500).oracle_values.tolist()
                    )
                    all_oracle_values_by_step_top1000[f"cycle{i}_step{j}"] = (
                        scaff_step.head(1000).oracle_values.tolist()
                    )
                    all_oracle_values_by_step_above05[f"cycle{i}_step{j}"] = (
                        scaff_step_above05.oracle_values.tolist()
                    )

                    mean_oracle_value_by_step_top250[f"cycle{i}_step{j}"] = np.mean(
                        scaff_step.head(250).oracle_values.tolist()
                    )
                    mean_oracle_value_by_step_top500[f"cycle{i}_step{j}"] = np.mean(
                        scaff_step.head(500).oracle_values.tolist()
                    )
                    mean_oracle_value_by_step_top1000[f"cycle{i}_step{j}"] = np.mean(
                        scaff_step.head(1000).oracle_values.tolist()
                    )
                    mean_oracle_value_by_step_above05[f"cycle{i}_step{j}"] = np.mean(
                        scaff_step_above05.oracle_values.tolist()
                    )

                    print("\nGet mean and distribution of oracle scores")
                    if model_type == "regression":
                        scaff_step["oracle_scores"] = [
                            utility(s, low=2, high=4)
                            for s in scaff_step.oracle_values.tolist()
                        ]
                        scaff_step_above05["oracle_scores"] = [
                            utility(s, low=2, high=4)
                            for s in scaff_step_above05.oracle_values.tolist()
                        ]

                        all_oracle_scores_by_step_top250[f"cycle{i}_step{j}"] = (
                            scaff_step.head(250).oracle_scores.tolist()
                        )
                        all_oracle_scores_by_step_top500[f"cycle{i}_step{j}"] = (
                            scaff_step.head(500).oracle_scores.tolist()
                        )
                        all_oracle_scores_by_step_top1000[f"cycle{i}_step{j}"] = (
                            scaff_step.head(1000).oracle_scores.tolist()
                        )
                        all_oracle_scores_by_step_above05[f"cycle{i}_step{j}"] = (
                            scaff_step_above05.oracle_scores.tolist()
                        )

                        mean_oracle_score_by_step_top250[f"cycle{i}_step{j}"] = np.mean(
                            scaff_step.head(250).oracle_scores.tolist()
                        )
                        mean_oracle_score_by_step_top500[f"cycle{i}_step{j}"] = np.mean(
                            scaff_step.head(500).oracle_scores.tolist()
                        )
                        mean_oracle_score_by_step_top1000[f"cycle{i}_step{j}"] = (
                            np.mean(scaff_step.head(1000).oracle_scores.tolist())
                        )
                        mean_oracle_score_by_step_above05[f"cycle{i}_step{j}"] = (
                            np.mean(scaff_step_above05.oracle_scores.tolist())
                        )

                    if model_type == "classification":
                        all_oracle_scores_by_step_top250[f"cycle{i}_step{j}"] = (
                            all_oracle_values_by_step_top250[f"cycle{i}_step{j}"]
                        )
                        all_oracle_scores_by_step_top500[f"cycle{i}_step{j}"] = (
                            all_oracle_values_by_step_top500[f"cycle{i}_step{j}"]
                        )
                        all_oracle_scores_by_step_top1000[f"cycle{i}_step{j}"] = (
                            all_oracle_values_by_step_top1000[f"cycle{i}_step{j}"]
                        )
                        all_oracle_scores_by_step_above05[f"cycle{i}_step{j}"] = (
                            all_oracle_values_by_step_above05[f"cycle{i}_step{j}"]
                        )

                        mean_oracle_score_by_step_top250[f"cycle{i}_step{j}"] = np.mean(
                            all_oracle_values_by_step_top250[f"cycle{i}_step{j}"]
                        )
                        mean_oracle_score_by_step_top500[f"cycle{i}_step{j}"] = np.mean(
                            all_oracle_values_by_step_top500[f"cycle{i}_step{j}"]
                        )
                        mean_oracle_score_by_step_top1000[f"cycle{i}_step{j}"] = (
                            np.mean(
                                all_oracle_values_by_step_top1000[f"cycle{i}_step{j}"]
                            )
                        )
                        mean_oracle_score_by_step_above05[f"cycle{i}_step{j}"] = (
                            np.mean(
                                all_oracle_values_by_step_above05[f"cycle{i}_step{j}"]
                            )
                        )

                    print("\nMeasure predictive performance of current predictor")
                    if model_type == "regression":
                        chembl_pred_values = model.predict(
                            fp_counter.get_fingerprints(chembl_data.SMILES.tolist())
                        )
                        acc_wrt_d0["mae"].append(
                            mean_absolute_error(y_test, model.predict(x_test))
                        )

                    if model_type == "classification":
                        chembl_pred_values = model.predict(
                            fp_counter.get_fingerprints(chembl_data.SMILES.tolist())
                        )
                        acc_wrt_d0["mae"].append(
                            mean_absolute_error(
                                [
                                    feedback_model.oracle_score(s)
                                    for s in test_set.SMILES.tolist()
                                ],
                                model.predict_proba(x_test)[:, 1],
                            )
                        )
                    acc_wrt_chembl["mae"].append(
                        mean_absolute_error(chembl_target_values, chembl_pred_values)
                    )

                    if model_type == "regression":
                        if len(scaff_step.head(250).SMILES.tolist()) > 0:
                            acc_wrt_current_by_step_top250["mae"].append(
                                mean_absolute_error(
                                    all_oracle_values_by_step_top250[
                                        f"cycle{i}_step{j}"
                                    ],
                                    all_predicted_values_by_step_top250[
                                        f"cycle{i}_step{j}"
                                    ],
                                )
                            )
                            acc_wrt_current_by_step_top500["mae"].append(
                                mean_absolute_error(
                                    all_oracle_values_by_step_top500[
                                        f"cycle{i}_step{j}"
                                    ],
                                    all_predicted_values_by_step_top500[
                                        f"cycle{i}_step{j}"
                                    ],
                                )
                            )
                            acc_wrt_current_by_step_top1000["mae"].append(
                                mean_absolute_error(
                                    all_oracle_values_by_step_top1000[
                                        f"cycle{i}_step{j}"
                                    ],
                                    all_predicted_values_by_step_top1000[
                                        f"cycle{i}_step{j}"
                                    ],
                                )
                            )
                        if len(scaff_step_above05.SMILES.tolist()) > 0:
                            acc_wrt_current_by_step_above05["mae"].append(
                                mean_absolute_error(
                                    all_oracle_values_by_step_above05[
                                        f"cycle{i}_step{j}"
                                    ],
                                    all_predicted_values_by_step_above05[
                                        f"cycle{i}_step{j}"
                                    ],
                                )
                            )

                    if model_type == "classification":
                        acc_wrt_current_by_step_top250["f1"].append(
                            f1_score(
                                np.round(
                                    all_oracle_values_by_step_top250[
                                        f"cycle{i}_step{j}"
                                    ]
                                ),
                                np.round(
                                    all_predicted_values_by_step_top250[
                                        f"cycle{i}_step{j}"
                                    ]
                                ),
                            )
                        )
                        acc_wrt_current_by_step_top500["f1"].append(
                            f1_score(
                                np.round(
                                    all_oracle_values_by_step_top500[
                                        f"cycle{i}_step{j}"
                                    ]
                                ),
                                np.round(
                                    all_predicted_values_by_step_top500[
                                        f"cycle{i}_step{j}"
                                    ]
                                ),
                            )
                        )
                        acc_wrt_current_by_step_top1000["f1"].append(
                            f1_score(
                                np.round(
                                    all_oracle_values_by_step_top1000[
                                        f"cycle{i}_step{j}"
                                    ]
                                ),
                                np.round(
                                    all_predicted_values_by_step_top1000[
                                        f"cycle{i}_step{j}"
                                    ]
                                ),
                            )
                        )
                        acc_wrt_current_by_step_above05["f1"].append(
                            f1_score(
                                np.round(
                                    all_oracle_values_by_step_above05[
                                        f"cycle{i}_step{j}"
                                    ]
                                ),
                                np.round(
                                    all_predicted_values_by_step_above05[
                                        f"cycle{i}_step{j}"
                                    ]
                                ),
                            )
                        )
                        try:
                            acc_wrt_current_by_step_top250["roc_auc"].append(
                                roc_auc_score(
                                    np.round(
                                        all_oracle_values_by_step_top250[
                                            f"cycle{i}_step{j}"
                                        ]
                                    ),
                                    np.round(
                                        all_predicted_values_by_step_top250[
                                            f"cycle{i}_step{j}"
                                        ]
                                    ),
                                )
                            )
                            acc_wrt_current_by_step_top500["roc_auc"].append(
                                roc_auc_score(
                                    np.round(
                                        all_oracle_values_by_step_top500[
                                            f"cycle{i}_step{j}"
                                        ]
                                    ),
                                    np.round(
                                        all_predicted_values_by_step_top500[
                                            f"cycle{i}_step{j}"
                                        ]
                                    ),
                                )
                            )
                            acc_wrt_current_by_step_top1000["roc_auc"].append(
                                roc_auc_score(
                                    np.round(
                                        all_oracle_values_by_step_top1000[
                                            f"cycle{i}_step{j}"
                                        ]
                                    ),
                                    np.round(
                                        all_predicted_values_by_step_top1000[
                                            f"cycle{i}_step{j}"
                                        ]
                                    ),
                                )
                            )
                            acc_wrt_current_by_step_above05["roc_auc"].append(
                                roc_auc_score(
                                    np.round(
                                        all_oracle_values_by_step_above05[
                                            f"cycle{i}_step{j}"
                                        ]
                                    ),
                                    np.round(
                                        all_predicted_values_by_step_above05[
                                            f"cycle{i}_step{j}"
                                        ]
                                    ),
                                )
                            )
                        except:
                            acc_wrt_current_by_step_top250["roc_auc"].append(0.0)
                            acc_wrt_current_by_step_top500["roc_auc"].append(0.0)
                            acc_wrt_current_by_step_top1000["roc_auc"].append(0.0)
                            acc_wrt_current_by_step_above05["roc_auc"].append(0.0)

                    print("\nGet MOSES structural distance and diversity metrics")
                    if i == 0:
                        if len(scaff_step.head(250).SMILES.tolist()) > 0:
                            snn_wrt_previous_set_by_step_top250.append(
                                get_pairwise_structural_metrics(
                                    scaff_step.head(250).SMILES.tolist(),
                                    train_smiles,
                                    "snn",
                                )
                            )
                            frag_wrt_previous_set_by_step_top250.append(
                                get_pairwise_structural_metrics(
                                    scaff_step.head(250).SMILES.tolist(),
                                    train_smiles,
                                    "frag",
                                )
                            )
                            fcd_wrt_previous_set_by_step_top250.append(
                                get_pairwise_structural_metrics(
                                    scaff_step.head(250).SMILES.tolist(),
                                    train_smiles,
                                    "fcd",
                                )
                            )
                        print("\nDONE")

                        if len(scaff_step.head(500).SMILES.tolist()) > 0:
                            snn_wrt_previous_set_by_step_top500.append(
                                get_pairwise_structural_metrics(
                                    scaff_step.head(500).SMILES.tolist(),
                                    train_smiles,
                                    "snn",
                                )
                            )
                            frag_wrt_previous_set_by_step_top500.append(
                                get_pairwise_structural_metrics(
                                    scaff_step.head(500).SMILES.tolist(),
                                    train_smiles,
                                    "frag",
                                )
                            )
                            fcd_wrt_previous_set_by_step_top500.append(
                                get_pairwise_structural_metrics(
                                    scaff_step.head(500).SMILES.tolist(),
                                    train_smiles,
                                    "fcd",
                                )
                            )
                        print("\nDONE")

                        if len(scaff_step.head(1000).SMILES.tolist()) > 0:
                            snn_wrt_previous_set_by_step_top1000.append(
                                get_pairwise_structural_metrics(
                                    scaff_step.head(1000).SMILES.tolist(),
                                    train_smiles,
                                    "snn",
                                )
                            )
                            frag_wrt_previous_set_by_step_top1000.append(
                                get_pairwise_structural_metrics(
                                    scaff_step.head(1000).SMILES.tolist(),
                                    train_smiles,
                                    "frag",
                                )
                            )
                            fcd_wrt_previous_set_by_step_top1000.append(
                                get_pairwise_structural_metrics(
                                    scaff_step.head(1000).SMILES.tolist(),
                                    train_smiles,
                                    "fcd",
                                )
                            )
                        print("\nDONE")

                        if len(scaff_step_above05.SMILES.tolist()) > 0:
                            snn_wrt_previous_set_by_step_above05.append(
                                get_pairwise_structural_metrics(
                                    scaff_step_above05.SMILES.tolist(),
                                    train_smiles,
                                    "snn",
                                )
                            )
                            frag_wrt_previous_set_by_step_above05.append(
                                get_pairwise_structural_metrics(
                                    scaff_step_above05.SMILES.tolist(),
                                    train_smiles,
                                    "frag",
                                )
                            )
                            fcd_wrt_previous_set_by_step_above05.append(
                                get_pairwise_structural_metrics(
                                    scaff_step_above05.SMILES.tolist(),
                                    train_smiles,
                                    "fcd",
                                )
                            )
                        print("\nDONE")
                    else:
                        if (
                            len(scaff_step.head(250).SMILES.tolist()) > 0
                            and len(prev_smiles_top250) > 0
                        ):
                            snn_wrt_previous_set_by_step_top250.append(
                                get_pairwise_structural_metrics(
                                    scaff_step.head(250).SMILES.tolist(),
                                    prev_smiles_top250,
                                    "snn",
                                )
                            )
                            frag_wrt_previous_set_by_step_top250.append(
                                get_pairwise_structural_metrics(
                                    scaff_step.head(250).SMILES.tolist(),
                                    prev_smiles_top250,
                                    "frag",
                                )
                            )
                            fcd_wrt_previous_set_by_step_top250.append(
                                get_pairwise_structural_metrics(
                                    scaff_step.head(250).SMILES.tolist(),
                                    prev_smiles_top250,
                                    "fcd",
                                )
                            )
                        print("\nDONE")

                        if (
                            len(scaff_step.head(500).SMILES.tolist()) > 0
                            and len(prev_smiles_top500) > 0
                        ):
                            snn_wrt_previous_set_by_step_top500.append(
                                get_pairwise_structural_metrics(
                                    scaff_step.head(500).SMILES.tolist(),
                                    prev_smiles_top500,
                                    "snn",
                                )
                            )
                            frag_wrt_previous_set_by_step_top500.append(
                                get_pairwise_structural_metrics(
                                    scaff_step.head(500).SMILES.tolist(),
                                    prev_smiles_top500,
                                    "frag",
                                )
                            )
                            fcd_wrt_previous_set_by_step_top500.append(
                                get_pairwise_structural_metrics(
                                    scaff_step.head(500).SMILES.tolist(),
                                    prev_smiles_top500,
                                    "fcd",
                                )
                            )
                        print("\nDONE")

                        if (
                            len(scaff_step.head(1000).SMILES.tolist()) > 0
                            and len(prev_smiles_top1000) > 0
                        ):
                            snn_wrt_previous_set_by_step_top1000.append(
                                get_pairwise_structural_metrics(
                                    scaff_step.head(1000).SMILES.tolist(),
                                    prev_smiles_top1000,
                                    "snn",
                                )
                            )
                            frag_wrt_previous_set_by_step_top1000.append(
                                get_pairwise_structural_metrics(
                                    scaff_step.head(1000).SMILES.tolist(),
                                    prev_smiles_top1000,
                                    "frag",
                                )
                            )
                            fcd_wrt_previous_set_by_step_top1000.append(
                                get_pairwise_structural_metrics(
                                    scaff_step.head(1000).SMILES.tolist(),
                                    prev_smiles_top1000,
                                    "fcd",
                                )
                            )
                        print("\nDONE")

                        if (
                            len(scaff_step_above05.SMILES.tolist()) > 0
                            and len(prev_smiles_above05) > 0
                        ):
                            snn_wrt_previous_set_by_step_above05.append(
                                get_pairwise_structural_metrics(
                                    scaff_step_above05.SMILES.tolist(),
                                    prev_smiles_above05,
                                    "snn",
                                )
                            )
                            frag_wrt_previous_set_by_step_above05.append(
                                get_pairwise_structural_metrics(
                                    scaff_step_above05.SMILES.tolist(),
                                    prev_smiles_above05,
                                    "frag",
                                )
                            )
                            fcd_wrt_previous_set_by_step_above05.append(
                                get_pairwise_structural_metrics(
                                    scaff_step_above05.SMILES.tolist(),
                                    prev_smiles_above05,
                                    "fcd",
                                )
                            )
                        print("\nDONE")

                    if acq != "None":
                        if len(scaff_step.head(250).SMILES.tolist()) > 0:
                            snn_wrt_queries_by_step_top250.append(
                                get_pairwise_structural_metrics(
                                    scaff_step.head(250).SMILES.tolist(),
                                    queried_smiles,
                                    "snn",
                                )
                            )
                            snn_wrt_queries_by_step_top500.append(
                                get_pairwise_structural_metrics(
                                    scaff_step.head(500).SMILES.tolist(),
                                    queried_smiles,
                                    "snn",
                                )
                            )
                            snn_wrt_queries_by_step_top1000.append(
                                get_pairwise_structural_metrics(
                                    scaff_step.head(1000).SMILES.tolist(),
                                    queried_smiles,
                                    "snn",
                                )
                            )
                        if len(scaff_step_above05.SMILES.tolist()) > 0:
                            snn_wrt_queries_by_step_above05.append(
                                get_pairwise_structural_metrics(
                                    scaff_step_above05.SMILES.tolist(),
                                    queried_smiles,
                                    "snn",
                                )
                            )
                        print("\nDONE")

                        if len(scaff_step.head(250).SMILES.tolist()) > 0:
                            frag_wrt_queries_by_step_top250.append(
                                get_pairwise_structural_metrics(
                                    scaff_step.head(250).SMILES.tolist(),
                                    queried_smiles,
                                    "frag",
                                )
                            )
                            frag_wrt_queries_by_step_top500.append(
                                get_pairwise_structural_metrics(
                                    scaff_step.head(500).SMILES.tolist(),
                                    queried_smiles,
                                    "frag",
                                )
                            )
                            frag_wrt_queries_by_step_top1000.append(
                                get_pairwise_structural_metrics(
                                    scaff_step.head(1000).SMILES.tolist(),
                                    queried_smiles,
                                    "frag",
                                )
                            )
                        if len(scaff_step_above05.SMILES.tolist()) > 0:
                            frag_wrt_queries_by_step_above05.append(
                                get_pairwise_structural_metrics(
                                    scaff_step_above05.SMILES.tolist(),
                                    queried_smiles,
                                    "frag",
                                )
                            )
                        print("\nDONE")

                        if len(scaff_step.head(250).SMILES.tolist()) > 0:
                            fcd_wrt_queries_by_step_top250.append(
                                get_pairwise_structural_metrics(
                                    scaff_step.head(250).SMILES.tolist(),
                                    queried_smiles,
                                    "fcd",
                                )
                            )
                            fcd_wrt_queries_by_step_top500.append(
                                get_pairwise_structural_metrics(
                                    scaff_step.head(500).SMILES.tolist(),
                                    queried_smiles,
                                    "fcd",
                                )
                            )
                            fcd_wrt_queries_by_step_top1000.append(
                                get_pairwise_structural_metrics(
                                    scaff_step.head(1000).SMILES.tolist(),
                                    queried_smiles,
                                    "fcd",
                                )
                            )
                        if len(scaff_step_above05.SMILES.tolist()) > 0:
                            fcd_wrt_queries_by_step_above05.append(
                                get_pairwise_structural_metrics(
                                    scaff_step_above05.SMILES.tolist(),
                                    queried_smiles,
                                    "fcd",
                                )
                            )
                        print("\nDONE")

                    if len(scaff_step.head(250).SMILES.tolist()) > 0:
                        internal_diversity_by_step_top250.append(
                            internal_diversity(scaff_step.head(250).SMILES.tolist())
                        )
                        internal_diversity_by_step_top500.append(
                            internal_diversity(scaff_step.head(500).SMILES.tolist())
                        )
                        internal_diversity_by_step_top1000.append(
                            internal_diversity(scaff_step.head(1000).SMILES.tolist())
                        )
                    if len(scaff_step_above05.SMILES.tolist()) > 0:
                        internal_diversity_by_step_above05.append(
                            internal_diversity(scaff_step_above05.SMILES.tolist())
                        )
                    print("\nDONE")

                    if len(scaff_step.head(250).SMILES.tolist()) > 0:
                        snn_wrt_train_by_step_top250.append(
                            get_pairwise_structural_metrics(
                                scaff_step.head(250).SMILES.tolist(),
                                train_smiles,
                                "snn",
                            )
                        )
                        snn_wrt_train_by_step_top500.append(
                            get_pairwise_structural_metrics(
                                scaff_step.head(500).SMILES.tolist(),
                                train_smiles,
                                "snn",
                            )
                        )
                        snn_wrt_train_by_step_top1000.append(
                            get_pairwise_structural_metrics(
                                scaff_step.head(1000).SMILES.tolist(),
                                train_smiles,
                                "snn",
                            )
                        )
                    if len(scaff_step_above05.SMILES.tolist()) > 0:
                        snn_wrt_train_by_step_above05.append(
                            get_pairwise_structural_metrics(
                                scaff_step_above05.SMILES.tolist(), train_smiles, "snn"
                            )
                        )
                    print("\nDONE")

                    if len(scaff_step.head(250).SMILES.tolist()) > 0:
                        frag_wrt_train_by_step_top250.append(
                            get_pairwise_structural_metrics(
                                scaff_step.head(250).SMILES.tolist(),
                                train_smiles,
                                "frag",
                            )
                        )
                        frag_wrt_train_by_step_top500.append(
                            get_pairwise_structural_metrics(
                                scaff_step.head(500).SMILES.tolist(),
                                train_smiles,
                                "frag",
                            )
                        )
                        frag_wrt_train_by_step_top1000.append(
                            get_pairwise_structural_metrics(
                                scaff_step.head(1000).SMILES.tolist(),
                                train_smiles,
                                "frag",
                            )
                        )
                    if len(scaff_step_above05.SMILES.tolist()) > 0:
                        frag_wrt_train_by_step_above05.append(
                            get_pairwise_structural_metrics(
                                scaff_step_above05.SMILES.tolist(), train_smiles, "frag"
                            )
                        )
                    print("\nDONE")

                    if len(scaff_step.head(250).SMILES.tolist()) > 0:
                        fcd_wrt_train_by_step_top250.append(
                            get_pairwise_structural_metrics(
                                scaff_step.head(250).SMILES.tolist(),
                                train_smiles,
                                "fcd",
                            )
                        )
                        fcd_wrt_train_by_step_top500.append(
                            get_pairwise_structural_metrics(
                                scaff_step.head(500).SMILES.tolist(),
                                train_smiles,
                                "fcd",
                            )
                        )
                        fcd_wrt_train_by_step_top1000.append(
                            get_pairwise_structural_metrics(
                                scaff_step.head(1000).SMILES.tolist(),
                                train_smiles,
                                "fcd",
                            )
                        )
                    if len(scaff_step_above05.SMILES.tolist()) > 0:
                        fcd_wrt_train_by_step_above05.append(
                            get_pairwise_structural_metrics(
                                scaff_step_above05.SMILES.tolist(), train_smiles, "fcd"
                            )
                        )
                    print("\nDONE")

                    print(
                        "\nGet MOSES distributions of key chemical properties for the generated molecules"
                    )

                    if len(scaff_step.head(250).SMILES.tolist()) > 0:
                        qed_by_step_top250.append(
                            np.mean(
                                [
                                    QED(Chem.MolFromSmiles(s))
                                    for s in scaff_step.head(250).SMILES.tolist()
                                ]
                            )
                        )
                        qed_by_step_top500.append(
                            np.mean(
                                [
                                    QED(Chem.MolFromSmiles(s))
                                    for s in scaff_step.head(500).SMILES.tolist()
                                ]
                            )
                        )
                        qed_by_step_top1000.append(
                            np.mean(
                                [
                                    QED(Chem.MolFromSmiles(s))
                                    for s in scaff_step.head(1000).SMILES.tolist()
                                ]
                            )
                        )
                    if len(scaff_step_above05.SMILES.tolist()) > 0:
                        qed_by_step_above05.append(
                            np.mean(
                                [
                                    QED(Chem.MolFromSmiles(s))
                                    for s in scaff_step_above05.SMILES.tolist()
                                ]
                            )
                        )
                    print("\nDONE")

                    if len(scaff_step.head(250).SMILES.tolist()) > 0:
                        mw_by_step_top250.append(
                            np.mean(
                                [
                                    Weight(Chem.MolFromSmiles(s))
                                    for s in scaff_step.head(250).SMILES.tolist()
                                ]
                            )
                        )
                        mw_by_step_top500.append(
                            np.mean(
                                [
                                    Weight(Chem.MolFromSmiles(s))
                                    for s in scaff_step.head(500).SMILES.tolist()
                                ]
                            )
                        )
                        mw_by_step_top1000.append(
                            np.mean(
                                [
                                    Weight(Chem.MolFromSmiles(s))
                                    for s in scaff_step.head(1000).SMILES.tolist()
                                ]
                            )
                        )
                    if len(scaff_step_above05.SMILES.tolist()) > 0:
                        mw_by_step_above05.append(
                            np.mean(
                                [
                                    Weight(Chem.MolFromSmiles(s))
                                    for s in scaff_step_above05.SMILES.tolist()
                                ]
                            )
                        )
                    print("\nDONE")

                    if len(scaff_step.head(250).SMILES.tolist()) > 0:
                        logp_by_step_top250.append(
                            np.mean(
                                [
                                    logP(Chem.MolFromSmiles(s))
                                    for s in scaff_step.head(250).SMILES.tolist()
                                ]
                            )
                        )
                        logp_by_step_top500.append(
                            np.mean(
                                [
                                    logP(Chem.MolFromSmiles(s))
                                    for s in scaff_step.head(500).SMILES.tolist()
                                ]
                            )
                        )
                        logp_by_step_top1000.append(
                            np.mean(
                                [
                                    logP(Chem.MolFromSmiles(s))
                                    for s in scaff_step.head(1000).SMILES.tolist()
                                ]
                            )
                        )
                    if len(scaff_step_above05.SMILES.tolist()) > 0:
                        logp_by_step_above05.append(
                            np.mean(
                                [
                                    logP(Chem.MolFromSmiles(s))
                                    for s in scaff_step_above05.SMILES.tolist()
                                ]
                            )
                        )
                    print("\nDONE")

                    if len(scaff_step.head(250).SMILES.tolist()) > 0:
                        sa_by_step_top250.append(
                            np.mean(
                                [
                                    SA(Chem.MolFromSmiles(s))
                                    for s in scaff_step.head(250).SMILES.tolist()
                                ]
                            )
                        )
                        sa_by_step_top500.append(
                            np.mean(
                                [
                                    SA(Chem.MolFromSmiles(s))
                                    for s in scaff_step.head(500).SMILES.tolist()
                                ]
                            )
                        )
                        sa_by_step_top1000.append(
                            np.mean(
                                [
                                    SA(Chem.MolFromSmiles(s))
                                    for s in scaff_step.head(1000).SMILES.tolist()
                                ]
                            )
                        )
                    if len(scaff_step_above05.SMILES.tolist()) > 0:
                        sa_by_step_above05.append(
                            np.mean(
                                [
                                    SA(Chem.MolFromSmiles(s))
                                    for s in scaff_step_above05.SMILES.tolist()
                                ]
                            )
                        )
                    print("\nDONE")

                    print("\nGet MOSES uniqueness/validity/novelty metrics")
                    if len(scaff_step.head(250).SMILES.tolist()) > 0:
                        uniqueness_by_step_top250.append(
                            fraction_unique(scaff_step.head(250).SMILES.tolist())
                        )
                        uniqueness_by_step_top500.append(
                            fraction_unique(scaff_step.head(500).SMILES.tolist())
                        )
                        uniqueness_by_step_top1000.append(
                            fraction_unique(scaff_step.head(1000).SMILES.tolist())
                        )
                    if len(scaff_step_above05.SMILES.tolist()) > 0:
                        uniqueness_by_step_above05.append(
                            fraction_unique(scaff_step_above05.SMILES.tolist())
                        )

                    if len(scaff_step.head(250).SMILES.tolist()) > 0:
                        novelty_by_step_top250.append(
                            fraction_unique(scaff_step.head(250).SMILES.tolist())
                        )
                        novelty_by_step_top500.append(
                            fraction_unique(scaff_step.head(500).SMILES.tolist())
                        )
                        novelty_by_step_top1000.append(
                            fraction_unique(scaff_step.head(1000).SMILES.tolist())
                        )
                    if len(scaff_step_above05.SMILES.tolist()) > 0:
                        novelty_by_step_above05.append(
                            fraction_unique(scaff_step_above05.SMILES.tolist())
                        )

                    prev_smiles_top250 = scaff_step.head(250).SMILES.tolist()
                    prev_smiles_top500 = scaff_step.head(500).SMILES.tolist()
                    prev_smiles_top1000 = scaff_step.head(1000).SMILES.tolist()
                    prev_smiles_above05 = scaff_step_above05.SMILES.tolist()

    results = {
        "acc_wrt_d0": acc_wrt_d0,
        "acc_wrt_chembl": acc_wrt_chembl,
        "scaffold_memory": {
            "all_oracle_values_top250": all_oracle_values_by_step_top250,
            "all_oracle_values_top500": all_oracle_values_by_step_top500,
            "all_oracle_values_top1000": all_oracle_values_by_step_top1000,
            "all_oracle_values_above05": all_oracle_values_by_step_above05,
            "mean_oracle_value_top250": mean_oracle_value_by_step_top250,
            "mean_oracle_value_top500": mean_oracle_value_by_step_top500,
            "mean_oracle_value_top1000": mean_oracle_value_by_step_top1000,
            "mean_oracle_value_above05": mean_oracle_value_by_step_above05,
            "all_oracle_scores_top250": all_oracle_scores_by_step_top250,
            "all_oracle_scores_top500": all_oracle_scores_by_step_top500,
            "all_oracle_scores_top1000": all_oracle_scores_by_step_top1000,
            "all_oracle_scores_above05": all_oracle_scores_by_step_above05,
            "mean_oracle_score_top250": mean_oracle_score_by_step_top250,
            "mean_oracle_score_top500": mean_oracle_score_by_step_top500,
            "mean_oracle_score_top1000": mean_oracle_score_by_step_top1000,
            "mean_oracle_score_above05": mean_oracle_score_by_step_above05,
            "all_predicted_values_top250": all_predicted_values_by_step_top250,
            "all_predicted_values_top500": all_predicted_values_by_step_top500,
            "all_predicted_values_top1000": all_predicted_values_by_step_top1000,
            "all_predicted_values_above05": all_predicted_values_by_step_above05,
            "mean_predicted_value_top250": mean_predicted_value_by_step_top250,
            "mean_predicted_value_top500": mean_predicted_value_by_step_top500,
            "mean_predicted_value_top1000": mean_predicted_value_by_step_top1000,
            "mean_predicted_value_above05": mean_predicted_value_by_step_above05,
            "all_predicted_scores_top250": all_predicted_scores_by_step_top250,
            "all_predicted_scores_top500": all_predicted_scores_by_step_top500,
            "all_predicted_scores_top1000": all_predicted_scores_by_step_top1000,
            "all_predicted_scores_above05": all_predicted_scores_by_step_above05,
            "mean_predicted_score_top250": mean_predicted_score_by_step_top250,
            "mean_predicted_score_top500": mean_predicted_score_by_step_top500,
            "mean_predicted_score_top1000": mean_predicted_score_by_step_top1000,
            "mean_predicted_score_above05": mean_predicted_score_by_step_above05,
            "acc_wrt_current_top250": acc_wrt_current_by_step_top250,
            "acc_wrt_current_top500": acc_wrt_current_by_step_top500,
            "acc_wrt_current_top1000": acc_wrt_current_by_step_top1000,
            "acc_wrt_current_above05": acc_wrt_current_by_step_above05,
            "snn_wrt_queries_top250": snn_wrt_queries_by_step_top250,
            "snn_wrt_queries_top500": snn_wrt_queries_by_step_top500,
            "snn_wrt_queries_top1000": snn_wrt_queries_by_step_top1000,
            "snn_wrt_queries_above05": snn_wrt_queries_by_step_above05,
            "snn_wrt_train_top250": snn_wrt_train_by_step_top250,
            "snn_wrt_train_top500": snn_wrt_train_by_step_top500,
            "snn_wrt_train_top1000": snn_wrt_train_by_step_top1000,
            "snn_wrt_train_above05": snn_wrt_train_by_step_above05,
            "frag_wrt_queries_top250": frag_wrt_queries_by_step_top250,
            "frag_wrt_queries_top500": frag_wrt_queries_by_step_top500,
            "frag_wrt_queries_top1000": frag_wrt_queries_by_step_top1000,
            "frag_wrt_queries_above05": frag_wrt_queries_by_step_above05,
            "frag_wrt_train_top250": frag_wrt_train_by_step_top250,
            "frag_wrt_train_top500": frag_wrt_train_by_step_top500,
            "frag_wrt_train_top1000": frag_wrt_train_by_step_top1000,
            "frag_wrt_train_above05": frag_wrt_train_by_step_above05,
            "fcd_wrt_queries_top250": fcd_wrt_queries_by_step_top250,
            "fcd_wrt_queries_top500": fcd_wrt_queries_by_step_top500,
            "fcd_wrt_queries_top1000": fcd_wrt_queries_by_step_top1000,
            "fcd_wrt_queries_above05": fcd_wrt_queries_by_step_above05,
            "fcd_wrt_train_top250": fcd_wrt_train_by_step_top250,
            "fcd_wrt_train_top500": fcd_wrt_train_by_step_top500,
            "fcd_wrt_train_top1000": fcd_wrt_train_by_step_top1000,
            "fcd_wrt_train_above05": fcd_wrt_train_by_step_above05,
            "snn_wrt_previous_set_top250": snn_wrt_previous_set_by_step_top250,
            "snn_wrt_previous_set_top500": snn_wrt_previous_set_by_step_top500,
            "snn_wrt_previous_set_top1000": snn_wrt_previous_set_by_step_top1000,
            "snn_wrt_previous_set_above05": snn_wrt_previous_set_by_step_above05,
            "frag_wrt_previous_set_top250": frag_wrt_previous_set_by_step_top250,
            "frag_wrt_previous_set_top500": frag_wrt_previous_set_by_step_top500,
            "frag_wrt_previous_set_top1000": frag_wrt_previous_set_by_step_top1000,
            "frag_wrt_previous_set_above05": frag_wrt_previous_set_by_step_above05,
            "fcd_wrt_previous_set_top250": fcd_wrt_previous_set_by_step_top250,
            "fcd_wrt_previous_set_top500": fcd_wrt_previous_set_by_step_top500,
            "fcd_wrt_previous_set_top1000": fcd_wrt_previous_set_by_step_top1000,
            "fcd_wrt_previous_set_above05": fcd_wrt_previous_set_by_step_above05,
            "internal_diversity_top250": internal_diversity_by_step_top250,
            "internal_diversity_top500": internal_diversity_by_step_top500,
            "internal_diversity_top1000": internal_diversity_by_step_top1000,
            "internal_diversity_above05": internal_diversity_by_step_above05,
            "uniqueness_top250": uniqueness_by_step_top250,
            "uniqueness_top500": uniqueness_by_step_top500,
            "uniqueness_top1000": uniqueness_by_step_top1000,
            "uniqueness_above05": uniqueness_by_step_above05,
            "novelty_top250": novelty_by_step_top250,
            "novelty_top500": novelty_by_step_top500,
            "novelty_top1000": novelty_by_step_top1000,
            "novelty_above05": novelty_by_step_above05,
            "moses_chemical_properties_top250": {
                "MW": mw_by_step_top250,
                "logP": logp_by_step_top250,
                "QED": qed_by_step_top250,
                "SA": sa_by_step_top250,
            },
            "moses_chemical_properties_top500": {
                "MW": mw_by_step_top500,
                "logP": logp_by_step_top500,
                "QED": qed_by_step_top500,
                "SA": sa_by_step_top500,
            },
            "moses_chemical_properties_top1000": {
                "MW": mw_by_step_top1000,
                "logP": logp_by_step_top1000,
                "QED": qed_by_step_top1000,
                "SA": sa_by_step_top1000,
            },
            "moses_chemical_properties_above05": {
                "MW": mw_by_step_above05,
                "logP": logp_by_step_above05,
                "QED": qed_by_step_above05,
                "SA": sa_by_step_above05,
            },
        },
        "sample": {
            "all_oracle_values": sample_all_oracle_values_by_step,
            "mean_oracle_value": sample_mean_oracle_value_by_step,
            "all_oracle_scores": sample_all_oracle_scores_by_step,
            "mean_oracle_score": sample_mean_oracle_score_by_step,
            "all_predicted_values": sample_all_predicted_values_by_step,
            "mean_predicted_value": sample_mean_predicted_value_by_step,
            "all_predicted_scores": sample_all_predicted_scores_by_step,
            "mean_predicted_score": sample_mean_predicted_score_by_step,
            "acc_wrt_current": sample_acc_wrt_current_by_step,
            "snn_wrt_queries": sample_snn_wrt_queries_by_step,
            "snn_wrt_train": sample_snn_wrt_train_by_step,
            "frag_wrt_queries": sample_frag_wrt_queries_by_step,
            "frag_wrt_train": sample_frag_wrt_train_by_step,
            "fcd_wrt_queries": sample_fcd_wrt_queries_by_step,
            "fcd_wrt_train": sample_fcd_wrt_train_by_step,
            "snn_wrt_previous_set": sample_snn_wrt_previous_set_by_step,
            "frag_wrt_previous_set": sample_frag_wrt_previous_set_by_step,
            "fcd_wrt_previous_set": sample_fcd_wrt_previous_set_by_step,
            "internal_diversity": sample_internal_diversity_by_step,
            "uniqueness": sample_uniqueness_by_step,
            "novelty": sample_novelty_by_step,
            "moses_chemical_properties": {
                "MW": sample_mw_by_step,
                "logP": sample_logp_by_step,
                "QED": sample_qed_by_step,
                "SA": sample_sa_by_step,
            },
        },
    }

    print(f"\nExit and save results in {path_to_output_dir}/data_for_figures")

    return results


@click.command()
@click.option(
    "--job_name",
    "-j",
    type=str,
    help="Name given to the experiment (e.g., demo_drd2 or demo_drd2_multi)",
)
@click.option("--seed", "-s", type=int, help="Seed used for the experiment")
@click.option("--rounds", "-R", default=4, type=int, help="Number of rounds")
@click.option(
    "--al_iterations", "-T", default=5, type=int, help="Number of AL iterations"
)
@click.option(
    "--n_opt_steps", default=250, type=int, help="Number of REINVENT optimization steps"
)
@click.option(
    "--task",
    type=click.Choice(["logp", "drd2"]),
    help="Goal of the molecule generation",
)
@click.option(
    "--model_type",
    type=click.Choice(["regression", "classification"]),
    help="Whether the scoring model is a regressor or classifier",
)
@click.option(
    "--path_to_predictor",
    type=str,
    help="Path to pickled target property predictor (without the .pkl extension)",
)
@click.option(
    "--path_to_simulator",
    type=str,
    help="Path to pickled oracle or assay simulator (without the .pkl extension)",
)
@click.option(
    "--path_to_train_data",
    type=str,
    help="Path to csv file of initial predictor training dataset (without the .csv extension)",
)
@click.option(
    "--path_to_test_data",
    type=str,
    help="Path to csv file of initial predictor test dataset (without the .csv extension)",
)
@click.option(
    "--score_component_name",
    type=str,
    help="Name given to the predictive scoring component in REINVENT output files",
)
@click.option(
    "--path_to_reinvent_env",
    type=str,
    help="Path to python virtual environment for Reinvent V3.2",
)
@click.option(
    "--path_to_reinvent_repo",
    type=str,
    help="Path to cloned Reinvent V3.2 repository",
)
@click.option(
    "--path_to_output_dir",
    type=str,
    help="Path to directory where to store the results",
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
    "--sigma_noise",
    default=0.0,
    type=float,
    help="Sigma value for the noise term in the expert model (if 0, expert model = Oracle)",
)
@click.option(
    "--by_step",
    type=bool,
    default=True,
    help="Whether to calculate all metrics at each REINVENT step",
)
def main(
    job_name,
    seed,
    rounds,
    al_iterations,
    n_opt_steps,
    task,
    model_type,
    path_to_predictor,
    path_to_simulator,
    path_to_train_data,
    path_to_test_data,
    score_component_name,
    path_to_reinvent_env,
    path_to_reinvent_repo,
    path_to_output_dir,
    acquisition,
    n_queries,
    sigma_noise,
    by_step,
):

    if acquisition != "None":
        results = get_result_data_for_figures(
            job_name,
            task,
            path_to_predictor,
            path_to_simulator,
            path_to_train_data,
            path_to_test_data,
            path_to_reinvent_env,
            path_to_reinvent_repo,
            path_to_output_dir,
            seed,
            rounds,
            al_iterations,
            n_opt_steps=n_opt_steps,
            acq=acquisition,
            sigma_noise=sigma_noise,
            n_queries=n_queries,
            score_component_name=score_component_name,
            model_type=model_type,
            by_step=by_step,
        )

        filename = f"{task}_results_R{rounds}_Steps{n_opt_steps}_T{al_iterations}_n{n_queries}_{acquisition}_noise{sigma_noise}_seed{seed}.pkl"
        if "multi" in job_name:
            filename = f"{task}_results_multi_R{rounds}_Steps{n_opt_steps}_T{al_iterations}_n{n_queries}_{acquisition}_noise{sigma_noise}_seed{seed}.pkl"
        if "prior" in job_name:
            filename = f"{task}_results_prior_R{rounds}_Steps{n_opt_steps}_T{al_iterations}_n{n_queries}_{acquisition}_noise{sigma_noise}_seed{seed}.pkl"
        if "tanimoto" in job_name:
            filename = f"{task}_results_tanimoto_R{rounds}_Steps{n_opt_steps}_T{al_iterations}_n{n_queries}_{acquisition}_noise{sigma_noise}_seed{seed}.pkl"

    else:
        results = get_result_data_for_figures(
            job_name,
            task,
            path_to_predictor,
            path_to_simulator,
            path_to_train_data,
            path_to_test_data,
            path_to_reinvent_env,
            path_to_reinvent_repo,
            path_to_output_dir,
            seed,
            rounds,
            al_iterations,
            n_opt_steps=n_opt_steps,
            acq=acquisition,
            sigma_noise=sigma_noise,
            n_queries=n_queries,
            score_component_name=score_component_name,
            model_type=model_type,
            by_step=by_step,
        )

        filename = f"{task}_results_R{rounds}_Steps{n_opt_steps}_base_seed{seed}.pkl"
        if "multi" in job_name:
            filename = (
                f"{task}_results_multi_R{rounds}_Steps{n_opt_steps}_base_seed{seed}.pkl"
            )
        if "prior" in job_name:
            filename = (
                f"{task}_results_prior_R{rounds}_Steps{n_opt_steps}_base_seed{seed}.pkl"
            )
        if "tanimoto" in job_name:
            filename = f"{task}_results_tanimoto_R{rounds}_Steps{n_opt_steps}_base_seed{seed}.pkl"
        if "calib" in job_name:
            filename = f"{task}_results_calibratedPredictor_R{rounds}_Steps{n_opt_steps}_base_seed{seed}.pkl"

    if not os.path.exists(f"{path_to_output_dir}/data_for_figures"):
        os.makedirs(f"{path_to_output_dir}/data_for_figures")

    with open(
        os.path.join(f"{path_to_output_dir}/data_for_figures", filename), "wb"
    ) as handle:
        pickle.dump(results, handle, protocol=pickle.HIGHEST_PROTOCOL)


if __name__ == "__main__":
    main()
