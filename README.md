[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.13988149.svg)](https://doi.org/10.5281/zenodo.13988149)

Human-in-the-loop Active Learning for Goal-Oriented Molecule Generation
====================================================================================================

We present an interactive workflow to fine-tune predictive machine learning models of target molecular properties based on expert feedback, and foster human-machine collaboration for goal-oriented molecular design and optimization.

![Overview of the human-in-the-loop active learning workflow to fine-tune molecular property predictors for goal-oriented molecule generation.](figures/graphical-abstract.png)

In this study, we simulate the process of producing novel drug candidates through machine learning (REINVENT) then validating them in the lab.
This workflow is based [REINVENT 3.2](https://github.com/yasminenahal/Reinvent) for molecule generation. In the meantime, [REINVENT 4](https://github.com/MolecularAI/REINVENT4) was released so the plan is to move to REINVENT 4 soon!

The goal of this study is to generate successful top-scoring molecules (i.e., promising with respect to a target molecular property) according to both the machine learning predictive model used in the scoring function, and a lab simulator that validates the promise of the produced molecules at the end of the REINVENT process. Both should be well aligned to avoid relying on suboptimal molecules during assay trials and increasing their success rate.

Since simulators are expensive to query at each iteration of fine-tuning the predictive model (i.e., active learning), we mitigate this by allowing "weaker" yet more accessible oracles (i.e., human experts) to be queried for iterative fine-tuning of the predictive model (i.e., human-in-the-loop active learning). The lab simulator is then only used at the end of the REINVENT process for final validation.

Human experts evaluate the relevance of top-scoring molecules identified by the predictive machine learning model by accepting or refuting some of them. Our results demonstrated significant improvements in the REINVENT process' outcome where the predicted success scores of the final generated molecules are better aligned with those of the lab simulator, while enhancing other metrics such as drug-likeness and synthetic accessibility.



# System Requirements

- `python>=3.9,<3.11`
- This code was tested on Linux `x86_64`

# Installation

1. Since this workflow is based on REINVENT 3.2, you need a working installation of REINVENT 3.2. Follow install instructions [here](https://github.com/yasminenahal/Reinvent).
2. Create a virtual environment with `python>=3.9,<3.11` and activate it, then install the package with

        pip install hitl-al-gomg

# Usage

Below are command examples to train a target property predictor then running the active learning workflow using a simulated expert to fine-tune it. Make sure to replace the provided paths with yours before running the command lines.
In this example, the target property is DRD2 bioactivity.

**For training a predictor of DRD2 bioactivity:**

        python -m hitl_al_gomg.models.train --path_to_train_data data/train/drd2_train --path_to_test_data data/test/drd2_test --path_to_predictor data/predictors/drd2 --path_to_simulator data/simulators/drd2 --train True --demo True

- To use the same DRD2 bioactivity simulator than that of the paper, you can download `drd2.pkl` from this [URL](https://huggingface.co/yasminenahal/hitl-al-gomg-simulators/tree/main).
- The directory `example_files/` contains examples of hyperparameter grids to run cross-validation for `scikit-learn` Random Forest models. If you wish to enable hyparameter search, you can add the argument ``--path_to_param_grid example_files/rfc_param_grid.json``.

**For running the HITL-AL workflow using a simulated expert:**

Once you have a pre-trained predictor for your target property, you can use it to run REINVENT to produce novel molecules that satisfy this property.

- First, you need to run the workflow without active learning so that you can generate the set of generated molecules based on your initial target property predictor.

        python -m hitl_al_gomg.run --seed 3 --rounds 4 --num_opt_steps 100 --path_to_output_dir results --path_to_reinvent_env /home/miniconda3/envs/reinvent-hitl --path_to_reinvent_repo /home/Test_my_code/Reinvent --task drd2 --path_to_scoring_model data/predictors/drd2 --path_to_simulator data/simulators/drd2 --model_type classification --scoring_component_name bioactivity --dirname demo_drd2 --path_to_train_data data/train/drd2_train --acquisition None

- Then, you can run the workflow using active learning. Below is an example where we use entropy-based sampling to select `10` query molecules to be evaluated by the simulated expert model.

        python -m hitl_al_gomg.run --seed 3 --rounds 4 --num_opt_steps 100 --path_to_output_dir results --path_to_reinvent_env /home/miniconda3/envs/reinvent-hitl --path_to_reinvent_repo /home/Test_my_code/Reinvent --task drd2 --path_to_scoring_model data/predictors/drd2 --path_to_simulator data/simulators/drd2 --model_type classification --scoring_component_name bioactivity --dirname demo_drd2 --path_to_train_data data/train/drd2_train --acquisition entropy --al_iterations 5 --n_queries 10 --noise 0.1


**For calculating simulator scores and metrics from [MOSES](https://github.com/molecularsets/moses):**

Once you the HITL-AL run is completed, you can generate a pickled dictionary that contains simulator/oracle scores and metrics to evaluate your generated molecules at the end of each round and track the progress of your predictor fine-tuning.

- To evaluate the output of your baseline run (without active learning feedback):

         python -m hitl_al_gomg.evaluate_results --job_name demo_drd2 --seed 3 --rounds 4 --n_opt_steps 100 --task drd2 --model_type classification --score_component_name bioactivity --path_to_predictor data/predictors/drd2 --path_to_simulator data/simulators/drd2 --path_to_train_data data/train/drd2_train --path_to_test_data data/test/drd2_test --path_to_reinvent_env /home/miniconda3/envs/reinvent-hitl --path_to_reinvent_repo /home/Test_my_code/Reinvent --path_to_output_dir results --acquisition None

- To evaluate the output of your active learning-enabled run:
  
         python -m hitl_al_gomg.evaluate_results --job_name demo_drd2 --seed 3 --rounds 4 --n_opt_steps 100 --task drd2 --model_type classification --score_component_name bioactivity --path_to_predictor data/predictors/drd2 --path_to_simulator data/simulators/drd2 --path_to_train_data data/train/drd2_train --path_to_test_data data/test/drd2_test --path_to_reinvent_env /home/miniconda3/envs/reinvent-hitl --path_to_reinvent_repo /home/Test_my_code/Reinvent --path_to_output_dir results --acquisition entropy --al_iterations 5 --n_queries 10 --sigma_noise 0.1

**For running the HITL-AL workflow using the Metis graphical interface:**

To run the workflow with real expert feedback through a graphical interface, you first need to install  [Metis](https://github.com/JanoschMenke/metis) in two quick steps:

1. Clone the Metis repository using ``git clone --branch nahal_experiment https://github.com/JanoschMenke/metis.git`` then navigate to its location.
2. On a remote machine accessible through SSH and that has SLURM, install [REINVENT V3.2](https://github.com/yasminenahal/Reinvent) as mentioned previously.

To run the HITL-AL workflow described in our paper, you can download the following [zipped folder](https://drive.google.com/file/d/1xWCoHodZTy9VwIm-CAicT1MFbzHg2bPn/view?usp=sharing) and upload it to your remote machine. This folder contains the models used for Reinvent (the prior Reinvent agent `random.prior.new`, the Reinvent agent `Agent_Initial.ckpt` after being optimized for 1200 epochs using the initial target property predictor `Model_Initial.pkl` as well as the hERG bioactivity oracle that we use in the multi-objective use case experiments `herg.pkl`).

You should change the following file contents according to your remote SSH login details and your paths to predictive models and data sets.

- `metis/setting.yml` should contain your Metis configuration, such as which features you wish to display on the interface, as well as your paths to your initial predictive model, initial training set and initial set of generated molecules before observing any human feedback.
- `metis/reinvent_connect/input_files/ssh_settings.yml` should contain your SSH login information, path to a folder where REINVENT is installed and where you wish to store your REINVENT outputs on your remote machine.
- `metis/reinvent_connect/input_files/test_qsar.json` should contain your initial REINVENT run configuration (after the first iteration, it will be updated automatically).
- `metis/reinvent_connect/input_files/reinvent_slurm.slurm` should contain your SLURM job specifications and REINVENT run commands.

To start the interface and the human workflow, run

        cd metis && python metis.py

Your evaluations through Metis will be stored in the `results` folder.

# Data

- We provide data sets for training the penalized LogP and DRD2 bioactivity predictors, as well as a sample from ChEMBL on which `REINVENT` prior agent was pre-trained.
- We also provide a copy of the pre-trained `REINVENT` prior agent in `data/priors/random.prior.new`.
- The experimental simulators or oracles for DRD2 bioactivity and the hERG model described in the multi-objective generation use case can both be downloaded from [https://huggingface.co/yasminenahal/hitl-al-gomg-simulators/tree/main].
  
# Notebooks

In `notebooks/`, we provide Jupyter notebooks with code to reproduce the paper's result figures for both simulation and real human experiments.

# Acknowledgements

- We acknowledge the following works which were extremely helpful to develop this workflow:
  * Sundin, I., Voronov, A., Xiao, H. et al. Human-in-the-loop assisted de novo molecular design. J Cheminform 14, 86 (2022). [https://doi.org/10.1186/s13321-022-00667-8](https://doi.org/10.1186/s13321-022-00667-8)
  * Bickford Smith, F., Kirsch, A., Farquhar, S., Gal, Y., Foster, A., Rainforth, T. Prediction-oriented Bayesian active learning. International Conference on Artificial Intelligence and Statistics (2023). [https://arxiv.org/abs/2304.08151](https://arxiv.org/abs/2304.08151)

- We thank [Vincenzo Palmacci](https://github.com/vincenzo-palmacci) for his contribution in refactoring parts of this code.

**For any inquiries, please contact yasmine.nahal@aalto.fi**
