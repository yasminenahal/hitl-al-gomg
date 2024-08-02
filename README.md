# Human-in-the-loop Active Learning for Goal-Oriented Molecule Generation

We present an interactive workflow to fine-tune predictive machine learning models for target molecular properties based on expert feedback and foster human-machine collaboration for goal-oriented molecular design and optimization.

![Overview of the human-in-the-loop active learning workflow to fine-tune molecular property predictors for goal-oriented molecule generation.](figures/graphical-abstract.png)

In this study, we simulated the process of producing novel drug candidates using machine learning then validating them in the lab.

The goal is to generate a high number of successful top-scoring molecules (i.e., promising with respect to a target molecular property) according to both the machine learning predictive model, which scores the molecules at each iteration of the drug design process, and the lab simulator, which evaluates the best molecules at the end of the process.

Since simulators are expensive to query at each iteration of the drug design process, as well as for fine-tuning the predictive model iteratively (i.e., active learning), we mitigate this by allowing "weaker" oracles (i.e., human experts) to be queried for fine-tuning the predictive model (i.e., human-in-the-loop active learning).

Human experts supervised the predictive machine learning model that scores the molecules by accepting or refuting some of its predictions of top-scoring molecules. This improved the process' outcome by progressively aligning the machine-predicted probabilities of success with those of an experimental simulator and enhancing other metrics such as drug-likeness and synthetic accessibility.
