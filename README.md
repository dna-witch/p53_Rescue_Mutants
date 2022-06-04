Predicting Cancerous p53 Mutants
==============================


## Purpose

This project aims to identify mutants that could lead to functional rescue and regions of the p53 core domain that could be altered to rescue function by modeling mutant p53 transcriptional activity.

## Data
The raw datasets for this project are from the . . . Group and are hosted online at the UC Irvine Machine Learning Repository.

## Methods
### Data Cleaning


### Exploratory Data Analysis


### Preprocessing Steps
    + Feature Selection
    + Scaling the Data
    + Resampling with SMOTE-Tomek

### Modeling 
#### Random Forest

#### Logistic Regression

#### Gaussian Naive Bayes

#### Nearest Centroid

## Conclusions
### Metrics for Success


## Future Improvements
+ Filtering Features Using Pairwise Mutual Information
+ Play with the number of features/attributes used to train the models to find optimum number of features
+ Instead of just a train-test-split, make a train-test-validation split of the data
+ Visualize the data with UMAP and Compare UMAP with t-SNE
+ Hyperparameter Tuning with Bayesian Optimization
+ Resampling Data with SMOTE-ENN and comparing noise with SMOTE-Tomek
+ Calibrating the Classifier Models
+ Neural Networks
+ Calculate the MCC scores
+ This was built using the 2010 dataset, but can combine with 2012 dataset
+ Investigate specific clusters of “active” p53 proteins, such as those seen in the t-SNE plots, more closely
+ Combine with protein visualization software for easier interpretation of the results
+ Maybe cross-check/sanity-check with information of which domains of p53 are the most important to preserving wild-type function
+ Use Cloud Computing services, containerize and deploy model


## Acknowledgments
Huge thank you to my mentor, Ricardo, for all of his encouragement and guidance throughout this capstone project! 

==============================
Project Organization

    ├── LICENSE
    ├── Makefile           <- Makefile with commands like `make data` or `make train`
    ├── README.md          <- The top-level README for developers using this project.
    ├── data
    │   ├── external       <- Data from third party sources.
    │   ├── interim        <- Intermediate data that has been transformed.
    │   ├── processed      <- The final, canonical data sets for modeling.
    │   └── raw            <- The original, immutable data dump.
    │
    ├── docs               <- A default Sphinx project; see sphinx-doc.org for details
    │
    ├── models             <- Trained and serialized models, model predictions, or model summaries
    │
    ├── notebooks          <- Jupyter notebooks. Naming convention is a number (for ordering),
    │                         the creator's initials, and a short `-` delimited description, e.g.
    │                         `1.0-jqp-initial-data-exploration`.
    │
    ├── references         <- Data dictionaries, manuals, and all other explanatory materials.
    │
    ├── reports            <- Generated analysis as HTML, PDF, LaTeX, etc.
    │   └── figures        <- Generated graphics and figures to be used in reporting
    │
    ├── requirements.txt   <- The requirements file for reproducing the analysis environment, e.g.
    │                         generated with `pip freeze > requirements.txt`
    │
    ├── setup.py           <- makes project pip installable (pip install -e .) so src can be imported
    ├── src                <- Source code for use in this project.
    │   ├── __init__.py    <- Makes src a Python module
    │   │
    │   ├── data           <- Scripts to download or generate data
    │   │   └── make_dataset.py
    │   │
    │   ├── features       <- Scripts to turn raw data into features for modeling
    │   │   └── build_features.py
    │   │
    │   ├── models         <- Scripts to train models and then use trained models to make
    │   │   │                 predictions
    │   │   ├── predict_model.py
    │   │   └── train_model.py
    │   │
    │   └── visualization  <- Scripts to create exploratory and results oriented visualizations
    │       └── visualize.py
    │
    └── tox.ini            <- tox file with settings for running tox; see tox.readthedocs.io


--------

<p><small>Project based on the <a target="_blank" href="https://drivendata.github.io/cookiecutter-data-science/">cookiecutter data science project template</a>. #cookiecutterdatascience</small></p>
"# p53_Rescue_Mutants" 
