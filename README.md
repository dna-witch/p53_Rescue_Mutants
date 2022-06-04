Predicting Cancerous p53 Mutants
==============================


## Purpose

This project aims to identify mutants that could lead to functional rescue and regions of the p53 core domain that could be altered to rescue function by modeling mutant p53 transcriptional activity.

![p53_structure](img/p53_structure.png)

![hallmarks of cancer hanahan 2011](img/cancer_hallmarks.png)

## Data
The data represents the biophysical attributes of mutant p53 proteins which are going to be used to predict the transcriptional activity of the mutant p53 proteins. The mutant proteins are all labeled “active” or “inactive”. Proteins labeled “inactive” are cancerous, while proteins labeled “active” have successfully had their normal transcriptional function “rescued”. There are 16772 instances, and each instance has 5409 attributes. Potential challenges for this dataset include known missing values, as well as the large number of attributes per instance.

The attributes are described as follows:
+ Attributes 1-4826 represent 2D electrostatic and surface based features. These are all represented by numerical values.
+ Attributes 4827-5408 represent 3D distance based features. These are all represented by numerical values.
+ Attribute 5409 is the class attribute, which is either active or inactive. This is a categorical variable.

## Methods
### Data Cleaning
![Dirty Data](img/dirty_data.png)
![Nametags](img/nametags.png)

Cleaning steps:
1) Concat numeric data with mutant labels
2) Now we have duplicate class attribute columns -> checked if they were equivalent -> dropped one col
3) Changed ?’s to np.NaN
4) Dropped rows that were entirely NaN (181 rows) -> no more NaN left
5) Check for duplicates (all values in the rows are equivalent) -> none
6) Check datatypes -> all “object”, which is good because less memory-intensive for computations than using floats + pandas stores strings as “object” datatype

![Cleaned Data](img/clean_data.png)

### Exploratory Data Analysis

![Number Mutations](img/num_mutations.png)

![PCA](img/pca.png)

![t-SNE](img/tsne.png)

### Preprocessing Steps
#### Scaling the Data
![not_scaled](img/min_max_stds.png)
![scaled](img/scaled_features.png)

#### Feature Selection
![after_drop](img/after_drop.png)

![upper 2d](img/upper_2d.png)

![final](img/final.png)

![resampling](img/resampling.png)

### Modeling 

![baseline](img/baseline%20models.png)

![balanced accuracy](img/balanced_accuracy.png)

#### Random Forest

![rf2](img/rf2.png)
![rf2_desc](img/rf2_desc.png)
![rf3](img/rf3.png)
![rf3 randomized search](img/rf3_random_search.png)

#### Logistic Regression

![logreg](img/logreg.png)

#### Gaussian Naive Bayes

![gnb](img/gnb.png)

#### Nearest Centroid

![nearest centroid](img/nearest_centroid.png)
![nc table](img/nearest_centroid_table.png)

## Conclusions



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
