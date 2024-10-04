# Recommender System with Explainable AI (XAI) Methods

This project implements a recommender system using three different algorithms: **Item-KNN**, **ALS**, and **Content-Based Filtering**. Additionally, four Explainable AI (XAI) methods are applied to each algorithm: **SHAP**, **LIME**, **Counterfactual Explanations**, and **Surrogate Models**. Each XAI method is evaluated using six key metrics: **Interpretability**, **Fidelity**, **Computational Efficiency**, **Accuracy Impact**, **Scope**, and **Model Complexity**.

## Project Overview

### Dataset
The [MovieLens 1M Dataset](https://grouplens.org/datasets/movielens/1m/) is used to build and evaluate recommender systems. The dataset includes 1 million ratings for 3,883 movies rated by 6,040 users. The following files are used:

- **ratings.dat**: User-movie ratings with timestamps
- **movies.dat**: Movie titles and genres
- **users.dat**: User demographic information

### Recommender Algorithms Implemented
1. **Item-KNN**: A neighbourhood-based collaborative filtering algorithm that recommends items based on the similarities between items rated by the same user.
2. **ALS (Alternating Least Squares)**: A matrix factorization method that decomposes the user-item matrix to provide recommendations.
3. **Content-Based Filtering**: Recommends items based on the similarity of their features (e.g., genres) to items the user has liked previously.

### XAI Methods Implemented
1. **SHAP (SHapley Additive exPlanations)**: Explains individual predictions by attributing each feature's contribution.
2. **LIME (Local Interpretable Model-Agnostic Explanations)**: Provides local explanations by approximating the model with an interpretable surrogate model for individual predictions.
3. **Counterfactual Explanations**: Explores minimal changes to features that would result in a different prediction.
4. **Surrogate Models**: Trains interpretable models (like decision trees) to mimic the behaviour of black-box models, providing global explanations.

## Structure of the Repository

- `notebook.ipynb`: Contains the Jupyter notebook with the complete implementation and analysis of the recommender system and XAI methods.
- `ml-1m/`: Directory containing the MovieLens dataset files (`ratings.dat`, `movies.dat`, `users.dat`).
- `images/`: Directory containing tree plots and other visualizations generated during the analysis.


## How to Run the Code

1. **Clone the repository**:

2. **Install the required libraries**:

3. **Run the Jupyter notebook**:
