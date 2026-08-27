- download dataset from: https://www.kaggle.com/datasets/netflix-inc/netflix-prize-data
- unzip it in folder data/
- run preprocessing.ipynb and the data will be exported to data/processed/

## ALS Model and Evaluation

I implemented the Alternating Least Squares (ALS) algorithm from scratch using NumPy and SciPy's sparse matrices. The script runs hyperparameter tuning across different latent features, iterations, and regularization terms, calculating MAE and RMSE for each combination. Finally, it uses Matplotlib to plot the error trends and help pick the best model.
