import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import KFold, LeaveOneOut,GridSearchCV
from sklearn.linear_model import RidgeCV
from sklearn.model_selection import cross_val_score, KFold
from sklearn.metrics import mean_squared_error
from joblib import dump
import matplotlib.pyplot as plt
import seaborn as sns
import xgboost as xgb
import os
from statistics import mean, stdev
import re
from sklearn.pipeline import make_pipeline
import warnings
warnings.filterwarnings('ignore')
from sklearn.preprocessing import LabelEncoder


# Create directories for saving results
os.makedirs('results/drug_response_pred_models', exist_ok=True)
os.makedirs('results/plots', exist_ok=True)
os.makedirs('results/logs', exist_ok=True)

def plot_cor(mean_data):
    # Ensure 'Drug' and 'Cancer' columns are available

    label_encoder = LabelEncoder()
    mean_data['Drug'] = mean_data['Drug'].astype(str)

    # Fit and transform the data
    mean_data['Drug'] = label_encoder.fit_transform(mean_data['Drug'])
    
    # Calculate the correlation between 'Actual' and 'Predicted'
    correlation = mean_data[['Actual', 'Predicted']].corr().iloc[0, 1]

    # Create the scatter plot
    plt.figure(figsize=(14, 8))
    
    # Get unique cancer types and assign colors
    cancer_types = mean_data['Cancer'].unique()
    palette = sns.color_palette('Set1', n_colors=len(cancer_types))
    color_mapping = {cancer: palette[i] for i, cancer in enumerate(cancer_types)}
    
    # Create a scatter plot with different colors for cancer types
    scatter = sns.scatterplot(
        x='Actual', 
        y='Predicted', 
        data=mean_data, 
        hue='Cancer',        # Color by cancer type
        palette=color_mapping, # Use custom color mapping
        s=100, 
        edgecolor='w', 
        alpha=0.7
    )

    # Add text annotations for drug numbers
    for i in range(mean_data.shape[0]):
        plt.text(
            mean_data['Actual'].iloc[i], 
            mean_data['Predicted'].iloc[i], 
            str(mean_data['Drug'].iloc[i]), 
            color=color_mapping[mean_data['Cancer'].iloc[i]],  # Use custom color mapping
            fontsize=12, 
            ha='center', 
            va='center'
        )

    # Add a line of equality
    max_val = max(mean_data['Actual'].max(), mean_data['Predicted'].max())
    min_val = min(mean_data['Actual'].min(), mean_data['Predicted'].min())
    plt.plot([min_val, max_val], [min_val, max_val], color='grey', linestyle='--', label='Perfect Prediction')

    # Customize plot
    plt.title(f'Drug Sensitivity by Cancer Type (Correlation: {correlation:.4f})')
    plt.xlabel('Actual Drug Sensitivity')
    plt.ylabel('Predicted Drug Sensitivity')

    # Adjust legend to show only cancer types
    plt.legend(title='Cancer Type', loc='center left', bbox_to_anchor=(1, 0.5), ncol=1, frameon=False)

    plt.grid(True)

    # Save plot
    plt_path = 'results/plots/resdat_plot.png'
    plt.savefig(plt_path, bbox_inches='tight')
    plt.close()

# def plot_cor(mean_data):
#     # Create the scatter plot
#     plt.figure(figsize=(12, 8))
#     scatter = sns.scatterplot(x='Actual', y='Predicted', data=mean_data, hue='Cancer', palette='Set1', s=100, edgecolor='w', alpha=0.7)

#     # Add a line of equality
#     max_val = max(mean_data['Actual'].max(), mean_data['Predicted'].max())
#     min_val = min(mean_data['Actual'].min(), mean_data['Predicted'].min())
#     plt.plot([min_val, max_val], [min_val, max_val], color='grey', linestyle='--', label='Perfect Prediction')

#     # Customize plot
#     plt.title('Drug Sensitivity by Cancer Type')
#     plt.xlabel('Actual Drug Sensitivity')
#     plt.ylabel('Predicted Drug Sensitivity')

#     # Adjust legend
#     plt.legend(title='Cancer Type', loc='center left', bbox_to_anchor=(1, 0.5), ncol=1, frameon=False)

#     plt.grid(True)

#     # Save plot
#     plt_path = 'results/plots/resdat_plot.png'
#     plt.savefig(plt_path, bbox_inches='tight')
#     plt.close()

def train_and_evaluate_model_kfold(X, y, drug,cancer):
    # Initialize RidgeCV with provided alphas

    alphas = [0.1, 1.0, 10.0]
    model_name = 'RidgeCV'

    model = RidgeCV(alphas=alphas, store_cv_values=True)
    
    # Initialize 5-Fold Cross-Validation
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    mse_list = []
    y_true_list = []
    y_pred_list = []

    # Perform 5-Fold Cross-Validation
    for train_index, test_index in kf.split(X):
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]

        # Fit RidgeCV model
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        
        # Calculate mean squared error
        mse = mean_squared_error(y_test, y_pred)
        mse_list.append(mse)
        y_true_list.extend(y_test)
        y_pred_list.extend(y_pred)

    # Output the results
    # Calculate mean and standard deviation of MSE
    mean_mse = np.mean(mse_list)
    mse_std = np.std(mse_list)

    # Save MSE and correlation
    log_path = f'results/logs/{model_name}_{drug}_log.txt'
    with open(log_path, 'w') as log_file:
        log_file.write(f"{model_name} Mean Squared Error: {mean_mse:.4f}\n")
        log_file.write(f"{model_name} std MSE: {mse_std:.4f}\n")
        correlation = np.corrcoef(y_true_list, y_pred_list)[0, 1]
        log_file.write(f'Correlation: {correlation:.4f}\n')

    df = pd.DataFrame({'Actual': y_test,'Predicted': y_pred,'Cancer': cancerlist})
    mean_data = df.groupby('Cancer').agg({'Actual': 'mean', 'Predicted': 'mean'}).reset_index()

    correlation = mean_data['Actual'].corr(mean_data['Predicted'])
    print(f'Model: {model_name}; Correlation: {correlation}')
    model_path = f'results/drug_response_pred_models/{model_name}_{drug}_model.pth'
    dump(best_model, model_path)
    return df

def train_and_evaluate_model_LOOCV(X, y, drug, cancer):
    alphas = [0.1, 1.0, 10.0]
    model_name = 'RidgeCV'
    
    # Initialize Leave-One-Out cross-validation
    loo = LeaveOneOut()
    mse_list = []
    y_true_list = []
    y_pred_list = []
    cancerlist = []

    # RidgeCV requires an array of alphas
    model = RidgeCV(alphas=alphas, store_cv_values=True)
    # Leave-One-Out cross-validation loop
    for train_index, test_index in loo.split(X):
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]
        c = cancer[test_index].values  # Assuming cancer is a pandas Series

        # Fit RidgeCV model
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        
        # Calculate mean squared error
        mse = mean_squared_error(y_test, y_pred)
        mse_list.append(mse)
        y_true_list.append(y_test[0])  # Append single value
        y_pred_list.append(y_pred[0])  # Append single value
        cancerlist.append(c[0])        # Append single value

    # Output the results
    # Calculate mean and standard deviation of MSE
    mean_mse = np.mean(mse_list)
    mse_std = np.std(mse_list)

    # Save MSE and correlation
    log_path = f'results/logs/{model_name}_{drug}_log.txt'
    os.makedirs(os.path.dirname(log_path), exist_ok=True)
    with open(log_path, 'w') as log_file:
        log_file.write(f"{model_name} Mean Squared Error: {mean_mse:.4f}\n")
        log_file.write(f"{model_name} std MSE: {mse_std:.4f}\n")
        correlation = np.corrcoef(y_true_list, y_pred_list)[0, 1]
        log_file.write(f'Correlation: {correlation:.4f}\n')

    df = pd.DataFrame({'Actual': y_true_list, 'Predicted': y_pred_list, 'Cancer': cancerlist})
    df['Drug'] = drug
    mean_data = df.groupby('Cancer').agg({'Actual': 'mean', 'Predicted': 'mean'}).reset_index()
    

    correlation = mean_data['Actual'].corr(mean_data['Predicted'])
    print(f'Model: {model_name}; Correlation: {correlation}')
    
    model_path = f'results/drug_response_pred_models/{model_name}_{drug}_model.pth'
    dump(model, model_path)  # Save the model instead of `best_model`, which was not defined

    return df

def PREDICT_RESPONSE():
    # Load embeddings
    embeddings_path = 'results/PatStrat_emb.csv'
    embeddings = pd.read_csv(embeddings_path, index_col=0).sort_index()

    # Load drug response data
    drug_response = pd.read_csv("data/drug_response.csv", index_col=0)
    drug_response = drug_response[drug_response['TCGA_DESC'] != 'UNCLASSIFIED']
    value_counts = drug_response['TCGA_DESC'].value_counts()
    values_to_keep = value_counts[value_counts > 10000].index
    drug_response = drug_response[drug_response['TCGA_DESC'].isin(values_to_keep)]

    resdat = pd.DataFrame(columns=['Actual','Predicted','Cancer'])

    for drug in sorted(list(set(drug_response['DRUG_NAME']))):
        #for cancer in set(drug_response['TCGA_DESC']):
        print(f"Processing drug: {drug}")
        subdf = drug_response[drug_response.index.isin(embeddings.index)]
        #subdf = subdf[subdf['TCGA_DESC'] == cancer]
        subdf = subdf[subdf['DRUG_NAME'] == drug][['LN_IC50','TCGA_DESC']]
        subdf = subdf[~subdf.index.duplicated(keep='first')]
        emb = embeddings[embeddings.index.isin(subdf.index)].sort_index()
        subdf = subdf[subdf.index.isin(emb.index)].sort_index()
        responses = subdf['LN_IC50'].values
        cancer = subdf['TCGA_DESC']
        #print(cancer)

        # Convert embeddings to a matrix format and match with responses
        X = emb.values
        y = responses
        # Train and evaluate each model
        drug = re.sub(r'[^a-zA-Z0-9]', '_', drug)
        df = train_and_evaluate_model_LOOCV(X, y, drug, cancer)
        resdat = pd.concat([resdat,df])
        plot_cor(df)
        print(resdat)
    resdat.to_csv("results/logs/all_results.csv")
    #plot_cor(resdat)

if __name__ == "__main__":
    main()