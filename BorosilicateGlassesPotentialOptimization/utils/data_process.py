import torch
import random
import numpy as np
import pandas as pd
from skopt.space import Real

def set_seed(seed):
    """Ensure reproducibility by setting seeds."""
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def load_data(data_path):
    """Load data from the Excel file."""
    return pd.read_excel(data_path, header=0, sheet_name='Sheet1')

def normalize_tensor(tensor, min_val, max_val):
    """Apply Min-Max normalization."""
    return (tensor - min_val) / (max_val - min_val)

def denormalize_tensor(tensor, min_val, max_val):
    """Reverse Min-Max normalization."""
    return tensor * (max_val - min_val) + min_val

def data_process_NNW(raw_data_df,remove_nan_row = True):
    """
    Process input data to prepare features (x) and targets (y) for modeling.

    Parameters:
        raw_data_df (DataFrame): Raw data in DataFrame format.
        remove_nan_row (bool): If True, removes rows with NaN values.
        feature_type (str): Determines which feature set to use ('num' or 'pct').

    Returns:
        x (ndarray): BO-A', 'BO-R','BO-C','BB-A','BB-R',boron_number
        y (ndarray): for the boron_number: density and b4 value
    """
    if remove_nan_row == True:
        raw_data_df.dropna(inplace = True)

    para_num_np = raw_data_df[['BO-A', 'BO-R','BO-C','BB-A','BB-R']].to_numpy() # select numercal potential parameter
    density_np = raw_data_df[['6B_dens','12B_dens','24B_dens','37B_dens','50B_dens','62B_dens','75B_dens']].to_numpy()      # select densities
    b4_np = raw_data_df[['6B_B4','12B_B4','24B_B4','37B_B4','50B_B4','62B_B4','75B_B4']].to_numpy() 

    if not (para_num_np.shape[0] == density_np.shape[0] == b4_np.shape[0]):
      raise ValueError("para_num_np, density_np, b4_np raw number different。")

    boron_number = np.array([6,12,24,37,50,62,75])


    # make x for density model
    repeated_para_num = np.repeat(para_num_np, len(boron_number), axis=0)
    repeated_boron_number = np.tile(boron_number, para_num_np.shape[0])
    repeated_boron_number = repeated_boron_number.reshape(-1, 1)
    x = np.hstack((repeated_para_num, repeated_boron_number))

    # make y for density model
    y_density = density_np.reshape(-1,1)
    y_b4= b4_np.reshape(-1,1)
    y = np.hstack((y_density, y_b4))

    assert x.shape[0] == y.shape[0]
    return x, y

def normalize_NNW_data(x_train, x_test, y_train, y_test):
    """
    Normalize the input and output data.
    """
    # Convert numpy arrays to PyTorch tensors
    x_train_tensor = torch.tensor(x_train, dtype=torch.float32)
    x_test_tensor = torch.tensor(x_test, dtype=torch.float32)
    y_train_tensor = torch.tensor(y_train, dtype=torch.float32)
    y_test_tensor = torch.tensor(y_test, dtype=torch.float32)

    # Min-Max Normalize the first column of y using training data
    # normalize density: [0,1]
    min_y = y_train_tensor[:, 0].min()
    max_y = y_train_tensor[:, 0].max()
    y_train_tensor[:, 0] = (y_train_tensor[:, 0] - min_y) / (max_y - min_y)
    y_test_tensor[:, 0] = (y_test_tensor[:, 0] - min_y) / (max_y - min_y)  # Apply the same scale to the test data

    #Normalize X: the first five parameter is by min-max normalization, based on each parameters range in training data
    min_x_train = x_train_tensor[:, :-1].min(dim=0, keepdim=True)[0]
    max_x_train = x_train_tensor[:, :-1].max(dim=0, keepdim=True)[0]

    x_train_norm = x_train_tensor.clone()
    x_test_norm = x_test_tensor.clone()

    x_train_norm[:, :-1] = (x_train_tensor[:, :-1] - min_x_train) / (max_x_train - min_x_train)
    x_test_norm[:, :-1] = (x_test_tensor[:, :-1] - min_x_train) / (max_x_train - min_x_train)

    #the last parameter: Boron number has range [0,100], so normalize based on this
    x_train_norm[:, -1] = x_train_tensor[:, -1] / 100
    x_test_norm[:, -1] = x_test_tensor[:, -1] / 100

    return x_train_norm, x_test_norm, y_train_tensor, y_test_tensor, {'x_min': min_x_train, 'x_max': max_x_train, 'y_min': min_y, 'y_max': max_y}
  

def data_process_bayes(raw_data_df):
  '''
  original parameter: the parameter value from original paper (for wrong mass)
  y_target_tensor: the target experiment value for density and B4 under different fraction
  '''
  original_parameter = raw_data_df.loc[1, ['BO-A', 'BO-R', 'BO-C', 'BB-A', 'BB-R']].to_numpy()
  density_target = raw_data_df.loc[0, ['6B_dens', '12B_dens', '24B_dens', '37B_dens', '50B_dens', '62B_dens', '75B_dens']].to_numpy()
  b4_target = raw_data_df.loc[0, ['6B_B4', '12B_B4', '24B_B4', '37B_B4', '50B_B4', '62B_B4', '75B_B4']].to_numpy()
    
  y_density = density_target.reshape(-1,1)
  y_b4= b4_target.reshape(-1,1)
  y_target = np.hstack((y_density, y_b4))
  y_target = y_target.astype(np.float32)
  y_target_tensor = torch.tensor(y_target, dtype=torch.float32)
  return original_parameter, y_target_tensor

def bayes_search_space(original_parameter, percentage_ranges):
  # Define the space of parameters over which to optimize
  actual_ranges = [(original * (1 + low/100), original * (1 + high/100)) for original, (low, high) in zip(original_parameter, percentage_ranges)]
  space = [Real(low=low, high=high, name=f'param_{i}') for i, (low, high) in enumerate(actual_ranges)]
  return actual_ranges, space

def store_optimization_results(result, output_parameter_path, top_n=5):
    """
    Process optimization results, display top N results, and save to Excel file.
    Preserves additional columns from existing Excel file that are not in the new results.
    
    Args:
        result: The result object from the optimization process.
        output_parameter_path: Path to save the Excel file with results.
        top_n: Number of top results to display and save (default is 5).
    """
    import pandas as pd
    # Sort all function evaluations and parameters
    all_results = sorted(zip(result.func_vals, result.x_iters), key=lambda x: x[0])
    
    # Display the best MSE and parameters
    print(f"Best MSE: {result.fun}")
    print("Best parameters:")
    best_params = {f'param_{i}': result.x[i] for i in range(len(result.x))}
    print(best_params)
    
    # Display the top N results
    print(f"\nTop {top_n} combinations and their MSEs:")
    top_results = []
    for i, (mse, params) in enumerate(all_results[:top_n]):
        param_dict = {f'param_{j}': params[j] for j in range(len(params))}
        print(f"Rank {i+1}: MSE = {mse}, Parameters = {param_dict}")
        # Prepare results for DataFrame
        result_dict = {'predicted MSE': mse, 'Rank': i+1}
        result_dict.update(param_dict)
        top_results.append(result_dict)
    
    # Define key mapping
    key_mapping = {
        'predicted MSE': 'predicted MSE',
        'param_0': 'BO-A',
        'param_1': 'BO-R',
        'param_2': 'BO-C',
        'param_3': 'BB-A',
        'param_4': 'BB-R'
    }
    
    # Convert results to a DataFrame and format keys
    results_df = pd.DataFrame(top_results)
    results_df = results_df.rename(columns=key_mapping)
    results_df = results_df[['predicted MSE', 'BO-A', 'BO-R', 'BO-C', 'BB-A', 'BB-R']]  # Reorder columns
    print("\nResults DataFrame:")
    print(results_df)
    
    # Read existing Excel file and preserve additional columns
    prev = pd.read_excel(output_parameter_path, index_col=0)
    
    # Get columns from prev that are not in results_df
    additional_columns = [col for col in prev.columns if col not in results_df.columns]
    
    # For new rows in results_df, fill additional columns with NaN
    for col in additional_columns:
        results_df[col] = pd.NA
        
    # Ensure all columns from prev are present in results_df
    results_df = results_df.reindex(columns=list(prev.columns))
    
    # Concatenate while preserving all columns
    df_final = pd.concat([prev, results_df], ignore_index=True)
    df_final.index = df_final.index + 1

    
    # Save to Excel
    df_final.to_excel(output_parameter_path)
    print(f"\nResults saved to {output_parameter_path}")
  
    return df_final

# Usage example:
# process_optimization_results(result, '/path/to/output.xlsx', top_n=5)

#To run seperate NNW for density and B4
def split_y_data(y_train, y_test):
    y_train_1 = y_train[:, 0:1]
    y_train_2 = y_train[:, 1:2]
    y_test_1 = y_test[:, 0:1]
    y_test_2 = y_test[:, 1:2]
    return y_train_1, y_train_2, y_test_1, y_test_2
