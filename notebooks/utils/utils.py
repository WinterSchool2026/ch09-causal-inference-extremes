import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import precision_recall_curve, average_precision_score, roc_auc_score
import math


def plot_class_distribution(train, val, test, target_col):
    """
    Plots the count and percentage of positive and negative classes 
    across Train, Validation, and Test sets.
    """
    # 1. Prepare data
    dfs = {'Train': train, 'Validation': val, 'Test': test}
    plot_data = []

    # Calculate totals for each dataset to compute percentages later
    dataset_totals = {}

    for name, df in dfs.items():
        # Get counts
        counts = df[target_col].value_counts().reset_index()
        counts.columns = ['Class', 'Count']
        counts['Dataset'] = name
        
        # Store total samples per dataset for % calculation
        total_samples = counts['Count'].sum()
        dataset_totals[name] = total_samples
        
        plot_data.append(counts)
        
    df_plot = pd.concat(plot_data)

    # 2. Plotting
    plt.figure(figsize=(12, 6))
    ax = sns.barplot(
        data=df_plot, 
        x='Dataset', 
        y='Count', 
        hue='Class', 
        palette=['#1f77b4', '#ff7f0e'] 
    )

    # 3. Add Annotations (Count + Percentage)
    # We iterate through the containers (groups of bars by Hue/Class)
    for container in ax.containers:
        # Each bar in the container corresponds to one X-axis category (Train, Val, Test)
        # We assume the order of bars matches the order of x-tick labels
        labels = []
        for i, bar in enumerate(container):
            height = bar.get_height()
            
            # Get the dataset name corresponding to this bar (e.g., "Train")
            current_dataset = ax.get_xticklabels()[i].get_text()
            
            # Get the total samples for this dataset
            total = dataset_totals[current_dataset]
            
            # Calculate percentage
            if total > 0:
                pct = (height / total) * 100
                label = f'{int(height)}\n({pct:.1f}%)'
            else:
                label = "0"
            
            # Add the text to the plot
            ax.text(
                bar.get_x() + bar.get_width() / 2,  # Center X
                height,                             # Y height
                label, 
                ha='center', 
                va='bottom', 
                fontsize=11, 
                fontweight='bold'
            )

    # 4. Styling
    plt.title(f'Class Distribution for {target_col[0]}', fontsize=16)
    plt.ylabel('Count', fontsize=12)
    plt.xlabel('Dataset Split', fontsize=12)
    plt.legend(title='Target Class')
    
    # Add a little extra space at the top so labels fit
    plt.ylim(0, df_plot['Count'].max() * 1.15)
    plt.grid(axis='y', linestyle='--', alpha=0.5)
    
    plt.show()
    
def plot_univariate_pr_curves(df, target_col, feature_cols=None, top_n=None):
    """
    Plots PR curves for individual features to act as baselines.
    Automatically handles negative correlations (flips feature if AUC < 0.5).
    """
    if feature_cols is None:
        feature_cols = [c for c in df.columns if c != target_col]
    
    # Create a temporary copy for plotting so you don't overwrite your original data
    df_fun = df.copy()
    df_fun[target_col] = (df_fun[target_col] > 0).astype(int)

    # Apply your logic: Anything > 0 becomes 1 (Positive Class), else 0

    y_true = df_fun[target_col].values
    
    # Setup plot
    plt.figure(figsize=(12, 7))
    
    # Calculate "No Skill" baseline (fraction of positives)
    baseline = np.sum(y_true) / len(y_true)
    plt.plot([0, 1], [baseline, baseline], linestyle='--', label=f'No Skill ({baseline:.2f})', color='gray')

    # Store results to sort legend by AUC later
    results = []

    for feat in feature_cols:
        # Drop NaNs for this specific pair
        mask = df_fun[[feat, target_col]].notna().all(axis=1)
        y_curr = y_true[mask]
        feat_curr = df_fun.loc[mask, feat].values

        if len(y_curr) == 0:
            print(f"Skipping {feat} (No valid data)")
            continue

        # 1. Determine Direction using ROC AUC
        # If ROC AUC < 0.5, the feature is negatively correlated with target.
        # We must flip it (-feature) so the PR curve works correctly.
        try:
            auc_check = roc_auc_score(y_curr, feat_curr)
            direction_str = "+"
            if auc_check < 0.5:
                feat_curr = -feat_curr
                direction_str = "-"
                
            # 2. Calculate Precision-Recall
            precision, recall, _ = precision_recall_curve(y_curr, feat_curr)
            pr_auc = average_precision_score(y_curr, feat_curr)
            
            results.append({
                'label': f"{feat} ({direction_str})",
                'auc': pr_auc,
                'precision': precision,
                'recall': recall
            })
        except ValueError:
            print(f"Skipping {feat} (Error in calculation, possibly constant value)")

    # 3. Sort by Score and Plot (Top N only if requested)
    results.sort(key=lambda x: x['auc'], reverse=True)
    if top_n:
        results = results[:top_n]

    for res in results:
        plt.plot(res['recall'], res['precision'], lw=2, 
                 label=f"{res['label']} (AUC={res['auc']:.2f})")

    plt.xlabel('Recall (Sensitivity)', fontsize=15)
    plt.ylabel('Precision (PPV)', fontsize=15)
    plt.title(f'Univariate Feature Baselines (PR Curves)\nTarget: {target_col}', fontsize=14)
    plt.legend(loc='center left', bbox_to_anchor=(1, 0.5), fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.xlim([0, 1])
    plt.ylim([0, 1.05])
    plt.tight_layout()
    plt.show()
    
def split_data_by_time(df, time_col, train_years, val_years, test_years):
    """
    Splits a dataframe into training and testing sets based on year ranges.
    
    Parameters:
    df (pd.DataFrame): The input dataframe.
    time_col (str): The name of the column containing date/time information.
    train_years (tuple): (start_year, end_year) inclusive.
    test_years (tuple): (start_year, end_year) inclusive.
    """
    # Ensure the time column is in datetime format
    df[time_col] = pd.to_datetime(df[time_col])
    
    # Extract the year to make filtering easier
    df_years = df[time_col].dt.year
    
    # Create masks for training and testing
    train_mask = (df_years >= train_years[0]) & (df_years <= train_years[1])
    val_mask = (df_years >= val_years[0]) & (df_years <= val_years[1])
    test_mask = (df_years >= test_years[0]) & (df_years <= test_years[1])
    
    train_data = df[train_mask].copy()
    val_data = df[val_mask].copy()
    test_data = df[test_mask].copy()
    
    print(f"Training set: {train_years[0]}-{train_years[1]}")
    print(f"Validation set: {val_years[0]}-{val_years[1]}")
    print(f"Testing set:  {test_years[0]}-{test_years[1]}")
    
    print(f"Lenght of training data: {train_data.shape}, Lenght of validation data: {val_data.shape}, lenght of test data: {test_data.shape}")
    
    return train_data, val_data, test_data

# =============================================
# 1. Helper Function: Flatten Xarray for Plotting
# =============================================
def xarray_to_long_df(ds, feature_name, target_name):
    """
    Takes an xarray dataset and two variable names.
    Returns a long-format pandas DataFrame suitable for seaborn,
    with NaNs dropped to align features and targets perfectly.
    """
    try:
        # Select only the two needed variables to save memory before converting
        df = ds[[feature_name, target_name]]
        
        # Convert to dataframe. This flattens lat/lon/time into a MultiIndex
        #df = subset.to_dataframe()
        
        # Drop rows where either the feature OR the target is NaN
        df_clean = df.dropna(subset=[feature_name, target_name])
        
        # Reset index so lat/lon/time become normal columns (optional, but good for debugging)
        df_final = df_clean.reset_index()
        
        # Ensure target is categorical for better plotting semantics
        df_final[target_name] = df_final[target_name].astype('category')
        
        return df_final
        
    except KeyError as e:
        print(f"Error: Variable not found in dataset: {e}")
        return pd.DataFrame() # Return empty DF on failure

# =============================================
# 2. Main Mosaic Plotting Function
# =============================================
def plot_split_violin_mosaic(ds, target_var, define_features_list=None, ncols=4):
    """
    Creates a mosaic of split violin plots for features vs a binary target.
    """
    # Determine features to plot
    if define_features_list is None:
        # Default: Plot everything except the target itself
        features_list = [v for v in ds.data_vars if v != target_var]
    else:
        features_list = define_features_list

    n_plots = len(features_list)
    nrows = math.ceil(n_plots / ncols)

    # Setup Figure using constrained_layout for automatic spacing
    fig, ax = plt.subplots(figsize=[ncols * 3, nrows * 2.5], 
                           ncols=ncols, nrows=nrows, 
                           constrained_layout=True)
    ax = ax.flatten()

    print(f"Generating {n_plots} violin plots against target: '{target_var}'...")

    # --- Main Loop ---
    for i, feature_name in enumerate(features_list):
        # 1. Prepare Data
        # We must flatten the specific feature + target combo into a 2-column DataFrame
        plot_df = xarray_to_long_df(ds, feature_name, target_var)

        if plot_df.empty:
            print(f"Skipping {feature_name} (No overlapping valid data with target)")
            ax[i].set_title(f"{feature_name} (No Data)", fontsize=9)
            ax[i].set_xticks([])
            ax[i].set_yticks([])
            continue

        # 2. Create Split Violin Plot
        # x = Binary Target (splits the plot left/right)
        # y = Continuous Feature (shows the distribution)
        # hue = Binary Target + split=True (creates the split violin effect)
        sns.violinplot(data=plot_df, x=target_var, y=feature_name,
                       hue=target_var, split=True,
                       ax=ax[i],
                       palette={0: "#809c73", 1: "#a05d5d"}, # Red for 0, Green for 1
                       inner="quart", # Adds quartile lines inside
                       linewidth=1,
                       legend=False) # Turn off individual legends to save space

        # 3. Formatting
        ax[i].set_title(f"{feature_name}", fontsize=11, fontweight='bold')
        ax[i].set_ylabel("") # Remove y-label to save space
        ax[i].set_xlabel("Target Class", fontsize=8)
        
        # Optional: nicer x-tick labels
        # ax[i].set_xticklabels(["Negative (0)", "Positive (1)"])
        
        # Optional: Grid for easier reading of quartiles
        ax[i].grid(True, axis='y', linestyle='--', alpha=0.5)


    # --- Cleanup ---
    # Hide empty axes if n_plots isn't a perfect multiple of ncols
    for j in range(i + 1, len(ax)):
        ax[j].axis('off')

    fig.suptitle(f"Feature Distributions Split by Target ({target_var})", 
                 fontsize=16, y=1.01, fontweight='bold')
    
    # Create a dummy legend for the whole figure
    from matplotlib.lines import Line2D
    custom_lines = [Line2D([0], [0], color="#809c73", lw=4),
                    Line2D([0], [0], color="#a05d5d", lw=4)]
    fig.legend(custom_lines, ['Target = 0 (Negative)', 'Target = 1 (Positive)'], 
               loc='lower center', ncol=2, bbox_to_anchor=(0.5, -0.02))

    plt.show()
    
    
def plot_correlation_matrix(df, variables, cmap="coolwarm", annot=True, figsize=(20,16)):
    
    # Select only the desired columns
    data = df[variables].copy()
    
    # Compute correlation matrix
    corr = data.corr()
    
    # Plot the heatmap
    plt.figure(figsize=figsize)
    sns.heatmap(corr, annot=annot, cmap=cmap, square=True, linewidths=0.5, vmin=-1, vmax=1)
    plt.title("Correlation Matrix")
    plt.tight_layout()
    plt.show()
    
def encode_categorical_raster(input_vector, prefix='class'):
    """
    Reclassifies categorical integers and creates one-hot encoded columns.
    
    Parameters:
    - input_vector: Array or Series containing categorical integer IDs (e.g., [1, 2, 3]).
    - prefix: String prefix for the resulting column names.
    
    Returns:
    - A DataFrame where each unique class has its own binary (0/1) column.
    """
    # Ensure the input is a pandas Series to use get_dummies effectively
    s = pd.Series(input_vector.astype(int))
    
    # Generate the one-hot encoded (dummy) variables
    # This prevents the model from assuming an ordinal rank between random IDs
    df_onehot = pd.get_dummies(s, prefix=prefix, dtype=int)
    
    return df_onehot

import math
import matplotlib.pyplot as plt

def plot_spatial_temporal_grid(df, variable, boundary_gdf, cols=3, cmap='viridis', marker_size=5):
    """
    Plots a grid of maps for each unique timestamp in the dataframe.
    
    Parameters:
    - df: DataFrame containing 'time', 'lon', 'lat', and the target variable.
    - variable: String, the column name to visualize.
    - boundary_gdf: GeoDataFrame for the background (e.g., country borders).
    - cols: Number of columns in the subplot grid.
    """
    
    # 1. Setup Data
    dates = sorted(df['time'].unique())
    vmin = df[variable].min()
    vmax = df[variable].max()

    # 2. Setup Subplots Grid
    rows = math.ceil(len(dates) / cols)
    fig, axs = plt.subplots(rows, cols, figsize=(12, 4 * rows), constrained_layout=True)
    
    # Handle cases where there is only 1 subplot
    if len(dates) == 1:
        axs = [axs]
    else:
        axs = axs.flatten()

    # 3. Loop through each date
    for i, date in enumerate(dates):
        ax = axs[i]
        df_year = df[df['time'] == date]
        
        # A. Plot Background (Countries)
        boundary_gdf.plot(ax=ax, edgecolor="black", facecolor="#ffffff", linewidth=0.8, zorder=1)
          
        # B. Plot the Data
        sc = ax.scatter(
            df_year.lon, 
            df_year.lat, 
            c=df_year[variable], 
            cmap=cmap,
            s=marker_size,
            vmin=vmin, 
            vmax=vmax,
            marker='s',
            zorder=2  # Higher zorder puts points on top of the boundary
        )
        
        # C. FIX: Force zoom to the data extent
        ax.set_xlim(df.lon.min() - 0.5, df.lon.max() + 0.5)
        ax.set_ylim(df.lat.min() - 0.5, df.lat.max() + 0.5)
        
        ax.set_title(f"Time: {date}", fontsize=12)
        ax.set_axis_off()
        ax.set_aspect('equal')
        
    # 4. Hide empty subplots
    for j in range(i + 1, len(axs)):
        axs[j].axis('off')

    # 5. Add a single Colorbar for the whole figure
    cbar = fig.colorbar(sc, ax=axs, orientation='horizontal', fraction=0.03, pad=0.04)
    cbar.set_label(f'{variable} Scale')

    return fig, axs

def plot_spatial_bias(df, treatment_col, outcome_col, time_col='time'):
    """
    Plots the spatial distribution of Treatment vs Outcome across years
    to identify if certain regions are consistently 'False Positives'.
    """
    # 1. Create the Confusion Categories
    # We use a copy to avoid SettingWithCopy warnings
    plot_df = df.copy()
    
    # Convert time to just the year for cleaner facets
    plot_df['year'] = pd.to_datetime(plot_df[time_col]).dt.year
    
    def get_cat(row):
        t, o = row[treatment_col], row[outcome_col]
        if t == 1 and o == 1: return '(T=1, O=1)'
        if t == 1 and o == 0: return '(T=1, O=0)'
        if t == 0 and o == 1: return '(T=0, O=1)'
        return '(T=0, O=0)'

    plot_df['Performance'] = plot_df.apply(get_cat, axis=1)

    # 2. Visualization Setup
    # Using a FacetGrid to see evolution over the years
    g = sns.FacetGrid(plot_df, col='year', col_wrap=3, height=4, 
                     hue='Performance', 
                     palette={'(T=1, O=1)': '#d18b00',        
                              '(T=1, O=0)': '#ebb95e', 
                              '(T=0, O=1)': '#1f77b4',        
                              '(T=0, O=0)': '#aec7e8'}) 
    
    # 3. Map the coordinates
    g.map(sns.scatterplot, 'lon', 'lat', s=25, alpha=0.8, edgecolor='none')
    
    # 4. Final touches
    g.add_legend(title=f"{treatment_col} vs {outcome_col}")
    g.set_titles("Year: {col_name}")
    plt.subplots_adjust(top=0.9)
    g.fig.suptitle(f'Spatial Performance Mapping: {treatment_col}', fontsize=16)
    
    plt.show()
    
def plot_causal_effects(df, effect_col='Estimated_Effect', lower_col='Lower_Bound', 
                        upper_col='Upper_Bound', label_col='zone_name', 
                        title="Causal Impact"):
    """
    Generates a sorted forest plot for causal effects with significance coloring.
    """
    # 1. Prepare and Sort Data
    plot_df = df.sort_values(effect_col, ascending=True).reset_index(drop=True)
    
    # 2. Determine Significance (if CI crosses 0)
    plot_df['significant'] = ~((plot_df[lower_col] <= 0) & (plot_df[upper_col] >= 0))
    colors = ['#2c7bb6' if sig else '#bababa' for sig in plot_df['significant']]

    # 3. Setup Plot
    fig, ax = plt.subplots(figsize=(10, 7), dpi=100)
    sns.set_style("white")

    # 4. Calculate Error Bars
    # xerr expects: [[lower_error_1, lower_error_2...], [upper_error_1, upper_error_2...]]
    errors_below = plot_df[effect_col] - plot_df[lower_col]
    errors_above = plot_df[upper_col] - plot_df[effect_col]
    xerr = [errors_below, errors_above]

    # 5. Plotting
    for i in range(len(plot_df)):
        ax.errorbar(
            plot_df.loc[i, effect_col], i, 
            xerr=[[errors_below[i]], [errors_above[i]]], 
            fmt='o', color=colors[i], capsize=5, elinewidth=2, markersize=8
        )
        
        # Add text labels for the exact values
        ax.text(
            plot_df.loc[i, upper_col] + (plot_df[effect_col].max() * 0.05), i, 
            f"{plot_df.loc[i, effect_col]:.3f}", 
            va='center', fontsize=10, color='#444444'
        )

    # 6. Styling
    ax.axvline(x=0, color='#d7191c', linestyle='--', linewidth=1.5, alpha=0.8)
    ax.set_yticks(range(len(plot_df)))
    ax.set_yticklabels(plot_df[label_col], fontsize=12)
    ax.set_xlabel('Treatment Effect (CATE)', fontsize=13, labelpad=10)
    ax.set_title(title, fontsize=15, fontweight='bold', pad=20)

    sns.despine(left=True, bottom=False)
    ax.grid(axis='x', linestyle=':', alpha=0.6)
    
    plt.tight_layout()
    return fig, ax