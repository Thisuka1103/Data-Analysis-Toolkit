from __future__ import annotations
from typing import Optional, Sequence, Tuple, Dict, Any, List
from pydantic import BaseModel, ValidationError, field_validator

import pandas as pd
import numpy as np
import io
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from google.colab import files
import scipy
from scipy.stats import chi2_contingency, pointbiserialr, f_oneway, multivariate_normal
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, MinMaxScaler, StandardScaler, RobustScaler
from sklearn.decomposition import FactorAnalysis
import json
import uuid
import inspect
import base64
import networkx as nx
import graphviz
from IPython.display import HTML, display

class DataInspector:
    """
    A comprehensive data cleaning and exploration tool for Google Colab.
    Provides interactive visualizations using Plotly and robust data sanitization.
    """

    def __init__(self):
        self.df = pd.DataFrame()
        self.numeric_df = pd.DataFrame()
        self.categorical_df = pd.DataFrame()
        self.categorical_normalized_df = pd.DataFrame()
        self.normalized_data_df = pd.DataFrame()
        self.numeric_normalized_df = pd.DataFrame()

    # Strings that should be treated as missing values
    _GARBAGE = {"?", "n/a", "na", "null", "none", "", " ", "nan", "N/A", "NULL", "None"}

    def upload_data(self) -> None:
        """
        Upload a CSV file from your local machine inside Google Colab.
        Automatically handles garbage strings and attempts type correction.
        """
        from google.colab import files
        uploaded = files.upload()
        if not uploaded:
            print("No file uploaded.")
            return
        filename = list(uploaded.keys())[0]
        content = uploaded[filename]
        self.df = pd.read_csv(
            io.BytesIO(content),
            na_values=list(self._GARBAGE),
            keep_default_na=True
        )
        self._auto_type_correction()
        self._split_column_types()
        print(f"✅ Loaded '{filename}': {self.df.shape[0]} rows × {self.df.shape[1]} columns")

    def load_from_path(self, path: str) -> None:
        """
        Load a CSV directly from a file path (e.g. a Colab sample dataset).
        Automatically handles garbage strings and type correction.
        """
        self.df = pd.read_csv(
            path,
            na_values=list(self._GARBAGE),
            keep_default_na=True
        )
        self._auto_type_correction()
        self._split_column_types()
        print(f"✅ Loaded '{path}': {self.df.shape[0]} rows × {self.df.shape[1]} columns")
        
    def load_from_path(self, path: str) -> None:
        """
        Load a CSV directly from a file path (e.g. a Colab sample dataset).
        Automatically handles garbage strings and type correction.
        """
        self.df = pd.read_csv(
            path,
            na_values=list(self._GARBAGE),
            keep_default_na=True
        )
        self._auto_type_correction()
        self._split_column_types()
        print(f"✅ Loaded '{path}': {self.df.shape[0]} rows × {self.df.shape[1]} columns")


    def _auto_type_correction(self) -> None:
        """Force-convert columns to numeric where possible without going all-null."""
        for col in self.df.columns:
            if self.df[col].dtype == object:
                converted = pd.to_numeric(self.df[col], errors="coerce")
                if converted.notna().sum() > 0:
                    self.df[col] = converted
                    
    def _split_column_types(self) -> None:
        """Separate numeric and categorical columns into dedicated DataFrames."""
        num_cols = self.df.select_dtypes(include=np.number).columns.tolist()
        cat_cols = self.df.select_dtypes(exclude=np.number).columns.tolist()
        self.numeric_df = self.df[num_cols].copy()
        self.categorical_df = self.df[cat_cols].copy()
    
    def get_summary(self):
        """
        Prints data dimensions and column type breakdown.
        Displays the first 20 rows of the DataFrame.
        """
        if self.df is None or self.df.empty:
            print("Error: No data loaded.")
            return

        numeric_columns = list(self.df.select_dtypes(include=[np.number]).columns)
        categorical_columns = list(self.df.select_dtypes(exclude=[np.number]).columns)

        print("--- Data Summary ---")
        print(f"Rows: {len(self.df)} | Columns: {len(self.df.columns)}")
        print(f"Numerical ({len(numeric_columns)}): {numeric_columns}")
        print(f"Categorical ({len(categorical_columns)}): {categorical_columns}")
        display(self.df.head(20))

    def show_missing_data(self):
        """
        Filters the DataFrame to show only rows containing at least one missing (NaN) value.
        """
        if self.df is None or self.df.empty:
            return
            
        mask_nan = self.df.isna().any(axis=1)
        mask_empty_str = (self.df == "").any(axis=1)
        combined_mask = mask_nan | mask_empty_str
        
        rows_with_missing = self.df[combined_mask]

        if len(rows_with_missing) == 0:
            print("No missing data found!")
        else:
            print(f"Found {len(rows_with_missing)} rows with missing values:")
            display(rows_with_missing)

    def delete_rows(self):
        """
        Deletes rows based on a comma-separated list of indices provided via user input.
        """
        if self.df is None or self.df.empty:
            return
            
        try:
            raw_input = input("Enter row indices to delete (e.g., 1, 3, 15): ")
            target_indices = [int(val.strip()) for val in raw_input.split(',') if val.strip().isdigit()]
            
            valid_indices = [idx for idx in target_indices if idx in self.df.index]
            self.df = self.df.drop(index=valid_indices).reset_index(drop=True)
            print(f"Deleted {len(valid_indices)} rows. New count: {len(self.df)}")
        except Exception as err:
            print(f"Error: {err}")

    def delete_columns(self):
        """
        Deletes columns based on a comma-separated list of names provided via user input.
        """
        if self.df is None or self.df.empty:
            print("No data loaded.")
            return

        try:
            print(f"Current columns: {', '.join(self.df.columns)}")
            raw_input = input("Enter column names to delete (e.g., Column1, Column2): ")
            target_cols = [c.strip() for c in raw_input.split(',')]
            
            valid_cols = [c for c in target_cols if c in self.df.columns]
            
            if not valid_cols:
                print("None of the provided column names were found.")
                return

            self.df = self.df.drop(columns=valid_cols)
            print(f"Deleted {len(valid_cols)} columns. Remaining count: {len(self.df.columns)}")
        except Exception as err:
            print(f"Error: {err}")

    def handle_missing_values(self, columns=None, strategy='median', fill_value=None):
        """
        Imputes missing values in specified columns to preserve data rows.

        Parameters:
        - columns: List of strings. If None, applies to all columns with NaNs.
        - strategy: 'mean', 'median', 'mode', or 'constant'.
        - fill_value: Used only if strategy is 'constant'.
        """
        if self.df is None or self.df.empty:
            return
            
        cols_to_process = columns if columns else [col for col in self.df.columns if self.df[col].isna().any()]

        for col in cols_to_process:
            is_num = pd.api.types.is_numeric_dtype(self.df[col])
            
            if strategy == 'mean' and is_num:
                impute_val = self.df[col].mean()
                self.df[col] = self.df[col].fillna(impute_val)
            elif strategy == 'median' and is_num:
                impute_val = self.df[col].median()
                self.df[col] = self.df[col].fillna(impute_val)
            elif strategy == 'mode':
                impute_val = self.df[col].mode().iloc[0] if not self.df[col].mode().empty else np.nan
                self.df[col] = self.df[col].fillna(impute_val)
            elif strategy == 'constant':
                self.df[col] = self.df[col].fillna(fill_value)

        print(f"🛠️ Imputation complete using '{strategy}' strategy for: {cols_to_process}")

    def remove_duplicates(self):
        """
        Identifies and removes exact duplicate rows from the DataFrame to prevent statistical bias.
        """
        if self.df is None or self.df.empty:
            return
            
        old_size = len(self.df)
        self.df = self.df.drop_duplicates(ignore_index=True)
        removed_count = old_size - len(self.df)
        print(f"Removed {removed_count} duplicate rows. New row count: {len(self.df)}")

    def export_cleaned_data(self, filename='cleaned_data.csv'):
        """
        Converts the current state of the DataFrame to a CSV file and
        triggers a browser download in the Google Colab environment.
        """
        if self.df is None or self.df.empty:
            return
            
        self.df.to_csv(filename, index=False)
        files.download(filename)
        print(f"💾 '{filename}' has been generated and download triggered.")

    def column_details(self):
        """
        Iterates through all columns to show numeric ranges or categorical unique value counts.
        """
        if self.df is None or self.df.empty:
            return
            
        for col_name in self.df.columns:
            if pd.api.types.is_numeric_dtype(self.df[col_name]):
                min_val = self.df[col_name].min()
                max_val = self.df[col_name].max()
                print(f"🔹 {col_name} (Numeric): Range [{min_val} to {max_val}]")
            else:
                unique_count = self.df[col_name].nunique()
                print(f"🔸 {col_name} (Categorical): {unique_count} unique values")

    def get_categorical_summary(self):
        """
        Generates a detailed statistical summary for categorical columns,
        including unique counts, the most frequent value (Mode), and its frequency.
        """
        if self.df is None or self.df.empty:
            return
            
        cat_data = self.df.select_dtypes(exclude=[np.number])
        if cat_data.empty:
            print("No categorical columns found.")
            return

        stat_summary = cat_data.describe().transpose()[['unique', 'top', 'freq']]
        print("--- Categorical Deep Dive ---")
        display(stat_summary)

    def extract_numeric_data(self):
        """
        Filters the DataFrame to include only numeric columns and triggers a download.
        """
        if self.df is None or self.df.empty:
            print("Error: No data loaded.")
            return None

        self.numeric_df = self.df.select_dtypes(include=[np.number]).copy()
        return self.numeric_df

    def extract_categorical_data(self):
        """
        Filters the DataFrame to include only categorical (non-numeric) columns and triggers a download.
        """
        if self.df is None or self.df.empty:
            print("Error: No data loaded.")
            return None

        self.categorical_df = self.df.select_dtypes(exclude=[np.number]).copy()
        return self.categorical_df

    def extract_normalized_numeric_data(self, method='minmax'):
        """
        Extracts numerical columns and scales them using the specified method.

        Parameters:
        - method: str, options are:
          * 'minmax': Scales features exactly to the [0, 1] range. 
                      Best for algorithms that assume a bounded range (e.g., Neural Networks).
          * 'standard': Centers features to a mean of 0 and standard deviation of 1.
                        Standard choice for PCA, Clustering, and Linear models.
          * 'robust': Uses the median and Interquartile Range (IQR). 
                      Best if your data has outliers that you don't want distorting the scaling.
        """
        if self.df is None or self.df.empty:
            print("Error: No data loaded.")
            return None

        num_subset = self.df.select_dtypes(include=[np.number]).copy()

        if num_subset.empty:
            print("⚠️ No numerical columns found to scale.")
            self.numeric_normalized_df = pd.DataFrame()
            return self.numeric_normalized_df

        if num_subset.isna().any().any():
            print("ℹ️ Missing values detected. Imputing with column medians before scaling...")
            num_subset = num_subset.fillna(num_subset.median())

        chosen_method = method.strip().lower()

        if chosen_method == 'minmax':
            scaler_obj = MinMaxScaler()
        elif chosen_method == 'standard':
            scaler_obj = StandardScaler()
        elif chosen_method == 'robust':
            scaler_obj = RobustScaler()
        else:
            print(f"❌ Unknown scaling method '{method}'. Defaulting to 'minmax'.")
            return self.extract_normalized_numeric_data(method='minmax')

        scaled_array = scaler_obj.fit_transform(num_subset)
        self.numeric_normalized_df = pd.DataFrame(scaled_array, columns=num_subset.columns, index=num_subset.index)

        print(f"✨ Successfully scaled numerical data using the '{chosen_method}' method.")
        return self.numeric_normalized_df

    def extract_normalized_categorical_data(self, method='uniform'):
        """
        Extracts categorical columns and applies the specified encoding method.
        
        Parameters:
        - method: str, options are:
          * 'uniform': Maps categories to numeric codes scaled to the [0, 1] range.
          * 'ordinal': Converts categories to distinct integers (0, 1, 2...) using OrdinalEncoder.
          * 'onehot': Converts categories to multiple binary (0 or 1) columns using OneHotEncoder.
          * 'minmax_ordinal': First encodes ordinally, then scales to exactly [0, 1] using MinMaxScaler.
        """
        if self.df is None or self.df.empty:
            print("Error: No data loaded.")
            return None

        cat_subset = self.df.select_dtypes(exclude=[np.number]).copy()

        if cat_subset.empty:
            print("⚠️ No categorical columns found to transform.")
            self.categorical_normalized_df = pd.DataFrame()
            return self.categorical_normalized_df

        chosen_method = method.strip().lower()

        if chosen_method == 'uniform':
            for column in cat_subset.columns:
                factorized_codes = cat_subset[column].astype('category').cat.codes
                max_val = factorized_codes.max()
                if max_val > 0:
                    cat_subset[column] = factorized_codes / max_val
                else:
                    cat_subset[column] = 0.0
            self.categorical_normalized_df = cat_subset

        elif chosen_method == 'ordinal':
            enc = OrdinalEncoder()
            filled_cats = cat_subset.fillna("Missing")
            transformed_arr = enc.fit_transform(filled_cats)
            self.categorical_normalized_df = pd.DataFrame(transformed_arr, columns=cat_subset.columns, index=cat_subset.index)

        elif chosen_method == 'onehot':
            enc = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
            filled_cats = cat_subset.fillna("Missing")
            transformed_arr = enc.fit_transform(filled_cats)
            cols_out = enc.get_feature_names_out(cat_subset.columns)
            self.categorical_normalized_df = pd.DataFrame(transformed_arr, columns=cols_out, index=cat_subset.index)

        elif chosen_method == 'minmax_ordinal':
            enc = OrdinalEncoder()
            scl = MinMaxScaler()
            filled_cats = cat_subset.fillna("Missing")
            ord_arr = enc.fit_transform(filled_cats)
            scaled_arr = scl.fit_transform(ord_arr)
            self.categorical_normalized_df = pd.DataFrame(scaled_arr, columns=cat_subset.columns, index=cat_subset.index)

        else:
            print(f"❌ Unknown method '{method}'. Defaulting to 'uniform'.")
            return self.extract_normalized_categorical_data(method='uniform')

        print(f"✨ Successfully encoded categorical data using the '{chosen_method}' method.")
        return self.categorical_normalized_df     

    def create_normalized_data_df(self):
        """
        Creates a single DataFrame containing the original numeric columns 
        merged side-by-side with the normalized categorical columns.
        """
        if self.df is None or self.df.empty:
            print("Error: No data loaded.")
            return None

        numerics = self.extract_numeric_data()
        categoricals = self.extract_normalized_categorical_data()

        if categoricals is None or categoricals.empty:
            print("ℹ️ No categorical columns found. Returning numeric DataFrame only.")
            self.normalized_data_df = numerics
            return self.normalized_data_df

        if numerics is None or numerics.empty:
            print("ℹ️ No numeric columns found. Returning normalized categorical DataFrame only.")
            self.normalized_data_df = categoricals
            return self.normalized_data_df

        self.normalized_data_df = pd.concat([numerics, categoricals], axis=1)
        print(f"✅ Success! Created merged DataFrame with {self.normalized_data_df.shape[1]} columns.")
        
        return self.normalized_data_df

    def plot_numerical(self, column_names):
        """
        Generates interactive Horizontal Violin, Scatter, and Histogram plots.
        Swapping the axis for Violin/Box plots to improve horizontal comparison.
        """
        if self.df is None or self.df.empty:
            return
            
        cols_to_plot = [column_names] if isinstance(column_names, str) else column_names
        numeric_cols_only = [c for c in cols_to_plot if c in self.df.columns and pd.api.types.is_numeric_dtype(self.df[c])]

        for c in numeric_cols_only:
            figure = make_subplots(
                rows=1, cols=3,
                subplot_titles=(f"Horizontal Violin/Box: {c}", f"Scatter Plot: {c}", f"Distribution: {c}")
            )

            figure.add_trace(
                go.Violin(x=self.df[c], box_visible=True, meanline_visible=True,
                        name=c, orientation='h', line_color='lightseagreen'),
                row=1, col=1
            )

            figure.add_trace(
                go.Scatter(y=self.df[c], mode='markers',
                        marker=dict(opacity=0.5, color='royalblue'), name=c),
                row=1, col=2
            )

            figure.add_trace(
                go.Histogram(x=self.df[c], name=c, marker_color='indianred'),
                row=1, col=3
            )

            figure.update_layout(
                height=450,
                title_text=f"<b>Statistical Analysis: {c}</b>",
                showlegend=False,
                template="plotly_white"
            )

            figure.update_xaxes(title_text="Value", row=1, col=1)
            figure.update_yaxes(title_text="Value", row=1, col=2)
            figure.update_xaxes(title_text="Value", row=1, col=3)

            figure.show()

    def plot_categorical(self, column_names):
        """
        Generates interactive Bar charts for categorical columns showing counts and percentages.
        """
        if self.df is None or self.df.empty:
            return
            
        cols_to_plot = [column_names] if isinstance(column_names, str) else column_names

        for c in cols_to_plot:
            freq_df = self.df[c].value_counts().reset_index()
            freq_df.columns = [c, 'count']
            freq_df['percentage'] = (freq_df['count'] / freq_df['count'].sum() * 100).round(1).astype(str) + '%'

            figure = px.bar(freq_df, x=c, y='count', text='percentage',
                         title=f"Frequency: {c}", color=c, color_discrete_sequence=px.colors.qualitative.Pastel)
            figure.show()

    def handle_outliers(self, columns=None, find_and_delete=False):
        """
        Flags outliers using IQR logic.
        Optionally deletes the flagged rows and updates the class instance.
        """
        if self.df is None or self.df.empty:
            return
            
        cols_to_check = columns if columns else self.df.select_dtypes(include=[np.number]).columns.tolist()
        outlier_indices = set()

        for c in cols_to_check:
            perc_25, perc_75 = self.df[c].quantile(0.25), self.df[c].quantile(0.75)
            inter_quartile_range = perc_75 - perc_25
            lower_bound = perc_25 - 1.5 * inter_quartile_range
            upper_bound = perc_75 + 1.5 * inter_quartile_range
            
            mask = (self.df[c] < lower_bound) | (self.df[c] > upper_bound)
            anomalies = self.df[mask]
            
            outlier_indices.update(anomalies.index.tolist())
            print(f"🚨 {c}: Found {len(anomalies)} outliers.")

        if outlier_indices:
            display(self.df.loc[list(outlier_indices)])
            if find_and_delete:
                self.df = self.df.drop(index=list(outlier_indices)).reset_index(drop=True)
                print(f"🗑️ Deleted {len(outlier_indices)} outlier rows.")

    def plot_relationship(self, col1, col2):
        """
        Intelligently selects the best interactive plot based on column types:
        - Num vs Num: Scatter with Trendline
        - Cat vs Num: Box plot with data points
        - Cat vs Cat: Grouped bar chart
        """
        if self.df is None or self.df.empty:
            return
            
        c1_is_num = pd.api.types.is_numeric_dtype(self.df[col1])
        c2_is_num = pd.api.types.is_numeric_dtype(self.df[col2])

        if c1_is_num and c2_is_num:
            figure = px.scatter(self.df, x=col1, y=col2, trendline="ols", title=f"Correlation: {col1} vs {col2}")
        elif not c1_is_num and not c2_is_num:
            figure = px.histogram(self.df, x=col1, color=col2, barmode="group", title=f"Relationship: {col1} vs {col2}")
        else:
            num_var, cat_var = (col1, col2) if c1_is_num else (col2, col1)
            figure = px.box(self.df, x=cat_var, y=num_var, points="all", color=cat_var, title=f"Distribution of {num_var} by {cat_var}")

        figure.show()

    def plot_numerical_correlation(self):
        """
        Displays an interactive Heatmap of the Pearson Correlation matrix
        for all numeric features in the dataset.
        """
        if self.df is None or self.df.empty:
            return
        
        num_data = self.df.select_dtypes(include=[np.number])
        correlation_matrix = num_data.corr(method='pearson')
        figure = px.imshow(correlation_matrix, text_auto=".2f", aspect="auto", color_continuous_scale='RdBu_r',
                        title="Pearson Correlation Heatmap")
        figure.show()

    def plot_categorical_correlation(self):
        """
        Calculates the Cramér's V association matrix for all categorical columns
        and displays it as an interactive Plotly Heatmap.
        """
        if self.df is None or self.df.empty:
            print("Error: No data loaded.")
            return None
            
        categorical_subset = self.df.select_dtypes(exclude=[np.number])
        
        if categorical_subset.empty:
            print("⚠️ No categorical columns found to compute associations.")
            return None
            
        headers = categorical_subset.columns
        size = len(headers)
        
        v_matrix = pd.DataFrame(np.zeros((size, size)), index=headers, columns=headers)
        
        for idx1 in range(size):
            for idx2 in range(idx1, size):
                h1 = headers[idx1]
                h2 = headers[idx2]
                
                if idx1 == idx2:
                    v_matrix.loc[h1, h2] = 1.0
                    continue
                    
                crosstab_res = pd.crosstab(categorical_subset[h1], categorical_subset[h2])
                
                if crosstab_res.size == 0 or min(crosstab_res.shape) <= 1:
                    v_matrix.loc[h1, h2] = 0.0
                    v_matrix.loc[h2, h1] = 0.0
                    continue
                    
                chi_stat = chi2_contingency(crosstab_res)[0]
                total_n = crosstab_res.sum().sum()
                
                if total_n > 0:
                    cramers_v = np.sqrt(chi_stat / (total_n * (min(crosstab_res.shape) - 1)))
                else:
                    cramers_v = 0.0
                    
                v_matrix.loc[h1, h2] = cramers_v
                v_matrix.loc[h2, h1] = cramers_v
                
        print("--- Cramér's V Association Matrix ---")
        display(v_matrix.round(3))
        
        figure = px.imshow(
            v_matrix,
            text_auto=".2f",
            aspect="auto",
            color_continuous_scale="RdBu_r",
            title="<b>Cramér's V Categorical Association Heatmap</b>",
            labels=dict(color="Cramér's V")
        )
        
        figure.update_layout(
            height=max(400, size * 80), 
            width=max(500, size * 80),
            template="plotly_white"
        )
        
        figure.show()
        return v_matrix

    def correlate_num_to_cat(self):
        """
        Computes associations between all numeric and categorical columns.
        - Uses Point-Biserial correlation for binary categories (-1 to 1).
        - Uses Eta (from ANOVA) for multi-class categories (0 to 1).
        """
        numerics = self.df.select_dtypes(include=[np.number]).columns
        categoricals = self.df.select_dtypes(exclude=[np.number]).columns

        if len(numerics) == 0 or len(categoricals) == 0:
            print("⚠️ Requires both numerical and categorical columns.")
            return pd.DataFrame()

        associations = []

        for c_col in categoricals:
            for n_col in numerics:
                pair_data = self.df[[c_col, n_col]].dropna()
                if pair_data.empty: 
                    continue
                
                unique_cats = pair_data[c_col].unique()
                if len(unique_cats) < 2:
                    continue 

                if len(unique_cats) == 2:
                    binary_series = pd.get_dummies(pair_data[c_col], drop_first=True).iloc[:, 0]
                    correlation_val, p_value = pointbiserialr(binary_series, pair_data[n_col])
                    associations.append({
                        'Categorical': c_col,
                        'Numerical': n_col,
                        'Type': 'Point-Biserial (Binary)',
                        'Correlation': round(correlation_val, 3),
                        'P-Value': round(p_value, 4)
                    })

                else:
                    group_slices = [pair_data[pair_data[c_col] == val][n_col] for val in unique_cats]
                    group_slices = [gs for gs in group_slices if len(gs) > 0]
                    
                    if len(group_slices) > 1:
                        f_stat, p_value = f_oneway(*group_slices)
                        
                        overall_avg = pair_data[n_col].mean()
                        sum_sq_total = ((pair_data[n_col] - overall_avg) ** 2).sum()
                        sum_sq_between = sum(len(gs) * (gs.mean() - overall_avg) ** 2 for gs in group_slices)
                        
                        if sum_sq_total > 0:
                            eta_squared = sum_sq_between / sum_sq_total
                            eta_stat = np.sqrt(eta_squared) 
                        else:
                            eta_stat = 0.0

                        associations.append({
                            'Categorical': c_col,
                            'Numerical': n_col,
                            'Type': "Eta (Multi-class ANOVA)",
                            'Correlation': round(eta_stat, 3),
                            'P-Value': round(p_value, 4)
                        })

        return pd.DataFrame(associations)

    def plot_all_associations_heatmap(self):
        """
        Creates a unified association matrix for BOTH categorical and numeric data
        and displays it as a single interactive Plotly Heatmap.
        """
        if self.df is None or self.df.empty:
            print("Error: No data loaded.")
            return None
            
        all_headers = self.df.columns
        dim = len(all_headers)
        
        unified_matrix = pd.DataFrame(np.zeros((dim, dim)), index=all_headers, columns=all_headers)
        
        for idx_x in range(dim):
            for idx_y in range(idx_x, dim):
                hx = all_headers[idx_x]
                hy = all_headers[idx_y]
                
                if idx_x == idx_y:
                    unified_matrix.loc[hx, hy] = 1.0
                    continue
                
                pair_df = self.df[[hx, hy]].dropna()
                if pair_df.empty:
                    continue
                
                x_num = pd.api.types.is_numeric_dtype(pair_df[hx])
                y_num = pd.api.types.is_numeric_dtype(pair_df[hy])
                
                if x_num and y_num:
                    strength = abs(pair_df[hx].corr(pair_df[hy], method='pearson'))
                    
                elif not x_num and not y_num:
                    ct = pd.crosstab(pair_df[hx], pair_df[hy])
                    if ct.size > 0 and min(ct.shape) > 1:
                        chi2_val = chi2_contingency(ct)[0]
                        total_obs = ct.sum().sum()
                        strength = np.sqrt(chi2_val / (total_obs * (min(ct.shape) - 1))) if total_obs > 0 else 0.0
                    else:
                        strength = 0.0
                        
                else:
                    cat_name, num_name = (hx, hy) if not x_num else (hy, hx)
                    
                    cat_distincts = pair_df[cat_name].unique()
                    if len(cat_distincts) > 1:
                        g_slices = [pair_df[pair_df[cat_name] == cd][num_name] for cd in cat_distincts]
                        g_slices = [g for g in g_slices if len(g) > 0]
                        
                        g_mean = pair_df[num_name].mean()
                        sst = ((pair_df[num_name] - g_mean) ** 2).sum()
                        ssb = sum(len(g) * (g.mean() - g_mean) ** 2 for g in g_slices)
                        
                        strength = np.sqrt(ssb / sst) if sst > 0 else 0.0
                    else:
                        strength = 0.0
                
                unified_matrix.loc[hx, hy] = round(strength, 3)
                unified_matrix.loc[hy, hx] = round(strength, 3)
                
        print("--- Global Association Matrix ---")
        display(unified_matrix)
        
        figure = px.imshow(
            unified_matrix,
            text_auto=".2f",
            aspect="auto",
            color_continuous_scale="viridis",
            title="<b>Unified Association Heatmap (Numeric & Categorical)</b>",
            labels=dict(color="Association Strength")
        )
        
        figure.update_layout(
            height=max(500, dim * 45),
            width=max(600, dim * 45),
            template="plotly_white"
        )
        
        figure.show()
        return unified_matrix

    def test_constant_mean(self, columns: Optional[Sequence[str]] = None, chunks: int = 10) -> Any:
        """
        Evaluates first moment homogeneity across sequential data blocks using MANOVA via Wilks' Lambda.
        Numerically stabilized via log-determinant tracking and shrinkage regularization.
        
        Parameters:
        - columns: Sequence of strings. If None, automatically extracts all numerical columns.
        - chunks: int, the number of non-overlapping consecutive blocks to split the data into.
        """
        if self.df is None or self.df.empty:
            raise ValueError("Error: No data loaded.")

        if columns is None:
            num_headers = list(self.df.select_dtypes(include=[np.number]).columns)
            if 'count' in num_headers:
                num_headers.remove('count')
            if not num_headers:
                raise ValueError("No numerical columns found automatically in the dataset.")
            cols_to_test = num_headers
        else:
            cols_to_test = list(columns)
            missing = [c for c in cols_to_test if c not in self.df.columns or not pd.api.types.is_numeric_dtype(self.df[c])]
            if missing:
                raise TypeError(f"Invalid columns specified: {missing}")

        n_rows = len(self.df)
        m_features = len(cols_to_test)
        split_size = n_rows // chunks
        
        if split_size < m_features:
            raise ValueError(f"Sample size per chunk ({split_size}) must be greater than features ({m_features}). Reduce chunks.")

        testing_df = self.df[cols_to_test].dropna().copy()
        n_rows = len(testing_df)
        
        testing_df['block_id'] = np.minimum(np.arange(n_rows) // split_size, chunks - 1)

        overall_avg = testing_df[cols_to_test].mean().values
        var_within = np.zeros((m_features, m_features))
        var_between = np.zeros((m_features, m_features))

        for block_num, block_data in testing_df.groupby('block_id'):
            block_mat = block_data[cols_to_test].values
            block_avg = block_mat.mean(axis=0)
            n_block = len(block_mat)
            
            var_within += np.dot((block_mat - block_avg).T, (block_mat - block_avg))
            avg_diff = (block_avg - overall_avg).reshape(-1, 1)
            var_between += n_block * np.dot(avg_diff, avg_diff.T)

        ridge_factor = 1e-6 * np.eye(m_features)
        var_within_safe = var_within + ridge_factor
        total_var_safe = var_within + var_between + ridge_factor

        sgn_w, logdet_w = np.linalg.slogdet(var_within_safe)
        sgn_t, logdet_t = np.linalg.slogdet(total_var_safe)

        if sgn_w <= 0 or sgn_t <= 0:
            raise np.linalg.LinAlgError("Variation matrices are non-invertible or poorly scaled.")

        log_lambda = logdet_w - logdet_t
        wilks_val = np.exp(log_lambda)

        dof_stat = m_features * (chunks - 1)
        chi2_approx = -(n_rows - 1 - (m_features + chunks) / 2) * log_lambda
        p_val = scipy.stats.chi2.sf(chi2_approx, dof_stat)

        print(f"\n--- MANOVA First Moment Homogeneity (Wilks' Lambda) ---")
        print(f"Chunks: {chunks} | Features: {m_features} | Valid Samples: {n_rows}")
        print(f"Wilks' Lambda (Λ): {wilks_val:.6f}")
        print(f"Bartlett's χ² Approx: {chi2_approx:.4f} (df={dof_stat})")
        print(f"p-value: {p_val:.6f}")

        if p_val > 0.05:
            print("✅ Success: Fail to reject H0. First moments (Means) are statistically stable.")
        else:
            print("🚨 Warning: Reject H0. Significant drift in first moments detected.")

        return {"wilks_lambda": wilks_val, "p_value": p_val, "chi2_stat": chi2_approx, "df": dof_stat}

    def test_constant_covariance(self, columns: Optional[Sequence[str]] = None, chunks: int = 10) -> Any:
        """
        Evaluates second moment homogeneity across sequential data blocks using Box's M-test.
        Numerically stabilized via shrinkage regularization and row-wise imputation protection.
        
        Parameters:
        - columns: Sequence of strings. If None, automatically extracts all numerical columns.
        - chunks: int, the number of non-overlapping consecutive blocks to split the data into.
        """
        if self.df is None or self.df.empty:
            raise ValueError("Error: No data loaded.")

        if columns is None:
            num_headers = list(self.df.select_dtypes(include=[np.number]).columns)
            if 'count' in num_headers:
                num_headers.remove('count')
            if not num_headers:
                raise ValueError("No numerical columns found.")
            cols_to_test = num_headers
        else:
            cols_to_test = list(columns)

        testing_df = self.df[cols_to_test].dropna().copy()
        n_rows = len(testing_df)
        m_features = len(cols_to_test)
        split_size = n_rows // chunks

        if split_size < m_features + 2:
            raise ValueError("Insufficient sample size per chunk for covariance estimation.")

        testing_df['block_id'] = np.minimum(np.arange(n_rows) // split_size, chunks - 1)

        pooled_cov = np.zeros((m_features, m_features))
        total_dof = 0
        block_covariances = []
        block_dofs = []

        for block_num, block_data in testing_df.groupby('block_id'):
            block_mat = block_data[cols_to_test].values
            dof = len(block_mat) - 1
            if dof < 1: continue
            
            cov_mat = np.cov(block_mat, rowvar=False) + 1e-6 * np.eye(m_features)
            block_covariances.append(cov_mat)
            block_dofs.append(dof)
            
            pooled_cov += dof * cov_mat
            total_dof += dof

        pooled_cov /= total_dof
        sgn_p, logdet_p = np.linalg.slogdet(pooled_cov)

        m_stat = total_dof * logdet_p
        for c_mat, d in zip(block_covariances, block_dofs):
            sgn_c, logdet_c = np.linalg.slogdet(c_mat)
            m_stat -= d * logdet_c

        c1 = (2 * m_features**2 + 3 * m_features - 1) / (6 * (m_features + 1) * (chunks - 1))
        scaling_sum = sum(1/d for d in block_dofs) - (1/total_dof)
        c1 *= scaling_sum

        chi2_approx = m_stat * (1 - c1)
        dof_stat = (chunks - 1) * m_features * (m_features + 1) / 2
        p_val = scipy.stats.chi2.sf(chi2_approx, dof_stat)

        print(f"\n--- Box's M-Test for Covariance Homogeneity ---")
        print(f"Box's M Statistic: {m_stat:.4f}")
        print(f"χ² Approx: {chi2_approx:.4f} (df={dof_stat})")
        print(f"p-value: {p_val:.6f}")

        if p_val > 0.05:
            print("✅ Success: Fail to reject H0. Covariance matrices are stable.")
        else:
            print("🚨 Warning: Reject H0. Significant heteroscedasticity detected.")

        return {"M_stat": m_stat, "p_value": p_val, "chi2_stat": chi2_approx, "df": dof_stat}

    def test_row_independence(self, columns: Optional[Sequence[str]] = None, max_lag: int = None) -> Any:
        """
        Evaluates row-to-row statistical independence using the Multivariate Ljung-Box Portmanteau test.
        Numerically stabilized via pseudo-inverse fallbacks and row-wise missing value removal.
        
        Parameters:
        - columns: Sequence of strings. If None, automatically extracts all numerical columns.
        - max_lag: int, maximum row-index offset to analyze. Defaults to ln(n).
        """
        if self.df is None or self.df.empty:
            raise ValueError("Error: No data loaded.")

        if columns is None:
            cols_to_test = list(self.df.select_dtypes(include=[np.number]).columns)
            if 'count' in cols_to_test: cols_to_test.remove('count')
        else:
            cols_to_test = list(columns)

        testing_df = self.df[cols_to_test].dropna().copy()
        n_rows = len(testing_df)
        m_features = len(cols_to_test)
        
        if max_lag is None:
            max_lag = int(np.log(n_rows))
            
        block_mat = testing_df.values
        block_avg = block_mat.mean(axis=0)
        centered_mat = block_mat - block_avg
        
        c0 = np.dot(centered_mat.T, centered_mat) / n_rows
        try:
            c0_inv = np.linalg.inv(c0)
        except np.linalg.LinAlgError:
            c0_inv = np.linalg.pinv(c0)
            
        q_stat = 0.0
        
        for lag in range(1, max_lag + 1):
            c_lag = np.dot(centered_mat[lag:].T, centered_mat[:-lag]) / n_rows
            term_mat = np.dot(np.dot(c_lag.T, c0_inv), np.dot(c_lag, c0_inv))
            q_stat += np.trace(term_mat) / (n_rows - lag)
            
        q_stat *= n_rows * (n_rows + 2)
        dof_stat = m_features**2 * max_lag
        p_val = scipy.stats.chi2.sf(q_stat, dof_stat)

        print(f"\n--- Multivariate Ljung-Box Test (Row Independence) ---")
        print(f"Lags Analyzed: {max_lag}")
        print(f"Q Statistic: {q_stat:.4f}")
        print(f"p-value: {p_val:.6f}")

        if p_val > 0.05:
            print("✅ Success: Fail to reject H0. Rows are independent.")
        else:
            print("🚨 Warning: Reject H0. Autocorrelation dependency detected.")

        return {"Q_stat": q_stat, "p_value": p_val, "df": dof_stat}

    def estimate_joint_normal(self, columns: Optional[Sequence[str]] = None) -> Any:
        """
        Operationalizes the micro-scale model X_i ~ N(mu_hat, S) by fitting a 
        parametric Multivariate Normal Distribution to the verified IID baseline.
        
        Utilizes finite-sample unbiased Maximum Likelihood Estimation (MLE) with 
        Bessel's correction to construct a continuous probabilistic boundary 
        for real-time anomaly detection.
        """
        if self.df is None or self.df.empty:
            raise ValueError("Error: No data loaded.")
            
        cols_to_test = columns if columns else list(self.df.select_dtypes(include=[np.number]).columns)
        if 'count' in cols_to_test: cols_to_test.remove('count')
        
        block_mat = self.df[cols_to_test].dropna().values
        mu_hat = np.mean(block_mat, axis=0)
        cov_hat = np.cov(block_mat, rowvar=False, ddof=1)
        
        cov_hat += np.eye(len(cols_to_test)) * 1e-6
        mvn_dist = scipy.stats.multivariate_normal(mean=mu_hat, cov=cov_hat)
        
        print("\n--- MLE Joint Normal Distribution Fitting ---")
        print(f"Estimated Mean Vector Dimensions: {mu_hat.shape}")
        print(f"Estimated Covariance Dimensions: {cov_hat.shape}")
        
        return {"mu_hat": mu_hat, "cov_hat": cov_hat, "distribution": mvn_dist}

    def instantiate_macro_clt_distribution(self, columns: Optional[Sequence[str]] = None) -> Any:
        """
        Operationalizes the macro-scale CLT model: mu_hat_n ~ N(mu_hat_n, (1/n) * S).
        
        This instantiates the continuous Gaussian sampling distribution of the empirical
        mean vector. It models parameter uncertainty rather than raw data variations,
        providing the mathematical foundation to track system drift and draw baseline 
        confidence ellipsoids.
        """
        if self.df is None or self.df.empty:
            raise ValueError("Error: No data loaded.")
            
        cols_to_test = columns if columns else list(self.df.select_dtypes(include=[np.number]).columns)
        if 'count' in cols_to_test: cols_to_test.remove('count')
        
        block_mat = self.df[cols_to_test].dropna().values
        n_obs = len(block_mat)
        mu_hat = np.mean(block_mat, axis=0)
        cov_hat = np.cov(block_mat, rowvar=False, ddof=1)
        
        macro_cov = cov_hat / n_obs
        macro_cov += np.eye(len(cols_to_test)) * 1e-8
        
        clt_dist = scipy.stats.multivariate_normal(mean=mu_hat, cov=macro_cov)
        
        print("\n--- Central Limit Theorem Macro Distribution ---")
        print(f"Sampling Mean Vector: {mu_hat.shape}")
        print(f"Sampling Covariance Matrix (Scaled by 1/n): {macro_cov.shape}")
        
        return {"mu_hat_n": mu_hat, "cov_hat_n": macro_cov, "clt_distribution": clt_dist}

    def compute_empirical_pca(self, columns: Optional[Sequence[str]] = None, show_plot: bool = True) -> Dict[str, Any]:
        """
        Operationalizes the geometric de-correlation framework via PCA.
        Decomposes the unbiased empirical covariance matrix S into its orthogonal 
        basis P_hat and diagonalized variance matrix Lambda_hat.
        
        Computes Hotelling's T^2 and Q (SPE) statistics across all possible 
        truncation boundaries k to optimize structural health monitoring thresholds.
        
        Generates an elite 2x3 Plotly Subplot Dashboard integrating the Feature Loading
        Heatmap alongside Eigenvalues, Variance Profiles, and T^2/Q tracking vs k.
        """
        if self.df is None or self.df.empty:
            raise ValueError("Error: No data loaded.")

        cols_to_use = columns if columns else list(self.df.select_dtypes(include=[np.number]).columns)
        if 'count' in cols_to_use: cols_to_use.remove('count')
        
        mat_x = self.df[cols_to_use].dropna().values
        num_n, num_m = mat_x.shape
        
        if num_n <= num_m:
            raise ValueError("Sample size must be larger than feature count.")
            
        avg_x = np.mean(mat_x, axis=0)
        centered_x = mat_x - avg_x
        
        cov_matrix = np.cov(centered_x, rowvar=False, ddof=1)
        eig_vals, eig_vecs = np.linalg.eigh(cov_matrix)
        
        sorted_idx = np.argsort(eig_vals)[::-1]
        lambda_arr = np.maximum(eig_vals[sorted_idx], 1e-15)
        p_basis = eig_vecs[:, sorted_idx]
        
        tot_variance = np.sum(lambda_arr)
        if tot_variance == 0:
            raise ValueError("Total variance is zero.")
            
        exp_var = lambda_arr / tot_variance
        cum_var = np.cumsum(exp_var)
        unexp_var = 1.0 - cum_var
        
        z_projected = np.dot(centered_x, p_basis)
        z_cov = np.cov(z_projected, rowvar=False, ddof=1)
        
        k_list = np.arange(1, num_m)
        t2_avgs = []
        q_avgs = []
        
        t2_full = np.zeros((num_n, len(k_list)))
        q_full = np.zeros((num_n, len(k_list)))
        
        for k_idx, k_val in enumerate(k_list):
            z_sub = z_projected[:, :k_val]
            l_sub = lambda_arr[:k_val]
            
            t2_vec = np.sum((z_sub**2) / l_sub, axis=1)
            t2_full[:, k_idx] = t2_vec
            t2_avgs.append(np.mean(t2_vec))
            
            z_res = z_projected[:, k_val:]
            q_vec = np.sum(z_res**2, axis=1)
            q_full[:, k_idx] = q_vec
            q_avgs.append(np.mean(q_vec))
            
        print("\n--- Operationalizing Geometric De-correlation Layer (PCA) ---")
        print(f"Total System Variance: {tot_variance:.4f}")
        
        if show_plot:
            names_pc = [f"PC {i+1}" for i in range(num_m)]
            names_k = [f"k={k}" for k in k_list]
            
            dashboard = make_subplots(
                rows=2, cols=3,
                subplot_titles=("Loadings |P_hat|", "Eigenvalues", "Explained Var", "Unexplained Var", "Mean T²", "Mean Q")
            )
            
            dashboard.add_trace(go.Heatmap(z=np.abs(p_basis), x=names_pc, y=cols_to_use, colorscale='YlOrRd'), row=1, col=1)
            dashboard.add_trace(go.Bar(x=names_pc, y=lambda_arr, marker_color='blue'), row=1, col=2)
            dashboard.add_trace(go.Bar(x=names_pc, y=exp_var*100, marker_color='orange'), row=1, col=3)
            dashboard.add_trace(go.Scatter(x=names_pc, y=cum_var*100, mode='lines+markers', marker_color='red'), row=1, col=3)
            dashboard.add_trace(go.Bar(x=names_pc, y=unexp_var*100, marker_color='green'), row=2, col=1)
            dashboard.add_trace(go.Scatter(x=names_k, y=t2_avgs, mode='lines+markers', marker_color='purple'), row=2, col=2)
            dashboard.add_trace(go.Scatter(x=names_k, y=q_avgs, mode='lines+markers', marker_color='pink'), row=2, col=3)
            
            dashboard.update_layout(height=750, width=1250, title_text="PCA Optimization Dashboard", showlegend=False)
            dashboard.show()
            
        return {
            "mean_vector": avg_x,
            "covariance_matrix_S": cov_matrix,
            "eigenvalues_lambda": lambda_arr,
            "eigenvectors_P": p_basis,
            "transformed_scores_Z": z_projected,
            "score_covariance_diagonal": np.diag(z_cov)
        }

    def compute_empirical_fa(self, k: int, columns: Optional[Sequence[str]] = None, show_plot: bool = True) -> Dict[str, Any]:
        """
        Operationalizes the Factor Analysis latent subspace framework.
        Decomposes the empirical correlation matrix R into shared structural 
        variance (Lambda * Lambda^T) and individual sensor uniqueness (Psi).
        
        Estimates the latent common factor scores using Thomson's MMSE regression method.
        
        Generates an elite 4-panel Plotly Subplot Dashboard analyzing Factor Loadings,
        Communality vs Uniqueness balances, Sensor Noise Scales, and Latent Factor Energy.
        """
        if self.df is None or self.df.empty:
            raise ValueError("Error: No data loaded.")

        cols_to_use = columns if columns else list(self.df.select_dtypes(include=[np.number]).columns)
        if 'count' in cols_to_use: cols_to_use.remove('count')
        
        mat_x = self.df[cols_to_use].dropna().values
        num_n, num_m = mat_x.shape
        
        if k >= num_m:
            raise ValueError("k must be less than feature count m.")
            
        avg_x = np.mean(mat_x, axis=0)
        std_x = np.std(mat_x, axis=0, ddof=1)
        std_x = np.maximum(std_x, 1e-15)
        
        z_norm = (mat_x - avg_x) / std_x
        r_mat = np.corrcoef(mat_x, rowvar=False)
        
        fa_model = FactorAnalysis(n_components=k, rotation='varimax', random_state=42)
        fa_model.fit(z_norm)
        
        loadings = fa_model.components_.T 
        uniqueness = fa_model.noise_variance_ 
        communality = np.sum(loadings**2, axis=1)
        f_scores = fa_model.transform(z_norm)
        
        print("\n--- Factor Analysis ---")
        print(f"Average Communality: {np.mean(communality)*100:.2f}%")
        
        if show_plot:
            f_names = [f"Factor {i+1}" for i in range(k)]
            
            dashboard = make_subplots(
                rows=2, cols=2,
                subplot_titles=("Factor Loadings", "Communality vs Uniqueness", "Uniqueness Profile", "Factor Variances")
            )
            
            dashboard.add_trace(go.Heatmap(z=np.abs(loadings), x=f_names, y=cols_to_use, colorscale='YlOrRd'), row=1, col=1)
            dashboard.add_trace(go.Bar(x=communality*100, y=cols_to_use, orientation='h', name='Communality'), row=1, col=2)
            dashboard.add_trace(go.Bar(x=uniqueness*100, y=cols_to_use, orientation='h', name='Uniqueness'), row=1, col=2)
            dashboard.add_trace(go.Scatter(x=cols_to_use, y=uniqueness, mode='lines+markers', name='Uniqueness'), row=2, col=1)
            
            f_vars = np.var(f_scores, axis=0, ddof=1)
            dashboard.add_trace(go.Bar(x=f_names, y=f_vars, name='Variance'), row=2, col=2)
            
            dashboard.update_layout(barmode='stack', height=750, width=1250, title_text="FA Diagnostics Dashboard", showlegend=False)
            dashboard.show()
            
        return {
            "mean_vector_mu": avg_x,
            "std_vector_D": std_x,
            "correlation_matrix_R": r_mat,
            "factor_loadings_lambda": loadings,
            "uniqueness_psi": uniqueness,
            "communality_h2": communality,
            "latent_factor_scores_F": f_scores
        }

class PlottingMethods:
    def get_methods_info(self, user_id=None):
        info_list = []
        for n, m in inspect.getmembers(self, inspect.ismethod):
            if not n.startswith('_'):
                sig = inspect.signature(m)
                doc = inspect.getdoc(m) or "No description available"
                info_list.append({"method": n, "signature": str(sig), "description": doc})
        return {'status': 'success', 'response': info_list}

    def _data_validate(self, data, message_dict):
        if data is None or data == "":
            message_dict['message'] = 'No data'
            return {'status': 'error', 'message_dict': message_dict}
            
        if isinstance(data, pd.DataFrame):
            if data.empty:
                message_dict['message'] = 'No data'
                return {'status': 'error', 'message_dict': message_dict}
            return {'status': 'success', 'data': data.to_dict(orient='records')}
            
        if isinstance(data, list):
            if not data:
                message_dict['message'] = 'No data'
                return {'status': 'error', 'message_dict': message_dict}
            return {'status': 'success', 'data': data}
            
        try:
            parsed = json.loads(data)
            records = parsed.get('records', parsed) if isinstance(parsed, dict) else parsed
            if not records:
                message_dict['message'] = 'No data'
                return {'status': 'error', 'message_dict': message_dict}
            return {'status': 'success', 'data': records}
        except BaseException:
            message_dict['message'] = 'Invalid data format'
            return {'status': 'error', 'message_dict': message_dict}

    def plot_bar_chart(self, x='date', y='value', color=None, text=None, title='', barmode='stack', hover_data=None, data_id=None, data='{"records":[]}', meta_data={}, user_id=None):
        """
        Given a list of dictionaries, plot a Plotly px bar chart with x as the x values,
        y as the y values, and color as the categories. The variable barmode specifies if the mode of grouping.

        Args:
            x (str): Column name for the x-axis.
            y (str): Column name for the y-axis.
            color (str): Optional - Column name for the stacking categories.
            text (str or None): Optional - Column name for text labels.
            title (str): Optional - Title of the chart.
            barmode (str or None): Optional - The bar mode either stack or group.
            hover_data (list or None): Optional - List of column names to include in hover data.
            data (str): JSON string containing a list of records in the format
                        {'records': [{'x': ..., 'y': ..., 'color': ...}, ...]}.

        Returns:
            dict: A dictionary with the status and the generated Plotly figure.
        """
        msg_dict = {'message': meta_data}
        valid_res = self._data_validate(data, msg_dict)
        if valid_res['status'] != 'success':
            return {'status': 'error', 'response': {'meta_data': valid_res['message_dict'], 'data': json.dumps({'figure': ''})}, 'message': valid_res['message_dict'].get('message', 'Err')}

        df_plt = pd.DataFrame(valid_res['data'])
        df_plt[y] = pd.to_numeric(df_plt[y], errors='coerce')
        
        if isinstance(hover_data, str):
            try:
                hd = json.loads(hover_data)
                hover_data = hd if isinstance(hd, list) else None
            except:
                hover_data = hover_data.split(',') if ',' in hover_data else None

        fig = px.bar(df_plt, x=x, y=y, color=color, title=title, text=text, hover_data=hover_data, barmode=barmode)
        raw_html = fig.to_html(full_html=False, include_plotlyjs=True)
        unique_div = raw_html.replace('<div>', f'<div id="{uuid.uuid4().hex[:8]}">')
        
        msg_dict['message'] = 'Bar chart plotted'
        return {'status': 'success', 'response': {'meta_data': msg_dict, 'data': json.dumps({'figure': unique_div}), 'message': json.dumps(msg_dict)}}

    def plot_pie_chart(self, names='date', values='value', title='', hole=None, data_id=None, data='{"records":[]}', meta_data={}, user_id=None):
        """
        Generates a responsive Plotly pie chart based on provided or previously stored data.

        This method creates a pie chart using the Plotly Express library. Data can be provided directly
        via a JSON string (`data`) or indirectly through a reference ID (`data_id`) which is used to
        retrieve previously stored data.

        Parameters:
            names (str): Column name in the dataset to use for pie chart segment labels (default: 'date').
            values (str): Column name in the dataset to use for segment sizes (default: 'value').
            title (str): Title of the pie chart (default: empty string).
            data_id (str, optional): Optional ID for retrieving stored data via `DBQ.get_ai_message_stored_data`.
            data (str, optional): JSON string in the format `{"records": [...]}` representing the dataset.
                                Used if `data_id` is not provided.
            meta_data (dict): Additional metadata dictionary to include in the response.

        Returns:
            dict: A dictionary with:
                - 'status' (str): 'success' if the chart is generated; 'error' if there is an issue.
                - 'response' (dict):
                    - 'meta_data' (dict): Includes original meta plus status or error message.
                    - 'data' (str): JSON-encoded string containing the HTML representation of the pie chart.
                    - 'message' (str): A stringified version of `meta_data` for convenience.

        Example:
            >>> plot_pie_chart(
                    names='category',
                    values='count',
                    title='Category Distribution',
                    data=json.dumps({"records": [{"category": "A", "count": 40}, {"category": "B", "count": 60}]})
                )
            {
                'status': 'success',
                'response': {
                    'meta_data': {'message': 'Pie chart plotted'},
                    'data': '{"figure": "<div id='abc123'>...</div>"}',
                    'message': '{"message": "Pie chart plotted"}'
                }
            }

        Notes:
            - The generated Plotly chart is converted into HTML for embedding in web pages.
            - If both `data` and `data_id` are missing or invalid, an appropriate error is returned.
            - Ensures responsiveness and disables the Plotly logo in the exported chart.
            - Each chart is given a unique `div` ID for safe embedding.

        Raises:
            Exception: Any errors during data retrieval or plotting are caught and returned in the response.
        """
        msg_dict = {'message': meta_data}
        valid_res = self._data_validate(data, msg_dict)
        if valid_res['status'] != 'success':
            return {'status': 'error', 'response': {'meta_data': valid_res['message_dict'], 'data': json.dumps({'figure': ''})}, 'message': valid_res['message_dict'].get('message', 'Err')}

        df_plt = pd.DataFrame(valid_res['data'])
        fig = px.pie(df_plt, names=names, values=values, title=title, hole=hole)
        
        raw_html = fig.to_html(full_html=False, include_plotlyjs=True)
        unique_div = raw_html.replace('<div>', f'<div id="{uuid.uuid4().hex[:8]}">')
        
        msg_dict['message'] = 'Pie chart plotted'
        return {'status': 'success', 'response': {'meta_data': msg_dict, 'data': json.dumps({'figure': unique_div}), 'message': json.dumps(msg_dict)}}

    def plot_histogram(self, x='value', title='', bins=None, data='{"records":[]}', meta_data={}, user_id=None):
        """
        Given a list of dictionaries, plot a Plotly px histogram with the specified column as the x-axis.

        Args:
            x (str): Column name for the x-axis.
            title (str): Title of the histogram.
            bins (list or None): Custom bin intervals for the histogram.
            data (str): JSON string containing a list of records.

        Returns:
            dict: A dictionary with the status and the generated Plotly figure.
        """
        msg_dict = {'message': meta_data}
        valid_res = self._data_validate(data, msg_dict)
        if valid_res['status'] != 'success':
            return {'status': 'error', 'response': {'meta_data': valid_res['message_dict'], 'data': json.dumps({'figure': ''})}, 'message': valid_res['message_dict'].get('message', 'Err')}

        df_plt = pd.DataFrame(valid_res['data'])
        if bins:
            df_plt[x] = pd.cut(df_plt[x], bins=bins).astype(str)
            
        fig = px.histogram(df_plt, x=x, title=title)
        
        raw_html = fig.to_html(full_html=False, include_plotlyjs=True)
        unique_div = raw_html.replace('<div>', f'<div id="{uuid.uuid4().hex[:8]}">')
        
        msg_dict['message'] = 'Histogram plotted'
        return {'status': 'success', 'response': {'meta_data': msg_dict, 'data': json.dumps({'figure': unique_div}), 'message': json.dumps(msg_dict)}}

    def plot_simple_sunburst_graph(self, path=["parent", "name"], values="marks", title='Hierarchy map', data_id=None, data='{"records":[]}', meta_data={}, user_id=None):
        """
        Plots a Plotly Sunburst graph given hierarchical data.

        Args:
            data (str): JSON string of hierarchical records.
            path (list): List defining the hierarchy in the dataset.
            values (str): Column name to use for slice sizes.
            title (str): Title of the Sunburst plot.

        Returns:
            dict: A response containing the Plotly figure.
        """
        msg_dict = {'message': meta_data}
        valid_res = self._data_validate(data, msg_dict)
        if valid_res['status'] != 'success':
            return {'status': 'error', 'response': {'meta_data': valid_res['message_dict'], 'data': json.dumps({'figure': ''})}, 'message': valid_res['message_dict'].get('message', 'Err')}

        df_plt = pd.DataFrame(valid_res['data']).fillna('')
        fig = px.sunburst(df_plt, path=path, values=values, title=title)
        
        raw_html = fig.to_html(full_html=False, include_plotlyjs=True)
        unique_div = raw_html.replace('<div>', f'<div id="{uuid.uuid4().hex[:8]}">')
        
        msg_dict['message'] = 'Sunburst plotted'
        return {'status': 'success', 'response': {'meta_data': msg_dict, 'data': json.dumps({'figure': unique_div}), 'message': json.dumps(msg_dict)}}

    def plot_tree_map(self, path=["parent", "name"], values="marks", title='Hierarchy map', data_id=None, data='{"records":[]}', meta_data={}, user_id=None):
        """
        Given a list of hierarchical dictionaries, plot a plotly px treemap with the specified path.
        """
        msg_dict = {'message': meta_data}
        valid_res = self._data_validate(data, msg_dict)
        if valid_res['status'] != 'success':
            return {'status': 'error', 'response': {'meta_data': valid_res['message_dict'], 'data': json.dumps({'figure': ''})}, 'message': valid_res['message_dict'].get('message', 'Err')}

        df_plt = pd.DataFrame(valid_res['data']).fillna('')
        fig = px.treemap(df_plt, path=path, values=values, title=title)
        
        raw_html = fig.to_html(full_html=False, include_plotlyjs=True)
        unique_div = raw_html.replace('<div>', f'<div id="{uuid.uuid4().hex[:8]}">')
        
        msg_dict['message'] = 'Treemap plotted'
        return {'status': 'success', 'response': {'meta_data': msg_dict, 'data': json.dumps({'figure': unique_div}), 'message': json.dumps(msg_dict)}}

    def plot_sankey_diagram(self, source_column="parent", target_column="name", values="marks", title="Sankey Diagram", data_id=None, data='{"records":[]}', meta_data={}, user_id=None):
        """
        Plot a Plotly Sankey diagram
        using the specified source and target columns.

        Args:
            source_column (str): The column name to use as the source in the Sankey diagram.
            target_column (str): The column name to use as the target in the Sankey diagram.
            values (str): The column name to use as the values in the Sankey diagram.
            title (str): The title of the Sankey diagram.
            data (str): JSON string containing the records of hierarchical data.

        Returns:
            dict: A dictionary containing:
                  - status (str): 'success' or 'error'.
                  - response (dict):
                      - message (str): Success or error message.
                      - data (dict): Contains the Sankey diagram figure as a Plotly figure object.
        """
        msg_dict = {'message': meta_data}
        valid_res = self._data_validate(data, msg_dict)
        if valid_res['status'] != 'success':
            return {'status': 'error', 'response': {'meta_data': valid_res['message_dict'], 'data': json.dumps({'figure': ''})}, 'message': valid_res['message_dict'].get('message', 'Err')}

        df_plt = pd.DataFrame(valid_res['data'])
        grped = df_plt.groupby([source_column, target_column], as_index=False)[values].sum()
        
        node_names = pd.concat([grped[source_column], grped[target_column]]).unique().tolist()
        node_map = {n: i for i, n in enumerate(node_names)}
        
        src_idx = grped[source_column].map(node_map).tolist()
        tgt_idx = grped[target_column].map(node_map).tolist()
        vals = grped[values].tolist()
        
        fig = go.Figure(go.Sankey(
            node=dict(label=node_names),
            link=dict(source=src_idx, target=tgt_idx, value=vals)
        ))
        fig.update_layout(title=title)
        
        raw_html = fig.to_html(full_html=False, include_plotlyjs=True)
        unique_div = raw_html.replace('<div>', f'<div id="{uuid.uuid4().hex[:8]}">')
        
        msg_dict['message'] = 'Sankey plotted'
        return {'status': 'success', 'response': {'meta_data': msg_dict, 'data': json.dumps({'figure': unique_div}), 'message': json.dumps(msg_dict)}}

    def plot_sunburst_from_hierarchy(self, path=["parent", "name"], values="marks", title="Sunburst Diagram",data_id=None, data="{'records':[]}", meta_data={}, user_id=None):
        """
        Generate a Plotly Sunburst chart from hierarchical data.
        """
        msg_dict = {'message': meta_data}
        valid_res = self._data_validate(data, msg_dict)
        if valid_res['status'] != 'success':
            return {'status': 'error', 'response': {'meta_data': valid_res['message_dict'], 'data': json.dumps({'figure': ''})}, 'message': valid_res['message_dict'].get('message', 'Err')}

        df_plt = pd.DataFrame(valid_res['data']).fillna('')
        fig = px.sunburst(df_plt, path=path, values=values, title=title)
        
        raw_html = fig.to_html(full_html=False, include_plotlyjs=True)
        unique_div = raw_html.replace('<div>', f'<div id="{uuid.uuid4().hex[:8]}">')
        
        msg_dict['message'] = 'Hierarchical sunburst plotted'
        return {'status': 'success', 'response': {'meta_data': msg_dict, 'data': json.dumps({'figure': unique_div}), 'message': json.dumps(msg_dict)}}

    def plot_heat_map(self, values, index, columns, aggregade_method='sum', fill_value=0, title='', width=None, data_id=None, data='{"records":[]}', meta_data={}, user_id=None):
        """
        Generates an interactive Plotly heatmap from tabular data, with optional aggregation and layout control.

        This method builds a heatmap visualization using a pivoted version of the input dataset.
        Data can be provided either directly (`data`) or fetched via a reference `data_id` using a stored data lookup.

        Parameters:
            values (str): The column whose values will be visualized in the heatmap (e.g., grades, sales).
            index (str): The column to use as the y-axis in the pivot table.
            columns (str): The column to use as the x-axis in the pivot table.
            aggregade_method (str): Aggregation method to apply when pivoting ('sum', 'mean', 'count', etc.). Defaults to 'sum'.
            fill_value (int or float): Value used to fill missing cells in the pivot table. Defaults to 0.
            title (str): Title of the heatmap.
            width (int, optional): Optional fixed width of the chart in pixels. If None, it's auto-sized.
            data_id (str, optional): Optional ID to fetch data from stored source using `DBQ.get_ai_message_stored_data`.
            data (str, optional): A JSON string in the format `{"records": [...]}`. Used if `data_id` is not provided.
            meta_data (dict): Metadata dictionary to propagate through the response.

        Returns:
            dict: A dictionary containing:
                - 'status' (str): 'success' if the heatmap was created, otherwise 'error'.
                - 'response' (dict):
                    - 'meta_data' (dict): Extended metadata including any error or success messages.
                    - 'data' (str): JSON string with an HTML-embedded Plotly figure (`figure`).
                    - 'message' (str): Same as meta_data['message'], serialized for frontend use.

        Example:
            >>> plot_heat_map(
                    values='grade',
                    index='student_name',
                    columns='module',
                    aggregade_method='mean',
                    title='Average Grades per Module',
                    data=json.dumps({"records": [{"student_name": "Alice", "module": "Math", "grade": 85}, ...]})
                )
            {
                'status': 'success',
                'response': {
                    'meta_data': {'message': 'Heat map plotted'},
                    'data': '{"figure": "<div id='abcd1234'>...</div>"}',
                    'message': '{"message": "Heat map plotted"}'
                }
            }

        Notes:
            - The pivoted DataFrame is plotted using `plotly.express.imshow()`.
            - The chart is rendered in HTML for easy embedding in web pages or dashboards.
            - Unique div IDs are assigned to avoid DOM conflicts in the frontend.
            - The layout is responsive and resizes fluidly within containers.

        Raises:
            Exception: Any errors during data processing or plotting are captured and returned in the response.
        """
        msg_dict = {'message': meta_data}
        valid_res = self._data_validate(data, msg_dict)
        if valid_res['status'] != 'success':
            return {'status': 'error', 'response': {'meta_data': valid_res['message_dict'], 'data': json.dumps({'figure': ''})}, 'message': valid_res['message_dict'].get('message', 'Err')}

        df_plt = pd.DataFrame(valid_res['data'])
        pivot_df = pd.pivot_table(df_plt, values=values, index=index, columns=columns, aggfunc=aggregade_method, fill_value=fill_value)
        
        fig = px.imshow(pivot_df, title=title)
        if width: fig.update_layout(width=width)
        
        raw_html = fig.to_html(full_html=False, include_plotlyjs=True)
        unique_div = raw_html.replace('<div>', f'<div id="{uuid.uuid4().hex[:8]}">')
        
        msg_dict['message'] = 'Heat map plotted'
        return {'status': 'success', 'response': {'meta_data': msg_dict, 'data': json.dumps({'figure': unique_div}), 'message': json.dumps(msg_dict)}}

    def plot_multi_column_bar_graph(self, xLabel, value_vars, title='', hover_data=None, barmode='group', data_id=None, data='{"records":[]}', meta_data={}, user_id=None):
        msg_dict = {'message': meta_data}
        valid_res = self._data_validate(data, msg_dict)
        if valid_res['status'] != 'success':
            return {'status': 'error', 'response': {'meta_data': valid_res['message_dict'], 'data': json.dumps({'figure': ''})}, 'message': valid_res['message_dict'].get('message', 'Err')}

        df_plt = pd.DataFrame(valid_res['data'])
        melted_df = df_plt.melt(id_vars=[xLabel], value_vars=value_vars)
        
        fig = px.bar(melted_df, x=xLabel, y='value', color='variable', barmode=barmode, title=title)
        
        raw_html = fig.to_html(full_html=False, include_plotlyjs=True)
        unique_div = raw_html.replace('<div>', f'<div id="{uuid.uuid4().hex[:8]}">')
        
        msg_dict['message'] = 'Multi column bar plotted'
        return {'status': 'success', 'response': {'meta_data': msg_dict, 'data': json.dumps({'figure': unique_div}), 'message': json.dumps(msg_dict)}}

    def plot_flow_chart(self, data_id=None, data='{"records":[]}', meta_data={}, user_id=None):
        """
        Generates and returns a flowchart visualization from given process data using Graphviz and NetworkX.

        This method accepts either a data_id to fetch pre-stored data, or a raw data string containing flowchart
        records in JSON format. The resulting flowchart is rendered as a PNG image, base64-encoded, and returned
        as HTML along with meta information and status.

        Edge and node attributes are accessed using `.get()` with sensible defaults, ensuring robust handling of missing fields.
        Edges now support optional custom color and thickness (penwidth).

        Args:
            data_id (str, optional): An identifier used to retrieve stored process data. If provided, overrides `data`.
            data (str, optional): A JSON string with a "records" dictionary, containing "edges" and "nodes" for the flowchart.
            meta_data (dict, optional): Metadata dictionary to include in the response.
            user_id (str, optional): Optional user identifier for logging or auditing purposes.

        Returns:
            dict: A dictionary containing:
                - 'status': 'success' or 'error'
                - 'response': {
                    'meta_data': Meta information and messages,
                    'data': JSON string with an embedded PNG image (base64-encoded HTML figure)
                }
                - 'message': JSON-encoded message describing the outcome.

        Raises:
            None: All exceptions are caught and reported in the 'status':'error' response.

        Data Structure:
            The input data should be a JSON object with the following format:

            {
                "records": {
                    "edges": [
                        {
                            "start": "<source_node_label>",      # (str) Required. The source node's label or ID.
                            "end": "<destination_node_label>",   # (str) Required. The destination node's label or ID.
                            "label": "<edge_label>",             # (str) Optional. The label displayed on the edge.
                            "color": "<color>",                  # (str) Optional. Edge color (e.g. '#ff0000', 'blue'). Default: 'black'
                            "penwidth": <number>,                # (float/int) Optional. Edge thickness. Default: 1
                        },
                        ...
                    ],
                    "nodes": [
                        {
                            "label": "<node_label>",         # (str) Required. Node's unique label or ID.
                            "shape": "<shape>",              # (str) Optional. Graphviz shape (e.g. 'box', 'ellipse'). Default: 'ellipse'
                            "style": "<style>",              # (str) Optional. Node style (e.g. 'filled'). Default: 'filled'
                            "fillcolor": "<color>",          # (str) Optional. Fill color (e.g. '#abcdef'). Default: '#bbbbbb'
                            "fontcolor": "<color>"           # (str) Optional. Font color (e.g. '#000000'). Default: 'black'
                        },
                        ...
                    ]
                }
            }

        Example:
            result = self.plot_flow_chart(data_id='abc123')
            # or
            result = self.plot_flow_chart(data='{"records": {"edges": [...], "nodes": [...]}}')

        Notes:
            - At least one of `data_id` or `data` must provide valid flowchart records.
            - Handles missing or invalid data gracefully, always returning a well-structured response.
            - Uses `.get()` with default values for node and edge attributes to prevent errors from missing fields.
            - Edges can be customized with `color` and `penwidth`.
            - Uses Graphviz to render the flowchart, and NetworkX for graph data manipulation.
        """
        msg_dict = {'message': meta_data}
        valid_res = self._data_validate(data, msg_dict)
        if valid_res['status'] != 'success':
            return valid_res

        rec = valid_res['data']
        edges = rec.get('edges', [])
        nodes = rec.get('nodes', [])
        
        dot = graphviz.Digraph(format='png')
        for n in nodes:
            dot.node(n.get('label', ''), shape=n.get('shape', 'ellipse'), style=n.get('style', 'filled'), fillcolor=n.get('fillcolor', 'white'), fontcolor=n.get('fontcolor', 'black'))
        for e in edges:
            dot.edge(e.get('start', ''), e.get('end', ''), label=e.get('label', ''), color=e.get('color', 'black'), penwidth=str(e.get('penwidth', 1)))
            
        png_bytes = dot.pipe()
        b64 = base64.b64encode(png_bytes).decode('utf-8')
        html_str = f'<img src="data:image/png;base64,{b64}">'
        
        msg_dict['message'] = 'Flowchart generated'
        return {'status': 'success', 'response': {'meta_data': msg_dict, 'data': json.dumps({'figure': html_str}), 'message': json.dumps(msg_dict)}}

    def plot_flow_chart_plotly(self, data_id=None, data='{"records":[]}', meta_data={}, user_id=None):
        """
        Generates and returns a flowchart visualization from given process data using Plotly and NetworkX.
        (Docstring omitted for brevity - keep yours.)
        """
        msg_dict = {'message': meta_data}
        valid_res = self._data_validate(data, msg_dict)
        if valid_res['status'] != 'success':
            return valid_res

        rec = valid_res['data']
        edges = rec.get('edges', [])
        
        G = nx.DiGraph()
        for e in edges:
            G.add_edge(e.get('start', ''), e.get('end', ''))
            
        pos = nx.spring_layout(G)
        
        edge_x = []
        edge_y = []
        for edge in G.edges():
            x0, y0 = pos[edge[0]]
            x1, y1 = pos[edge[1]]
            edge_x.extend([x0, x1, None])
            edge_y.extend([y0, y1, None])
            
        edge_trace = go.Scatter(x=edge_x, y=edge_y, line=dict(width=1, color='#888'), mode='lines')
        
        node_x = []
        node_y = []
        for node in G.nodes():
            x, y = pos[node]
            node_x.append(x)
            node_y.append(y)
            
        node_trace = go.Scatter(x=node_x, y=node_y, mode='markers+text', text=list(G.nodes()), textposition="bottom center", marker=dict(size=10, color='lightblue'))
        
        fig = go.Figure(data=[edge_trace, node_trace])
        raw_html = fig.to_html(full_html=False, include_plotlyjs=True)
        unique_div = raw_html.replace('<div>', f'<div id="{uuid.uuid4().hex[:8]}">')
        
        msg_dict['message'] = 'Plotly flowchart generated'
        return {'status': 'success', 'response': {'meta_data': msg_dict, 'data': json.dumps({'figure': unique_div}), 'message': json.dumps(msg_dict)}}

    def display_image(self, result):
        if result['status'] == 'success':
            html_content = json.loads(result['response']['data']).get('figure', '')
            display(HTML(html_content))
        else:
            print("Error generating graph:", result.get('message'))







