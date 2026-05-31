# DataInspector — User Guide

A beginner-friendly reference for every function in the `DataInspector` and `PlottingMethods` classes.

---

## Table of Contents

1. [What is DataInspector?](#what-is-datainspector)
2. [How to Get Started](#how-to-get-started)
3. [The Full Workflow](#the-full-workflow)
4. [Category 1 — Data Ingestion](#category-1--data-ingestion)
5. [Category 2 — Structural Analysis](#category-2--structural-analysis)
6. [Category 3 — Data Cleaning](#category-3--data-cleaning)
7. [Category 4 — Feature Engineering & Normalization](#category-4--feature-engineering--normalization)
8. [Category 5 — Visualization](#category-5--visualization)
9. [Category 6 — Correlation & Association](#category-6--correlation--association)
10. [Category 7 — Custom Plotting (PlottingMethods)](#category-7--custom-plotting-plottingmethods)
11. [Category 8 — Export](#category-8--export)
12. [Quick Reference Table](#quick-reference-table)

---

## What is DataInspector?

`DataInspector` is a Python class built for Google Colab that automates the most repetitive parts of data science work — cleaning messy data, handling missing values, normalizing features, and visualizing patterns — all in one place.

Instead of writing 50 lines of pandas code every time you get a new dataset, you call simple methods like `inspector.handle_missing_values()` or `inspector.plot_all_associations_heatmap()` and the tool does the work for you.

`PlottingMethods` is a companion class for generating standalone charts (bar, pie, histogram, heatmap) that return embeddable HTML figures.

---

## How to Get Started

### Step 1 — Install dependencies
```python
!pip install plotly scikit-learn scipy pydantic networkx graphviz --quiet
```

### Step 2 — Paste the class code
Copy and paste both the `DataInspector` and `PlottingMethods` class definitions into a Colab cell and run it.

### Step 3 — Create an instance
```python
inspector = DataInspector()
```

You now have a fresh inspector object ready to load and process any CSV dataset.

---

## The Full Workflow

Follow this order for best results. Each step builds on the previous one.

```
Load Data → Inspect Structure → Clean → Normalize → Visualize → Export
```

---

## Category 1 — Data Ingestion

> **Goal:** Load your CSV file into the inspector so all other methods can work on it.

---

### `upload_data()`

**What it does:**
Opens a file picker dialog in Google Colab so you can upload a CSV file directly from your computer. After uploading, it automatically:
- Converts common garbage strings (`"?"`, `"n/a"`, `"NULL"`, `""`, `" "`) into proper `NaN` (missing) values
- Attempts to convert any text columns that are actually numbers into numeric type
- Separates columns into numeric and categorical groups internally

**When to use it:**
When you have a CSV file on your local machine and want to upload it into Colab.

```python
inspector.upload_data()
```

**What you will see:**
```
✅ Loaded 'your_file.csv': 891 rows × 12 columns
🔄 Auto-converted to numeric: ['Age', 'Fare']
⏭️  Kept as categorical: ['Name', 'Sex', 'Embarked']
```

---

### `load_from_path(path)`

**What it does:**
Same as `upload_data()` but loads a CSV that is already on the Colab server's filesystem — for example a sample dataset or a file you previously uploaded.

**When to use it:**
When your file is already accessible via a file path, not on your local machine.

```python
inspector.load_from_path('/content/your_file.csv')
```

| Parameter | Type | Description |
|-----------|------|-------------|
| `path` | `str` | The full file path to your CSV file |

---

## Category 2 — Structural Analysis

> **Goal:** Understand what your dataset looks like before touching it. Always run these first.

---

### `get_summary()`

**What it does:**
Prints a quick overview of your dataset — how many rows and columns it has, which columns are numeric vs categorical, and shows you the first 20 rows so you can see the raw data.

**When to use it:**
Immediately after loading data. This is your first look at the dataset.

```python
inspector.get_summary()
```

**What you will see:**
```
--- Data Summary ---
Rows: 891 | Columns: 12
Numerical (5): ['PassengerId', 'Survived', 'Pclass', 'Age', 'Fare']
Categorical (7): ['Name', 'Sex', 'Ticket', 'Cabin', 'Embarked']
[first 20 rows displayed as a table]
```

---

### `column_details()`

**What it does:**
Goes through every column and prints either:
- The **min and max value** for numeric columns
- The **number of unique values** for categorical columns

**When to use it:**
After `get_summary()` to get a deeper look at the value ranges and cardinality of each column.

```python
inspector.column_details()
```

**What you will see:**
```
🔹 Age (Numeric): Range [0.42 to 80.0]
🔹 Fare (Numeric): Range [0.0 to 512.33]
🔸 Sex (Categorical): 2 unique values
🔸 Embarked (Categorical): 3 unique values
```

---

### `get_categorical_summary()`

**What it does:**
For each categorical column, shows:
- How many unique values it has
- What the most frequent value (mode) is
- How many times that mode appears

**When to use it:**
When you want to understand the distribution of your text columns at a glance.

```python
inspector.get_categorical_summary()
```

**What you will see:**
```
--- Categorical Deep Dive ---
          unique    top   freq
Sex            2   male    577
Embarked       3      S    644
```

---

### `show_missing_data()`

**What it does:**
Filters and displays only the rows in your dataset that contain at least one missing value. If there are no missing values, it tells you so.

**When to use it:**
Before cleaning, to see exactly which rows are affected by missing data.

```python
inspector.show_missing_data()
```

**What you will see:**
```
Found 177 rows with missing values:
[table of rows that contain NaN]
```

> **Tip:** Run this again after imputation to confirm all missing values have been handled.

---

## Category 3 — Data Cleaning

> **Goal:** Fix problems in your data — missing values, duplicates, outliers, and unwanted rows or columns.

---

### `handle_missing_values(columns, strategy, fill_value)`

**What it does:**
Fills in missing (`NaN`) values in your dataset using a strategy you choose. Instead of deleting rows with missing data (which loses information), it replaces the gaps with calculated or fixed values.

**When to use it:**
After `show_missing_data()` confirms there are missing values that need to be filled.

```python
# fill numeric columns with their median
inspector.handle_missing_values(strategy='median')

# fill categorical columns with their most common value
inspector.handle_missing_values(strategy='mode')

# fill a specific column with a constant
inspector.handle_missing_values(columns=['Age'], strategy='constant', fill_value=0)
```

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `columns` | `list` | `None` | List of column names to impute. If `None`, applies to all columns with NaNs |
| `strategy` | `str` | `'median'` | How to fill gaps: `'mean'`, `'median'`, `'mode'`, or `'constant'` |
| `fill_value` | `any` | `None` | The value to use when `strategy='constant'` |

**Strategy guide:**

| Strategy | Best for | Works on |
|----------|----------|----------|
| `median` | Numeric columns with outliers | Numeric only |
| `mean` | Normally distributed numeric columns | Numeric only |
| `mode` | Categorical columns or any column | Both types |
| `constant` | When you know exactly what to fill | Both types |

> **Important:** Calling `strategy='mean'` or `strategy='median'` on a categorical column silently does nothing. Use `strategy='mode'` for categorical columns.

---

### `remove_duplicates()`

**What it does:**
Scans the entire dataset for rows where every single column value is identical to another row. If found, it removes the extra copies and resets the row index so there are no gaps.

**When to use it:**
Early in your cleaning pipeline to ensure the same data point is not counted multiple times, which would bias your analysis.

```python
inspector.remove_duplicates()
```

**What you will see:**
```
Removed 3 duplicate rows. New row count: 888
```

> **Note:** Only removes exact duplicates — every column must match. A row with one different value will not be flagged.

---

### `handle_outliers(columns, find_and_delete)`

**What it does:**
Detects extreme values in numeric columns using the IQR (Interquartile Range) method. Any value that sits more than 1.5× the IQR below Q1 or above Q3 is flagged as an outlier.

**How IQR works:**
```
Q1 = 25th percentile of the column
Q3 = 75th percentile of the column
IQR = Q3 - Q1

Lower bound = Q1 - (1.5 × IQR)
Upper bound = Q3 + (1.5 × IQR)

Any value outside these bounds = outlier
```

**When to use it:**
After handling missing values. Run it first without deletion to review, then decide whether to delete.

```python
# step 1 — review outliers without deleting anything
inspector.handle_outliers(find_and_delete=False)

# step 2 — delete after reviewing
inspector.handle_outliers(find_and_delete=True)

# target specific columns only
inspector.handle_outliers(columns=['Fare', 'Age'], find_and_delete=True)
```

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `columns` | `list` | `None` | Columns to check. If `None`, checks all numeric columns |
| `find_and_delete` | `bool` | `False` | If `True`, removes outlier rows permanently |

**What you will see:**
```
🚨 Fare: Found 17 outliers.
🚨 Age: Found 10 outliers.
[table of all flagged rows displayed]
🗑️ Deleted 27 outlier rows.
```

---

### `delete_rows()`

**What it does:**
Prompts you to type a comma-separated list of row index numbers. Those rows are then permanently removed from the dataset and the index is reset.

**When to use it:**
When you have identified specific rows that need to be manually removed — for example, test entries, corrupted records, or rows you reviewed and decided to exclude.

```python
inspector.delete_rows()
# prompted: Enter row indices to delete (e.g., 1, 3, 15):
# type: 5, 23, 100
```

**What you will see:**
```
Deleted 3 rows. New count: 888
```

> **Tip:** Check `get_summary()` first to see current row indices before deciding which to delete.

---

### `delete_columns()`

**What it does:**
Prints the current list of column names, then prompts you to type which ones to remove. Those columns are permanently dropped from the dataset.

**When to use it:**
When certain columns are irrelevant, have too many missing values, or contain identifiers (like IDs or names) that don't contribute to analysis.

```python
inspector.delete_columns()
# prompted: Enter column names to delete (e.g., Column1, Column2):
# type: Cabin, Ticket, Name
```

**What you will see:**
```
Current columns: PassengerId, Survived, Pclass, Name, Sex, Age, Fare, Cabin
Deleted 3 columns. Remaining count: 5
```

---

## Category 4 — Feature Engineering & Normalization

> **Goal:** Convert your cleaned data into a numeric format that machine learning models and statistical methods can process.

---

### `extract_numeric_data()`

**What it does:**
Pulls out only the numeric columns from your dataset and returns them as a separate DataFrame. Does not modify the original data.

**When to use it:**
When you need to work with just the numbers — for example before feeding data into a model that only accepts numeric input.

```python
numeric_df = inspector.extract_numeric_data()
display(numeric_df.head())
```

---

### `extract_categorical_data()`

**What it does:**
The opposite of `extract_numeric_data()` — pulls out only the text/categorical columns and returns them as a separate DataFrame.

**When to use it:**
When you want to inspect or process categorical columns separately.

```python
categorical_df = inspector.extract_categorical_data()
display(categorical_df.head())
```

---

### `extract_normalized_numeric_data(method)`

**What it does:**
Takes all numeric columns and scales their values so they are comparable to each other. Without scaling, a column with values in the thousands (like `Fare`) would dominate a column with values 0-1 (like `Survived`) in most algorithms.

If any NaN values remain, they are automatically filled with the column median before scaling.

**When to use it:**
Before feeding data into machine learning models, PCA, or clustering.

```python
# scales everything to [0, 1]
numeric_normalized = inspector.extract_normalized_numeric_data(method='minmax')

# centers around mean=0, std=1
numeric_normalized = inspector.extract_normalized_numeric_data(method='standard')

# uses median and IQR — robust to outliers
numeric_normalized = inspector.extract_normalized_numeric_data(method='robust')

display(numeric_normalized.head())
```

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `method` | `str` | `'minmax'` | Scaling method: `'minmax'`, `'standard'`, or `'robust'` |

**Method comparison:**

| Method | Formula | Output range | Best when |
|--------|---------|-------------|-----------|
| `minmax` | `(x - min) / (max - min)` | `[0, 1]` | Neural networks, bounded range needed |
| `standard` | `(x - mean) / std` | Unbounded | PCA, clustering, linear models |
| `robust` | `(x - median) / IQR` | Unbounded | Dataset has outliers you kept |

---

### `extract_normalized_categorical_data(method)`

**What it does:**
Converts text categories into numbers so they can be used in mathematical operations. Different encoding methods make different assumptions about your data.

**When to use it:**
Before any analysis or modelling that requires purely numeric input.

```python
# binary column per category (recommended for unordered categories)
categorical_encoded = inspector.extract_normalized_categorical_data(method='onehot')

# one integer per category
categorical_encoded = inspector.extract_normalized_categorical_data(method='ordinal')

# category codes scaled to [0, 1]
categorical_encoded = inspector.extract_normalized_categorical_data(method='uniform')

# ordinal then scaled to [0, 1]
categorical_encoded = inspector.extract_normalized_categorical_data(method='minmax_ordinal')

display(categorical_encoded.head())
```

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `method` | `str` | `'uniform'` | Encoding method: `'onehot'`, `'ordinal'`, `'uniform'`, or `'minmax_ordinal'` |

**Method comparison:**

| Method | Example input | Example output | Best when |
|--------|--------------|---------------|-----------|
| `onehot` | `["S", "C", "Q"]` | 3 new binary columns | Categories have no natural order |
| `ordinal` | `["S", "C", "Q"]` | `[2, 0, 1]` | Categories have a natural order |
| `uniform` | `["S", "C", "Q"]` | `[1.0, 0.0, 0.5]` | Quick encoding, bounded range needed |
| `minmax_ordinal` | `["S", "C", "Q"]` | `[1.0, 0.0, 0.5]` | Ordinal with bounded scaling |

> **Note:** Missing values are automatically filled with the string `"Missing"` before encoding so the encoder does not crash.

---

### `create_normalized_data_df()`

**What it does:**
Combines raw numeric columns with encoded categorical columns into one unified DataFrame. This gives you a single clean table ready for modelling.

**When to use it:**
After you have decided on your scaling and encoding methods. This is the final preparation step before feeding data into a model.

```python
full_normalized = inspector.create_normalized_data_df()
display(full_normalized.head())
print(f"Shape: {full_normalized.shape}")
```

**What the result looks like:**
```
Before:                              After:
| Age | Fare | Sex    | Embarked |   | Age | Fare | Sex_encoded | Embarked_encoded |
|-----|------|--------|----------|   |-----|------|-------------|------------------|
| 22  | 7.25 | male   | S        |   | 22  | 7.25 | 1.0         | 1.0              |
| 38  | 71.3 | female | C        |   | 38  | 71.3 | 0.0         | 0.0              |
```

> **Note:** Numeric columns are kept as-is (not scaled). Only categorical columns are encoded. Scale numeric columns separately using `extract_normalized_numeric_data()` if needed.

---

## Category 5 — Visualization

> **Goal:** See your data visually to understand distributions, frequencies, and relationships between columns.

---

### `plot_numerical(column_names)`

**What it does:**
For each numeric column you specify, generates a 3-panel interactive chart:
- **Panel 1 — Horizontal Violin/Box:** Shows the shape of the distribution, median, quartiles, and mean line
- **Panel 2 — Scatter:** Plots each row's index vs its value — useful to spot trends or drift across rows
- **Panel 3 — Histogram:** Shows frequency counts across value bins

**When to use it:**
To understand the distribution of your numeric columns — whether they are skewed, have outliers, or are normally distributed.

```python
# single column
inspector.plot_numerical('Age')

# multiple columns
inspector.plot_numerical(['Age', 'Fare', 'Pclass'])

# all numeric columns automatically
numeric_cols = inspector.df.select_dtypes(include='number').columns.tolist()
inspector.plot_numerical(numeric_cols)
```

| Parameter | Type | Description |
|-----------|------|-------------|
| `column_names` | `str` or `list` | One column name or a list of column names |

---

### `plot_categorical(column_names)`

**What it does:**
For each categorical column you specify, generates a bar chart showing how many times each unique category appears, with the percentage of total displayed on top of each bar.

**When to use it:**
To see how your categorical data is distributed — for example, whether your dataset is balanced between classes.

```python
# single column
inspector.plot_categorical('Sex')

# multiple columns
inspector.plot_categorical(['Sex', 'Embarked', 'Pclass'])

# all categorical columns automatically
cat_cols = inspector.df.select_dtypes(exclude='number').columns.tolist()
inspector.plot_categorical(cat_cols)
```

| Parameter | Type | Description |
|-----------|------|-------------|
| `column_names` | `str` or `list` | One column name or a list of column names |

---

### `plot_relationship(col1, col2)`

**What it does:**
Automatically detects the types of the two columns you give it and picks the most appropriate chart:

| col1 type | col2 type | Chart chosen |
|-----------|-----------|-------------|
| Numeric | Numeric | Scatter plot with OLS trendline |
| Categorical | Numeric | Box plot with all data points shown |
| Categorical | Categorical | Grouped bar chart |

**When to use it:**
When you want to explore whether two columns have a relationship — for example, does `Sex` affect `Fare`? Does `Age` correlate with `Fare`?

```python
# numeric vs numeric — scatter with trendline
inspector.plot_relationship('Age', 'Fare')

# categorical vs numeric — box plot
inspector.plot_relationship('Sex', 'Age')

# categorical vs categorical — grouped bar
inspector.plot_relationship('Sex', 'Embarked')
```

| Parameter | Type | Description |
|-----------|------|-------------|
| `col1` | `str` | First column name |
| `col2` | `str` | Second column name |

---

## Category 6 — Correlation & Association

> **Goal:** Measure the strength of relationships between all pairs of columns in your dataset using the correct statistical method for each data type combination.

---

### `plot_numerical_correlation()`

**What it does:**
Computes the Pearson correlation coefficient between every pair of numeric columns and displays the result as an interactive heatmap. Values range from -1 to +1.

```
+1.0  → as one column increases, the other increases perfectly
 0.0  → no linear relationship
-1.0  → as one column increases, the other decreases perfectly
```

**When to use it:**
To identify which numeric features are strongly related to each other — useful for feature selection before modelling.

```python
inspector.plot_numerical_correlation()
```

**How to read the heatmap:**
- Deep red cells = strong positive correlation
- White cells = no correlation
- Deep blue cells = strong negative correlation
- Diagonal is always 1.0 (a column is perfectly correlated with itself)

---

### `plot_categorical_correlation()`

**What it does:**
Computes Cramér's V between every pair of categorical columns and displays it as a heatmap. Cramér's V is the categorical equivalent of Pearson correlation — it measures how strongly two categories are associated.

Values range from 0 to 1 (no direction, only strength):
```
0.0 → no association
1.0 → perfect association
```

**When to use it:**
To find relationships between text columns — for example, does `Sex` associate with `Embarked`?

```python
inspector.plot_categorical_correlation()
```

**How it works internally:**
1. Builds a crosstab (frequency table) for each pair of categories
2. Runs a Chi-squared test on the crosstab
3. Normalizes the Chi-squared value to a 0-1 scale using the Cramér's V formula

---

### `correlate_num_to_cat()`

**What it does:**
Computes the association between every numeric column and every categorical column. Returns a table (DataFrame) of results — not a heatmap. Uses two different statistical tests depending on how many categories exist:

- **Binary category (2 values)** → Point-Biserial correlation (range: -1 to +1, direction matters)
- **Multi-class category (3+ values)** → Eta via ANOVA (range: 0 to 1, strength only)

Also computes a p-value for each pair to tell you whether the association is statistically significant.

**When to use it:**
When you want to know which numeric features are most influenced by categorical groupings — for example, does `Sex` significantly affect `Fare`?

```python
assoc_table = inspector.correlate_num_to_cat()
display(assoc_table)
```

**What the output looks like:**
```
Categorical | Numerical | Type                  | Correlation | P-Value
Sex         | Age       | Point-Biserial        | -0.081      | 0.0385
Embarked    | Fare      | Eta (Multi-class ANOVA)| 0.234      | 0.0001
```

> **Reading the p-value:** A p-value below 0.05 means the association is statistically significant and unlikely to be random chance.

---

### `plot_all_associations_heatmap()`

**What it does:**
The most powerful visualization in the class. Creates a single unified heatmap covering every column pair in your entire dataset — regardless of whether the columns are numeric, categorical, or mixed. It automatically selects the right association measure for each pair:

| Pair type | Measure used |
|-----------|-------------|
| Numeric vs Numeric | Pearson r (absolute value) |
| Categorical vs Categorical | Cramér's V |
| Numeric vs Categorical | Eta (ANOVA-based) |

All values are on a 0 to 1 scale (strength only) so they are visually comparable across the whole matrix.

**When to use it:**
As a final exploratory step to get a complete picture of all relationships in your dataset at once.

```python
inspector.plot_all_associations_heatmap()
```

**How to read it:**
- Bright/light cells = strong association between those two columns
- Dark cells = weak or no association
- Diagonal is always 1.0

> **Tip:** High association between a feature and your target column (e.g. `Survived`) is a good signal that feature will be useful in a model.

---

## Category 7 — Custom Plotting (PlottingMethods)

> **Goal:** Generate standalone charts from any data you provide, formatted as HTML figures that can be embedded anywhere.

All `PlottingMethods` functions accept data as a JSON string in this format:
```python
import json
data = json.dumps({"records": [
    {"col1": value1, "col2": value2},
    ...
]})
```

And return a result dictionary. Use `PLT.display_image(result)` to render the chart inline in Colab.

---

### Setup

```python
PLT = PlottingMethods()
```

---

### `get_methods_info()`

**What it does:**
Returns a list of all available plotting methods with their parameter signatures and descriptions. Useful as a quick reference without reading the full documentation.

```python
response = PLT.get_methods_info()
for method in response['response']:
    print(f"Method    : {method['method']}")
    print(f"Signature : {method['signature']}")
    print(f"Description: {method['description']}")
    print("-" * 60)
```

---

### `plot_bar_chart(...)`

**What it does:**
Creates a vertical bar chart. Supports stacked or grouped mode, optional color grouping, and hover data.

```python
data = json.dumps({"records": [
    {"category": "A", "value": 40},
    {"category": "B", "value": 60},
    {"category": "C", "value": 25}
]})

result = PLT.plot_bar_chart(
    x='category',
    y='value',
    title='My Bar Chart',
    barmode='group',     # or 'stack'
    data=data
)
PLT.display_image(result)
```

| Parameter | Description |
|-----------|-------------|
| `x` | Column name for the x-axis |
| `y` | Column name for the y-axis |
| `color` | Column name to split bars by color |
| `barmode` | `'stack'` or `'group'` |
| `title` | Chart title |

---

### `plot_pie_chart(...)`

**What it does:**
Creates a pie chart. Pass `hole=0.4` to make it a donut chart.

```python
result = PLT.plot_pie_chart(
    names='category',
    values='value',
    title='My Pie Chart',
    hole=0.4,            # remove this line for a full pie
    data=data
)
PLT.display_image(result)
```

| Parameter | Description |
|-----------|-------------|
| `names` | Column name for segment labels |
| `values` | Column name for segment sizes |
| `hole` | Float 0-1 for donut hole size. `None` for full pie |
| `title` | Chart title |

---

### `plot_histogram(...)`

**What it does:**
Creates a histogram showing how frequently values fall into each bin. Optionally accepts custom bin boundaries.

```python
data_hist = json.dumps({"records": [{"value": v} for v in [10, 20, 20, 30, 40, 40, 40, 50]]})

result = PLT.plot_histogram(
    x='value',
    title='My Histogram',
    data=data_hist
)
PLT.display_image(result)
```

| Parameter | Description |
|-----------|-------------|
| `x` | Column name to plot |
| `title` | Chart title |
| `bins` | Optional list of bin boundaries |

---

### `plot_heat_map(...)`

**What it does:**
Creates a heatmap from a pivot table of your data. Useful for showing how a value changes across two categorical dimensions — for example grades across students and subjects.

```python
data_heat = json.dumps({"records": [
    {"student": "Alice", "subject": "Math",    "grade": 85},
    {"student": "Alice", "subject": "Science", "grade": 90},
    {"student": "Bob",   "subject": "Math",    "grade": 70},
    {"student": "Bob",   "subject": "Science", "grade": 75}
]})

result = PLT.plot_heat_map(
    values='grade',
    index='student',
    columns='subject',
    aggregade_method='mean',
    title='Average Grades',
    data=data_heat
)
PLT.display_image(result)
```

| Parameter | Description |
|-----------|-------------|
| `values` | Column whose values fill the heatmap cells |
| `index` | Column for the y-axis (rows of pivot table) |
| `columns` | Column for the x-axis (columns of pivot table) |
| `aggregade_method` | How to aggregate: `'sum'`, `'mean'`, `'count'` etc. |
| `fill_value` | Value for empty cells (default `0`) |
| `title` | Chart title |

---

### `plot_multi_column_bar_graph(...)`

**What it does:**
Creates a grouped or stacked bar chart comparing multiple numeric columns side by side for each group label.

```python
data_multi = json.dumps({"records": [
    {"group": "Q1", "revenue": 100, "cost": 60},
    {"group": "Q2", "revenue": 130, "cost": 70},
    {"group": "Q3", "revenue": 120, "cost": 65}
]})

result = PLT.plot_multi_column_bar_graph(
    xLabel='group',
    value_vars=['revenue', 'cost'],
    title='Revenue vs Cost by Quarter',
    barmode='group',
    data=data_multi
)
PLT.display_image(result)
```

| Parameter | Description |
|-----------|-------------|
| `xLabel` | Column name for the x-axis groups |
| `value_vars` | List of column names to compare |
| `barmode` | `'group'` or `'stack'` |
| `title` | Chart title |

---

## Category 8 — Export

> **Goal:** Save and download your cleaned dataset after all processing is complete.

---

### `export_cleaned_data(filename)`

**What it does:**
Saves the current state of your DataFrame as a CSV file on the Colab server and immediately triggers a browser download so the file saves to your local machine.

The exported file reflects everything you have done — imputation, duplicate removal, outlier deletion, and column drops — captured at the exact moment you call this function.

**When to use it:**
At the very end of your pipeline, after all cleaning and processing is complete.

```python
inspector.export_cleaned_data('cleaned_data.csv')
```

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `filename` | `str` | `'cleaned_data.csv'` | Name of the file to save and download |

**What you will see:**
```
💾 'cleaned_data.csv' has been generated and download triggered.
```

> **Tip:** Run `get_summary()` one final time before exporting to confirm your dataset looks correct.

---

## Quick Reference Table

| Function | Category | What it does in one line |
|----------|----------|--------------------------|
| `upload_data()` | Ingestion | Upload CSV from local machine |
| `load_from_path(path)` | Ingestion | Load CSV from Colab filesystem |
| `get_summary()` | Analysis | Row/column counts + first 20 rows |
| `column_details()` | Analysis | Range for numeric, unique count for categorical |
| `get_categorical_summary()` | Analysis | Mode and frequency for categorical columns |
| `show_missing_data()` | Analysis | Display all rows with at least one NaN |
| `handle_missing_values()` | Cleaning | Fill NaNs with mean/median/mode/constant |
| `remove_duplicates()` | Cleaning | Drop exact duplicate rows |
| `handle_outliers()` | Cleaning | Flag or delete IQR-based outliers |
| `delete_rows()` | Cleaning | Manually remove rows by index |
| `delete_columns()` | Cleaning | Manually remove columns by name |
| `extract_numeric_data()` | Normalization | Return numeric columns only |
| `extract_categorical_data()` | Normalization | Return categorical columns only |
| `extract_normalized_numeric_data()` | Normalization | Scale numeric columns |
| `extract_normalized_categorical_data()` | Normalization | Encode categorical columns |
| `create_normalized_data_df()` | Normalization | Merge numeric + encoded categorical |
| `plot_numerical()` | Visualization | Violin + scatter + histogram per column |
| `plot_categorical()` | Visualization | Frequency bar chart per column |
| `plot_relationship()` | Visualization | Auto-select best chart for two columns |
| `plot_numerical_correlation()` | Correlation | Pearson heatmap for numeric columns |
| `plot_categorical_correlation()` | Correlation | Cramér's V heatmap for categorical columns |
| `correlate_num_to_cat()` | Correlation | Point-Biserial / Eta association table |
| `plot_all_associations_heatmap()` | Correlation | Unified heatmap for all column types |
| `get_methods_info()` | Plotting | List all PlottingMethods with signatures |
| `plot_bar_chart()` | Plotting | Grouped or stacked bar chart |
| `plot_pie_chart()` | Plotting | Pie or donut chart |
| `plot_histogram()` | Plotting | Frequency histogram |
| `plot_heat_map()` | Plotting | Pivot-based heatmap |
| `plot_multi_column_bar_graph()` | Plotting | Multi-metric grouped bar chart |
| `export_cleaned_data()` | Export | Save and download cleaned CSV |

---

*Built for Google Colab. Requires: pandas, numpy, plotly, scikit-learn, scipy.*
