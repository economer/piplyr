
# piplyr

piplyr is a Python package that provides dplyr-like data manipulation capabilities for pandas DataFrames. It simplifies data manipulation tasks in Python by offering a set of intuitive methods for data filtering, selection, transformation, and aggregation.

## Installation

You can install `piplyr` directly from PyPI:

```bash
pip install piplyr
```

## Usage

Here are some basic examples of how to use `piplyr`:

### Initializing piplyr

```python
import pandas as pd
from piplyr import piplyr

# Sample DataFrame
df = pd.DataFrame({'A': [1, 2, 3], 'B': [4, 5, 6]})

# Initialize piplyr with the DataFrame
pi = piplyr(df)
```

### Grouping Data

```python
# Group by a column
grouped_pi = pi.group_by('A')
```

### Sorting Data

```python
# Sort the DataFrame by a column

sorted_pi = pi.sort_by('B').to_df
```

### to_df

Use to_df to convert your generated data to a Pandas' DataFrame

### Selecting Columns

```python
# Select specific columns
selected_pi = pi.select('A', 'B')
```

### Dropping Columns

```python
# Drop specific columns
dropped_pi = pi.drop_col('B')
```

### Renaming Columns

```python
# Rename columns in the DataFrame
renamed_pi = pi.rename_col({'A': 'new_A'})
```

### Filtering Rows

```python
# Filter rows based on a condition
filtered_pi = pi.filter_row('A > 1')
```

### Mutating Columns

```python
# Add a new column or modify existing ones
mutated_pi = pi.mutate(new_col=lambda x: x['A'] * 2)
```

### Summarizing Data

```python
# Summarize the DataFrame
summarized_pi = pi.summarize(mean_A=lambda x: x['A'].mean())
```

### Executing SQL Queries

```python
# Execute an SQL query on the DataFrame. You have to use from df regardless of your DataFrame name
sql_pi = pi.sql_plyr('SELECT * FROM df WHERE A > 1')
```

### Chaining Methods

```python
# Chain multiple operations
result_pi = pi.select('A', 'B').filter_row('A > 1').summarize(avg_B=lambda x: x['B'].mean())
```

### Additional Methods

The package also includes several other methods like `join`, `count_na`, `distinct`, `pivot_longer`, `pivot_wider`, `clean_names`, `separate`, `str_pad`, `str_sub`, `str_extract`, `str_detect`, `str_len`, `str_lower`, `str_upper`, `str_startswith`, `str_endswith`, `str_contains`, `fct_lump`, `fct_infreq`, `fct_relevel`, `fct_recode`, `fct_reorder` and others. 

Each of these methods provides specific data manipulation functionalities and can be explored further in the package documentation.

## More Examples

Please consult docstrings of various methods that include explnations and examples. 

## Contributing

Contributions to `piplyr` are welcome! Please refer to the [contribution guidelines](https://github.com/YourGitHubUsername/piplyr/CONTRIBUTING.md) for more information.

## License

This project is licensed under the [MIT License](https://github.com/YourGitHubUsername/piplyr/LICENSE).
