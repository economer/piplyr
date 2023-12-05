import pandas as pd
import numpy as np
import sqlite3
import re
import warnings

class piplyr:
    """
    A class providing dplyr-like data manipulation capabilities for pandas DataFrames.
    """

    def __init__(self, df):
        """
        Initializes the piplyr class with a pandas DataFrame.
        
        Args:
            df (pd.DataFrame): A pandas DataFrame to be manipulated.
            Examples:
            >>> df = pd.DataFrame({'A': [1, 2, 3], 'B': [4, 5, 6]})
            >>> pi = piplyr(df)
        """
        if not isinstance(df, pd.DataFrame):
            raise TypeError("Input must be a pandas DataFrame")
        self.df = df.copy()
        self.grouped = None

    def group_by(self, *group_vars):
        """
        Groups the DataFrame by specified columns.

        Args:
            group_vars: Columns to group by. Multiple columns can be specified.

        Returns:
            self: The piplyr object with the DataFrame grouped.
            Examples:
            >>> df = pd.DataFrame({'A': [1, 2, 1], 'B': [5, 6, 7]})
            >>> pi = piplyr(df).group_by('A')
        """
        if not all(col in self.df.columns for col in group_vars):
            raise ValueError("One or more grouping columns not found in DataFrame")
        self.grouped = self.df.groupby(list(group_vars))
        return self

    def sort_by(self, column, ascending=True):
        """
        Sorts the DataFrame by a specified column.

        Args:
            column: The column to sort by.
            ascending: Whether to sort in ascending order (default is True).

        Returns:
            self: The piplyr object with the DataFrame sorted.
             Examples:
            >>> df = pd.DataFrame({'A': [3, 1, 2]})
            >>> pi = piplyr(df).sort_by('A')
        """
        if column not in self.df.columns:
            raise ValueError(f"Column {column} not found in DataFrame")
        self.df = self.df.sort_values(by=column, ascending=ascending)
        return self

    def select(self, *columns):
        """
        Selects specified columns from the DataFrame.

        Args:
            columns: A list of column names to keep in the DataFrame.

        Returns:
            self: The modified piplyr object with only the selected columns.
            Examples:
            >>> df = pd.DataFrame({'A': [1, 2, 3], 'B': [4, 5, 6], 'C': [7, 8, 9]})
            >>> pi = piplyr(df).select('A', 'B')
        """
        missing_cols = [col for col in columns if col not in self.df.columns]
        if missing_cols:
            raise ValueError(f"Columns {missing_cols} not found in DataFrame")
        self.df = self.df[list(columns)]
        return self

    def drop_col(self, *columns):
        """
        Drops specified columns from the DataFrame.

        Args:
            columns: A list of column names to drop from the DataFrame.

        Returns:
            self: The modified piplyr object with specified columns removed.
            Examples:
            >>> df = pd.DataFrame({'A': [1, 2, 3], 'B': [4, 5, 6], 'C': [7, 8, 9]})
            >>> pi = piplyr(df).drop_col('C')
        """
        missing_cols = [col for col in columns if col not in self.df.columns]
        if missing_cols:
            raise ValueError(f"Columns {missing_cols} not found in DataFrame")
        self.df = self.df.drop(columns=list(columns))
        return self

    def rename_col(self, rename_dict):
        """
        Renames columns in the DataFrame.

        Args:
            rename_dict: A dictionary mapping old column names to new column names.

        Returns:
            self: The modified piplyr object with columns renamed.
            Examples:
            >>> df = pd.DataFrame({'A': [1, 2, 3]})
            >>> pi = piplyr(df).rename_col({'A': 'new_A'})
        """
        self.df = self.df.rename(columns=rename_dict)
        return self

    def filter_row(self, condition):
        """
        Filters rows based on a given condition.

        Args:
            condition: A string of condition to filter the DataFrame rows. 
                       The condition should be in a format that can be used inside DataFrame.query().

        Returns:
            self: The modified piplyr object with rows filtered based on the condition.
            Examples:
            >>> df = pd.DataFrame({'A': [1, 2, 3], 'B': [4, 5, 6]})
            >>> pi = piplyr(df).filter_row('A > 1')
        """
        self.df = self.df.query(condition)
        return self

    def mutate(self, **kwargs):
        """
        Adds new columns or modifies existing ones based on given expressions.
        When used after 'group_by', applies the mutation within each group.

        Args:
            kwargs: Key-value pairs where keys are new or existing column names 
                    and values are expressions or functions to compute their values.

        Returns:
            self: The modified piplyr object with new or modified columns.

        Examples:
            Without grouping:
            >>> df = pd.DataFrame({'A': [1, 2, 3], 'B': [4, 5, 6]})
            >>> pi = piplyr(df).mutate(new_col=lambda x: x['A'] * 2)

            With grouping:
            >>> df = pd.DataFrame({'A': [1, 2, 3], 'B': [4, 5, 6], 'C': ['X', 'Y', 'X']})
            >>> pi = piplyr(df).group_by('C').mutate(mean_val=lambda x: np.mean(x['A']))
        """
        if self.grouped:
            for new_col, func in kwargs.items():
                if callable(func):
                    # Apply the function to each group and assign the result
                    self.df[new_col] = self.grouped.apply(lambda x: func(x)).reset_index(level=0, drop=True)
                else:
                    raise ValueError("With grouping, provide a callable function for mutation.")
        else:
            for new_col, expression in kwargs.items():
                if callable(expression):
                    self.df[new_col] = expression(self.df)
                else:
                    raise ValueError("Mutation expressions must be callable.")

        return self



    def summarize(self, **kwargs):
        """
        Performs summary/aggregation operations on the DataFrame.
        If used after 'group_by', provides aggregated statistics for each group.
        Without 'group_by', provides aggregated statistics for the entire DataFrame.

        Args:
            kwargs: Key-value pairs where keys are new column names for the aggregated values and
                    values are aggregation functions.

        Returns:
            self: The modified piplyr object with a DataFrame containing summary statistics.

        Examples:
            Without grouping:
            >>> df = pd.DataFrame({'A': [1, 2, 3], 'B': [4, 5, 6]})
            >>> pi = piplyr(df).summarize(mean_A=np.mean('A'), sum_B=np.sum('B'))

            With grouping:
            >>> df = pd.DataFrame({'A': [1, 2, 3], 'B': [4, 5, 6], 'C': ['X', 'Y', 'X']})
            >>> pi = piplyr(df).group_by('C').summarize(mean_val=lambda x: np.mean(x['A']))
            >>> print(pi.to_df)
        """
        if self.grouped:
            def agg_func(group):
                result = {}
                for new_col, func in kwargs.items():
                    if callable(func):
                        result[new_col] = func(group)
                    else:
                        raise ValueError("Aggregation function must be callable")
                return pd.Series(result)

            self.df = self.grouped.apply(agg_func).reset_index()
        else:
            result = {}
            for new_col, func in kwargs.items():
                if callable(func):
                    result[new_col] = func(self.df)
                else:
                    raise ValueError("Aggregation function must be callable")
            self.df = pd.DataFrame(result, index=[0])

        return self
    
    def agg_funcs(self, **kwargs):
        """
        Apply multiple aggregation functions to columns.

        Args:
            kwargs: Key-value pairs where the key is the column name and the value is a list of aggregation functions.

        Returns:
            self: The modified piplyr object with aggregated DataFrame.

        Examples:
            >>> df = pd.DataFrame({'A': [1, 2, 3], 'B': [4, 5, 6]})
            >>> pi = piplyr(df).agg_funcs(A=['sum', 'mean'], B=['min', 'max'])
        """
        agg_dict = {col: funcs for col, funcs in kwargs.items()}
        self.df = self.df.agg(agg_dict)
        return self
    
    def rowwise(self, func, *args, **kwargs):
        """
        Apply a function row-wise to the DataFrame.

        Args:
            func: A function to apply to each row.
            *args, **kwargs: Additional arguments and keyword arguments for the function.

        Returns:
            self: The modified piplyr object.

        Examples:
            >>> df = pd.DataFrame({'A': [1, 2], 'B': [3, 4]})
            >>> func = lambda row: row['A'] + row['B']
            >>> pi = piplyr(df).rowwise(func)
        """
        self.df = self.df.apply(lambda row: func(row, *args, **kwargs), axis=1)
        return self
    
    
    def sample_n(self, n):
        """
        Randomly sample n rows from the DataFrame.

        Args:
            n: Number of rows to sample.

        Returns:
            self: The modified piplyr object.

        Examples:
            >>> df = pd.DataFrame({'A': range(10), 'B': range(10, 20)})
            >>> pi = piplyr(df).sample_n(5)
        """
        self.df = self.df.sample(n=n)
        return self
    
    def sample_frac(self, frac):
        """
        Randomly sample a fraction of rows from the DataFrame.

        Args:
            frac: Fraction of rows to sample.

        Returns:
            self: The modified piplyr object.

        Examples:
            >>> df = pd.DataFrame({'A': range(10), 'B': range(10, 20)})
            >>> pi = piplyr(df).sample_frac(0.5)
        """
        self.df = self.df.sample(frac=frac)
        return self
    
    def mutate_conditional(self, **kwargs):
        """
        Apply conditional mutations to columns.

        Args:
            kwargs: Key-value pairs where keys are column names and values are tuples of (condition, value).

        Returns:
            self: The modified piplyr object.

        Examples:
            >>> df = pd.DataFrame({'A': [10, 20], 'B': [30, 40]})
            >>> pi = piplyr(df).mutate_conditional(A=('A > 15', 100), B=('B < 35', 200))
        """
        for col, (condition, value) in kwargs.items():
            self.df.loc[self.df.eval(condition), col] = value
        return self
        
    def bin_data(self, column, bins):
        """
        Bin numeric data into categories.

        Args:
            column: The column to bin.
            bins: The edges defining the bins.

        Returns:
            self: The modified piplyr object.

        Examples:
            >>> df = pd.DataFrame({'A': [1, 4, 6, 8]})
            >>> pi = piplyr(df).bin_data('A', bins=[0, 3, 6, 9])
        """
        self.df[column + '_binned'] = pd.cut(self.df[column], bins=bins)
        return self
    
    
    def groupwise_custom(self, group_vars, func, *args, **kwargs):
        """
        Apply a custom function to groups within the DataFrame.

        Args:
            group_vars: Columns to group by.
            func: Custom function to apply to each group.
            *args, **kwargs: Additional arguments and keyword arguments for the function.

        Returns:
            self: The modified piplyr object.

        Examples:
            >>> df = pd.DataFrame({'Group': ['A', 'A', 'B', 'B'], 'Value': [1, 2, 3, 4]})
            >>> func = lambda x: x.sum()
            >>> pi = piplyr(df).groupwise_custom('Group', func)
        """
        self.grouped = self.df.groupby(group_vars)
        self.df = self.grouped.apply(lambda x: func(x, *args, **kwargs)).reset_index(drop=True)
        return self
    
    def sql_plyr(self, expression):
        """
        Executes an SQL query on the DataFrame.

        Args:
            expression: The SQL query to execute.

        Returns:
            self: The piplyr object with the DataFrame modified by the SQL query.
        """
        with sqlite3.connect(':memory:') as con:
            self.df.to_sql('df', con, index=False)
            self.df = pd.read_sql_query(expression, con)
        return self

    def convert_dtype(self, column, dtype):
        """
        Convert the data type of a specified column.

        Args:
            column: The column whose data type is to be converted.
            dtype: The target data type.

        Returns:
            self: The modified piplyr object.

        Examples:
            >>> df = pd.DataFrame({'A': ['1', '2', '3']})
            >>> pi = piplyr(df).convert_dtype('A', int)
        """
        self.df[column] = self.df[column].astype(dtype)
        return self

    def pipe(self, func, *args, **kwargs):
        """
        Allows the use of external functions in a chain.

        Args:
            func: A function to apply to the DataFrame.
            *args, **kwargs: Additional arguments and keyword arguments for the function.

        Returns:
            self: The modified piplyr object.
        """
        self.df = func(self.df, *args, **kwargs)
        return self



    def case_when(self, cases, target_var):
        """
        Applies conditions and assigns values based on them, similar to SQL's CASE WHEN.

        Args:
            cases: A list of tuples containing conditions and corresponding values.
            target_var: The name of the new or existing column to store the result.

        Returns:
            self: The modified piplyr object.
        Examples:
        >>> df = pd.DataFrame({'A': [10, 15, 20]})
        >>> cases = [('A > 15', 'High'), ('A <= 15', 'Low')]
        >>> pi = piplyr(df).case_when(cases, 'Category')    
        """
        self.df[target_var] = np.nan
        for condition, value in cases:
            self.df.loc[self.df.eval(condition), target_var] = value
        return self

    def join(self, other_df, by, join_type='inner'):
        """
        Joins the current DataFrame with another DataFrame.

        Args:
            other_df: The DataFrame to join with.
            by: The column name(s) to join on.
            join_type: Type of join to perform ('inner', 'left', 'right', 'outer').

        Returns:
            self: The modified piplyr object.
        Example:    
            >>> df = pd.DataFrame({'A': [1, 2, 3], 'B': [4, 5, 6],'F':['a','b','c']})
            >>> df2 = pd.DataFrame({'C': [10, 20], 'D': [40, 50],'F':['b','c']})
            >>> piplyr(df).join(df2,'F','outer')
        """
        if join_type not in ['inner', 'left', 'right', 'outer']:
            raise ValueError("join_type must be one of 'inner', 'left', 'right', 'outer'")
        self.df = self.df.merge(other_df, on=by, how=join_type)
        return self

    def count_na(self):
        """
        Counts the number of NA values in each column of the DataFrame.

        Returns:
            pd.Series: A Series with the count of NA values for each column.
            
        Examples:
        >>> df = pd.DataFrame({'A': [1, np.nan, 3], 'B': [4, 5, np.nan]})
        >>> na_count = piplyr(df).count_na()    
        """
        return self.df.isna().sum()

    # ... [Other existing methods] ...

    def distinct(self, columns=None):
        """
        Removes duplicate rows in the DataFrame.

        Args:
            columns: The columns to consider for identifying duplicates. 
                     If None, all columns are considered.

        Returns:
            self: The modified piplyr object.
        
        Examples:
        >>> df = pd.DataFrame({'A': [1, 1, 2], 'B': [3, 3, 3]})
        >>> pi = piplyr(df).distinct()
        """
        self.df = self.df.drop_duplicates(subset=columns)
        return self

    def skim(self):
        """
        Provides a summary of the DataFrame's statistics.

        Returns:
            pd.DataFrame: A DataFrame containing summary statistics for each column.
        
        Examples:
        >>> df = pd.DataFrame({'A': [1, 2, 3], 'B': [4, 5, 6]})
        >>> summary = piplyr(df).skim()
        """
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")

        stats = {
            'Types': self.df.dtypes,
            'Missing Values': self.df.isna().sum(),
            'Unique Values': self.df.nunique(),
            'Min': self.df.select_dtypes(include=[np.number]).min(),
            'Max': self.df.select_dtypes(include=[np.number]).max(),
            'Mean': self.df.select_dtypes(include=[np.number]).mean(),
            'Std': self.df.select_dtypes(include=[np.number]).std(),
            '25%': self.df.select_dtypes(include=[np.number]).quantile(0.25),
            '50%': self.df.select_dtypes(include=[np.number]).quantile(0.50),
            '75%': self.df.select_dtypes(include=[np.number]).quantile(0.75)
        }

        # Convert stats dict to DataFrame
        stats_df = pd.DataFrame(stats)

        # Fill non-applicable numeric stats with NaN for non-numeric columns
        for col in self.df.columns:
            if self.df[col].dtype not in [np.number]:
                stats_df.loc[['Min', 'Max', 'Mean', 'Std', '25%', '50%', '75%'], col] = np.nan

        return stats_df
    

    def pivot_longer(self, cols, id_vars=None, var_name='variable', value_name='value'):
        """
        Transforms the DataFrame from a wide format to a long format.

        Args:
            cols: Columns to unpivot.
            id_vars: Columns to leave unchanged (identifier variables).
            var_name: Name of the new column that will contain the variable names.
            value_name: Name of the new column that will contain the values.

        Returns:
            self: The modified piplyr object.
        
        Examples:
        >>> df = pd.DataFrame({'Name': ['A', 'B'], 'Val1': [1, 2], 'Val2': [3, 4]})
        >>> pi = piplyr(df).pivot_longer(cols=['Val1', 'Val2'], id_vars='Name')    
        """
        self.df = pd.melt(self.df, id_vars=id_vars, value_vars=cols, var_name=var_name, value_name=value_name)
        return self

    def pivot_wider(self, index, columns, values):
        """
        Transforms the DataFrame from a long format to a wide format.

        Args:
            index: Column(s) to use as index (identifier variables).
            columns: Column whose unique values will become new columns in the wide format.
            values: Column(s) that will be spread across the new columns.

        Returns:
            self: The modified piplyr object.
        
        Examples:
        >>> df = pd.DataFrame({'Name': ['A', 'A', 'B', 'B'], 'Variable': ['Val1', 'Val2', 'Val1', 'Val2'], 'Value': [1, 2, 3, 4]})
        >>> pi = piplyr(df).pivot_wider(index='Name', columns='Variable', values='Value')    
        """
        self.df = self.df.pivot(index=index, columns=columns, values=values)
        return self

    def clean_names(self):
        """
        Cleans and standardizes column names by converting them to lowercase and replacing 
        non-alphanumeric characters with underscores.

        Returns:
            self: The modified piplyr object.
            
        Examples:
        >>> df = pd.DataFrame({'First Name': [1, 2, 3], 'Last-Name': [4, 5, 6]})
        >>> pi = piplyr(df).clean_names()    
        """
        self.df.columns = [re.sub('[^0-9a-zA-Z]+', '_', col).lower() for col in self.df.columns]
        return self
    

    def separate(self, col, into, sep=None, remove=False, extra='warn'):
        """
        Separates a column into multiple columns based on a separator.

        Args:
            col: Column name to be separated.
            into: List of new column names after separation.
            sep: Separator to split the column (default: split on whitespace).
            remove: Flag to remove the original column (default: False).
            extra: Specifies behavior if there are extra splits. Options are 'drop', 'merge', and 'warn'.

        Returns:
            self: The piplyr object with the DataFrame modified.
        
        Examples:
        >>> df = pd.DataFrame({'Name': ['John Doe', 'Jane Smith']})
        >>> pi = piplyr(df).separate('Name', into=['FirstName', 'LastName'], sep=' ')
        """
        split_cols = self.df[col].str.split(sep, expand=True)
        num_cols = len(into)

        if split_cols.shape[1] > num_cols:
            if extra == 'drop':
                split_cols = split_cols.iloc[:, :num_cols]
            elif extra == 'merge':
                split_cols.iloc[:, num_cols-1] = split_cols.iloc[:, num_cols-1:].apply(lambda x: sep.join(x.dropna().astype(str)), axis=1)
                split_cols = split_cols.iloc[:, :num_cols]
            elif extra == 'warn':
                warnings.warn("Number of splits exceeds the length of 'into'; extra splits are being dropped.")

        self.df[into] = split_cols
        if remove:
            self.df = self.df.drop(columns=[col])
        return self

    def str_pad(self, column, width, side='left', pad=" "):
        """
        Pads strings in a DataFrame column to a specified width.

        Args:
            column: The column to pad.
            width: The width to pad the strings to.
            side: The side to pad on ('left' or 'right').
            pad: The character used for padding (default: space).

        Returns:
            self: The piplyr object with the DataFrame modified.
        
        Examples:
        >>> df = pd.DataFrame({'Name': ['John', 'Jane']})
        >>> pi = piplyr(df).str_pad('Name', 10, side='right', pad='_')
        """
        if side not in ['left', 'right']:
            raise ValueError("Side must be either 'left' or 'right'")

        if side == 'left':
            self.df[column] = self.df[column].astype(str).str.pad(width, side='left', fillchar=pad)
        else:  # side == 'right'
            self.df[column] = self.df[column].astype(str).str.pad(width, side='right', fillchar=pad)
        return self

    def str_replace(self, pattern, replacement):
        """
        Replaces a pattern in strings with a replacement string.

        Args:
            pattern: The regex pattern to replace.
            replacement: The replacement string.

        Returns:
            self: The piplyr object with the DataFrame modified.
        
        Examples:
        >>> df = pd.DataFrame({'Text': ['foo123', 'bar456', 'baz789']})
        >>> pi = piplyr(df).str_sub('\\d+', 'XYZ')
        """
        self.df = self.df.applymap(lambda x: re.sub(pattern, replacement, str(x)) if isinstance(x, str) else x)
        return self

    def str_extract(self, pattern, col=None):
        """
        Extracts a pattern from a string column.

        Args:
            pattern: The regex pattern to extract.
            col: The column to apply extraction. If None, applies to all string columns.

        Returns:
            self: The piplyr object with the DataFrame modified.
        
        Examples:
        >>> df = pd.DataFrame({'Text': ['apple123', 'banana456', 'cherry789']})
        >>> pi = piplyr(df).str_extract('(\\d+)', 'Text')
        """
        if col:
            self.df[col + '_extracted'] = self.df[col].str.extract(pattern)
        else:
            for c in self.df.columns:
                if self.df[c].dtype == object:
                    self.df[c + '_extracted'] = self.df[c].str.extract(pattern)
        return self

    def str_detect(self, col, pattern):
        """
        Detects if a pattern exists in a string column.

        Args:
            col: The column to check for the pattern.
            pattern: The regex pattern to detect.

        Returns:
            self: The piplyr object with a new column indicating detection.
            Examples:
            >>> df = pd.DataFrame({'A': ['foo', 'bar', 'baz']})
            >>> pi = piplyr(df).str_detect('A','foo')
            >>> print(pi.to_df)
        """
        self.df[col + '_detected'] = self.df[col].str.contains(pattern, na=False)
        return self
    
    def str_len(self, col):
        """
        Calculates the length of strings in a specified DataFrame column.

        Args:
            col (str): The column to calculate string lengths for.

        Returns:
            self: The piplyr object with a new column appended indicating string lengths.

        Examples:
            >>> df = pd.DataFrame({'A': ['foo', 'bar', 'baz']})
            >>> pi = piplyr(df).str_len('A')
            >>> print(pi.to_df)
        """
        self.df[col + '_len'] = self.df[col].str.len()
        return self

    def str_lower(self, col):
        """
        Converts strings in the specified column to lowercase.

        Args:
            col (str): The column to convert strings to lowercase.

        Returns:
            self: The piplyr object with the specified column converted to lowercase.

        Examples:
            >>> df = pd.DataFrame({'A': ['FOO', 'Bar', 'BAZ']})
            >>> pi = piplyr(df).str_lower('A')
            >>> print(pi.to_df)
        """
        self.df[col] = self.df[col].str.lower()
        return self

    def str_upper(self, col):
        """
        Converts strings in the specified column to uppercase.

        Args:
            col (str): The column to convert strings to uppercase.

        Returns:
            self: The piplyr object with the specified column converted to uppercase.

        Examples:
            >>> df = pd.DataFrame({'A': ['foo', 'bar', 'baz']})
            >>> pi = piplyr(df).str_upper('A')
            >>> print(pi.to_df)
        """
        self.df[col] = self.df[col].str.upper()
        return self

    def str_startswith(self, col, prefix):
        """
        Checks if strings in a specified column start with a given prefix.

        Args:
            col (str): The column to check.
            prefix (str): The prefix to check for.

        Returns:
            self: The piplyr object with a new column appended indicating if strings start with the prefix.

        Examples:
            >>> df = pd.DataFrame({'A': ['apple', 'banana', 'cherry']})
            >>> pi = piplyr(df).str_startswith('A', 'a')
            >>> print(pi.to_df)
        """
        self.df[col + '_startswith'] = self.df[col].str.startswith(prefix)
        return self

    def str_endswith(self, col, suffix):
        """
        Checks if strings in a specified column end with a given suffix.

        Args:
            col (str): The column to check.
            suffix (str): The suffix to check for.

        Returns:
            self: The piplyr object with a new column appended indicating if strings end with the suffix.

        Examples:
            >>> df = pd.DataFrame({'A': ['apple', 'banana', 'cherry']})
            >>> pi = piplyr(df).str_endswith('A', 'e')
            >>> print(pi.to_df)
        """
        self.df[col + '_endswith'] = self.df[col].str.endswith(suffix)
        return self

    def str_contains(self, col, pattern):
        """
        Checks if strings in a specified column contain a given pattern.

        Args:
            col (str): The column to check.
            pattern (str): The pattern to check for.

        Returns:
            self: The piplyr object with a new column appended indicating if strings contain the pattern.

        Examples:
            >>> df = pd.DataFrame({'A': ['apple', 'banana', 'cherry']})
            >>> pi = piplyr(df).str_contains('A', 'an')
            >>> print(pi.to_df)
        """
        self.df[col + '_contains'] = self.df[col].str.contains(pattern)
        return self

    
    

    def fct_lump(self, column, n=10, other_level='Other'):
        """
        Lumps less frequent levels of a categorical column into an 'Other' category.

        Args:
            column: The name of the categorical column.
            n: The minimum count to not be lumped into 'Other'.
            other_level: The name for the lumped category (default: 'Other').

        Returns:
            self: The piplyr object with the DataFrame modified.

        Examples:
            >>> df = pd.DataFrame({'Category': ['A', 'B', 'B', 'C', 'C', 'C', 'D', 'D', 'D', 'D']})
            >>> pi = piplyr(df).fct_lump('Category', n=3)
            >>> print(pi.to_df)
        """
        value_counts = self.df[column].value_counts()
        self.df[column] = np.where(self.df[column].isin(value_counts.index[value_counts >= n]), self.df[column], other_level)
        return self

    def fct_infreq(self, column, frac=0.01, other_level='Other'):
        """
        Lumps infrequent levels of a categorical column based on a fraction of total occurrences.

        Args:
            column: The name of the categorical column.
            frac: Fraction of total occurrences to be considered infrequent.
            other_level: The name for the lumped category (default: 'Other').

        Returns:
            self: The piplyr object with the DataFrame modified.

        Examples:
            >>> df = pd.DataFrame({'Category': ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J']})
            >>> pi = piplyr(df).fct_infreq('Category', frac=0.1)
            >>> print(pi.to_df)
        """
        value_counts = self.df[column].value_counts(normalize=True)
        self.df[column] = np.where(self.df[column].isin(value_counts.index[value_counts >= frac]), self.df[column], other_level)
        return self

    def fct_relevel(self, column, ref_level, after=True):
        """
        Reorders levels of a categorical column, moving a specified level to the first or last.

        Args:
            column: The name of the categorical column.
            ref_level: The reference level to move.
            after: Whether to move the reference level after the other levels.

        Returns:
            self: The piplyr object with the DataFrame modified.

        Examples:
            >>> df = pd.DataFrame({'Category': ['B', 'A', 'C']})
            >>> pi = piplyr(df).fct_relevel('Category', 'B', after=False)
            >>> print(pi.to_df)
        """
        self.df[column] = pd.Categorical(self.df[column], categories=self.df[column].unique(), ordered=True)
        if ref_level not in self.df[column].cat.categories:
            raise ValueError(f"Reference level '{ref_level}' not found in column '{column}'")
        if after:
            new_order = [cat for cat in self.df[column].cat.categories if cat != ref_level] + [ref_level]
        else:
            new_order = [ref_level] + [cat for cat in self.df[column].cat.categories if cat != ref_level]
        self.df[column] = self.df[column].cat.reorder_categories(new_order)
        return self

    def fct_recode(self, column, recode_dict, drop_unused=False):
        """
        Recodes levels of a categorical column.

        Args:
            column: The name of the categorical column.
            recode_dict: Dictionary mapping old levels to new levels.
            drop_unused: Whether to drop unused categories after recoding.

        Returns:
            self: The piplyr object with the DataFrame modified.

        Examples:
            >>> df = pd.DataFrame({'Category': ['A', 'B', 'C']})
            >>> pi = piplyr(df).fct_recode('Category', {'A': 'X', 'B': 'Y'})
            >>> print(pi.to_df)
        """
        if not all(level in self.df[column].cat.categories for level in recode_dict.keys()):
            raise ValueError("One or more levels to recode not found in column categories")
        self.df[column] = self.df[column].cat.rename_categories(recode_dict)
        if drop_unused:
            self.df[column] = self.df[column].cat.remove_unused_categories()
        return self

    def fct_reorder(self, column, order_by, ascending=True):
        """
        Reorders the levels of a categorical column based on another column.

        Args:
            column: The name of the categorical column.
            order_by: The column by which to order the categories.
            ascending: Whether to sort the categories in ascending order (default: True).

        Returns:
            self: The piplyr object with the DataFrame modified.

        Examples:
            >>> df = pd.DataFrame({'Category': ['A', 'B', 'C'], 'Values': [3, 1, 2]})
            >>> pi = piplyr(df).fct_reorder('Category', 'Values')
            >>> print(pi.to_df)
        """
        ordering = self.df[order_by].argsort()
        if not ascending:
            ordering = ordering[::-1]
        self.df[column] = pd.Categorical(self.df[column], categories=np.array(self.df[column])[ordering], ordered=True)
        return self

    


    def __call__(self, df):
        """
        Allows the piplyr object to be called with a new DataFrame, replacing the current one.

        Args:
            df: A new pandas DataFrame to replace the current one.

        Returns:
            self: The piplyr object with the new DataFrame.
        """
        if not isinstance(df, pd.DataFrame):
            raise TypeError("Input must be a pandas DataFrame")
        self.df = df
        self.grouped = None
        return self

    def __repr__(self):
        """
        Returns a string representation of the DataFrame.

        Returns:
            str: A string representation of the DataFrame.
        """
        return self.df.__repr__()

    @property
    def to_df(self):
        """
        Converts the piplyr object's DataFrame to a standard pandas DataFrame.

        Returns:
            pd.DataFrame: The DataFrame contained within the piplyr object.
            Examples:
            >>> df = pd.DataFrame({'A': [1, 2, 3]})
            >>> pi = piplyr(df)
            >>> print(pi.to_df())
        """
        return pd.DataFrame(self.df)
