
import pandas as pd
import numpy as np
from sqlite3 import connect
import sqlite3
import re


class piplyr:
    
    def __init__(self, df):
        """
        Initializes the class with a dataframe and creates a SQLite connection
        """
        self.df = df
        

        ## group by

    def group_by(self,group_var=None):
        """
        Groups the dataframe by the specified column
        
        Args:
            group_var: The column name to group the dataframe by
        
        Returns:
            self: the piplyr object with the grouped dataframe
        
        Example:
            df = pd.DataFrame({'A': [1, 2, 3, 4], 'B': [5, 6, 7, 8],'C': ['a','b','c','d']})
            pi = piplyr(df)
            pi.group_by("C")
        """
        self.df = self.df.groupby(group_var)
        return self        
        
        
    def sort_by(self, column, ascending=True):
        
         """
        Sorts the dataframe by the specified column
        
        Args:
            column: The column name to sort the dataframe by
            ascending: A boolean value specifying the sorting order. Default value is True for ascending order.
        
        Returns:
            self: the piplyr object with the sorted dataframe
        
        Example:
            df = pd.DataFrame({'A': [1, 2, 3, 4], 'B': [5, 6, 7, 8],'C': ['a','b','c','d']})
            pi = piplyr(df)
            pi.sort_by("A", False)
        """

         self.df = self.df.sort_values(by=column, ascending=ascending)
         return self   


    def select(self, *columns):
        """
        Selects specified columns from the dataframe
        
        Args:
            columns: The column names to select from the dataframe. This can be passed as multiple arguments.
        
        Returns:
            self: the piplyr object with the selected columns
        
        Example:
            df = pd.DataFrame({'A': [1, 2, 3, 4], 'B': [5, 6, 7, 8],'C': ['a','b','c','d']})
            pi = piplyr(df)
            pi.select("A", "C")
        """
        self.df = self.df[list(columns)]
        return self
    
## drop columns
    def drop_col(self, column):
        """
        This method is used to drop a specified column from a DataFrame.

        Parameters:
        column (str): The name of the column to be dropied.

        Returns:
        DataFrame: The updated DataFrame with the specified column removed.

        Example:
        df = DataFrame({"A": [1, 2, 3], "B": [4, 5, 6]})
        df_obj = piplyr(df)
        df_obj.drop_col("A")
        # The DataFrame now only contains column "B"
        """
        self.df = self.df.drop(columns=column,inplace=False)
        return self

    def rename_col(self,rename_cols):
        '''
        to rename columns, similar to rename in Pandas
        rename_cols: a dictionaray for renaiming the variable
        
        Example: 
        df = pd.DataFrame({'A': [1, 2, 3, 4], 'B': [5, 6, 7, 8],'C': ['a','b','c','d']})
        pi = piplyr(df)
        pi.rename_col(
            {
                "A":"A_A",
                "B":"NEW_B_NAME"
            }
        )
        '''
        self.df = self.df.rename(columns=rename_cols)
        return self
    
 
    def filter(self, query):
        ''' 
        filter(query)
        Method to filter the DataFrame based on a given query.

        Parameters:
        query(str): A string containing the query to filter the DataFrame by. 
                    The query should be in the format of a valid pandas query, 
                    using the syntax df.query("column_name pierator value")
                    
        Returns:
        - PandasDataFrame: The filtered DataFrame and a piplyr object

        df = pd.DataFrame({'A': [1, 2, 3, 4], 'B': [5, 6, 7, 8],'C': ['a','b','c','d']})
        pi = piplyr(df)
        pi.filter("A > 2 & B > 7")

        df = pd.DataFrame({'A': [1, 2, 3, 4], 'B': [5, 6, 7, 8],'C': ['a','b','c','d']})
        pi = piplyr(df)
        pi.filter("A>2 | C =='d'")
        '''
        self.df = self.df.query(query)
        return self

    
# mutate a new column based on an expression
   
    def mutate(self, expression,new_var_name):
        
        """
        This method allows you to create a new column in your dataframe by apilying an expression to existing columns.

        :param new_var_name: str, name of the new column
        :param expression: str, valid pandas expression to create new column
        :return: the piplyr object with the new column added to the dataframe

        Example:
        df = pandas.DataFrame({'A': [1, 2, 3], 'B': [4, 5, 6]})
        pi = piplyr(df)
        pi.mutate('C', 'A + B')
        pi.df
        # Output:
        A B C
        0 1 4 5
        1 2 5 7
        2 3 6 9
        """
        self.df[new_var_name] = self.df.eval(expression,inplace=False)
        return self
    
            
# write any sql expression on the data frame fed to pilyr          
    def sql_plyr(self,expression):
        

        """
        This method allows to perform SQL queries on the dataframe using the SQLite connection. 
        The resulting dataframe is updated with the query results.
        When refering to the datat frame inside the query, you should use 'df' no matter what the name of your
        dataframe is.
        
        Parameters:
        expression (str): SQL expression to be executed
        
        Returns:
        piplyr: Returns the updated piplyr object
        
        Example:
        df_with_a_name = pd.DataFrame({'x': [1, 2, 3], 'y': [4, 5, 6]})
        pi = piplyr(df_with_a_name)
        pi.sql_plyr('SELECT * FROM df WHERE x > 2')
        
        """
        self.con = sqlite3.connect(':memory:')
        self.df.to_sql('df', self.con)
        #self.query = 'SELECT * FROM df'
        #self.query_exp = 'SELECT * FROM df'
        
        self.query_exp = f'{expression}'
        self.df = pd.read_sql_query(self.query_exp, self.con)
        return self
    
    def case_when(self, cases, target_var):
        '''
        this is similar to case when function of SQL, 
        cases: the conditions is introduced in cases argument in a list of tuples [(),(),...]
        
        target_var: the target_var is the name of new column for the output of case_when
        if you would like to replace an existing column with the results, use it as a target_var
        
        Example: 
        df = pd.DataFrame({'A': [1, 2, 3, 4], 'B': [5, 6, 7, 8],'C': ['a','b','c','d']})
        pi = piplyr(df)

        pi.case_when(
            [
          (" C == 'a' ","AAA"),
          ("C in ('b','c','d')","OTHER"),
            ],
        target_var="new_col"
        ) ## you could also replace new_col with C if you want to change the contents of C
        
        Example 2:
        
        df = pd.DataFrame({'a': [1,2,3,4], 'b':[10,20,30,40]})
        pi = piplyr(df)
        (pi.case_when(
            [
                ('a > 2', round(333.333,1)),
                ('b < 25', np.mean(df['b']))
            ],target_var="new_col")
        )
        '''
        if target_var is None:
            self.df[target_var] = np.nan # initialize new variable as NaN
            for cond, val in cases:
                self.df.loc[self.df.eval(cond), target_var] = val
        else:
            for cond, val in cases:
                self.df.loc[self.df.eval(cond), target_var] = val
        return self

    
    def summarize(self,group_var=None, var=None, agg_func=None):
        """
    This method allows to perform groupby and aggregation on the dataframe.

        Parameters:
        group_var (str): The variable to group the dataframe by.
        var (str): The variable to perform the aggregation on.
        agg_func (str or function): The aggregation function to apily to the variable.
        Can be a string for built-in aggregation functions such as 'mean' or 'sum'
        or a user-defined function.

        Returns:
        piplyr: Returns the updated piplyr object with the summarized dataframe.

        Example:
        df = pd.DataFrame({'x': [1, 2, 3], 'y': [4, 5, 6],'z':['a','b','a']})
        pi = piplyr(df)
        pi.summarize(group_var='z',var='y',agg_func='mean').to_df
        or to apily the agg_func to the whole dataframe
        pi.summarize(var='y',agg_func='mean').to_df
        """
        if group_var and var and agg_func:
            self.df = self.df.groupby(group_var)[var].agg(agg_func)
        elif var and agg_func:
            self.df = self.df[var].agg(agg_func)
        return self
        

    def join(self, other_df, by, join_type='inner'):
        '''
        join method to join the df of pilyr to other data frames
        by: the key by which the dataframes can be joined
        join_type: type of the join including 'inner','left', etc 
        
        Example: 
        df1 = pd.DataFrame({'key': ['A', 'B', 'C', 'D'], 'value': [1, 2, 3, 4]})
        df2 = pd.DataFrame({'key': ['B', 'D', 'E', 'F'], 'value': [5, 6, 7, 8]})
        pi = piplyr(df1)
        pi.join(df2, by='key')
        print(pi.to_df)
        '''
        self.df = self.df.merge(other_df, on=by, how=join_type)
        return self
    

    def count_na(self, var=None):
      if var is None:
          # count missing values for all columns
          cols = self.df.columns
      else:
          cols = var
      for col in cols:
          missing_values = self.df[col].isnull().sum()
          if self.df[col].dtype == 'object' or self.df[col].dtype == 'category':
              categories = self.df[col].unique()
              print(f"Column: {col}\n \t **Number of Missing Values: {missing_values}\n\t Categories: {categories}")
          else:
              print(f"Column: {col}\n \t Missing Values: {missing_values}")
    
    def distinct(self, column=None):
        '''
        This will return a new dataframe with only the unique rows of the original dataframe
        that was passed to the pilyr class.
        Example:
        df = pd.DataFrame({'key': ['A', 'B', 'C', 'D'], 'value': [1, 1, 1, 4]})
        piplyr(df).distinct('value')
        '''
        self.df = self.df.drop_duplicates(subset=column)
        return self


    def skim(self):
        """
        Summary statistics of a dataframe.

        This method provides a compact overview of the key characteristics of a dataframe. It includes information about data types, missing values, unique values, categories (for categorical variables), minimum and maximum values, mean, median, skewness, standard deviation, 25th, 50th, and 75th percentiles. The output table adjusts its width to display the full content of its columns. 

        Returns:
    -------
        DataFrame: A dataframe with the summary statistics for each column in the original dataframe.

        """
        import warnings
        warnings.filterwarnings("ignore", category=FutureWarning)
        result = {}
        result['Types'] = self.df.dtypes
        result['Number of Missing Values'] = self.df.isna().sum()
        result['Number of Unique Values'] = self.df.nunique()
        categories = {}
        for col in self.df.columns:
            if self.df[col].dtype == "object" or self.df[col].dtype == "category":
                unique_vals = self.df[col].dropna().unique()
                categories[col] = [f"{i+1}-{val}" for i, val in enumerate(unique_vals)]
        
        result['Categories'] = pd.Series(categories)
        result['Minimum'] = self.df.min()
        result['Maximum'] = self.df.max()
        result['Mean'] = self.df.mean()
        result['Skewness'] = self.df.skew()
        result['Standard Deviation'] = self.df.std()
        result['25th Percentile'] = self.df.quantile(0.25)
        result['50th Percentile'] = self.df.quantile(0.50)
        result['75th Percentile'] = self.df.quantile(0.75)
        
        
        result_df = pd.DataFrame(result).sort_values("Types").fillna(value='-')
        pd.options.display.max_colwidth = 1000

        return result_df
    
    
    
    
    def pivot_longer(self, cols, id_vars=None, values_to='value', names_to='name'):
        """
        Pivots the dataframe from wide format to long format

        Args:
            value_vars: The columns to be pivoted into the long format
            id_vars: The columns to be used as identifier variables (default is None)
            var_name: The name of the new column that contains the variable names (default is 'variable')
            value_name: The name of the new column that contains the pivoted values (default is 'value')
        
        Returns:
            self: the piplyr object with the pivoted dataframe
        
        Example:
            df = pd.DataFrame({'A': [1, 2, 3, 4], 'B': [5, 6, 7, 8], 'C': [9, 10, 11, 12]})
            pi = piplyr(df)
            pi.pivot_longer(cols=['B', 'C'], id_vars='A', values_to='value', names_to='name')
        """
        if id_vars is None:
            id_vars = self.df.columns.to_list()
        self.df = pd.melt(self.df, id_vars=id_vars, value_vars=cols, var_name=names_to, value_name=values_to)
        return self


    def pivot_wider(self, names_from, values_from, names_repair=None, id_vars=None):
        """
        Pivots the dataframe from long format to wide format

        Args:
            names_from: The name of the column containing the variable names
            values_from: The name of the column containing the values
            names_repair: A function that takes in a variable name and returns a modified variable name
                (default is None, which means no modification will be made)
            id_vars: The columns to be used as identifier variables (default is None, which means all columns except `names_from` and `values_from`)

        Returns:
            self: the piplyr object with the pivoted dataframe

        Example:
            df = pd.DataFrame({'A': [1, 1, 2, 2], 'variable': ['B', 'C', 'B', 'C'], 'value': [5, 9, 6, 10]})
            pi = piplyr(df)
            pi.pivot_wider(names_from='variable', values_from='value')
        """
        if id_vars is None:
            id_vars = [col for col in self.df.columns if col not in [names_from, values_from]]
        if names_repair is None:
            names_repair = lambda x: x
        self.df = self.df.pivot(index=id_vars, columns=names_from, values=values_from)
        self.df.columns = [names_repair(col) for col in self.df.columns]
        self.df.reset_index(inplace=True)
        return self
 

    def clean_names(self):
        """
        it cleans the name of variables, 
        """
        self.df.columns = [re.sub('[^0-9a-zA-Z]+', '_', col) for col in self.df.columns]
        self.df.columns = [col.lower() for col in df.columns]
        return self
    
    
    def separate(self, col, into, sep=None, remove=False, extra='warn'):
        """
        Separates a column into multiple columns based on a separator.

        Args:
            col: The name of the column to be separated
            into: A list of names for the new columns that will be created
            sep: The separator used to split the values in `col` (default is None, meaning to split on whitespace)
            remove: A flag to remove the original column (default is False)
            extra: What to do if there are extra values that don't fit in the new columns, options are 'drop', 'merge' and 'warn' (default is 'warn')
        
        Returns:
            self: the piplyr object with the separated dataframe
        
        Example:
            df = pd.DataFrame({'A': [1, 2, 3, 4], 'B': ['a_/*b', 'c_d', 'e_f', 'g*_h']})
            pi = piplyr(df)
            pi.separate(col='B', into=['B1', 'B2'], sep='_', remove=True)
        """
        self.df[into] = self.df[col].str.split(sep, expand=True)
        if remove:
            self.df.drop(col, axis=1, inplace=True)
        return self



    def str_pad(self, width, side='left', pad=" "):
        """
        Pads strings in the dataframe to the specified width

        Args:
            width: The width to pad the strings to
            side: The side to pad the strings (either 'left' or 'right', default is 'left')
            pad: The character to pad the strings with (default is ' ')

        Returns:
            self: the piplyr object with the padded strings in the dataframe

        Example:
            df = pd.DataFrame({'A': ['hello', 'world', 'foo', 'bar']})
            pi = piplyr(df)
            pi.str_pad(10, side='right', pad='*')
        """
        if side == 'left':
            self.df = self.df.applymap(lambda x: str(x).rjust(width, pad) if isinstance(x, str) else x)
        elif side == 'right':
            self.df = self.df.applymap(lambda x: str(x).ljust(width, pad) if isinstance(x, str) else x)
        return self

    def str_sub(self, pattern, replacement):
        """
        Replaces a pattern in strings in the dataframe with a replacement string

        Args:
            pattern: The pattern to search for in the strings
            replacement: The replacement string

        Returns:
            self: the piplyr object with the replaced strings in the dataframe

        Example:
            df = pd.DataFrame({'A': ['hello', 'world', 'foo', 'bar']})
            pi = piplyr(df)
            pi.str_sub('o', '0')
        """
        self.df = self.df.applymap(lambda x: re.sub(pattern, replacement, str(x)) if isinstance(x, str) else x)
        return self



    def str_extract(self, pattern, col=None):
        """
        Extracts the first occurrence of a pattern from each string in a column
        
        Args:
            pattern: A regular expression pattern to match in the strings
            col: The column name to extract the substring from. If None, apilies to all columns with object dtype.
        
        Returns:
            self: The piplyr object with the extracted substrings added as a new column

        Example:
            df = pd.DataFrame({'col1': ['abc123', 'def456', 'ghi789']})
            pi = piplyr(df)
            pi.str_extract(pattern='[a-z]+', col='col1')
        """
        if col:
            self.df[col + '_extracted'] = self.df[col].str.extract(f'({pattern})')
        else:
            for c in self.df.columns:
                if self.df[c].dtype == object:
                    self.df[c + '_extracted'] = self.df[c].str.extract(f'({pattern})')
        return self



    def str_detect(self, col, pattern):
        """
        Check if a string pattern is present in a specific column

        Args:
            col: the name of the column to be searched
            pattern: the string pattern to be searched for

        Returns:
            self: the piplyr object with the modified dataframe
        
        Example:
            df = pd.DataFrame({'col1': ['abc', 'def', 'ghi', 'jkl']})
            pi = piplyr(df)
            pi.str_detect(col='col1', pattern='a')
        """
        self.df['detected'] = self.df[col].str.contains(pattern)
        return self

    
    def fct_lump(self, column, n=10, other_level='Other'):
        """
        Lumps levels of a factor column into 'Other' level for levels that apiear less than n times

        Args:
            column: The name of the column containing the factor levels
            n: The number of times a level should apiear to avoid being lumped (default is 10)
            other_level: The name of the new level for the lumped levels (default is 'Other')
        
        Returns:
            self: the piplyr object with the lumped column
        
        Example:
            df = pd.DataFrame({'A': [1, 2, 3, 4], 'B': ['A', 'B', 'C', 'A']})
            pi = piplyr(df)
            pi.fct_lump(column='B', n=2, other_level='Rare')
        """
        counts = self.df[column].value_counts()
        self.df[column] = np.where(self.df[column].isin(counts[counts >= n].index), self.df[column], other_level)
        return self
    
    def fct_infreq(self, column, frac=0.01, other_level='Other'):
        """
        Lumps levels of a factor column into 'Other' level for levels that apiear less than a fraction of total

        Args:
            column: The name of the column containing the factor levels
            frac: The fraction of the total number of times a level should apiear to avoid being lumped (default is 0.01)
            other_level: The name of the new level for the lumped levels (default is 'Other')
        
        Returns:
            self: the piplyr object with the lumped column
        
        Example:
            df = pd.DataFrame({'A': [1, 2, 3, 4], 'B': ['A', 'B', 'C', 'A']})
            pi = piplyr(df)
            pi.fct_infreq(column='B', frac=0.5, other_level='Rare')
        """
        counts = self.df[column].value_counts(normalize=True)
        self.df[column] = np.where(self.df[column].isin(counts[counts >= frac].index), self.df[column], other_level)
        return self

    def fct_relevel(self, col, ref, after=True):
        """
        Relevels the factor column to put the reference level first

        Args:
            col: The name of the factor column
            ref: The reference level to put first
            after: Whether to put the reference level after (True) or before (False) the other levels (default is True)
        
        Returns:
            self: the piplyr object with the releveled factor column
        
        Example:
            df = pd.DataFrame({'col1': [1, 2, 3, 4], 'col2': ['A', 'B', 'C', 'D']})
            pi = piplyr(df)
            pi.fct_relevel(col='col2', ref='B', after=False)
        """
        self.df[col] = self.df[col].astype('category')
        if after:
            self.df[col] = self.df[col].cat.reorder_categories([ref] + [x for x in self.df[col].cat.categories if x != ref])
        else:
            self.df[col] = self.df[col].cat.reorder_categories([x for x in self.df[col].cat.categories if x != ref] + [ref])
        return self

    
    def fct_recode(self, col, values, labels=None, drop=False):
        """
        Recodes factor levels in a given column

        Args:
            col: The column name to be recoded
            values: The old factor levels to be recoded
            labels: The new factor levels (default is None)
            drop: Specifies whether to drop unused levels (default is False)

        Returns:
            self: The piplyr object with the recoded dataframe
        """
        self.df[col] = self.df[col].astype('category')
        self.df[col] = self.df[col].cat.reorder_categories(values, ordered=True)
        if labels:
            self.df[col] = self.df[col].cat.rename_categories(labels)
        if drop:
            self.df[col].cat.remove_unused_categories(inplace=True)
        return self
        
    def __call__(self, df):
        self.df = df
        return self
    
    def __repr__(self):
        return self.df.__repr__()
    
    @property
    def to_df(self):
        """
        @property to_df
        This property allows to convert the dataframe of the piplyr object to a standard pandas dataframe.
        It returns a pandas DataFrame object with the same data as the original dataframe.
        Example:
        df_with_a_name = pd.DataFrame({'x': [1, 2, 3], 'y': [4, 5, 6]})
        pi = piplyr(df_with_a_name)
        df = pi.to_df
        """
        return pd.DataFrame(self.df)
