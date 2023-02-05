# piplyr package

#### **The "piplyr" package offers a convenient way to manipulate data using a syntax similar to the popular R package "dplyr". It's built as a simple wrapper for the "pandas" library, providing a "piplyr" class that can be initialized with a pandas DataFrame. The package includes methods for tasks like grouping, sorting, selecting, dropping, renaming, filtering, and even SQL-like functionality.**

## Installation
You can install the package using pip:

```
# make sure to install the latest version
pip install piplyr==0.0.8

```

## Usage

The package can be used by first importing the piplyr class and then initializing it with a DataFrame. The package provides several methods that can be used to manipulate the DataFrame.

```
import pandas as pd
from piplyr.main import piplyr
```


```
## Create a sample DataFrame
df = pd.DataFrame({'A': [1, 2, 3, 4], 'B': [5, 6, 7, 8],'C': ['a','b','c','d']})

## Initialize the piplyr class with the DataFrame
pi = piplyr(df)
```


## Methods

### group_by
The group_by method can be used to group the DataFrame by a specified column.

```
pi.group_by("C")
```

### sort_by
The sort_by method can be used to sort the DataFrame by a specified column in ascending or descending order.

```
pi.sort_by("A", False)
```

### select
The select method can be used to select specified columns from the DataFrame.

```
pi.select("A", "C")
```
### drop_col
The drop_col method can be used to drop a specified column from the DataFrame.

```
pi.drop_col("A")
```
### rename_col
The rename_col method can be used to rename columns of the DataFrame.

```
pi.rename_col(
    {
 "A":"A_A",
 "B":"NEW_B_NAME"
    }
)
```

### filter
The filter method can be used to filter the DataFrame based on a given query. The query should be in the format of a valid pandas query, using the syntax df.query("column_name pierator value")

```
pi.filter("A > 2")

```

### mutate_eval

This method allows you to create a new column in your dataframe by apilying an expression to existing columns.

Args:
new_col: str, name of the new column
expression: str, valid pandas expression to create new column

```
## Example:
df = pandas.DataFrame({'A': [1, 2, 3], 'B': [4, 5, 6]})
pi = piplyr(df)
pi.mutate_eval( 'A + B','C')

```
### mutate_func(self,col,func,new_col,*arg,**kwarg):

This method allows the generate of a new variable using a function
        
 Args: 
 col: the coloumn whose values are used in generation of new column, 
 func: a function defined by a user or a pre-defined function.
 new_col: the name of new column. 
```
##Example:
df = pd.DataFrame({'A': [1, 2, 3, 4], 'B': [5, 6, 7, 8],'C': ['a','b','c','d']})
pi= piplyr(df)
pi.mutate_func('A',np.mean,'mean_A')


df = pd.DataFrame({'A': [1, 2, 3, 4], 'B': [5, 6, 7, 8],'C': ['a','b','c','d']})
pi= piplyr(df)
pi.mutate_func('A',lambda x: np.mean(x)+10,'mean_A_plus_10')

```

### sql_plyr
SQL like functionality
The piplyr also provides SQL like functionality by creating a SQLite connection and saving the DataFrame as a table in memory. The query method can be used to execute a SQL query on the DataFrame.
When refering to the datat frame inside the query, you should use 'df' no matter what the name of your
dataframe is.

```
# Execute a SQL query
df_with_a_name = pd.DataFrame({'x': [1, 2, 3], 'y': [4, 5, 6]})
pi = piplyr(df_with_a_name)
pi.sql_query('SELECT * FROM df WHERE x > 2')
```

### case_when
This is similar to case when function of SQL, 
if you would like to replace an existing column with the results, use it as a target_var

``` 
df = pd.DataFrame({'A': [1, 2, 3, 4], 'B': [5, 6, 7, 8],'C': ['a','b','c','d']})
pi = piplyr(df)

pi.case_when(
 [
   (" C == 'a' ","AAA"),
   ("C in ('b','c','d')","OTHER"),
 ],
 target_var="new_col"
 ) 

```

### summarize
This method allows to perform groupby and aggregation on the dataframe.

Parameters:
 group_var (str): The variable to group the dataframe by.
 var (str): The variable to perform the aggregation on.
 agg_func (str or function): The aggregation function to apily to the variable.
 Can be a string for built-in aggregation functions such as 'mean' or 'sum'
 or a user-defined function.

 ```
 Example:
        df = pd.DataFrame({'x': [1, 2, 3], 'y': [4, 5, 6],'z':['a','b','a']})
        pi = piplyr(df)
        pi.summarize(group_var='z',var='y',agg_func='mean').to_df
        or to apily the agg_func to the whole dataframe
        pi.summarize(var='y',agg_func='mean').to_df
```

### Chaining
All the methods provided by the piplyr class return the piplyr object, allowing you to chain multiple methods together to perform multiple data manipulation tasks in one line. Please see the examples provided below. 

```
pi.group_by("C").sort_by("A", False).select("A", "C")
```
### clean_names()
This method can be used to clean the name of variables

```
df = pd.DataFrame({'key is': ['A', 'B', 'C', 'D'], 'Value_ dD': [1, 1, 1, 4]})
piplyr(df).clean_names().to_df
```

### skim
This method provides a compact overview of the key characteristics of a dataframe.
It can also be used after data manipulation operations such as select, sql_dplyr, filter, etc. 
```
pi.skim()
```


## separate():

 Separates a column into multiple columns based on a separator.

 Args:
     col: The name of the column to be separated
     into: A list of names for the new columns that will be created
     sep: The separator used to split the values in `col` (default is None, meaning to split on whitespace)
     remove: A flag to remove the original column (default is False)
     extra: What to do if there are extra values that don't fit in the new columns, pitions are 'drop', 'merge' and 'warn' (default is 'warn')
 

```
  ### Example:
     df = pd.DataFrame({'A': [1, 2, 3, 4], 'B': ['a_/*b', 'c_d', 'e_f', 'g*_h']})
     pi = piplyr(df)
     pi.separate(col='B', into=['B1', 'B2'], sep='_', remove=True)
 
```

 ## str_pad
  
 Pads strings in the dataframe to the specified width

 Args:
     width: The width to pad the strings to
     side: The side to pad the strings (either 'left' or 'right', default is 'left')
     pad: The character to pad the strings with (default is ' ')


```
  ### Example:
     df = pd.DataFrame({'A': ['hello', 'world', 'foo', 'bar']})
     pi = piplyr(df)
     pi.str_pad(10, side='right', pad='*')
 ``` 
 

 ## str_sub


 Replaces a pattern in strings in the dataframe with a replacement string

 Args:
     pattern: The pattern to search for in the strings
     replacement: The replacement string


```
  ### Example:
     df = pd.DataFrame({'A': ['hello', 'world', 'foo', 'bar']})
     pi = piplyr(df)
     pi.str_sub('o', '0')
  
 ```

 ## str_extract
  
 Extracts the first occurrence of a pattern from each string in a column
 
 Args:
     pattern: A regular expression pattern to match in the strings
     col: The column name to extract the substring from. If None, apilies to all columns with object dtype.
 
 ```
  ### Example:
     df = pd.DataFrame({'col1': ['abc123', 'def456', 'ghi789']})
     pi = piplyr(df)
     pi.str_extract(pattern='[a-z]+', col='col1')
  
 ```



 ## str_detect
 Check if a string pattern is present in a specific column

 Args:
     col: the name of the column to be searched
     pattern: the string pattern to be searched for

 ```
  ### Example:
     df = pd.DataFrame({'col1': ['abc', 'def', 'ghi', 'jkl']})
     pi = piplyr(df)
     pi.str_detect(col='col1', pattern='a')
  
 ```
    
 ## fct_lump
  
 Lumps levels of a factor column into 'Other' level for levels that apiear less than n times

 Args:
     column: The name of the column containing the factor levels
     n: The number of times a level should apiear to avoid being lumped (default is 10)
     other_level: The name of the new level for the lumped levels (default is 'Other')
 
 ```
  ### Example:
     df = pd.DataFrame({'A': [1, 2, 3, 4], 'B': ['A', 'B', 'C', 'A']})
     pi = piplyr(df)
     pi.fct_lump(column='B', n=2, other_level='Rare')
  
 ```

 ## fct_infreq
  
 Lumps levels of a factor column into 'Other' level for levels that apiear less than a fraction of total

 Args:
     column: The name of the column containing the factor levels
     frac: The fraction of the total number of times a level should apiear to avoid being lumped (default is 0.01)
     other_level: The name of the new level for the lumped levels (default is 'Other')
 
 ```
  ### Example:
     df = pd.DataFrame({'A': [1, 2, 3, 4], 'B': ['A', 'B', 'C', 'A']})
     pi = piplyr(df)
     pi.fct_infreq(column='B', frac=0.5, other_level='Rare')
  
 ```
 ## fct_relevel
  
 Relevels the factor column to put the reference level first

 Args:
     col: The name of the factor column
     ref: The reference level to put first
     after: Whether to put the reference level after (True) or before (False) the other levels (default is True)
 
 ```
  ### Example:
     df = pd.DataFrame({'col1': [1, 2, 3, 4], 'col2': ['A', 'B', 'C', 'D']})
     pi = piplyr(df)
     pi.fct_relevel(col='col2', ref='B', after=False)
  
 ```

    
 ## fct_recode
  
 Recodes factor levels in a given column

 Args:
     col: The column name to be recoded
     values: The old factor levels to be recoded
     labels: The new factor levels (default is None)
     drop: Specifies whether to drop unused levels (default is False)


## Recommened style of writing codes: 
### Example 1
```

df = pd.DataFrame({'x': [1, 2, 3, 4, 5, 6],
 'y': [4, 5, 6, 7, 8, 9],
 'z':['a','b','a','b','a','a']
       })

pi  = piplyr(df)

(
pi.
mutate('x+y','x2').
sql_plyr('SELECT x,x2,y,z, (AVG(x2) over()) as x3 FROM df').
select('x','x2','x3','z').
filter('x > 2')
)

```
### Example 2
```

df = pd.DataFrame({'x': [1, 2, 3, 4, 5, 6],
 'y': [4, 5, 6, 7, 8, 9],
 'z':['a','b','a','b','a','a']
       })


pi  = piplyr(df)
(
pi.
mutate('x+y','x2').
sql_plyr('SELECT x,x2,y,z, (AVG(x2) over()) as x3 FROM df').
select('x','x2','x3','z').
filter('x > 2').
summarize(group_var='z',var='x2',agg_func='mean')
).to_df.reset_index()

```
