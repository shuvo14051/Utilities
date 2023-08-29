"""
Display the percent of null vlaues 
and percent of zeros for each column

If a gender column has a value 0 that's nan
Because a gender can only be M or F
"""

import pandas as pd

def check_null_values(df):
  null_values = pd.DataFrame({
      'columns':df.columns,
      'percent_null': df.isnull().mean(),
      'percent_zero': df.isin([0]).mean()
  })

  return null_values
