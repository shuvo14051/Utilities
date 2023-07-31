from scipy.stats import chi2_contingency

def chi_square(column):
    data = df_6[[column, 'label']]    
    contingency_table = pd.crosstab(data['label'], data[column])
    chi2_stat, p_value, dof, expected = chi2_contingency(contingency_table)
    print("Chi-square Statistic:", chi2_stat)
    print("P-value:", p_value)
    
    alpha = 0.05
    
    if p_value < alpha:
        print('There is a significant difference "{}"' .format(column))
    else:
        print('There is no significant difference "{}"' .format(column))
