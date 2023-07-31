from scipy.stats import f_oneway

def anova_test(column):
    # select the feature and the label
    data = df_6[[column, 'label']]    
    grouped_data = data.groupby('label')[column]

    # Depends on the number of clusters we have
    cluster1_age = grouped_data.get_group(0)
    cluster2_age = grouped_data.get_group(1)
    cluster3_age = grouped_data.get_group(2)
    cluster4_age = grouped_data.get_group(3)
    cluster5_age = grouped_data.get_group(4)
    cluster6_age = grouped_data.get_group(5)
    
    f_statistic, p_value = f_oneway(cluster1_age, cluster2_age, cluster3_age, cluster4_age) 
    
    print("F-Statistic:", f_statistic)
    print("P-value:", p_value)
    
    alpha = 0.05
    
    if p_value < alpha:
        print('There is a significant difference in the mean "{}"' .format(column))
    else:
        print('There is no significant difference in the mean "{}"' .format(column))
