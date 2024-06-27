#Introduction to Python 

# We import the libraries that we're going to use for the analysis.


import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from scipy.stats import chi2_contingency
import plotly.express as px

# The data base was extracted from the following link 
# https://archive.ics.uci.edu/dataset/544/estimation+of+obesity+levels+based+on+eating+habits+and+physical+condition
# We read it with pandas with the commnad pd.read_csv

df=pd.read_csv('ObesityDataSet_raw_and_data_sinthetic.csv')

# And print it 
df

# We print the description of the dataframe 
summary_stats = df.describe()
summary_stats

# With this line we check if there is any NA value in the data set
# In this case there are no missing values.
na_counts_df = df.isna().sum()
print("NA values in each column:")
#df_item_copy.shape
print(na_counts_df)


# We check the types of the information we have.
df_types=df.dtypes
df_types


# Plot the pie chart
 
plt.figure(figsize=(4, 4)) # We use plt.figure to specify the size of the figure
gender_counts.plot(kind='pie', autopct='%1.1f%%', startangle=90) # Plot the 'gender_counts' 
#object as a pie chart, with percentage format '%1.1f%%'
plt.title('Distribution of Orders by Time Interval') # We set the title of the plot
plt.ylabel('') # Remove the y-axis label
plt.show() # Display the plot


df.hist(figsize=(15, 10), bins=20) # Use the .hist() function to plot histograms 
#for all the numerical columns in our dataframe called 'df', specifying the size and bin intervals.
plt.show() #We display the plot


# With this first line we define a list called 'categorical_columns'
# containing all 'object' type columns from df
categorical_columns = ['Gender', 'family_history_with_overweight', 'FAVC', 'CAEC', 'SMOKE', 'SCC', 'CALC', 'MTRANS', 'NObeyesdad']

# We create a dictionary 'value_counts' which will help us to count
# all the ocurrences for each column in 'categorical_columns'
value_counts = {col: df[col].value_counts() for col in categorical_columns}

# The command FOR will help us to iterate through each columns, 'col' 
# in our list 'categorical_columns' this will display 9 plots.
for col in categorical_columns:
    plt.figure(figsize=(8, 4))# We set the size of the plot
    ax = sns.countplot(data=df, x=col) # We creat a bar plot specifying information 'df', values in x axis 'col'
    total = len(df[col]) # We calculate the number of entries in the column 'col'
 

    for p in ax.patches: # We use this for to iterate each bar(p), which represnts a category count and percentage
        height = p.get_height()# The command .get_height() represents the count of occurrences per category 'p'
        #We use the function '.annotate()' to put the % labels with only 1 decimal and in a specific position.
        ax.annotate(f'{height/total:.1%}', (p.get_x() + p.get_width() / 2., height),
                    ha='center', va='bottom')
    
    plt.xticks(rotation=45) # We rotate the x axis labels 45º for a better readability
    plt.show() # We display the plot


sns.pairplot(df)# We create a grid with the commnad .pairplot() *There's a similar option with pandas = pd.plotting.scatter_matrix(df,figsize=(12,8))
# we use seaborn (sns) for a better visualization.
plt.show() # We display the plot
correlation_matrix = df.corr() # We do a corralation between each column
plt.figure(figsize=(12, 8)) # We set the size of the plot
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm') # To better visualize the correlation of the columns we use a heat map.
# using the dara from 'correlation_matrix'
plt.show() # We dipslay the heatmap



# We create an interactive box plot using px.box(), selecting the data for comparison.
# We define the x and y axes, and specify the title of the plot.
fig = px.box(df, x='NObeyesdad', y='Age', title='Age Distribution Across NObeyesdad Categories')

fig.show() #We display the plot

# We use the function px.histogram from Plotly to create a histogram. Then we define the x-axis values, in this case, 'Age'.
# We want to see the comparison in 2 columns, so we use facet_col='Gender' to create 2 columns.
# Finally, we specify the rows with facet_row='NObeyesdad'.
fig = px.histogram(df, x='Age', facet_col='Gender', facet_row='NObeyesdad', category_orders={'Gender': ['Male', 'Female'], 'NObeyesdad': sorted(df['NObeyesdad'].unique())}, height=800)


fig.update_layout(
    title_text='Age Distribution by Gender and NObeyesdad',#We set the Plot title
    margin=dict(t=50, l=50, r=1000, b=50),  # We set space for labels
    height=1800  # And set the heightfor facets
)

fig.update_xaxes(title_text='Age') # We set x axis title
fig.update_yaxes(title_text='Count') # We set y axis title

fig.update_layout( # The following lines of core will give the size of the plot 
    autosize=False, 
    margin=dict(
        l=.01,
        r=.01,  
        b=100,
        t=100,
        pad=.01
    ),
    showlegend=False
)

fig.show()  # We diplsay the interactive plot.
# If the user hovers the mouse over the histograms, more precise information will be displayed.


# We create a contingency table with the pandas function pd.crosstab()
# and we select the combination values
contingency_table = pd.crosstab(df['family_history_with_overweight'], df['NObeyesdad'])

# From scipy.stats we use the chi2_contingency() function
# this will give us back the values of chi2, which measures the discrepancy between the variables,
# p value, which tells us how significant the relationship is,
# dof, which provides the number of values in the calculation that are free to vary,
# and expected, which gives the frequencies we would expect if there was no association between the variables.
chi2, p, dof, expected = chi2_contingency(contingency_table)


#These 2 lines will display the value of the chi2 and p which will be described in the article.
print(f"Chi-Square Test Statistic: {chi2}") 
print(f"P-Value: {p}")


plt.figure(figsize=(10, 6)) # We specify the plot size
sns.heatmap(contingency_table, annot=True, fmt='d', cmap='YlGnBu') # We create a heatmap from the contingency_table data
# The annot parameter gives the count values, while the fmt parameter ensures the values are displayed as integers.
plt.title('Contingency Table Heatmap of family_history_with_overweight vs NObeyesdad') # We set the plot title
plt.show() # We display the heatmap
