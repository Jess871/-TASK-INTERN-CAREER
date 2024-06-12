#!/usr/bin/env python
# coding: utf-8

# # Task 1

# ## Date: 26 May 2024

# ### Name: Jessica Bawden

# ### Email: jessbawden987@gmail.com

# ## Task 1: YouTube Streamer Analysis

# ### Dataset: Top 1000 YouTuber Statistics

# Description: This dataset contains valuableinformation about the top YouTube streamers,including their ranking, categories, subscribers,country, visits, likes, comments, and more.Your task is to perform a comprehensiveanalysis of the dataset to extract insightsabout the top YouTube content creators.

# ## Import Libraries

# In[57]:


import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt


# ### Importing Data

# In[58]:


df = pd.read_csv(r"C:\Users\jessb\Downloads\youtubers_df.csv")


# ### 1) Exploring data

# In[59]:


print(df.head())


# In[60]:


df.info() # Getting information about the dataset


# #### Check for misssing data

# In[61]:


df.isnull().sum()


# In[62]:


df['Categories'].unique()


# In[63]:


# Handle the missing data
df.fillna('Desconocida', inplace=True) #Desconocide means unknown in Spanish
#Check the output
df.isnull().sum()


# In[64]:


# Correting the column name 'Suscribers' to 'Subsribers'
df.rename(columns={'Suscribers':'Subscribers'},inplace=True)
df.columns #Checking columns


# In[65]:


#Checking for duplicates
df.duplicated().sum()


# #### Observation from the information gained about the dataset:

# The DataFrame provided contains data about YouTubers. Here are some initial observations:
# 
# 1. Entries: There are 1000 entries (rows) numbered from 0 to 999.
# 2. Columns: The dataset consists of 9 columns with the following information:
# - Rank: Rank of the YouTuber with no missing values.
# - Username: Username of the YouTuber with no missing values.
# - Categories: Categories of the YouTuber with 306 missing values replaced with "unknown".
# - Subscribers: Number of subscribers of the YouTuber with no missing values.
# - Country: Country of the YouTuber with no missing values.
# - Visits: Number of visits to the YouTuber's channel with no missing values.
# - Likes: Number of likes of the YouTuber with no missing values.
# - Comments: Number of comments of the YouTuber with no missing values.
# - Links: YouTube link of the YouTuber.

# In[66]:


# First 10 in the dataset
df.head(10)


# In[67]:


# Summary statistics for the numeric columns in the dataset
df.describe()


# In[68]:


# Displaying the summary stats for the object columns in the dataset
df.describe(include="object")


# ### 2) Trend Analysis

# In[69]:


# Most popular categories
most_popular_categories = df['Categories'].value_counts()
most_popular_categories


# In[70]:


# Graphical representation of most popular categories
plt.figure(figsize=(18,10))
sns.barplot(x=most_popular_categories.index, y=most_popular_categories.values, palette='viridis')
plt.title('Most Popular Categories')
plt.xlabel('Categories')
plt.ylabel('Number of Streamers')
plt.xticks(rotation = 90)
plt.show()


# #### Correlation between Subsribers and the number of Likes/Comments

# In[71]:


# Subscribers vs Likes
subscribers_likes_corr = df['Subscribers'].corr(df['Likes'])
print('Correlation between Subscribers and Likes:', subscribers_likes_corr)

#Subscribers vs Comments
subscribers_comments_corr = df['Subscribers'].corr(df['Comments'])
print('Correlation between Subscribers and Comments:', subscribers_comments_corr)


# ##### Subscribers vs. Likes: The correlation coefficient is approximately 0.2116, indicating a weak positive correlation.

# ##### Subscribers vs. Comments: The correlation coefficient is approximately 0.0363, indicating a very weak positive correlation.

# ### 3) Audience Study

# In[72]:


country_counts = df['Country'].value_counts()
print(country_counts)


# In[73]:


# Calculate the total count per country and sort in descending order
country_order = df['Country'].value_counts().index
# Geographical distribution
plt.figure(figsize=(14, 8))
sns.countplot(x='Country', data=df, order=country_order, palette='viridis')
plt.title('Distribution of Streamers by Countries')
plt.ylabel('Number of Streamers')
plt.xlabel('Country')
plt.xticks(rotation=90, ha='right')  # Rotate x-axis labels for better readability
plt.tight_layout()
plt.show()


# In[74]:


# Create a pivot table to count the number of streamers in each country-category combination
pivot_table = df.pivot_table(index='Country', columns='Categories', aggfunc='size', fill_value=0)

# Create a heatmap
plt.figure(figsize=(12, 8))
sns.heatmap(pivot_table, cmap='YlGnBu', annot=True, fmt='d', linewidths=0.5)
plt.title('Regional Preferences for Content Categories')
plt.xlabel('Content Categories')
plt.ylabel('Country')
plt.xticks(rotation=90)  # Rotate x-axis labels for better readability
plt.tight_layout()
plt.show()


# ### 4) Performance Metrics

# #### Average number of subsribers, visits, likes, and comments

# In[75]:


average_metrics = df[['Subscribers', 'Visits', 'Likes', 'Comments']].mean()
colors = sns.color_palette("viridis", 5)
plt.figure(figsize=(10, 6), facecolor='white')
average_metrics.plot(kind='bar', color=colors)
plt.title('Average Performance Metrics')
plt.ylabel('Average Number')
plt.xlabel('Metrics')
plt.xticks(rotation=45)
plt.grid(True, linestyle='--', alpha=0.6)
plt.show()


# In[76]:


# Patterns and anomalies
category_metrics = df.groupby('Categories')[['Subscribers', 'Visits', 'Likes', 'Comments']].mean().sort_values(
    by='Subscribers', ascending=False)

plt.figure(figsize=(14, 8), facecolor='white')
sns.heatmap(category_metrics, annot=True, fmt='.0f', cmap='coolwarm', linewidths=.5)
plt.title('Heatmap of Average Performance Metrics by Category')
plt.ylabel('Categories')
plt.xlabel('Performance Metrics')
plt.show()


# ### 5) Content Categories

# #### Distriution of content categories

# In[77]:


category_counts = df['Categories'].value_counts()
print(category_counts)

#Plotting
colors = sns.color_palette("viridis")

plt.figure(figsize=(12, 8), facecolor='white')
category_counts.plot(kind='bar', color=colors)
plt.title('Distribution of Categories')
plt.ylabel('Number of Streamers')
plt.xlabel('Content Categories')
plt.xticks(rotation=90)
plt.grid(True, linestyle='--', alpha=0.6)
plt.show()


# In[78]:


# Top 10 Categories
category_counts = df['Categories'].value_counts()
print(category_counts.head(10))

colors = sns.color_palette("viridis")

plt.figure(figsize=(12, 8), facecolor='white')
category_counts.head(10).plot(kind='bar', color=colors)
plt.title('Top 10 Content Categories by Number of Streamers')
plt.ylabel('Number of Streamers')
plt.xlabel('Content Categories')
plt.xticks(rotation=45)
plt.grid(True, linestyle='--', alpha=0.6)
plt.show()


# #### Categories with exception performance metrics

# In[79]:


# Group by 'Categories' and aggregate the metrics
category_metrics = df.groupby('Categories').agg({
    'Subscribers': 'sum',
    'Visits': 'sum',
    'Likes': 'sum',
    'Comments': 'sum'
}).reset_index()

# Filtering the exceptional metrics
exceptional_metrics = category_metrics[
    (category_metrics['Subscribers'] > category_metrics['Subscribers'].quantile(0.75)) & 
    (category_metrics['Visits'] > category_metrics['Visits'].quantile(0.75)) & 
    (category_metrics['Likes'] > category_metrics['Likes'].quantile(0.75)) & 
    (category_metrics['Comments'] > category_metrics['Comments'].quantile(0.75))
]

# Print the exceptional metrics
print(exceptional_metrics)


# ### 6) Brand and Collaborations

# In[80]:


# Define thresholds for high and low performance based on quartiles
high_subscribers = df['Subscribers'].quantile(0.75, interpolation='nearest')
high_likes = df['Likes'].quantile(0.75, interpolation='nearest')
high_comments = df['Comments'].quantile(0.75, interpolation='nearest')

low_subscribers = df['Subscribers'].quantile(0.25, interpolation='nearest')
low_likes = df['Likes'].quantile(0.25, interpolation='nearest')
low_comments = df['Comments'].quantile(0.25, interpolation='nearest')

# Categorize high and low performance
df['HighPerformance'] = ((df['Subscribers'] >= high_subscribers) & (df['Likes'] >= high_likes) & (df['Comments'] >= high_comments))
df['LowPerformance'] = ((df['Subscribers'] < low_subscribers) | (df['Likes'] < low_likes) | (df['Comments'] < low_comments))

# Calculate engagement
df['Engagement'] = df['Likes'] + df['Comments']
high_performance = df[df['HighPerformance']]['Engagement'].sum()
low_performance = df[df['LowPerformance']]['Engagement'].sum()

# Barplot of High Performance and Low Performance
plt.figure(figsize=(10,5))
sns.barplot(x=['High Performance', 'Low Performance'], y=[high_performance, low_performance], palette='viridis')
plt.title('Comparison of Engagement between High and Low Performing Streamers')
plt.xlabel('Performance')
plt.ylabel('Total Engagement')
plt.show()


# In[81]:


df.head()


# In[82]:


# Remove 'Links' column from the dataset
df.drop(columns=["Links"], inplace=True)
print(df.head())


# ### 7) Benchmarking

# #### Streamers with above average metrics in terms of subscribers, visits, likes and comments

# In[83]:


above_average = df[(df['Subscribers'] > df['Subscribers'].mean()) & (df['Visits'] > df['Visits'].mean()) & (
            df['Likes'] > df['Likes'].mean()) & (df['Comments'] > df['Comments'].mean())]
print(above_average[['Username', 'Subscribers', 'Visits', 'Likes', 'Comments']])



# #### Top performing content creators

# In[84]:


df['Total_Engagement'] = df['Subscribers'] + df['Visits'] + df['Likes'] + df['Comments']
top_performers = df.nlargest(5, 'Total_Engagement')[
    ['Username', 'Subscribers', 'Visits', 'Likes', 'Comments', 'Total_Engagement']]
print('\n The top performers are: ', '\n', top_performers)


# In[85]:


# Visualization
plt.figure(figsize=(14, 8), facecolor='white')
top_performers.set_index('Username')[['Subscribers', 'Visits', 'Likes', 'Comments']].plot(kind='bar', stacked=True, colormap='viridis')
plt.title('Top 5 Performers by Total Engagement')
plt.xlabel('Username')
plt.ylabel('Engagement Metrics')
plt.xticks(rotation=45, ha='right')  # Rotate x-axis labels for better readability
plt.legend(title='Engagement Metrics', bbox_to_anchor=(1.05, 1), loc='upper left')
plt.tight_layout()
plt.grid(True, linestyle='--', alpha=0.6)
plt.show()


# ### 8) Content Recommendations

# In[86]:


# Creating the user-item matrix based on Subscribers in different content categories
user_item_matrix = df.pivot_table(index='Username', columns='Categories', values='Subscribers', fill_value=0)

# Normalize the user-item matrix
normalized_matrix = user_item_matrix.div(np.linalg.norm(user_item_matrix, axis=1), axis=0)


# In[87]:


# Calculate cosine similarity using matrix multiplication
cosine_sim = normalized_matrix @ normalized_matrix.T

def get_recommendations(username, cosine_sim=cosine_sim):
    sim_scores = cosine_sim.loc[username].sort_values(ascending=False)
    sim_scores = sim_scores.iloc[1:16] # Considering top 15 similar streamers
    return sim_scores.index

recommend_streamers = get_recommendations('tseries')
recommend_streamers


# In[ ]:




