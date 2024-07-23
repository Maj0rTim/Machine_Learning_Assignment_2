import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.model_selection import train_test_split

plt.style.use('seaborn-darkgrid')

# Note that Best Practices, in terms of order of steps were violated in this program, please fix that.

# # Data Acquisition (I broke the CSV on purpose)
wine_df = pd.read_csv('winequality-red.csv')
with pd.option_context('display.max_rows', None, 'display.max_columns', None):
    print(wine_df.head(3))

X = wine_df.drop('quality', axis=1).values
y = np.ravel(wine_df[['quality']])

# # Cleanup

# # Exploratory Data Analysis
# with pd.option_context('display.max_rows', None, 'display.max_columns', None):
    # print(wine_df.describe())
# sns.pairplot(wine_df, hue = 'quality', height = 3, palette="husl")
# sns.violinplot(data=wine_df, x='quality', y='alcohol')
sns.FacetGrid(wine_df, hue='quality', height=6).map(plt.scatter, 'alcohol', 'fixed acidity').add_legend()
plt.show()


# ### Distribution of wine quality (target variable)
plt.hist(wine_df['quality'], bins=6, edgecolor='black')
plt.xlabel('quality', fontsize=20)
plt.ylabel('count', fontsize=20)
plt.xticks(fontsize=15)
plt.yticks(fontsize=15)
plt.show()

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)