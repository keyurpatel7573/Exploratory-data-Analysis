import pandas as pd
import seaborn as sns
import scipy 

tips = sns.load_dataset('tips')
flights = sns.load_dataset('flights')
iris = sns.load_dataset('iris')
titanic_data = pd.read_csv('train.csv')
tips.head()
titanic_data.head()
sns.scatterplot(
    data=tips,
    x='total_bill',
    y='tip',
    hue='sex',
    style='smoker'
)
sns.barplot(
    data=titanic_data,
    x= 'Pclass',
    y='Fare',
    hue='Sex',
)
sns.boxplot(
    data=titanic_data,
    x='Sex',
    y='Age',
    hue='Survived'
)
sns.distplot(
    titanic_data[titanic_data['Survived']== 0]['Age'],
    hist = False
)
sns.distplot(
    titanic_data[titanic_data['Survived']== 1]['Age'],
    hist= False
)
sns.heatmap(pd.crosstab(titanic_data['Pclass'],titanic_data['Survived']))
titanic_data.groupby('Pclass')['Survived'].mean()*100

sns.pairplot(iris,hue='species')
new_data = flights.groupby('year')['passengers'].sum().reset_index()
sns.lineplot(
    data=new_data,
    x='year',
    y='passengers'
)
sns.heatmap(flights.pivot_table(values='passengers',index='month', columns= 'year'))