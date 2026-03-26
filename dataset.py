import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler, StandardScaler

 
url = "https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv"
df = pd.read_csv(url)

 
print("--- Aperçu des données ---")
print(df.head())  
print("\n--- Types de données ---")
print(df.dtypes) 
print("\n--- Valeurs manquantes par colonne ---")
print(df.isnull().sum()) 





df['Age'] = df['Age'].fillna(df['Age'].median())

 
df['Embarked'] = df['Embarked'].fillna(df['Embarked'].mode()[0])

 
if df['Cabin'].isnull().sum() / len(df) > 0.4:
    df.drop(columns=['Cabin'], inplace=True)

 

plt.figure(figsize=(6, 4))
sns.boxplot(x=df['Fare'])
plt.title("Détection des valeurs aberrantes (Fare)")
plt.show()


Q1 = df['Fare'].quantile(0.25)
Q3 = df['Fare'].quantile(0.75)
IQR = Q3 - Q1
limite_inf = Q1 - 1.5 * IQR
limite_sup = Q3 + 1.5 * IQR

df.loc[(df['Fare'] < limite_inf) | (df['Fare'] > limite_sup), 'Fare'] = df['Fare'].median()




 
df = pd.get_dummies(df, columns=['Sex', 'Embarked'], drop_first=True)


scaler_minmax = MinMaxScaler() 
scaler_std = StandardScaler()   

df['Age_MinMax'] = scaler_minmax.fit_transform(df[['Age']])
df['Age_Zscore'] = scaler_std.fit_transform(df[['Age']])


fig, axes = plt.subplots(1, 3, figsize=(15, 5))
sns.histplot(df['Age'], kde=True, ax=axes[0]).set_title("Original (Age)")
sns.histplot(df['Age_MinMax'], kde=True, ax=axes[1]).set_title("Normalisation (Min-Max)")
sns.histplot(df['Age_Zscore'], kde=True, ax=axes[2]).set_title("Standardisation (Z-score)")
plt.show()




print("\n--- Déséquilibre de la variable cible (Survived) ---")
print(df['Survived'].value_counts())

 

df_majoritaire = df[df['Survived'] == 0]
df_minoritaire = df[df['Survived'] == 1]


df_minoritaire_upsampled = df_minoritaire.sample(n=len(df_majoritaire), replace=True, random_state=42)


df_equilibre = pd.concat([df_majoritaire, df_minoritaire_upsampled])

print("\n--- Résultat Final (Après Oversampling manuel) ---")
print(f"Distribution des classes :\n{df_equilibre['Survived'].value_counts()}")