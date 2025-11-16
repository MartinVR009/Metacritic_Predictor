import pandas as pd

# Cargar CSV
df_games = pd.read_csv("games.csv")
df_movies = pd.read_csv("movies.csv")
df_tv = pd.read_csv("tv.csv")

print("=== Games Data ===")
print(df_games.head())
print(df_games.info())
print("Null values in Games Data:")
print(df_games.isnull().sum())

print("\n=== Movies Data ===")
print(df_movies.head())
print(df_movies.info())
print("Null values in Movies Data:")
print(df_movies.isnull().sum())

print("\n=== TV Data ===")
print(df_tv.head())
print(df_tv.info())
print("Null values in TV Data:")
print(df_tv.isnull().sum())