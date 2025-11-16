import duckdb
import pandas as pd

# Load CSV
df = pd.read_csv("games.csv")

# Clean/normalize columns: parse dates and convert numeric columns
df["release_date"] = pd.to_datetime(df["release_date"], errors="coerce")
df["metascore"] = pd.to_numeric(df["metascore"], errors="coerce")
df["user_score"] = pd.to_numeric(df["user_score"], errors="coerce")

# Create a DuckDB connection and register the DataFrame as a table named 'df'
con = duckdb.connect()
con.register("df", df)

# Run SQL against the registered DataFrame. Use DATE literal for clarity.
result = con.execute("""
    SELECT platform,
           AVG(metascore) AS avg_meta,
           AVG(user_score) AS avg_user
    FROM df
    WHERE release_date >= DATE '2015-01-01'
    GROUP BY platform
    ORDER BY avg_meta DESC
""").fetchdf()

print(result.head())
