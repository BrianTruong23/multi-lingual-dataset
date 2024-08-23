import pandas as pd
import pyarrow.parquet as pq

file_name = "data-00000-of-00001.arrow"

# Read the Arrow file into a Pandas DataFrame
table = pq.read_table(file_name)
df = table.to_pandas()

print(df.head())
