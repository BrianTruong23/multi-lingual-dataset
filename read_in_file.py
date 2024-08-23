import pyarrow.ipc as ipc

file_name = "vie/MMMU/Accounting/dev/data-00000-of-00001.arrow"

# Open the Arrow IPC file and read it into a table
with ipc.open_file(file_name) as file:
    table = file.read_all()

# Convert the table to a Pandas DataFrame
df = table.to_pandas()

print(df.head())
