import pandas as pd
import matplotlib.pyplot as plt

# Replace 'your_file.csv' with the actual path to your CSV file
csv_file_path = 'reports/figures/Book6.csv'

# Reading data from CSV file
df = pd.read_csv(csv_file_path)

# Visualize the data as a stacked lines with markers chart with vertical x-axis labels
num_rows = len(df)

fig, ax = plt.subplots(figsize=(12, 6))

for i in range(num_rows):
    ax.plot(df.columns, df.iloc[i, :], marker='o', label=f'Row {i + 1}')

ax.set_title('Stacked Lines with Markers Chart')
ax.set_xlabel('Columns')
ax.set_ylabel('Values')

# Rotate x-axis labels
plt.xticks(rotation='vertical')

# Adjust layout to avoid cutting off x-axis labels
plt.tight_layout(rect=[0, 0.1, 1, 1])

plt.show()
