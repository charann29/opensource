import pandas as pd
import matplotlib.pyplot as plt
df = pd.read_csv('data.csv')
grouped_df = df.groupby('category_column')['value_column'].mean().reset_index()
plt.figure(figsize=(10, 6))
plt.bar(grouped_df['category_column'], grouped_df['value_column'], color='skyblue')
plt.xlabel('Category')
plt.ylabel('Average Value')
plt.title('Average Value by Category')
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig('average_value_by_category.png')
plt.show()
