import pandas as pd

# Load your dataset
data = pd.read_csv(r'C:\Users\arige\Documents\PPTS\FILES\field project\yes\final1\parkinsons.data')

# Print columns to verify names
print("Columns in dataset:", data.columns)

# Adjust columns based on your dataset
# Replace with the actual column names your model expects
columns_needed = ['MDVP:Fo(Hz)', 'MDVP:Fhi(Hz)', 'MDVP:Flo(Hz)', 'MDVP:Jitter(%)',
                  'MDVP:Jitter(Abs)', 'MDVP:RAP', 'MDVP:PPQ', 'Jitter:DDP',
                  'MDVP:Shimmer', 'MDVP:Shimmer(dB)', 'Shimmer:APQ3', 'Shimmer:APQ5',
                  'MDVP:APQ', 'Shimmer:DDA', 'NHR', 'HNR', 'RPDE', 'DFA',
                  'spread1', 'spread2', 'D2', 'PPE']

batch_data = data[columns_needed]  # Adjust columns as per your dataset

# Save to CSV file
batch_data.to_csv(r'C:\Users\arige\Documents\PPTS\FILES\field project\yes\FINAL1\batch_data.csv', index=False)

# Display information about the batch data
print("Batch data saved successfully.")
print("Sample data:")
print(batch_data.head())
