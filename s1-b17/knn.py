import tkinter as tk
from tkinter import ttk
import numpy as np
from sklearn.neighbors import KNeighborsRegressor
from tensorflow_estimator.python.estimator.canned.timeseries import model

# Sample data
data = [
    {"Criminal Name": "William Reed", "Crime": "Fraud", "Date": "2022-12-08", "Region": "New Mexico"},
    {"Criminal Name": "Ryan Morris", "Crime": "Homicide", "Date": "2022-03-25", "Region": "Missouri"},
    {"Criminal Name": "Daniel Wells", "Crime": "Burglary", "Date": "2019-10-08", "Region": "Massachusetts"},
    {"Criminal Name": "Michael Hull", "Crime": "Identity Theft", "Date": "2021-09-23", "Region": "Illinois"},
    {"Criminal Name": "Jose Vasquez MD", "Crime": "Embezzlement", "Date": "2019-06-08", "Region": "Florida"},
    {"Criminal Name": "Leah Wells PhD", "Crime": "Drug Possession", "Date": "2020-06-01", "Region": "Maryland"},
    {"Criminal Name": "Kelly Kaufman", "Crime": "Fraud", "Date": "2020-06-30", "Region": "Missouri"},
    {"Criminal Name": "Caroline Montoya", "Crime": "Grand Theft Auto", "Date": "2020-02-03", "Region": "Kentucky"},
    {"Criminal Name": "Robert Bradshaw", "Crime": "Embezzlement", "Date": "2019-04-18", "Region": "Georgia"},
    {"Criminal Name": "Roger French", "Crime": "Embezzlement", "Date": "2023-01-26", "Region": "Florida"},
    {"Criminal Name": "Patricia Mathis", "Crime": "Vandalism", "Date": "2019-01-11", "Region": "Colorado"},
    {"Criminal Name": "Heather Smith", "Crime": "Arson", "Date": "2021-08-21", "Region": "Texas"},
    {"Criminal Name": "Kevin Singh", "Crime": "Fraud", "Date": "2019-10-01", "Region": "Kansas"},
]

# Create a mapping of crime categories to numerical labels
crime_labels = {"Fraud": 1, "Homicide": 2, "Burglary": 3, "Identity Theft": 4, "Embezzlement": 5,
                "Drug Possession": 6, "Grand Theft Auto": 7, "Vandalism": 8, "Arson": 9}

# Create a mapping of regions to numerical labels
region_labels = {"New Mexico": 1, "Missouri": 2, "Massachusetts": 3, "Illinois": 4, "Florida": 5,
                 "Maryland": 6, "Kentucky": 7, "Georgia": 8, "Rhode Island": 9, "Colorado": 10,
                 "Texas": 11, "Kansas": 12}

# Convert data to numerical features and labels
X = []
y = []
for entry in data:
    crime_label = crime_labels.get(entry["Crime"], -1)
    region_label = region_labels.get(entry["Region"], -1)
    if crime_label != -1 and region_label != -1:
        X.append([crime_label, region_label])
        y.append(np.random.uniform(0, 1))

X = np.array(X, dtype=np.float32)
y = np.array(y, dtype=np.float32)

# Create a kNN model
knn_model = KNeighborsRegressor(n_neighbors=3)

# Fit the kNN model
knn_model.fit(X, y)

# Initialize Tkinter
root = tk.Tk()
root.title("Crime Prediction with kNN")

# Rest of the GUI code


def predict_crime_probability():
    # Rest of the predict_crime_probability function
    selected_region = region_var.get()
    selected_crime = crime_var.get()

    results_text.delete("1.0", tk.END)

    results_text.insert(tk.END, f"Predictions for {selected_crime} in {selected_region}:\n\n")

    for entry in data:
        if entry["Region"] == selected_region and entry["Crime"] == selected_crime:
            criminal_name = entry["Criminal Name"]
            past_crimes = np.array([[crime_labels.get(entry["Crime"], -1), region_labels.get(entry["Region"], -1)]],
                                   dtype=np.float32)
            if past_crimes.size > 0:
                prediction = model.predict(past_crimes)
                future_probability = prediction[0][0] * 100
                results_text.insert(tk.END, f"Criminal Name: {criminal_name}\n")
                results_text.insert(tk.END, f"Future Crime Probability: {future_probability:.2f}%\n\n")
            else:
                results_text.insert(tk.END, f"Criminal Name: {criminal_name}\n")
                results_text.insert(tk.END, f"No past crimes data available.\n\n")
        # Use the kNN model to predict
        future_probability = knn_model.predict(
            np.array([[crime_labels.get(selected_crime), region_labels.get(selected_region)]]))
        results_text.insert(tk.END, f"Future Crime Probability (kNN): {future_probability[0]:.2f}%\n\n")


# Rest of the Tkinter GUI code
# Create labels and input fields
region_label = tk.Label(root, text="Select Region:")
region_label.pack()

region_var = tk.StringVar()
region_combobox = ttk.Combobox(root, textvariable=region_var, values=list(region_labels.keys()))
region_combobox.pack()

crime_label = tk.Label(root, text="Select Crime:")
crime_label.pack()

crime_var = tk.StringVar()
crime_combobox = ttk.Combobox(root, textvariable=crime_var, values=list(crime_labels.keys()))
crime_combobox.pack()

predict_button = tk.Button(root, text="Predict", command=predict_crime_probability)
predict_button.pack()

results_text = tk.Text(root, height=10, width=40)
results_text.pack()

root.mainloop()

