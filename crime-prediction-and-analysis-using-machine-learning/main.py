import numpy as np
import tensorflow as tf
import tkinter as tk
from tkinter import ttk

data = [
    {"Criminal Name": "Nilesh", "Crime": "Fraud", "Date": "2022-12-08", "Region": "Mumbai"},
    {"Criminal Name": "Rishi", "Crime": "Homicide", "Date": "2022-11-20", "Region": "Mumbai"},
    {"Criminal Name": "Anil", "Crime": "Burglary", "Date": "2022-09-08", "Region": "Mumbai"},
    {"Criminal Name": "Suresh", "Crime": "Identity Theft", "Date": "2022-08-15", "Region": "Mumbai"},
    {"Criminal Name": "Ashraf", "Crime": "Embezzlement", "Date": "2022-07-10", "Region": "Mumbai"},
    {"Criminal Name": "Rupesh", "Crime": "Drug Possession", "Date": "2022-06-25", "Region": "Mumbai"},
    {"Criminal Name": "Ganesh", "Crime": "Grand Theft Auto", "Date": "2022-05-18", "Region": "Mumbai"},
    {"Criminal Name": "Rahul", "Crime": "Arson", "Date": "2022-04-30", "Region": "Mumbai"},
    {"Criminal Name": "Deepak", "Crime": "Vandalism", "Date": "2022-03-22", "Region": "Mumbai"},
    {"Criminal Name": "Arjun", "Crime": "Homicide", "Date": "2022-02-14", "Region": "Hyderabad"},
    {"Criminal Name": "Kamal", "Crime": "Burglary", "Date": "2021-11-01", "Region": "Delhi"},
    {"Criminal Name": "Rajesh", "Crime": "Fraud", "Date": "2020-10-01", "Region": "Chennai"},
    {"Criminal Name": "Amit", "Crime": "Arson", "Date": "2019-05-15", "Region": "Bengaluru"},
    {"Criminal Name": "Priya", "Crime": "Drug Possession", "Date": "2019-06-08", "Region": "Kolkata"},
    {"Criminal Name": "Ravi", "Crime": "Embezzlement", "Date": "2021-06-01", "Region": "Pune"},
    {"Criminal Name": "Meena", "Crime": "Grand Theft Auto", "Date": "2020-07-30", "Region": "Ahmedabad"},
    {"Criminal Name": "Vijay", "Crime": "Vandalism", "Date": "2020-12-15", "Region": "Surat"},
    {"Criminal Name": "Neha", "Crime": "Fraud", "Date": "2019-10-11", "Region": "Jaipur"},
    {"Criminal Name": "Vikas", "Crime": "Identity Theft", "Date": "2022-05-18", "Region": "Mumbai"},
    {"Criminal Name": "Anita", "Crime": "Drug Possession", "Date": "2021-03-12", "Region": "Delhi"},
    {"Criminal Name": "Kiran", "Crime": "Burglary", "Date": "2020-08-14", "Region": "Chennai"},
    {"Criminal Name": "Ramesh", "Crime": "Homicide", "Date": "2019-09-21", "Region": "Hyderabad"},
    {"Criminal Name": "Sunita", "Crime": "Embezzlement", "Date": "2021-04-05", "Region": "Bengaluru"},
    {"Criminal Name": "Manoj", "Crime": "Grand Theft Auto", "Date": "2022-02-11", "Region": "Kolkata"},
    {"Criminal Name": "Pankaj", "Crime": "Fraud", "Date": "2022-12-08", "Region": "Delhi"},
    {"Criminal Name": "Vikram", "Crime": "Homicide", "Date": "2022-11-20", "Region": "Delhi"},
    {"Criminal Name": "Sanjay", "Crime": "Burglary", "Date": "2022-09-08", "Region": "Delhi"},
    {"Criminal Name": "Yogesh", "Crime": "Identity Theft", "Date": "2022-08-15", "Region": "Delhi"},
    {"Criminal Name": "Farhan", "Crime": "Embezzlement", "Date": "2022-07-10", "Region": "Delhi"},
    {"Criminal Name": "Raghav", "Crime": "Drug Possession", "Date": "2022-06-25", "Region": "Delhi"},
    {"Criminal Name": "Shiv", "Crime": "Grand Theft Auto", "Date": "2022-05-18", "Region": "Delhi"},
    {"Criminal Name": "Kartik", "Crime": "Arson", "Date": "2022-04-30", "Region": "Delhi"},
    {"Criminal Name": "Harsh", "Crime": "Fraud", "Date": "2022-12-08", "Region": "Chennai"},
    {"Criminal Name": "Mohan", "Crime": "Homicide", "Date": "2022-11-20", "Region": "Chennai"},
    {"Criminal Name": "Rohit", "Crime": "Burglary", "Date": "2022-09-08", "Region": "Chennai"},
    {"Criminal Name": "Jatin", "Crime": "Identity Theft", "Date": "2022-08-15", "Region": "Chennai"},
    {"Criminal Name": "Arif", "Crime": "Embezzlement", "Date": "2022-07-10", "Region": "Chennai"},
    {"Criminal Name": "Lokesh", "Crime": "Drug Possession", "Date": "2022-06-25", "Region": "Chennai"},
    {"Criminal Name": "Nitin", "Crime": "Grand Theft Auto", "Date": "2022-05-18", "Region": "Chennai"},
    {"Criminal Name": "Siddharth", "Crime": "Arson", "Date": "2022-04-30", "Region": "Chennai"},
    {"Criminal Name": "Deepa", "Crime": "Vandalism", "Date": "2020-01-25", "Region": "Pune"},
    {"Criminal Name": "Aakash", "Crime": "Fraud", "Date": "2022-12-08", "Region": "Bengaluru"},
    {"Criminal Name": "Rohit", "Crime": "Homicide", "Date": "2022-11-20", "Region": "Bengaluru"},
    {"Criminal Name": "Sandeep", "Crime": "Burglary", "Date": "2022-09-08", "Region": "Bengaluru"},
    {"Criminal Name": "Vineet", "Crime": "Identity Theft", "Date": "2022-08-15", "Region": "Bengaluru"},
    {"Criminal Name": "Arjun", "Crime": "Embezzlement", "Date": "2022-07-10", "Region": "Bengaluru"},
    {"Criminal Name": "Puneet", "Crime": "Drug Possession", "Date": "2022-06-25", "Region": "Bengaluru"},
    {"Criminal Name": "Rajat", "Crime": "Grand Theft Auto", "Date": "2022-05-18", "Region": "Bengaluru"},
    {"Criminal Name": "Varun", "Crime": "Arson", "Date": "2022-04-30", "Region": "Hyderabad"},
    {"Criminal Name": "Jaya", "Crime": "Vandalism", "Date": "2020-01-25", "Region": "Pune"},
    {"Criminal Name": "Shreya", "Crime": "Fraud", "Date": "2022-12-08", "Region": "Hyderabad"},
    {"Criminal Name": "Rachit", "Crime": "Homicide", "Date": "2022-11-20", "Region": "Hyderabad"},
    {"Criminal Name": "Nirav", "Crime": "Burglary", "Date": "2022-09-08", "Region": "Hyderabad"},
    {"Criminal Name": "Tarun", "Crime": "Identity Theft", "Date": "2022-08-15", "Region": "Hyderabad"},
    {"Criminal Name": "Irfan", "Crime": "Embezzlement", "Date": "2022-07-10", "Region": "Hyderabad"},
    {"Criminal Name": "Yusuf", "Crime": "Drug Possession", "Date": "2022-06-25", "Region": "Hyderabad"},

]

# Create a mapping of crime categories to numerical labels
crime_labels = {"Fraud": 1, "Homicide": 2, "Burglary": 3, "Identity Theft": 4, "Embezzlement": 5,
                "Drug Possession": 6, "Grand Theft Auto": 7, "Vandalism": 8, "Arson": 9}

# Create a mapping of Indian cities to numerical labels
region_labels = {"Mumbai": 1, "Hyderabad": 2, "Delhi": 3, "Chennai": 4, "Bengaluru": 5,
                 "Kolkata": 6, "Pune": 7, "Ahmedabad": 8,}

X = []
y = []
for entry in data:
    crime_label = crime_labels.get(entry["Crime"], -1)
    region_label = region_labels.get(entry["Region"], -1)
    if crime_label != -1 and region_label != -1:
        X.append([crime_label, region_label])
        # For demonstration purposes, we'll assign a random crime probability
        y.append(np.random.uniform(0, 1))

X = np.array(X, dtype=np.float32)
y = np.array(y, dtype=np.float32)

# Create a neural network model using TensorFlow
model = tf.keras.Sequential([
    tf.keras.layers.Dense(64, activation="relu", input_shape=(2,)),
    tf.keras.layers.Dense(32, activation="relu"),
    tf.keras.layers.Dense(1, activation="sigmoid")
])

# Compile the model
model.compile(optimizer="adam", loss="mean_squared_error")

# Train the model
model.fit(X, y, epochs=100, batch_size=1)

# Function to predict crime probability
def predict_crime_probability():
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


root = tk.Tk()
root.title("Crime Prediction")


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
