from flask import Flask, render_template, request
import pandas as pd
import pickle
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split

app = Flask(__name__)

# Load and preprocess the dataset
def load_data():
    try:
        df = pd.read_csv("spam.csv", encoding="latin-1")
        print("CSV file loaded successfully.")
        print("Columns in the CSV:", df.columns)
        return df
    except FileNotFoundError:
        print("The 'spam.csv' file was not found. Please ensure it is in the correct directory.")
        return None
    except Exception as e:
        print("An error occurred while loading the CSV file:", e)
        return None

df = load_data()
if df is not None:
    df.drop(['Unnamed: 2', 'Unnamed: 3', 'Unnamed: 4', 'Unnamed: 5'], axis=1, inplace=True, errors='ignore')
    
    # Directly use the columns 'v1' for class and 'v2' for message
    if 'v1' in df.columns and 'v2' in df.columns:
        df['label'] = df['v1'].map({'ham': 0, 'spam': 1})
        X = df['v2']
        y = df['label']
        
        # Extract features with CountVectorizer
        cv = CountVectorizer()
        X = cv.fit_transform(X)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)

        # Train the Naive Bayes Classifier
        clf = MultinomialNB()
        clf.fit(X_train, y_train)

        # Save the model and vectorizer
        with open('NB_spam_model.pkl', 'wb') as model_file:
            pickle.dump(clf, model_file)
        with open('count_vectorizer.pkl', 'wb') as vectorizer_file:
            pickle.dump(cv, vectorizer_file)
    else:
        print("The necessary columns are not found in the CSV file. Please check the file structure.")

@app.route('/')
def home():
    return render_template('front_page.html')

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        try:
            # Load the model and vectorizer
            with open('NB_spam_model.pkl', 'rb') as model_file:
                clf = pickle.load(model_file)
            with open('count_vectorizer.pkl', 'rb') as vectorizer_file:
                cv = pickle.load(vectorizer_file)

            message = request.form['message']
            data = [message]
            vect = cv.transform(data).toarray()
            my_prediction = clf.predict(vect)
            return render_template('page_result.html', prediction=my_prediction)
        except Exception as e:
            print("An error occurred during prediction:", e)
            return "An error occurred during prediction. Please try again."

if __name__ == '__main__':
    app.run(debug=True)
