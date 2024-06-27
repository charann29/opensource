# Crime Prediction Using Machine Learning

## Project Description

The crime prediction project leverages machine learning techniques to predict the likelihood of criminal activities in different geographical areas and times. By analyzing historical crime data, the system can identify patterns and trends, helping law enforcement agencies allocate resources more efficiently and potentially prevent crimes before they occur.

## Objectives

1. **Data Collection**: Gather historical crime data from various sources such as police departments, government databases, and public records.
2. **Data Preprocessing**: Clean and preprocess the data to handle missing values, outliers, and inconsistencies. Transform the data into a suitable format for analysis.
3. **Feature Selection**: Identify and select relevant features that influence crime rates, such as location, time of day, day of the week, weather conditions, socio-economic factors, and more.
4. **Model Selection**: Choose appropriate machine learning algorithms to train the prediction models.
5. **Model Training**: Train the models using the preprocessed data and selected features.
6. **Model Evaluation**: Evaluate the models using appropriate metrics (e.g., accuracy, precision, recall, F1-score) to determine their performance.
7. **Deployment**: Deploy the best-performing model to a production environment where it can make real-time predictions.
8. **Visualization**: Develop dashboards and visualizations to present the predictions and insights to stakeholders.

## Algorithms Used

- **Logistic Regression**
- **Decision Trees**
- **Random Forests**
- **Support Vector Machines (SVM)**
- **K-Nearest Neighbors (KNN)**
- **Naive Bayes**
- **Neural Networks**
- **Gradient Boosting Machines (GBM)**
- **XGBoost**

## Libraries Used

- **Pandas**: Data manipulation and analysis
- **NumPy**: Numerical computing
- **Scikit-learn**: Machine learning algorithms
- **XGBoost**: Extreme Gradient Boosting
- **Matplotlib**: Data visualization
- **Seaborn**: Statistical data visualization
- **TensorFlow/Keras**: Neural networks
- **Flask/Django**: Web framework for deployment
- **Dash/Plotly**: Interactive web-based data visualizations


## How to Run the Project

1. **Clone the repository**:
    ```sh
    git clone https://github.com/yourusername/crime-prediction.git
    cd crime-prediction
    ```

2. **Install the required libraries**:
    ```sh
    pip install -r requirements.txt
    ```

3. **Prepare the data**:
    - Place your raw data files in the `data/raw/` directory.
    - Run the data preprocessing script to clean and prepare the data:
      ```sh
      python src/data_preprocessing.py
      ```

4. **Perform exploratory data analysis**:
    - Open and run the `notebooks/exploratory_data_analysis.ipynb` notebook to analyze the data and visualize trends.

5. **Train the models**:
    - Run the model training script to train various machine learning models:
      ```sh
      python src/model_training.py
      ```

6. **Evaluate the models**:
    - Open and run the `notebooks/model_evaluation.ipynb` notebook to evaluate the performance of the trained models.

7. **Deploy the best model**:
    - Run the deployment script to start the web application:
      ```sh
      python app.py
      ```

8. **Access the web application**:
    - Open your web browser and go to `http://127.0.0.1:5000` to access the crime prediction web application.

## Visualization

Interactive visualizations and dashboards are created using Plotly and Dash to present the predictions and insights to stakeholders. These visualizations can be accessed via the web application.

## Conclusion

By using machine learning for crime prediction, we can gain valuable insights into crime patterns and potentially reduce crime rates through proactive measures. The success of such a project relies on the quality of data, the choice of features, and the robustness of the models used.

## OUTPUTS
![alt text](<WhatsApp Image 2024-06-27 at 13.33.05_69f6beec.jpg>)
![alt text](<WhatsApp Image 2024-06-27 at 13.33.55_90af236d.jpg>)
![alt text](<WhatsApp Image 2024-06-27 at 13.34.15_7404a84c.jpg>)


