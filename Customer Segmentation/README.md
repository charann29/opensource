# Customer Segmentation using K-Means Clustering

This project demonstrates customer segmentation using the K-Means clustering algorithm. The dataset used for this analysis is `customers.csv`, which contains information about customers, including their annual income and spending score.

## Project Structure

- `customers.csv`: The dataset file containing customer data.
- `README.md`: This file, providing an overview of the project.
- `customer_segmentation.py`: The Python script containing the data preprocessing, clustering, and visualization code.

## Dependencies

This project requires the following Python libraries:

- pandas
- numpy
- matplotlib
- seaborn
- scikit-learn

You can install these libraries using pip:

```bash
pip install pandas numpy matplotlib seaborn scikit-learn
```

## Usage
- Load the dataset: The dataset is loaded into a pandas DataFrame.
- Select features for clustering: Only the 'Annual Income (k$)' and 'Spending Score (1-100)' features are used for clustering.
- Standardize the features: The features are standardized using StandardScaler.
- Determine the optimal number of clusters: The Elbow method is used to find the optimal number of clusters. The within-cluster sum of squares (WCSS) is plotted for different numbers of clusters.
- Apply K-Means clustering: K-Means clustering is performed with the optimal number of clusters (5 in this example).
- Add cluster labels to the DataFrame: The cluster labels are added to the original DataFrame.
- Visualize the clusters: The clusters are visualized using a scatter plot.
- Display cluster statistics: The mean values of numeric columns for each cluster are displayed.

## Example
To run the script, execute the following command:
This will produce the following outputs:

- An Elbow graph to help determine the optimal number of clusters.
- A scatter plot visualizing the customer segments.
- Mean values of numeric columns for each cluster.
## Results
The scatter plot visualizes the customer segments based on annual income and spending score. The mean values of numeric columns for each cluster are displayed to understand the characteristics of each segment.

## Conclusion
This project demonstrates the use of K-Means clustering for customer segmentation. By understanding customer segments, businesses can tailor their marketing strategies to better meet the needs of different customer groups.