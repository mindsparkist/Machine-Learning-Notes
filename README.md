# Machine-Learning-Notes
## What is Machine Learning?
Machine learning is a subset of artificial intelligence (AI) that enables computers to learn from data and make decisions or predictions without being explicitly programmed.

Supervised and unsupervised learning are two fundamental approaches to machine learning that differ in their training data and learning objectives:

### Supervised Learning
- **Definition**: Supervised learning involves training a machine learning model on a labeled dataset, where each data point has a corresponding label or output value.
- **Goal**: The algorithm learns to map the input data to the desired output, allowing it to make predictions for new, unseen data.
- **Examples**:
  - Classification: Assigning categories to data points (e.g., support vector machines, logistic regression, decision trees).
  - Regression: Predicting continuous numerical values (e.g., linear regression, polynomial regression, ridge regression).
- **Advantages**:
  - Highly accurate predictions.
  - Can be used for tasks where the desired output is known.
- **Disadvantages**:
  - Requires labeled training data.
  - Computationally complex.

### Unsupervised Learning
- **Definition**: Unsupervised learning involves training a machine learning model on an unlabeled dataset, where the data points do not have corresponding labels or output values.
- **Goal**: The algorithm learns to identify patterns and structures in the data without explicit guidance.
- **Examples**:
  - Clustering: Grouping data points into clusters based on their similarity (e.g., k-means clustering, hierarchical clustering).
  - Dimensionality reduction: Reducing the number of features in a dataset while preserving the most important information (e.g., principal component analysis, autoencoders).
- **Advantages**:
  - Does not require labeled training data.
  - Can find previously unknown patterns in data.
  - Can help gain insights from unlabeled data.
- **Disadvantages**:
  - Difficult to measure accuracy or effectiveness due to lack of predefined answers during training.
  - Results often have lesser accuracy.
  - Requires time to interpret and label classes.

These two approaches are used in different scenarios and with different datasets, and the choice between them depends on the specific problem being solved, the data available, and the tools and experience needed to build and manage the models[1][2][3].

Citations:
[1] https://www.geeksforgeeks.org/supervised-unsupervised-learning/
[2] https://www.javatpoint.com/difference-between-supervised-and-unsupervised-learning
[3] https://cloud.google.com/discover/supervised-vs-unsupervised-learning
[4] https://aws.amazon.com/what-is/machine-learning/
[5] https://www.ibm.com/topics/machine-learning

Imagine you're a software engineer working on a fitness app. You want to predict how many calories someone burns based on their weight. Linear regression can help you build a model for this!

**Here's the idea:**

* **Dependent Variable (Y):**  Calories burned (what you want to predict)
* **Independent Variable (X):**  Weight (what you're basing the prediction on)

Linear regression finds a straight line that best fits the data points you have. These data points would be pairs of weight and corresponding calorie burn for different people. The line represents the **relationship** between weight and calorie burn.

**Building the Model:**

1. **Data Collection:**  You collect data on weight and calorie burn for a group of people (let's say 100 people).
2. **Line Fitting:**  The algorithm finds the equation of the straight line that minimizes the difference between the actual calorie burn (for each person) and the calorie burn predicted by the equation based on their weight.
3. **Imagine the Line:**  This line represents the average trend. Someone heavier will likely burn more calories than someone lighter, but it won't be a perfect straight line.

**Using the Model:**

Once you have the equation for the line, you can plug in a weight value (X) and get a predicted calorie burn (Y) based on the model.

**Example:**

Let's say the equation of your model is: Calories Burned (Y) = 10 * Weight (X) + 50. This means for every 1 unit increase in weight, calorie burn increases by 10 units, on average.

* If someone weighs 50 kgs (X=50), the model predicts they burn 550 calories (Y = 10 * 50 + 50).

**Important Notes:**

* This is a simplified example. Real-world data might not have a perfect linear relationship.
* Linear regression has limitations, but it's a good starting point for understanding relationships between variables.
* There are more complex regression models for non-linear relationships.

**Benefits for Software Engineers:**

* Linear regression helps you build models to predict things based on available data. 
* It's a foundational concept in machine learning, a field used in many software applications today.

By understanding linear regression, you can explore how to leverage data to make informed predictions within your software projects.

