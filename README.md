[![Review Assignment Due Date](https://classroom.github.com/assets/deadline-readme-button-24ddc0f5d75046c5622901739e7c5dd533143b0c8e959d652212380cedb1ea36.svg)](https://classroom.github.com/a/ml3o3YLw)
# Task
The task I'm addressing in this code is a text classification problem where the goal is to classify movie reviews into one of three categories:
- Not a movie/TV show review
- Positive movie/TV show review
- Negative movie/TV show review
The provided datasets, which include a training set and a test set, contain text documents that need to be classified into the appropriate categories.

# Approach 

I begin by importing the necessary libraries for the task at hand. These include pandas for data manipulation, scikit-learn's TfidfVectorizer for feature extraction, SelectKBest and chi2 from scikit-learn for feature selection, LogisticRegression for model training, train_test_split for splitting the data into training and validation sets, accuracy_score and f1_score for model evaluation, and dump from joblib for saving the trained model.

Next, I define a function called load_and_preprocess_data that takes the file paths of the training and test data as input. This function reads the CSV files using pandas, and fills any missing values in the 'TEXT' column with empty strings. It returns the preprocessed training and test data.

I then define two functions for feature extraction and selection. The extract_features function takes the preprocessed training and test data as input and applies the TF-IDF vectorizer to convert the text data into numerical features. It uses n-grams ranging from 1 to 3, a minimum document frequency of 5, a maximum document frequency of 95%, sublinear term frequency scaling, and removes English stop words. The function returns the transformed training and test features, as well as the fitted vectorizer.

The select_features function takes the extracted features, the training labels, and the number of top features to select (default is 20,000) as input. It uses the chi-squared statistical test to select the most informative features and returns the selected features for both the training and test sets, along with the fitted selector object.

I define a function called train_and_evaluate_model that takes the selected training features and labels as input. It initializes a logistic regression model with specific hyperparameters, such as the regularization strength (C=1.0), the multi-class strategy ('multinomial'), the solver ('saga'), and the maximum number of iterations (1000). The function then splits the training data into a smaller training set and a validation set using train_test_split with a test size of 20% and a fixed random state for reproducibility. It trains the model on the smaller training set and evaluates its performance on the validation set using accuracy and macro-averaged F1-score. The function prints the validation accuracy and F1-score and returns the trained model.

# Results
The model achieves a validation accuracy of 92.38% and a macro-averaged F1-score of 91.19%, indicating its effectiveness in distinguishing between the three categories of documents.
# Code
https://github.com/uazhlt-ms-program/grad-level-term-project-kaggle-competition-Ahmed8M
# Notes
- **At least one of your submitted solutions must use one or more of the classification algorithms covered in this course**
- You are not obligated to use Python
- You may delete or alter any files in this repository
- You are free to add dependencies
  - Ensure that your code can be installed/used on another machine running Linux or MacOS (consider containerizing your project with Docker or an equivalent technology)
