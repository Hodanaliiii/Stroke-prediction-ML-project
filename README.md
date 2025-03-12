# Stroke-prediction-ML-project
CS-C3240 - Machine Learning
Stage 2
Comparing Logistic Regression and Decision Tree Classifier in Stroke prediction
# 1. Introduction
Stroke is a number no.5 cause of death in the United States [1]. Stroke is also common in Finland and approximately 24 000
people suffer from stroke yearly [2]. The average healthcare costs from the onset of a stroke to the end of a patient’s life are
significant per individual. Due to these reasons, we think it’s crucial to look more deeply into the impact of the risk factors of
stroke. It's important to focus on actions before the occurrence of a stroke. The goal of this project is to compare the performance of Logistic Regression and Decision Tree Classifier in predicting stroke occurrence based on health-related risk factor
The structure of our report is divided insections. In section 2 we formulate the machine learning problem that our project deals
with. The section 3 and 4 is devoted to the two methods that we use. In those sections, we cover the datasets and the process that
we go through each method in its own section. In section 5 we will discuss the results that we discovered, and conclusions that
summarize our whole report can be found in the same section. In the end we have sourced our references and attached our
appendix .
# 2. Problem Formulation
A stroke is the result of blockage of cerebral blood vessels. The blockage leads to a lack of oxygen in the brain which is followed
by a damage to the brain function of the specific area where stroke occurred. Stroke can be disabling and even fatal, and needs to
be treated immediately. Thus we wanted to compare different health related factors and predict their impact on whether a stroke
occurs or not.
The dataset was collected from a machine learning project from Kaggle and in it we found interesting data from which we
collected the health related data we found the most useful in this project. In the next chapters we will tell more on how we chose
the data features and data points.
3. Methods
3.1 Dataset & data
We obtained our dataset for the machine learning project from Kaggle and it’s called “ Stroke prediction” [3]. We have 5110 data
points and each data point corresponds to an individual or patient used for stroke prediction. In our dataset there are 9 columns
each containing specific information about a person.
We have 8 features and one target variable in our dataset, which are Body Mass Index (BMI), smoking status, gender, age, hypertension, heart disease,
work type, residence type and average glucose level. Heart disease and hypertension are binary features and smoking status,
work type, residence type and gender are categorical features. Age is a numerical feature and in our data set every individual's
age is represented with a specific number. Our target variable is Stroke, thus our label is whether stroke occurs or not. Stroke
occurrence can be described with the numbers 0 or 1, since our label is binary. If the number 0 occurs, it means that the person
doesn’t have Stroke and if the number 1 occurs, the person has had Stroke. When we conducted our analysis, we noticed that our
data is extremely unbalanced since we have only 209 Stroke instances and 4699 non-Stroke instances in the dataset.
## 3.2 Feature selection
There is a similar problem solved in Kaggle, they use all the features and all the classification methods. [3] However, we had a
clear vision on what features we wanted to focus on, since we were interested in health-related features that may directly affect
the likelihood of stroke. We excluded three features from our data: ever married, residence type and worktype. These features
have the potential to give beneficial information, but we wanted to maintain a clear emphasis on the health aspects of Stroke risk.
Therefore, we maintained Body Mass Index (BMI), smoking status, gender, age, hypertension, heart disease, and average glucose
level. We know from a domain knowledge that hypertension, high bmi, smoking, heart disease, average glucose levels are risk
factors for Stroke. We confirmed our understanding from Käypähoito (Current Care guidelines) website, which is a trusted source of
clinical practice guidelines in Finland. In additio to the mentioned health related factors, age is a major factor in the occurrence of
a stroke. In the Käypä Hoito article it was stated that under the age of 75 men have two times higher risk of stroke compared to
women, although after reaching the age of 85 women have greater risk than men. [12]
## 3.3 Preprocessing the data
We converted categorical features into binary form. We changed the gender into binary form, so 0 is male and 1 is
female. There was “smokes”, “formerly smoked”, “never smoked” and “unknown” in our categorical feature smoking status. We
replaced the unknown values to random values with the assistance of the TA. After that we combined “ formerly smoked” and
“smokes” and replaced them with 1. We did this combination choice, because it simplifies interpretation and modeling. In
addition, our interest lies in how past or current smoking has an impact on a risk of a stroke. We also replaced “never smokes”
with 0, to binarize our categorical feature.
It is important to check if the data contains any null data. [luentodia]From the output of our data, we noticed that we had null
values in our dataset. The null values were in the BMI feature, because there were 201 columns with the value of zero. We
decided to remove these missing values to ensure that we could have more reliable data. After removing the missing values, our
data points were then reduced to 4908.
## 3.4 Correlation between features
The correlation matrix is used to understand the relationships between the features
and one can also interpret the features that have the most impact on the target
variable. The correlation coefficient 1 indicates a strong correlation between the
features. Thus, in this correlation matrix it means that the correlation between
feature and our target variable increases, the lighter and the higher the correlation
coefficient is.
We can see that age, hypertension, heart disease and average glucose level have
the most impact on stroke occurrence. Gender, BMI and smoking status have
darker areas and smaller coefficient numbers, so the correlation is smaller with
those features.
## 3.5 Constructing data sets / Training, validation and test sets
We split our dataset into training, validation and test datasets with the use of sklearn library. We put 70% of our data into a
training set, because based on our machine learning lectures larger sets can be helpful in detecting and capturing more complex
patterns[1]. Furthermore, we put 15% of data into the validation set and keep 15% of the data for the testing. We tried various
split size rations, but 70% turned out to be the best, since accuracy scores and F1-scores were the highest. We have 4908 data
points as mentioned, and if our dataset would be very large, we would allocate a smaller portion for the training dataset to ensure
a satisfactory amount of testing and validation set. Furthermore,  We need to use a larger percentage
for training to ensure that our data has enough data to learn from.
## 3.6 First Method: Logistic regression
We chose logistic regression for our first method in this machine learning problem, because in our ML task we are predicting a
categorical target variable to one of two possible outcomes - if our target variable, a stroke occurs then it’s described as 1, which
stands for true and if it doesn’t then the target variable is described as 0 which stands for false. Furthermore, 0 means that the
person has not had a stroke and 1 means that person has had a stroke. Additionally, the benefits of logistic regression is that it is
less prone to overfitting and efficient to train. The method is also more robust to correlated features. These reasons reinforced our
willingness to use this method [13]
Logistic regression is a binary classification, so it’s used for data points that consist of binary labels. Logistic regression can be
interpret with this equation . [14]The left side of the equation represents the probability of
the target variable Y =1. The right side of the logistic regression calculates the denominator of the logistic function to ensure that
the range of the predicted values is between 0 and 1.
Before building the final Logistic Regression model for testing, we had to standardize our features preprocessing the data,
because they had significantly different ranges of values. We noticed this, when we trained our Logistic regression model. We
used StandardScaler function for training and test sets (can be seen in the appendix).
## 3.7 The Loss function
We are using logistic loss as our loss function, as it is a powerful and common evaluating metric for binary classification
methods, which our Logistic regression is. Logistic loss is the negative average of the corrected probabilities for each occurrence,
which are in log.
Logistic loss can be interpret with this equation estimate of probability and here the true label is y=(0,1)-}.
[10], where p is the
The closer the loss function value is to 1, the better the classifier and the closer to 0 the worse the classifier is. For our testing set,
we got 0.43745 based on our output.
We could have used the squared error loss, as our loss function. Nonetheless, logistic loss is a better choice, since we have
classified data points with binary labels.
# 4. Second method: Decision Tree Classifier
The second method was selected to be Decision Tree Classifier which is also a classification method. The process of our selection
was inspired by the information in our dataset. This method works well with missing data, which was our case in our features.
The fact that Decision Tree Classifier can handle both numerical and categorical data made us choose this method since our data
information included both.[4] However, the categorial features have to be encoded to represent binary values by using one-hot
encoding. This means that our categorical variables have been changed to values 0 and 1. This method divides data points into
categories effectively to illustrate the information in the data with decision making. DecisionTreeClassifier is usable with the
defined binary variables. This also supports plotting, which makes it possible to plot the tree for proper visual representation.
This satisfies our project aim since we can predict the occurrence of stroke. In this case we’ve got two categories, which one is
the occurrence of the stroke and the other is when the stroke doesn’t occur. The amount of our selected data points, which is in
our case 4908 data points referring to the quantity of patients.
Since we would like to know the quality of our classifiers performance on the prediction of our stroke occurrence with these
features we divide the data set to training and test sets. Dataset was split into a training set of 70%. The remaining data was
splitted to a validation set of 15% and a test set of 15%. The reasoning behind this choice is that it resulted in the highest
accuracy which was what we aimed for. Training data has been divided and tested with the train_test_split() function from
scikit-learn.[15] The accuracy score is 0.90780 by using Decision Tree method and training accuracy was computed to the value
1.0. This means that the tree is overfitting which can be seen in the structure on image below.
.
The overfit can be avoided by pruning the tree. [7] This was done with GridSearch and finished fitting a new model where the
criterion on pruning was set to “gini”, the maximum depth of the branches to be 5 and maximum features to be 7. [8] We still
chose the maximum depth to be two since we wanted to simplify our tree. Our new computed accuration increased to 0.77021
which is not to our liking, so we stuck with our first accuracy. In the next section we tested our new Decision Tree with our loss
function.
## 4.1 Loss function
In our Decision Tree model we are using Gini Impurity as a loss function. Gini Impurity indicates the likelihood of
misclassification to happen in the dataset. The probability of misclassifying the features is low when the gini impurity is low. [9]
The reasoning of us using gini impurity lies on the fact that it evaluates the best split when we have categorical features. [10]The
formula of Gini impurity calculation is Gini Impurity = 1- Gini index, here the gini index gets values 0, which is the purity and 1
shows how features have randomly spread to different classes.[11]
As you can see in our plotted decision tree, we have two different features on display. These are hypertension and BMI. By this it
means that these features are most likely to have the most significant effect on our prediction.