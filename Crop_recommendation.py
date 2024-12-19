#importing libraries
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import warnings
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score
from sklearn.model_selection import cross_val_score
import pickle

# Load dataset
data = pd.read_csv(r"C:/Users/Md Ganim/Desktop/Program/AI_project/Final/Dataset\FinalCrop_Reommenationn.csv")

# Renaming columns
data.columns = ['Nitrogen','Phosphorus','Potassium','Temperature','Humidity','pH','Rainfall','Soil','Label']

# Display basic information and statistics
print("Shape of the dataset:", data.shape)
print("Missing/Null values in the dataset:")
print(data.isnull().sum())
print(data.describe())
print(data['Label'].value_counts())
data.info()

# Check for duplicates
print("Number of duplicates:", data.duplicated().sum())

# Average values for each feature
print("Average Ratio of Nitrogen in the soil : {0:.2f}".format(data['Nitrogen'].mean()))
print("Average Ratio of Phosphorus in the soil : {0:.2f}".format(data['Phosphorus'].mean()))
print("Average Ratio of Potassium in the soil : {0:.2f}".format(data['Potassium'].mean()))
print("Average Temperature in Celsius : {0:.2f}".format(data['Temperature'].mean()))
print("Average Humidity : {0:.2f}".format(data['Humidity'].mean()))
print("Average pH in the soil : {0:.2f}".format(data['pH'].mean()))
print("Average Rainfall : {0:.2f}".format(data['Rainfall'].mean()))

# Interactive function to display statistics by crop label
from ipywidgets import interact

@interact
def summary(crops=list(data['Label'].value_counts().index)):
    x = data[data['Label'] == crops]
    print(f"Statistics for {crops}:")
    print("Minimum Nitrogen:", x['Nitrogen'].min(), " | Maximum Nitrogen:", x['Nitrogen'].max())
    print("Minimum Phosphorus:", x['Phosphorus'].min(), " | Maximum Phosphorus:", x['Phosphorus'].max())
    print("Minimum Potassium:", x['Potassium'].min(), " | Maximum Potassium:", x['Potassium'].max())
    print("Minimum Temperature:", x['Temperature'].min(), " | Maximum Temperature:", x['Temperature'].max())
    print("Minimum Humidity:", x['Humidity'].min(), " | Maximum Humidity:", x['Humidity'].max())
    print("Minimum pH:", x['pH'].min(), " | Maximum pH:", x['pH'].max())
    print("Minimum Rainfall:", x['Rainfall'].min(), " | Maximum Rainfall:", x['Rainfall'].max())

# Visualize data distributions
plt.style.use('dark_background')
sns.set_palette("Set2")
for i in data.columns[:-1]:
    fig, ax = plt.subplots(1, 3, figsize=(18, 4))
    sns.histplot(data=data, x=i, kde=True, bins=20, ax=ax[0])
    sns.violinplot(data=data, x=i, ax=ax[1])
    sns.boxplot(data=data, x=i, ax=ax[2])
    plt.suptitle(f'Visualizing {i}', size=20)
plt.show()

# Grouped bar plot of means for each crop label
# print(data.dtypes)
# grouped = data.groupby(by='Label').mean().reset_index()
# fig, ax = plt.subplots(7, 1, figsize=(25, 25))
# for index, i in enumerate(grouped.columns[1:]):
#     sns.barplot(data=grouped, x='Label', y=i, ax=ax[index])
# plt.suptitle("Comparison of Mean Attributes of Various Crops", size=25)
# plt.show()

# # Top crops based on attributes
# for i in grouped.columns[1:]:
#     print(f'Top 5 crops requiring highest {i}:')
#     for j, k in grouped.sort_values(by=i, ascending=False)[:5][['Label', i]].values:
#         print(f'{j} --> {k}')
#     print('--------------------------------')
    

names = data['Label'].unique()
from sklearn.preprocessing import LabelEncoder
encoder=LabelEncoder()
soil_encoder=LabelEncoder()
data['Soil']=soil_encoder.fit_transform(data['Soil'])
data['Label']=encoder.fit_transform(data['Label'])
data.head()

# Splitting the dataset for predictive modeling
y = data['Label']
x = data.drop(['Label'], axis=1)

# Train-test split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=0)
print('Shape of Splitting:')
print('x_train = {}, y_train = {}, x_test = {}, y_test = {}'.format(x_train.shape, y_train.shape, x_test.shape, y_test.shape))

acc = []  # TEST
model = []
acc1 = []  # TRAIN

# Decision Tree
ds = DecisionTreeClassifier(criterion="entropy", random_state=2, max_depth=5)
ds.fit(x_train, y_train)
predicted_values = ds.predict(x_test)
x = accuracy_score(y_test, predicted_values)
acc.append(x)
model.append('Decision Tree')

predicted_values_train = ds.predict(x_train)
y = accuracy_score(y_train, predicted_values_train)
acc1.append(y)
print("Decision Tree's Accuracy: ", x * 100, y * 100)

# Naive Bayes
NaiveBayes = GaussianNB()
NaiveBayes.fit(x_train, y_train)

predicted_values = NaiveBayes.predict(x_test)
x = accuracy_score(y_test, predicted_values)
acc.append(x)

predicted_values_train = NaiveBayes.predict(x_train)
y = accuracy_score(y_train, predicted_values_train)
acc1.append(y)

model.append('Naive Bayes')
print("Naive Bayes's Accuracy: ", x * 100, y * 100)

# SVM
norm = MinMaxScaler().fit(x_train)
X_train_norm = norm.transform(x_train)
X_test_norm = norm.transform(x_test)

SVM = SVC(kernel='poly', degree=3, C=1)
SVM.fit(X_train_norm, y_train)

predicted_values = SVM.predict(X_test_norm)
x = accuracy_score(y_test, predicted_values)
acc.append(x)

predicted_values_train = SVM.predict(X_train_norm)
y = accuracy_score(y_train, predicted_values_train)
acc1.append(y)

model.append('SVM')
print("SVM's Accuracy: ", x * 100, y * 100)

# Logistic Regression
LogReg = LogisticRegression(solver='liblinear')
LogReg.fit(x_train, y_train)

predicted_values = LogReg.predict(x_test)
x = accuracy_score(y_test, predicted_values)
acc.append(x)

predicted_values_train = LogReg.predict(x_train)
y = accuracy_score(y_train, predicted_values_train)
acc1.append(y)

model.append('Logistic Regression')
print("Logistic Regression's Accuracy: ", x * 100, y * 100)

# Random Forest
RF = RandomForestClassifier(n_estimators=20, random_state=0)
RF.fit(x_train, y_train)

predicted_values = RF.predict(x_test)
x = accuracy_score(y_test, predicted_values)
acc.append(x)

predicted_values_train = RF.predict(x_train)
y = accuracy_score(y_train, predicted_values_train)
acc1.append(y)

model.append('Random Forest')
print("Random Forest's Accuracy: ", x * 100, y * 100)

# Display classification reports for each model
for i, m in enumerate(model):
    predicted_values = None
    if m == 'Decision Tree':
        predicted_values = ds.predict(x_test)
    elif m == 'Naive Bayes':
        predicted_values = NaiveBayes.predict(x_test)
    elif m == 'SVM':
        predicted_values = SVM.predict(X_test_norm)
    elif m == 'Logistic Regression':
        predicted_values = LogReg.predict(x_test)
    elif m == 'Random Forest':
        predicted_values = RF.predict(x_test)

    print(f"Classification Report for {m}:\n", classification_report(y_test, predicted_values,zero_division=1))

# Create a DataFrame for the accuracies
accuracy_df = pd.DataFrame({
    'Model': model,
    'Train Accuracy': acc1,
    'Test Accuracy': acc
})

# Set the figure size
plt.figure(figsize=(12, 6))

# Plotting the accuracies
bar_width = 0.35
index = np.arange(len(model))

# Create bars for training and testing accuracy
plt.bar(index, acc1, bar_width, label='Train Accuracy', color='b', alpha=0.6)
plt.bar(index + bar_width, acc, bar_width, label='Test Accuracy', color='r', alpha=0.6)

# Adding labels and title
plt.xlabel('Model')
plt.ylabel('Accuracy')
plt.title('Comparison of Model Accuracies')
plt.xticks(index + bar_width / 2, model)
plt.legend()

# Show the plot
plt.tight_layout()
plt.show()


# After fitting and evaluating all models, add the following:

# Cross-validation scores for each model
cv_scores = []

# Decision Tree Cross-Validation
ds_cv_scores = cross_val_score(ds, data.drop('Label', axis=1), data['Label'], cv=5)
cv_scores.append(ds_cv_scores.mean())

# Naive Bayes Cross-Validation
nb_cv_scores = cross_val_score(NaiveBayes, data.drop('Label', axis=1), data['Label'], cv=5)
cv_scores.append(nb_cv_scores.mean())

# SVM Cross-Validation
svm_cv_scores = cross_val_score(SVM, data.drop('Label', axis=1), data['Label'], cv=5)
cv_scores.append(svm_cv_scores.mean())

# Logistic Regression Cross-Validation
logreg_cv_scores = cross_val_score(LogReg, data.drop('Label', axis=1), data['Label'], cv=5)
cv_scores.append(logreg_cv_scores.mean())

# Random Forest Cross-Validation
rf_cv_scores = cross_val_score(RF, data.drop('Label', axis=1), data['Label'], cv=5)
cv_scores.append(rf_cv_scores.mean())


# Now you can print or visualize the cross-validation scores along with other accuracies
for i, m in enumerate(model):
    print(f"{m} Cross-Validation Score: {cv_scores[i] * 100:.2f}%")


# Create a DataFrame for the accuracies and cross-validation scores
accuracy_cv_df = pd.DataFrame({
    'Model': model,
    'Train Accuracy': acc1,
    'Test Accuracy': acc,
    'Cross-Validation Score': cv_scores
})

# Set the figure size
plt.figure(figsize=(14, 7))

# Set bar width and positions
bar_width = 0.25
index = np.arange(len(model))

# Create bars for training, testing, and cross-validation accuracy
plt.bar(index, acc1, bar_width, label='Train Accuracy', color='b', alpha=0.6)
plt.bar(index + bar_width, acc, bar_width, label='Test Accuracy', color='r', alpha=0.6)
plt.bar(index + 2 * bar_width, cv_scores, bar_width, label='Cross-Validation Score', color='g', alpha=0.6)

# Adding labels and title
plt.xlabel('Model')
plt.ylabel('Accuracy')
plt.title('Comparison of Model Accuracies and Cross-Validation Scores')
plt.xticks(index + bar_width, model)
plt.legend()

# Show the plot
plt.tight_layout()
plt.show()
data.head()
# Make the prediction
import pandas as pd

# Assuming you have the feature names in a list
feature_names = ['Nitrogen', 'Phosphorus', 'Potassium', 'Temperature', 'Humidity', 'pH', 'Rainfall', 'Soil']

# Create a DataFrame for input features
input_features_df = pd.DataFrame(data=[[50, 45, 10, 20, 80, 7, 200, soil_encoder.transform(['clayey'])[0]]], columns=feature_names)

# Make the prediction using the DataFrame
prediction_numeric = RF.predict(input_features_df)

# Convert the numeric prediction back to the original label
prediction_label = encoder.inverse_transform(prediction_numeric)

print("The suggested crop for the given climatic condition is:", prediction_label[0])

# For predicted probabilities
predicted_probabilities_RF = RF.predict_proba(input_features_df)
top_indices_1 = np.argsort(predicted_probabilities_RF[0])[::-1][:3]  # Get the indices of the top 3 crops
top_predictions1 = encoder.inverse_transform(top_indices_1)

print("The suggested crops for the given climatic condition are:")
for crop in top_predictions1:
    print(crop)
# dumping pickle file
pickle_out = open('C:/Users/Md Ganim/Desktop/Program/AI_project/crop_recom.pkl', 'wb')
pickle.dump(RF, pickle_out)
pickle_out.close()