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

warnings.filterwarnings('ignore')

# importing the dataset
data = pd.read_csv("C:/Users/Md Ganim/Desktop/Program/AI_project/Final/Dataset/FinalFertilizer.csv")
data.info()

# changing the column names
data.rename(columns={'Humidity ': 'Humidity', 'Soil Type': 'Soil_Type', 'Crop Type': 'Crop_Type', 'Fertilizer Name': 'Fertilizer'}, inplace=True)

# checking unique values and nulls
data.nunique()
data.isna().sum()

# statistical parameters
data.describe(include='all')
plt.figure(figsize=(13, 5))
sns.set(style="whitegrid")
sns.countplot(data=data, x='Crop_Type')
plt.title('Count Plot for Crop_Type')
plt.xlabel('Crop_Type')
plt.ylabel('Count')
plt.show()

# first 4 crop types
part1_data = data[data['Crop_Type'].isin(data['Crop_Type'].value_counts().index[:4])]
plt.figure(figsize=(10, 4))
sns.set(style="whitegrid")
sns.countplot(data=part1_data, x='Crop_Type', hue='Fertilizer', width=0.8, palette='Set2')
plt.title('First 4 Crop Types')
plt.xlabel('Crop_Type')
plt.ylabel('Count')
plt.legend(title='Fertilizer')
plt.xticks(rotation=45, horizontalalignment='right')
plt.tight_layout()
plt.show()

# Split the data into three parts: next 4 crop types
part2_data = data[data['Crop_Type'].isin(data['Crop_Type'].value_counts().index[4:8])]
plt.figure(figsize=(8, 4))
sns.set(style="whitegrid")
sns.countplot(data=part2_data, x='Crop_Type', hue='Fertilizer', width=0.8, palette='Set2')
plt.title('Next 4 Crop Types')
plt.xlabel('Crop_Type')
plt.ylabel('Count')
plt.legend(title='Fertilizer')
plt.xticks(rotation=45, horizontalalignment='right')
plt.tight_layout()
plt.show()

# Split the data into three parts: remaining crop types
part3_data = data[data['Crop_Type'].isin(data['Crop_Type'].value_counts().index[8:13])]
plt.figure(figsize=(8, 4))
sns.set(style="whitegrid")
sns.countplot(data=part3_data, x='Crop_Type', hue='Fertilizer', width=0.8, palette='Set2')
plt.title('Remaining Crop Types')
plt.xlabel('Crop_Type')
plt.ylabel('Count')
plt.legend(title='Fertilizer')
plt.xticks(rotation=45, horizontalalignment='right')
plt.tight_layout()
plt.show()

# Heatmap for Correlation Analysis
numerical_data = data.select_dtypes(include=[np.number])
sns.heatmap(numerical_data.corr(), annot=True)
plt.title('Correlation Heatmap')
plt.show()

# Encoding the labels for categorical variables
encode_soil = LabelEncoder()
data.Soil_Type = encode_soil.fit_transform(data.Soil_Type)

encode_crop = LabelEncoder()
data.Crop_Type = encode_crop.fit_transform(data.Crop_Type)

encode_ferti = LabelEncoder()
data.Fertilizer = encode_ferti.fit_transform(data.Fertilizer)

# Splitting the data into train and test
x_train, x_test, y_train, y_test = train_test_split(data.drop('Fertilizer', axis=1), data.Fertilizer, test_size=0.2, random_state=1)
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
LogReg = LogisticRegression(random_state=2)
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

    print(f"Classification Report for {m}:\n", classification_report(y_test, predicted_values))

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

# Cross-validation scores for each model
cv_scores = []

# Decision Tree Cross-Validation
ds_cv_scores = cross_val_score(ds, data.drop('Fertilizer', axis=1), data.Fertilizer, cv=5)
cv_scores.append(ds_cv_scores.mean())

# Naive Bayes Cross-Validation
nb_cv_scores = cross_val_score(NaiveBayes, data.drop('Fertilizer', axis=1), data.Fertilizer, cv=5)
cv_scores.append(nb_cv_scores.mean())

# SVM Cross-Validation
svm_cv_scores = cross_val_score(SVM, data.drop('Fertilizer', axis=1), data.Fertilizer, cv=5)
cv_scores.append(svm_cv_scores.mean())

# Logistic Regression Cross-Validation
logreg_cv_scores = cross_val_score(LogReg, data.drop('Fertilizer', axis=1), data.Fertilizer, cv=5)
cv_scores.append(logreg_cv_scores.mean())

# Random Forest Cross-Validation
rf_cv_scores = cross_val_score(RF, data.drop('Fertilizer', axis=1), data.Fertilizer, cv=5)
cv_scores.append(rf_cv_scores.mean())

# Now you can print or visualize the cross-validation scores along with other accuracies
for i, m in enumerate(model):
    print(f"{m} Cross-Validation Score: {cv_scores[i] * 100:.2f}%")

# Add this code after calculating cross-validation scores

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



# Function to predict fertilizer without using encoded values
def predict_fertilizer(temperature, humidity, soil_type, crop_type, nitrogen, potassium, phosphorous):
    # Convert categorical variables to numerical using the encoder
    soil_type_encoded = encode_soil.transform([soil_type])[0]  # Encode Soil Type
    crop_type_encoded = encode_crop.transform([crop_type])[0]  # Encode Crop Type

    # Create an input feature array
    input_features = np.array([[temperature, humidity, soil_type_encoded, crop_type_encoded, nitrogen, potassium, phosphorous]])

    # Predict using the Random Forest model (or any other model you want)
    fertilizer_prediction = RF.predict(input_features)

    # Decode the predicted fertilizer label
    predicted_fertilizer = encode_ferti.inverse_transform(fertilizer_prediction)
    
    return predicted_fertilizer[0]


predicted = predict_fertilizer(25, 60,'Clayey','Wheat', 50, 20, 40)
print("The suggested fertilizer for the given conditions is:", predicted)

#dumping into model
pickle_out = open('C:/Users/Md Ganim/Desktop/Program/AI_project/Final/models/fertilizer.pkl', 'wb')
pickle.dump(RF, pickle_out)
pickle_out.close()

pickle_out = open('C:/Users/Md Ganim/Desktop/Program/AI_project/Final/models/encoder_fert.pkl','wb')
pickle.dump(encode_ferti,pickle_out)
pickle_out.close()

pickle_out = open('C:/Users/Md Ganim/Desktop/Program/AI_project/Final/models/encoder_soil.pkl','wb')
pickle.dump(encode_soil,pickle_out)
pickle_out.close()

pickle_out = open('C:/Users/Md Ganim/Desktop/Program/AI_project/Final/models/encoder_crop.pkl','wb')
pickle.dump(encode_crop,pickle_out)
pickle_out.close()