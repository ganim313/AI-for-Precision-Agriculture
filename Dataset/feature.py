import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import mutual_info_classif, f_classif, chi2

class FeatureImportanceAnalyzer:
    def __init__(self, file_path, target_column):
        # Load the dataset
        self.data = pd.read_csv(file_path)
        
        # Separate features and target
        self.X = self.data.drop(target_column, axis=1)
        self.y = self.data[target_column]
        
        # Preprocess the data
        self.preprocess_data()
        
        # Standardize features
        self.scaler = StandardScaler()
        self.X_scaled = self.scaler.fit_transform(self.X)
    
    def preprocess_data(self):
        # Encode categorical variables
        for column in self.X.select_dtypes(include=['object']):
            le = LabelEncoder()
            self.X[column] = le.fit_transform(self.X[column].astype(str))
        
        # Encode target variable
        le_target = LabelEncoder()
        self.y = le_target.fit_transform(self.y.astype(str))
    
    def get_feature_importance(self):
        # Combine multiple feature importance methods
        results = {}
        
        # 1. Random Forest Feature Importance
        rf = RandomForestClassifier(n_estimators=100, random_state=42)
        rf.fit(self.X, self.y)
        results['Random Forest Importance'] = dict(zip(
            self.X.columns, 
            rf.feature_importances_
        ))
        
        # 2. Mutual Information
        mi_scores = mutual_info_classif(self.X, self.y)
        results['Mutual Information'] = dict(zip(
            self.X.columns, 
            mi_scores
        ))
        
        # 3. ANOVA F-Test
        f_scores, _ = f_classif(self.X_scaled, self.y)
        results['ANOVA F-Test'] = dict(zip(
            self.X.columns, 
            f_scores
        ))
        
        # 4. Chi-Square Test (for non-negative features)
        X_positive = self.X_scaled - self.X_scaled.min()
        chi2_scores, _ = chi2(X_positive, self.y)
        results['Chi-Square'] = dict(zip(
            self.X.columns, 
            chi2_scores
        ))
        
        # 5. Correlation with Target
        correlation = self.X.apply(lambda col: np.abs(col.corr(pd.Series(self.y))))
        results['Correlation'] = dict(zip(
            self.X.columns, 
            correlation
        ))
        
        return results
    
    def print_feature_importance(self):
        # Get feature importance
        importance_results = self.get_feature_importance()
        
        # Print results for each method
        for method, scores in importance_results.items():
            print(f"\n{method}:")
            # Sort scores in descending order
            sorted_scores = sorted(scores.items(), key=lambda x: x[1], reverse=True)
            
            # Print each feature and its score
            for feature, score in sorted_scores:
                print(f"{feature}: {score:.4f}")
    
    def aggregate_importance(self):
        # Combine importance from different methods
        importance_results = self.get_feature_importance()
        
        # Normalize and aggregate scores
        aggregated_scores = {}
        for method, scores in importance_results.items():
            # Normalize scores
            max_score = max(scores.values())
            normalized_scores = {k: v/max_score for k, v in scores.items()}
            
            # Aggregate scores
            for feature, score in normalized_scores.items():
                if feature not in aggregated_scores:
                    aggregated_scores[feature] = []
                aggregated_scores[feature].append(score)
        
        # Calculate average importance
        avg_importance = {
            feature: np.mean(scores) 
            for feature, scores in aggregated_scores.items()
        }
        
        # Sort and print aggregated importance
        print("\nAggregated Feature Importance:")
        sorted_importance = sorted(avg_importance.items(), key=lambda x: x[1], reverse=True)
        for feature, importance in sorted_importance:
            print(f"{feature}: {importance:.4f}")

def main():
    # Replace with your actual file path and target column name
    file_path = 'C:/Users/Md Ganim/Desktop/Program/AI_project/Final/Dataset/f2.csv'  # Update this with your CSV file path
    target_column = 'Fertilizer'  # Update this with your target column name
    
    # Initialize the feature importance analyzer
    analyzer = FeatureImportanceAnalyzer(file_path, target_column)
    
    # Print detailed feature importance for each method
    print("Detailed Feature Importance by Method:")
    analyzer.print_feature_importance()
    
    # Print aggregated feature importance
    analyzer.aggregate_importance()

if __name__ == "__main__":
    main()