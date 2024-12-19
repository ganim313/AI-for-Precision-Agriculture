import pandas as pd

# Read the CSV data
df = pd.read_csv('C:/Users/Md Ganim/Desktop/Program/AI_project/Final/Dataset/f2.csv')

# Drop the 'Moisture' column
df_dropped = df.drop('Moisture', axis=1)

# Optional: Save the modified dataframe back to a CSV
df_dropped.to_csv('C:/Users/Md Ganim/Desktop/Program/AI_project/Final/Dataset/FinalFertilizer.csv', index=False)