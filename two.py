import pandas as pd
from sklearn.tree import DecisionTreeClassifier, export_text
from sklearn.preprocessing import OrdinalEncoder

df = pd.read_csv("play_tennis.csv")

features = ['Outlook', 'Temperature', 'Humidity', 'Wind']
encoder = OrdinalEncoder()
X_encoded = encoder.fit_transform(df[features])
y = df['PlayTennis']

clf = DecisionTreeClassifier(criterion='entropy', random_state=42)
clf.fit(X_encoded, y)

print("Decision Tree Rules for Playing Tennis from CSV:")
print(export_text(clf, feature_names=features))

sample_condition = pd.DataFrame({
    'Outlook': ['Sunny'],
    'Temperature': ['Cool'],
    'Humidity': ['Normal'],
    'Wind': ['Weak']
})
sample_encoded = encoder.transform(sample_condition)
prediction = clf.predict(sample_encoded)
print(f"Prediction for condition [Sunny, Cool, Normal, Weak]: PlayTennis = {prediction[0]}")
