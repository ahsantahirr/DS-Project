# Import necessary libraries
import pandas as pd
import re
from nltk.corpus import stopwords
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score
import nltk

# Download stopwords
nltk.download('stopwords')

# Load the dataset
file_path = "./bbc_news (1).csv"  # Replace this with your dataset path
data = pd.read_csv(file_path)

# Step 1: Combine 'title' and 'description' columns into a single 'text' column
data['text'] = data['title'] + " " + data['description']

# Step 2: Assign categories based on keywords in the text
def assign_category(text):
    text = text.lower()
    if 'ukraine' in text or 'war' in text:
        return 'World'
    elif 'business' in text or 'oil price' in text:
        return 'Business'
    elif 'arena bombing' in text or 'attack' in text:
        return 'Crime'
    elif 'technology' in text or 'innovation' in text:
        return 'Technology'
    else:
        return 'General'

data['category'] = data['text'].apply(assign_category)

# Step 3: Preprocess the text
stop_words = set(stopwords.words('english'))

def clean_text(text):
    text = re.sub(r'http\S+', '', text)  # Remove URLs
    text = re.sub(r'[^\w\s]', '', text)  # Remove punctuation
    text = text.lower().strip()  # Convert to lowercase and strip whitespace
    text = ' '.join(word for word in text.split() if word not in stop_words)  # Remove stopwords
    return text

data['cleaned_text'] = data['text'].apply(clean_text)
# print(data.head())
# print(data.tail())
# Step 4: Split the data into training and testing sets
X = data['cleaned_text']
y = data['category']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Step 5: Convert text into numerical features using TfidfVectorizer
vectorizer = TfidfVectorizer(max_features=10000)
X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)

# Step 6: Train a Random Forest Classifier
model = RandomForestClassifier(n_estimators=200, random_state=42)
model.fit(X_train_vec, y_train)

# Step 7: Evaluate the model
y_pred = model.predict(X_test_vec)

accuracy = accuracy_score(y_test, y_pred)
report = classification_report(y_test, y_pred)

print("Model Accuracy:", accuracy)
print("\nClassification Report:\n", report)
