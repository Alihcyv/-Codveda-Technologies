%%writefile sentiment_pipeline.py
import re
import nltk
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics import accuracy_score, classification_report, f1_score

nltk.download('stopwords', quiet=True)
nltk.download('wordnet', quiet=True)
nltk.download('omw-1.4', quiet=True)

stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()

def split_hashtags(text: str) -> str:
    hashtags = re.findall(r"#(\w+)", text)
    for tag in hashtags:
        split_tag = re.sub(r'([a-z])([A-Z])', r'\1 \2', tag)
        text = text.replace(f'#{tag}', split_tag)
    return text

def clean_text(text: str) -> str:
    text = text.lower().strip()
    text = re.sub(r'[^a-z\s]', '', text)
    words = text.split()
    cleaned_words = [lemmatizer.lemmatize(w) for w in words if w not in stop_words]
    return " ".join(cleaned_words)


pos_words = [
    'Positive', 'Happiness', 'Joy', 'Love', 'Amusement', 'Enjoyment', 'Admiration', 'Affection', 'Awe', 'Acceptance', 'Adoration', 'Runway Creativity',
    'Anticipation', 'Calmness', 'Excitement', 'Kind', 'Pride', 'Elation', 'Euphoria', 'Contentment', 'Serenity', 'Gratitude', "Ocean's Freedom",
    'Hope', 'Empowerment', 'Compassion', 'Tenderness', 'Enthusiasm', 'Fulfillment', 'Reverence', 'Curiosity', 'Determination', 'Zest', 'Relief',
    'Hopeful', 'Proud', 'Grateful', 'Empathetic', 'Compassionate', 'Playful', 'Free-spirited', 'Inspired', 'Confident', 'Thrill', 'Mischievous',
    'Overjoyed', 'Inspiration', 'Motivation', 'Satisfaction', 'Blessed', 'Appreciation', 'Confidence', 'Accomplishment', 'Wonderment', 'Happy',
    'Optimism', 'Enchantment', 'Intrigue', 'PlayfulJoy', 'Mindfulness', 'DreamChaser', 'Elegance', 'Whimsy', 'Harmony', 'Creativity', 'JoyfulReunion',
    'Radiance', 'Wonder', 'Rejuvenation', 'Coziness', 'Adventure', 'Melodic', 'FestiveJoy', 'InnerJourney', 'Freedom', 'Dazzle', 'Renewed Effort',
    'Adrenaline', 'ArtisticBurst', 'CulinaryOdyssey', 'Resilience', 'Immersion', 'Spark', 'Marvel', 'Positivity', 'Kindness', 'Runway Creativity',
    'Friendship', 'Success', 'Exploration', 'Amazement', 'Romance', 'Captivation', 'Tranquility', 'Grandeur', 'Energy', 'Celebration', 
    'Charm', 'Ecstasy', 'Colorful', 'Hypnotic', 'Connection', 'Iconic', 'Journey', 'Engagement', 'Touched', 'Triumph', 
    'Heartwarming', 'Solace', 'Breakthrough', 'Joy in Baking', 'Envisioning History', 'Imagination', 'Vibrancy', 'Mesmerizing', 
    'Culinary Adventure', 'Winter Magic', 'Thrilling Journey', "Nature's Beauty", 'Celestial Wonder', 'Creative Inspiration', 
]

neg_words = [
    'Negative', 'Anger', 'Fear', 'Sadness', 'Disgust', 'Disappointed', 'Bitter', 'Shame', 'Despair', 'Grief', 'Loneliness', 'Jealousy', 'Bad',
    'Resentment', 'Frustration', 'Boredom', 'Anxiety', 'Intimidation', 'Helplessness', 'Envy', 'Regret', 'Numbness', 'Melancholy', 'Yearning',
    'Bitterness', 'Fearful', 'Apprehensive', 'Overwhelmed', 'Jealous', 'Devastated', 'Frustrated', 'Envious', 'Dismissive', 'Bittersweet', 'Hate',
    'Heartbreak', 'Betrayal', 'Suffering', 'EmotionalStorm', 'Isolation', 'Disappointment', 'LostLove', 'Exhaustion', 'Sorrow', 'Darkness', 'Sad',
    'Desperation', 'Ruins', 'Desolation', 'Loss', 'Heartache', 'Solitude', 'Obstacle', 'Pressure', 'Miscalculation', 'Challenge', 'Embarrassed',
]


def map_sentiment(val: str) -> int:
    if val in pos_words:
        return 0            #positive
    elif val in neg_words:
        return 1            #negative
    else:
        return 2            #neutral


def preprocessing_sentiment(df: pd.DataFrame) -> tuple:
    print("*** Preprocessing Sentiment dataset.csv ***")
    df = df.copy()

    df.drop(['Unnamed: 0.1', 'Unnamed: 0'], axis=1, inplace=True)
    df['Sentiment'] = df['Sentiment'].apply(lambda x: ' '.join(str(x).split()))
    df['target'] = df['Sentiment'].map(map_sentiment)
    
    print("Missing value:", df.isnull().sum().sum())
    print("Class Imbalance:\n", df['target'].value_counts())

    df['Hashtags'] = df['Hashtags'].apply(split_hashtags).apply(clean_text)
    df['cleaned_text'] = df['Text'].apply(split_hashtags).apply(clean_text)
    
    df['final_text'] = df['cleaned_text'] + ' ' + df['Hashtags']
    df['final_text'] = df['final_text'].apply(lambda x: ' '.join(x.split()))
    return df

def evaluate_model(y_pred, y_test):
    acc = accuracy_score(y_test, y_pred)
    F1_score = f1_score(y_test, y_pred, average='macro')
    report = classification_report(y_test, y_pred)
    
    print(f"Accuracy: {acc:.4f}")
    print(f"F1-Score: {F1_score}")
    print("\nClassification Report:\n", report)

df = pd.read_csv("/kaggle/input/datasets/elihaciyev/ml-intern/Sentiment dataset.csv")
df = preprocessing_sentiment(df)

X_train_text, X_test_text, y_train, y_test = train_test_split(
        df['final_text'], df['target'],
        test_size=0.2, random_state=42,
        stratify=df['target'], shuffle=True
)

tfidf = TfidfVectorizer(min_df=3, ngram_range=(1, 2), binary=True)
X_train = tfidf.fit_transform(X_train_text)
X_test = tfidf.transform(X_test_text)

print("\n", 30*"#", "Logistic Regression", "#"*30, "\n")
model = LogisticRegression(
    C = 10.0, max_iter=1000, class_weight='balanced', random_state=42
)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
evaluate_model(y_pred, y_test)


print("\n", 30*"#", "Random Forest Classifier", "#"*30, "\n")
model = RandomForestClassifier(
    n_estimators=1000,
    max_depth=6, 
    class_weight='balanced',
    random_state=42
)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
evaluate_model(y_pred, y_test)


print("\n", 30*"#", "Support Vector Classifier", "#"*30, "\n")
model = SVC(
    C=10.0, kernel = 'linear', class_weight='balanced', random_state=42
)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
evaluate_model(y_pred, y_test)


print("\n", 30*"#", "KNeighbors Classifier", "#"*30, "\n")
model = KNeighborsClassifier(n_neighbors=3)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
evaluate_model(y_pred, y_test)

f1_scores = []
k_range = range(1, 20) 

for k in k_range:
    model = KNeighborsClassifier(n_neighbors=k)
    model.fit(X_train, y_train)
    pred = model.predict(X_test)
    f1_scores.append(f1_score(y_test, pred, average='macro'))

plt.figure(figsize=(7, 5))
plt.plot(k_range, f1_scores, marker='o', color='blue', linestyle='--')
plt.title('Finding the Optimal k')
plt.xlabel('Value of k')
plt.ylabel('Test F1-Score')
plt.xticks(k_range)
plt.grid(True)
plt.show()


print('\n', 30*"#", 'Recommendations System', "#"*30, '\n')
tfidf = TfidfVectorizer(min_df=3, ngram_range=(1, 2), binary=True)
X_all = tfidf.fit_transform(df['final_text'])
cosine_sim = cosine_similarity(X_all, X_all)

def get_recommendations(post_index: int, 
                        df_texts: pd.Series,
                        cosine_sim,
                       top_n: int) -> pd.Series:
    
    sim_scores = list(enumerate(cosine_sim[post_index]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    
    top_indices = [i[0] for i in sim_scores[1:top_n + 1]]
    return df_texts.iloc[top_indices]


text_column = df['Text']
post_idx = 20
top_n = 5
print(f"Original text: {text_column.iloc[post_idx]}")
print("\nSimilar posts:\n")
recommendations = get_recommendations(post_idx, text_column, cosine_sim, top_n)
for i, text in enumerate(recommendations, 1):
    print(f"{i}. {text}")
