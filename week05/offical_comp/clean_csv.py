import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import pandas as pd
import ast
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer

# Download necessary tools
nltk.download('stopwords')
nltk.download('wordnet')

stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()

# read csv files
#df = pd.read_csv("train.csv")
df = pd.read_csv("test_with_labels.csv")

# 1. Drop attributes immediately
df.drop(columns=['attributes'], inplace=True)

def simple_clean(val):
    if pd.isna(val):
        return val

    # Convert to string and strip byte/unicode artifacts
    s = str(val).strip()
    if s.startswith(("b'", 'b"', "u'", 'u"')):
        # Remove b or u prefix
        s = s[2:] if s.startswith(('b', 'u')) else s
        # Remove trailing quotes
        s = s.rstrip("'").rstrip('"')

    # Specific fix for u'value' inside strings that aren't dicts
    if isinstance(s, str) and s.startswith("u'"):
        s = s[2:].rstrip("'")

    # Only use literal_eval if it's a stringified dict/list
    if isinstance(s, str) and '{' in s and '}' in s:
        try:
            return ast.literal_eval(s)
        except:
            return s
    return s

# Apply cleaning to everything
df = df.map(simple_clean)

# 2. Flatten any column that contains dictionaries
cols_to_check = df.columns
for col in cols_to_check:
    # Check if the first valid entry is a dictionary
    sample = df[col].dropna().iloc[0] if not df[col].dropna().empty else None
    
    if isinstance(sample, dict):
        # Expand dict to individual columns with .is_ prefix
        flattened = pd.json_normalize(df[col])
        flattened.columns = [f"{col}.is_{c}" for c in flattened.columns]
        
        # Replace original column with flattened ones
        df = pd.concat([df.drop(columns=[col]), flattened], axis=1)

# 3. Final conversion: True -> 1, False -> 0, NaN -> -1
# We map the actual Booleans and the string representations
map_dict = {
    True: 1, 'True': 1, '1': 1, 1: 1,
    False: 0, 'False': 0, '0': 0, 0: 0,
    'nan': -1, 'None': -1
}

# Fill actual NaNs with -1 first, then replace based on the dict
df = df.fillna(-1).replace(map_dict)
"""
for i,j in enumerate(df.columns):
    print(j)
    print(df.iloc[0,i])
    print(type(df.iloc[0,i]))
    print(df[j].value_counts())
    #for lab in df.label.unique():
    #    print(lab)
    #    print(df[df["label"]==lab][j].value_counts())


"""
int_label = {
    'american (traditional)': 1, 
    'mexican': 2,
    'chinese': 3,
    'italian': 4,
    'thai': 5,
    'american (new)': 6, 
    'asian fusion': 7, 
    'mediterranean': 8, 
    'canadian (new)': 9,
    'japanese': 10
}

df['int_col'] = df['label'].map(int_label)

custom_stops = {
    # Logistics, Staff & Transactional
    'order', 'ordered', 'takeout', 'delivery', 'menu', 'price', 'bill', 'check',
    'minutes', 'wait', 'waiting', 'server', 'waiter', 'waitress', 'staff', 
    'management', 'manager', 'owner', 'bartender', 'service', 'business', 
    'counter', 'customer', 'line', 'reservation', 'tip', 'dollars', 'money', 
    'worth', 'expensive', 'cheap', 'pricey', 'cost', 'togo', 'pickup',
    
    # Narrative, Frequency & Reviewer Verbs
    'ive', 'dont', 'didnt', 'wasnt', 'cant', 'couldnt', 'get', 'got', 'give', 
    'gave', 'given', 'star', 'stars', 'rating', 'review', 'reviewing', 'comment', 
    'told', 'said', 'asked', 'went', 'came', 'back', 'told', 'try', 'tried', 
    'trying', 'look', 'looked', 'looking', 'make', 'makes', 'made', 'take', 
    'took', 'want', 'wanted', 'let', 'know', 'think', 'thought', 'feel', 'felt',
    'always', 'never', 'ever', 'sometimes', 'often', 'recently', 'today', 
    'night', 'weekend', 'time', 'times', 'day', 'days', 'year', 'years',
    
    # Generic Sentiment & Adjectives (Scrubbed to force feature importance to food)
    'good', 'great', 'amazing', 'best', 'nice', 'bad', 'disappointed', 'happy', 
    'favorite', 'fantastic', 'excellent', 'terrible', 'awesome', 'wonderful', 
    'perfectly', 'decent', 'special', 'tasty', 'delicious', 'yummy', 'flavorful',
    'pretty', 'definitely', 'actually', 'personally', 'maybe', 'probably', 
    'super', 'massive', 'huge', 'little', 'bit', 'lot', 'lots', 'selection',
    
    # People & Entities
    'people', 'person', 'someone', 'somebody', 'guy', 'girl', 'kid', 'kids', 
    'boy', 'boys', 'husband', 'wife', 'family', 'friend', 'friends', 'daughter', 
    'neighbor', 'neighbors', 'coworker', 'coworkers', 'everyone', 'everybody',
    
    # Spatial, Decor & Location (Noise for Cuisine Detection)
    'restaurant', 'place', 'location', 'spot', 'area', 'valley', 'house', 'home', 
    'store', 'inside', 'outside', 'patio', 'bar', 'table', 'tables', 'booth', 
    'booths', 'seating', 'seatings', 'atmosphere', 'ambience', 'decor', 'decoration', 
    'fixture', 'fixtures', 'interior', 'design', 'modern', 'clean', 'dirty',

    # extras
    'butread', 'hometook', 'incase', 'wthe', 'onnwhat', 'nmin', 'nnso', 'exactly',
    'expected', 'given', 'actually', 'forever', 'blown', 'around', 'expecting',
    'size', 'portion', 'giant', 'together', 'half', 'double', 'mass', 'total', 'less',
    'maybe', 'nothing', 'kudos', 'short', 'many'
}

all_stops = stop_words.union(custom_stops)

def clean_review(text):
    if not isinstance(text, str): return ""

    # 1. Lowercase and handle specific artifacts
    text = text.lower()
    
    # 2. Advanced Regex Cleanup
    # Collapses newlines, returns, hex codes, and your specific CSV artifacts
    text = re.sub(r'\\x[0-9a-fA-F]+', ' ', text)
    text = re.sub(r'\\n+|\\r+|nnthey|nnim|nnso|onnnwhat', ' ', text)
    # Remove b' and u' prefixes
    text = text.replace("u'", " ").replace("b'", " ")
    
    # 3. Remove punctuation and numbers
    text = re.sub(r'[^a-z\s]', '', text)

    # 4. Tokenize, Lemmatize, and Filter
    words = text.split()
    cleaned = []
    for w in words:
        # Lemmatize first so we can catch plurals of stop words
        lemma = lemmatizer.lemmatize(w)
        if lemma not in all_stops and len(lemma) > 2:
            cleaned.append(lemma)

    return " ".join(cleaned)
# Apply to the dataframe
df['review_cleaned'] = df['review'].apply(clean_review)
df.to_csv("test_clean.csv")
for i in  df[df["label"].isin(["asian fusion","canadian (new)","american (new)","american (traditional)"])]['review_cleaned'][:20]:
    print(i)

# 5. Feature Importance Logic
# Drop non-numeric features and target columns for the X matrix
specific_importances = {}

labels = {
    1: 'american (traditional)', 2: 'mexican', 3: 'chinese', 4: 'italian',
    5: 'thai', 6: 'american (new)', 7: 'asian fusion', 8: 'mediterranean',
    9: 'canadian (new)', 10: 'japanese'
}

# 5. Feature Importance Logic
# Drop non-numeric features and target columns for the X matrix
#X = df.drop(columns=['label', 'int_col','id', 'review', 'business_id', 'name', 'address', 'city', 'state', 'latitude', 'longitude', 'postal_code'], errors='ignore')
X = df.drop(columns=['label', 'int_col','id', 'review', 'business_id', 'name', 'address'], errors='ignore')

# Ensure all data in X is numeric (convert any missed strings to codes)
X = X.apply(pd.to_numeric, errors='coerce').fillna(-1)
y = df['int_col']

# 1. Initialize the vectorizer
# ngram_range=(1, 3) tells it to grab unigrams, bigrams, and trigrams
# max_features=1000 prevents the dataframe from getting too massive/slow
tfidf = TfidfVectorizer(
    ngram_range=(1, 3), 
    max_features=3000,
    min_df=5,
    max_df=0.5
)


# 2. Create the new columns
tfidf_matrix = tfidf.fit_transform(df['review_cleaned'])

# 3. Convert to a DataFrame with actual word names as headers
tfidf_df = pd.DataFrame(tfidf_matrix.toarray(), columns=tfidf.get_feature_names_out())

# 4. Join this to your original X (the one without lat/long)
X_final = pd.concat([X.reset_index(drop=True), tfidf_df], axis=1)

for val, name in labels.items():
    # Create a binary target: 1 if it is the current label, 0 otherwise
    y_binary = (df['int_col'] == val).astype(int)
    
    if y_binary.sum() > 0:  # Ensure the label exists in the data
        clf = RandomForestClassifier(n_estimators=50, random_state=42)
        clf.fit(X_final, y_binary)
        
        # Save top 5 features for this specific category
        feat_imp = pd.Series(clf.feature_importances_, index=X_final.columns).sort_values(ascending=False)
        specific_importances[name] = feat_imp.head(30)

# Display specific drivers for a few examples

for category in int_label.keys():
    print(f"\n--- Top Drivers for {category.upper()} ---")
    print(specific_importances.get(category, "No data"))


