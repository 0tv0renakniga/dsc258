import pandas as pd
import ast
import re

def clean_value(val):
    """Safely cleans byte-strings, unicode prefixes, and evaluates dicts/booleans."""
    if pd.isna(val):
        return val
    
    if isinstance(val, str):
        val = val.strip()
        
        # Strip external double quotes if present (e.g., "b'...'")
        if val.startswith('"') and val.endswith('"'):
            val = val[1:-1]
            
        # Strip the b'...' or b"..." byte wrapper
        if val.startswith("b'") and val.endswith("'"):
            val = val[2:-1]
        elif val.startswith('b"') and val.endswith('"'):
            val = val[2:-1]
            
        # Remove Python 2 unicode prefixes (e.g., u'no' -> 'no')
        # We use regex to only target 'u' preceded by a word boundary or quote
        val = re.sub(r"(?<![a-zA-Z])u'([^']*)'", r"'\1'", val)
        val = re.sub(r'(?<![a-zA-Z])u"([^"]*)"', r'"\1"', val)
        
        # Try evaluating stringified dicts, booleans, or cleaned strings
        if val.startswith('{') or val in ['True', 'False', 'None'] or val.startswith("'"):
            try:
                return ast.literal_eval(val)
            except (ValueError, SyntaxError):
                pass
                
    return val

# 1. Load the data
df = pd.read_csv('test_with_labels.csv')

# 2. Drop the redundant aggregate 'attributes' column if it exists
if 'attributes' in df.columns:
    df = df.drop(columns=['attributes'])

# 3. Apply cleaning to all object columns (except review)
for col in df.select_dtypes(include=['object']).columns:
    if col != 'review':
        df[col] = df[col].apply(clean_value)

# 4. Flatten dictionary columns
dict_cols = [c for c in df.columns if df[c].apply(lambda x: isinstance(x, dict)).any()]

for col in dict_cols:
    # Normalize dicts; missing values become NaNs in the new columns
    flattened = pd.json_normalize(df[col].apply(lambda x: x if isinstance(x, dict) else {}))
    flattened = flattened.add_prefix(f"{col}_")
    
    # Drop original and append new
    df = pd.concat([df.drop(columns=[col]), flattened], axis=1)

# 5. Handle Missing Values Carefully
if 'review' in df.columns:
    df['review'] = df['review'].fillna('')

# Fill categorical/string columns with 'Missing' and numeric/boolean with -1
for col in df.columns:
    if df[col].dtype == 'object' and col != 'review':
        df[col] = df[col].fillna('Missing')
    elif col != 'review':
        df[col] = df[col].fillna(-1)

# 6. Save
df.to_csv('test_cleaned.csv', index=False)
print(f"Cleaned dataset shape: {df.shape}")
