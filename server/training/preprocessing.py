import re
import os
import nltk
import string
import numpy as np
import pandas as pd
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer, PorterStemmer

# Download required NLTK data
try:
    nltk.download('stopwords', quiet=True)
    nltk.download('punkt', quiet=True)
    nltk.download('wordnet', quiet=True)  # Lemmatizes convert words inito dictionary format
    nltk.download('averaged_perceptron_tagger', quiet=True)
except:
    pass

class TextPreprocessor:
    def __init__(self, folder_path, min_words=10, max_words=10000):
        """
        Initialize preprocessor
        
        Args:
            folder_path: Path to folder with .txt files
            min_words: Minimum word count to keep document
            max_words: Maximum word count to keep document
        """
        self.folder_path = folder_path
        self.min_words = min_words
        self.max_words = max_words
        self.stop_words = set(stopwords.words('english'))
        self.lemmatizer = WordNetLemmatizer()
        self.stemmer = PorterStemmer()
        
    def clean_text(self, text, method='lemmatize'):
        """
        Comprehensive text cleaning
        
        Args:
            text: Input text string
            method: 'lemmatize' or 'stem' or 'none'
        """
        # 1. Lowercase
        text = text.lower()
        
        # 2. Remove URLs
        text = re.sub(r'http\S+|www\S+|https\S+', '', text)
        
        # 3. Remove email addresses
        text = re.sub(r'\S+@\S+', '', text)
        
        # 4. Remove special characters and digits (keep spaces)
        text = re.sub(r'[^a-zA-Z\s]', ' ', text)
        
        # 5. Remove extra whitespace
        text = re.sub(r'\s+', ' ', text).strip()
        
        # 6. Tokenize
        tokens = text.split()
        
        # 7. Remove stop words and short words
        tokens = [word for word in tokens if word not in self.stop_words and len(word) > 2]
        
        # 8. Lemmatize or Stem
        if method == 'lemmatize':
            tokens = [self.lemmatizer.lemmatize(word) for word in tokens]
        elif method == 'stem':
            tokens = [self.stemmer.stem(word) for word in tokens]
        
        return ' '.join(tokens)
    
    def filter_by_length(self, text):
        """Check if document meets length requirements"""
        word_count = len(text.split())
        return self.min_words <= word_count <= self.max_words
    
    def remove_duplicates(self, df):
        """Remove duplicate documents"""
        original_count = len(df)
        df = df.drop_duplicates(subset=['cleaned_text'], keep='first')
        removed = original_count - len(df)
        print(f"Removed {removed} duplicate documents")
        return df
    
    def preprocess_dataset(self, method='lemmatize', remove_dupes=True):
        """
        Main preprocessing pipeline
        
        Args:
            method: 'lemmatize', 'stem', or 'none'
            remove_dupes: Whether to remove duplicate texts
        """
        print("="*60)
        print("STARTING PREPROCESSING PIPELINE")
        print("="*60)
        
        data = []
        skipped_files = []
        
        # Load and clean files
        print("\n1. Loading and cleaning files...")
        for filename in os.listdir(self.folder_path):
            if filename.endswith('.txt'):
                file_path = os.path.join(self.folder_path, filename)
                try:
                    with open(file_path, 'r', encoding='utf-8') as f:
                        content = f.read()
                        
                    # Clean text
                    cleaned = self.clean_text(content, method=method)
                    
                    # Check length filter
                    if self.filter_by_length(cleaned):
                        data.append({
                            'filename': filename,
                            'original_text': content,
                            'cleaned_text': cleaned,
                            'word_count': len(cleaned.split()),
                            'char_count': len(cleaned)
                        })
                    else:
                        skipped_files.append((filename, len(cleaned.split())))
                        
                except Exception as e:
                    print(f"Error processing {filename}: {e}")
        
        df = pd.DataFrame(data)
        print(f"   ✓ Processed {len(df)} files successfully")
        print(f"   ✗ Skipped {len(skipped_files)} files (length filter)")
        
        # Remove duplicates
        if remove_dupes:
            print("\n2. Removing duplicates...")
            df = self.remove_duplicates(df)
        
        # Summary statistics
        print("\n3. Final Dataset Statistics:")
        print(f"   Total documents: {len(df)}")
        print(f"   Word count - Mean: {df['word_count'].mean():.1f}, "
              f"Median: {df['word_count'].median():.1f}")
        print(f"   Word count - Min: {df['word_count'].min()}, "
              f"Max: {df['word_count'].max()}")
        
        return df, skipped_files
    
    def save_processed_data(self, df, output_folder='processed_data'):
        """Save cleaned data"""
        os.makedirs(output_folder, exist_ok=True)
        
        # Save as CSV
        csv_path = os.path.join(output_folder, 'processed_texts.csv')
        df[['filename', 'cleaned_text', 'word_count']].to_csv(csv_path, index=False)
        print(f"\n✓ Saved CSV to: {csv_path}")
        
        # Save individual cleaned files
        txt_folder = os.path.join(output_folder, 'cleaned_texts')
        os.makedirs(txt_folder, exist_ok=True)
        
        for idx, row in df.iterrows():
            txt_path = os.path.join(txt_folder, f"cleaned_{row['filename']}")
            with open(txt_path, 'w', encoding='utf-8') as f:
                f.write(row['cleaned_text'])
        
        print(f"✓ Saved {len(df)} cleaned text files to: {txt_folder}")
        
        return csv_path
    
    def get_vocabulary_stats(self, df):
        """Get vocabulary statistics after preprocessing"""
        all_words = ' '.join(df['cleaned_text'].tolist()).split()
        unique_words = set(all_words)
        
        print("\n4. Vocabulary Statistics:")
        print(f"   Total words: {len(all_words):,}")
        print(f"   Unique words: {len(unique_words):,}")
        print(f"   Vocabulary richness: {len(unique_words)/len(all_words):.4f}")
        
        return all_words, unique_words


# ============================================
# USAGE EXAMPLE
# ============================================
if __name__ == "__main__":
    # Configuration
    FOLDER_PATH = "D:\\Coding\\MERN\\Docs_classification\\server\\dataset"
    MIN_WORDS = 50      # Adjust based on your needs
    MAX_WORDS = 5000    # Adjust based on your needs
    
    # Initialize preprocessor
    preprocessor = TextPreprocessor(
        folder_path=FOLDER_PATH,
        min_words=MIN_WORDS,
        max_words=MAX_WORDS
    )
    
    # Run preprocessing
    df, skipped = preprocessor.preprocess_dataset(
        method='lemmatize',  # Options: 'lemmatize', 'stem', 'none'
        remove_dupes=True
    )
    
    # Get vocabulary stats
    words, vocab = preprocessor.get_vocabulary_stats(df)
    
    # Save processed data
    preprocessor.save_processed_data(df, output_folder='processed_data')
    
    # Display sample
    print("\n" + "="*60)
    print("SAMPLE CLEANED TEXT")
    print("="*60)
    print(f"\nOriginal:\n{df.iloc[0]['original_text'][:300]}...")
    print(f"\nCleaned:\n{df.iloc[0]['cleaned_text'][:300]}...")
    
    print("\n" + "="*60)
    print("PREPROCESSING COMPLETE!")
    print("="*60)