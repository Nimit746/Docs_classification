import re
import os
import numpy as np
import pandas as pd
import seaborn as sns
from collections import Counter
from wordcloud import WordCloud
import matplotlib.pyplot as plt

# Set style
sns.set_style('whitegrid')
plt.rcParams['figure.figsize'] = (12, 6)

class TextDatasetEDA:
    def __init__(self, folder_path):
        """Initialize with folder containing .txt files"""
        self.folder_path = folder_path
        self.data = []
        self.df = None
        
    def load_data(self):
        """Load all .txt files from the folder"""
        print("Loading text files...")
        for filename in os.listdir(self.folder_path):
            if filename.endswith('.txt'):
                file_path = os.path.join(self.folder_path, filename)
                try:
                    with open(file_path, 'r', encoding='utf-8') as f:
                        content = f.read()
                        self.data.append({
                            'filename': filename,
                            'content': content,
                            'char_count': len(content),
                            'word_count': len(content.split()),
                            'line_count': len(content.split('\n'))
                        })
                except Exception as e:
                    print(f"Error reading {filename}: {e}")
        
        self.df = pd.DataFrame(self.data)
        print(f"Loaded {len(self.data)} files successfully!")
        return self.df
    
    def basic_statistics(self):
        """Display basic statistics about the dataset"""
        print("\n" + "="*50)
        print("BASIC STATISTICS")
        print("="*50)
        print(f"Total files: {len(self.df)}")
        print(f"\nCharacter count statistics:")
        print(self.df['char_count'].describe())
        print(f"\nWord count statistics:")
        print(self.df['word_count'].describe())
        print(f"\nLine count statistics:")
        print(self.df['line_count'].describe())
        
    def plot_distributions(self):
        """Plot distributions of text lengths"""
        fig, axes = plt.subplots(1, 3, figsize=(15, 4))
        
        # Character count distribution
        axes[0].hist(self.df['char_count'], bins=30, color='skyblue', edgecolor='black')
        axes[0].set_title('Character Count Distribution')
        axes[0].set_xlabel('Characters')
        axes[0].set_ylabel('Frequency')
        
        # Word count distribution
        axes[1].hist(self.df['word_count'], bins=30, color='lightgreen', edgecolor='black')
        axes[1].set_title('Word Count Distribution')
        axes[1].set_xlabel('Words')
        axes[1].set_ylabel('Frequency')
        
        # Line count distribution
        axes[2].hist(self.df['line_count'], bins=30, color='salmon', edgecolor='black')
        axes[2].set_title('Line Count Distribution')
        axes[2].set_xlabel('Lines')
        axes[2].set_ylabel('Frequency')
        
        plt.tight_layout()
        plt.savefig('text_distributions.png', dpi=300, bbox_inches='tight')
        plt.show()
        print("Saved: text_distributions.png")
    
    def analyze_vocabulary(self):
        """Analyze vocabulary and word frequencies"""
        print("\n" + "="*50)
        print("VOCABULARY ANALYSIS")
        print("="*50)
        
        # Combine all text
        all_text = ' '.join(self.df['content'].tolist())
        
        # Clean and tokenize
        words = re.findall(r'\b[a-zA-Z]+\b', all_text.lower())
        
        # Word frequency
        word_freq = Counter(words)
        
        print(f"Total words: {len(words)}")
        print(f"Unique words: {len(word_freq)}")
        print(f"Vocabulary richness: {len(word_freq)/len(words):.4f}")
        
        print("\nTop 20 most common words:")
        for word, count in word_freq.most_common(20):
            print(f"{word:20} : {count:5}")
        
        return word_freq
    
    def plot_top_words(self, word_freq, top_n=20):
        """Plot top N most frequent words"""
        top_words = word_freq.most_common(top_n)
        words, counts = zip(*top_words)
        
        plt.figure(figsize=(12, 6))
        plt.bar(words, counts, color='steelblue', edgecolor='black')
        plt.title(f'Top {top_n} Most Frequent Words', fontsize=16, fontweight='bold')
        plt.xlabel('Words', fontsize=12)
        plt.ylabel('Frequency', fontsize=12)
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        plt.savefig('top_words.png', dpi=300, bbox_inches='tight')
        plt.show()
        print("Saved: top_words.png")
    
    def generate_wordcloud(self):
        """Generate word cloud from all text"""
        all_text = ' '.join(self.df['content'].tolist())
        
        wordcloud = WordCloud(width=800, height=400, 
                            background_color='white',
                            colormap='viridis').generate(all_text)
        
        plt.figure(figsize=(12, 6))
        plt.imshow(wordcloud, interpolation='bilinear')
        plt.axis('off')
        plt.title('Word Cloud', fontsize=16, fontweight='bold')
        plt.tight_layout()
        plt.savefig('wordcloud.png', dpi=300, bbox_inches='tight')
        plt.show()
        print("Saved: wordcloud.png")
    
    def check_missing_data(self):
        """Check for empty or problematic files"""
        print("\n" + "="*50)
        print("DATA QUALITY CHECK")
        print("="*50)
        
        empty_files = self.df[self.df['char_count'] == 0]
        print(f"Empty files: {len(empty_files)}")
        if len(empty_files) > 0:
            print("Empty file names:")
            print(empty_files['filename'].tolist())
        
        very_short = self.df[self.df['word_count'] < 10]
        print(f"\nFiles with < 10 words: {len(very_short)}")
        
    def export_summary(self, output_file='eda_summary.csv'):
        """Export summary to CSV"""
        self.df.to_csv(output_file, index=False)
        print(f"\nExported summary to: {output_file}")


# ============================================
# USAGE EXAMPLE
# ============================================
if __name__ == "__main__":
    # Specify your folder path
    FOLDER_PATH = "D:\Coding\MERN\Docs_classification\server\dataset"  # <-- CHANGE THIS
    
    # Initialize EDA
    eda = TextDatasetEDA(FOLDER_PATH)
    
    # Load data
    df = eda.load_data()
    
    # Display first few rows
    print("\nFirst 5 files:")
    print(df[['filename', 'char_count', 'word_count', 'line_count']].head())
    
    # Perform EDA
    eda.basic_statistics()
    eda.check_missing_data()
    eda.plot_distributions()
    
    word_freq = eda.analyze_vocabulary()
    eda.plot_top_words(word_freq, top_n=20)
    eda.generate_wordcloud()
    
    # Export summary
    eda.export_summary()
    
    print("\n" + "="*50)
    print("EDA COMPLETE!")
    print("="*50)