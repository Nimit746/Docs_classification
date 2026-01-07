import os
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC
from sklearn.metrics import (classification_report, confusion_matrix, 
                            accuracy_score, precision_recall_fscore_support)
from sklearn.preprocessing import LabelEncoder
import warnings
warnings.filterwarnings('ignore')

sns.set_style('whitegrid')


class SVMTrainer:
    def __init__(self):
        """Initialize SVM Trainer"""
        self.vectorizer = None
        self.model = None
        self.label_encoder = LabelEncoder()
        self.class_names = None
        
    def load_preprocessed_csv(self, csv_path, text_col='cleaned_text', label_col='label'):
        """
        Load preprocessed data from CSV
        
        Args:
            csv_path: Path to CSV file
            text_col: Name of text column
            label_col: Name of label column
        """
        print("="*60)
        print("LOADING PREPROCESSED DATA")
        print("="*60)
        
        df = pd.read_csv(csv_path)
        print(f"Loaded {len(df)} documents from {csv_path}")
        
        # Check if label column exists
        if label_col not in df.columns:
            print(f"\n⚠ Warning: '{label_col}' column not found!")
            print(f"Available columns: {df.columns.tolist()}")
            
            # Try to extract label from filename
            if 'filename' in df.columns:
                print("\nAttempting to extract labels from 'filename' column...")
                df['label'] = df['filename'].apply(lambda x: x.split('_')[0] if '_' in x else 'unknown')
                print(f"Extracted labels: {df['label'].unique()}")
        
        print(f"\nClasses found: {df[label_col].unique().tolist()}")
        print(f"\nClass distribution:")
        print(df[label_col].value_counts())
        
        return df, text_col, label_col
    
    def load_from_folder_structure(self, dataset_path):
        """
        Load data from folder structure (alternative method)
        
        dataset/
        ├── class1/
        │   ├── file1.txt
        │   └── file2.txt
        └── class2/
            ├── file1.txt
            └── file2.txt
        """
        print("="*60)
        print("LOADING FROM FOLDER STRUCTURE")
        print("="*60)
        
        data = []
        for class_name in os.listdir(dataset_path):
            class_path = os.path.join(dataset_path, class_name)
            
            if os.path.isdir(class_path):
                print(f"Loading class: {class_name}")
                for filename in os.listdir(class_path):
                    if filename.endswith('.txt'):
                        file_path = os.path.join(class_path, filename)
                        try:
                            with open(file_path, 'r', encoding='utf-8') as f:
                                content = f.read()
                                data.append({
                                    'filename': filename,
                                    'cleaned_text': content,
                                    'label': class_name
                                })
                        except Exception as e:
                            print(f"  Error reading {filename}: {e}")
        
        df = pd.DataFrame(data)
        print(f"\nLoaded {len(df)} documents")
        print(f"Classes: {df['label'].unique().tolist()}")
        print(f"\nClass distribution:")
        print(df['label'].value_counts())
        
        return df, 'cleaned_text', 'label'
    
    def visualize_class_distribution(self, df, label_col, save_path='class_distribution.png'):
        """Visualize class distribution"""
        plt.figure(figsize=(12, 6))
        class_counts = df[label_col].value_counts()
        
        colors = sns.color_palette('Set2', n_colors=len(class_counts))
        bars = plt.bar(range(len(class_counts)), class_counts.values, 
                      color=colors, edgecolor='black', linewidth=1.5)
        
        plt.xticks(range(len(class_counts)), class_counts.index, rotation=45, ha='right')
        plt.title('Class Distribution', fontsize=16, fontweight='bold', pad=20)
        plt.xlabel('Class', fontsize=12)
        plt.ylabel('Number of Documents', fontsize=12)
        plt.grid(axis='y', alpha=0.3)
        
        # Add value labels on bars
        for i, (bar, count) in enumerate(zip(bars, class_counts.values)):
            plt.text(i, count + class_counts.max()*0.01, 
                    f'{int(count)}', ha='center', va='bottom', 
                    fontsize=10, fontweight='bold')
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
        print(f"✓ Saved class distribution: {save_path}")
    
    def extract_features(self, texts, max_features=5000, ngram_range=(1, 2)):
        """
        Convert texts to TF-IDF features
        
        Args:
            texts: List or Series of text documents
            max_features: Maximum number of features
            ngram_range: N-gram range (1,1) for unigrams, (1,2) for unigrams+bigrams
        """
        print("\n" + "="*60)
        print("FEATURE EXTRACTION")
        print("="*60)
        
        self.vectorizer = TfidfVectorizer(
            max_features=max_features,
            ngram_range=ngram_range,
            min_df=2,            # Ignore terms that appear in < 2 documents
            max_df=0.8,          # Ignore terms that appear in > 80% of documents
            sublinear_tf=True,   # Use log scaling for term frequency
            strip_accents='unicode',
            lowercase=True
        )
        
        X = self.vectorizer.fit_transform(texts)
        
        print(f"✓ Feature matrix shape: {X.shape}")
        print(f"  - Documents: {X.shape[0]}")
        print(f"  - Features: {X.shape[1]}")
        print(f"  - Sparsity: {(1 - X.nnz / (X.shape[0] * X.shape[1]))*100:.2f}%")
        
        return X
    
    def prepare_labels(self, labels):
        """Encode labels to numeric format"""
        y = self.label_encoder.fit_transform(labels)
        self.class_names = self.label_encoder.classes_
        
        print(f"✓ Encoded {len(self.class_names)} classes: {self.class_names.tolist()}")
        
        return y
    
    def train_with_grid_search(self, X_train, y_train, cv=5):
        """Train SVM with hyperparameter tuning"""
        print("\n" + "="*60)
        print("TRAINING SVM WITH GRID SEARCH")
        print("="*60)
        
        param_grid = {
            'C': [0.1, 1, 10, 100],
            'kernel': ['linear', 'rbf'],
            'gamma': ['scale', 'auto'],
            'class_weight': [None, 'balanced']
        }
        
        print("Searching best parameters...")
        print(f"Grid size: {np.prod([len(v) for v in param_grid.values()])} combinations")
        print(f"Cross-validation folds: {cv}")
        
        grid_search = GridSearchCV(
            SVC(random_state=42, probability=True),
            param_grid,
            cv=cv,
            scoring='accuracy',
            n_jobs=-1,
            verbose=2
        )
        
        grid_search.fit(X_train, y_train)
        self.model = grid_search.best_estimator_
        
        print(f"\n✓ Best parameters found:")
        for param, value in grid_search.best_params_.items():
            print(f"  - {param}: {value}")
        print(f"\n✓ Best cross-validation score: {grid_search.best_score_:.4f}")
        
        return grid_search
    
    def train_fast(self, X_train, y_train):
        """Train SVM with default parameters (fast)"""
        print("\n" + "="*60)
        print("TRAINING SVM (FAST MODE)")
        print("="*60)
        
        self.model = SVC(
            kernel='linear',
            C=1.0,
            class_weight='balanced',
            random_state=42,
            probability=True
        )
        
        print("Training model...")
        self.model.fit(X_train, y_train)
        print("✓ Training complete!")
        
        return self.model
    
    def evaluate(self, X_test, y_test, save_cm='confusion_matrix.png'):
        """Comprehensive model evaluation"""
        print("\n" + "="*60)
        print("MODEL EVALUATION")
        print("="*60)
        
        # Predictions
        y_pred = self.model.predict(X_test)
        
        # Overall accuracy
        accuracy = accuracy_score(y_test, y_pred)
        print(f"\n📊 Overall Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)\n")
        
        # Per-class metrics
        print("Classification Report:")
        print("-" * 60)
        report = classification_report(
            y_test, y_pred, 
            target_names=self.class_names,
            digits=4
        )
        print(report)
        
        # Confusion matrix
        cm = confusion_matrix(y_test, y_pred)
        self.plot_confusion_matrix(cm, save_cm)
        
        # Per-class accuracy
        print("\nPer-Class Accuracy:")
        print("-" * 60)
        for i, class_name in enumerate(self.class_names):
            class_mask = y_test == i
            class_acc = accuracy_score(y_test[class_mask], y_pred[class_mask])
            print(f"{class_name:20s}: {class_acc:.4f} ({class_acc*100:.2f}%)")
        
        return accuracy, y_pred
    
    def plot_confusion_matrix(self, cm, save_path):
        """Plot and save confusion matrix"""
        plt.figure(figsize=(10, 8))
        
        # Normalize confusion matrix
        cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        
        # Create heatmap
        sns.heatmap(
            cm_normalized, 
            annot=cm,  # Show counts
            fmt='d',
            cmap='Blues',
            xticklabels=self.class_names,
            yticklabels=self.class_names,
            cbar_kws={'label': 'Normalized Value'},
            square=True
        )
        
        plt.title('Confusion Matrix', fontsize=16, fontweight='bold', pad=20)
        plt.ylabel('True Label', fontsize=12)
        plt.xlabel('Predicted Label', fontsize=12)
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
        print(f"\n✓ Saved confusion matrix: {save_path}")
    
    def cross_validate(self, X, y, cv=5):
        """Perform cross-validation"""
        print("\n" + "="*60)
        print("CROSS-VALIDATION")
        print("="*60)
        
        print(f"Performing {cv}-fold cross-validation...")
        scores = cross_val_score(self.model, X, y, cv=cv, scoring='accuracy', n_jobs=-1)
        
        print(f"\nCV Scores: {scores}")
        print(f"Mean CV Score: {scores.mean():.4f} (+/- {scores.std() * 2:.4f})")
        
        return scores
    
    def save_model(self, save_dir='models'):
        """Save model, vectorizer, and label encoder"""
        os.makedirs(save_dir, exist_ok=True)
        
        # Save model
        model_path = os.path.join(save_dir, 'svm_model.pkl')
        with open(model_path, 'wb') as f:
            pickle.dump(self.model, f)
        
        # Save vectorizer
        vectorizer_path = os.path.join(save_dir, 'tfidf_vectorizer.pkl')
        with open(vectorizer_path, 'wb') as f:
            pickle.dump(self.vectorizer, f)
        
        # Save label encoder
        encoder_path = os.path.join(save_dir, 'label_encoder.pkl')
        with open(encoder_path, 'wb') as f:
            pickle.dump(self.label_encoder, f)
        
        print("\n" + "="*60)
        print("MODEL SAVED")
        print("="*60)
        print(f"✓ Model: {model_path}")
        print(f"✓ Vectorizer: {vectorizer_path}")
        print(f"✓ Label encoder: {encoder_path}")


# ============================================
# MAIN TRAINING PIPELINE
# ============================================
def main():
    print("\n" + "="*60)
    print("   SVM TEXT CLASSIFIER - TRAINING PIPELINE")
    print("="*60 + "\n")
    
    # ==================== CONFIGURATION ====================
    # Option 1: Use preprocessed CSV
    USE_CSV = True
    CSV_PATH = 'processed_data/processed_texts.csv'
    TEXT_COL = 'cleaned_text'  # Column name with text
    LABEL_COL = 'label'        # Column name with labels
    
    # Option 2: Load from folder structure
    DATASET_PATH = 'dataset'
    
    # Training parameters
    TEST_SIZE = 0.2
    RANDOM_STATE = 42
    USE_GRID_SEARCH = True  # Set False for faster training
    MAX_FEATURES = 5000
    NGRAM_RANGE = (1, 2)    # (1,1) for unigrams only, (1,2) for unigrams+bigrams
    
    # ==================== INITIALIZE ====================
    trainer = SVMTrainer()
    
    # ==================== LOAD DATA ====================
    if USE_CSV and os.path.exists(CSV_PATH):
        df, text_col, label_col = trainer.load_preprocessed_csv(
            CSV_PATH, TEXT_COL, LABEL_COL
        )
    else:
        print(f"⚠ CSV not found at {CSV_PATH}, loading from folder structure...")
        df, text_col, label_col = trainer.load_from_folder_structure(DATASET_PATH)
    
    # ==================== VISUALIZE ====================
    trainer.visualize_class_distribution(df, label_col)
    
    # ==================== PREPARE DATA ====================
    X = trainer.extract_features(
        df[text_col], 
        max_features=MAX_FEATURES,
        ngram_range=NGRAM_RANGE
    )
    y = trainer.prepare_labels(df[label_col])
    
    # ==================== SPLIT DATA ====================
    print("\n" + "="*60)
    print("DATA SPLITTING")
    print("="*60)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, 
        test_size=TEST_SIZE, 
        random_state=RANDOM_STATE, 
        stratify=y
    )
    print(f"Training samples: {X_train.shape[0]} ({(1-TEST_SIZE)*100:.0f}%)")
    print(f"Testing samples:  {X_test.shape[0]} ({TEST_SIZE*100:.0f}%)")
    
    # ==================== TRAIN MODEL ====================
    if USE_GRID_SEARCH:
        grid_search = trainer.train_with_grid_search(X_train, y_train, cv=5)
    else:
        trainer.train_fast(X_train, y_train)
    
    # ==================== EVALUATE ====================
    accuracy, y_pred = trainer.evaluate(X_test, y_test)
    
    # ==================== CROSS-VALIDATE ====================
    trainer.cross_validate(X, y, cv=5)
    
    # ==================== SAVE MODEL ====================
    trainer.save_model(save_dir='models')
    
    # ==================== SUMMARY ====================
    print("\n" + "="*60)
    print("   TRAINING COMPLETE!")
    print("="*60)
    print(f"✓ Final Test Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")
    print(f"✓ Models saved to: models/")
    print(f"✓ Ready for prediction!")
    print("="*60 + "\n")


if __name__ == "__main__":
    main()