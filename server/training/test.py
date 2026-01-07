import os
import pickle
import pandas as pd
import numpy as np


class SVMPredictor:
    def __init__(self, model_dir='models'):
        """
        Initialize SVM Predictor
        
        Args:
            model_dir: Directory containing saved model files
        """
        self.model_dir = model_dir
        self.model = None
        self.vectorizer = None
        self.label_encoder = None
        
        self._load_models()
    
    def _load_models(self):
        """Load all required model components"""
        print("="*60)
        print("LOADING TRAINED MODEL")
        print("="*60)
        
        try:
            # Load SVM model
            model_path = os.path.join(self.model_dir, 'svm_model.pkl')
            with open(model_path, 'rb') as f:
                self.model = pickle.load(f)
            print(f"✓ Model loaded: {model_path}")
            
            # Load TF-IDF vectorizer
            vectorizer_path = os.path.join(self.model_dir, 'tfidf_vectorizer.pkl')
            with open(vectorizer_path, 'rb') as f:
                self.vectorizer = pickle.load(f)
            print(f"✓ Vectorizer loaded: {vectorizer_path}")
            
            # Load label encoder
            encoder_path = os.path.join(self.model_dir, 'label_encoder.pkl')
            with open(encoder_path, 'rb') as f:
                self.label_encoder = pickle.load(f)
            print(f"✓ Label encoder loaded: {encoder_path}")
            
            print(f"\n📋 Available classes: {self.label_encoder.classes_.tolist()}")
            print("="*60 + "\n")
            
        except FileNotFoundError as e:
            print(f"\n❌ Error: Model files not found!")
            print(f"Make sure you have trained the model first using train_svm.py")
            print(f"Looking in: {self.model_dir}/")
            raise e
    
    def predict(self, text, return_probabilities=False):
        """
        Predict class for a single text
        
        Args:
            text: Input text (can be raw or preprocessed)
            return_probabilities: If True, return class probabilities
        
        Returns:
            predicted_class (and probabilities if requested)
        """
        # Transform to features
        features = self.vectorizer.transform([text])
        
        # Predict
        prediction = self.model.predict(features)[0]
        predicted_class = self.label_encoder.inverse_transform([prediction])[0]
        
        if return_probabilities:
            # Get prediction probabilities
            probabilities = self.model.predict_proba(features)[0]
            prob_dict = {
                class_name: prob 
                for class_name, prob in zip(self.label_encoder.classes_, probabilities)
            }
            return predicted_class, prob_dict
        
        return predicted_class
    
    def predict_file(self, file_path, return_probabilities=False):
        """
        Predict class for a text file
        
        Args:
            file_path: Path to .txt file
            return_probabilities: If True, return class probabilities
        """
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                text = f.read()
            return self.predict(text, return_probabilities)
        except Exception as e:
            print(f"Error reading file {file_path}: {e}")
            return None
    
    def predict_batch(self, texts, return_probabilities=False):
        """
        Predict classes for multiple texts
        
        Args:
            texts: List of text strings
            return_probabilities: If True, return class probabilities
        
        Returns:
            predictions (and probabilities if requested)
        """
        # Transform all texts
        features = self.vectorizer.transform(texts)
        
        # Predict
        predictions = self.model.predict(features)
        predicted_classes = self.label_encoder.inverse_transform(predictions)
        
        if return_probabilities:
            probabilities = self.model.predict_proba(features)
            results = []
            for pred_class, probs in zip(predicted_classes, probabilities):
                prob_dict = {
                    class_name: prob 
                    for class_name, prob in zip(self.label_encoder.classes_, probs)
                }
                results.append((pred_class, prob_dict))
            return results
        
        return predicted_classes.tolist()
    
    def predict_folder(self, folder_path, output_csv='predictions.csv', 
                      show_probabilities=True):
        """
        Predict classes for all .txt files in a folder
        
        Args:
            folder_path: Path to folder containing .txt files
            output_csv: Path to save predictions
            show_probabilities: Include probability scores in output
        """
        print("="*60)
        print(f"PREDICTING FILES IN: {folder_path}")
        print("="*60 + "\n")
        
        if not os.path.exists(folder_path):
            print(f"❌ Error: Folder not found: {folder_path}")
            return None
        
        results = []
        file_count = 0
        
        for filename in sorted(os.listdir(folder_path)):
            if filename.endswith('.txt'):
                file_path = os.path.join(folder_path, filename)
                file_count += 1
                
                try:
                    if show_probabilities:
                        predicted_class, probabilities = self.predict_file(
                            file_path, return_probabilities=True
                        )
                        
                        result = {
                            'filename': filename,
                            'predicted_class': predicted_class,
                            'confidence': max(probabilities.values())
                        }
                        
                        # Add individual class probabilities
                        for class_name, prob in probabilities.items():
                            result[f'prob_{class_name}'] = prob
                        
                        results.append(result)
                        print(f"✓ {filename:40s} → {predicted_class:15s} "
                              f"(confidence: {max(probabilities.values()):.3f})")
                    else:
                        predicted_class = self.predict_file(file_path)
                        results.append({
                            'filename': filename,
                            'predicted_class': predicted_class
                        })
                        print(f"✓ {filename:40s} → {predicted_class}")
                        
                except Exception as e:
                    print(f"❌ {filename:40s} → Error: {e}")
                    results.append({
                        'filename': filename,
                        'predicted_class': 'ERROR',
                        'error': str(e)
                    })
        
        if not results:
            print(f"\n⚠ No .txt files found in {folder_path}")
            return None
        
        # Save to CSV
        df = pd.DataFrame(results)
        df.to_csv(output_csv, index=False)
        
        print("\n" + "="*60)
        print("PREDICTION SUMMARY")
        print("="*60)
        print(f"Total files processed: {file_count}")
        print(f"\nPredicted class distribution:")
        if 'predicted_class' in df.columns:
            class_dist = df['predicted_class'].value_counts()
            for class_name, count in class_dist.items():
                print(f"  {class_name:20s}: {count:3d} files ({count/file_count*100:.1f}%)")
        
        print(f"\n✓ Results saved to: {output_csv}")
        print("="*60 + "\n")
        
        return df
    
    def predict_from_csv(self, csv_path, text_column='cleaned_text', 
                        output_csv='predictions_from_csv.csv'):
        """
        Predict classes for texts in a CSV file
        
        Args:
            csv_path: Path to CSV file with texts
            text_column: Name of column containing text
            output_csv: Path to save predictions
        """
        print(f"Loading texts from: {csv_path}")
        df = pd.read_csv(csv_path)
        
        if text_column not in df.columns:
            print(f"❌ Error: Column '{text_column}' not found!")
            print(f"Available columns: {df.columns.tolist()}")
            return None
        
        print(f"Predicting {len(df)} texts...")
        
        # Batch prediction
        predictions = self.predict_batch(df[text_column].tolist(), 
                                        return_probabilities=True)
        
        # Add predictions to dataframe
        df['predicted_class'] = [pred[0] for pred in predictions]
        df['confidence'] = [max(pred[1].values()) for pred in predictions]
        
        # Add individual class probabilities
        for class_name in self.label_encoder.classes_:
            df[f'prob_{class_name}'] = [pred[1][class_name] for pred in predictions]
        
        # Save results
        df.to_csv(output_csv, index=False)
        print(f"\n✓ Predictions saved to: {output_csv}")
        
        return df
    
    def interactive_predict(self):
        """Interactive prediction mode"""
        print("\n" + "="*60)
        print("INTERACTIVE PREDICTION MODE")
        print("="*60)
        print("Enter text to classify (or 'quit' to exit)")
        print("="*60 + "\n")
        
        while True:
            text = input("\n📝 Enter text: ").strip()
            
            if text.lower() in ['quit', 'exit', 'q']:
                print("Exiting...")
                break
            
            if not text:
                print("⚠ Please enter some text!")
                continue
            
            predicted_class, probabilities = self.predict(text, return_probabilities=True)
            
            print(f"\n🎯 Predicted Class: {predicted_class}")
            print(f"\n📊 Class Probabilities:")
            for class_name, prob in sorted(probabilities.items(), 
                                          key=lambda x: x[1], reverse=True):
                bar = '█' * int(prob * 40)
                print(f"  {class_name:20s} {prob:6.2%} {bar}")
    
    def get_model_info(self):
        """Display model information"""
        print("\n" + "="*60)
        print("MODEL INFORMATION")
        print("="*60)
        print(f"Model type: {type(self.model).__name__}")
        
        if hasattr(self.model, 'kernel'):
            print(f"Kernel: {self.model.kernel}")
        if hasattr(self.model, 'C'):
            print(f"Regularization (C): {self.model.C}")
        
        print(f"\nClasses ({len(self.label_encoder.classes_)}):")
        for i, class_name in enumerate(self.label_encoder.classes_, 1):
            print(f"  {i}. {class_name}")
        
        print(f"\nVectorizer settings:")
        print(f"  Max features: {self.vectorizer.max_features}")
        print(f"  N-gram range: {self.vectorizer.ngram_range}")
        print(f"  Min document frequency: {self.vectorizer.min_df}")
        print(f"  Max document frequency: {self.vectorizer.max_df}")
        print("="*60 + "\n")


# ============================================
# USAGE EXAMPLES
# ============================================
def main():
    # Initialize predictor
    predictor = SVMPredictor(model_dir='D:\Coding\MERN\Docs_classification\server\models')
    
    # Show model info
    predictor.get_model_info()
    
    # =========================
    # Example 1: Single text
    # =========================
    print("EXAMPLE 1: Single Text Prediction")
    print("-" * 60)
    
    sample_text = """
    The company announced record profits this quarter, 
    with stock prices surging to an all-time high.
    """
    
    predicted_class, probabilities = predictor.predict(
        sample_text, return_probabilities=True
    )
    
    print(f"Text: {sample_text.strip()[:100]}...")
    print(f"\nPredicted: {predicted_class}")
    print(f"Probabilities: {probabilities}")
    
    # =========================
    # Example 2: Predict folder
    # =========================
    print("\n\nEXAMPLE 2: Predict Entire Folder")
    print("-" * 60)
    
    # Uncomment and set your test folder path
    test_folder = "D:\Coding\MERN\Docs_classification\server\\training\processed_data\cleaned_texts"
    df = predictor.predict_folder(test_folder, output_csv='predictions.csv')
    
    # =========================
    # Example 3: Predict from CSV
    # =========================
    print("\n\nEXAMPLE 3: Predict from CSV")
    print("-" * 60)
    
    # Uncomment to predict from your preprocessed CSV
    # df = predictor.predict_from_csv(
    #     'processed_data/processed_texts.csv',
    #     text_column='cleaned_text',
    #     output_csv='predictions_from_processed.csv'
    # )
    
    # =========================
    # Example 4: Interactive mode
    # =========================
    
    print("\n\nExample 4: Interactive mode")
    print("-" * 60)
    # Uncomment to try interactive prediction
    # predictor.interactive_predict()
    
    print("\n✓ All examples complete!")
    print("Uncomment the examples above to test with your data.\n")


if __name__ == "__main__":
    main()