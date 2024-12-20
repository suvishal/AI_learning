import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split
import pickle
import os

class SimpleOfflineAI:
    def __init__(self):
        self.vectorizer = TfidfVectorizer(max_features=5000)
        self.classifier = MultinomialNB()
        self.is_trained = False
    
    def train(self, texts, labels):
        """
        Train the model on provided texts and labels
        
        Args:
            texts: List of text documents
            labels: List of corresponding labels
        """
        # Convert texts to TF-IDF features
        X = self.vectorizer.fit_transform(texts)
        
        # Split into training and validation sets
        X_train, X_val, y_train, y_val = train_test_split(
            X, labels, test_size=0.2, random_state=42
        )
        
        # Train the classifier
        self.classifier.fit(X_train, y_train)
        
        # Calculate accuracy on validation set
        accuracy = self.classifier.score(X_val, y_val)
        self.is_trained = True
        
        return accuracy
    
    def predict(self, text):
        """
        Make predictions on new text
        
        Args:
            text: Text to classify
        """
        if not self.is_trained:
            raise ValueError("Model needs to be trained first!")
            
        # Transform text using the same vectorizer
        X = self.vectorizer.transform([text])
        
        # Make prediction
        prediction = self.classifier.predict(X)[0]
        confidence = np.max(self.classifier.predict_proba(X))
        
        return prediction, confidence
    
    def save_model(self, path="model"):
        """Save the trained model to disk"""
        if not self.is_trained:
            raise ValueError("Model needs to be trained first!")
            
        os.makedirs(path, exist_ok=True)
        
        # Save the vectorizer
        with open(f"{path}/vectorizer.pkl", "wb") as f:
            pickle.dump(self.vectorizer, f)
            
        # Save the classifier
        with open(f"{path}/classifier.pkl", "wb") as f:
            pickle.dump(self.classifier, f)
    
    def load_model(self, path="model"):
        """Load a trained model from disk"""
        try:
            # Load the vectorizer
            with open(f"{path}/vectorizer.pkl", "rb") as f:
                self.vectorizer = pickle.load(f)
                
            # Load the classifier
            with open(f"{path}/classifier.pkl", "rb") as f:
                self.classifier = pickle.load(f)
                
            self.is_trained = True
        except Exception as e:
            raise ValueError(f"Error loading model: {e}")

# Example usage
def main():
    # Sample training data
    texts = [
        "This is a positive review about the product",
        "I really enjoyed this movie",
        "The service was terrible",
        "I would not recommend this",
        # Add more training examples...
    ]
    
    labels = ["positive", "positive", "negative", "negative"]
    
    # Create and train the model
    model = SimpleOfflineAI()
    accuracy = model.train(texts, labels)
    print(f"Model trained with accuracy: {accuracy:.2f}")
    
    # Save the model
    model.save_model()
    
    # Make predictions
    test_text = "I really liked this product"
    prediction, confidence = model.predict(test_text)
    print(f"Prediction: {prediction}")
    print(f"Confidence: {confidence:.2f}")

if __name__ == "__main__":
    main()
