import re
from typing import Dict
from transformers import pipeline
import torch

class EngineeringSentimentAnalyzer:
    def __init__(self):
        """Initialize Hugging Face sentiment pipeline optimized for technical text"""
        # Technical domain sentiment model (handles engineering terms better)
        self.analyzer = pipeline(
            "sentiment-analysis",
            model="cardiffnlp/twitter-roberta-base-sentiment-latest",
            device=0 if torch.cuda.is_available() else -1,
            torch_dtype=torch.float16
        )
        
    def preprocess_engineering_text(self, text: str) -> str:
        """Clean CAD text for better sentiment analysis"""
        # Remove technical noise
        text = re.sub(r'[\d:\-xX×mm]+', ' ', text)  # Dimensions/scales
        text = re.sub(r'[A-Z]{2,4}\d+', '', text)    # Drawing refs (A1, REV B)
        text = re.sub(r'\s+', ' ', text).strip()
        return text
    
    def analyze(self, text: str) -> Dict:
        """Analyze sentiment of engineering drawing text"""
        clean_text = self.preprocess_engineering_text(text)
        
        if len(clean_text) < 10:
            return {
                'compound': 0.0,
                'label': 'NEUTRAL',
                'score': 0.5,
                'pos': 0.0,
                'neg': 0.0,
                'neu': 1.0,
                'engineering_flags': []
            }
        
        # Run Hugging Face pipeline
        result = self.analyzer(clean_text)[0]
        
        # Extract engineering sentiment keywords
        neg_flags = re.findall(r'\b(urgent|critical|reject|fail|error|defect|revise|fix|redesign)\b', 
                              text.lower())
        pos_flags = re.findall(r'\b(approved|ok|good|final|complete|acceptable)\b', 
                              text.lower())
        
        return {
            'label': result['label'],
            'score': result['score'],
            'compound': 1.0 if result['label'] == 'POSITIVE' else 
                       -1.0 if result['label'] == 'NEGATIVE' else 0.0,
            'pos': result['score'] if result['label'] == 'POSITIVE' else 0.0,
            'neg': result['score'] if result['label'] == 'NEGATIVE' else 0.0,
            'neu': 1.0 - result['score'],
            'engineering_flags': neg_flags + pos_flags,
            'raw_text': clean_text[:200],
            'model': 'twitter-roberta-base-sentiment'
        }

# Usage example
def analyze_engineering_sentiment(text: str) -> Dict:
    """Main function for app integration"""
    analyzer = EngineeringSentimentAnalyzer()
    return anal
