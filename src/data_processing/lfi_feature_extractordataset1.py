import pandas as pd
import numpy as np
from textstat import textstat
from lexicalrichness import LexicalRichness
import spacy
from pathlib import Path
import sys
import os
current_file_path = Path(__file__).resolve()
PROJECT_ROOT = current_file_path.parent.parent.parent 
sys.path.append(str(PROJECT_ROOT))
from src.utils.paths import (
    
    PROCESSED_DATA_DIR,
    
)

nlp = spacy.load("en_core_web_md")  

class LFIExtractor:
    def __init__(self):
        self.lexicons = self._load_lexicons()
    
    def _load_lexicons(self):
       
        return {
            'emotional': [
                'amazing', 'awful', 'beautiful', 'disgust', 'excellent', 
                'fear', 'furious', 'happy', 'hate', 'joy', 'love', 'pain',
                'perfect', 'terrible', 'worst'
            ],
            'cognitive': [
                'because', 'cause', 'conclude', 'deduce', 'estimate',
                'hypothesis', 'infer', 'know', 'reason', 'think',
                'understand', 'believe', 'consider', 'decide', 'determine'
            ]
        }
    
    def extract_features(self, texts):
        """Extract 21-dimensional LFI features"""
        features = []
        
        for text in texts:
            doc = nlp(text)
            text_lower = text.lower()
            word_lengths = [len(token) for token in doc if token.is_alpha]
            pos_tags = [token.pos_ for token in doc]
            
            # Stylometric Features (8-dim)
            stylo = [
                np.mean(word_lengths) if word_lengths else 0,
                np.std(word_lengths) if word_lengths else 0,
                len(doc),
                len(list(doc.sents)) if doc.sents else 1,
                textstat.flesch_reading_ease(text),
                textstat.smog_index(text),
                LexicalRichness(text).ttr,
                LexicalRichness(text).mtld(threshold=0.72)
            ]
            
            # Complexity Features (7-dim)
            complexity = [
                pos_tags.count('NOUN')/len(doc) if doc else 0,
                pos_tags.count('VERB')/len(doc) if doc else 0,
                sum(1 for t in doc if t.dep_ == 'nsubj')/len(doc) if doc else 0,
                textstat.dale_chall_readability_score(text),
                textstat.difficult_words(text)/max(1, len(doc)),
                textstat.lexicon_count(text)/max(1, len(doc)),
                textstat.syllable_count(text)/max(1, len(doc))
            ]
            
            # Psychological Features (6-dim)
            psycho = [
                sum(1 for w in self.lexicons['emotional'] if w in text_lower)/max(1, len(doc)),
                sum(1 for w in self.lexicons['cognitive'] if w in text_lower)/max(1, len(doc)),
                textstat.lexicon_count(text)/max(1, len(doc)),
                textstat.polysyllabcount(text)/max(1, len(doc)),
                len(text)/max(1, len(doc)),
                sum(c.isalpha() for c in text)/max(1, len(doc))
            ]
            
            features.append(stylo + complexity + psycho)
        
        return pd.DataFrame(features, columns=[
            'avg_word_len', 'word_len_std', 'token_count', 'sent_count',
            'flesch', 'smog', 'ttr', 'mtld', 'noun_ratio', 'verb_ratio',
            'nsubj_ratio', 'dale_chall', 'diff_word_ratio', 'lexicon_ratio',
            'syllable_ratio', 'emo_word_ratio', 'cog_word_ratio',
            'word_length_ratio', 'poly_ratio', 'char_ratio', 'letter_ratio'
        ])
    
    def process_dataset(self, split_name):
        """Process a dataset split (train/val/test)"""
        input_path = PROCESSED_DATA_DIR / "ProcessedDatasets1" / f"{split_name}_cleaned.csv"
        output_path = PROCESSED_DATA_DIR /"lfi_features" / f"lfi_{split_name}.csv"
        
        output_path.parent.mkdir(exist_ok=True)
        df = pd.read_csv(input_path)
        features = self.extract_features(df['cleaned_text'])
        features.to_csv(output_path, index=False)
        return features