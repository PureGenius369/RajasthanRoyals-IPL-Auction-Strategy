# File: src/data_processor.py
import pandas as pd
import numpy as np
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')
import gc  # Garbage collection

class BallByBallProcessor:
    def __init__(self, file_path, chunksize=100000):
        self.file_path = file_path
        self.chunksize = chunksize
        self.dtypes = self._infer_dtypes()
        
    def _infer_dtypes(self):
        """Create optimized dtypes for memory efficiency"""
        return {
            'match_id': 'int32',
            'innings': 'int8',
            'overs': 'float32',
            'over_num': 'int8',
            'innings_ball_num': 'int16',
            'over_ball_num': 'int8',
            'batting_team_id': 'int16',
            'bowling_team_id': 'int16',
            'striker_id': 'int32',
            'non_striker_id': 'int32',
            'bowler_id': 'int32',
            'runs': 'int8',
            'is_four': 'bool',
            'is_six': 'bool',
            'is_wicket': 'bool',
            'dismissal_type_id': 'int8',
            'extras_type': 'category',
            'pace_or_spin': 'category',
            # String columns will be handled separately
        }
    
    def load_in_chunks(self):
        """Load data in chunks for memory efficiency"""
        print(f"Loading {self.file_path} in chunks...")
        
        chunks = []
        for i, chunk in enumerate(pd.read_csv(self.file_path, 
                                             chunksize=self.chunksize,
                                             low_memory=False)):
            print(f"Processing chunk {i+1}...")
            
            # Optimize memory
            chunk = self._optimize_dataframe(chunk)
            chunks.append(chunk)
            
        print("Concatenating chunks...")
        df = pd.concat(chunks, ignore_index=True)
        
        print(f"Final shape: {df.shape}")
        print(f"Memory usage: {df.memory_usage(deep=True).sum() / 1024**2:.2f} MB")
        
        return df
    
    def _optimize_dataframe(self, df):
        """Optimize dataframe memory usage"""
        # Convert to categorical where appropriate
        for col in df.columns:
            if df[col].dtype == 'object':
                num_unique = df[col].nunique()
                if num_unique < len(df) * 0.5:  # If less than 50% unique
                    df[col] = df[col].astype('category')
            
            # Apply predefined dtypes
            if col in self.dtypes and df[col].dtype != self.dtypes[col]:
                try:
                    df[col] = df[col].astype(self.dtypes[col])
                except:
                    pass
        
        return df
    
    def get_sample_stats(self, sample_size=10000):
        """Get quick statistics without loading full file"""
        sample = pd.read_csv(self.file_path, nrows=sample_size)
        
        print("📊 SAMPLE STATISTICS")
        print("=" * 50)
        print(f"Sample shape: {sample.shape}")
        print(f"Columns: {list(sample.columns)}")
        
        # Key column analysis
        key_columns = ['striker_name', 'bowler_name', 'batting_team_name', 'bowling_team_name']
        for col in key_columns:
            if col in sample.columns:
                print(f"\n{col}:")
                print(f"  Unique values: {sample[col].nunique()}")
                print(f"  Top 5: {sample[col].value_counts().head(5).to_dict()}")
        
        return sample