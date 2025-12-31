#!/usr/bin/env python3
"""
Data Transformation Module for ETL Pipeline
Converts raw data into ML-ready structure with derived columns and proper data types
"""

import logging
import pandas as pd
import numpy as np
from datetime import datetime
from typing import Dict, Any, Tuple
import re


class DataTransformer:
    """Handles all data transformation operations for support tickets"""
    
    def __init__(self, config: Dict[str, Any]):
        """Initialize the data transformer with configuration"""
        self.config = config
        self.logger = logging.getLogger(self.__class__.__name__)
        self.transformation_report = {
            'columns_converted': [],
            'derived_columns_created': [],
            'categorical_encoded': [],
            'columns_standardized': []
        }
        
    def convert_data_types(self, df: pd.DataFrame) -> pd.DataFrame:
        """Convert data types according to schema requirements"""
        self.logger.info("Converting data types...")
        
        type_mappings = self.config.get('data_types', {})
        
        for column, target_type in type_mappings.items():
            if column in df.columns:
                try:
                    original_type = df[column].dtype
                    
                    if target_type == 'datetime':
                        df[column] = pd.to_datetime(df[column], errors='coerce')
                    elif target_type == 'int':
                        df[column] = pd.to_numeric(df[column], errors='coerce').astype('Int64')
                    elif target_type == 'float':
                        df[column] = pd.to_numeric(df[column], errors='coerce')
                    elif target_type == 'string':
                        df[column] = df[column].astype('string')
                    elif target_type == 'category':
                        df[column] = df[column].astype('category')
                    elif target_type == 'bool':
                        df[column] = df[column].astype('boolean')
                    
                    self.transformation_report['columns_converted'].append({
                        'column': column,
                        'from': str(original_type),
                        'to': target_type
                    })
                    
                    self.logger.info(f"Converted '{column}': {original_type} -> {target_type}")
                    
                except Exception as e:
                    self.logger.warning(f"Failed to convert '{column}' to {target_type}: {e}")
        
        return df
    
    def create_derived_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create derived columns for ML readiness"""
        self.logger.info("Creating derived columns...")
        
        # Ensure datetime columns are properly converted
        df['created_at'] = pd.to_datetime(df['created_at'], errors='coerce', utc=True).dt.tz_localize(None)
        df['resolved_at'] = pd.to_datetime(df['resolved_at'], errors='coerce', utc=True).dt.tz_localize(None)
        
        # 1. Resolution days calculation
        df['resolution_days'] = (df['resolved_at'] - df['created_at']).dt.days
        df['resolution_days'] = df['resolution_days'].where(df['resolution_days'] >= 0)  # Remove negative values
        self.transformation_report['derived_columns_created'].append('resolution_days')
        
        # 2. Is resolved flag
        df['is_resolved'] = df['resolved_at'].notna().astype('boolean')
        self.transformation_report['derived_columns_created'].append('is_resolved')
        
        # 3. Time-based features from created_at
        df['created_year'] = df['created_at'].dt.year
        df['created_month'] = df['created_at'].dt.month
        df['created_day_of_week'] = df['created_at'].dt.day_name()
        df['created_hour'] = df['created_at'].dt.hour
        df['created_quarter'] = df['created_at'].dt.quarter
        
        time_features = ['created_year', 'created_month', 'created_day_of_week', 'created_hour', 'created_quarter']
        self.transformation_report['derived_columns_created'].extend(time_features)
        
        # 4. Resolution time-based features (for resolved tickets)
        df['resolved_year'] = df['resolved_at'].dt.year
        df['resolved_month'] = df['resolved_at'].dt.month
        df['resolved_day_of_week'] = df['resolved_at'].dt.day_name()
        df['resolved_hour'] = df['resolved_at'].dt.hour
        
        resolution_time_features = ['resolved_year', 'resolved_month', 'resolved_day_of_week', 'resolved_hour']
        self.transformation_report['derived_columns_created'].extend(resolution_time_features)
        
        # 5. Business logic features
        df['is_sla_breached'] = df['sla_breached'].astype('boolean')
        df['is_weekend_created'] = df['created_day_of_week'].isin(['Saturday', 'Sunday'])
        df['is_business_hours'] = df['created_hour'].between(9, 17)
        
        business_features = ['is_sla_breached', 'is_weekend_created', 'is_business_hours']
        self.transformation_report['derived_columns_created'].extend(business_features)
        
        # 6. Text-based features
        if 'subject' in df.columns:
            df['subject_length'] = df['subject'].astype(str).str.len()
            df['subject_word_count'] = df['subject'].astype(str).str.split().str.len()
            self.transformation_report['derived_columns_created'].extend(['subject_length', 'subject_word_count'])
        
        if 'description' in df.columns:
            df['description_length'] = df['description'].astype(str).str.len()
            df['description_word_count'] = df['description'].astype(str).str.split().str.len()
            self.transformation_report['derived_columns_created'].extend(['description_length', 'description_word_count'])
        
        # 7. Priority scoring
        priority_mapping = {'Low': 1, 'Medium': 2, 'High': 3, 'Critical': 4}
        df['priority_score'] = df['priority'].map(priority_mapping)
        self.transformation_report['derived_columns_created'].append('priority_score')
        
        # 8. Severity scoring
        severity_mapping = {'Trivial': 1, 'Minor': 2, 'Major': 3, 'Critical': 4}
        df['severity_score'] = df['severity'].map(severity_mapping)
        self.transformation_report['derived_columns_created'].append('severity_score')
        
        # 9. Status grouping
        status_groups = {
            'Open': 'Active',
            'New': 'Active',
            'Pending': 'Active', 
            'In Progress': 'Active',
            'On Hold': 'Active',
            'Resolved': 'Closed',
            'Closed': 'Closed'
        }
        df['status_group'] = df['status'].map(status_groups).fillna('Unknown')
        self.transformation_report['derived_columns_created'].append('status_group')
        
        # 10. Age in days (from creation to now for unresolved tickets)
        current_date = pd.Timestamp.now().tz_localize(None)
        df['age_days'] = (current_date - df['created_at']).dt.days
        self.transformation_report['derived_columns_created'].append('age_days')
        
        self.logger.info(f"Created {len(self.transformation_report['derived_columns_created'])} derived columns")
        
        return df
    
    def encode_categorical_variables(self, df: pd.DataFrame) -> pd.DataFrame:
        """Encode categorical variables for ML readiness"""
        self.logger.info("Encoding categorical variables...")
        
        categorical_columns = self.config.get('categorical_encoding', {})
        
        for column, encoding_type in categorical_columns.items():
            if column in df.columns:
                try:
                    if encoding_type == 'label':
                        # Label encoding
                        df[f'{column}_encoded'] = pd.Categorical(df[column]).codes
                        self.transformation_report['categorical_encoded'].append({
                            'column': column,
                            'type': 'label_encoding',
                            'new_column': f'{column}_encoded'
                        })
                    
                    elif encoding_type == 'onehot':
                        # One-hot encoding
                        dummies = pd.get_dummies(df[column], prefix=column, dummy_na=True)
                        df = pd.concat([df, dummies], axis=1)
                        self.transformation_report['categorical_encoded'].append({
                            'column': column,
                            'type': 'one_hot_encoding',
                            'new_columns': list(dummies.columns)
                        })
                    
                    self.logger.info(f"Encoded '{column}' using {encoding_type} encoding")
                    
                except Exception as e:
                    self.logger.warning(f"Failed to encode '{column}': {e}")
        
        return df
    
    def standardize_column_names(self, df: pd.DataFrame) -> pd.DataFrame:
        """Standardize column names to follow consistent naming convention"""
        self.logger.info("Standardizing column names...")
        
        # Store original column names
        original_columns = df.columns.tolist()
        
        # Apply naming conventions
        new_columns = []
        for col in df.columns:
            # Convert to lowercase and replace spaces/special chars with underscores
            new_col = re.sub(r'[^a-zA-Z0-9_]', '_', col.lower())
            # Remove multiple consecutive underscores
            new_col = re.sub(r'_+', '_', new_col)
            # Remove leading/trailing underscores
            new_col = new_col.strip('_')
            new_columns.append(new_col)
        
        # Update column names
        df.columns = new_columns
        
        # Track changes
        for original, new in zip(original_columns, new_columns):
            if original != new:
                self.transformation_report['columns_standardized'].append({
                    'original': original,
                    'standardized': new
                })
        
        self.logger.info(f"Standardized {len(self.transformation_report['columns_standardized'])} column names")
        
        return df
    
    def transform_data(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """Execute the complete data transformation pipeline"""
        self.logger.info("Starting data transformation pipeline...")
        initial_columns = len(df.columns)
        
        # Step 1: Convert data types
        df = self.convert_data_types(df)
        
        # Step 2: Create derived columns
        df = self.create_derived_columns(df)
        
        # Step 3: Encode categorical variables (optional)
        df = self.encode_categorical_variables(df)
        
        # Step 4: Standardize column names
        df = self.standardize_column_names(df)
        
        final_columns = len(df.columns)
        self.transformation_report['initial_columns'] = initial_columns
        self.transformation_report['final_columns'] = final_columns
        self.transformation_report['columns_added'] = final_columns - initial_columns
        
        self.logger.info(f"Transformation completed: {initial_columns} -> {final_columns} columns")
        
        return df, self.transformation_report


def main():
    """Test the transformation module independently"""
    
    # Test configuration
    config = {
        'data_types': {
            'ticket_id': 'int',
            'created_at': 'datetime',
            'resolved_at': 'datetime',
            'priority': 'category',
            'status': 'category'
        },
        'categorical_encoding': {
            'priority': 'label',
            'category': 'onehot'
        }
    }
    
    # Test with sample data
    test_data = {
        'ticket_id': ['1', '2', '3'],
        'subject': ['Payment Issue', 'Login Problem', 'Feature Request'],
        'priority': ['High', 'Medium', 'Low'],
        'status': ['Open', 'Closed', 'Pending'],
        'category': ['Technical', 'Billing', 'Feature'],
        'severity': ['Major', 'Minor', 'Trivial'],
        'created_at': ['2023-01-01 10:00:00', '2023-01-02 14:30:00', '2023-01-03 09:15:00'],
        'resolved_at': ['2023-01-02 16:00:00', '2023-01-03 11:00:00', None],
        'sla_breached': [True, False, False]
    }
    
    df = pd.DataFrame(test_data)
    print("Original data:")
    print(df.dtypes)
    print(df.head())
    
    transformer = DataTransformer(config)
    transformed_df, report = transformer.transform_data(df)
    
    print(f"\nTransformed data ({len(transformed_df.columns)} columns):")
    print(transformed_df.dtypes)
    print(f"\nTransformation report:")
    print(f"Derived columns: {len(report['derived_columns_created'])}")
    print(f"Columns converted: {len(report['columns_converted'])}")


if __name__ == "__main__":
    main()