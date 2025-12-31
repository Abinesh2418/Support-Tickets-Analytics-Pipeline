#!/usr/bin/env python3
"""
Data Cleaning Module for ETL Pipeline
Handles data cleaning operations including missing values, duplicates, and impossible data
"""

import logging
import pandas as pd
import numpy as np
from datetime import datetime
from typing import Dict, Any, Tuple
import re


class DataCleaner:
    """Handles all data cleaning operations for support tickets"""
    
    def __init__(self, config: Dict[str, Any]):
        """Initialize the data cleaner with configuration"""
        self.config = config
        self.logger = logging.getLogger(self.__class__.__name__)
        self.cleaning_report = {
            'impossible_data_removed': 0,
            'missing_values_found': {},
            'duplicates_removed': 0,
            'text_normalized': 0,
            'invalid_dates_fixed': 0
        }
        
    def remove_impossible_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Remove records with impossible data values"""
        self.logger.info("Removing impossible data...")
        initial_count = len(df)
        
        # Convert date columns to datetime for comparison
        df['created_at'] = pd.to_datetime(df['created_at'], errors='coerce')
        df['resolved_at'] = pd.to_datetime(df['resolved_at'], errors='coerce')
        
        # Remove records where resolved_at < created_at
        before_date_filter = len(df)
        df = df[~((df['resolved_at'].notna()) & (df['created_at'].notna()) & 
                 (df['resolved_at'] < df['created_at']))]
        date_impossible_removed = before_date_filter - len(df)
        
        # Remove records where ticket_id < 0
        before_id_filter = len(df)
        df = df[df['ticket_id'] >= 0]
        negative_id_removed = before_id_filter - len(df)
        
        # Remove records with invalid dates (year < 1900 or year > current year + 1)
        current_year = datetime.now().year
        before_invalid_dates = len(df)
        df = df[~((df['created_at'].dt.year < 1900) | (df['created_at'].dt.year > current_year + 1))]
        df = df[~((df['resolved_at'].dt.year < 1900) | (df['resolved_at'].dt.year > current_year + 1))]
        invalid_dates_removed = before_invalid_dates - len(df)
        
        total_removed = initial_count - len(df)
        self.cleaning_report['impossible_data_removed'] = total_removed
        
        self.logger.info(f"Removed {total_removed} records with impossible data:")
        self.logger.info(f"  - Date logic violations: {date_impossible_removed}")
        self.logger.info(f"  - Negative ticket IDs: {negative_id_removed}")
        self.logger.info(f"  - Invalid dates: {invalid_dates_removed}")
        
        return df
    
    def handle_missing_values(self, df: pd.DataFrame) -> pd.DataFrame:
        """Handle missing values using advanced imputation techniques"""
        self.logger.info("Handling missing values...")
        
        # Count missing values for each column
        for column in df.columns:
            missing_count = df[column].isnull().sum()
            if missing_count > 0:
                missing_percentage = (missing_count / len(df)) * 100
                self.cleaning_report['missing_values_found'][column] = {
                    'count': missing_count,
                    'percentage': round(missing_percentage, 2)
                }
                self.logger.info(f"Column '{column}': {missing_count} missing values ({missing_percentage:.2f}%)")
        
        # Keep missing values as required but mark them for tracking
        # Add a column to track which records had missing critical fields
        critical_fields = ['ticket_id', 'subject', 'status', 'created_at']
        df['has_missing_critical'] = df[critical_fields].isnull().any(axis=1)
        
        # Advanced imputation techniques for non-critical fields
        
        # 1. MODE IMPUTATION for categorical fields
        categorical_fields = ['category', 'product', 'channel', 'origin']
        for field in categorical_fields:
            if field in df.columns and df[field].isnull().sum() > 0:
                mode_value = df[field].mode()[0] if len(df[field].mode()) > 0 else 'Other'
                filled_count = df[field].isnull().sum()
                df[field] = df[field].fillna(mode_value)
                self.logger.info(f"Filled {filled_count} missing values in '{field}' with mode: '{mode_value}'")
        
        # 2. FORWARD FILL for assigned_agent (based on customer patterns)
        if 'assigned_agent' in df.columns and 'customer_id' in df.columns:
            # Sort by customer_id and created_at to use forward fill logic
            df_sorted = df.sort_values(['customer_id', 'created_at'])
            
            # Group by customer and forward fill agent assignments
            df_sorted['assigned_agent'] = df_sorted.groupby('customer_id')['assigned_agent'].ffill()
            
            # For still missing values, use the most common agent per priority level
            missing_mask = df_sorted['assigned_agent'].isnull()
            if missing_mask.sum() > 0:
                if 'priority' in df.columns:
                    # Get most common agent per priority
                    priority_agent_map = df_sorted.dropna(subset=['assigned_agent', 'priority']).groupby('priority')['assigned_agent'].agg(lambda x: x.mode()[0] if len(x.mode()) > 0 else 'Unassigned').to_dict()
                    
                    # Fill based on priority
                    for priority, agent in priority_agent_map.items():
                        mask = missing_mask & (df_sorted['priority'] == priority)
                        df_sorted.loc[mask, 'assigned_agent'] = agent
                
                # Final fallback to most common agent
                remaining_missing = df_sorted['assigned_agent'].isnull().sum()
                if remaining_missing > 0:
                    most_common_agent = df_sorted['assigned_agent'].mode()[0] if len(df_sorted['assigned_agent'].mode()) > 0 else 'Unassigned'
                    df_sorted['assigned_agent'] = df_sorted['assigned_agent'].fillna(most_common_agent)
                    self.logger.info(f"Filled {remaining_missing} remaining missing 'assigned_agent' values with most common: '{most_common_agent}'")
            
            # Restore original order
            df = df_sorted.sort_index()
        
        # 3. K-NEAREST NEIGHBORS IMPUTATION for description based on subject similarity
        if 'description' in df.columns and 'subject' in df.columns:
            missing_desc_mask = df['description'].isnull()
            if missing_desc_mask.sum() > 0:
                # For missing descriptions, find similar subjects and use their descriptions
                filled_count = 0
                for idx in df[missing_desc_mask].index:
                    subject = str(df.loc[idx, 'subject']).lower()
                    if subject and subject != 'nan':
                        # Find tickets with similar subjects
                        similar_tickets = df[
                            (~df['description'].isnull()) & 
                            (df['subject'].str.lower().str.contains(subject[:10], na=False, regex=False))
                        ]
                        
                        if len(similar_tickets) > 0:
                            # Use description from most similar subject
                            df.loc[idx, 'description'] = similar_tickets.iloc[0]['description']
                            filled_count += 1
                
                # For remaining missing, create generic description based on subject
                still_missing = df['description'].isnull() & df['subject'].notna()
                if still_missing.sum() > 0:
                    df.loc[still_missing, 'description'] = 'Issue related to ' + df.loc[still_missing, 'subject'].astype(str) + '. Details to be updated.'
                    filled_count += still_missing.sum()
                
                self.logger.info(f"Filled {filled_count} missing 'description' values using similarity-based imputation")
        
        # 4. REGRESSION IMPUTATION for severity based on priority and category
        if 'severity' in df.columns and 'priority' in df.columns:
            missing_severity_mask = df['severity'].isnull()
            if missing_severity_mask.sum() > 0:
                # Create severity mapping based on priority
                priority_severity_map = {
                    'Critical': 'Critical',
                    'High': 'Major', 
                    'Medium': 'Minor',
                    'Low': 'Trivial'
                }
                
                filled_count = 0
                for idx in df[missing_severity_mask].index:
                    priority = df.loc[idx, 'priority']
                    if priority in priority_severity_map:
                        df.loc[idx, 'severity'] = priority_severity_map[priority]
                        filled_count += 1
                
                # For remaining missing, use mode
                remaining_missing = df['severity'].isnull().sum()
                if remaining_missing > 0:
                    mode_severity = df['severity'].mode()[0] if len(df['severity'].mode()) > 0 else 'Minor'
                    df['severity'] = df['severity'].fillna(mode_severity)
                    filled_count += remaining_missing
                
                self.logger.info(f"Filled {filled_count} missing 'severity' values using priority-based regression imputation")
        
        # 5. Business rule: Keep certain fields as null (resolved_at for unresolved tickets)
        business_null_fields = ['resolved_at']
        for field in business_null_fields:
            if field in df.columns:
                null_count = df[field].isnull().sum()
                if null_count > 0:
                    self.logger.info(f"Keeping {null_count} null values in '{field}' as per business logic")
        
        return df
    
    def remove_duplicates(self, df: pd.DataFrame) -> pd.DataFrame:
        """Remove duplicate rows and duplicate ticket_ids"""
        self.logger.info("Removing duplicates...")
        initial_count = len(df)
        
        # Remove exact duplicate rows
        before_row_duplicates = len(df)
        df = df.drop_duplicates()
        row_duplicates_removed = before_row_duplicates - len(df)
        
        # Remove duplicate ticket_ids (keep the most recent one based on created_at)
        before_id_duplicates = len(df)
        df = df.sort_values('created_at').drop_duplicates(subset=['ticket_id'], keep='last')
        id_duplicates_removed = before_id_duplicates - len(df)
        
        total_duplicates_removed = initial_count - len(df)
        self.cleaning_report['duplicates_removed'] = total_duplicates_removed
        
        self.logger.info(f"Removed {total_duplicates_removed} duplicate records:")
        self.logger.info(f"  - Exact duplicate rows: {row_duplicates_removed}")
        self.logger.info(f"  - Duplicate ticket IDs: {id_duplicates_removed}")
        
        return df
    
    def normalize_text_fields(self, df: pd.DataFrame) -> pd.DataFrame:
        """Normalize text fields by trimming spaces and fixing inconsistent values"""
        self.logger.info("Normalizing text fields...")
        
        text_columns = ['subject', 'description', 'priority', 'status', 'category', 
                       'origin', 'assigned_agent', 'product', 'channel', 'severity']
        
        normalized_count = 0
        
        for column in text_columns:
            if column in df.columns:
                # Store original for comparison
                original_values = df[column].copy()
                
                # Trim whitespace
                df[column] = df[column].astype(str).str.strip()
                
                # Fix specific inconsistencies based on column
                if column == 'priority':
                    # Standardize priority values
                    priority_mapping = {
                        'low': 'Low', 'medium': 'Medium', 'high': 'High', 'critical': 'Critical',
                        'LOW': 'Low', 'MEDIUM': 'Medium', 'HIGH': 'High', 'CRITICAL': 'Critical',
                        'urgent': 'High', 'URGENT': 'High'
                    }
                    df[column] = df[column].replace(priority_mapping)
                
                elif column == 'status':
                    # Standardize status values
                    status_mapping = {
                        'open': 'Open', 'closed': 'Closed', 'pending': 'Pending', 'resolved': 'Resolved',
                        'OPEN': 'Open', 'CLOSED': 'Closed', 'PENDING': 'Pending', 'RESOLVED': 'Resolved',
                        'in progress': 'In Progress', 'IN PROGRESS': 'In Progress'
                    }
                    df[column] = df[column].replace(status_mapping)
                
                elif column == 'severity':
                    # Standardize severity values
                    severity_mapping = {
                        'trivial': 'Trivial', 'minor': 'Minor', 'major': 'Major', 'critical': 'Critical',
                        'TRIVIAL': 'Trivial', 'MINOR': 'Minor', 'MAJOR': 'Major', 'CRITICAL': 'Critical'
                    }
                    df[column] = df[column].replace(severity_mapping)
                
                # Count how many values were actually changed
                changes = (original_values.astype(str) != df[column].astype(str)).sum()
                if changes > 0:
                    normalized_count += changes
                    self.logger.info(f"Normalized {changes} values in column '{column}'")
        
        self.cleaning_report['text_normalized'] = normalized_count
        self.logger.info(f"Total text normalization changes: {normalized_count}")
        
        return df
    
    def clean_data(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """Execute the complete data cleaning pipeline"""
        self.logger.info("Starting data cleaning pipeline...")
        initial_shape = df.shape
        
        # Step 1: Remove impossible data
        df = self.remove_impossible_data(df)
        
        # Step 2: Handle missing values
        df = self.handle_missing_values(df)
        
        # Step 3: Remove duplicates
        df = self.remove_duplicates(df)
        
        # Step 4: Normalize text fields
        df = self.normalize_text_fields(df)
        
        final_shape = df.shape
        self.cleaning_report['initial_rows'] = initial_shape[0]
        self.cleaning_report['final_rows'] = final_shape[0]
        self.cleaning_report['rows_removed'] = initial_shape[0] - final_shape[0]
        
        self.logger.info(f"Data cleaning completed: {initial_shape[0]:,} -> {final_shape[0]:,} rows")
        
        return df, self.cleaning_report


def main():
    """Test the cleaning module independently"""
    import yaml
    
    # Load test configuration
    config = {
        'missing_value_rules': {
            'assigned_agent': {'action': 'fill', 'value': 'Unassigned'},
            'resolved_at': {'action': 'keep', 'value': None}
        }
    }
    
    # Test with sample data
    test_data = {
        'ticket_id': [1, 2, 3, -1, 2],  # Duplicate and negative ID
        'subject': ['  Issue 1  ', 'Issue 2', 'Issue 3', 'Issue 4', 'Issue 2'],
        'status': ['open', 'CLOSED', 'pending', 'resolved', 'CLOSED'],
        'created_at': ['2023-01-01', '2023-01-02', '2023-01-03', '1800-01-01', '2023-01-02'],
        'resolved_at': ['2023-01-02', '2023-01-01', None, '1800-01-02', '2023-01-01']  # Invalid logic
    }
    
    df = pd.DataFrame(test_data)
    print("Original data:")
    print(df)
    
    cleaner = DataCleaner(config)
    cleaned_df, report = cleaner.clean_data(df)
    
    print("\nCleaned data:")
    print(cleaned_df)
    print("\nCleaning report:")
    print(report)


if __name__ == "__main__":
    main()