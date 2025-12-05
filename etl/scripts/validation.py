#!/usr/bin/env python3
"""
Data Validation Module for ETL Pipeline
Validates cleaned and transformed data against business rules and data quality standards
"""

import logging
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, Any, List, Tuple
import json


class DataValidator:
    """Handles all data validation operations for support tickets"""
    
    def __init__(self, config: Dict[str, Any]):
        """Initialize the data validator with configuration"""
        self.config = config
        self.logger = logging.getLogger(self.__class__.__name__)
        self.validation_report = {
            'validation_timestamp': datetime.now().isoformat(),
            'total_records': 0,
            'validation_passed': True,
            'required_fields_check': {},
            'missing_values_check': {},
            'status_values_check': {},
            'date_rules_check': {},
            'outliers_check': {},
            'data_quality_score': 0.0,
            'errors': [],
            'warnings': []
        }
        
    def validate_required_fields(self, df: pd.DataFrame) -> bool:
        """Validate that all required fields are present and not null"""
        self.logger.info("Validating required fields...")
        
        required_fields = self.config.get('required_fields', [
            'ticket_id', 'subject', 'priority', 'status', 'created_at'
        ])
        
        validation_passed = True
        
        for field in required_fields:
            if field not in df.columns:
                error_msg = f"Required field '{field}' is missing from dataset"
                self.validation_report['errors'].append(error_msg)
                self.logger.error(error_msg)
                validation_passed = False
                continue
            
            # Check for null values in required fields
            null_count = df[field].isnull().sum()
            null_percentage = (null_count / len(df)) * 100
            
            self.validation_report['required_fields_check'][field] = {
                'is_present': True,
                'null_count': null_count,
                'null_percentage': round(null_percentage, 2),
                'passed': null_count == 0
            }
            
            if null_count > 0:
                warning_msg = f"Required field '{field}' has {null_count} null values ({null_percentage:.2f}%)"
                self.validation_report['warnings'].append(warning_msg)
                self.logger.warning(warning_msg)
                
                # Critical failure if more than 5% nulls in required field
                if null_percentage > 5:
                    validation_passed = False
        
        self.logger.info(f"Required fields validation: {'PASSED' if validation_passed else 'FAILED'}")
        return validation_passed
    
    def validate_missing_values(self, df: pd.DataFrame) -> bool:
        """Validate missing values are within acceptable thresholds"""
        self.logger.info("Validating missing values...")
        
        missing_value_thresholds = self.config.get('missing_value_thresholds', {
            'resolved_at': 50,  # Up to 50% can be missing (unresolved tickets)
            'assigned_agent': 10,  # Up to 10% can be missing
            'customer_id': 5  # Up to 5% can be missing
        })
        
        validation_passed = True
        
        for column in df.columns:
            null_count = df[column].isnull().sum()
            null_percentage = (null_count / len(df)) * 100
            
            threshold = missing_value_thresholds.get(column, 20)  # Default 20% threshold
            
            self.validation_report['missing_values_check'][column] = {
                'null_count': null_count,
                'null_percentage': round(null_percentage, 2),
                'threshold': threshold,
                'passed': null_percentage <= threshold
            }
            
            if null_percentage > threshold:
                error_msg = f"Column '{column}' has {null_percentage:.2f}% missing values (threshold: {threshold}%)"
                self.validation_report['errors'].append(error_msg)
                self.logger.error(error_msg)
                validation_passed = False
            elif null_percentage > threshold * 0.8:  # Warning at 80% of threshold
                warning_msg = f"Column '{column}' approaching missing value threshold: {null_percentage:.2f}%"
                self.validation_report['warnings'].append(warning_msg)
                self.logger.warning(warning_msg)
        
        self.logger.info(f"Missing values validation: {'PASSED' if validation_passed else 'FAILED'}")
        return validation_passed
    
    def validate_status_values(self, df: pd.DataFrame) -> bool:
        """Validate that status values are within allowed values"""
        self.logger.info("Validating status values...")
        
        allowed_status_values = self.config.get('allowed_status_values', [
            'Open', 'Closed', 'Pending', 'Resolved', 'In Progress'
        ])
        
        validation_passed = True
        
        if 'status' in df.columns:
            unique_statuses = df['status'].dropna().unique()
            invalid_statuses = [status for status in unique_statuses if status not in allowed_status_values]
            
            self.validation_report['status_values_check'] = {
                'allowed_values': allowed_status_values,
                'found_values': list(unique_statuses),
                'invalid_values': invalid_statuses,
                'passed': len(invalid_statuses) == 0
            }
            
            if invalid_statuses:
                error_msg = f"Invalid status values found: {invalid_statuses}"
                self.validation_report['errors'].append(error_msg)
                self.logger.error(error_msg)
                validation_passed = False
            
            # Check status distribution
            status_counts = df['status'].value_counts()
            for status, count in status_counts.items():
                percentage = (count / len(df)) * 100
                self.logger.info(f"Status '{status}': {count} records ({percentage:.1f}%)")
        
        else:
            error_msg = "Status column not found in dataset"
            self.validation_report['errors'].append(error_msg)
            self.logger.error(error_msg)
            validation_passed = False
        
        self.logger.info(f"Status values validation: {'PASSED' if validation_passed else 'FAILED'}")
        return validation_passed
    
    def validate_date_rules(self, df: pd.DataFrame) -> bool:
        """Validate date-related business rules"""
        self.logger.info("Validating date rules...")
        
        validation_passed = True
        date_issues = []
        
        if 'created_at' in df.columns and 'resolved_at' in df.columns:
            # Ensure datetime types
            df['created_at'] = pd.to_datetime(df['created_at'], errors='coerce')
            df['resolved_at'] = pd.to_datetime(df['resolved_at'], errors='coerce')
            
            # Rule 1: resolved_at should not be before created_at
            invalid_resolution_dates = df[
                (df['resolved_at'].notna()) & 
                (df['created_at'].notna()) & 
                (df['resolved_at'] < df['created_at'])
            ]
            
            if len(invalid_resolution_dates) > 0:
                error_msg = f"{len(invalid_resolution_dates)} tickets have resolved_at before created_at"
                date_issues.append(error_msg)
                self.validation_report['errors'].append(error_msg)
                self.logger.error(error_msg)
                validation_passed = False
            
            # Rule 2: created_at should not be in the future
            future_created = df[df['created_at'] > datetime.now()]
            if len(future_created) > 0:
                error_msg = f"{len(future_created)} tickets have future created_at dates"
                date_issues.append(error_msg)
                self.validation_report['errors'].append(error_msg)
                self.logger.error(error_msg)
                validation_passed = False
            
            # Rule 3: resolved_at should not be in the future
            future_resolved = df[df['resolved_at'] > datetime.now()]
            if len(future_resolved) > 0:
                error_msg = f"{len(future_resolved)} tickets have future resolved_at dates"
                date_issues.append(error_msg)
                self.validation_report['errors'].append(error_msg)
                self.logger.error(error_msg)
                validation_passed = False
            
            # Rule 4: Check for reasonable date ranges (last 10 years)
            min_date = datetime.now() - timedelta(days=10*365)
            old_tickets = df[df['created_at'] < min_date]
            if len(old_tickets) > 0:
                warning_msg = f"{len(old_tickets)} tickets are older than 10 years"
                self.validation_report['warnings'].append(warning_msg)
                self.logger.warning(warning_msg)
        
        self.validation_report['date_rules_check'] = {
            'issues_found': date_issues,
            'passed': validation_passed
        }
        
        self.logger.info(f"Date rules validation: {'PASSED' if validation_passed else 'FAILED'}")
        return validation_passed
    
    def identify_outliers(self, df: pd.DataFrame) -> bool:
        """Identify outliers in resolution time and other metrics"""
        self.logger.info("Identifying outliers...")
        
        outliers_found = []
        validation_passed = True
        
        # Check resolution time outliers (> 900 days as specified)
        if 'resolution_days' in df.columns:
            outlier_threshold = 900  # days
            resolution_outliers = df[df['resolution_days'] > outlier_threshold]
            
            if len(resolution_outliers) > 0:
                warning_msg = f"Found {len(resolution_outliers)} tickets with resolution time > {outlier_threshold} days"
                outliers_found.append({
                    'type': 'resolution_time',
                    'count': len(resolution_outliers),
                    'threshold': outlier_threshold,
                    'max_value': resolution_outliers['resolution_days'].max(),
                    'ticket_ids': resolution_outliers['ticket_id'].tolist()[:10]  # First 10 IDs
                })
                self.validation_report['warnings'].append(warning_msg)
                self.logger.warning(warning_msg)
        
        # Check for unusually long subject/description
        if 'subject_length' in df.columns:
            subject_q99 = df['subject_length'].quantile(0.99)
            long_subjects = df[df['subject_length'] > subject_q99 * 2]  # 2x the 99th percentile
            
            if len(long_subjects) > 0:
                outliers_found.append({
                    'type': 'subject_length',
                    'count': len(long_subjects),
                    'threshold': subject_q99 * 2,
                    'max_value': long_subjects['subject_length'].max()
                })
        
        if 'description_length' in df.columns:
            desc_q99 = df['description_length'].quantile(0.99)
            long_descriptions = df[df['description_length'] > desc_q99 * 2]
            
            if len(long_descriptions) > 0:
                outliers_found.append({
                    'type': 'description_length',
                    'count': len(long_descriptions),
                    'threshold': desc_q99 * 2,
                    'max_value': long_descriptions['description_length'].max()
                })
        
        self.validation_report['outliers_check'] = {
            'outliers_found': outliers_found,
            'total_outlier_records': sum(outlier['count'] for outlier in outliers_found),
            'passed': True  # Outliers are warnings, not failures
        }
        
        self.logger.info(f"Outlier detection completed: {len(outliers_found)} types of outliers found")
        return True
    
    def calculate_data_quality_score(self, df: pd.DataFrame) -> float:
        """Calculate overall data quality score (0-100)"""
        score = 100.0
        
        # Deduct points for missing required fields
        required_field_score = len([check for check in self.validation_report['required_fields_check'].values() if check.get('passed', False)])
        total_required = len(self.validation_report['required_fields_check'])
        if total_required > 0:
            required_field_percentage = (required_field_score / total_required) * 100
            score = min(score, required_field_percentage)
        
        # Deduct points for errors (each error -10 points, max -50)
        error_penalty = min(len(self.validation_report['errors']) * 10, 50)
        score -= error_penalty
        
        # Deduct points for warnings (each warning -2 points, max -20)
        warning_penalty = min(len(self.validation_report['warnings']) * 2, 20)
        score -= warning_penalty
        
        # Ensure score is not negative
        score = max(score, 0.0)
        
        return round(score, 1)
    
    def save_validation_report(self, output_path: str = None) -> str:
        """Save validation report to JSON file"""
        if output_path is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_path = f"validation_report_{timestamp}.json"
        
        try:
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(self.validation_report, f, indent=2, default=str)
            
            self.logger.info(f"Validation report saved to: {output_path}")
            return output_path
            
        except Exception as e:
            self.logger.error(f"Failed to save validation report: {e}")
            raise
    
    def validate_data(self, df: pd.DataFrame) -> Tuple[bool, Dict[str, Any]]:
        """Execute the complete data validation pipeline"""
        self.logger.info("Starting data validation pipeline...")
        
        self.validation_report['total_records'] = len(df)
        
        # Run all validation checks
        checks = [
            self.validate_required_fields(df),
            self.validate_missing_values(df),
            self.validate_status_values(df),
            self.validate_date_rules(df),
            self.identify_outliers(df)
        ]
        
        # Overall validation result
        overall_passed = all(checks)
        self.validation_report['validation_passed'] = overall_passed
        
        # Calculate data quality score
        self.validation_report['data_quality_score'] = self.calculate_data_quality_score(df)
        
        # Summary logging
        self.logger.info(f"Validation completed:")
        self.logger.info(f"  - Overall result: {'PASSED' if overall_passed else 'FAILED'}")
        self.logger.info(f"  - Data quality score: {self.validation_report['data_quality_score']}/100")
        self.logger.info(f"  - Errors: {len(self.validation_report['errors'])}")
        self.logger.info(f"  - Warnings: {len(self.validation_report['warnings'])}")
        
        return overall_passed, self.validation_report


def main():
    """Test the validation module independently"""
    
    # Test configuration
    config = {
        'required_fields': ['ticket_id', 'subject', 'status', 'created_at'],
        'allowed_status_values': ['Open', 'Closed', 'Pending', 'Resolved'],
        'missing_value_thresholds': {
            'resolved_at': 50,
            'assigned_agent': 10
        }
    }
    
    # Test with sample data
    test_data = {
        'ticket_id': [1, 2, 3, 4, 5],
        'subject': ['Issue 1', 'Issue 2', None, 'Issue 4', 'Issue 5'],
        'status': ['Open', 'Closed', 'Pending', 'InvalidStatus', 'Resolved'],
        'created_at': ['2023-01-01', '2023-01-02', '2023-01-03', '2025-01-01', '2020-01-01'],
        'resolved_at': ['2023-01-02', '2023-01-01', None, None, '2020-01-05'],  # Invalid date logic
        'resolution_days': [1, 950, None, None, 5]  # One outlier
    }
    
    df = pd.DataFrame(test_data)
    print("Test data:")
    print(df)
    
    validator = DataValidator(config)
    passed, report = validator.validate_data(df)
    
    print(f"\nValidation result: {'PASSED' if passed else 'FAILED'}")
    print(f"Data quality score: {report['data_quality_score']}/100")
    print(f"Errors: {len(report['errors'])}")
    print(f"Warnings: {len(report['warnings'])}")
    
    # Save report
    report_path = validator.save_validation_report('test_validation_report.json')
    print(f"Report saved to: {report_path}")


if __name__ == "__main__":
    main()