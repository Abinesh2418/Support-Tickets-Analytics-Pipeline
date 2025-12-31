#!/usr/bin/env python3
"""
Main ETL Pipeline Orchestrator
Coordinates cleaning, transformation, and validation modules for support ticket processing
"""

import logging
import pandas as pd
import yaml
import sys
import argparse
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, Optional

# Import the modular ETL components
from etl.scripts.cleaning import DataCleaner
from etl.scripts.transformation import DataTransformer
from etl.scripts.validation import DataValidator


class ETLPipeline:
    """Main ETL Pipeline orchestrator for support tickets"""
    
    def __init__(self, config_path: str = None):
        """Initialize the ETL pipeline with configuration"""
        if config_path is None:
            # Default to config.yaml in the etl/config directory
            self.config_path = Path(__file__).parent / "etl" / "config" / "config.yaml"
        else:
            self.config_path = Path(config_path)
            # If relative path provided, make it relative to project root
            if not self.config_path.is_absolute():
                self.config_path = Path(__file__).parent / self.config_path
        
        self.config = self._load_config()
        self.project_root = Path(__file__).parent
        
        # Setup logging
        self._setup_logging()
        self.logger = logging.getLogger(self.__class__.__name__)
        
        # Initialize pipeline modules
        self.cleaner = DataCleaner(self.config)
        self.transformer = DataTransformer(self.config)
        self.validator = DataValidator(self.config)
        
        # Pipeline state
        self.pipeline_report = {
            'pipeline_id': f"etl_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            'start_time': None,
            'end_time': None,
            'duration_seconds': 0,
            'input_file': None,
            'output_file': None,
            'initial_rows': 0,
            'final_rows': 0,
            'cleaning_report': {},
            'transformation_report': {},
            'validation_report': {},
            'overall_success': False
        }
    
    def _load_config(self) -> Dict[str, Any]:
        """Load configuration from YAML file"""
        if not self.config_path.exists():
            raise FileNotFoundError(f"Configuration file not found: {self.config_path}")
            
        with open(self.config_path, 'r', encoding='utf-8') as f:
            return yaml.safe_load(f)
    
    def _setup_logging(self) -> None:
        """Configure logging for the pipeline"""
        log_config = self.config.get('logging', {})
        
        # Create logs directory
        logs_dir = self.project_root / "logs"
        logs_dir.mkdir(exist_ok=True)
        
        # Configure logging
        log_level = getattr(logging, log_config.get('level', 'INFO'))
        log_format = log_config.get('format', '%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        
        handlers = []
        
        # Console handler
        if log_config.get('console_output', True):
            console_handler = logging.StreamHandler(sys.stdout)
            console_handler.setFormatter(logging.Formatter(log_format))
            handlers.append(console_handler)
        
        # File handler
        if log_config.get('file_output', True):
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            log_file = logs_dir / f"etl_pipeline_{timestamp}.log"
            file_handler = logging.FileHandler(log_file, encoding='utf-8')
            file_handler.setFormatter(logging.Formatter(log_format))
            handlers.append(file_handler)
        
        # Configure root logger
        logging.basicConfig(
            level=log_level,
            format=log_format,
            handlers=handlers,
            force=True
        )
    
    def load_source_data(self) -> pd.DataFrame:
        """Load the source data from configured location"""
        self.logger.info("Loading source data...")
        
        input_path = self.project_root / self.config['data_source']['input_path']
        encoding = self.config['data_source'].get('encoding', 'utf-8')
        
        if not input_path.exists():
            raise FileNotFoundError(f"Source data file not found: {input_path}")
        
        try:
            df = pd.read_csv(input_path, encoding=encoding)
            self.pipeline_report['input_file'] = str(input_path)
            self.pipeline_report['initial_rows'] = len(df)
            
            self.logger.info(f"Loaded {len(df):,} rows from {input_path.name}")
            self.logger.info(f"Columns: {list(df.columns)}")
            
            return df
            
        except Exception as e:
            self.logger.error(f"Failed to load source data: {e}")
            raise
    
    def save_raw_data(self, df: pd.DataFrame) -> Path:
        """Save the raw data with timestamp for traceability"""
        self.logger.info("Saving raw data copy...")
        
        # Create raw data directory
        raw_dir = self.project_root / self.config['output']['raw_data_path']
        raw_dir.mkdir(parents=True, exist_ok=True)
        
        # Generate raw filename with timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        raw_file = raw_dir / f"support_tickets_raw_{timestamp}.parquet"
        
        try:
            # Save as Parquet with compression for consistency
            compression = self.config['output'].get('compression', 'snappy')
            df.to_parquet(raw_file, compression=compression, index=False)
            
            # Log file statistics
            file_size = raw_file.stat().st_size
            self.logger.info(f"Saved raw data: {raw_file.name}")
            self.logger.info(f"File size: {file_size:,} bytes")
            self.logger.info(f"Raw dataset: {len(df):,} rows, {len(df.columns)} columns")
            
            return raw_file
            
        except Exception as e:
            self.logger.error(f"Failed to save raw data: {e}")
            raise
    
    def save_processed_data(self, df: pd.DataFrame) -> Path:
        """Save the processed data to output location"""
        self.logger.info("Saving processed data...")
        
        # Create output directory
        output_dir = self.project_root / self.config['output']['processed_data_path']
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Generate output filename with timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_file = output_dir / f"support_tickets_processed_{timestamp}.parquet"
        
        try:
            # Save as Parquet with compression
            compression = self.config['output'].get('compression', 'snappy')
            df.to_parquet(output_file, compression=compression, index=False)
            
            self.pipeline_report['output_file'] = str(output_file)
            self.pipeline_report['final_rows'] = len(df)
            
            # Log file statistics
            file_size = output_file.stat().st_size
            self.logger.info(f"Saved processed data: {output_file.name}")
            self.logger.info(f"File size: {file_size:,} bytes")
            self.logger.info(f"Final dataset: {len(df):,} rows, {len(df.columns)} columns")
            
            return output_file
            
        except Exception as e:
            self.logger.error(f"Failed to save processed data: {e}")
            raise
    
    def save_pipeline_report(self) -> Path:
        """Save comprehensive pipeline report"""
        reports_dir = self.project_root / self.config['output']['reports_path']
        reports_dir.mkdir(parents=True, exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        report_file = reports_dir / f"etl_pipeline_report_{timestamp}.json"
        
        try:
            import json
            
            # Calculate duration
            if self.pipeline_report['start_time'] and self.pipeline_report['end_time']:
                duration = (self.pipeline_report['end_time'] - self.pipeline_report['start_time']).total_seconds()
                self.pipeline_report['duration_seconds'] = round(duration, 2)
            
            # Save report as JSON
            with open(report_file, 'w', encoding='utf-8') as f:
                json.dump(self.pipeline_report, f, indent=2, default=str)
            
            self.logger.info(f"Pipeline report saved: {report_file.name}")
            return report_file
            
        except Exception as e:
            self.logger.warning(f"Failed to save pipeline report: {e}")
            return None
    
    def run_cleaning_step(self, df: pd.DataFrame) -> pd.DataFrame:
        """Execute the data cleaning step"""
        if not self.config['pipeline'].get('enable_cleaning', True):
            self.logger.info("Cleaning step disabled, skipping...")
            return df
        
        self.logger.info("="*50)
        self.logger.info("STARTING CLEANING STEP")
        self.logger.info("="*50)
        
        try:
            cleaned_df, cleaning_report = self.cleaner.clean_data(df)
            self.pipeline_report['cleaning_report'] = cleaning_report
            
            self.logger.info(f"Cleaning completed successfully")
            self.logger.info(f"Rows: {len(df):,} -> {len(cleaned_df):,}")
            
            return cleaned_df
            
        except Exception as e:
            self.logger.error(f"Cleaning step failed: {e}")
            raise
    
    def run_transformation_step(self, df: pd.DataFrame) -> pd.DataFrame:
        """Execute the data transformation step"""
        if not self.config['pipeline'].get('enable_transformation', True):
            self.logger.info("Transformation step disabled, skipping...")
            return df
        
        self.logger.info("="*50)
        self.logger.info("STARTING TRANSFORMATION STEP")
        self.logger.info("="*50)
        
        try:
            transformed_df, transformation_report = self.transformer.transform_data(df)
            self.pipeline_report['transformation_report'] = transformation_report
            
            self.logger.info(f"Transformation completed successfully")
            self.logger.info(f"Columns: {len(df.columns)} -> {len(transformed_df.columns)}")
            
            return transformed_df
            
        except Exception as e:
            self.logger.error(f"Transformation step failed: {e}")
            raise
    
    def run_validation_step(self, df: pd.DataFrame) -> bool:
        """Execute the data validation step"""
        if not self.config['pipeline'].get('enable_validation', True):
            self.logger.info("Validation step disabled, skipping...")
            return True
        
        self.logger.info("="*50)
        self.logger.info("STARTING VALIDATION STEP")
        self.logger.info("="*50)
        
        try:
            validation_passed, validation_report = self.validator.validate_data(df)
            self.pipeline_report['validation_report'] = validation_report
            
            # Save detailed validation report
            reports_dir = self.project_root / self.config['output']['reports_path']
            reports_dir.mkdir(parents=True, exist_ok=True)
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            validation_file = reports_dir / f"validation_report_{timestamp}.json"
            
            import json
            with open(validation_file, 'w', encoding='utf-8') as f:
                json.dump(validation_report, f, indent=2, default=str)
            
            self.logger.info(f"Validation completed: {'PASSED' if validation_passed else 'FAILED'}")
            self.logger.info(f"Data quality score: {validation_report.get('data_quality_score', 0)}/100")
            self.logger.info(f"Detailed validation report saved: {validation_file.name}")
            
            return validation_passed
            
        except Exception as e:
            self.logger.error(f"Validation step failed: {e}")
            raise
    
    def select_final_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        """Select final columns for output based on configuration"""
        self.logger.info("Selecting final columns...")
        
        final_config = self.config.get('final_columns', {})
        
        # Collect all configured columns
        columns_to_keep = []
        
        # Add core fields
        columns_to_keep.extend(final_config.get('core_fields', []))
        
        # Add datetime fields
        columns_to_keep.extend(final_config.get('datetime_fields', []))
        
        # Add metadata fields
        columns_to_keep.extend(final_config.get('metadata_fields', []))
        
        # Add derived features if enabled
        if final_config.get('include_derived_features', True):
            # Add all derived columns that exist
            derived_patterns = [
                '_year', '_month', '_day_of_week', '_hour', '_quarter',
                '_score', '_category', '_length', '_word_count', 
                'is_', 'has_', '_days', '_group'
            ]
            
            for col in df.columns:
                if any(pattern in col for pattern in derived_patterns):
                    if col not in columns_to_keep:
                        columns_to_keep.append(col)
        
        # Only keep columns that actually exist in the dataframe
        available_columns = [col for col in columns_to_keep if col in df.columns]
        
        if not available_columns:
            # Fallback: keep all columns
            self.logger.warning("No columns configured for final output, keeping all columns")
            return df
        
        final_df = df[available_columns].copy()
        
        self.logger.info(f"Selected {len(available_columns)} columns for final output")
        self.logger.info(f"Final columns: {available_columns}")
        
        return final_df
    
    def run_pipeline(self) -> bool:
        """Execute the complete ETL pipeline"""
        self.pipeline_report['start_time'] = datetime.now()
        
        self.logger.info("="*60)
        self.logger.info("STARTING ETL PIPELINE")
        self.logger.info("="*60)
        self.logger.info(f"Pipeline ID: {self.pipeline_report['pipeline_id']}")
        self.logger.info(f"Configuration: {self.config_path}")
        
        try:
            # Step 1: Load source data
            df = self.load_source_data()
            
            # Step 1.1: Save raw data copy for traceability
            raw_file = self.save_raw_data(df)
            self.logger.info(f"Raw data archived: {raw_file.name}")
            
            # Step 2: Execute pipeline steps based on configuration
            execution_order = self.config['pipeline'].get('execution_order', ['cleaning', 'transformation', 'validation'])
            
            for step in execution_order:
                if step == 'cleaning':
                    df = self.run_cleaning_step(df)
                elif step == 'transformation':
                    df = self.run_transformation_step(df)
                elif step == 'validation':
                    validation_passed = self.run_validation_step(df)
                    # Continue pipeline even if validation fails (for analysis purposes)
                    if not validation_passed:
                        self.logger.warning("Validation failed, but continuing pipeline...")
                else:
                    self.logger.warning(f"Unknown pipeline step: {step}")
            
            # Step 3: Select final columns
            df = self.select_final_columns(df)
            
            # Step 4: Save processed data
            output_file = self.save_processed_data(df)
            
            # Mark pipeline as successful
            self.pipeline_report['overall_success'] = True
            self.pipeline_report['end_time'] = datetime.now()
            
            # Save pipeline report
            self.save_pipeline_report()
            
            self.logger.info("="*60)
            self.logger.info("ETL PIPELINE COMPLETED SUCCESSFULLY")
            self.logger.info("="*60)
            self.logger.info(f"Input: {self.pipeline_report['initial_rows']:,} rows")
            self.logger.info(f"Output: {self.pipeline_report['final_rows']:,} rows")
            self.logger.info(f"Processed file: {output_file}")
            
            return True
            
        except Exception as e:
            self.pipeline_report['end_time'] = datetime.now()
            self.pipeline_report['overall_success'] = False
            
            self.logger.error("="*60)
            self.logger.error("ETL PIPELINE FAILED")
            self.logger.error("="*60)
            self.logger.error(f"Error: {e}")
            
            # Still try to save the report for debugging
            self.save_pipeline_report()
            
            raise


def main():
    """Main entry point for ETL pipeline"""
    parser = argparse.ArgumentParser(description='Run ETL pipeline for support tickets')
    parser.add_argument('--config', '-c', help='Path to config file', default='etl/config/config.yaml')
    parser.add_argument('--verbose', '-v', action='store_true', help='Enable verbose logging')
    
    args = parser.parse_args()
    
    try:
        # Create and run pipeline
        pipeline = ETLPipeline(args.config)
        
        if args.verbose:
            logging.getLogger().setLevel(logging.DEBUG)
        
        success = pipeline.run_pipeline()
        
        if success:
            print("[SUCCESS] ETL Pipeline completed successfully!")
            sys.exit(0)
        else:
            print("[ERROR] ETL Pipeline failed!")
            sys.exit(1)
            
    except KeyboardInterrupt:
        print("[WARNING] Pipeline interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"[ERROR] Pipeline failed with error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()