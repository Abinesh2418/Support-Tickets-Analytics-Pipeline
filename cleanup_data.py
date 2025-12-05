#!/usr/bin/env python3
"""
Data Cleanup Script
Removes all files from raw and processed data folders for testing purposes
"""

import os
import shutil
from pathlib import Path
import logging
from datetime import datetime

def setup_logging():
    """Setup logging for cleanup operations"""
    log_dir = Path("logs")
    log_dir.mkdir(exist_ok=True)
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_dir / "cleanup.log"),
            logging.StreamHandler()
        ]
    )
    return logging.getLogger(__name__)

def safe_print(message):
    """Print message safely"""
    try:
        print(message)
    except UnicodeEncodeError:
        safe_message = message.encode('ascii', 'replace').decode('ascii')
        print(safe_message)

def delete_files_in_directory(directory_path, file_pattern="*", logger=None):
    """Delete all files matching pattern in the given directory"""
    dir_path = Path(directory_path)
    
    if not dir_path.exists():
        if logger:
            logger.warning(f"Directory does not exist: {dir_path}")
        safe_print(f"⚠️  Directory not found: {dir_path}")
        return 0
    
    deleted_count = 0
    total_size = 0
    
    try:
        # Get all files matching pattern
        files_to_delete = list(dir_path.glob(file_pattern))
        
        if not files_to_delete:
            if logger:
                logger.info(f"No files found to delete in: {dir_path}")
            safe_print(f"✓ No files to delete in: {dir_path}")
            return 0
        
        # Delete each file
        for file_path in files_to_delete:
            if file_path.is_file():
                file_size = file_path.stat().st_size
                total_size += file_size
                
                file_path.unlink()
                deleted_count += 1
                
                if logger:
                    logger.info(f"Deleted: {file_path.name} ({file_size:,} bytes)")
                safe_print(f"🗑️  Deleted: {file_path.name}")
        
        if logger:
            logger.info(f"Deleted {deleted_count} files from {dir_path} (Total: {total_size:,} bytes)")
        safe_print(f"✓ Deleted {deleted_count} files from {dir_path} ({total_size/1024/1024:.2f} MB)")
        
    except Exception as e:
        if logger:
            logger.error(f"Error deleting files from {dir_path}: {e}")
        safe_print(f"❌ Error deleting files from {dir_path}: {e}")
    
    return deleted_count

def cleanup_unnecessary_files(logger=None):
    """Remove unnecessary files that are no longer needed"""
    files_to_remove = [
        "__pycache__",
        "*.pyc"
    ]
    
    removed_count = 0
    
    for file_pattern in files_to_remove:
        if file_pattern == "__pycache__":
            # Remove __pycache__ directories
            for pycache_dir in Path(".").rglob("__pycache__"):
                if pycache_dir.is_dir():
                    try:
                        shutil.rmtree(pycache_dir)
                        removed_count += 1
                        if logger:
                            logger.info(f"Removed directory: {pycache_dir}")
                        safe_print(f"🗑️  Removed: {pycache_dir}")
                    except Exception as e:
                        if logger:
                            logger.error(f"Error removing {pycache_dir}: {e}")
                        safe_print(f"❌ Error removing {pycache_dir}: {e}")
        
        elif "*" in file_pattern:
            # Handle wildcard patterns
            for file_path in Path(".").rglob(file_pattern):
                if file_path.is_file():
                    try:
                        file_path.unlink()
                        removed_count += 1
                        if logger:
                            logger.info(f"Removed file: {file_path}")
                        safe_print(f"🗑️  Removed: {file_path}")
                    except Exception as e:
                        if logger:
                            logger.error(f"Error removing {file_path}: {e}")
                        safe_print(f"❌ Error removing {file_path}: {e}")
        
        else:
            # Handle specific files
            file_path = Path(file_pattern)
            if file_path.exists():
                try:
                    if file_path.is_file():
                        file_size = file_path.stat().st_size
                        file_path.unlink()
                        removed_count += 1
                        if logger:
                            logger.info(f"Removed file: {file_path} ({file_size:,} bytes)")
                        safe_print(f"🗑️  Removed: {file_path} ({file_size/1024:.1f} KB)")
                    elif file_path.is_dir():
                        shutil.rmtree(file_path)
                        removed_count += 1
                        if logger:
                            logger.info(f"Removed directory: {file_path}")
                        safe_print(f"🗑️  Removed directory: {file_path}")
                except Exception as e:
                    if logger:
                        logger.error(f"Error removing {file_path}: {e}")
                    safe_print(f"❌ Error removing {file_path}: {e}")
    
    return removed_count

def cleanup_data_folders(keep_source=True):
    """Main cleanup function"""
    logger = setup_logging()
    
    safe_print("🧹 Data Cleanup Script")
    safe_print("=" * 40)
    safe_print(f"Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    safe_print("")
    
    total_deleted = 0
    
    # Define directories to clean
    directories_to_clean = [
        {
            "path": "data/raw",
            "pattern": "*.parquet",
            "description": "Raw data files"
        },
        {
            "path": "data/processed", 
            "pattern": "*.parquet",
            "description": "Processed data files"
        },
        {
            "path": "data/reports",
            "pattern": "*.json",
            "description": "Report files"
        },
        {
            "path": "logs",
            "pattern": "*.log",
            "description": "Log files"
        }
    ]
    
    # Optional: Also clean source if not keeping it
    if not keep_source:
        directories_to_clean.append({
            "path": "source",
            "pattern": "*.csv", 
            "description": "Source data files"
        })
    
    # Clean each directory
    for dir_info in directories_to_clean:
        safe_print(f"🗂️  Cleaning {dir_info['description']}...")
        count = delete_files_in_directory(
            dir_info["path"], 
            dir_info["pattern"], 
            logger
        )
        total_deleted += count
        safe_print("")
    
    # Clean unnecessary files
    safe_print("🧹 Removing unnecessary files...")
    unnecessary_count = cleanup_unnecessary_files(logger)
    total_deleted += unnecessary_count
    safe_print("")
    
    # Summary
    safe_print("=" * 40)
    if total_deleted > 0:
        safe_print(f"🎉 Cleanup completed! Deleted {total_deleted} files total.")
        if keep_source:
            safe_print("📁 Source files preserved for reprocessing.")
    else:
        safe_print("✨ All folders are already clean!")
    
    safe_print(f"Finished at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    logger.info(f"Cleanup completed. Total files deleted: {total_deleted}")
    
    return total_deleted

def interactive_cleanup():
    """Interactive cleanup with user confirmation"""
    safe_print("🧹 Interactive Data Cleanup")
    safe_print("=" * 40)
    
    # Show current data status
    raw_dir = Path("data/raw")
    processed_dir = Path("data/processed")
    reports_dir = Path("data/reports")
    logs_dir = Path("logs")
    source_dir = Path("source")
    
    safe_print("📊 Current Data Status:")
    
    # Check raw files
    if raw_dir.exists():
        raw_files = list(raw_dir.glob("*.parquet"))
        safe_print(f"   Raw files: {len(raw_files)} Parquet files")
        for f in raw_files[:3]:  # Show first 3
            safe_print(f"      - {f.name}")
        if len(raw_files) > 3:
            safe_print(f"      ... and {len(raw_files) - 3} more")
    else:
        safe_print("   Raw files: No raw directory")
    
    # Check processed files  
    if processed_dir.exists():
        processed_files = list(processed_dir.glob("*.parquet"))
        safe_print(f"   Processed files: {len(processed_files)} Parquet files")
        for f in processed_files[:3]:  # Show first 3
            safe_print(f"      - {f.name}")
        if len(processed_files) > 3:
            safe_print(f"      ... and {len(processed_files) - 3} more")
    else:
        safe_print("   Processed files: No processed directory")
    
    # Check report files
    if reports_dir.exists():
        report_files = list(reports_dir.glob("*.json"))
        safe_print(f"   Report files: {len(report_files)} JSON files")
        for f in report_files[:3]:  # Show first 3
            safe_print(f"      - {f.name}")
        if len(report_files) > 3:
            safe_print(f"      ... and {len(report_files) - 3} more")
    else:
        safe_print("   Report files: No reports directory")
    
    # Check log files
    if logs_dir.exists():
        log_files = list(logs_dir.glob("*.log"))
        safe_print(f"   Log files: {len(log_files)} log files")
        for f in log_files[:3]:  # Show first 3
            safe_print(f"      - {f.name}")
        if len(log_files) > 3:
            safe_print(f"      ... and {len(log_files) - 3} more")
    else:
        safe_print("   Log files: No logs directory")
    
    # Check source files
    if source_dir.exists():
        source_files = list(source_dir.glob("*.csv"))
        safe_print(f"   Source files: {len(source_files)} CSV files")
        for f in source_files:
            safe_print(f"      - {f.name}")
    else:
        safe_print("   Source files: No source directory")
    
    # Check unnecessary files
    unnecessary_files = []
    
    pycache_dirs = list(Path(".").rglob("__pycache__"))
    if pycache_dirs:
        unnecessary_files.append(f"{len(pycache_dirs)} __pycache__ directories")
    
    if unnecessary_files:
        safe_print(f"   Unnecessary files: {', '.join(unnecessary_files)}")
    else:
        safe_print("   Unnecessary files: None")
    
    safe_print("")
    
    # Get user confirmation
    try:
        choice = input("🤔 What would you like to do?\n" +
                      "   1. Clean raw and processed (keep source)\n" +
                      "   2. Clean all folders (including source)\n" +
                      "   3. Cancel\n" +
                      "Enter choice (1-3): ").strip()
        
        if choice == "1":
            safe_print("\n🚀 Cleaning raw and processed folders...")
            cleanup_data_folders(keep_source=True)
        elif choice == "2":
            confirm = input("\n⚠️  This will delete ALL data including source files. Are you sure? (yes/no): ").strip().lower()
            if confirm in ['yes', 'y']:
                safe_print("\n🚀 Cleaning all folders...")
                cleanup_data_folders(keep_source=False)
            else:
                safe_print("❌ Cleanup cancelled.")
        elif choice == "3":
            safe_print("❌ Cleanup cancelled.")
        else:
            safe_print("❌ Invalid choice. Cleanup cancelled.")
            
    except KeyboardInterrupt:
        safe_print("\n❌ Cleanup cancelled by user.")
    except Exception as e:
        safe_print(f"❌ Error during cleanup: {e}")

def quick_cleanup():
    """Quick cleanup without prompts - for automation"""
    return cleanup_data_folders(keep_source=True)

def main():
    """Main function with command line options"""
    import sys
    
    if len(sys.argv) > 1:
        option = sys.argv[1].lower()
        
        if option in ['-q', '--quick']:
            # Quick cleanup mode
            quick_cleanup()
        elif option in ['-a', '--all']:
            # Clean all including source
            cleanup_data_folders(keep_source=False)
        elif option in ['-h', '--help']:
            # Show help
            safe_print("🧹 Data Cleanup Script")
            safe_print("")
            safe_print("Usage:")
            safe_print("  python cleanup_data.py           # Interactive mode")
            safe_print("  python cleanup_data.py -q        # Quick cleanup (keep source)")
            safe_print("  python cleanup_data.py -a        # Clean all folders")
            safe_print("  python cleanup_data.py -h        # Show this help")
            safe_print("")
            safe_print("Directories cleaned:")
            safe_print("  - data/raw/       (Parquet files)")
            safe_print("  - data/processed/ (Parquet files)")
            safe_print("  - data/reports/   (JSON files)")
            safe_print("  - logs/           (Log files)")
            safe_print("  - source/         (CSV files, only with -a flag)")
            safe_print("")
            safe_print("Files removed:")
            safe_print("  - __pycache__/    (Python cache directories)")
            safe_print("  - *.pyc           (Python compiled files)")
        else:
            safe_print(f"❌ Unknown option: {option}")
            safe_print("Use -h for help")
    else:
        # Interactive mode
        interactive_cleanup()

if __name__ == "__main__":
    main()