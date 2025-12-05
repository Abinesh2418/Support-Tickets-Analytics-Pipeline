#!/usr/bin/env python3
"""
Professional Demo Script for ETL Pipeline Project
Run this to demonstrate the complete system to your mentor
"""

import pandas as pd
import os
from pathlib import Path
from datetime import datetime
import subprocess
import time

def print_header(title):
    print("\n" + "="*60)
    print(f"🎯 {title}")
    print("="*60)

def print_section(title):
    print(f"\n📋 {title}")
    print("-" * 40)

def main():
    print_header("ETL PIPELINE PROJECT DEMONSTRATION")
    print("🎓 Student: [Your Name]")
    print("📅 Date: " + datetime.now().strftime("%B %d, %Y"))
    print("📊 Project: Modular ETL Pipeline with Analytics Dashboard")
    
    print_section("1. PROJECT OVERVIEW")
    print("📊 Modular ETL Pipeline with Analytics Dashboard")
    print("✅ Components: Cleaning → Transformation → Validation → Dashboard")
    
    print_section("2. DATA PROCESSING")
    print("✅ Advanced Imputation: Mode, Forward Fill, Similarity-based, Regression")
    print("✅ Quality Assurance: Validation, Deduplication, Normalization")
    
    print_section("3. DATA ANALYSIS")
    
    # Find latest processed file
    processed_dir = Path("data/processed")
    if processed_dir.exists():
        parquet_files = list(processed_dir.glob("support_tickets_processed_*.parquet"))
        if parquet_files:
            latest_file = sorted(parquet_files, key=lambda x: x.stat().st_mtime, reverse=True)[0]
            df = pd.read_parquet(latest_file)
            file_time = datetime.fromtimestamp(latest_file.stat().st_mtime)
            
            print(f"📁 Latest Processed File: {latest_file.name}")
            print(f"📅 Processing Timestamp: {file_time.strftime('%Y-%m-%d %H:%M:%S')}")
            print(f"📊 Dataset Size: {len(df):,} records × {len(df.columns)} columns")
            print(f"💾 File Size: {latest_file.stat().st_size / 1024 / 1024:.2f} MB")
            
            print(f"\n🎯 Data Quality Metrics:")
            print(f"   - Records Processed: {len(df):,}")
            print(f"   - Date Range: {df['created_at'].min().date()} to {df['created_at'].max().date()}")
            print(f"   - Unresolved Tickets: {df['resolved_at'].isnull().sum():,}")
            print(f"   - Categories Filled: {len(df[df['category'].notna()]):,}")
            print(f"   - Products Filled: {len(df[df['product'].notna()]):,}")
            
            print(f"\n📈 Business Insights:")
            priority_dist = df['priority'].value_counts()
            for priority, count in priority_dist.head(3).items():
                print(f"   - {priority} Priority: {count:,} tickets ({count/len(df)*100:.1f}%)")
    
    print_section("4. FILTER VERIFICATION")
    
    if processed_dir.exists() and parquet_files:
        print("🔍 Dashboard Filter Verification - All Combinations:")
        
        # Priority-based filtering verification
        print(f"\n📊 PRIORITY-BASED FILTERING:")
        priority_counts = df['priority'].value_counts()
        for priority in priority_counts.index:
            filtered_df = df[df['priority'] == priority]
            print(f"   📌 {priority} Priority Only:")
            print(f"      - Records: {len(filtered_df):,}")
            print(f"      - Percentage: {len(filtered_df)/len(df)*100:.1f}%")
            print(f"      - Date Range: {filtered_df['created_at'].min().date()} to {filtered_df['created_at'].max().date()}")
            
            # Top statuses for this priority
            top_statuses = filtered_df['status'].value_counts().head(3)
            print(f"      - Top Statuses: {', '.join([f'{s}({c})' for s, c in top_statuses.items()])}")
        
        # Status-based filtering verification  
        print(f"\n📋 STATUS-BASED FILTERING:")
        status_counts = df['status'].value_counts()
        for status in status_counts.index:
            filtered_df = df[df['status'] == status]
            print(f"   🎫 {status} Status Only:")
            print(f"      - Records: {len(filtered_df):,}")
            print(f"      - Percentage: {len(filtered_df)/len(df)*100:.1f}%")
            
            # Resolution rate for this status
            if status in ['Resolved', 'Closed']:
                resolved_count = len(filtered_df[filtered_df['resolved_at'].notna()])
                print(f"      - Actually Resolved: {resolved_count:,} ({resolved_count/len(filtered_df)*100:.1f}%)")
            else:
                unresolved_count = len(filtered_df[filtered_df['resolved_at'].isna()])
                print(f"      - Unresolved: {unresolved_count:,} ({unresolved_count/len(filtered_df)*100:.1f}%)")
            
            # Top priorities for this status
            top_priorities = filtered_df['priority'].value_counts().head(3)
            print(f"      - Top Priorities: {', '.join([f'{p}({c})' for p, c in top_priorities.items()])}")
        
        # COMPREHENSIVE STATUS-PRIORITY COMBINATIONS
        print(f"\n🎯 COMPREHENSIVE STATUS-PRIORITY COMBINATIONS:")
        print("    📊 All Status × Priority Combinations for Mentor Verification")
        print("-" * 70)
        
        # All status values
        statuses = ['Closed', 'Resolved', 'Open', 'In Progress', 'On Hold', 'New']
        priorities = ['High', 'Medium', 'Low']
        
        # Create verification table
        print(f"\n{'Status':<12} {'High':<8} {'Medium':<8} {'Low':<8} {'Total':<8}")
        print("-" * 50)
        
        for status in statuses:
            high_count = len(df[(df['status'] == status) & (df['priority'] == 'High')])
            medium_count = len(df[(df['status'] == status) & (df['priority'] == 'Medium')])
            low_count = len(df[(df['status'] == status) & (df['priority'] == 'Low')])
            total_count = len(df[df['status'] == status])
            
            print(f"{status:<12} {high_count:<8} {medium_count:<8} {low_count:<8} {total_count:<8}")
        
        print("\n📋 DETAILED COMBINATION ANALYSIS:")
        
        for status in statuses:
            status_df = df[df['status'] == status]
            if len(status_df) > 0:
                print(f"\n   📌 {status.upper()} STATUS:")
                for priority in priorities:
                    combo_df = df[(df['status'] == status) & (df['priority'] == priority)]
                    if len(combo_df) > 0:
                        percentage = (len(combo_df) / len(status_df) * 100)
                        print(f"      └─ {priority}: {len(combo_df):,} tickets ({percentage:.1f}%)")
                        
                        # Additional insights for each combination
                        if 'age_days' in combo_df.columns:
                            avg_age = combo_df['age_days'].mean()
                            print(f"         Average age: {avg_age:.1f} days")
                        
                        if status in ['Resolved', 'Closed'] and 'resolution_days' in combo_df.columns:
                            resolved_with_time = combo_df[combo_df['resolution_days'].notna()]
                            if len(resolved_with_time) > 0:
                                avg_resolution = resolved_with_time['resolution_days'].mean()
                                print(f"         Average resolution time: {avg_resolution:.1f} days")
                    else:
                        print(f"      └─ {priority}: 0 tickets")
        
        # Quick verification commands for mentor
        print(f"\n🎯 MENTOR VERIFICATION COMMANDS:")
        print("   Use these exact numbers to verify dashboard filters:")
        
        for status in statuses:
            for priority in priorities:
                combo_count = len(df[(df['status'] == status) & (df['priority'] == priority)])
                if combo_count > 0:
                    print(f"   • {status} + {priority}: {combo_count:,} records")
        
        # Date range filtering example
        print(f"\n📅 DATE RANGE FILTERING EXAMPLES:")
        recent_year = df[df['created_at'].dt.year == 2024]
        print(f"   📆 2024 Tickets Only:")
        print(f"      - Records: {len(recent_year):,}")
        print(f"      - Priority Distribution: {recent_year['priority'].value_counts().to_dict()}")
        
        old_year = df[df['created_at'].dt.year == 2020]
        print(f"   📆 2020 Tickets Only:")
        print(f"      - Records: {len(old_year):,}")
        print(f"      - Status Distribution: {dict(old_year['status'].value_counts().head(3))}")
        
        print(f"\n🎯 DASHBOARD FILTER VERIFICATION COMMANDS:")
        print(f"   1. Select 'High' priority → Verify {priority_counts.get('High', 0):,} records")
        print(f"   2. Select 'New' status → Verify {status_counts.get('New', 0):,} records") 
        
        # Calculate the combination for verification
        medium_in_progress = len(df[(df['priority'] == 'Medium') & (df['status'] == 'In Progress')])
        print(f"   3. Select 'Medium' + 'In Progress' → Verify {medium_in_progress:,} records")
        print(f"   4. Select date range 2024 → Verify {len(recent_year):,} records")
        print(f"   5. Clear all filters → Verify {len(df):,} total records")


    
    print_header("DEMONSTRATION COMPLETE")
    
if __name__ == "__main__":
    main()