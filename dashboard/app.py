#!/usr/bin/env python3
"""
Streamlit Dashboard for Customer Support Tickets Analytics
3-Page Comprehensive Dashboard: Data Quality, Operations, and Performance
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import matplotlib.pyplot as plt
from pathlib import Path
from datetime import datetime, timedelta
import yaml
import subprocess
import sys
import time
import os


class TicketsDashboard:
    """Streamlit dashboard for customer support tickets analytics"""
    
    def __init__(self):
        """Initialize the dashboard"""
        self.config = self._load_config()
        self.project_root = Path(__file__).parent.parent
        
    def _load_config(self) -> dict:
        """Load configuration from YAML file"""
        # Try different possible locations for config file
        possible_paths = [
            Path("etl/config/config.yaml"),  # When running from project root
            Path("../etl/config/config.yaml"),  # When running from dashboard/ directory
            Path(__file__).parent.parent / "etl" / "config" / "config.yaml"  # Relative to script location
        ]
        
        for config_path in possible_paths:
            if config_path.exists():
                with open(config_path, 'r', encoding='utf-8') as f:
                    return yaml.safe_load(f)
        
        # Fallback configuration if no config file found
        return {
            'output': {
                'processed_data_path': 'data/processed'
            }
        }
    
    @st.cache_data
    def load_processed_data(_self) -> pd.DataFrame:
        """Load the latest processed data"""
        try:
            processed_dir = _self.project_root / _self.config['output']['processed_data_path']
            
            if not processed_dir.exists():
                return pd.DataFrame()
            
            # Find the latest parquet file
            parquet_files = list(processed_dir.glob("support_tickets_processed_*.parquet"))
            
            if not parquet_files:
                return pd.DataFrame()
            
            # Get the most recent file
            latest_file = sorted(parquet_files, key=lambda x: x.stat().st_mtime, reverse=True)[0]
            
            # Load the data
            df = pd.read_parquet(latest_file)
            return df
            
        except Exception as e:
            st.error(f"Error loading processed data: {str(e)}")
            return pd.DataFrame()
    
    def check_data_availability(self) -> bool:
        """Check if processed data is available"""
        processed_dir = self.project_root / self.config['output']['processed_data_path']
        
        if not processed_dir.exists():
            return False
        
        parquet_files = list(processed_dir.glob("support_tickets_processed_*.parquet"))
        return len(parquet_files) > 0
    
    def render_filters(self, df: pd.DataFrame) -> pd.DataFrame:
        """Render sidebar filters and return filtered dataframe"""
        if df.empty:
            return df
            
        st.sidebar.header("🔍 Filters")
        
        filtered_df = df.copy()
        
        # Date range filter
        date_columns = ['created_at', 'created_date', 'date_created']
        date_col = None
        for col in date_columns:
            if col in df.columns:
                date_col = col
                break
        
        if date_col:
            # Get both created and resolved date ranges to show full data span
            created_dates = pd.to_datetime(df[date_col], errors='coerce').dropna()
            
            # Also check resolved dates if available
            resolved_dates = None
            if 'resolved_at' in df.columns:
                resolved_dates = pd.to_datetime(df['resolved_at'], errors='coerce').dropna()
            
            # Determine the full date range from both created and resolved dates
            if not created_dates.empty:
                min_date = created_dates.min().date()
                max_date = created_dates.max().date()
                
                # Extend range if resolved dates go beyond created dates
                if resolved_dates is not None and not resolved_dates.empty:
                    resolved_min = resolved_dates.min().date()
                    resolved_max = resolved_dates.max().date()
                    min_date = min(min_date, resolved_min)
                    max_date = max(max_date, resolved_max)
                
                # Set reasonable default range - full available range for better user experience
                from datetime import datetime, timedelta
                today = datetime.now().date()
                
                # Use the full data range as default, but allow user to see all available data
                default_start = min_date
                default_end = max_date
                
                date_range = st.sidebar.date_input(
                    "📅 Created Date Range",
                    value=(default_start, default_end),
                    min_value=min_date,
                    max_value=max_date,
                    help=f"Full data range: {min_date} to {max_date} (includes future resolved dates)"
                )
                
                if len(date_range) == 2:
                    start_date, end_date = date_range
                    df_dates = pd.to_datetime(filtered_df[date_col], errors='coerce')
                    mask = (df_dates.dt.date >= start_date) & (df_dates.dt.date <= end_date)
                    filtered_df = filtered_df.loc[mask.fillna(False)]
        
        # Status filter
        if 'status' in df.columns:
            statuses = df['status'].dropna().unique().tolist()
            if len(statuses) > 0:
                selected_statuses = st.sidebar.multiselect(
                    "Ticket Status",
                    options=statuses,
                    default=statuses
                )
                
                if selected_statuses:
                    filtered_df = filtered_df[filtered_df['status'].isin(selected_statuses)]
        
        # Priority filter
        if 'priority' in df.columns:
            priorities = df['priority'].dropna().unique().tolist()
            if len(priorities) > 0:
                selected_priorities = st.sidebar.multiselect(
                    "Priority Level",
                    options=priorities,
                    default=priorities
                )
                
                if selected_priorities:
                    filtered_df = filtered_df[filtered_df['priority'].isin(selected_priorities)]
        
        # Show filtered data info
        if len(filtered_df) != len(df):
            st.sidebar.info(f"Showing {len(filtered_df):,} of {len(df):,} tickets")
        
        return filtered_df
    
    def render_data_quality_page(self, df: pd.DataFrame) -> None:
        """Page 1: Dataset Overview & Data Quality Dashboard"""
        st.title("🟦 Dataset Overview & Data Quality Dashboard")
        st.markdown("*Complete understanding of dataset after ETL including missing values, duplicates, and data distributions*")
        st.markdown("---")
        
        # ⭐ 1. Top-Level KPIs
        st.header("📊 Top-Level KPIs")
        col1, col2, col3, col4, col5 = st.columns(5)
        
        with col1:
            total_tickets = len(df)
            st.metric("Total Tickets", f"{total_tickets:,}")
        
        with col2:
            if 'description' in df.columns:
                missing_descriptions = (df['description'].isna().sum() / len(df)) * 100
                st.metric("% Missing Descriptions", f"{missing_descriptions:.1f}%")
            else:
                st.metric("% Missing Descriptions", "N/A")
        
        with col3:
            if 'assigned_agent' in df.columns:
                unassigned_tickets = (df['assigned_agent'].isna().sum() / len(df)) * 100
                st.metric("% Tickets Unassigned", f"{unassigned_tickets:.1f}%")
            else:
                st.metric("% Tickets Unassigned", "N/A")
        
        with col4:
            if 'status' in df.columns:
                resolved_tickets = ((df['status'] == 'Resolved').sum() / len(df)) * 100
                st.metric("% Resolved Tickets", f"{resolved_tickets:.1f}%")
            else:
                st.metric("% Resolved Tickets", "N/A")
        
        with col5:
            if 'sla_breach' in df.columns:
                sla_breach_rate = (df['sla_breach'].sum() / len(df)) * 100
                st.metric("SLA Breach Rate", f"{sla_breach_rate:.1f}%")
            else:
                st.metric("SLA Breach Rate", "N/A")
        
        st.markdown("---")
        
        # ⭐ 2. Missing Values Analysis
        st.header("📉 Missing Values Analysis")
        
        # Calculate missing values
        missing_data = df.isnull().sum()
        missing_percent = (missing_data / len(df)) * 100
        missing_df = pd.DataFrame({
            'Column': missing_data.index,
            'Missing Count': missing_data.values,
            'Missing %': missing_percent.values
        }).sort_values('Missing %', ascending=True)
        
        # Filter out columns with no missing values for the chart
        missing_df_chart = missing_df[missing_df['Missing %'] > 0]
        
        if not missing_df_chart.empty:
            fig = px.bar(
                missing_df_chart,
                x='Missing %',
                y='Column',
                orientation='h',
                title="Missing Values by Column (%)",
                color='Missing %',
                color_continuous_scale='Reds'
            )
            fig.update_layout(height=max(400, len(missing_df_chart) * 30))
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.success("🎉 No missing values found in the dataset!")
        
        # ⭐ 3. Data Types Summary
        st.header("🔍 Data Types Summary")
        
        dtype_df = pd.DataFrame({
            'Column': df.columns,
            'Type': df.dtypes.astype(str),
            'Nullable?': df.isnull().any(),
            'Non-Null Count': df.count(),
            'Unique Values': [df[col].nunique() for col in df.columns]
        })
        
        st.dataframe(dtype_df, use_container_width=True)
        
        # ⭐ 4. Duplicate Records Report
        st.header("🔄 Duplicate Records Report")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            duplicate_rows = df.duplicated().sum()
            st.metric("Duplicate Rows", f"{duplicate_rows:,}")
        
        with col2:
            if 'ticket_id' in df.columns:
                unique_ticket_ids = df['ticket_id'].nunique()
                st.metric("Unique Ticket IDs", f"{unique_ticket_ids:,}")
            else:
                st.metric("Unique Ticket IDs", "N/A")
        
        with col3:
            data_quality_score = ((len(df) - duplicate_rows) / len(df)) * 100
            st.metric("Data Quality Score", f"{data_quality_score:.1f}%")
        
        # ⭐ 5. Column-Wise Distribution Summary
        st.header("📋 Column-Wise Distribution Summary")
        
        summary_data = []
        for col in df.columns:
            summary_data.append({
                'Column Name': col,
                'Unique Count': df[col].nunique(),
                'Missing Count': df[col].isnull().sum(),
                'Example Values': str(df[col].dropna().head(3).tolist())[:100] + "..." if len(str(df[col].dropna().head(3).tolist())) > 100 else str(df[col].dropna().head(3).tolist())
            })
        
        summary_df = pd.DataFrame(summary_data)
        st.dataframe(summary_df, use_container_width=True, height=400)
    
    def render_operational_analytics_page(self, df: pd.DataFrame) -> None:
        """Page 2: Ticket Insights & Operational Analytics"""
        st.title("🟩 Ticket Insights & Operational Analytics")
        st.markdown("*Insights into ticket behavior across categories, priorities, channels, and status*")
        st.markdown("---")
        
        # ⭐ 1. Ticket Status Distribution
        st.header("📊 Ticket Status Distribution")
        
        if 'status' in df.columns:
            status_counts = df['status'].value_counts()
            
            col1, col2 = st.columns(2)
            
            with col1:
                fig = px.bar(
                    x=status_counts.index,
                    y=status_counts.values,
                    title="Tickets by Status",
                    color=status_counts.values,
                    color_continuous_scale='Blues'
                )
                st.plotly_chart(fig, use_container_width=True)
            
            with col2:
                fig = px.pie(
                    values=status_counts.values,
                    names=status_counts.index,
                    title="Status Distribution"
                )
                st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("Status information not available")
        
        # ⭐ 2. Priority Distribution
        st.header("⚡ Priority Distribution")
        
        if 'priority' in df.columns:
            priority_counts = df['priority'].value_counts()
            
            # Color mapping for priorities
            color_map = {'High': 'red', 'Medium': 'orange', 'Low': 'green'}
            colors = [color_map.get(priority, 'blue') for priority in priority_counts.index]
            
            fig = px.bar(
                x=priority_counts.index,
                y=priority_counts.values,
                title="Tickets by Priority Level",
                color=priority_counts.index,
                color_discrete_map=color_map
            )
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("Priority information not available")
        
        # ⭐ 3. Severity Breakdown
        st.header("🔥 Severity Breakdown")
        
        if 'severity' in df.columns:
            severity_counts = df['severity'].value_counts()
            
            fig = px.pie(
                values=severity_counts.values,
                names=severity_counts.index,
                title="Severity Distribution",
                color_discrete_sequence=px.colors.sequential.Reds_r
            )
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("Severity information not available")
        
        # ⭐ 4. Tickets by Origin/Channel
        st.header("📱 Tickets by Origin")
        
        channel_columns = ['channel', 'origin', 'source']
        channel_col = None
        for col in channel_columns:
            if col in df.columns:
                channel_col = col
                break
        
        if channel_col:
            channel_counts = df[channel_col].value_counts()
            
            fig = px.bar(
                x=channel_counts.values,
                y=channel_counts.index,
                orientation='h',
                title=f"Tickets by {channel_col.title()}",
                color=channel_counts.values,
                color_continuous_scale='Viridis'
            )
            fig.update_layout(height=400)
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("Channel/Origin information not available")
        
        # ⭐ 5. Tickets by Category
        st.header("📂 Tickets by Category")
        
        category_columns = ['category', 'ticket_type', 'issue_type']
        category_col = None
        for col in category_columns:
            if col in df.columns:
                category_col = col
                break
        
        if category_col:
            category_counts = df[category_col].value_counts()
            
            # Show missing category too
            missing_category = df[category_col].isnull().sum()
            if missing_category > 0:
                category_counts['(Missing/Blank)'] = missing_category
            
            fig = px.bar(
                x=category_counts.values,
                y=category_counts.index,
                orientation='h',
                title=f"Tickets by {category_col.title()}",
                color=category_counts.values,
                color_continuous_scale='Plasma'
            )
            fig.update_layout(height=max(400, len(category_counts) * 30))
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("Category information not available")
        
        # ⭐ 6. Agent Assignment Analysis
        st.header("👥 Agent Assignment Analysis")
        
        if 'assigned_agent' in df.columns:
            col1, col2 = st.columns(2)
            
            with col1:
                assigned_count = df['assigned_agent'].notna().sum()
                unassigned_count = df['assigned_agent'].isna().sum()
                
                assignment_data = pd.DataFrame({
                    'Assignment Status': ['Assigned', 'Unassigned'],
                    'Count': [assigned_count, unassigned_count]
                })
                
                fig = px.pie(
                    assignment_data,
                    values='Count',
                    names='Assignment Status',
                    title="Assignment Status Overview"
                )
                st.plotly_chart(fig, use_container_width=True)
            
            with col2:
                # Agent workload
                agent_counts = df['assigned_agent'].value_counts().head(10)
                
                if not agent_counts.empty:
                    fig = px.bar(
                        x=agent_counts.values,
                        y=agent_counts.index,
                        orientation='h',
                        title="Top 10 Agents by Ticket Count"
                    )
                    st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("Agent assignment information not available")
    
    def render_performance_analytics_page(self, df: pd.DataFrame) -> None:
        """Page 3: SLA, Time, Resolution & Performance Metrics"""
        st.title("🟧 SLA & Performance Analytics")
        st.markdown("*Analysis of resolution times, SLA performance, and support quality metrics*")
        st.markdown("---")
        
        # ⭐ 1. Resolution Time KPIs
        st.header("⏱️ Resolution Time KPIs")
        
        if 'resolution_days' in df.columns:
            resolution_data = df['resolution_days'].dropna()
            
            if not resolution_data.empty:
                col1, col2, col3, col4, col5 = st.columns(5)
                
                with col1:
                    avg_resolution = resolution_data.mean()
                    st.metric("Average Resolution Time", f"{avg_resolution:.1f} days")
                
                with col2:
                    median_resolution = resolution_data.median()
                    st.metric("Median Resolution Time", f"{median_resolution:.1f} days")
                
                with col3:
                    resolved_within_7 = (resolution_data <= 7).sum() / len(resolution_data) * 100
                    st.metric("% Resolved Within 7 Days", f"{resolved_within_7:.1f}%")
                
                with col4:
                    longest_resolution = resolution_data.max()
                    st.metric("Longest Resolution Time", f"{longest_resolution:.0f} days")
                
                with col5:
                    if 'status' in df.columns:
                        open_tickets = ((df['status'] != 'Resolved') & (df['status'] != 'Closed')).sum() / len(df) * 100
                        st.metric("% Tickets Still Open", f"{open_tickets:.1f}%")
                    else:
                        st.metric("% Tickets Still Open", "N/A")
        else:
            st.info("Resolution time data not available")
        
        st.markdown("---")
        
        # ⭐ 2. Resolution Time Distribution
        st.header("📊 Resolution Time Distribution")
        
        if 'resolution_days' in df.columns:
            resolution_data = df['resolution_days'].dropna()
            
            if not resolution_data.empty:
                col1, col2 = st.columns(2)
                
                with col1:
                    fig = px.histogram(
                        x=resolution_data,
                        title="Resolution Time Distribution",
                        nbins=30,
                        labels={'x': 'Days to Resolution', 'y': 'Number of Tickets'}
                    )
                    fig.add_vline(x=resolution_data.mean(), line_dash="dash", line_color="red", annotation_text="Average")
                    fig.add_vline(x=resolution_data.median(), line_dash="dash", line_color="blue", annotation_text="Median")
                    st.plotly_chart(fig, use_container_width=True)
                
                with col2:
                    # Box plot for outliers
                    fig = px.box(
                        y=resolution_data,
                        title="Resolution Time Box Plot (Outlier Detection)"
                    )
                    st.plotly_chart(fig, use_container_width=True)
                
                # Identify outliers
                outliers = resolution_data[resolution_data > 900]
                if len(outliers) > 0:
                    st.warning(f"⚠️ Found {len(outliers)} tickets with resolution time > 900 days (likely data quality issues)")
        
        # ⭐ 3. SLA Breach Analysis
        st.header("📈 SLA Breach Analysis")
        
        if 'sla_breach' in df.columns:
            breach_counts = df['sla_breach'].value_counts()
            
            col1, col2 = st.columns(2)
            
            with col1:
                fig = px.pie(
                    values=breach_counts.values,
                    names=['SLA Met' if x == False else 'SLA Breached' for x in breach_counts.index],
                    title="SLA Performance Overview",
                    color_discrete_map={'SLA Met': 'green', 'SLA Breached': 'red'}
                )
                st.plotly_chart(fig, use_container_width=True)
            
            with col2:
                breach_rate = (breach_counts.get(True, 0) / len(df)) * 100
                st.metric("SLA Breach Rate", f"{breach_rate:.1f}%")
                
                if breach_rate < 10:
                    st.success("🎉 Excellent SLA performance!")
                elif breach_rate < 20:
                    st.warning("⚠️ Moderate SLA performance")
                else:
                    st.error("❌ Poor SLA performance - needs attention")
        else:
            st.info("SLA breach information not available")
        
        # ⭐ 4. Time Series Analysis
        st.header("📈 Created vs Resolved Tickets Over Time")
        
        date_columns = ['created_date', 'date_created', 'created_at']
        created_col = None
        for col in date_columns:
            if col in df.columns:
                created_col = col
                break
        
        if created_col:
            df_time = df.copy()
            df_time[created_col] = pd.to_datetime(df_time[created_col], errors='coerce')
            df_time = df_time.dropna(subset=[created_col])
            
            if not df_time.empty:
                # Monthly trends
                df_time['month'] = df_time[created_col].dt.to_period('M')
                monthly_created = df_time.groupby('month').size()
                
                if 'status' in df.columns:
                    monthly_resolved = df_time[df_time['status'].isin(['Resolved', 'Closed'])].groupby('month').size()
                    
                    fig = go.Figure()
                    fig.add_trace(go.Scatter(
                        x=monthly_created.index.astype(str),
                        y=monthly_created.values,
                        mode='lines+markers',
                        name='Created',
                        line=dict(color='blue')
                    ))
                    fig.add_trace(go.Scatter(
                        x=monthly_resolved.index.astype(str),
                        y=monthly_resolved.values,
                        mode='lines+markers',
                        name='Resolved',
                        line=dict(color='green')
                    ))
                    
                    fig.update_layout(title="Monthly Tickets: Created vs Resolved")
                    st.plotly_chart(fig, use_container_width=True)
                else:
                    fig = px.line(
                        x=monthly_created.index.astype(str),
                        y=monthly_created.values,
                        title="Monthly Ticket Creation Trend"
                    )
                    st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("Date information not available for time series analysis")
        
        # ⭐ 5. Priority vs Resolution Time Analysis
        st.header("🎯 Priority vs Resolution Time Analysis")
        
        if 'priority' in df.columns and 'resolution_days' in df.columns:
            priority_resolution = df.groupby('priority')['resolution_days'].agg(['mean', 'median', 'count']).round(2)
            priority_resolution.columns = ['Avg Resolution Time (days)', 'Median Resolution Time (days)', 'Ticket Count']
            
            st.dataframe(priority_resolution, use_container_width=True)
            
            # Box plot by priority
            fig = px.box(
                df,
                x='priority',
                y='resolution_days',
                title="Resolution Time Distribution by Priority"
            )
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("Priority and resolution time data not available for analysis")
        
        # ⭐ 6. Open Tickets Aging Report
        st.header("📅 Open Tickets Aging Report")
        
        if 'status' in df.columns and created_col:
            open_tickets = df[(df['status'] != 'Resolved') & (df['status'] != 'Closed')].copy()
            
            if not open_tickets.empty and created_col in open_tickets.columns:
                open_tickets[created_col] = pd.to_datetime(open_tickets[created_col], errors='coerce')
                open_tickets = open_tickets.dropna(subset=[created_col])
                
                if not open_tickets.empty:
                    # Calculate age
                    today = pd.Timestamp.now()
                    open_tickets['age_days'] = (today - open_tickets[created_col]).dt.days
                    
                    # Categorize by age
                    age_categories = []
                    for age in open_tickets['age_days']:
                        if age < 7:
                            age_categories.append('< 7 days')
                        elif age < 30:
                            age_categories.append('7-30 days')
                        elif age < 60:
                            age_categories.append('30-60 days')
                        else:
                            age_categories.append('60+ days')
                    
                    open_tickets['age_category'] = age_categories
                    age_distribution = open_tickets['age_category'].value_counts()
                    
                    fig = px.bar(
                        x=age_distribution.index,
                        y=age_distribution.values,
                        title="Open Tickets by Age Category",
                        color=age_distribution.values,
                        color_continuous_scale='Reds'
                    )
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # Summary metrics
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("Total Open Tickets", len(open_tickets))
                    with col2:
                        st.metric("Avg Age (days)", f"{open_tickets['age_days'].mean():.1f}")
                    with col3:
                        critical_aging = (open_tickets['age_days'] > 60).sum()
                        st.metric("Critical Aging (60+ days)", critical_aging)
            else:
                st.info("No open tickets found or date data unavailable")
        else:
            st.info("Status and date information required for aging analysis")
    
    def run_dashboard(self) -> None:
        """Main dashboard execution"""
        # Configure Streamlit page
        st.set_page_config(
            page_title="Customer Support Tickets Dashboard",
            page_icon="🎫",
            layout="wide",
            initial_sidebar_state="expanded"
        )
        
        # Dashboard header
        st.title("🎟️ Support Tickets ETL Pipeline & Analytics Dashboard")
        st.markdown("---")
        
        # Check if processed data is available
        if not self.check_data_availability():
            # Show comprehensive upload instructions and pipeline information
            st.markdown("""
            <div style="text-align: center; padding: 30px; background-color: #f8f9fa; border-radius: 10px; margin: 20px 0;">
                <h2 style="color: black;">📊 No Data to Display</h2>
                <h3 style="color: black;">🚀 Support Tickets ETL Pipeline Ready for Processing</h3>
                <p style="font-size: 18px; color: #333; margin: 20px 0;">This pipeline is specifically designed for <strong>Customer Support Tickets Data</strong></p>
            </div>
            """, unsafe_allow_html=True)
            
            # Upload Instructions Section
            st.markdown("### 📋 Data Upload Instructions")
            
            st.markdown("""
            **📁 To get started:**
            - Go to the `source` folder and place your CSV file there
            - Then run the ETL pipeline to process your data
            
            **📄 CSV File Format:**
            Your CSV file should contain columns like: `ticket_id`, `customer_id`, `created_at`, `resolved_at`, `priority`, `status`, `category`, `product`, `assigned_agent`, `description`
            """)
            
            st.markdown("---")
            
            if st.button("🔄 Refresh Dashboard", help="Click after uploading data and running pipeline"):
                st.rerun()
            return
        
        # Load processed data
        try:
            df = self.load_processed_data()
            if df.empty:
                st.markdown("""
                <div style="text-align: center; padding: 30px;">
                    <h2>📊 No Data to Display</h2>
                    <h3>Complete the ETL pipeline and explore your analytics dashboard.</h3>
                    <p style="font-size: 16px; color: #666;">Data processing incomplete. Please try again later.</p>
                </div>
                """, unsafe_allow_html=True)
                return
        except Exception as e:
            st.markdown("""
            <div style="text-align: center; padding: 30px;">
                <h2>📊 No Data to Display</h2>
                <h3>Complete the ETL pipeline and explore your analytics dashboard.</h3>
                <p style="font-size: 16px; color: #666;">Unable to load data. Please ensure processing is complete.</p>
            </div>
            """, unsafe_allow_html=True)
            return
        
        # Show data info
        with st.expander("📊 Dataset Information", expanded=False):
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Total Records", f"{len(df):,}")
            with col2:
                st.metric("Total Columns", len(df.columns))
            with col3:
                memory_usage = df.memory_usage(deep=True).sum() / (1024 ** 2)
                st.metric("Memory Usage", f"{memory_usage:.1f} MB")
            with col4:
                # Get latest file timestamp
                processed_dir = self.project_root / self.config['output']['processed_data_path']
                parquet_files = list(processed_dir.glob("support_tickets_processed_*.parquet"))
                if parquet_files:
                    latest_file = sorted(parquet_files, key=lambda x: x.stat().st_mtime, reverse=True)[0]
                    last_updated = datetime.fromtimestamp(latest_file.stat().st_mtime)
                    st.metric("Last Updated", last_updated.strftime('%Y-%m-%d %H:%M'))
            
            # Add data range information
            st.markdown("#### 📅 Data Coverage")
            date_info_cols = st.columns(2)
            with date_info_cols[0]:
                if 'created_at' in df.columns:
                    created_dates = pd.to_datetime(df['created_at'], errors='coerce').dropna()
                    if not created_dates.empty:
                        st.info(f"**Created Dates**: {created_dates.min().strftime('%Y-%m-%d')} to {created_dates.max().strftime('%Y-%m-%d')}")
                        st.caption(f"Created span: {(created_dates.max() - created_dates.min()).days:,} days")
            with date_info_cols[1]:
                if 'resolved_at' in df.columns:
                    resolved_dates = pd.to_datetime(df['resolved_at'], errors='coerce').dropna()
                    if not resolved_dates.empty:
                        st.info(f"**Resolved Dates**: {resolved_dates.min().strftime('%Y-%m-%d')} to {resolved_dates.max().strftime('%Y-%m-%d')}")
                        future_resolved = resolved_dates > pd.Timestamp.now()
                        if future_resolved.any():
                            st.warning(f"⚠️ {future_resolved.sum():,} tickets have future resolution dates")
                        
            # Show overall data span
            if 'created_at' in df.columns and 'resolved_at' in df.columns:
                created_dates = pd.to_datetime(df['created_at'], errors='coerce').dropna()
                resolved_dates = pd.to_datetime(df['resolved_at'], errors='coerce').dropna()
                if not created_dates.empty and not resolved_dates.empty:
                    overall_min = min(created_dates.min(), resolved_dates.min())
                    overall_max = max(created_dates.max(), resolved_dates.max())
                    st.success(f"**📊 Complete Data Range**: {overall_min.strftime('%Y-%m-%d')} to {overall_max.strftime('%Y-%m-%d')} ({(overall_max - overall_min).days:,} days total)")
        
        # Create sidebar navigation
        st.sidebar.title("📊 Dashboard Navigation")
        
        # Add cache clear option
        if st.sidebar.button("🔄 Clear Cache & Refresh"):
            st.cache_data.clear()
            st.rerun()
        
        page = st.sidebar.radio(
            "Select Analytics Page:",
            ["🟦 Dataset Overview & Data Quality", "🟩 Ticket Insights & Operations", "🟧 SLA & Performance Analytics"]
        )
        
        # Apply filters
        filtered_df = self.render_filters(df)
        
        # Page 1: Dataset Overview & Data Quality Dashboard
        if page == "🟦 Dataset Overview & Data Quality":
            self.render_data_quality_page(filtered_df)
        
        # Page 2: Ticket Insights & Operational Analytics
        elif page == "🟩 Ticket Insights & Operations":
            self.render_operational_analytics_page(filtered_df)
        
        # Page 3: SLA, Time, Resolution & Performance Metrics
        elif page == "🟧 SLA & Performance Analytics":
            self.render_performance_analytics_page(filtered_df)


def main():
    """Main function to run the dashboard"""
    dashboard = TicketsDashboard()
    dashboard.run_dashboard()


if __name__ == "__main__":
    main()