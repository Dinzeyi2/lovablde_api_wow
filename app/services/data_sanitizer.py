"""
Data Sanitizer - Automatically clean and validate data before training
Prevents garbage-in-garbage-out scenarios
"""

import pandas as pd
import numpy as np
from typing import Dict, Any, Tuple, List
import os
from datetime import datetime

class DataSanitizer:
    """
    Automatically detect and fix common data quality issues
    Provides detailed report of what was fixed
    """
    
    def __init__(self):
        self.min_rows = 10
        self.max_missing_percent = 50
        self.max_rows = 1_000_000  # Safety limit
    
    def sanitize_file(self, file_path: str) -> Tuple[str, Dict[str, Any]]:
        """
        Sanitize a CSV file and return cleaned file path + report
        """
        
        # Load data
        try:
            df = pd.read_csv(file_path)
        except Exception as e:
            return file_path, {
                "critical_issues": [f"Failed to load CSV: {str(e)}"],
                "usable": False
            }
        
        # Sanitize dataframe
        df_clean, report = self.sanitize_dataframe(df)
        
        # Save cleaned version
        clean_path = file_path.replace('.csv', '_clean.csv')
        df_clean.to_csv(clean_path, index=False)
        
        return clean_path, report
    
    def sanitize_dataframe(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """
        Clean a dataframe and return cleaned version + detailed report
        """
        
        report = {
            "original_rows": len(df),
            "original_columns": len(df.columns),
            "issues": [],
            "fixes_applied": [],
            "warnings": [],
            "critical_issues": [],
            "data_quality_score": 100,
            "usable": True
        }
        
        # Critical Check 1: Minimum rows
        if len(df) < self.min_rows:
            report["critical_issues"].append(
                f"Too few rows: {len(df)} (need at least {self.min_rows})"
            )
            report["usable"] = False
            return df, report
        
        # Critical Check 2: Maximum rows (prevent memory issues)
        if len(df) > self.max_rows:
            report["warnings"].append(
                f"Dataset too large: {len(df)} rows. Using first {self.max_rows} rows."
            )
            df = df.head(self.max_rows)
            report["fixes_applied"].append(f"Truncated to {self.max_rows} rows")
        
        # Check 3: Missing values
        missing_counts = df.isnull().sum()
        total_cells = len(df) * len(df.columns)
        missing_percent = (missing_counts.sum() / total_cells) * 100
        
        if missing_percent > self.max_missing_percent:
            report["critical_issues"].append(
                f"Too many missing values: {missing_percent:.1f}% (max {self.max_missing_percent}%)"
            )
            report["data_quality_score"] -= 50
        elif missing_percent > 0:
            report["issues"].append(f"Missing values: {missing_percent:.1f}%")
            report["data_quality_score"] -= min(missing_percent, 30)
            
            # Fix: Fill missing values
            for col in df.columns:
                if df[col].isnull().any():
                    if df[col].dtype in ['int64', 'float64']:
                        # Fill numeric with median
                        df[col].fillna(df[col].median(), inplace=True)
                        report["fixes_applied"].append(f"Filled {col} nulls with median")
                    else:
                        # Fill categorical with mode or 'unknown'
                        mode_val = df[col].mode()[0] if not df[col].mode().empty else 'unknown'
                        df[col].fillna(mode_val, inplace=True)
                        report["fixes_applied"].append(f"Filled {col} nulls with mode/unknown")
        
        # Check 4: Data types and encoding
        for col in df.columns:
            # Detect and fix mixed types
            if df[col].dtype == 'object':
                # Try to convert to numeric
                try:
                    df[col] = pd.to_numeric(df[col], errors='ignore')
                    if df[col].dtype in ['int64', 'float64']:
                        report["fixes_applied"].append(f"Converted {col} to numeric")
                except:
                    pass
                
                # Check for too many unique values (potential data issue)
                unique_ratio = df[col].nunique() / len(df)
                if unique_ratio > 0.95:
                    report["warnings"].append(
                        f"{col} has {df[col].nunique()} unique values (might be an ID column)"
                    )
        
        # Check 5: Duplicates
        duplicates = df.duplicated().sum()
        if duplicates > 0:
            dup_percent = (duplicates / len(df)) * 100
            report["issues"].append(f"Duplicate rows: {duplicates} ({dup_percent:.1f}%)")
            df = df.drop_duplicates()
            report["fixes_applied"].append(f"Removed {duplicates} duplicate rows")
            report["data_quality_score"] -= min(dup_percent, 20)
        
        # Check 6: Column name issues
        problematic_cols = []
        for col in df.columns:
            # Check for spaces, special characters
            if ' ' in col or not col.replace('_', '').replace('-', '').isalnum():
                clean_col = col.strip().replace(' ', '_').replace('-', '_')
                clean_col = ''.join(c for c in clean_col if c.isalnum() or c == '_')
                df.rename(columns={col: clean_col}, inplace=True)
                problematic_cols.append(col)
        
        if problematic_cols:
            report["fixes_applied"].append(f"Cleaned {len(problematic_cols)} column names")
        
        # Check 7: Outliers (for numeric columns)
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        outliers_removed = 0
        
        for col in numeric_cols:
            Q1 = df[col].quantile(0.25)
            Q3 = df[col].quantile(0.75)
            IQR = Q3 - Q1
            
            lower_bound = Q1 - 3 * IQR  # 3 IQR (more lenient than 1.5)
            upper_bound = Q3 + 3 * IQR
            
            outliers = ((df[col] < lower_bound) | (df[col] > upper_bound)).sum()
            
            if outliers > 0 and outliers < len(df) * 0.05:  # Remove if < 5% of data
                df = df[(df[col] >= lower_bound) & (df[col] <= upper_bound)]
                outliers_removed += outliers
        
        if outliers_removed > 0:
            report["fixes_applied"].append(f"Removed {outliers_removed} extreme outliers")
        
        # Check 8: Target column validation (for supervised learning)
        if 'target' in df.columns or 'label' in df.columns:
            target_col = 'target' if 'target' in df.columns else 'label'
            
            # Check target distribution
            value_counts = df[target_col].value_counts()
            min_class_size = value_counts.min()
            
            if min_class_size < 5:
                report["warnings"].append(
                    f"Target column has class with only {min_class_size} samples"
                )
                report["data_quality_score"] -= 15
        
        # Final validation
        report["final_rows"] = len(df)
        report["final_columns"] = len(df.columns)
        
        if report["data_quality_score"] < 50:
            report["critical_issues"].append(
                f"Data quality score too low: {report['data_quality_score']}/100"
            )
            report["usable"] = False
        
        # Add recommendations
        report["recommendations"] = self._generate_recommendations(df, report)
        
        return df, report
    
    def _generate_recommendations(self, df: pd.DataFrame, report: Dict) -> List[str]:
        """Generate actionable recommendations for improving data quality"""
        
        recommendations = []
        
        if len(df) < 100:
            recommendations.append("Collect more data - ML models work better with 100+ samples")
        
        if len(df.columns) > 50:
            recommendations.append("Consider feature selection - too many columns can hurt performance")
        
        if df.select_dtypes(include=['object']).shape[1] > 10:
            recommendations.append("Many categorical columns detected - consider one-hot encoding")
        
        if report["data_quality_score"] < 80:
            recommendations.append("Data quality is below optimal - review the issues list above")
        
        return recommendations
    
    def validate_for_model_type(self, df: pd.DataFrame, model_type: str) -> Dict[str, Any]:
        """
        Validate data is suitable for specific model type
        """
        
        validation = {"valid": True, "issues": []}
        
        if model_type == "recommendation":
            # Need user_id, item_id, rating columns (or similar)
            required_cols = 3
            if len(df.columns) < required_cols:
                validation["valid"] = False
                validation["issues"].append(
                    f"Recommendation models need at least {required_cols} columns (user, item, rating)"
                )
        
        elif model_type in ["classification", "regression"]:
            # Need at least 2 columns (features + target)
            if len(df.columns) < 2:
                validation["valid"] = False
                validation["issues"].append(
                    f"{model_type} needs at least 2 columns (features + target)"
                )
        
        elif model_type == "clustering":
            # Need numeric features
            if df.select_dtypes(include=[np.number]).shape[1] < 1:
                validation["valid"] = False
                validation["issues"].append("Clustering needs at least 1 numeric column")
        
        return validation
