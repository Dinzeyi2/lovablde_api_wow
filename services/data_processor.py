import pandas as pd
import numpy as np
from typing import Dict, Any, List
import json

class DataProcessor:
    """Handle complex data processing operations"""
    
    def process(self, file_path: str, operation: str, params: Dict[str, Any]) -> Dict[str, Any]:
        """Process data file with specified operation"""
        
        # Load data
        if file_path.endswith('.csv'):
            df = pd.read_csv(file_path)
        elif file_path.endswith('.json'):
            df = pd.read_json(file_path)
        elif file_path.endswith(('.xlsx', '.xls')):
            df = pd.read_excel(file_path)
        else:
            raise ValueError("Unsupported file format")
        
        # Execute operation
        if operation == 'transform':
            result = self.transform(df, params)
        elif operation == 'aggregate':
            result = self.aggregate(df, params)
        elif operation == 'filter':
            result = self.filter_data(df, params)
        elif operation == 'merge':
            result = self.merge(df, params)
        elif operation == 'clean':
            result = self.clean(df, params)
        elif operation == 'analyze':
            result = self.analyze(df, params)
        else:
            raise ValueError(f"Unsupported operation: {operation}")
        
        return result
    
    def transform(self, df: pd.DataFrame, params: Dict[str, Any]) -> Dict[str, Any]:
        """Transform data columns"""
        
        transformations = params.get('transformations', [])
        # transformations: [{"column": "price", "operation": "multiply", "value": 1.1}]
        
        df_transformed = df.copy()
        
        for trans in transformations:
            col = trans['column']
            op = trans['operation']
            value = trans.get('value')
            
            if op == 'multiply':
                df_transformed[col] = df_transformed[col] * value
            elif op == 'divide':
                df_transformed[col] = df_transformed[col] / value
            elif op == 'add':
                df_transformed[col] = df_transformed[col] + value
            elif op == 'subtract':
                df_transformed[col] = df_transformed[col] - value
            elif op == 'round':
                df_transformed[col] = df_transformed[col].round(value)
            elif op == 'lowercase':
                df_transformed[col] = df_transformed[col].str.lower()
            elif op == 'uppercase':
                df_transformed[col] = df_transformed[col].str.upper()
            elif op == 'strip':
                df_transformed[col] = df_transformed[col].str.strip()
        
        # Convert to records
        records = df_transformed.head(100).to_dict('records')  # Limit for API response
        
        return {
            'rows_processed': len(df_transformed),
            'columns': list(df_transformed.columns),
            'sample_data': records[:10],
            'transformations_applied': len(transformations)
        }
    
    def aggregate(self, df: pd.DataFrame, params: Dict[str, Any]) -> Dict[str, Any]:
        """Aggregate data by groups"""
        
        group_by = params.get('group_by', [])
        aggregations = params.get('aggregations', {})
        # aggregations: {"price": "mean", "quantity": "sum"}
        
        if not group_by:
            # Overall aggregations
            results = {}
            for col, func in aggregations.items():
                if func == 'mean':
                    results[f'{col}_mean'] = float(df[col].mean())
                elif func == 'sum':
                    results[f'{col}_sum'] = float(df[col].sum())
                elif func == 'count':
                    results[f'{col}_count'] = int(df[col].count())
                elif func == 'min':
                    results[f'{col}_min'] = float(df[col].min())
                elif func == 'max':
                    results[f'{col}_max'] = float(df[col].max())
            
            return {
                'total_rows': len(df),
                'aggregations': results
            }
        
        else:
            # Group by aggregations
            grouped = df.groupby(group_by)
            agg_result = grouped.agg(aggregations).reset_index()
            
            records = agg_result.to_dict('records')
            
            return {
                'groups': len(agg_result),
                'group_by': group_by,
                'results': records
            }
    
    def filter_data(self, df: pd.DataFrame, params: Dict[str, Any]) -> Dict[str, Any]:
        """Filter data based on conditions"""
        
        conditions = params.get('conditions', [])
        # conditions: [{"column": "price", "operator": ">", "value": 100}]
        
        df_filtered = df.copy()
        
        for condition in conditions:
            col = condition['column']
            op = condition['operator']
            value = condition['value']
            
            if op == '>':
                df_filtered = df_filtered[df_filtered[col] > value]
            elif op == '<':
                df_filtered = df_filtered[df_filtered[col] < value]
            elif op == '>=':
                df_filtered = df_filtered[df_filtered[col] >= value]
            elif op == '<=':
                df_filtered = df_filtered[df_filtered[col] <= value]
            elif op == '==':
                df_filtered = df_filtered[df_filtered[col] == value]
            elif op == '!=':
                df_filtered = df_filtered[df_filtered[col] != value]
            elif op == 'contains':
                df_filtered = df_filtered[df_filtered[col].str.contains(str(value), na=False)]
            elif op == 'in':
                df_filtered = df_filtered[df_filtered[col].isin(value)]
        
        records = df_filtered.head(100).to_dict('records')
        
        return {
            'original_rows': len(df),
            'filtered_rows': len(df_filtered),
            'sample_data': records[:10],
            'conditions_applied': len(conditions)
        }
    
    def merge(self, df: pd.DataFrame, params: Dict[str, Any]) -> Dict[str, Any]:
        """Merge with another dataset"""
        
        # This would need second file in production
        # For now, return structure
        
        return {
            'status': 'merge_operation',
            'message': 'Merge requires second dataset - use /api/v1/data/merge endpoint with two files'
        }
    
    def clean(self, df: pd.DataFrame, params: Dict[str, Any]) -> Dict[str, Any]:
        """Clean data (remove nulls, duplicates, etc.)"""
        
        operations = params.get('operations', ['remove_duplicates', 'remove_nulls'])
        
        df_cleaned = df.copy()
        stats = {}
        
        if 'remove_duplicates' in operations:
            before = len(df_cleaned)
            df_cleaned = df_cleaned.drop_duplicates()
            stats['duplicates_removed'] = before - len(df_cleaned)
        
        if 'remove_nulls' in operations:
            before = len(df_cleaned)
            df_cleaned = df_cleaned.dropna()
            stats['null_rows_removed'] = before - len(df_cleaned)
        
        if 'fill_nulls' in operations:
            fill_value = params.get('fill_value', 0)
            df_cleaned = df_cleaned.fillna(fill_value)
            stats['nulls_filled'] = True
        
        if 'remove_outliers' in operations:
            # Simple IQR-based outlier removal
            numeric_cols = df_cleaned.select_dtypes(include=[np.number]).columns
            before = len(df_cleaned)
            
            for col in numeric_cols:
                Q1 = df_cleaned[col].quantile(0.25)
                Q3 = df_cleaned[col].quantile(0.75)
                IQR = Q3 - Q1
                lower_bound = Q1 - 1.5 * IQR
                upper_bound = Q3 + 1.5 * IQR
                df_cleaned = df_cleaned[
                    (df_cleaned[col] >= lower_bound) & 
                    (df_cleaned[col] <= upper_bound)
                ]
            
            stats['outliers_removed'] = before - len(df_cleaned)
        
        return {
            'original_rows': len(df),
            'cleaned_rows': len(df_cleaned),
            'operations': operations,
            'statistics': stats
        }
    
    def analyze(self, df: pd.DataFrame, params: Dict[str, Any]) -> Dict[str, Any]:
        """Generate statistical analysis"""
        
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        categorical_cols = df.select_dtypes(include=['object']).columns.tolist()
        
        analysis = {
            'total_rows': len(df),
            'total_columns': len(df.columns),
            'numeric_columns': len(numeric_cols),
            'categorical_columns': len(categorical_cols),
            'memory_usage_mb': df.memory_usage(deep=True).sum() / 1024**2,
            'missing_values': df.isnull().sum().to_dict(),
            'numeric_stats': {},
            'categorical_stats': {}
        }
        
        # Numeric statistics
        for col in numeric_cols[:10]:  # Limit to first 10
            analysis['numeric_stats'][col] = {
                'mean': float(df[col].mean()),
                'median': float(df[col].median()),
                'std': float(df[col].std()),
                'min': float(df[col].min()),
                'max': float(df[col].max()),
                'null_count': int(df[col].isnull().sum())
            }
        
        # Categorical statistics
        for col in categorical_cols[:10]:  # Limit to first 10
            value_counts = df[col].value_counts().head(10)
            analysis['categorical_stats'][col] = {
                'unique_values': int(df[col].nunique()),
                'top_values': value_counts.to_dict(),
                'null_count': int(df[col].isnull().sum())
            }
        
        return analysis
