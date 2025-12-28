"""
Pre-built algorithms catalog with descriptions and pricing tiers
"""

PREBUILT_ALGORITHMS = {
    'fraud-detection': {
        'name': 'Fraud Detection',
        'description': 'Detect fraudulent transactions using multi-signal risk scoring',
        'category': 'security',
        'pricing_tier': 'pro',
        'parameters': {
            'transaction_amount': 'float',
            'user_location': 'string (country code)',
            'device_fingerprint': 'string',
            'account_age_days': 'integer',
            'transaction_time': 'integer (hour 0-23)'
        },
        'returns': {
            'is_fraud': 'boolean',
            'risk_score': 'float (0-1)',
            'risk_factors': 'list',
            'recommendation': 'string (approve/review/block)'
        },
        'use_cases': [
            'E-commerce payment validation',
            'Financial transaction screening',
            'Account security monitoring'
        ]
    },
    
    'dynamic-pricing': {
        'name': 'Dynamic Pricing',
        'description': 'Calculate optimal prices based on competition, inventory, and demand',
        'category': 'ecommerce',
        'pricing_tier': 'pro',
        'parameters': {
            'base_price': 'float',
            'competitor_prices': 'list of floats',
            'inventory_level': 'integer (percentage)',
            'demand_score': 'float (0-1)',
            'cost': 'float'
        },
        'returns': {
            'recommended_price': 'float',
            'profit_margin_percent': 'float',
            'price_change_percent': 'float',
            'reasoning': 'string'
        },
        'use_cases': [
            'E-commerce pricing optimization',
            'SaaS plan pricing',
            'Retail markdown strategy'
        ]
    },
    
    'recommendation-collab': {
        'name': 'Collaborative Filtering Recommendations',
        'description': 'Generate personalized recommendations based on user behavior',
        'category': 'ml',
        'pricing_tier': 'starter',
        'parameters': {
            'user_id': 'string',
            'item_ratings': 'dict (item_id: rating)',
            'catalog': 'list of item_ids',
            'n_recommendations': 'integer'
        },
        'returns': {
            'recommendations': 'list of items with predicted ratings'
        },
        'use_cases': [
            'Product recommendations',
            'Content suggestions',
            'Personalized feeds'
        ]
    },
    
    'sentiment-analysis': {
        'name': 'Sentiment Analysis',
        'description': 'Analyze text sentiment (positive, negative, neutral)',
        'category': 'nlp',
        'pricing_tier': 'starter',
        'parameters': {
            'text': 'string'
        },
        'returns': {
            'sentiment': 'string (positive/negative/neutral)',
            'score': 'float (0-1)',
            'confidence': 'float (0-1)'
        },
        'use_cases': [
            'Customer review analysis',
            'Social media monitoring',
            'Support ticket categorization'
        ]
    },
    
    'churn-prediction': {
        'name': 'Customer Churn Prediction',
        'description': 'Predict likelihood of customer churn',
        'category': 'analytics',
        'pricing_tier': 'pro',
        'parameters': {
            'days_since_last_activity': 'integer',
            'total_purchases': 'integer',
            'avg_purchase_value': 'float',
            'support_tickets': 'integer',
            'account_age_months': 'integer'
        },
        'returns': {
            'will_churn': 'boolean',
            'churn_probability': 'float (0-1)',
            'risk_level': 'string (high/medium/low)',
            'recommended_action': 'string'
        },
        'use_cases': [
            'SaaS retention campaigns',
            'Subscription service optimization',
            'Customer success prioritization'
        ]
    },
    
    'lead-scoring': {
        'name': 'Lead Scoring',
        'description': 'Score and qualify sales leads based on engagement and demographics',
        'category': 'sales',
        'pricing_tier': 'starter',
        'parameters': {
            'email_opens': 'integer',
            'page_views': 'integer',
            'company_size': 'string (small/medium/large)',
            'job_title': 'string',
            'industry': 'string'
        },
        'returns': {
            'lead_score': 'integer (0-100)',
            'quality': 'string (hot/warm/cold)',
            'recommended_action': 'string'
        },
        'use_cases': [
            'Sales pipeline prioritization',
            'Marketing automation',
            'Lead nurturing workflows'
        ]
    },
    
    'inventory-optimization': {
        'name': 'Inventory Optimization',
        'description': 'Calculate optimal inventory levels and reorder points',
        'category': 'logistics',
        'pricing_tier': 'pro',
        'parameters': {
            'current_stock': 'integer',
            'daily_sales_avg': 'float',
            'lead_time_days': 'integer',
            'safety_stock_days': 'integer'
        },
        'returns': {
            'reorder_point': 'float',
            'recommended_order_quantity': 'float',
            'needs_reorder': 'boolean',
            'days_until_stockout': 'float',
            'urgency': 'string'
        },
        'use_cases': [
            'E-commerce inventory management',
            'Retail stock optimization',
            'Warehouse management'
        ]
    },
    
    'route-optimization': {
        'name': 'Route Optimization',
        'description': 'Optimize delivery routes using TSP algorithm',
        'category': 'logistics',
        'pricing_tier': 'pro',
        'parameters': {
            'locations': 'list of {lat, lon, address}',
            'start_location': 'object {lat, lon, address}'
        },
        'returns': {
            'optimized_route': 'list of locations in order',
            'total_distance_km': 'float',
            'estimated_time_hours': 'float'
        },
        'use_cases': [
            'Delivery route planning',
            'Field service scheduling',
            'Logistics optimization'
        ]
    },
    
    'credit-scoring': {
        'name': 'Credit Scoring',
        'description': 'Calculate credit scores based on financial factors',
        'category': 'fintech',
        'pricing_tier': 'pro',
        'parameters': {
            'income': 'float',
            'debt': 'float',
            'payment_history_score': 'integer (0-100)',
            'credit_age_years': 'integer',
            'num_accounts': 'integer'
        },
        'returns': {
            'credit_score': 'integer (300-850)',
            'rating': 'string (excellent/good/fair/poor)',
            'approval_recommendation': 'boolean',
            'interest_rate_tier': 'string'
        },
        'use_cases': [
            'Loan approval automation',
            'Risk assessment',
            'Credit line determination'
        ]
    },
    
    'demand-forecasting': {
        'name': 'Demand Forecasting',
        'description': 'Forecast future demand using time series analysis',
        'category': 'analytics',
        'pricing_tier': 'starter',
        'parameters': {
            'historical_sales': 'list of numbers',
            'forecast_periods': 'integer'
        },
        'returns': {
            'forecast': 'list of predicted values',
            'baseline': 'float',
            'trend': 'string (increasing/decreasing/stable)',
            'confidence': 'string (high/medium/low)'
        },
        'use_cases': [
            'Inventory planning',
            'Revenue forecasting',
            'Capacity planning'
        ]
    }
}

# Categories for filtering
ALGORITHM_CATEGORIES = {
    'security': ['fraud-detection'],
    'ecommerce': ['dynamic-pricing', 'inventory-optimization'],
    'ml': ['recommendation-collab'],
    'nlp': ['sentiment-analysis'],
    'analytics': ['churn-prediction', 'demand-forecasting'],
    'sales': ['lead-scoring'],
    'logistics': ['route-optimization'],
    'fintech': ['credit-scoring']
}
