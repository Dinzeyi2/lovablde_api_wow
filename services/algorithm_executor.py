from typing import Dict, Any
import numpy as np
from datetime import datetime
import hashlib

class AlgorithmExecutor:
    """Execute pre-built complex algorithms"""
    
    def __init__(self):
        self.algorithms = {
            'fraud-detection': self.fraud_detection,
            'dynamic-pricing': self.dynamic_pricing,
            'recommendation-collab': self.recommendation_collaborative,
            'sentiment-analysis': self.sentiment_analysis,
            'churn-prediction': self.churn_prediction,
            'lead-scoring': self.lead_scoring,
            'inventory-optimization': self.inventory_optimization,
            'route-optimization': self.route_optimization,
            'credit-scoring': self.credit_scoring,
            'demand-forecasting': self.demand_forecasting
        }
    
    def execute(self, algorithm_name: str, params: Dict[str, Any]) -> Dict[str, Any]:
        """Execute algorithm by name"""
        if algorithm_name not in self.algorithms:
            raise KeyError(f"Algorithm '{algorithm_name}' not found")
        
        return self.algorithms[algorithm_name](params)
    
    def fraud_detection(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """
        Detect fraudulent transactions using multiple signals
        """
        transaction_amount = params.get('transaction_amount', 0)
        user_location = params.get('user_location', '')
        device_fingerprint = params.get('device_fingerprint', '')
        account_age_days = params.get('account_age_days', 0)
        transaction_time = params.get('transaction_time', datetime.utcnow().hour)
        
        risk_score = 0.0
        risk_factors = []
        
        # High-risk location check
        high_risk_countries = ['NG', 'PK', 'ID', 'VN']  # Example
        if user_location.upper() in high_risk_countries:
            risk_score += 0.25
            risk_factors.append('high_risk_location')
        
        # New account check
        if account_age_days < 7:
            risk_score += 0.20
            risk_factors.append('new_account')
        
        # Unusual transaction amount
        if transaction_amount > 5000:
            risk_score += 0.15
            risk_factors.append('high_amount')
        
        # Odd hours transaction (2 AM - 5 AM)
        if 2 <= transaction_time <= 5:
            risk_score += 0.10
            risk_factors.append('unusual_time')
        
        # Device fingerprint check (simplified)
        if len(device_fingerprint) < 10:
            risk_score += 0.15
            risk_factors.append('suspicious_device')
        
        # Velocity check (simplified - would need history in production)
        if transaction_amount > 1000 and account_age_days < 30:
            risk_score += 0.15
            risk_factors.append('velocity_anomaly')
        
        is_fraud = risk_score > 0.5
        
        return {
            'is_fraud': is_fraud,
            'risk_score': min(risk_score, 1.0),
            'risk_factors': risk_factors,
            'recommendation': 'block' if risk_score > 0.7 else 'review' if risk_score > 0.5 else 'approve'
        }
    
    def dynamic_pricing(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """
        Calculate optimal price based on multiple factors
        """
        base_price = params.get('base_price', 100)
        competitor_prices = params.get('competitor_prices', [])
        inventory_level = params.get('inventory_level', 50)  # percentage
        demand_score = params.get('demand_score', 0.5)  # 0-1
        cost = params.get('cost', base_price * 0.6)
        
        # Start with base price
        optimal_price = base_price
        
        # Competitor pricing adjustment
        if competitor_prices:
            avg_competitor_price = np.mean(competitor_prices)
            min_competitor_price = np.min(competitor_prices)
            
            # Price within 5% of average competitor
            competitive_price = avg_competitor_price * 0.98
            optimal_price = (optimal_price + competitive_price) / 2
        
        # Inventory adjustment
        if inventory_level > 80:
            # High inventory - reduce price to move stock
            optimal_price *= 0.92
        elif inventory_level < 20:
            # Low inventory - increase price
            optimal_price *= 1.08
        
        # Demand adjustment
        demand_multiplier = 1 + (demand_score - 0.5) * 0.3
        optimal_price *= demand_multiplier
        
        # Ensure minimum margin (20%)
        min_price = cost * 1.20
        optimal_price = max(optimal_price, min_price)
        
        # Calculate metrics
        profit_margin = ((optimal_price - cost) / optimal_price) * 100
        
        return {
            'recommended_price': round(optimal_price, 2),
            'base_price': base_price,
            'profit_margin_percent': round(profit_margin, 2),
            'price_change_percent': round(((optimal_price - base_price) / base_price) * 100, 2),
            'reasoning': self._generate_pricing_reasoning(
                optimal_price, base_price, inventory_level, demand_score
            )
        }
    
    def _generate_pricing_reasoning(self, optimal, base, inventory, demand):
        reasons = []
        if inventory > 80:
            reasons.append("High inventory requires clearance")
        if demand > 0.7:
            reasons.append("High demand allows premium pricing")
        if optimal > base:
            reasons.append("Market conditions support price increase")
        elif optimal < base:
            reasons.append("Competitive pressure requires discount")
        return "; ".join(reasons) if reasons else "Optimal price based on market analysis"
    
    def recommendation_collaborative(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """
        Collaborative filtering recommendations (simplified)
        """
        user_id = params.get('user_id')
        item_ratings = params.get('item_ratings', {})  # {item_id: rating}
        catalog = params.get('catalog', [])  # list of all item_ids
        n_recommendations = params.get('n_recommendations', 5)
        
        # Simplified recommendation based on ratings
        # In production, this would use trained model
        rated_items = set(item_ratings.keys())
        unrated_items = [item for item in catalog if item not in rated_items]
        
        # Simple popularity-based + rating-based recommendation
        avg_rating = np.mean(list(item_ratings.values())) if item_ratings else 3.0
        
        recommendations = unrated_items[:n_recommendations]
        
        return {
            'user_id': user_id,
            'recommendations': [
                {
                    'item_id': item,
                    'predicted_rating': round(avg_rating + np.random.uniform(-0.5, 0.5), 2),
                    'confidence': round(np.random.uniform(0.7, 0.95), 2)
                }
                for item in recommendations
            ]
        }
    
    def sentiment_analysis(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """
        Simple sentiment analysis (would use NLP model in production)
        """
        text = params.get('text', '')
        
        # Simple keyword-based sentiment (placeholder)
        positive_words = ['good', 'great', 'excellent', 'amazing', 'love', 'best', 'awesome', 'fantastic']
        negative_words = ['bad', 'terrible', 'awful', 'hate', 'worst', 'poor', 'disappointing']
        
        text_lower = text.lower()
        positive_count = sum(1 for word in positive_words if word in text_lower)
        negative_count = sum(1 for word in negative_words if word in text_lower)
        
        total = positive_count + negative_count
        if total == 0:
            sentiment = 'neutral'
            score = 0.5
        elif positive_count > negative_count:
            sentiment = 'positive'
            score = 0.5 + (positive_count / (total * 2))
        else:
            sentiment = 'negative'
            score = 0.5 - (negative_count / (total * 2))
        
        return {
            'sentiment': sentiment,
            'score': round(score, 2),
            'confidence': round(min(total / 5, 1.0), 2)
        }
    
    def churn_prediction(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """
        Predict customer churn probability
        """
        days_since_last_activity = params.get('days_since_last_activity', 0)
        total_purchases = params.get('total_purchases', 0)
        avg_purchase_value = params.get('avg_purchase_value', 0)
        support_tickets = params.get('support_tickets', 0)
        account_age_months = params.get('account_age_months', 1)
        
        churn_score = 0.0
        
        # Inactivity
        if days_since_last_activity > 90:
            churn_score += 0.30
        elif days_since_last_activity > 60:
            churn_score += 0.20
        elif days_since_last_activity > 30:
            churn_score += 0.10
        
        # Low engagement
        if total_purchases < 5:
            churn_score += 0.15
        
        # Low value
        if avg_purchase_value < 50:
            churn_score += 0.10
        
        # Support issues
        if support_tickets > 5:
            churn_score += 0.20
        
        # New customer risk
        if account_age_months < 3:
            churn_score += 0.15
        
        will_churn = churn_score > 0.5
        
        return {
            'will_churn': will_churn,
            'churn_probability': min(churn_score, 1.0),
            'risk_level': 'high' if churn_score > 0.7 else 'medium' if churn_score > 0.4 else 'low',
            'recommended_action': self._get_churn_action(churn_score)
        }
    
    def _get_churn_action(self, score):
        if score > 0.7:
            return "Immediate intervention: Send personalized offer"
        elif score > 0.5:
            return "Re-engagement campaign: Email with discount"
        elif score > 0.3:
            return "Monitor: Check-in email"
        else:
            return "No action needed"
    
    def lead_scoring(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """
        Score leads based on engagement and demographics
        """
        email_opens = params.get('email_opens', 0)
        page_views = params.get('page_views', 0)
        company_size = params.get('company_size', 'small')  # small, medium, large
        job_title = params.get('job_title', '').lower()
        industry = params.get('industry', '')
        
        score = 0
        
        # Engagement scoring
        score += min(email_opens * 5, 30)
        score += min(page_views * 2, 20)
        
        # Company size
        size_scores = {'small': 10, 'medium': 20, 'large': 30}
        score += size_scores.get(company_size, 10)
        
        # Job title
        if any(title in job_title for title in ['ceo', 'cto', 'vp', 'director']):
            score += 20
        elif any(title in job_title for title in ['manager', 'lead']):
            score += 10
        
        # Industry (example high-value industries)
        high_value_industries = ['technology', 'finance', 'healthcare']
        if any(ind in industry.lower() for ind in high_value_industries):
            score += 15
        
        score = min(score, 100)
        
        quality = 'hot' if score >= 70 else 'warm' if score >= 40 else 'cold'
        
        return {
            'lead_score': score,
            'quality': quality,
            'recommended_action': 'Contact immediately' if quality == 'hot' else 'Nurture campaign' if quality == 'warm' else 'Low priority'
        }
    
    def inventory_optimization(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """
        Calculate optimal inventory levels
        """
        current_stock = params.get('current_stock', 0)
        daily_sales_avg = params.get('daily_sales_avg', 0)
        lead_time_days = params.get('lead_time_days', 7)
        safety_stock_days = params.get('safety_stock_days', 3)
        
        # Calculate reorder point
        reorder_point = (daily_sales_avg * lead_time_days) + (daily_sales_avg * safety_stock_days)
        
        # Calculate economic order quantity (simplified)
        eoq = daily_sales_avg * lead_time_days * 2
        
        # Days until stockout
        if daily_sales_avg > 0:
            days_until_stockout = current_stock / daily_sales_avg
        else:
            days_until_stockout = 999
        
        needs_reorder = current_stock <= reorder_point
        
        return {
            'current_stock': current_stock,
            'reorder_point': round(reorder_point, 0),
            'recommended_order_quantity': round(eoq, 0),
            'needs_reorder': needs_reorder,
            'days_until_stockout': round(days_until_stockout, 1),
            'urgency': 'critical' if days_until_stockout < 3 else 'high' if days_until_stockout < 7 else 'normal'
        }
    
    def route_optimization(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """
        Simple route optimization (TSP approximation)
        """
        locations = params.get('locations', [])  # [{lat, lon, address}]
        start_location = params.get('start_location', locations[0] if locations else {})
        
        # Simplified nearest neighbor algorithm
        if not locations:
            return {'route': [], 'total_distance': 0}
        
        route = [start_location]
        remaining = locations[1:]
        current = start_location
        total_distance = 0
        
        while remaining:
            # Find nearest unvisited location
            nearest = min(remaining, key=lambda loc: self._haversine_distance(
                current.get('lat', 0), current.get('lon', 0),
                loc.get('lat', 0), loc.get('lon', 0)
            ))
            
            distance = self._haversine_distance(
                current.get('lat', 0), current.get('lon', 0),
                nearest.get('lat', 0), nearest.get('lon', 0)
            )
            
            total_distance += distance
            route.append(nearest)
            remaining.remove(nearest)
            current = nearest
        
        return {
            'optimized_route': route,
            'total_distance_km': round(total_distance, 2),
            'estimated_time_hours': round(total_distance / 50, 2)  # Assume 50 km/h avg
        }
    
    def _haversine_distance(self, lat1, lon1, lat2, lon2):
        """Calculate distance between two coordinates in km"""
        R = 6371  # Earth's radius in km
        
        lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
        dlat = lat2 - lat1
        dlon = lon2 - lon1
        
        a = np.sin(dlat/2)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon/2)**2
        c = 2 * np.arcsin(np.sqrt(a))
        
        return R * c
    
    def credit_scoring(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """
        Calculate credit score based on multiple factors
        """
        income = params.get('income', 0)
        debt = params.get('debt', 0)
        payment_history_score = params.get('payment_history_score', 50)  # 0-100
        credit_age_years = params.get('credit_age_years', 0)
        num_accounts = params.get('num_accounts', 0)
        
        score = 300  # Base score
        
        # Income contribution (max 150 points)
        score += min(income / 1000, 150)
        
        # Debt-to-income ratio (max 200 points)
        if income > 0:
            dti = debt / income
            if dti < 0.3:
                score += 200
            elif dti < 0.5:
                score += 150
            else:
                score += 50
        
        # Payment history (max 250 points)
        score += payment_history_score * 2.5
        
        # Credit age (max 100 points)
        score += min(credit_age_years * 10, 100)
        
        # Account diversity (max 50 points)
        score += min(num_accounts * 5, 50)
        
        score = min(score, 850)  # Cap at 850
        
        rating = 'excellent' if score >= 750 else 'good' if score >= 650 else 'fair' if score >= 550 else 'poor'
        
        return {
            'credit_score': round(score),
            'rating': rating,
            'approval_recommendation': score >= 650,
            'interest_rate_tier': 'prime' if score >= 700 else 'subprime'
        }
    
    def demand_forecasting(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """
        Forecast future demand using simple time series analysis
        """
        historical_sales = params.get('historical_sales', [])  # list of numbers
        forecast_periods = params.get('forecast_periods', 7)
        
        if len(historical_sales) < 3:
            return {'forecast': [], 'error': 'Need at least 3 historical data points'}
        
        # Simple moving average forecast
        window_size = min(7, len(historical_sales))
        moving_avg = np.mean(historical_sales[-window_size:])
        
        # Calculate trend
        if len(historical_sales) >= window_size:
            recent_avg = np.mean(historical_sales[-window_size:])
            older_avg = np.mean(historical_sales[-window_size*2:-window_size] if len(historical_sales) >= window_size*2 else historical_sales[:window_size])
            trend = (recent_avg - older_avg) / window_size
        else:
            trend = 0
        
        # Generate forecast
        forecast = []
        for i in range(forecast_periods):
            predicted = moving_avg + (trend * i)
            forecast.append(max(0, round(predicted, 2)))  # Can't have negative demand
        
        return {
            'forecast': forecast,
            'baseline': round(moving_avg, 2),
            'trend': 'increasing' if trend > 0 else 'decreasing' if trend < 0 else 'stable',
            'confidence': 'high' if len(historical_sales) > 30 else 'medium' if len(historical_sales) > 10 else 'low'
        }
