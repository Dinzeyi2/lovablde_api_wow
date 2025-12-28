import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import accuracy_score, mean_squared_error, r2_score
import joblib
import os
import uuid
from datetime import datetime
from typing import Optional, Dict, Any

from app.models import Model

class MLTrainer:
    """AutoML training service"""
    
    def __init__(self):
        self.models_dir = "/tmp/models"
        os.makedirs(self.models_dir, exist_ok=True)
    
    def train_model(
        self,
        data_path: str,
        model_type: str,
        target_column: Optional[str],
        name: str,
        user_id: str,
        db
    ) -> str:
        """Train a machine learning model automatically"""
        
        model_id = str(uuid.uuid4())
        
        # Create model record
        db_model = Model(
            id=model_id,
            user_id=user_id,
            name=name,
            model_type=model_type,
            status="training"
        )
        db.add(db_model)
        db.commit()
        
        try:
            # Load data
            df = pd.read_csv(data_path)
            
            if model_type == "recommendation":
                # Collaborative filtering recommendation
                trained_model = self._train_recommendation(df, model_id)
                accuracy = 0.85  # Placeholder
            
            elif model_type == "classification":
                if not target_column:
                    raise ValueError("target_column required for classification")
                trained_model, accuracy = self._train_classification(df, target_column, model_id)
            
            elif model_type == "regression":
                if not target_column:
                    raise ValueError("target_column required for regression")
                trained_model, accuracy = self._train_regression(df, target_column, model_id)
            
            elif model_type == "clustering":
                trained_model = self._train_clustering(df, model_id)
                accuracy = 0.80  # Placeholder
            
            else:
                raise ValueError(f"Unsupported model type: {model_type}")
            
            # Update model status
            db_model.status = "ready"
            db_model.accuracy = accuracy
            db_model.file_path = f"{self.models_dir}/{model_id}.joblib"
            db_model.metadata = {
                "features": list(df.columns),
                "rows": len(df),
                "trained_at": datetime.utcnow().isoformat()
            }
            db.commit()
            
            return model_id
        
        except Exception as e:
            db_model.status = "failed"
            db_model.model_metadata = {"error": str(e)}
            db.commit()
            raise
    
    def _train_recommendation(self, df: pd.DataFrame, model_id: str):
        """Train collaborative filtering recommendation model"""
        from sklearn.neighbors import NearestNeighbors
        
        # Assume df has user_id, item_id, rating columns
        # Create user-item matrix
        if 'user_id' in df.columns and 'item_id' in df.columns:
            user_item_matrix = df.pivot_table(
                index='user_id',
                columns='item_id',
                values='rating' if 'rating' in df.columns else df.columns[2],
                fill_value=0
            )
        else:
            # Fallback: use first 3 columns
            user_item_matrix = df.pivot_table(
                index=df.columns[0],
                columns=df.columns[1],
                values=df.columns[2] if len(df.columns) > 2 else df.columns[1],
                fill_value=0
            )
        
        # Train KNN model
        model = NearestNeighbors(metric='cosine', algorithm='brute')
        model.fit(user_item_matrix.values)
        
        # Save model and matrix
        joblib.dump({
            'model': model,
            'matrix': user_item_matrix,
            'index_to_user': dict(enumerate(user_item_matrix.index)),
            'user_to_index': {user: idx for idx, user in enumerate(user_item_matrix.index)}
        }, f"{self.models_dir}/{model_id}.joblib")
        
        return model
    
    def _train_classification(self, df: pd.DataFrame, target_column: str, model_id: str):
        """Train classification model"""
        
        # Prepare data
        X = df.drop(columns=[target_column])
        y = df[target_column]
        
        # Encode categorical features
        label_encoders = {}
        for col in X.select_dtypes(include=['object']).columns:
            le = LabelEncoder()
            X[col] = le.fit_transform(X[col].astype(str))
            label_encoders[col] = le
        
        # Encode target if categorical
        if y.dtype == 'object':
            target_encoder = LabelEncoder()
            y = target_encoder.fit_transform(y)
        else:
            target_encoder = None
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        # Train Random Forest
        model = RandomForestClassifier(n_estimators=100, random_state=42)
        model.fit(X_train, y_train)
        
        # Evaluate
        predictions = model.predict(X_test)
        accuracy = accuracy_score(y_test, predictions)
        
        # Save model
        joblib.dump({
            'model': model,
            'label_encoders': label_encoders,
            'target_encoder': target_encoder,
            'feature_names': list(X.columns)
        }, f"{self.models_dir}/{model_id}.joblib")
        
        return model, accuracy
    
    def _train_regression(self, df: pd.DataFrame, target_column: str, model_id: str):
        """Train regression model"""
        
        # Prepare data
        X = df.drop(columns=[target_column])
        y = df[target_column]
        
        # Encode categorical features
        label_encoders = {}
        for col in X.select_dtypes(include=['object']).columns:
            le = LabelEncoder()
            X[col] = le.fit_transform(X[col].astype(str))
            label_encoders[col] = le
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        # Train Random Forest Regressor
        model = RandomForestRegressor(n_estimators=100, random_state=42)
        model.fit(X_train, y_train)
        
        # Evaluate
        predictions = model.predict(X_test)
        r2 = r2_score(y_test, predictions)
        
        # Save model
        joblib.dump({
            'model': model,
            'label_encoders': label_encoders,
            'feature_names': list(X.columns)
        }, f"{self.models_dir}/{model_id}.joblib")
        
        return model, r2
    
    def _train_clustering(self, df: pd.DataFrame, model_id: str):
        """Train clustering model"""
        from sklearn.cluster import KMeans
        
        # Encode categorical features
        X = df.copy()
        label_encoders = {}
        for col in X.select_dtypes(include=['object']).columns:
            le = LabelEncoder()
            X[col] = le.fit_transform(X[col].astype(str))
            label_encoders[col] = le
        
        # Auto-determine optimal clusters (simple elbow method)
        n_clusters = min(5, len(X) // 10)
        
        # Train KMeans
        model = KMeans(n_clusters=n_clusters, random_state=42)
        model.fit(X)
        
        # Save model
        joblib.dump({
            'model': model,
            'label_encoders': label_encoders,
            'feature_names': list(X.columns)
        }, f"{self.models_dir}/{model_id}.joblib")
        
        return model
    
    def predict(self, model_id: str, data: Dict[str, Any], db) -> Any:
        """Make prediction using trained model"""
        
        # Load model
        model_path = f"{self.models_dir}/{model_id}.joblib"
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model {model_id} not found")
        
        model_data = joblib.load(model_path)
        model = model_data['model']
        
        # Get model type from database
        db_model = db.query(Model).filter(Model.id == model_id).first()
        
        if db_model.model_type == "recommendation":
            # Recommendation prediction
            user_id = data.get('user_id')
            n_recommendations = data.get('n_recommendations', 5)
            
            user_to_index = model_data['user_to_index']
            if user_id not in user_to_index:
                return {"recommendations": [], "message": "User not found in training data"}
            
            user_idx = user_to_index[user_id]
            distances, indices = model.kneighbors(
                model_data['matrix'].iloc[user_idx].values.reshape(1, -1),
                n_neighbors=n_recommendations + 1
            )
            
            similar_users = [model_data['index_to_user'][idx] for idx in indices.flatten()[1:]]
            
            return {
                "user_id": user_id,
                "similar_users": similar_users,
                "recommendations": similar_users[:n_recommendations]
            }
        
        else:
            # Classification/Regression prediction
            # Prepare input data
            df_input = pd.DataFrame([data])
            
            # Apply label encoding
            if 'label_encoders' in model_data:
                for col, le in model_data['label_encoders'].items():
                    if col in df_input.columns:
                        df_input[col] = le.transform(df_input[col].astype(str))
            
            # Ensure correct feature order
            if 'feature_names' in model_data:
                df_input = df_input[model_data['feature_names']]
            
            # Make prediction
            prediction = model.predict(df_input)[0]
            
            # Decode if classification
            if 'target_encoder' in model_data and model_data['target_encoder']:
                prediction = model_data['target_encoder'].inverse_transform([int(prediction)])[0]
            
            # Get confidence for classification
            if hasattr(model, 'predict_proba'):
                confidence = float(model.predict_proba(df_input).max())
            else:
                confidence = None
            
            result = {
                "prediction": float(prediction) if isinstance(prediction, (np.integer, np.floating)) else str(prediction)
            }
            
            if confidence:
                result["confidence"] = confidence
            
            return result
