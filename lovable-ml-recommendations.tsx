/**
 * Example: Train Custom Recommendation Model in Lovable.dev
 * 
 * This example shows how to:
 * 1. Upload user behavior data
 * 2. Train a recommendation model
 * 3. Get personalized recommendations
 */

import React, { useState, useEffect } from 'react';
import { AlgoAPI } from '@algoapi/client';

function MealRecommendationApp() {
  const [api] = useState(() => new AlgoAPI(process.env.REACT_APP_ALGOAPI_KEY));
  const [modelId, setModelId] = useState(localStorage.getItem('meal_model_id'));
  const [recommendations, setRecommendations] = useState([]);
  const [training, setTraining] = useState(false);

  useEffect(() => {
    if (modelId) {
      loadRecommendations();
    }
  }, [modelId]);

  // Step 1: Train model from user data
  const trainRecommendationModel = async () => {
    setTraining(true);

    try {
      // Get user meal rating data from your database
      const userRatings = await fetchUserMealRatings();
      
      // Convert to CSV
      const csv = convertToCSV(userRatings);
      const file = new File([csv], 'meal_ratings.csv', { type: 'text/csv' });

      // Train model via AlgoAPI
      const result = await api.trainModel(file, {
        modelType: 'recommendation',
        name: 'meal-recommender'
      });

      // Save model ID
      setModelId(result.model_id);
      localStorage.setItem('meal_model_id', result.model_id);

      alert('Model training started! It will be ready in 2-5 minutes.');
      
      // Poll for completion
      pollModelStatus(result.model_id);
    } catch (error) {
      console.error('Training failed:', error);
      alert('Failed to train model');
    }

    setTraining(false);
  };

  // Step 2: Poll until model is ready
  const pollModelStatus = async (id) => {
    const checkStatus = async () => {
      const model = await api.getModel(id);
      
      if (model.status === 'ready') {
        alert('Model is ready! Loading recommendations...');
        loadRecommendations();
      } else if (model.status === 'failed') {
        alert('Model training failed');
      } else {
        setTimeout(checkStatus, 10000); // Check every 10 seconds
      }
    };

    checkStatus();
  };

  // Step 3: Get recommendations
  const loadRecommendations = async () => {
    if (!modelId) return;

    try {
      const currentUser = getCurrentUser();
      
      const result = await api.predict(modelId, {
        user_id: currentUser.id,
        n_recommendations: 5
      });

      // Get meal details for recommended items
      const mealDetails = await fetchMealDetails(result.prediction.recommendations);
      setRecommendations(mealDetails);
    } catch (error) {
      console.error('Failed to load recommendations:', error);
    }
  };

  // Step 4: Rate a meal (adds to training data)
  const rateMeal = async (mealId, rating) => {
    const currentUser = getCurrentUser();
    
    // Save rating to your database
    await saveMealRating(currentUser.id, mealId, rating);

    // Retrain model with new data (optional - could do this periodically)
    // await trainRecommendationModel();
  };

  return (
    <div className="meal-app">
      <h1>Personalized Meal Recommendations</h1>

      {!modelId ? (
        <div className="setup-section">
          <p>Train your personalized meal recommendation model</p>
          <button 
            onClick={trainRecommendationModel} 
            disabled={training}
            className="btn-primary"
          >
            {training ? 'Training...' : 'Train My Model'}
          </button>
        </div>
      ) : (
        <div className="recommendations-section">
          <h2>Recommended for You</h2>
          
          {recommendations.length === 0 ? (
            <p>Loading recommendations...</p>
          ) : (
            <div className="meal-grid">
              {recommendations.map(meal => (
                <div key={meal.id} className="meal-card">
                  <img src={meal.image} alt={meal.name} />
                  <h3>{meal.name}</h3>
                  <p>{meal.description}</p>
                  <div className="nutrition">
                    <span>{meal.calories} cal</span>
                    <span>{meal.protein}g protein</span>
                  </div>
                  
                  {/* Rating system */}
                  <div className="rating">
                    {[1, 2, 3, 4, 5].map(star => (
                      <button
                        key={star}
                        onClick={() => rateMeal(meal.id, star)}
                        className="star-button"
                      >
                        ⭐
                      </button>
                    ))}
                  </div>
                </div>
              ))}
            </div>
          )}

          <button onClick={loadRecommendations} className="btn-secondary">
            Refresh Recommendations
          </button>
        </div>
      )}
    </div>
  );
}

// Helper functions
async function fetchUserMealRatings() {
  // Fetch from your database
  // Format: [{ user_id, meal_id, rating }]
  return [
    { user_id: 'user1', meal_id: 'meal1', rating: 5 },
    { user_id: 'user1', meal_id: 'meal2', rating: 4 },
    // ... more ratings
  ];
}

function convertToCSV(data) {
  const headers = Object.keys(data[0]).join(',');
  const rows = data.map(row => Object.values(row).join(','));
  return [headers, ...rows].join('\n');
}

function getCurrentUser() {
  // Get current user from your auth system
  return { id: 'user123', name: 'John Doe' };
}

async function fetchMealDetails(mealIds) {
  // Fetch meal details from your database
  return mealIds.map(id => ({
    id,
    name: `Meal ${id}`,
    description: 'Delicious and nutritious',
    image: '/placeholder.jpg',
    calories: 450,
    protein: 25
  }));
}

async function saveMealRating(userId, mealId, rating) {
  // Save to your database
  console.log(`User ${userId} rated meal ${mealId}: ${rating} stars`);
}

export default MealRecommendationApp;
