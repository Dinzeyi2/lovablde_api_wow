/**
 * Example: Dynamic Pricing for E-commerce Product in Lovable.dev
 * 
 * Automatically adjust product prices based on:
 * - Competitor prices
 * - Inventory levels
 * - Demand signals
 */

import React, { useState, useEffect } from 'react';
import { useAlgoAPI } from '@algoapi/client';

function ProductCard({ product }) {
  const { calculatePrice } = useAlgoAPI(process.env.REACT_APP_ALGOAPI_KEY);
  const [optimalPrice, setOptimalPrice] = useState(product.basePrice);
  const [priceAnalysis, setPriceAnalysis] = useState(null);
  const [loading, setLoading] = useState(false);

  useEffect(() => {
    calculateOptimalPrice();
    // Recalculate every hour
    const interval = setInterval(calculateOptimalPrice, 60 * 60 * 1000);
    return () => clearInterval(interval);
  }, [product.id]);

  const calculateOptimalPrice = async () => {
    setLoading(true);

    try {
      // Fetch competitor prices (from your scraper or API)
      const competitorPrices = await fetchCompetitorPrices(product.id);
      
      // Get current inventory level
      const inventory = await getInventoryLevel(product.id);
      
      // Calculate demand score (based on views, add-to-carts, etc.)
      const demandScore = calculateDemandScore(product);

      // Call AlgoAPI for optimal pricing
      const result = await calculatePrice({
        base_price: product.basePrice,
        competitor_prices: competitorPrices,
        inventory_level: inventory.percentageFull,
        demand_score: demandScore,
        cost: product.cost
      });

      setOptimalPrice(result.result.recommended_price);
      setPriceAnalysis(result.result);
    } catch (error) {
      console.error('Pricing calculation failed:', error);
      setOptimalPrice(product.basePrice); // Fallback to base price
    }

    setLoading(false);
  };

  const priceChange = ((optimalPrice - product.basePrice) / product.basePrice) * 100;
  const isDiscounted = optimalPrice < product.basePrice;

  return (
    <div className="product-card">
      <img src={product.image} alt={product.name} />
      <h3>{product.name}</h3>
      
      <div className="price-container">
        {isDiscounted && (
          <span className="original-price">${product.basePrice.toFixed(2)}</span>
        )}
        <span className="current-price">
          ${optimalPrice.toFixed(2)}
          {loading && <span className="loading-spinner">⟳</span>}
        </span>
        {isDiscounted && (
          <span className="discount-badge">
            Save {Math.abs(priceChange).toFixed(0)}%
          </span>
        )}
      </div>

      {priceAnalysis && (
        <div className="price-details">
          <small>
            {priceAnalysis.reasoning}
          </small>
          <small>
            Profit Margin: {priceAnalysis.profit_margin_percent.toFixed(1)}%
          </small>
        </div>
      )}

      <button className="add-to-cart">Add to Cart</button>
    </div>
  );
}

// Helper functions
async function fetchCompetitorPrices(productId) {
  // Call your competitor price scraping service
  // Return array of competitor prices
  return [99.99, 105.00, 95.50];
}

async function getInventoryLevel(productId) {
  // Fetch from your inventory system
  return {
    current: 85,
    total: 100,
    percentageFull: 85
  };
}

function calculateDemandScore(product) {
  // Calculate based on recent activity
  // Simple formula: (views + add_to_carts * 5) / 100
  const score = (product.recentViews + product.recentAddToCarts * 5) / 100;
  return Math.min(Math.max(score, 0), 1); // Clamp to 0-1
}

export default ProductCard;
