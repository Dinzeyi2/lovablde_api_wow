/**
 * AlgoAPI SDK for Lovable.dev and other vibe coding platforms
 * 
 * Installation:
 * npm install @algoapi/client
 * 
 * Usage:
 * import { AlgoAPI } from '@algoapi/client';
 * const api = new AlgoAPI('your-api-key');
 */

export interface TrainModelOptions {
  modelType: 'recommendation' | 'classification' | 'regression' | 'clustering';
  name: string;
  targetColumn?: string;
  features?: string[];
}

export interface PredictOptions {
  data: Record<string, any>;
}

export interface AlgorithmParams {
  [key: string]: any;
}

export interface Model {
  id: string;
  name: string;
  type: string;
  status: 'training' | 'ready' | 'failed';
  created_at: string;
  prediction_count: number;
  accuracy?: number;
}

export class AlgoAPI {
  private apiKey: string;
  private baseUrl: string;

  constructor(apiKey: string, baseUrl: string = 'https://algoapi.railway.app') {
    this.apiKey = apiKey;
    this.baseUrl = baseUrl;
  }

  /**
   * Train a new ML model from CSV data
   */
  async trainModel(file: File, options: TrainModelOptions): Promise<{
    status: string;
    model_id: string;
    model_type: string;
    endpoint: string;
    estimated_time: string;
  }> {
    const formData = new FormData();
    formData.append('file', file);
    formData.append('model_type', options.modelType);
    formData.append('name', options.name);
    if (options.targetColumn) formData.append('target_column', options.targetColumn);

    const response = await fetch(`${this.baseUrl}/api/v1/train`, {
      method: 'POST',
      headers: {
        'X-API-Key': this.apiKey,
      },
      body: formData,
    });

    if (!response.ok) {
      throw new Error(`Training failed: ${response.statusText}`);
    }

    return response.json();
  }

  /**
   * Make predictions using a trained model
   */
  async predict(modelId: string, data: Record<string, any>): Promise<{
    model_id: string;
    prediction: any;
    timestamp: string;
    confidence?: number;
  }> {
    const response = await fetch(`${this.baseUrl}/api/v1/predict/${modelId}`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
        'X-API-Key': this.apiKey,
      },
      body: JSON.stringify({ data }),
    });

    if (!response.ok) {
      throw new Error(`Prediction failed: ${response.statusText}`);
    }

    return response.json();
  }

  /**
   * Execute a pre-built algorithm
   */
  async executeAlgorithm(algorithmName: string, params: AlgorithmParams): Promise<{
    algorithm: string;
    result: any;
    timestamp: string;
  }> {
    const response = await fetch(`${this.baseUrl}/api/v1/algorithm/execute`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
        'X-API-Key': this.apiKey,
      },
      body: JSON.stringify({
        algorithm_name: algorithmName,
        params,
      }),
    });

    if (!response.ok) {
      throw new Error(`Algorithm execution failed: ${response.statusText}`);
    }

    return response.json();
  }

  /**
   * List all available pre-built algorithms
   */
  async listAlgorithms(): Promise<{
    algorithms: Array<{
      name: string;
      description: string;
      category: string;
      pricing_tier: string;
    }>;
  }> {
    const response = await fetch(`${this.baseUrl}/api/v1/algorithms`);
    
    if (!response.ok) {
      throw new Error(`Failed to list algorithms: ${response.statusText}`);
    }

    return response.json();
  }

  /**
   * Get details about a specific algorithm
   */
  async getAlgorithm(algorithmName: string): Promise<any> {
    const response = await fetch(`${this.baseUrl}/api/v1/algorithms/${algorithmName}`);
    
    if (!response.ok) {
      throw new Error(`Failed to get algorithm: ${response.statusText}`);
    }

    return response.json();
  }

  /**
   * List all models for authenticated user
   */
  async listModels(): Promise<{ models: Model[] }> {
    const response = await fetch(`${this.baseUrl}/api/v1/models`, {
      headers: {
        'X-API-Key': this.apiKey,
      },
    });

    if (!response.ok) {
      throw new Error(`Failed to list models: ${response.statusText}`);
    }

    return response.json();
  }

  /**
   * Get details about a specific model
   */
  async getModel(modelId: string): Promise<Model & { endpoint: string }> {
    const response = await fetch(`${this.baseUrl}/api/v1/models/${modelId}`, {
      headers: {
        'X-API-Key': this.apiKey,
      },
    });

    if (!response.ok) {
      throw new Error(`Failed to get model: ${response.statusText}`);
    }

    return response.json();
  }

  /**
   * Delete a model
   */
  async deleteModel(modelId: string): Promise<{ status: string; model_id: string }> {
    const response = await fetch(`${this.baseUrl}/api/v1/models/${modelId}`, {
      method: 'DELETE',
      headers: {
        'X-API-Key': this.apiKey,
      },
    });

    if (!response.ok) {
      throw new Error(`Failed to delete model: ${response.statusText}`);
    }

    return response.json();
  }

  /**
   * Get usage statistics
   */
  async getUsage(): Promise<{
    total_models: number;
    total_predictions: number;
    models_by_status: {
      training: number;
      ready: number;
      failed: number;
    };
  }> {
    const response = await fetch(`${this.baseUrl}/api/v1/usage`, {
      headers: {
        'X-API-Key': this.apiKey,
      },
    });

    if (!response.ok) {
      throw new Error(`Failed to get usage: ${response.statusText}`);
    }

    return response.json();
  }

  // ==================== Pre-built Algorithm Shortcuts ====================

  /**
   * Detect fraudulent transactions
   */
  async detectFraud(params: {
    transaction_amount: number;
    user_location: string;
    device_fingerprint: string;
    account_age_days: number;
    transaction_time?: number;
  }) {
    return this.executeAlgorithm('fraud-detection', params);
  }

  /**
   * Calculate optimal pricing
   */
  async calculatePrice(params: {
    base_price: number;
    competitor_prices?: number[];
    inventory_level?: number;
    demand_score?: number;
    cost?: number;
  }) {
    return this.executeAlgorithm('dynamic-pricing', params);
  }

  /**
   * Get personalized recommendations
   */
  async getRecommendations(params: {
    user_id: string;
    item_ratings: Record<string, number>;
    catalog: string[];
    n_recommendations?: number;
  }) {
    return this.executeAlgorithm('recommendation-collab', params);
  }

  /**
   * Analyze sentiment of text
   */
  async analyzeSentiment(text: string) {
    return this.executeAlgorithm('sentiment-analysis', { text });
  }

  /**
   * Predict customer churn
   */
  async predictChurn(params: {
    days_since_last_activity: number;
    total_purchases: number;
    avg_purchase_value: number;
    support_tickets: number;
    account_age_months: number;
  }) {
    return this.executeAlgorithm('churn-prediction', params);
  }

  /**
   * Score a sales lead
   */
  async scoreLead(params: {
    email_opens: number;
    page_views: number;
    company_size: 'small' | 'medium' | 'large';
    job_title: string;
    industry: string;
  }) {
    return this.executeAlgorithm('lead-scoring', params);
  }
}

// React Hook for easy integration in Lovable.dev
export function useAlgoAPI(apiKey: string) {
  const api = new AlgoAPI(apiKey);
  
  return {
    api,
    trainModel: api.trainModel.bind(api),
    predict: api.predict.bind(api),
    executeAlgorithm: api.executeAlgorithm.bind(api),
    detectFraud: api.detectFraud.bind(api),
    calculatePrice: api.calculatePrice.bind(api),
    getRecommendations: api.getRecommendations.bind(api),
    analyzeSentiment: api.analyzeSentiment.bind(api),
    predictChurn: api.predictChurn.bind(api),
    scoreLead: api.scoreLead.bind(api),
  };
}

export default AlgoAPI;
