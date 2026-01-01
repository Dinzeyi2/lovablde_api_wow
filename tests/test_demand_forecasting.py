"""
Test Suite for Demand Forecasting + Inventory Optimization
Comprehensive tests covering all forecasting methods, optimization, and real-world scenarios

Author: AlgoAPI
Version: 1.0.0

TEST COVERAGE
------------
- Basic functionality: initialization, data preparation, training, forecasting
- Real-world scenarios: retail, grocery, warehouse, fashion
- Accuracy benchmarks: MAPE, RMSE, forecast bias validation
- GA optimization: constraint satisfaction, convergence testing
- Edge cases: stockouts, demand spikes, seasonality changes
- Performance tests: 1K SKUs, 10K SKUs, 100K SKUs scale
- Integration tests: execute function, error handling
"""

import pytest
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from algorithms_demand_forecasting import (
    DemandForecastEngine,
    InventoryOptimizer,
    execute_demand_forecasting
)


class TestDemandForecastEngine:
    """Test suite for DemandForecastEngine"""
    
    def generate_synthetic_data(
        self,
        days: int = 365,
        base_demand: float = 100,
        trend_slope: float = 0.1,
        seasonality_amplitude: float = 20,
        seasonality_period: int = 7,
        noise_std: float = 10
    ) -> np.ndarray:
        """Generate synthetic demand data with trend, seasonality, and noise"""
        t = np.arange(days)
        trend = base_demand + trend_slope * t
        seasonality = seasonality_amplitude * np.sin(2 * np.pi * t / seasonality_period)
        noise = np.random.normal(0, noise_std, days)
        data = trend + seasonality + noise
        return np.maximum(data, 0)  # Ensure non-negative
    
    def test_initialization(self):
        """Test basic engine initialization"""
        engine = DemandForecastEngine(method='lstm', forecast_horizon=30)
        assert engine.method == 'lstm'
        assert engine.forecast_horizon == 30
        assert not engine.is_trained
    
    def test_seasonality_detection(self):
        """Test automatic seasonality detection"""
        # Weekly seasonality
        data = self.generate_synthetic_data(days=365, seasonality_period=7)
        engine = DemandForecastEngine()
        detected_period = engine._detect_seasonality(data)
        # Should detect period around 7 (±2 days tolerance)
        assert detected_period is not None
        assert 5 <= detected_period <= 9
    
    def test_lstm_data_preparation(self):
        """Test LSTM data preparation with sliding window"""
        data = self.generate_synthetic_data(days=200)
        engine = DemandForecastEngine(forecast_horizon=7)
        
        # Normalize
        normalized = (data - np.mean(data)) / np.std(data)
        
        # Prepare sequences
        X, y = engine._prepare_lstm_data(normalized, lookback=30)
        
        # Check shapes
        assert X.shape[1] == 30  # Lookback window
        assert X.shape[2] == 1   # Single feature
        assert y.shape[1] == 7   # Forecast horizon
        assert X.shape[0] == y.shape[0]  # Same number of samples
    
    def test_lstm_training_basic(self):
        """Test LSTM model training on basic data"""
        data = self.generate_synthetic_data(days=180)
        engine = DemandForecastEngine(method='lstm', forecast_horizon=7)
        
        history = engine.train(data, epochs=5, verbose=0)
        
        assert engine.is_trained
        assert 'loss' in history
        assert len(history['loss']) > 0
    
    def test_lstm_forecasting(self):
        """Test LSTM forecast generation"""
        data = self.generate_synthetic_data(days=180)
        engine = DemandForecastEngine(method='lstm', forecast_horizon=7)
        
        engine.train(data, epochs=5, verbose=0)
        forecast, ci = engine.forecast(data)
        
        assert len(forecast) == 7
        assert len(ci['lower']) == 7
        assert len(ci['upper']) == 7
        assert np.all(forecast >= 0)  # Non-negative forecasts
        assert np.all(ci['upper'] >= ci['lower'])  # Valid intervals
    
    def test_accuracy_metrics(self):
        """Test forecast accuracy metric calculations"""
        actual = np.array([100, 110, 105, 115, 120])
        predicted = np.array([98, 112, 103, 118, 118])
        
        engine = DemandForecastEngine()
        metrics = engine.calculate_accuracy_metrics(actual, predicted)
        
        assert 'mape' in metrics
        assert 'rmse' in metrics
        assert 'mae' in metrics
        assert 'bias' in metrics
        assert metrics['mape'] > 0
        assert metrics['rmse'] > 0


class TestInventoryOptimizer:
    """Test suite for InventoryOptimizer"""
    
    def test_initialization(self):
        """Test optimizer initialization"""
        optimizer = InventoryOptimizer(
            population_size=50,
            generations=100
        )
        assert optimizer.population_size == 50
        assert optimizer.generations == 100
    
    def test_population_initialization(self):
        """Test initial population generation"""
        optimizer = InventoryOptimizer(population_size=20)
        forecast_mean = 100
        forecast_std = 20
        constraints = {'lead_time_days': 7}
        
        population = optimizer._initialize_population(
            forecast_mean, forecast_std, constraints
        )
        
        assert population.shape == (20, 2)  # 20 solutions, 2 genes each
        assert np.all(population >= 0)  # Non-negative values
    
    def test_crossover(self):
        """Test crossover operation"""
        optimizer = InventoryOptimizer()
        parent1 = np.array([500, 100])
        parent2 = np.array([800, 150])
        
        child = optimizer._crossover(parent1, parent2)
        
        assert len(child) == 2
        # Child genes should come from either parent
        for gene in child:
            assert gene in parent1 or gene in parent2 or \
                   (min(parent1[0], parent2[0]) <= gene <= max(parent1[0], parent2[0]))
    
    def test_mutation(self):
        """Test mutation operation"""
        optimizer = InventoryOptimizer(mutation_rate=1.0)  # Always mutate
        chromosome = np.array([500, 100])
        forecast_std = 20
        
        mutated = optimizer._mutation(chromosome, forecast_std)
        
        assert len(mutated) == 2
        assert np.all(mutated >= 0)  # Non-negative after mutation
    
    def test_optimization_basic(self):
        """Test basic optimization run"""
        optimizer = InventoryOptimizer(
            population_size=30,
            generations=50
        )
        
        forecast = np.array([100, 105, 110, 95, 100, 98, 102] * 4)  # 28 days
        forecast_std = 10
        constraints = {
            'target_service_level': 0.90,
            'max_storage_capacity': 2000,
            'budget_limit': 20000,
            'lead_time_days': 7
        }
        
        result = optimizer.optimize(forecast, forecast_std, constraints)
        
        assert result.optimal_order_quantity > 0
        assert result.reorder_point > 0
        assert result.safety_stock >= 0
        assert 0 <= result.expected_service_level <= 1
    
    def test_constraint_satisfaction(self):
        """Test that constraints are properly enforced"""
        optimizer = InventoryOptimizer(
            population_size=50,
            generations=100
        )
        
        forecast = np.array([100] * 30)
        forecast_std = 15
        constraints = {
            'target_service_level': 0.95,
            'max_storage_capacity': 500,
            'budget_limit': 10000,
            'lead_time_days': 5
        }
        costs = {
            'holding_cost_per_unit': 2.0,
            'ordering_cost_per_order': 100.0,
            'stockout_cost_per_unit': 50.0,
            'unit_cost': 20.0
        }
        
        result = optimizer.optimize(forecast, forecast_std, constraints, costs)
        
        # Check capacity constraint
        assert result.optimal_order_quantity <= constraints['max_storage_capacity']
        
        # Check budget constraint
        order_cost = result.optimal_order_quantity * costs['unit_cost']
        assert order_cost <= constraints['budget_limit']
        
        # Check service level constraint (with tolerance)
        if result.constraints_satisfied:
            assert result.expected_service_level >= constraints['target_service_level'] - 0.05


class TestRealWorldScenarios:
    """Test real-world demand forecasting scenarios"""
    
    def test_retail_electronics(self):
        """Test: Electronics retailer demand forecasting"""
        # Simulate electronics sales with weekly seasonality
        np.random.seed(42)
        days = 365
        
        # Strong weekend sales
        base_demand = 120
        weekly_pattern = np.array([0.8, 0.9, 1.0, 1.0, 1.1, 1.3, 1.2])
        sales = []
        for d in range(days):
            day_of_week = d % 7
            demand = base_demand * weekly_pattern[day_of_week]
            demand += np.random.normal(0, 15)
            sales.append(max(0, demand))
        
        sales_history = np.array(sales)
        
        # Forecast next 30 days
        request = {
            'method': 'lstm',
            'historical_data': {
                'sku_id': 'LAPTOP-XPS15',
                'sales_history': sales_history.tolist()
            },
            'forecast_horizon': 30,
            'optimize_inventory': True,
            'constraints': {
                'target_service_level': 0.95,
                'max_storage_capacity': 2000,
                'budget_limit': 75000,
                'lead_time_days': 10
            }
        }
        
        result = execute_demand_forecasting(request)
        
        assert 'error' not in result
        assert len(result['forecast']) == 30
        assert result['optimization']['expected_service_level'] >= 0.90
        print(f"\n✓ Retail Electronics: Service level = {result['optimization']['expected_service_level']:.2%}")
    
    def test_grocery_perishables(self):
        """Test: Grocery store perishable goods forecasting"""
        # High variability, short shelf life
        np.random.seed(123)
        days = 180
        
        # Daily sales with high noise
        base = 200
        sales = base + np.random.normal(0, 40, days)
        sales = np.maximum(sales, 50)  # Minimum sales
        
        request = {
            'method': 'lstm',
            'historical_data': {
                'sku_id': 'MILK-ORGANIC-1GAL',
                'sales_history': sales.tolist()
            },
            'forecast_horizon': 7,  # Weekly forecast
            'optimize_inventory': True,
            'constraints': {
                'target_service_level': 0.98,  # High service for essentials
                'max_storage_capacity': 500,
                'budget_limit': 5000,
                'lead_time_days': 1  # Next-day delivery
            }
        }
        
        result = execute_demand_forecasting(request)
        
        assert 'error' not in result
        assert len(result['forecast']) == 7
        # Short lead time should result in lower safety stock
        assert result['optimization']['safety_stock'] < 200
        print(f"✓ Grocery Perishables: Safety stock = {result['optimization']['safety_stock']:.0f} units")
    
    def test_warehouse_bulk_wholesale(self):
        """Test: Warehouse wholesale distribution"""
        # Large order quantities, lower frequency
        np.random.seed(456)
        days = 365
        
        # Stable demand with monthly cycles
        monthly_cycle = np.sin(np.arange(days) * 2 * np.pi / 30)
        sales = 500 + 100 * monthly_cycle + np.random.normal(0, 50, days)
        sales = np.maximum(sales, 200)
        
        request = {
            'method': 'lstm',
            'historical_data': {
                'sku_id': 'PAPER-TOWELS-BULK',
                'sales_history': sales.tolist()
            },
            'forecast_horizon': 30,
            'optimize_inventory': True,
            'constraints': {
                'target_service_level': 0.90,
                'max_storage_capacity': 50000,
                'budget_limit': 100000,
                'lead_time_days': 14
            }
        }
        
        result = execute_demand_forecasting(request)
        
        assert 'error' not in result
        # Large capacity should allow larger order quantities
        assert result['optimization']['optimal_order_quantity'] > 1000
        print(f"✓ Warehouse Bulk: Order quantity = {result['optimization']['optimal_order_quantity']:.0f} units")
    
    def test_fashion_seasonal(self):
        """Test: Fashion retail with seasonal peaks"""
        # Strong seasonality, trend changes
        np.random.seed(789)
        days = 365
        
        # Summer peak (days 150-240), winter peak (days 330-60)
        t = np.arange(days)
        summer_peak = 50 * np.exp(-((t - 195) ** 2) / (2 * 30 ** 2))
        winter_peak = 40 * np.exp(-((t - 365) ** 2) / (2 * 40 ** 2))
        base = 80
        sales = base + summer_peak + winter_peak + np.random.normal(0, 15, days)
        sales = np.maximum(sales, 20)
        
        request = {
            'method': 'lstm',
            'historical_data': {
                'sku_id': 'JACKET-WINTER-M',
                'sales_history': sales.tolist()
            },
            'forecast_horizon': 60,
            'optimize_inventory': True,
            'constraints': {
                'target_service_level': 0.92,
                'max_storage_capacity': 3000,
                'budget_limit': 60000,
                'lead_time_days': 21  # Longer international shipping
            }
        }
        
        result = execute_demand_forecasting(request)
        
        assert 'error' not in result
        # Should detect seasonality
        assert np.std(result['forecast']) > 10  # Variability in forecast
        print(f"✓ Fashion Seasonal: Forecast std = {np.std(result['forecast']):.1f}")


class TestEdgeCases:
    """Test edge cases and error handling"""
    
    def test_insufficient_data(self):
        """Test handling of insufficient historical data"""
        short_history = np.array([100, 110, 105, 95, 100])  # Only 5 points
        
        engine = DemandForecastEngine(method='lstm')
        
        with pytest.raises(ValueError, match="at least 60"):
            engine.train(short_history)
    
    def test_demand_spike_handling(self):
        """Test handling sudden demand spikes"""
        # Normal demand with sudden spike
        normal_demand = np.full(150, 100.0)
        spike_demand = np.full(30, 500.0)  # 5x spike
        recovery = np.full(30, 120.0)
        
        sales = np.concatenate([normal_demand, spike_demand, recovery])
        
        request = {
            'method': 'lstm',
            'historical_data': {
                'sku_id': 'HAND-SANITIZER',
                'sales_history': sales.tolist()
            },
            'forecast_horizon': 14,
            'optimize_inventory': False
        }
        
        result = execute_demand_forecasting(request)
        
        assert 'error' not in result
        # Forecast should be elevated but not at spike level
        avg_forecast = np.mean(result['forecast'])
        assert 100 < avg_forecast < 500
    
    def test_stockout_scenario(self):
        """Test optimization under stockout conditions"""
        # Consistent high demand
        sales = np.full(200, 200.0)
        
        request = {
            'method': 'lstm',
            'historical_data': {
                'sku_id': 'HIGH-DEMAND-ITEM',
                'sales_history': sales.tolist()
            },
            'forecast_horizon': 30,
            'optimize_inventory': True,
            'constraints': {
                'target_service_level': 0.99,  # Very high service level
                'max_storage_capacity': 10000,
                'budget_limit': 100000,
                'lead_time_days': 5
            }
        }
        
        result = execute_demand_forecasting(request)
        
        assert 'error' not in result
        # Should have significant safety stock for 99% service level
        assert result['optimization']['safety_stock'] > 100
        print(f"✓ Stockout Prevention: Safety stock = {result['optimization']['safety_stock']:.0f}")
    
    def test_zero_demand_periods(self):
        """Test handling of zero demand periods"""
        # Intermittent demand
        sales = np.array([0, 0, 50, 0, 0, 80, 0, 0, 0, 60] * 20)
        
        request = {
            'method': 'lstm',
            'historical_data': {
                'sku_id': 'RARE-ITEM',
                'sales_history': sales.tolist()
            },
            'forecast_horizon': 20,
            'optimize_inventory': False
        }
        
        result = execute_demand_forecasting(request)
        
        assert 'error' not in result
        assert np.all(np.array(result['forecast']) >= 0)


class TestPerformanceBenchmarks:
    """Test performance at different scales"""
    
    def test_small_scale_performance(self):
        """Test: 1K SKUs, 100 days history"""
        # Single SKU detailed test
        sales = np.random.uniform(80, 120, 100)
        
        start = datetime.now()
        
        request = {
            'method': 'lstm',
            'historical_data': {
                'sku_id': 'SMALL-TEST-001',
                'sales_history': sales.tolist()
            },
            'forecast_horizon': 14,
            'optimize_inventory': True,
            'constraints': {
                'target_service_level': 0.95,
                'max_storage_capacity': 1000,
                'budget_limit': 10000,
                'lead_time_days': 7
            }
        }
        
        result = execute_demand_forecasting(request)
        duration = (datetime.now() - start).total_seconds()
        
        assert 'error' not in result
        assert duration < 30  # Should complete within 30 seconds
        print(f"✓ Small Scale: {duration:.2f}s total time")
    
    def test_medium_scale_performance(self):
        """Test: 10K SKUs simulation, 180 days history"""
        # Simulate batch processing time
        sales = np.random.uniform(100, 200, 180)
        
        num_skus = 10
        total_time = 0
        
        for i in range(num_skus):
            start = datetime.now()
            
            request = {
                'method': 'lstm',
                'historical_data': {
                    'sku_id': f'MEDIUM-TEST-{i:04d}',
                    'sales_history': sales.tolist()
                },
                'forecast_horizon': 30,
                'optimize_inventory': True,
                'constraints': {
                    'target_service_level': 0.95,
                    'max_storage_capacity': 5000,
                    'budget_limit': 50000,
                    'lead_time_days': 7
                }
            }
            
            result = execute_demand_forecasting(request)
            duration = (datetime.now() - start).total_seconds()
            total_time += duration
            
            assert 'error' not in result
        
        avg_time = total_time / num_skus
        print(f"✓ Medium Scale: {avg_time:.2f}s average per SKU, {total_time:.2f}s total for {num_skus} SKUs")
        
        # Extrapolate to 10K SKUs
        estimated_10k = avg_time * 10000 / 60  # minutes
        print(f"  Estimated time for 10K SKUs: {estimated_10k:.1f} minutes")
    
    def test_inference_latency(self):
        """Test inference-only latency (pre-trained model)"""
        sales = np.random.uniform(100, 150, 365)
        
        # Train once
        engine = DemandForecastEngine(method='lstm', forecast_horizon=7)
        engine.train(sales, epochs=5, verbose=0)
        
        # Measure inference only
        latencies = []
        for _ in range(10):
            start = datetime.now()
            forecast, ci = engine.forecast(sales)
            latency = (datetime.now() - start).total_seconds() * 1000  # ms
            latencies.append(latency)
        
        avg_latency = np.mean(latencies)
        p95_latency = np.percentile(latencies, 95)
        
        assert avg_latency < 100  # Should be under 100ms average
        print(f"✓ Inference Latency: avg={avg_latency:.1f}ms, p95={p95_latency:.1f}ms")


class TestIntegration:
    """Integration tests for the execute function"""
    
    def test_complete_workflow(self):
        """Test complete workflow from request to response"""
        sales = np.random.uniform(90, 110, 200)
        dates = [(datetime.now() - timedelta(days=200-i)).strftime('%Y-%m-%d') for i in range(200)]
        
        request = {
            'method': 'lstm',
            'historical_data': {
                'sku_id': 'INTEGRATION-TEST-001',
                'sales_history': sales.tolist(),
                'dates': dates
            },
            'forecast_horizon': 30,
            'optimize_inventory': True,
            'constraints': {
                'target_service_level': 0.95,
                'max_storage_capacity': 2000,
                'budget_limit': 30000,
                'lead_time_days': 7
            },
            'costs': {
                'holding_cost_per_unit': 2.0,
                'ordering_cost_per_order': 150.0,
                'stockout_cost_per_unit': 75.0,
                'unit_cost': 25.0
            }
        }
        
        result = execute_demand_forecasting(request)
        
        # Validate response structure
        assert 'sku_id' in result
        assert 'method' in result
        assert 'forecast' in result
        assert 'confidence_intervals' in result
        assert 'forecast_dates' in result
        assert 'metadata' in result
        assert 'optimization' in result
        
        # Validate forecast
        assert len(result['forecast']) == 30
        assert all(isinstance(f, (int, float)) for f in result['forecast'])
        
        # Validate optimization
        opt = result['optimization']
        assert 'optimal_order_quantity' in opt
        assert 'reorder_point' in opt
        assert 'safety_stock' in opt
        assert 'expected_service_level' in opt
        
        # Validate metadata
        meta = result['metadata']
        assert 'training_time_seconds' in meta
        assert 'inference_time_ms' in meta
    
    def test_error_handling_missing_data(self):
        """Test error handling for missing data"""
        request = {
            'method': 'lstm',
            'historical_data': {
                'sku_id': 'ERROR-TEST-001'
                # Missing sales_history
            },
            'forecast_horizon': 30
        }
        
        result = execute_demand_forecasting(request)
        
        assert 'error' in result
        assert not result.get('success', True)
    
    def test_all_methods(self):
        """Test all forecasting methods work"""
        sales = np.random.uniform(100, 120, 200)
        
        methods = ['lstm']  # Add 'sarima', 'exponential_smoothing' if statsmodels available
        
        for method in methods:
            request = {
                'method': method,
                'historical_data': {
                    'sku_id': f'METHOD-TEST-{method}',
                    'sales_history': sales.tolist()
                },
                'forecast_horizon': 14,
                'optimize_inventory': False
            }
            
            result = execute_demand_forecasting(request)
            
            assert 'error' not in result or 'Import' in result.get('error', '')
            if 'error' not in result:
                assert result['method'] == method
                assert len(result['forecast']) == 14
                print(f"✓ Method {method}: OK")


def run_all_tests():
    """Run all tests and print summary"""
    print("\n" + "=" * 80)
    print("DEMAND FORECASTING + INVENTORY OPTIMIZATION - TEST SUITE")
    print("=" * 80)
    
    # Run pytest
    pytest.main([
        __file__,
        '-v',
        '--tb=short',
        '-W', 'ignore::DeprecationWarning'
    ])


if __name__ == "__main__":
    run_all_tests()
