"""
Knapsack Problem Algorithms
Classic optimization problem with multiple variants

Variants:
1. 0/1 Knapsack: Each item can be taken once or not at all
2. Fractional Knapsack: Items can be divided (continuous)
3. Unbounded Knapsack: Unlimited copies of each item

Use cases:
- Resource allocation
- Portfolio optimization
- Cargo loading
- Budget planning
"""

from typing import List, Literal, Tuple
from pydantic import BaseModel, Field, validator


class KnapsackItem(BaseModel):
    """Single item in knapsack problem"""
    name: str = Field(..., description="Item identifier")
    weight: float = Field(..., gt=0, description="Item weight")
    value: float = Field(..., gt=0, description="Item value")


class KnapsackInput(BaseModel):
    """Input schema for knapsack algorithms"""
    
    capacity: float = Field(..., gt=0, description="Maximum weight capacity")
    items: List[KnapsackItem] = Field(..., min_items=1, description="Available items")
    variant: Literal["01", "fractional", "unbounded"] = Field(
        default="01",
        description="Knapsack variant to solve"
    )


class KnapsackOutput(BaseModel):
    """Output schema for knapsack algorithms"""
    
    max_value: float = Field(..., description="Maximum achievable value")
    selected_items: List[dict] = Field(..., description="Items selected with quantities")
    total_weight: float = Field(..., description="Total weight of selected items")
    capacity_used_percent: float = Field(..., description="Percentage of capacity used")
    execution_time_ms: float = Field(..., description="Algorithm execution time")


def knapsack_01(items: List[KnapsackItem], capacity: float) -> Tuple[float, List[dict], float]:
    """
    0/1 Knapsack using Dynamic Programming
    Each item can be taken once or not at all
    
    Time Complexity: O(n * W) where n=items, W=capacity
    Space Complexity: O(n * W)
    """
    n = len(items)
    
    # Create DP table
    # dp[i][w] = max value using first i items with capacity w
    dp = [[0.0 for _ in range(int(capacity) + 1)] for _ in range(n + 1)]
    
    # Fill DP table
    for i in range(1, n + 1):
        item = items[i - 1]
        item_weight = int(item.weight)
        item_value = item.value
        
        for w in range(int(capacity) + 1):
            # Don't take item
            dp[i][w] = dp[i - 1][w]
            
            # Take item if possible
            if item_weight <= w:
                dp[i][w] = max(
                    dp[i][w],
                    dp[i - 1][w - item_weight] + item_value
                )
    
    # Backtrack to find selected items
    selected = []
    w = int(capacity)
    total_weight = 0
    
    for i in range(n, 0, -1):
        if dp[i][w] != dp[i - 1][w]:
            item = items[i - 1]
            selected.append({
                "name": item.name,
                "weight": item.weight,
                "value": item.value,
                "quantity": 1
            })
            total_weight += item.weight
            w -= int(item.weight)
    
    max_value = dp[n][int(capacity)]
    return max_value, selected, total_weight


def knapsack_fractional(items: List[KnapsackItem], capacity: float) -> Tuple[float, List[dict], float]:
    """
    Fractional Knapsack using Greedy approach
    Items can be divided (take fractions)
    
    Time Complexity: O(n log n) for sorting
    Space Complexity: O(n)
    """
    # Sort by value-to-weight ratio (descending)
    sorted_items = sorted(
        items,
        key=lambda x: x.value / x.weight,
        reverse=True
    )
    
    max_value = 0.0
    total_weight = 0.0
    selected = []
    remaining_capacity = capacity
    
    for item in sorted_items:
        if remaining_capacity == 0:
            break
        
        if item.weight <= remaining_capacity:
            # Take whole item
            max_value += item.value
            total_weight += item.weight
            remaining_capacity -= item.weight
            
            selected.append({
                "name": item.name,
                "weight": item.weight,
                "value": item.value,
                "quantity": 1.0,
                "fraction": 1.0
            })
        else:
            # Take fraction of item
            fraction = remaining_capacity / item.weight
            max_value += item.value * fraction
            total_weight += remaining_capacity
            
            selected.append({
                "name": item.name,
                "weight": remaining_capacity,
                "value": item.value * fraction,
                "quantity": fraction,
                "fraction": round(fraction, 4)
            })
            
            remaining_capacity = 0
    
    return max_value, selected, total_weight


def knapsack_unbounded(items: List[KnapsackItem], capacity: float) -> Tuple[float, List[dict], float]:
    """
    Unbounded Knapsack using Dynamic Programming
    Unlimited copies of each item available
    
    Time Complexity: O(n * W)
    Space Complexity: O(W)
    """
    W = int(capacity)
    
    # dp[w] = max value with capacity w
    dp = [0.0] * (W + 1)
    
    # Track which item was last added
    last_item = [-1] * (W + 1)
    
    # Fill DP table
    for w in range(W + 1):
        for i, item in enumerate(items):
            item_weight = int(item.weight)
            if item_weight <= w:
                new_value = dp[w - item_weight] + item.value
                if new_value > dp[w]:
                    dp[w] = new_value
                    last_item[w] = i
    
    # Backtrack to find selected items
    selected_counts = {item.name: 0 for item in items}
    w = W
    total_weight = 0
    
    while w > 0 and last_item[w] != -1:
        item_idx = last_item[w]
        item = items[item_idx]
        selected_counts[item.name] += 1
        total_weight += item.weight
        w -= int(item.weight)
    
    # Format output
    selected = []
    for item in items:
        if selected_counts[item.name] > 0:
            quantity = selected_counts[item.name]
            selected.append({
                "name": item.name,
                "weight": item.weight,
                "value": item.value,
                "quantity": quantity,
                "total_weight": item.weight * quantity,
                "total_value": item.value * quantity
            })
    
    max_value = dp[W]
    return max_value, selected, total_weight


def execute_knapsack(input_data: KnapsackInput) -> KnapsackOutput:
    """
    Execute knapsack algorithm based on variant
    """
    import time
    
    start_time = time.perf_counter()
    
    # Select algorithm based on variant
    if input_data.variant == "01":
        max_value, selected, total_weight = knapsack_01(input_data.items, input_data.capacity)
    elif input_data.variant == "fractional":
        max_value, selected, total_weight = knapsack_fractional(input_data.items, input_data.capacity)
    else:  # unbounded
        max_value, selected, total_weight = knapsack_unbounded(input_data.items, input_data.capacity)
    
    execution_time = (time.perf_counter() - start_time) * 1000
    
    capacity_used = (total_weight / input_data.capacity) * 100 if input_data.capacity > 0 else 0
    
    return KnapsackOutput(
        max_value=round(max_value, 2),
        selected_items=selected,
        total_weight=round(total_weight, 2),
        capacity_used_percent=round(capacity_used, 2),
        execution_time_ms=round(execution_time, 3)
    )


# Example usage
if __name__ == "__main__":
    from typing import Tuple
    
    # Test 0/1 Knapsack
    test_input = KnapsackInput(
        capacity=50,
        items=[
            KnapsackItem(name="item1", weight=10, value=60),
            KnapsackItem(name="item2", weight=20, value=100),
            KnapsackItem(name="item3", weight=30, value=120)
        ],
        variant="01"
    )
    
    result = execute_knapsack(test_input)
    print(f"Variant: {test_input.variant}")
    print(f"Max value: {result.max_value}")
    print(f"Selected: {result.selected_items}")
    print(f"Capacity used: {result.capacity_used_percent}%")
