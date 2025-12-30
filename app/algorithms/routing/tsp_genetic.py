"""
Traveling Salesman Problem (TSP) using Genetic Algorithm
=========================================================

Solves the TSP to find the shortest route visiting all locations exactly once.
Uses genetic algorithm optimization with tournament selection, ordered crossover,
and swap mutation.

Use Cases:
- Delivery route optimization (DoorDash, Uber Eats, Amazon)
- Sales territory planning
- Warehouse order picking
- Service technician routing
- Multi-stop journey planning

Time Complexity: O(g * p * n²) where g=generations, p=population, n=cities
Space Complexity: O(p * n)

Author: AlgoAPI
Version: 1.0
"""

import random
import time
from typing import List, Dict, Tuple, Optional, Any
from pydantic import BaseModel, Field, field_validator
from pydantic import ConfigDict
import math


# ============================================================================
# PYDANTIC SCHEMAS
# ============================================================================

class Location(BaseModel):
    """A location with coordinates."""
    model_config = ConfigDict(json_schema_extra={
        "example": {
            "name": "Store A",
            "x": 40.7128,
            "y": -74.0060
        }
    })
    
    name: str = Field(..., description="Name of the location")
    x: float = Field(..., description="X coordinate (latitude)")
    y: float = Field(..., description="Y coordinate (longitude)")


class TSPGeneticInput(BaseModel):
    """Input schema for TSP genetic algorithm."""
    locations: List[Location] = Field(
        ...,
        min_length=2,
        max_length=100,
        description="List of locations to visit (2-100 locations)"
    )
    start_location: Optional[str] = Field(
        None,
        description="Name of starting location (optional, defaults to first location)"
    )
    population_size: int = Field(
        default=100,
        ge=10,
        le=500,
        description="Size of population per generation (10-500)"
    )
    generations: int = Field(
        default=500,
        ge=10,
        le=2000,
        description="Number of generations to evolve (10-2000)"
    )
    mutation_rate: float = Field(
        default=0.01,
        ge=0.001,
        le=0.5,
        description="Probability of mutation (0.001-0.5)"
    )
    tournament_size: int = Field(
        default=5,
        ge=2,
        le=20,
        description="Number of individuals in tournament selection (2-20)"
    )
    elitism_count: int = Field(
        default=2,
        ge=0,
        le=10,
        description="Number of best individuals to preserve (0-10)"
    )
    
    @field_validator('locations')
    @classmethod
    def validate_unique_names(cls, v):
        """Ensure all location names are unique."""
        names = [loc.name for loc in v]
        if len(names) != len(set(names)):
            raise ValueError("All location names must be unique")
        return v
    
    @field_validator('start_location')
    @classmethod
    def validate_start_location(cls, v, info):
        """Ensure start location exists in locations list."""
        if v is not None and 'locations' in info.data:
            location_names = [loc.name for loc in info.data['locations']]
            if v not in location_names:
                raise ValueError(f"Start location '{v}' not found in locations list")
        return v
    
    model_config = ConfigDict(json_schema_extra={
            "example": {
                "locations": [
                    {"name": "Warehouse", "x": 0, "y": 0},
                    {"name": "Store A", "x": 2, "y": 3},
                    {"name": "Store B", "x": 5, "y": 1},
                    {"name": "Store C", "x": 3, "y": 7}
                ],
                "start_location": "Warehouse",
                "population_size": 100,
                "generations": 500,
                "mutation_rate": 0.01
            }
        }
    )


class RouteSegment(BaseModel):
    """A segment of the route."""
    from_location: str
    to_location: str
    distance: float


class TSPGeneticOutput(BaseModel):
    """Output schema for TSP genetic algorithm."""
    best_route: List[str] = Field(..., description="Optimal route (list of location names)")
    total_distance: float = Field(..., description="Total distance of the route")
    route_segments: List[RouteSegment] = Field(..., description="Detailed route segments")
    coordinates: List[Dict[str, Any]] = Field(..., description="Coordinates for visualization")
    improvement_percentage: float = Field(..., description="% improvement over initial random route")
    generations_evolved: int = Field(..., description="Number of generations evolved")
    final_population_fitness: Dict[str, float] = Field(..., description="Statistics on final population")
    execution_time_ms: float = Field(..., description="Execution time in milliseconds")
    algorithm_parameters: Dict[str, Any] = Field(..., description="Parameters used")
    
    model_config = ConfigDict(json_schema_extra={
            "example": {
                "best_route": ["Warehouse", "Store B", "Store A", "Store C", "Warehouse"],
                "total_distance": 15.23,
                "route_segments": [
                    {"from_location": "Warehouse", "to_location": "Store B", "distance": 5.39}
                ],
                "coordinates": [{"x": 0, "y": 0}, {"x": 5, "y": 1}],
                "improvement_percentage": 23.5,
                "generations_evolved": 500,
                "final_population_fitness": {
                    "best": 15.23,
                    "worst": 18.76,
                    "average": 16.45
                },
                "execution_time_ms": 234.56
            }
        }
    )


# ============================================================================
# GENETIC ALGORITHM IMPLEMENTATION
# ============================================================================

class TSPGeneticAlgorithm:
    """
    Genetic Algorithm solver for Traveling Salesman Problem.
    
    Uses:
    - Tournament selection for parent selection
    - Ordered crossover (OX) for reproduction
    - Swap mutation for genetic diversity
    - Elitism to preserve best solutions
    """
    
    def __init__(
        self,
        locations: List[Location],
        start_location: Optional[str] = None,
        population_size: int = 100,
        generations: int = 500,
        mutation_rate: float = 0.01,
        tournament_size: int = 5,
        elitism_count: int = 2
    ):
        self.locations = locations
        self.location_dict = {loc.name: loc for loc in locations}
        self.start_location = start_location or locations[0].name
        self.population_size = population_size
        self.generations = generations
        self.mutation_rate = mutation_rate
        self.tournament_size = tournament_size
        self.elitism_count = elitism_count
        
        # Distance matrix for O(1) lookups
        self.distance_matrix = self._compute_distance_matrix()
        
        # Statistics
        self.best_fitness_history = []
        self.avg_fitness_history = []
    
    def _compute_distance_matrix(self) -> Dict[Tuple[str, str], float]:
        """Pre-compute all pairwise distances using Euclidean distance."""
        distance_matrix = {}
        for loc1 in self.locations:
            for loc2 in self.locations:
                if loc1.name != loc2.name:
                    dist = math.sqrt(
                        (loc1.x - loc2.x)**2 + (loc1.y - loc2.y)**2
                    )
                    distance_matrix[(loc1.name, loc2.name)] = dist
        return distance_matrix
    
    def _calculate_route_distance(self, route: List[str]) -> float:
        """Calculate total distance of a route."""
        total_distance = 0.0
        for i in range(len(route) - 1):
            total_distance += self.distance_matrix[(route[i], route[i + 1])]
        # Return to start
        total_distance += self.distance_matrix[(route[-1], route[0])]
        return total_distance
    
    def _create_individual(self) -> List[str]:
        """Create a random individual (route)."""
        locations_copy = [loc.name for loc in self.locations if loc.name != self.start_location]
        random.shuffle(locations_copy)
        return [self.start_location] + locations_copy
    
    def _initialize_population(self) -> List[List[str]]:
        """Create initial population of random routes."""
        return [self._create_individual() for _ in range(self.population_size)]
    
    def _tournament_selection(self, population: List[List[str]], fitnesses: List[float]) -> List[str]:
        """Select an individual using tournament selection."""
        tournament_indices = random.sample(range(len(population)), self.tournament_size)
        tournament_fitnesses = [(fitnesses[i], population[i]) for i in tournament_indices]
        # Select best (minimum distance)
        winner = min(tournament_fitnesses, key=lambda x: x[0])
        return winner[1].copy()
    
    def _ordered_crossover(self, parent1: List[str], parent2: List[str]) -> List[str]:
        """
        Perform ordered crossover (OX) to create offspring.
        Preserves relative order from parents while avoiding duplicates.
        """
        size = len(parent1)
        
        # If only 2 locations, just return parent1 (no crossover needed)
        if size <= 2:
            return parent1.copy()
        
        # Keep start location fixed
        offspring = [self.start_location] + [None] * (size - 1)
        
        # Select random segment from parent1 (excluding start location)
        start_idx = random.randint(1, size - 2)
        end_idx = random.randint(start_idx, size - 1)
        
        # Copy segment from parent1
        offspring[start_idx:end_idx + 1] = parent1[start_idx:end_idx + 1]
        
        # Fill remaining positions with parent2's order
        parent2_filtered = [city for city in parent2 if city not in offspring]
        
        offspring_idx = 1
        for city in parent2_filtered:
            while offspring_idx < size and offspring[offspring_idx] is not None:
                offspring_idx += 1
            if offspring_idx < size:
                offspring[offspring_idx] = city
        
        return offspring
    
    def _swap_mutation(self, individual: List[str]) -> List[str]:
        """
        Perform swap mutation: randomly swap two cities (excluding start location).
        """
        mutated = individual.copy()
        
        if len(mutated) > 2 and random.random() < self.mutation_rate:
            # Only mutate non-start locations
            idx1, idx2 = random.sample(range(1, len(mutated)), 2)
            mutated[idx1], mutated[idx2] = mutated[idx2], mutated[idx1]
        
        return mutated
    
    def solve(self) -> Tuple[List[str], float, Dict]:
        """
        Run the genetic algorithm to find optimal route.
        
        Returns:
            Tuple of (best_route, best_distance, statistics)
        """
        # Initialize population
        population = self._initialize_population()
        
        # Track initial random route for comparison
        initial_distance = self._calculate_route_distance(population[0])
        
        best_individual = None
        best_fitness = float('inf')
        
        # Evolution loop
        for generation in range(self.generations):
            # Calculate fitness for all individuals
            fitnesses = [self._calculate_route_distance(ind) for ind in population]
            
            # Track best solution
            gen_best_idx = fitnesses.index(min(fitnesses))
            gen_best_fitness = fitnesses[gen_best_idx]
            
            if gen_best_fitness < best_fitness:
                best_fitness = gen_best_fitness
                best_individual = population[gen_best_idx].copy()
            
            # Statistics
            self.best_fitness_history.append(best_fitness)
            self.avg_fitness_history.append(sum(fitnesses) / len(fitnesses))
            
            # Create new population
            new_population = []
            
            # Elitism: preserve best individuals
            elite_indices = sorted(range(len(fitnesses)), key=lambda i: fitnesses[i])[:self.elitism_count]
            new_population.extend([population[i].copy() for i in elite_indices])
            
            # Generate offspring
            while len(new_population) < self.population_size:
                # Selection
                parent1 = self._tournament_selection(population, fitnesses)
                parent2 = self._tournament_selection(population, fitnesses)
                
                # Crossover
                offspring = self._ordered_crossover(parent1, parent2)
                
                # Mutation
                offspring = self._swap_mutation(offspring)
                
                new_population.append(offspring)
            
            population = new_population
        
        # Calculate improvement
        improvement = ((initial_distance - best_fitness) / initial_distance) * 100
        
        # Final population statistics
        final_fitnesses = [self._calculate_route_distance(ind) for ind in population]
        statistics = {
            "best": min(final_fitnesses),
            "worst": max(final_fitnesses),
            "average": sum(final_fitnesses) / len(final_fitnesses),
            "initial_distance": initial_distance,
            "improvement": improvement
        }
        
        return best_individual, best_fitness, statistics


# ============================================================================
# MAIN ALGORITHM FUNCTION
# ============================================================================

def tsp_genetic(input_data: TSPGeneticInput) -> TSPGeneticOutput:
    """
    Solve Traveling Salesman Problem using Genetic Algorithm.
    
    Args:
        input_data: TSPGeneticInput with locations and parameters
        
    Returns:
        TSPGeneticOutput with optimal route and statistics
        
    Example:
        >>> locations = [
        ...     Location(name="A", x=0, y=0),
        ...     Location(name="B", x=2, y=3),
        ...     Location(name="C", x=5, y=1)
        ... ]
        >>> result = tsp_genetic(TSPGeneticInput(locations=locations))
        >>> print(result.best_route)
        ['A', 'C', 'B', 'A']
    """
    start_time = time.time()
    
    # Initialize genetic algorithm
    ga = TSPGeneticAlgorithm(
        locations=input_data.locations,
        start_location=input_data.start_location,
        population_size=input_data.population_size,
        generations=input_data.generations,
        mutation_rate=input_data.mutation_rate,
        tournament_size=input_data.tournament_size,
        elitism_count=input_data.elitism_count
    )
    
    # Solve
    best_route, best_distance, statistics = ga.solve()
    
    # Build route segments
    route_segments = []
    for i in range(len(best_route)):
        from_loc = best_route[i]
        to_loc = best_route[(i + 1) % len(best_route)]
        distance = ga.distance_matrix[(from_loc, to_loc)]
        route_segments.append(RouteSegment(
            from_location=from_loc,
            to_location=to_loc,
            distance=round(distance, 2)
        ))
    
    # Get coordinates for visualization
    coordinates = [
        {"name": loc_name, "x": ga.location_dict[loc_name].x, "y": ga.location_dict[loc_name].y}
        for loc_name in best_route
    ]
    
    # Calculate execution time
    execution_time_ms = (time.time() - start_time) * 1000
    
    return TSPGeneticOutput(
        best_route=best_route,
        total_distance=round(best_distance, 2),
        route_segments=route_segments,
        coordinates=coordinates,
        improvement_percentage=round(statistics["improvement"], 2),
        generations_evolved=input_data.generations,
        final_population_fitness={
            "best": round(statistics["best"], 2),
            "worst": round(statistics["worst"], 2),
            "average": round(statistics["average"], 2)
        },
        execution_time_ms=round(execution_time_ms, 2),
        algorithm_parameters={
            "population_size": input_data.population_size,
            "generations": input_data.generations,
            "mutation_rate": input_data.mutation_rate,
            "tournament_size": input_data.tournament_size,
            "elitism_count": input_data.elitism_count
        }
    )
