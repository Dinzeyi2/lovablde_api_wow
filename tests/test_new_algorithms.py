"""
Test Suite for TSP Genetic Algorithm and KMP String Matching
=============================================================

Comprehensive tests covering:
- Basic functionality
- Edge cases
- Performance benchmarks
- Input validation
- Real-world scenarios

Run with: pytest test_new_algorithms.py -v
"""

import pytest
import time
from tsp_genetic import (
    TSPGeneticInput,
    TSPGeneticOutput,
    Location,
    tsp_genetic
)
from kmp_matching import (
    KMPInput,
    KMPOutput,
    kmp_search,
    kmp_replace,
    kmp_count,
    KMPMatcher
)


# ============================================================================
# TSP GENETIC ALGORITHM TESTS
# ============================================================================

class TestTSPGenetic:
    """Test cases for TSP Genetic Algorithm."""
    
    def test_simple_tsp_4_locations(self):
        """Test TSP with 4 locations forming a square."""
        locations = [
            Location(name="A", x=0, y=0),
            Location(name="B", x=0, y=10),
            Location(name="C", x=10, y=10),
            Location(name="D", x=10, y=0)
        ]
        
        input_data = TSPGeneticInput(
            locations=locations,
            population_size=50,
            generations=100
        )
        
        result = tsp_genetic(input_data)
        
        # Assertions
        assert result.best_route[0] == "A"  # Starts at A
        assert result.best_route[0] == result.best_route[-1] or len(result.best_route) == len(locations)
        assert len(result.best_route) == len(locations)
        assert result.total_distance == pytest.approx(40.0, abs=2.0)  # Square perimeter
        assert result.generations_evolved == 100
        assert result.execution_time_ms > 0
        assert len(result.route_segments) == len(locations)
    
    def test_tsp_triangle_optimal_route(self):
        """Test TSP with 3 locations forming a triangle."""
        locations = [
            Location(name="Origin", x=0, y=0),
            Location(name="Point1", x=3, y=0),
            Location(name="Point2", x=0, y=4)
        ]
        
        result = tsp_genetic(TSPGeneticInput(locations=locations, generations=200))
        
        # Total distance should be 3 + 4 + 5 = 12 (triangle sides)
        assert result.total_distance == pytest.approx(12.0, abs=1.0)
        assert result.best_route is not None
        assert len(result.best_route) == 3
    
    def test_tsp_with_specific_start(self):
        """Test TSP with specified starting location."""
        locations = [
            Location(name="Warehouse", x=0, y=0),
            Location(name="Store1", x=5, y=5),
            Location(name="Store2", x=10, y=0)
        ]
        
        result = tsp_genetic(TSPGeneticInput(
            locations=locations,
            start_location="Warehouse",
            generations=100
        ))
        
        assert result.best_route[0] == "Warehouse"
        assert "Store1" in result.best_route
        assert "Store2" in result.best_route
    
    def test_tsp_improvement_percentage(self):
        """Test that genetic algorithm improves over initial random solution."""
        locations = [
            Location(name=f"L{i}", x=i*2, y=i*3) 
            for i in range(8)
        ]
        
        result = tsp_genetic(TSPGeneticInput(
            locations=locations,
            population_size=100,
            generations=500
        ))
        
        # Should show improvement
        assert result.improvement_percentage > 0
        assert result.final_population_fitness["best"] <= result.final_population_fitness["average"]
        assert result.final_population_fitness["average"] <= result.final_population_fitness["worst"]
    
    def test_tsp_route_segments(self):
        """Test that route segments are correctly calculated."""
        locations = [
            Location(name="A", x=0, y=0),
            Location(name="B", x=3, y=4)
        ]
        
        result = tsp_genetic(TSPGeneticInput(locations=locations, generations=50))
        
        assert len(result.route_segments) == 2  # A->B and B->A
        total_segment_distance = sum(seg.distance for seg in result.route_segments)
        assert total_segment_distance == pytest.approx(result.total_distance, abs=0.1)
    
    def test_tsp_coordinates_output(self):
        """Test that coordinates are correctly output for visualization."""
        locations = [
            Location(name="A", x=1.5, y=2.5),
            Location(name="B", x=3.7, y=4.2)
        ]
        
        result = tsp_genetic(TSPGeneticInput(locations=locations, generations=50))
        
        assert len(result.coordinates) == len(locations)
        assert all("x" in coord and "y" in coord for coord in result.coordinates)
    
    def test_tsp_invalid_start_location(self):
        """Test error handling for invalid start location."""
        locations = [
            Location(name="A", x=0, y=0),
            Location(name="B", x=1, y=1)
        ]
        
        with pytest.raises(ValueError, match="Start location 'C' not found"):
            TSPGeneticInput(locations=locations, start_location="C")
    
    def test_tsp_duplicate_location_names(self):
        """Test error handling for duplicate location names."""
        locations = [
            Location(name="A", x=0, y=0),
            Location(name="A", x=1, y=1)  # Duplicate name
        ]
        
        with pytest.raises(ValueError, match="All location names must be unique"):
            TSPGeneticInput(locations=locations)
    
    def test_tsp_parameter_validation(self):
        """Test that algorithm parameters are validated."""
        locations = [Location(name="A", x=0, y=0), Location(name="B", x=1, y=1)]
        
        # Test population size bounds
        with pytest.raises(ValueError):
            TSPGeneticInput(locations=locations, population_size=5)  # Too small
        
        with pytest.raises(ValueError):
            TSPGeneticInput(locations=locations, population_size=1000)  # Too large
        
        # Test mutation rate bounds
        with pytest.raises(ValueError):
            TSPGeneticInput(locations=locations, mutation_rate=0.0005)  # Too small
        
        with pytest.raises(ValueError):
            TSPGeneticInput(locations=locations, mutation_rate=0.6)  # Too large


# ============================================================================
# KMP STRING MATCHING TESTS
# ============================================================================

class TestKMPStringMatching:
    """Test cases for KMP String Matching."""
    
    def test_simple_pattern_match(self):
        """Test basic pattern matching."""
        result = kmp_search(KMPInput(
            text="ABABDABACDABABCABAB",
            pattern="ABABCABAB"
        ))
        
        assert result.found == True
        assert result.occurrences == 1
        assert result.positions == [10]
        assert len(result.matches) == 1
        assert result.matches[0].position == 10
        assert result.matches[0].matched_text == "ABABCABAB"
    
    def test_multiple_occurrences(self):
        """Test finding multiple occurrences."""
        result = kmp_search(KMPInput(
            text="banana",
            pattern="ana"
        ))
        
        assert result.found == True
        assert result.occurrences == 2
        assert result.positions == [1, 3]
        assert len(result.matches) == 2
    
    def test_pattern_at_beginning(self):
        """Test pattern at the start of text."""
        result = kmp_search(KMPInput(
            text="hello world",
            pattern="hello"
        ))
        
        assert result.found == True
        assert result.positions == [0]
        assert result.matches[0].position == 0
    
    def test_pattern_at_end(self):
        """Test pattern at the end of text."""
        result = kmp_search(KMPInput(
            text="hello world",
            pattern="world"
        ))
        
        assert result.found == True
        assert result.positions == [6]
        assert result.matches[0].context_after == ""
    
    def test_pattern_not_found(self):
        """Test when pattern is not in text."""
        result = kmp_search(KMPInput(
            text="hello world",
            pattern="goodbye"
        ))
        
        assert result.found == False
        assert result.occurrences == 0
        assert result.positions == []
        assert len(result.matches) == 0
    
    def test_case_insensitive_search(self):
        """Test case-insensitive matching."""
        result = kmp_search(KMPInput(
            text="Hello World, HELLO Universe",
            pattern="hello",
            case_sensitive=False
        ))
        
        assert result.found == True
        assert result.occurrences == 2
        assert result.positions == [0, 13]
    
    def test_case_sensitive_search(self):
        """Test case-sensitive matching."""
        result = kmp_search(KMPInput(
            text="Hello World, hello Universe",
            pattern="hello",
            case_sensitive=True
        ))
        
        assert result.found == True
        assert result.occurrences == 1
        assert result.positions == [13]
    
    def test_find_first_only(self):
        """Test finding only the first occurrence."""
        result = kmp_search(KMPInput(
            text="banana",
            pattern="ana",
            find_all=False
        ))
        
        assert result.found == True
        assert result.occurrences == 1
        assert result.positions == [1]
    
    def test_max_matches_limit(self):
        """Test limiting maximum number of matches."""
        result = kmp_search(KMPInput(
            text="aaaaa",
            pattern="aa",
            max_matches=2
        ))
        
        assert result.found == True
        assert result.occurrences == 2
        assert len(result.positions) == 2
    
    def test_overlapping_patterns(self):
        """Test finding overlapping occurrences."""
        result = kmp_search(KMPInput(
            text="AAAA",
            pattern="AA"
        ))
        
        assert result.found == True
        assert result.occurrences == 3  # Positions 0, 1, 2
        assert result.positions == [0, 1, 2]
    
    def test_single_character_pattern(self):
        """Test single character pattern."""
        result = kmp_search(KMPInput(
            text="hello",
            pattern="l"
        ))
        
        assert result.found == True
        assert result.occurrences == 2
        assert result.positions == [2, 3]
    
    def test_pattern_equals_text(self):
        """Test when pattern is the entire text."""
        result = kmp_search(KMPInput(
            text="hello",
            pattern="hello"
        ))
        
        assert result.found == True
        assert result.occurrences == 1
        assert result.positions == [0]
    
    def test_empty_pattern_validation(self):
        """Test that empty pattern is rejected."""
        with pytest.raises(ValueError):
            KMPInput(text="hello", pattern="")
    
    def test_pattern_longer_than_text(self):
        """Test that pattern longer than text is rejected."""
        with pytest.raises(ValueError, match="Pattern cannot be longer than text"):
            KMPInput(text="hi", pattern="hello")
    
    def test_match_context(self):
        """Test that match context is correctly extracted."""
        result = kmp_search(KMPInput(
            text="The quick brown fox jumps over the lazy dog",
            pattern="fox"
        ))
        
        match = result.matches[0]
        assert match.matched_text == "fox"
        assert "brown " in match.context_before
        assert " jumps" in match.context_after
    
    def test_performance_metrics(self):
        """Test that performance metrics are calculated."""
        result = kmp_search(KMPInput(
            text="a" * 1000,
            pattern="aaa"
        ))
        
        assert "comparisons_made" in result.performance_metrics
        assert "efficiency_gain_vs_naive" in result.performance_metrics
        assert result.performance_metrics["comparisons_made"] > 0
        assert result.execution_time_ms > 0


# ============================================================================
# KMP UTILITY FUNCTIONS TESTS
# ============================================================================

class TestKMPUtilities:
    """Test utility functions for KMP."""
    
    def test_kmp_replace_all(self):
        """Test replacing all occurrences."""
        text = "Hello world, hello universe"
        result, count = kmp_replace(text, "hello", "goodbye", case_sensitive=False)
        
        assert result == "goodbye world, goodbye universe"
        assert count == 2
    
    def test_kmp_replace_max(self):
        """Test replacing limited occurrences."""
        text = "aaa"
        result, count = kmp_replace(text, "a", "b", max_replacements=2)
        
        assert result == "bba"
        assert count == 2
    
    def test_kmp_replace_not_found(self):
        """Test replace when pattern not found."""
        text = "hello"
        result, count = kmp_replace(text, "goodbye", "hi")
        
        assert result == "hello"
        assert count == 0
    
    def test_kmp_count(self):
        """Test counting occurrences."""
        count = kmp_count("banana", "ana")
        assert count == 2
        
        count = kmp_count("hello", "x")
        assert count == 0


# ============================================================================
# PERFORMANCE BENCHMARKS
# ============================================================================

class TestPerformance:
    """Performance benchmark tests."""
    
    def test_tsp_performance_10_cities(self):
        """Test TSP performance with 10 cities."""
        locations = [
            Location(name=f"City{i}", x=i*10, y=i*5) 
            for i in range(10)
        ]
        
        start = time.time()
        result = tsp_genetic(TSPGeneticInput(
            locations=locations,
            population_size=100,
            generations=200
        ))
        elapsed = time.time() - start
        
        assert elapsed < 5.0  # Should complete in under 5 seconds
        assert result.execution_time_ms < 5000
        print(f"\nTSP (10 cities): {result.execution_time_ms:.2f}ms")
    
    def test_tsp_performance_20_cities(self):
        """Test TSP performance with 20 cities."""
        locations = [
            Location(name=f"City{i}", x=i*7, y=i*11) 
            for i in range(20)
        ]
        
        start = time.time()
        result = tsp_genetic(TSPGeneticInput(
            locations=locations,
            population_size=100,
            generations=200
        ))
        elapsed = time.time() - start
        
        assert elapsed < 15.0  # Should complete in under 15 seconds
        print(f"\nTSP (20 cities): {result.execution_time_ms:.2f}ms")
    
    def test_kmp_performance_large_text(self):
        """Test KMP performance with large text."""
        text = "ABCD" * 25000  # 100,000 characters
        pattern = "ABCDABCD"
        
        start = time.time()
        result = kmp_search(KMPInput(text=text, pattern=pattern))
        elapsed = time.time() - start
        
        assert elapsed < 1.0  # Should be very fast
        assert result.execution_time_ms < 1000
        print(f"\nKMP (100K chars): {result.execution_time_ms:.2f}ms, {result.occurrences} matches")
    
    def test_kmp_efficiency_vs_naive(self):
        """Test that KMP is more efficient than naive search."""
        text = "A" * 10000 + "B"
        pattern = "A" * 100 + "B"
        
        result = kmp_search(KMPInput(text=text, pattern=pattern))
        
        # KMP should show significant efficiency gain
        assert result.performance_metrics["efficiency_gain_vs_naive"] > 50.0
        print(f"\nKMP efficiency gain: {result.performance_metrics['efficiency_gain_vs_naive']:.2f}%")


# ============================================================================
# REAL-WORLD SCENARIO TESTS
# ============================================================================

class TestRealWorldScenarios:
    """Test real-world use cases."""
    
    def test_delivery_route_optimization(self):
        """Test delivery route optimization scenario."""
        # Simulate food delivery locations in a city
        locations = [
            Location(name="Restaurant", x=0, y=0),
            Location(name="Customer1", x=2, y=3),
            Location(name="Customer2", x=5, y=1),
            Location(name="Customer3", x=3, y=7),
            Location(name="Customer4", x=8, y=4)
        ]
        
        result = tsp_genetic(TSPGeneticInput(
            locations=locations,
            start_location="Restaurant",
            population_size=100,
            generations=300
        ))
        
        # Verify result makes sense
        assert result.best_route[0] == "Restaurant"
        assert len(result.best_route) == 5
        assert result.total_distance > 0
        assert result.improvement_percentage > 0
        
        # Check that all customers are visited
        customers = {"Customer1", "Customer2", "Customer3", "Customer4"}
        assert customers.issubset(set(result.best_route))
    
    def test_log_file_analysis(self):
        """Test log file analysis with KMP."""
        log_text = """
        2024-12-30 10:15:23 ERROR: Connection timeout
        2024-12-30 10:15:45 INFO: Retry attempt 1
        2024-12-30 10:16:01 ERROR: Connection timeout
        2024-12-30 10:16:30 INFO: Retry attempt 2
        2024-12-30 10:17:00 ERROR: Connection timeout
        2024-12-30 10:17:45 ERROR: Max retries exceeded
        """
        
        # Search for error messages
        result = kmp_search(KMPInput(
            text=log_text,
            pattern="ERROR:"
        ))
        
        assert result.found == True
        assert result.occurrences == 4
        
        # All errors should be found
        assert len(result.positions) == 4
    
    def test_dna_sequence_matching(self):
        """Test DNA sequence matching scenario."""
        dna_sequence = "ATCGATCGATCGTAGCTAGCTAGCTACGATCGTAGCTAGC"
        gene_pattern = "TAGCTAGC"
        
        result = kmp_search(KMPInput(
            text=dna_sequence,
            pattern=gene_pattern
        ))
        
        assert result.found == True
        assert result.occurrences >= 1
        
        # Verify matches are actual gene sequences
        for match in result.matches:
            assert match.matched_text == gene_pattern


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
