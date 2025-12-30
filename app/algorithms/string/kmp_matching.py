"""
Knuth-Morris-Pratt (KMP) String Matching Algorithm
===================================================

Efficiently finds all occurrences of a pattern in a text string.
Uses preprocessing of the pattern to avoid unnecessary comparisons.

Use Cases:
- Document search (Ctrl+F functionality)
- Log file analysis and parsing
- DNA sequence matching
- Plagiarism detection
- Code search in repositories
- Network packet inspection
- Real-time text filtering

Time Complexity: O(n + m) where n=text length, m=pattern length
Space Complexity: O(m) for LPS array

Author: AlgoAPI
Version: 1.0
"""

import time
from typing import List, Dict, Optional, Any, Tuple
from pydantic import BaseModel, Field, field_validator
from pydantic import ConfigDict


# ============================================================================
# PYDANTIC SCHEMAS
# ============================================================================

class KMPInput(BaseModel):
    """Input schema for KMP string matching."""
    text: str = Field(
        ...,
        min_length=1,
        max_length=1000000,
        description="The text to search in (1-1,000,000 characters)"
    )
    pattern: str = Field(
        ...,
        min_length=1,
        max_length=10000,
        description="The pattern to search for (1-10,000 characters)"
    )
    case_sensitive: bool = Field(
        default=True,
        description="Whether the search should be case-sensitive"
    )
    find_all: bool = Field(
        default=True,
        description="Find all occurrences (True) or just the first (False)"
    )
    max_matches: Optional[int] = Field(
        default=None,
        ge=1,
        le=100000,
        description="Maximum number of matches to return (optional)"
    )
    
    @field_validator('pattern')
    @classmethod
    def validate_pattern_length(cls, v, info):
        """Ensure pattern is not longer than text."""
        if 'text' in info.data and len(v) > len(info.data['text']):
            raise ValueError("Pattern cannot be longer than text")
        return v
    
    model_config = ConfigDict(json_schema_extra={
            "example": {
                "text": "ABABDABACDABABCABAB",
                "pattern": "ABABCABAB",
                "case_sensitive": True,
                "find_all": True
            }
        }
    )


class MatchInfo(BaseModel):
    """Information about a single match."""
    position: int = Field(..., description="Starting position of match (0-indexed)")
    matched_text: str = Field(..., description="The matched substring")
    context_before: str = Field(..., description="Text before match (up to 20 chars)")
    context_after: str = Field(..., description="Text after match (up to 20 chars)")


class KMPOutput(BaseModel):
    """Output schema for KMP string matching."""
    found: bool = Field(..., description="Whether pattern was found in text")
    occurrences: int = Field(..., description="Total number of occurrences found")
    positions: List[int] = Field(..., description="List of starting positions (0-indexed)")
    matches: List[MatchInfo] = Field(..., description="Detailed information about each match")
    pattern_length: int = Field(..., description="Length of the search pattern")
    text_length: int = Field(..., description="Length of the text searched")
    search_parameters: Dict[str, Any] = Field(..., description="Parameters used for search")
    performance_metrics: Dict[str, float] = Field(..., description="Performance statistics")
    execution_time_ms: float = Field(..., description="Execution time in milliseconds")
    
    model_config = ConfigDict(json_schema_extra={
            "example": {
                "found": True,
                "occurrences": 1,
                "positions": [10],
                "matches": [
                    {
                        "position": 10,
                        "matched_text": "ABABCABAB",
                        "context_before": "ABABDABACD",
                        "context_after": ""
                    }
                ],
                "pattern_length": 9,
                "text_length": 19,
                "execution_time_ms": 0.52
            }
        }
    )


# ============================================================================
# KMP ALGORITHM IMPLEMENTATION
# ============================================================================

class KMPMatcher:
    """
    Knuth-Morris-Pratt string matching algorithm.
    
    The KMP algorithm preprocesses the pattern to create a "partial match" table
    (also called LPS - Longest Proper Prefix which is also Suffix). This allows
    the algorithm to skip unnecessary comparisons when a mismatch occurs.
    
    Example:
        Pattern: "ABABCABAB"
        LPS:     [0,0,1,2,0,1,2,3,4]
        
        This means if we mismatch at position 4 (the 'C'), we know that the
        first 2 characters already match, so we can skip ahead.
    """
    
    def __init__(self, pattern: str, case_sensitive: bool = True):
        """
        Initialize KMP matcher with a pattern.
        
        Args:
            pattern: The pattern to search for
            case_sensitive: Whether to perform case-sensitive matching
        """
        self.original_pattern = pattern
        self.case_sensitive = case_sensitive
        
        # Normalize pattern if case-insensitive
        self.pattern = pattern if case_sensitive else pattern.lower()
        self.pattern_length = len(self.pattern)
        
        # Build LPS (Longest Proper Prefix which is also Suffix) array
        self.lps = self._compute_lps_array()
        
        # Statistics
        self.comparisons_made = 0
        self.characters_scanned = 0
    
    def _compute_lps_array(self) -> List[int]:
        """
        Compute the LPS (Longest Proper Prefix which is also Suffix) array.
        
        The LPS array indicates the length of the longest proper prefix of the
        pattern that is also a suffix of the pattern, for each position.
        
        Time Complexity: O(m) where m is pattern length
        
        Returns:
            List of integers representing LPS values
            
        Example:
            Pattern: "ABABCABAB"
            LPS:     [0,0,1,2,0,1,2,3,4]
            
            Position 8: "ABABCABA" has "ABAB" as both prefix and suffix (length 4)
        """
        lps = [0] * self.pattern_length
        length = 0  # Length of previous longest prefix suffix
        i = 1
        
        while i < self.pattern_length:
            if self.pattern[i] == self.pattern[length]:
                length += 1
                lps[i] = length
                i += 1
            else:
                if length != 0:
                    # Try with shorter prefix
                    length = lps[length - 1]
                else:
                    lps[i] = 0
                    i += 1
        
        return lps
    
    def search(
        self,
        text: str,
        find_all: bool = True,
        max_matches: Optional[int] = None
    ) -> List[int]:
        """
        Search for pattern in text using KMP algorithm.
        
        Args:
            text: The text to search in
            find_all: Whether to find all occurrences or just the first
            max_matches: Maximum number of matches to return
            
        Returns:
            List of starting positions where pattern was found
            
        Time Complexity: O(n) where n is text length
        
        Example:
            >>> matcher = KMPMatcher("ABABCABAB")
            >>> positions = matcher.search("ABABDABACDABABCABAB")
            >>> print(positions)
            [10]
        """
        # Normalize text if case-insensitive
        search_text = text if self.case_sensitive else text.lower()
        text_length = len(search_text)
        
        matches = []
        i = 0  # Index for text
        j = 0  # Index for pattern
        
        self.comparisons_made = 0
        self.characters_scanned = 0
        
        while i < text_length:
            self.characters_scanned += 1
            self.comparisons_made += 1
            
            if search_text[i] == self.pattern[j]:
                i += 1
                j += 1
            
            if j == self.pattern_length:
                # Pattern found at position (i - j)
                matches.append(i - j)
                
                # Check if we should stop
                if not find_all or (max_matches and len(matches) >= max_matches):
                    break
                
                # Continue searching
                j = self.lps[j - 1]
            
            elif i < text_length and search_text[i] != self.pattern[j]:
                # Mismatch after j matches
                if j != 0:
                    # Use LPS to skip unnecessary comparisons
                    j = self.lps[j - 1]
                else:
                    i += 1
        
        return matches
    
    def get_match_context(
        self,
        text: str,
        position: int,
        context_length: int = 20
    ) -> Tuple[str, str, str]:
        """
        Get context around a match position.
        
        Args:
            text: The original text
            position: Starting position of the match
            context_length: Number of characters to include before/after
            
        Returns:
            Tuple of (matched_text, context_before, context_after)
        """
        matched_text = text[position:position + self.pattern_length]
        
        context_start = max(0, position - context_length)
        context_before = text[context_start:position]
        
        context_end = min(len(text), position + self.pattern_length + context_length)
        context_after = text[position + self.pattern_length:context_end]
        
        return matched_text, context_before, context_after


# ============================================================================
# MAIN ALGORITHM FUNCTION
# ============================================================================

def kmp_search(input_data: KMPInput) -> KMPOutput:
    """
    Perform KMP string matching to find pattern in text.
    
    Args:
        input_data: KMPInput with text, pattern, and search options
        
    Returns:
        KMPOutput with match positions and statistics
        
    Example:
        >>> result = kmp_search(KMPInput(
        ...     text="ABABDABACDABABCABAB",
        ...     pattern="ABABCABAB"
        ... ))
        >>> print(result.found)
        True
        >>> print(result.positions)
        [10]
    """
    start_time = time.time()
    
    # Initialize KMP matcher
    matcher = KMPMatcher(
        pattern=input_data.pattern,
        case_sensitive=input_data.case_sensitive
    )
    
    # Perform search
    positions = matcher.search(
        text=input_data.text,
        find_all=input_data.find_all,
        max_matches=input_data.max_matches
    )
    
    # Build detailed match information
    matches = []
    for pos in positions:
        matched_text, context_before, context_after = matcher.get_match_context(
            text=input_data.text,
            position=pos,
            context_length=20
        )
        matches.append(MatchInfo(
            position=pos,
            matched_text=matched_text,
            context_before=context_before,
            context_after=context_after
        ))
    
    # Calculate execution time
    execution_time_ms = (time.time() - start_time) * 1000
    
    # Calculate efficiency metrics
    text_length = len(input_data.text)
    pattern_length = len(input_data.pattern)
    
    # Naive algorithm would make (n - m + 1) * m comparisons worst case
    naive_comparisons = max(1, (text_length - pattern_length + 1) * pattern_length)
    efficiency_gain = ((naive_comparisons - matcher.comparisons_made) / naive_comparisons) * 100
    
    return KMPOutput(
        found=len(positions) > 0,
        occurrences=len(positions),
        positions=positions,
        matches=matches,
        pattern_length=pattern_length,
        text_length=text_length,
        search_parameters={
            "case_sensitive": input_data.case_sensitive,
            "find_all": input_data.find_all,
            "max_matches": input_data.max_matches
        },
        performance_metrics={
            "comparisons_made": matcher.comparisons_made,
            "characters_scanned": matcher.characters_scanned,
            "efficiency_gain_vs_naive": round(efficiency_gain, 2),
            "comparisons_per_character": round(matcher.comparisons_made / text_length, 2) if text_length > 0 else 0
        },
        execution_time_ms=round(execution_time_ms, 2)
    )


# ============================================================================
# ADDITIONAL UTILITY FUNCTIONS
# ============================================================================

def kmp_replace(
    text: str,
    pattern: str,
    replacement: str,
    case_sensitive: bool = True,
    max_replacements: Optional[int] = None
) -> Tuple[str, int]:
    """
    Replace all occurrences of pattern with replacement text.
    
    Args:
        text: The text to perform replacements in
        pattern: The pattern to search for
        replacement: The text to replace pattern with
        case_sensitive: Whether matching should be case-sensitive
        max_replacements: Maximum number of replacements to perform
        
    Returns:
        Tuple of (modified_text, replacement_count)
        
    Example:
        >>> text = "Hello world, hello universe"
        >>> result, count = kmp_replace(text, "hello", "goodbye", case_sensitive=False)
        >>> print(result)
        'goodbye world, goodbye universe'
        >>> print(count)
        2
    """
    matcher = KMPMatcher(pattern, case_sensitive)
    positions = matcher.search(text, find_all=True, max_matches=max_replacements)
    
    if not positions:
        return text, 0
    
    # Build result string by replacing from right to left (to preserve positions)
    result = text
    for pos in reversed(positions):
        result = result[:pos] + replacement + result[pos + len(pattern):]
    
    return result, len(positions)


def kmp_count(
    text: str,
    pattern: str,
    case_sensitive: bool = True
) -> int:
    """
    Count occurrences of pattern in text.
    
    Args:
        text: The text to search in
        pattern: The pattern to count
        case_sensitive: Whether matching should be case-sensitive
        
    Returns:
        Number of occurrences
        
    Example:
        >>> count = kmp_count("banana", "ana")
        >>> print(count)
        2
    """
    matcher = KMPMatcher(pattern, case_sensitive)
    positions = matcher.search(text, find_all=True)
    return len(positions)
