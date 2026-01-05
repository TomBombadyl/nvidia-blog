"""
Date Filter Extractor Module
Extracts date-related filters from user queries for BigQuery date filtering.
"""

import re
import logging
from typing import Dict, Optional, Tuple
from datetime import datetime, timedelta, timezone
from dateutil import parser as date_parser

logger = logging.getLogger(__name__)


class DateFilterExtractor:
    """Extracts date filters from user queries."""
    
    def __init__(self):
        """Initialize date filter extractor."""
        self.month_names = {
            'january': 1, 'jan': 1,
            'february': 2, 'feb': 2,
            'march': 3, 'mar': 3,
            'april': 4, 'apr': 4,
            'may': 5,
            'june': 6, 'jun': 6,
            'july': 7, 'jul': 7,
            'august': 8, 'aug': 8,
            'september': 9, 'sep': 9, 'sept': 9,
            'october': 10, 'oct': 10,
            'november': 11, 'nov': 11,
            'december': 12, 'dec': 12
        }
        
        # Relative date keywords
        self.relative_keywords = {
            'latest': 30,  # last 30 days
            'recent': 30,
            'newest': 30,
            'new': 30,
            'today': 1,
            'this month': None,  # Special handling
            'this week': 7,
            'last week': 7,
            'last month': 30,
            'last 30 days': 30,
            'last 7 days': 7,
            'last week': 7
        }
    
    def extract_date_filters(
        self, query: str
    ) -> Optional[Dict[str, datetime]]:
        """
        Extract date filters from query string.
        
        Args:
            query: User query string
            
        Returns:
            Dictionary with 'start_date' and 'end_date' keys, or None if no dates found
        """
        query_lower = query.lower()
        
        # Try relative dates first (latest, recent, etc.)
        relative_filter = self._extract_relative_date(query_lower)
        if relative_filter:
            return relative_filter
        
        # Try month/year patterns
        month_year_filter = self._extract_month_year(query_lower, query)
        if month_year_filter:
            return month_year_filter
        
        # Try specific date patterns
        specific_date_filter = self._extract_specific_date(query)
        if specific_date_filter:
            return specific_date_filter
        
        # Try date range patterns
        date_range_filter = self._extract_date_range(query_lower, query)
        if date_range_filter:
            return date_range_filter
        
        return None
    
    def _extract_relative_date(
        self, query_lower: str
    ) -> Optional[Dict[str, datetime]]:
        """Extract relative date filters (latest, recent, etc.)."""
        now = datetime.now(timezone.utc)
        
        # Check for "this month"
        if 'this month' in query_lower:
            start_date = datetime(now.year, now.month, 1, tzinfo=timezone.utc)
            # End date is first day of next month
            if now.month == 12:
                end_date = datetime(now.year + 1, 1, 1, tzinfo=timezone.utc)
            else:
                end_date = datetime(now.year, now.month + 1, 1, tzinfo=timezone.utc)
            return {'start_date': start_date, 'end_date': end_date}
        
        # Check for other relative keywords
        for keyword, days in self.relative_keywords.items():
            if keyword in query_lower and days is not None:
                end_date = now
                start_date = now - timedelta(days=days)
                return {'start_date': start_date, 'end_date': end_date}
        
        return None
    
    def _extract_month_year(
        self, query_lower: str, query_original: str
    ) -> Optional[Dict[str, datetime]]:
        """Extract month/year patterns like 'December 2025'."""
        # Pattern: month name + year (e.g., "December 2025", "Dec 2025")
        for month_name, month_num in self.month_names.items():
            pattern = rf'\b{month_name}\s+(\d{{4}})\b'
            match = re.search(pattern, query_lower)
            if match:
                year = int(match.group(1))
                start_date = datetime(year, month_num, 1, tzinfo=timezone.utc)
                # End date is first day of next month
                if month_num == 12:
                    end_date = datetime(year + 1, 1, 1, tzinfo=timezone.utc)
                else:
                    end_date = datetime(year, month_num + 1, 1, tzinfo=timezone.utc)
                return {'start_date': start_date, 'end_date': end_date}
        
        # Pattern: MM/YYYY or MM-YYYY (e.g., "12/2025", "12-2025")
        pattern = r'\b(\d{1,2})[/-](\d{4})\b'
        match = re.search(pattern, query_original)
        if match:
            month = int(match.group(1))
            year = int(match.group(2))
            if 1 <= month <= 12:
                start_date = datetime(year, month, 1, tzinfo=timezone.utc)
                if month == 12:
                    end_date = datetime(year + 1, 1, 1, tzinfo=timezone.utc)
                else:
                    end_date = datetime(year, month + 1, 1, tzinfo=timezone.utc)
                return {'start_date': start_date, 'end_date': end_date}
        
        return None
    
    def _extract_specific_date(
        self, query: str
    ) -> Optional[Dict[str, datetime]]:
        """Extract specific date patterns like 'December 18, 2025'."""
        # Try to parse common date formats
        date_patterns = [
            r'\b(January|February|March|April|May|June|July|August|September|October|November|December)\s+(\d{1,2}),?\s+(\d{4})\b',
            r'\b(Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)\.?\s+(\d{1,2}),?\s+(\d{4})\b',
            r'\b(\d{4})-(\d{1,2})-(\d{1,2})\b',  # ISO format: 2025-12-18
        ]
        
        for pattern in date_patterns:
            match = re.search(pattern, query, re.IGNORECASE)
            if match:
                try:
                    # Try to parse the matched date string
                    date_str = match.group(0)
                    parsed_date = date_parser.parse(date_str, default=datetime.now(timezone.utc))
                    
                    if parsed_date.tzinfo is None:
                        parsed_date = parsed_date.replace(tzinfo=timezone.utc)
                    
                    # Return single day range
                    start_date = parsed_date.replace(hour=0, minute=0, second=0, microsecond=0)
                    end_date = start_date + timedelta(days=1)
                    return {'start_date': start_date, 'end_date': end_date}
                except (ValueError, AttributeError):
                    continue
        
        return None
    
    def _extract_date_range(
        self, query_lower: str, query_original: str
    ) -> Optional[Dict[str, datetime]]:
        """Extract date range patterns like 'from December to January'."""
        # Pattern: "from [month] to [month]" or "between [month] and [month]"
        range_patterns = [
            r'from\s+(\w+)\s+to\s+(\w+)',
            r'between\s+(\w+)\s+and\s+(\w+)',
        ]
        
        for pattern in range_patterns:
            match = re.search(pattern, query_lower)
            if match:
                month1_str = match.group(1)
                month2_str = match.group(2)
                
                # Try to extract year from query
                year_match = re.search(r'\b(20\d{2})\b', query_original)
                year = int(year_match.group(1)) if year_match else datetime.now().year
                
                month1 = self.month_names.get(month1_str)
                month2 = self.month_names.get(month2_str)
                
                if month1 and month2:
                    start_date = datetime(year, month1, 1, tzinfo=timezone.utc)
                    # End date is first day of month after month2
                    if month2 == 12:
                        end_date = datetime(year + 1, 1, 1, tzinfo=timezone.utc)
                    else:
                        end_date = datetime(year, month2 + 1, 1, tzinfo=timezone.utc)
                    return {'start_date': start_date, 'end_date': end_date}
        
        return None

