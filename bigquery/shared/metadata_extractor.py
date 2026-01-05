"""
Metadata Extractor for Cleaned Text Files
Extracts publication date, title, and source URI from cleaned text headers.
The cleaned text files have metadata prepended in this format:
  Publication Date: {date}
  Title: {title}
  Source: {source}

  ---

  {content}
"""

import re
import logging
from typing import Dict, Optional
from datetime import datetime, timezone

logger = logging.getLogger(__name__)


def extract_metadata_from_text(cleaned_text: str) -> Dict[str, Optional[str]]:
    """
    Extract metadata from cleaned text header.

    Args:
        cleaned_text: Text content with optional metadata header

    Returns:
        Dictionary with:
        - publication_date: Parsed datetime or None
        - title: Title string or None
        - source_uri: Source URI string or None
    """
    metadata = {
        "publication_date": None,
        "title": None,
        "source_uri": None
    }

    if not cleaned_text:
        return metadata

    # Look for metadata header pattern
    # Format: "Publication Date: ...\nTitle: ...\nSource: ...\n\n---\n\n"
    header_match = re.match(
        r'^Publication Date:\s*(.+?)\n'
        r'Title:\s*(.+?)\n'
        r'Source:\s*(.+?)\n'
        r'\n---\n\n',
        cleaned_text,
        re.MULTILINE | re.DOTALL
    )

    if header_match:
        pub_date_str = header_match.group(1).strip()
        title = header_match.group(2).strip()
        source = header_match.group(3).strip()

        # Parse publication date - try multiple date formats
        # Handle 'Z' timezone indicator (UTC) specially
        if pub_date_str.endswith('Z'):
            # ISO 8601 with Z (UTC) - replace Z with +00:00 for parsing
            pub_date_str_parsed = pub_date_str[:-1] + '+00:00'
            try:
                publication_date = datetime.strptime(
                    pub_date_str_parsed, "%Y-%m-%dT%H:%M:%S%z"
                )
            except ValueError:
                # Try without timezone
                try:
                    dt = datetime.strptime(
                        pub_date_str[:-1], "%Y-%m-%dT%H:%M:%S"
                    )
                    publication_date = dt.replace(
                        tzinfo=timezone.utc
                    )
                except ValueError:
                    publication_date = None
        else:
            # Try other date formats
            date_formats = [
                # RFC 2822 with timezone name
                ("%a, %d %b %Y %H:%M:%S %Z", True),
                ("%a, %d %b %Y %H:%M:%S %z", False),  # RFC 2822 with timezone
                ("%Y-%m-%dT%H:%M:%S%z", False),  # ISO 8601 with timezone
                ("%Y-%m-%d %H:%M:%S", True),           # Simple (assume UTC)
                ("%Y-%m-%d", True),                    # Date only (assume UTC)
            ]

            publication_date = None
            for fmt, assume_utc in date_formats:
                try:
                    publication_date = datetime.strptime(pub_date_str, fmt)
                    if assume_utc and publication_date.tzinfo is None:
                        # Make timezone-aware (UTC)
                        publication_date = publication_date.replace(
                            tzinfo=timezone.utc
                        )
                    break
                except ValueError:
                    continue

        if publication_date:
            metadata["publication_date"] = publication_date
        else:
            logger.warning(
                f"Could not parse publication date: {pub_date_str}"
            )

        # Store title
        if title and title != "None":
            metadata["title"] = title

        # Extract source URI from source string
        if source and source != "None":
            # Try to extract URL if present in source
            url_match = re.search(r'https?://[^\s]+', source)
            if url_match:
                metadata["source_uri"] = url_match.group(0)
            else:
                # Store source name for reference
                metadata["source_uri"] = source

    return metadata


def extract_source_uri_from_item_id(
    item_id: str, feed: str
) -> Optional[str]:
    """
    Try to reconstruct source URI from item_id.

    Args:
        item_id: Sanitized item ID (may contain URL fragments)
        feed: Feed name (dev or official)

    Returns:
        Reconstructed URI or None
    """
    # Item IDs are sanitized URLs, so we can try to reconstruct
    # Format: "https___developer.nvidia.com_blog__p_12345"
    if item_id.startswith("https___") or item_id.startswith("http___"):
        # Replace underscores with appropriate characters
        uri = item_id.replace("___", "://").replace("__", "/")
        uri = uri.replace("_", ".")
        # Try to fix common patterns
        uri = uri.replace("://.", "://")
        return uri

    return None
