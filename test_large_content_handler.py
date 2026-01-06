"""
Test script for LargeContentHandler

Tests the large content handler with various content sizes and scenarios.
"""

import sys
import os

# Add the bigquery/mcp_server directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'bigquery', 'mcp_server'))

from large_content_handler import LargeContentHandler, ChunkingConfig, create_batches_for_embedding


def create_mock_chunks(num_chunks: int, tokens_per_chunk: int = 1000) -> list:
    """Create mock chunks for testing."""
    chunks = []
    for i in range(num_chunks):
        # Create text with approximately the right number of tokens
        # Rough estimate: 4 characters per token
        text_length = tokens_per_chunk * 4
        text = f"Chunk {i}: " + "x" * (text_length - 10)
        chunks.append({
            'chunk_index': i,
            'text': text,
            'item_id': 'test_item'
        })
    return chunks


def test_token_counting():
    """Test token counting functionality."""
    print("=" * 60)
    print("Test 1: Token Counting")
    print("=" * 60)
    
    handler = LargeContentHandler()
    
    test_cases = [
        ("Short text", "This is a short text", 5),  # ~5 tokens
        ("Medium text", "x" * 400, 100),  # ~100 tokens (400 chars / 4)
        ("Long text", "x" * 4000, 1000),  # ~1000 tokens
    ]
    
    for name, text, expected_tokens in test_cases:
        tokens = handler.count_tokens(text)
        print(f"{name}: {len(text)} chars -> {tokens} tokens (expected ~{expected_tokens})")
        assert tokens > 0, f"Token count should be positive for {name}"
    
    print("[PASS] Token counting test passed\n")


def test_chunking_config_selection():
    """Test adaptive chunking config selection."""
    print("=" * 60)
    print("Test 2: Chunking Config Selection")
    print("=" * 60)
    
    handler = LargeContentHandler()
    
    test_cases = [
        (50 * 1024, "normal"),  # 50KB - normal
        (150 * 1024, "normal"),  # 150KB - normal
        (250 * 1024, "large"),   # 250KB - large
        (300 * 1024, "large"),   # 300KB - large
        (450 * 1024, "very_large"),  # 450KB - very large
        (600 * 1024, "very_large"),  # 600KB - very large
    ]
    
    for size_bytes, expected_type in test_cases:
        config = handler.determine_chunking_config(size_bytes)
        size_kb = size_bytes / 1024
        
        if expected_type == "normal":
            assert config.chunk_size_tokens == 1000, f"Normal size should use 1000 tokens"
            print(f"[OK] {size_kb:.1f}KB -> Normal config (1000 tokens/chunk)")
        elif expected_type == "large":
            assert config.chunk_size_tokens == 750, f"Large size should use 750 tokens"
            print(f"[OK] {size_kb:.1f}KB -> Large config (750 tokens/chunk)")
        elif expected_type == "very_large":
            assert config.chunk_size_tokens == 500, f"Very large should use 500 tokens"
            print(f"[OK] {size_kb:.1f}KB -> Very large config (500 tokens/chunk)")
    
    print("[PASS] Chunking config selection test passed\n")


def test_batch_creation_normal():
    """Test batch creation with normal-sized content."""
    print("=" * 60)
    print("Test 3: Batch Creation - Normal Content")
    print("=" * 60)
    
    handler = LargeContentHandler()
    
    # Create chunks that should fit in one batch
    # 10 chunks * 1000 tokens = 10,000 tokens (under 20K limit)
    chunks = create_mock_chunks(10, tokens_per_chunk=1000)
    
    batches = handler.create_embedding_batches(chunks)
    
    print(f"Created {len(batches)} batches from {len(chunks)} chunks")
    assert len(batches) == 1, "Normal content should create 1 batch"
    
    total_tokens = sum(handler.count_tokens(c['text']) for c in batches[0])
    print(f"Total tokens in batch: {total_tokens} (limit: 20000)")
    assert total_tokens <= 20000, "Batch should not exceed token limit"
    
    print("[PASS] Normal content batch creation test passed\n")


def test_batch_creation_large():
    """Test batch creation with large content (simulating the 83KB post)."""
    print("=" * 60)
    print("Test 4: Batch Creation - Large Content (83KB Post Scenario)")
    print("=" * 60)
    
    handler = LargeContentHandler()
    
    # Simulate the 83KB post scenario:
    # 18 chunks with ~1333 tokens each = ~23,996 tokens total
    # This should create 2 batches
    chunks = create_mock_chunks(18, tokens_per_chunk=1333)
    
    batches = handler.create_embedding_batches(chunks)
    
    print(f"Created {len(batches)} batches from {len(chunks)} chunks")
    assert len(batches) >= 2, "Large content should create multiple batches"
    
    # Verify each batch is under the limit
    for i, batch in enumerate(batches):
        total_tokens = sum(handler.count_tokens(c['text']) for c in batch)
        print(f"Batch {i+1}: {len(batch)} chunks, {total_tokens} tokens")
        assert total_tokens <= 20000, f"Batch {i+1} exceeds token limit"
    
    print("[PASS] Large content batch creation test passed\n")


def test_batch_creation_very_large():
    """Test batch creation with very large content."""
    print("=" * 60)
    print("Test 5: Batch Creation - Very Large Content")
    print("=" * 60)
    
    handler = LargeContentHandler()
    
    # Create many chunks that will require multiple batches
    # 50 chunks * 1000 tokens = 50,000 tokens (needs 3 batches)
    chunks = create_mock_chunks(50, tokens_per_chunk=1000)
    
    batches = handler.create_embedding_batches(chunks)
    
    print(f"Created {len(batches)} batches from {len(chunks)} chunks")
    assert len(batches) >= 3, "Very large content should create multiple batches"
    
    # Verify all batches are under limit
    for i, batch in enumerate(batches):
        total_tokens = sum(handler.count_tokens(c['text']) for c in batch)
        print(f"Batch {i+1}: {len(batch)} chunks, {total_tokens} tokens")
        assert total_tokens <= 20000, f"Batch {i+1} exceeds token limit"
    
    print("[PASS] Very large content batch creation test passed\n")


def test_chunk_validation():
    """Test chunk validation functionality."""
    print("=" * 60)
    print("Test 6: Chunk Validation")
    print("=" * 60)
    
    handler = LargeContentHandler()
    config = ChunkingConfig(chunk_size_tokens=1000, overlap_tokens=200)
    
    # Test with valid chunks
    valid_chunks = create_mock_chunks(5, tokens_per_chunk=1000)
    validation = handler.validate_chunks(valid_chunks, config)
    
    print(f"Valid chunks: {validation['total_chunks']} chunks, {validation['total_tokens']} tokens")
    assert validation['is_valid'], "Valid chunks should pass validation"
    assert len(validation['issues']) == 0, "Valid chunks should have no issues"
    
    # Test with chunks that exceed limit
    invalid_chunks = create_mock_chunks(2, tokens_per_chunk=3000)  # Exceeds 2048 limit
    validation = handler.validate_chunks(invalid_chunks, config)
    
    print(f"Invalid chunks: {validation['total_chunks']} chunks, {len(validation['issues'])} issues")
    assert not validation['is_valid'], "Invalid chunks should fail validation"
    assert len(validation['issues']) > 0, "Invalid chunks should have issues"
    
    print("[PASS] Chunk validation test passed\n")


def test_convenience_function():
    """Test the convenience function."""
    print("=" * 60)
    print("Test 7: Convenience Function")
    print("=" * 60)
    
    chunks = create_mock_chunks(20, tokens_per_chunk=1000)
    
    # Test without content size
    batches = create_batches_for_embedding(chunks)
    print(f"Without content size: {len(batches)} batches from {len(chunks)} chunks")
    assert len(batches) >= 1, "Should create at least one batch"
    
    # Test with content size (triggers adaptive chunking)
    content_size = 300 * 1024  # 300KB - should use large config
    batches = create_batches_for_embedding(chunks, content_size_bytes=content_size)
    print(f"With content size ({content_size/1024:.1f}KB): {len(batches)} batches")
    assert len(batches) >= 1, "Should create at least one batch"
    
    print("[PASS] Convenience function test passed\n")


def test_edge_cases():
    """Test edge cases."""
    print("=" * 60)
    print("Test 8: Edge Cases")
    print("=" * 60)
    
    handler = LargeContentHandler()
    
    # Test with empty chunks list
    batches = handler.create_embedding_batches([])
    assert len(batches) == 0, "Empty chunks should create no batches"
    print("[OK] Empty chunks handled correctly")
    
    # Test with single chunk
    chunks = create_mock_chunks(1, tokens_per_chunk=500)
    batches = handler.create_embedding_batches(chunks)
    assert len(batches) == 1, "Single chunk should create one batch"
    print("[OK] Single chunk handled correctly")
    
    # Test with chunk exactly at limit
    chunks = create_mock_chunks(20, tokens_per_chunk=1000)  # Exactly 20K tokens
    batches = handler.create_embedding_batches(chunks)
    assert len(batches) == 1, "Exactly 20K tokens should fit in one batch"
    print("[OK] Chunks at limit handled correctly")
    
    print("[PASS] Edge cases test passed\n")


def run_all_tests():
    """Run all tests."""
    print("\n" + "=" * 60)
    print("LARGE CONTENT HANDLER - TEST SUITE")
    print("=" * 60 + "\n")
    
    tests = [
        test_token_counting,
        test_chunking_config_selection,
        test_batch_creation_normal,
        test_batch_creation_large,
        test_batch_creation_very_large,
        test_chunk_validation,
        test_convenience_function,
        test_edge_cases,
    ]
    
    passed = 0
    failed = 0
    
    for test in tests:
        try:
            test()
            passed += 1
        except AssertionError as e:
            print(f"[FAIL] Test failed: {e}\n")
            failed += 1
        except Exception as e:
            print(f"[ERROR] Test error: {e}\n")
            import traceback
            traceback.print_exc()
            failed += 1
    
    print("=" * 60)
    print(f"TEST RESULTS: {passed} passed, {failed} failed")
    print("=" * 60)
    
    if failed == 0:
        print("\n[SUCCESS] ALL TESTS PASSED!")
        return True
    else:
        print(f"\n[FAILED] {failed} TEST(S) FAILED")
        return False


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)
