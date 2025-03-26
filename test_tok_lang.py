"""
Simple test script for the fixed TokenLang implementation.
"""

from token_lang import generate_token_pattern

def test_simple_patterns():
    print("Testing basic patterns:")
    
    # Simple sequence
    simple = "A B C D E"
    print(f"\nSimple sequence: {simple}")
    result = generate_token_pattern(simple, length=10)
    print(f"Result: {result}")
    
    # Arithmetic
    arithmetic = "2 3 + 4 *"
    print(f"\nArithmetic: {arithmetic}")
    result = generate_token_pattern(arithmetic, length=10)
    print(f"Result: {result}")
    
    # Variable assignment
    variables = "x = 5 x x x"
    print(f"\nVariables: {variables}")
    result = generate_token_pattern(variables, length=10)
    print(f"Result: {result}")
    
    # Loop
    loop = "! 3 A B ; C"
    print(f"\nLoop: {loop}")
    result = generate_token_pattern(loop, length=10)
    print(f"Result: {result}")
    
    # Token mapping
    mapping = "A B +"
    print(f"\nToken mapping: {mapping}")
    result = generate_token_pattern(mapping, length=10)
    print(f"Result: {result}")

def test_fibonacci():
    print("\nTesting Fibonacci sequence:")
    
    fibonacci = """
    1 1
    a = 1
    b = 1
    ! 10
      c = a
      a = b
      b = c a +
      b
    ;
    """
    
    result = generate_token_pattern(fibonacci, length=15)
    print(f"Result: {result}")
    
    # Verify
    expected = [1, 1, 2, 3, 5, 8, 13, 21, 34, 55, 89, 144, 233]
    correct = all(a == b for a, b in zip(result[:len(expected)], expected))
    print(f"Correct Fibonacci sequence: {'Yes' if correct else 'No'}")

def test_alphabet():
    print("\nTesting alphabet sequence:")
    
    alphabet = """
    A
    curr = A
    ! 25
      curr = curr 1 +
      curr
    ;
    """
    
    result = generate_token_pattern(alphabet, length=15)
    print(f"Result: {result}")
    
    # Verify - should be A through P (15 letters)
    expected = [chr(ord('A') + i) for i in range(15)]
    correct = all(a == b for a, b in zip(result, expected))
    print(f"Correct alphabet sequence: {'Yes' if correct else 'No'}")

def test_count_by_twos():
    print("\nTesting count by twos:")
    
    count = """
    0
    curr = 0
    ! 10
      curr = curr 2 +
      curr
    ;
    """
    
    result = generate_token_pattern(count, length=15)
    print(f"Result: {result}")
    
    # Verify - should be 0, 2, 4, ..., 20
    expected = [i*2 for i in range(11)]
    correct = all(a == b for a, b in zip(result, expected))
    print(f"Correct count sequence: {'Yes' if correct else 'No'}")

def run_all_tests():
    """Run all tests to verify the fixed implementation."""
    print("Running tests for fixed TokenLang implementation")
    print("==============================================")
    
    test_simple_patterns()
    test_fibonacci()
    test_alphabet()
    test_count_by_twos()
    
    print("\nAll tests completed.")

if __name__ == "__main__":
    run_all_tests()
