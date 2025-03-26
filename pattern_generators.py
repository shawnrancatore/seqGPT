import re
from typing import List, Dict, Union, Any
import operator

# Import the TokenLang implementation
from token_lang import generate_token_pattern, create_token_patterns


class ExpressionEvaluator:
    """A more robust expression evaluator with proper operator precedence."""

    def __init__(self):
        # Define operations with proper precedence levels
        self.operations = {
            '+': (1, operator.add),
            '-': (1, operator.sub),
            '*': (2, operator.mul),
            '/': (2, operator.truediv),
            '%': (2, operator.mod),
            '**': (3, operator.pow),
            '#': (3, self.repeat_operator),
        }

    def repeat_operator(self, a, b):
        """Handles the repeat operator # for both lists and scalars."""
        if isinstance(a, list):
            # If 'a' is a list, repeat it 'b' times
            return a * int(b)
        elif isinstance(b, list):
            # If 'b' is a list, it's not a typical operation
            raise ValueError("Right operand of # cannot be a list")
        else:
            # If both are scalars, use power operation
            return a ** b

    def tokenize_expression(self, expression: str) -> List[str]:
        """Tokenize an expression into a list of tokens."""
        # Add spaces around operators and parentheses for easy splitting
        expression = re.sub(r'(\*\*|[+\-*/%()\[\],#])', r' \1 ', expression)
        # Split by whitespace and filter out empty strings
        return [token for token in expression.split() if token]

    def parse_expression(self, tokens: List[str], variables: Dict[str, Union[int, float]]) -> Any:
        """Parse and evaluate an expression with proper operator precedence."""

        def parse_factor():
            token = tokens.pop(0)
            if token == '(':
                value = parse_expression()
                if tokens.pop(0) != ')':
                    raise ValueError("Missing closing parenthesis")
                return value
            elif token == '[':
                # Parse a list literal
                list_values = []
                if tokens and tokens[0] != ']':
                    list_values.append(parse_expression())
                    while tokens and tokens[0] == ',':
                        tokens.pop(0)  # Consume the comma
                        if tokens[0] != ']':
                            list_values.append(parse_expression())
                if not tokens or tokens.pop(0) != ']':
                    raise ValueError("Missing closing bracket for list")
                return list_values
            elif token.isalpha():
                # Handle variables
                return variables.get(token, 0)
            else:
                # Handle numbers
                try:
                    return float(token)
                except ValueError:
                    raise ValueError(f"Unknown token: {token}")

        def parse_term():
            left = parse_factor()
            while tokens and tokens[0] in ('**', '#'):
                op = tokens.pop(0)
                right = parse_factor()
                _, func = self.operations[op]
                left = func(left, right)
            return left

        def parse_product():
            left = parse_term()
            while tokens and tokens[0] in ('*', '/', '%'):
                op = tokens.pop(0)
                right = parse_term()
                _, func = self.operations[op]
                left = func(left, right)
            return left

        def parse_expression():
            left = parse_product()
            while tokens and tokens[0] in ('+', '-'):
                op = tokens.pop(0)
                right = parse_product()
                _, func = self.operations[op]
                left = func(left, right)
            return left

        # Create a copy of tokens to avoid modifying the original
        tokens_copy = tokens.copy()
        return parse_expression()


def generate_pattern(rule: str, length: int = 50, language: str = "basic_numeric") -> List[Union[int, float, str]]:
    """
    Generate a sequence of values based on the specified rule and language.

    Parameters:
    ----------
    rule : str or dict
        The rule string in the formal language syntax or a dictionary with 'rule' and 'language' keys
    length : int, optional
        The number of elements to generate (default 50)
    language : str, optional
        The language to use for pattern generation ("basic_numeric" or "token_lang")

    Returns:
    --------
    list
        A list of values generated according to the rule
    """
    # For dictionary-type rules with explicit language specification
    if isinstance(rule, dict):
        language = rule.get('language', language)
        rule = rule.get('rule', '')
    
    # Use the appropriate language interpreter
    if language == "token_lang":
        return generate_token_pattern(rule, length)
    
    # Default to "basic_numeric" language
    sequence = []
    variables = {'A': 0, 'B': 0, 'C': 0}  # Default variables

    evaluator = ExpressionEvaluator()
    tokens = evaluator.tokenize_expression(rule)

    # Initialize the first two values to start the Fibonacci sequence correctly
    if 'B' in rule or 'C' in rule:  # Only initialize if B or C are used
        sequence.append(1)  # Initial value for i = 0
        sequence.append(1)  # Initial value for i = 1

    for i in range(length):
        variables['A'] = i  # Update the index variable

        # For rules that involve B as a cumulative value
        if 'B' in rule and i > 0:
            variables['B'] = sequence[i - 1]

        # For rules that involve C as a secondary cumulative/historical value
        if 'C' in rule and i > 1:
            variables['C'] = sequence[i - 2]

        try:
            # Evaluate the rule with current variable values
            value = evaluator.parse_expression(tokens.copy(), variables)

            # Handle special cases for list results
            if isinstance(value, list):
                if i < len(value):
                    value = value[i % len(value)]
                else:
                    value = value[i % len(value)]

            # Ensure integer results when appropriate
            if isinstance(value, float) and value.is_integer():
                value = int(value)

            sequence.append(value)

        except Exception as e:
            print(f"Error evaluating rule '{rule}' at step {i}: {e}")
            sequence.append(0)  # Fallback value

    return sequence


def create_pattern_library():
    """Create a library of patterns with escalating complexity."""
    patterns = {
        # Basic patterns (basic_numeric language)
        "linear": "A + 1",
        "even_numbers": "A * 2",
        "odd_numbers": "A * 2 + 1",
        "constant": "5",

        # Modular arithmetic
        "modulo_3": "A % 3",
        "modulo_5": "A % 5",
        "modulo_with_offset": "(A % 3) + 2",

        # List patterns
        "alternating": "[0, 1]",
        "triple_alternating": "[0, 1, 2]",
        "repeated_list": "[1, 2, 3] # 2",

        # Arithmetic combinations
        "quadratic": "A * A",
        "polynomial": "A * A + 2 * A + 1",
        "complex_modulo": "(A * 2) % 10",

        # Exponential patterns
        "powers_of_2": "2 ** (A % 6)",
        "powers_of_3_mod": "(3 ** A) % 10",

        # Fibonacci-like patterns (requires B and C)
        "fibonacci_type": "B + C",  # B is previous, C is second previous

        # Composite patterns
        "complex_composite": "(A * 3 + 2) % 7",
        "nested_operations": "(A % 3) * (A % 2 + 1)",

        # Advanced patterns
        "conditional_alternating": "(A % 2) * (A + 1) + (1 - (A % 2)) * (A - 1)",
        "complex_exponential": "2 ** ((A % 4) + 1)",
    }
    
    # Add TokenLang patterns
    token_patterns = create_token_patterns()
    patterns.update(token_patterns)
    
    return patterns


def test_pattern_generator():
    """Test the pattern generator with various rules."""
    test_patterns = create_pattern_library()
    
    # Test basic numeric patterns
    print("TESTING BASIC NUMERIC PATTERNS:")
    basic_numeric_patterns = {k: v for k, v in test_patterns.items() 
                             if isinstance(v, str) or (isinstance(v, dict) and v.get('language') != 'token_lang')}
    
    for name, rule in list(basic_numeric_patterns.items())[:5]:  # Test first 5 patterns
        try:
            sequence = generate_pattern(rule, length=10)
            
            # Get the rule string for display
            rule_str = rule.get('rule', rule) if isinstance(rule, dict) else rule
            
            print(f"Pattern: {name} ({rule_str})")
            print(f"Sequence: {sequence}")
            print("-" * 50)
        except Exception as e:
            print(f"Error testing pattern '{name}': {e}")
    
    # Test TokenLang patterns
    print("\nTESTING TOKEN LANGUAGE PATTERNS:")
    token_patterns = {k: v for k, v in test_patterns.items() 
                      if isinstance(v, dict) and v.get('language') == 'token_lang'}
    
    for name, pattern_info in token_patterns.items():
        try:
            sequence = generate_pattern(pattern_info, length=15)
            
            # Handle rule display for multiline rules
            rule_str = pattern_info.get('rule', '')
            if '\n' in rule_str:
                rule_display = "multi-line rule"
            else:
                rule_display = rule_str
                
            print(f"Pattern: {name} ({rule_display})")
            print(f"Sequence: {sequence}")
            print("-" * 50)
        except Exception as e:
            print(f"Error testing pattern '{name}': {e}")


if __name__ == "__main__":
    test_pattern_generator()
