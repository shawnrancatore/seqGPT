"""
TokenLang: A formal language for generating arbitrary token sequences.
Uses the same token space for coding the sequence and the output, with simple syntax.
"""

from typing import List, Dict, Any, Union, Optional

class TokenLang:
    """
    TokenLang interpreter for generating arbitrary token sequences.

    This language:
    - Uses the same token space for code and output
    - Has a simple syntax with single-token commands and whitespace delimiters
    - Can generate arbitrary sequences of characters/tokens
    - Includes a continuation token ("...")
    """

    def __init__(self):
        self.output = []
        self.variables = {}
        self.stack = []

    def evaluate(self, code: str, max_length: int = 50) -> List[Any]:
        """
        Evaluate TokenLang code and return the generated sequence.

        Args:
            code: TokenLang code as a string
            max_length: Maximum length of the output sequence

        Returns:
            The generated token sequence
        """
        self.output = []
        self.variables = {}
        self.stack = []

        # Preprocess the code: remove comments and normalize spacing
        lines = []
        for line in code.split('\n'):
            if '#' in line:
                line = line[:line.index('#')].strip()
            if line:
                lines.append(line)

        tokens = ' '.join(lines).split()

        # Process tokens
        i = 0
        while i < len(tokens) and len(self.output) < max_length:
            token = tokens[i]

            # Variable assignment: x = value
            if i + 2 < len(tokens) and tokens[i + 1] == '=':
                var_name = token
                value = self._evaluate_token(tokens[i + 2])
                self.variables[var_name] = value
                i += 3  # Skip variable, =, and value

            # Loop: ! count body ;
            elif token == '!':
                if i + 1 < len(tokens):
                    # Get iteration count
                    count_token = tokens[i + 1]
                    if count_token in self.variables:
                        iterations = self._to_int(self.variables[count_token])
                    else:
                        iterations = self._to_int(count_token)

                    # Find loop body (between ! count and matching ;)
                    loop_start = i + 2  # Start after ! count
                    loop_end = loop_start
                    nesting = 1  # Track nested loops

                    while loop_end < len(tokens) and nesting > 0:
                        if tokens[loop_end] == '!':
                            nesting += 1
                        elif tokens[loop_end] == ';':
                            nesting -= 1

                        if nesting > 0:  # Only increment if we haven't reached the end
                            loop_end += 1

                    if loop_end >= len(tokens):
                        # No matching end found, skip this loop
                        i = len(tokens)  # Force exit of main loop
                        continue

                    # Extract loop body tokens
                    loop_body = tokens[loop_start:loop_end]

                    # Execute the loop iterations
                    for _ in range(iterations):
                        if len(self.output) >= max_length:
                            break

                        # Recursively evaluate the loop body in the current context
                        self.evaluate_tokens(loop_body)

                    i = loop_end + 1  # Skip to after the loop end (;)

                else:
                    i += 1  # Skip malformed loop

            # Continuation token
            elif token == '...':
                if self.output:
                    pattern = self.output[-min(10, len(self.output)):]
                    while len(self.output) < max_length:
                        for pattern_token in pattern:
                            if len(self.output) < max_length:
                                self.output.append(pattern_token)
                            else:
                                break
                i += 1

            # Addition
            elif token == '+':
                if len(self.stack) >= 2:
                    b = self._to_int(self.stack.pop())
                    a = self._to_int(self.stack.pop())
                    result = a + b
                    self.stack.append(result)
                    self.output.append(result)
                i += 1

            # Subtraction
            elif token == '-':
                if len(self.stack) >= 2:
                    b = self._to_int(self.stack.pop())
                    a = self._to_int(self.stack.pop())
                    result = a - b
                    self.stack.append(result)
                    self.output.append(result)
                i += 1

            # Multiplication
            elif token == '*':
                if len(self.stack) >= 2:
                    b = self._to_int(self.stack.pop())
                    a = self._to_int(self.stack.pop())
                    result = a * b
                    self.stack.append(result)
                    self.output.append(result)
                i += 1

            # Division
            elif token == '/':
                if len(self.stack) >= 2:
                    b = self._to_int(self.stack.pop())
                    a = self._to_int(self.stack.pop())
                    result = a // b if b != 0 else 0
                    self.stack.append(result)
                    self.output.append(result)
                i += 1

            # Modulo
            elif token == '%':
                if len(self.stack) >= 2:
                    b = self._to_int(self.stack.pop())
                    a = self._to_int(self.stack.pop())
                    result = a % b if b != 0 else 0
                    self.stack.append(result)
                    self.output.append(result)
                i += 1

            # Variable reference
            elif token in self.variables:
                value = self.variables[token]
                self.stack.append(value)
                self.output.append(value)
                i += 1

            # Loop end or other special tokens to ignore
            elif token == ';':
                i += 1

            # Regular token (not a variable or command)
            else:
                value = self._evaluate_token(token)
                self.stack.append(value)
                self.output.append(value)
                i += 1

        return self.output[:max_length]

    def evaluate_tokens(self, tokens: List[str]) -> None:
        """Evaluate a list of tokens in the current context."""
        i = 0
        while i < len(tokens) and len(self.output) < 1000:  # Safe limit
            token = tokens[i]

            # Variable assignment
            if i + 2 < len(tokens) and tokens[i + 1] == '=':
                var_name = token

                # Special case for calculation in assignment: now supports pattern "var = operand1 operand2 operator"
                if i + 4 < len(tokens) and tokens[i + 4] in ['+', '-', '*', '/', '%']:
                    operand1 = self._evaluate_token(tokens[i + 2])
                    operand2 = self._evaluate_token(tokens[i + 3])
                    op = tokens[i + 4]
                    a_val = self._to_int(operand1)
                    b_val = self._to_int(operand2)
                    if op == '+':
                        value = a_val + b_val
                    elif op == '-':
                        value = a_val - b_val
                    elif op == '*':
                        value = a_val * b_val
                    elif op == '/':
                        value = a_val // b_val if b_val != 0 else 0
                    elif op == '%':
                        value = a_val % b_val if b_val != 0 else 0

                    # If the original variable was a letter, convert the result back to a letter.
                    orig = self.variables.get(var_name, None)
                    if isinstance(orig, str) and len(orig) == 1 and orig.isalpha():
                        if orig.isupper():
                            value = chr((value - 1) % 26 + ord('A'))
                        else:
                            value = chr((value - 1) % 26 + ord('a'))

                    self.variables[var_name] = value
                    i += 5  # Skip var, =, operand1, operand2, operator
                else:
                    # Simple assignment: var = value
                    value = self._evaluate_token(tokens[i + 2])
                    self.variables[var_name] = value
                    i += 3  # Skip var, =, value

            # Arithmetic operations
            elif token in ['+', '-', '*', '/', '%'] and len(self.stack) >= 2:
                b = self._to_int(self.stack.pop())
                a = self._to_int(self.stack.pop())

                if token == '+':
                    result = a + b
                elif token == '-':
                    result = a - b
                elif token == '*':
                    result = a * b
                elif token == '/':
                    result = a // b if b != 0 else 0
                elif token == '%':
                    result = a % b if b != 0 else 0

                self.stack.append(result)
                self.output.append(result)
                i += 1

            # Variable reference
            elif token in self.variables:
                value = self.variables[token]
                self.stack.append(value)
                self.output.append(value)
                i += 1

            # Skip control tokens
            elif token in ['!', ';']:
                i += 1

            # Regular token
            else:
                value = self._evaluate_token(token)
                self.stack.append(value)
                self.output.append(value)
                i += 1

    def _evaluate_token(self, token: str) -> Any:
        """Evaluate a token, handling variable references."""
        if token in self.variables:
            return self.variables[token]

        # Try to convert to int if it looks like a number
        try:
            if token.isdigit() or (token[0] == '-' and token[1:].isdigit()):
                return int(token)
        except (ValueError, IndexError):
            pass

        # Otherwise, return the token as is
        return token

    def _to_int(self, value: Any) -> int:
        """Convert a value to an integer if possible."""
        if isinstance(value, int):
            return value

        try:
            return int(value)
        except (ValueError, TypeError):
            # Try to convert a token to an integer based on its position in the alphabet
            if isinstance(value, str) and len(value) == 1:
                if value.isupper():
                    return ord(value) - ord('A') + 1  # A=1, B=2, ...
                elif value.islower():
                    return ord(value) - ord('a') + 1  # a=1, b=2, ...

            return 0  # Default value


def generate_token_pattern(rule: str, length: int = 50) -> List[Any]:
    """
    Generate a token sequence based on a TokenLang program.

    Args:
        rule: The TokenLang program as a string
        length: Maximum number of tokens to generate

    Returns:
        The generated token sequence
    """
    interpreter = TokenLang()
    return interpreter.evaluate(rule, max_length=length)


def create_token_patterns():
    """Create a library of example token patterns."""
    patterns = {
        # Simple sequence
        "token_simple": {
            "rule": "A B C D E F G H I J",
            "language": "token_lang"
        },

        # Arithmetic operations
        "token_arithmetic": {
            "rule": "2 3 + 4 * 5 +",
            "language": "token_lang"
        },

        # Loop construct
        "token_loop": {
            "rule": "! 5 a b ; c d",
            "language": "token_lang"
        },

        # Continuation token
        "token_continuation": {
            "rule": "X Y ...",
            "language": "token_lang"
        },

        # Fibonacci sequence - correctly implemented
        "token_fibonacci": {
            "rule": """
            # Fibonacci sequence
            1 1
            a = 1
            b = 1
            ! 10
              c = a
              a = b
              b = c a +
              b
            ;
            """,
            "language": "token_lang"
        },

        # Alphabet sequence - correctly implemented
        "token_alphabet": {
            "rule": """
            # Alphabet sequence
            A
            curr = A
            ! 25
              curr = curr 1 +
              curr
            ;
            """,
            "language": "token_lang"
        },

        # Count by twos
        "token_count_by_twos": {
            "rule": """
            # Count by twos
            0
            curr = 0
            ! 10
              curr = curr 2 +
              curr
            ;
            """,
            "language": "token_lang"
        },
    }

    return patterns


if __name__ == "__main__":
    # Test some patterns to verify
    print("Testing TokenLang implementation...")

    # Test Fibonacci sequence
    print("\nFibonacci sequence:")
    fibonacci_rule = """
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
    result = generate_token_pattern(fibonacci_rule, length=15)
    print(result)

    # Test alphabet sequence
    print("\nAlphabet sequence:")
    alphabet_rule = """
    A
    curr = A
    ! 25
      curr = curr 1 +
      curr
    ;
    """
    result = generate_token_pattern(alphabet_rule, length=15)
    print(result)
