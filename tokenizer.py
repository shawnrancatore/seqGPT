import re
import sys # Import sys module
from typing import List, Dict, Union, Any, Tuple

# Increase the limit for integer string conversion (if needed)
# try:
#     sys.set_int_max_str_digits(10000) # Set to a large value, adjust as needed
# except AttributeError:
#     print("Warning: sys.set_int_max_str_digits() not available. May encounter issues with very large numbers.")


class NumberStreamTokenizer:
    """
    Tokenizer for number streams and pattern rules.
    Handles conversion between tokens and numerical IDs.
    """

    def __init__(self, max_number: int = 100):
        # Define token types
        self.operators = ['+', '-', '*', '/', '%', '**', '#', '[', ']', ',', '(', ')']
        self.digits = [str(i) for i in range(10)]
        self.letters = [chr(ord('A') + i) for i in range(26)]  # A-Z for variables
        self.special_tokens = ['<PAD>', '<START>', '<END>', '<UNK>', '<INF>'] # Added <INF> token

        # Add number tokens for efficient number representation
        self.number_tokens = [f"NUM_{i}" for i in range(max_number)]

        # Create vocabulary
        self.vocabulary = (
                self.special_tokens +
                self.digits +
                self.letters +
                self.operators +
                self.number_tokens
        )

        # Create mappings
        self.token_to_id = {token: idx for idx, token in enumerate(self.vocabulary)}
        self.id_to_token = {idx: token for idx, token in enumerate(self.vocabulary)}

        # Store max number for number token generation
        self.max_number = max_number

    def tokenize_rule(self, rule_string: str) -> List[str]:
        """
        Tokenize a pattern rule string into a list of tokens.

        Parameters:
        -----------
        rule_string : str
            The rule string to tokenize

        Returns:
        --------
        list
            A list of tokens
        """
        # Standardize rule to uppercase for variable consistency during tokenization
        rule_string = rule_string.upper()

        # Add spaces around operators and brackets for easier tokenization
        rule = re.sub(r'([\+\-\*/%\(\)\[\],#])', r' \1 ', rule_string)

        # Handle ** operator (which was split by the previous regex)
        rule = rule.replace(' * *', ' ** ') # Correctly reassemble **


        # Split by whitespace and filter out empty strings
        tokens = [token for token in rule.split() if token]

        # Process numerical tokens
        processed_tokens = []
        i = 0
        while i < len(tokens):
            token = tokens[i]

            # Check if token is a number
            if token.isdigit() or (token[0] == '-' and token[1:].isdigit()):
                num = int(token)
                if 0 <= num < self.max_number:
                    processed_tokens.append(f"NUM_{num}")
                else:
                    # For numbers outside our range, tokenize digit by digit
                    for digit in token:
                        processed_tokens.append(digit)
            else:
                processed_tokens.append(token)

            i += 1

        return processed_tokens

    def tokenize_sequence(self, sequence: List[Union[int, float]]) -> List[str]:
        """
        Tokenize a number sequence into a list of tokens.

        Parameters:
        -----------
        sequence : list
            A list of numbers to tokenize

        Returns:
        --------
        list
            A list of tokens
        """
        tokenized_seq = []

        for num in sequence:
            # Check if we can represent this as a simple number token
            if isinstance(num, int) and 0 <= num < self.max_number:
                tokenized_seq.append(f"NUM_{num}")
            else:
                # Otherwise, tokenize digit by digit
                if num == float('inf'): # Handle infinity case
                    tokenized_seq.append('<INF>')
                    continue # Skip digit-by-digit tokenization for infinity
                num_str = str(num) # Proceed with digit tokenization for regular numbers
                for digit in num_str:
                    tokenized_seq.append(digit)

            tokenized_seq.append(',')  # Add separator between numbers

        return tokenized_seq[:-1]  # Remove last comma

    def numericalize(self, tokens: List[str]) -> List[int]:
        """
        Convert tokens to numerical IDs.

        Parameters:
        -----------
        tokens : list
            A list of tokens

        Returns:
        --------
        list
            A list of token IDs
        """
        return [self.token_to_id.get(token, self.token_to_id['<UNK>']) for token in tokens]

    def denumericalize(self, ids: List[int]) -> List[str]:
        """
        Convert numerical IDs back to tokens.

        Parameters:
        -----------
        ids : list
            A list of token IDs

        Returns:
        --------
        list
            A list of tokens
        """
        return [self.id_to_token.get(idx, '<UNK>') for idx in ids]

    def reconstruct_sequence(self, tokens: List[str], complete_only: bool = True) -> List[Union[int, float]]:
        """
        Reconstruct a number sequence from tokens.

        Parameters:
        -----------
        tokens : list
            A list of tokens
        complete_only : bool
            If True, only return completely parsed numbers (discard partial ones at the end)

        Returns:
        --------
        list
            A list of numbers
        """
        numbers = []
        current_digits = []

        for token in tokens:
            if token == '<INF>': # Handle infinity token during reconstruction
                numbers.append(float('inf')) # Reconstruct as float('inf')
            if token.startswith('NUM_'):
                if current_digits:
                    # Process any accumulated digits before handling the NUM token
                    num_str = ''.join(current_digits)
                    numbers.append(int(num_str) if num_str.isdigit() else float(num_str))
                    current_digits = []

                # Extract number from the NUM token
                num = int(token.split('_')[1])
                numbers.append(num)

            elif token in self.digits or token == '.' or token == '-':
                current_digits.append(token)

            elif token == ',':
                if current_digits:
                    num_str = ''.join(current_digits)
                    numbers.append(int(num_str) if num_str.isdigit() else float(num_str))
                    current_digits = []

        # Handle any remaining digits if we're keeping incomplete numbers
        if current_digits and not complete_only:
            num_str = ''.join(current_digits)
            numbers.append(int(num_str) if num_str.isdigit() else float(num_str))

        return numbers

    def get_partial_number(self, tokens: List[str]) -> Tuple[str, bool]:
        """
        Extract the partial number at the end of a token sequence.

        Parameters:
        -----------
        tokens : list
            A list of tokens

        Returns:
        --------
        tuple
            (partial_number_string, is_complete_number)
        """
        # Find the last comma to identify where the last number starts
        last_comma_idx = -1
        for i in range(len(tokens) - 1, -1, -1):
            if tokens[i] == ',':
                last_comma_idx = i
                break

        # Extract tokens after the last comma
        if last_comma_idx == -1:
            # No comma found, check the entire token list
            partial_tokens = tokens
        else:
            partial_tokens = tokens[last_comma_idx + 1:]

        # Reconstruct the partial number
        partial_str = ""
        for token in partial_tokens:
            if token.startswith('NUM_'):
                # Complete number token
                return token.split('_')[1], True  # Return the number and flag it as complete
            elif token in self.digits or token == '.' or token == '-':
                partial_str += token

        # Check if the partial_str is actually a complete number
        # For this pattern, we know numbers should be integers
        if partial_str.isdigit() or (partial_str.startswith('-') and partial_str[1:].isdigit()):
            return partial_str, True  # It's a complete number

        return partial_str, False  # It's a partial number
    @property
    def vocab_size(self) -> int:
        """Get the size of the vocabulary."""
        return len(self.vocabulary)


def test_tokenizer():
    """Test the tokenizer functionality."""
    tokenizer = NumberStreamTokenizer(max_number=100)

    # Test rule tokenization
    rules = [
        "A + 1",
        "(A * 2) % 10",
        "[1, 2, 3] # 2",
        "2 ** (A % 6)"
    ]

    print("=== Rule Tokenization Tests ===")
    for rule in rules:
        tokens = tokenizer.tokenize_rule(rule)
        ids = tokenizer.numericalize(tokens)
        recovered_tokens = tokenizer.denumericalize(ids)

        print(f"Rule: {rule}")
        print(f"Tokens: {tokens}")
        print(f"IDs: {ids}")
        print(f"Recovered tokens: {recovered_tokens}")
        print("-" * 50)

    # Test sequence tokenization
    sequences = [
        [1, 2, 3, 4, 5],
        [10, 20, 30, 40, 50],
        [0, 1, 0, 1, 0],
        [101, 202, 303, 404],  # Beyond max_number
        [float('inf')] # Test infinity tokenization
     ]

    print("\n=== Sequence Tokenization Tests ===")
    for seq in sequences:
        tokens = tokenizer.tokenize_sequence(seq)
        ids = tokenizer.numericalize(tokens)
        recovered_tokens = tokenizer.denumericalize(ids)
        reconstructed_seq = tokenizer.reconstruct_sequence(recovered_tokens)

        print(f"Sequence: {seq}")
        print(f"Tokens: {tokens}")
        print(f"Reconstructed: {reconstructed_seq}")
        print("-" * 50)

    print(f"Vocabulary size: {tokenizer.vocab_size}")


if __name__ == "__main__":
    test_tokenizer()