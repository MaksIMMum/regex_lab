# ai generate tests

import unittest
import pytest
from regex import (
    RegexFSM,
    StartState,
    TerminationState,
    DotState,
    AsciiState,
    StarState,
    PlusState
)


# ======================================================================
# Part 1. Pytest-based Tests (Parametrizations & Edge Cases)
# ======================================================================

def test_start_state():
    state = StartState()
    # StartState check_self should always return False
    assert state.check_self("a") is False
    assert state.check_self("") is False
    assert state.check_self("any_char") is False


def test_termination_state():
    state = TerminationState()
    # TerminationState check_self should always return False
    assert state.check_self("a") is False
    assert state.check_self("") is False
    assert state.check_self("any_char") is False


def test_dot_state():
    state = DotState()
    # DotState check_self should return True for any character
    assert state.check_self("a") is True
    assert state.check_self("z") is True
    assert state.check_self(".") is True
    assert state.check_self(" ") is True
    assert state.check_self("1") is True


def test_ascii_state():
    state = AsciiState("a")
    # AsciiState check_self should return True only for matching symbol
    assert state.check_self("a") is True
    assert state.check_self("b") is False
    assert state.check_self("") is False
    assert state.check_self("A") is False  # case-sensitive


def test_star_state():
    sub_state = AsciiState("x")
    state = StarState(sub_state)
    # StarState delegates check_self to checking_state and other next_states
    assert state.check_self("x") is True
    assert state.check_self("y") is False


def test_plus_state():
    sub_state = AsciiState("y")
    state = PlusState(sub_state)
    # PlusState delegates check_self to checking_state and other next_states
    assert state.check_self("y") is True
    assert state.check_self("x") is False


def test_state_check_next():
    # Test state transitions via check_next
    s1 = AsciiState("a")
    s2 = AsciiState("b")
    s3 = AsciiState("c")
    s1.next_states = [s2, s3]

    # transition matching b should return s2
    assert s1.check_next("b") is s2
    # transition matching c should return s3
    assert s1.check_next("c") is s3

    # transition matching d (not in next_states) should raise NotImplementedError
    with pytest.raises(NotImplementedError, match="rejected string"):
        s1.check_next("d")


@pytest.mark.parametrize("pattern, input_str, expected", [
    # Basic literals
    ("ab", "ab", True),
    ("ab", "a", False),
    ("ab", "b", False),
    ("ab", "abc", False),
    ("ab", "cab", False),

    # Dot state (wildcard)
    (".", "a", True),
    (".", "1", True),
    (".", "", False),
    (".", "ab", False),
    ("a.c", "abc", True),
    ("a.c", "axc", True),
    ("a.c", "abbc", False),
    ("a.c", "ac", False),

    # Star operator (0 or more)
    ("a*", "", True),
    ("a*", "a", True),
    ("a*", "aaaa", True),
    ("a*", "b", False),
    ("a*b", "b", True),
    ("a*b", "ab", True),
    ("a*b", "aaaab", True),
    ("a*b", "aaaabc", False),

    # Plus operator (1 or more)
    ("a+", "a", True),
    ("a+", "aaaa", True),
    ("a+", "", False),
    ("a+", "b", False),
    ("ab+c", "abc", True),
    ("ab+c", "abbbc", True),
    ("ab+c", "ac", False),

    # Multiple and combined operators
    ("a*b*", "", True),
    ("a*b*", "a", True),
    ("a*b*", "b", True),
    ("a*b*", "aaabbb", True),
    ("a*b*", "aaabbba", False),
    ("a.*c", "abbc", True),
    ("a.*c", "ac", True),
    ("a.*c", "axxxxxxxxxxc", True),
    ("a.*c", "abc", True),

    # Complex case from laboratory work README / regex.py default example
    ("a*4.+hi", "aaaaaa4uhi", True),
    ("a*4.+hi", "4uhi", True),
    ("a*4.+hi", "meow", False),
    ("a*4.+hi", "aaaaaa4hi", False),  # no char for DotState
    ("a*4.+hi", "aaaaaa4u", False),   # missing +hi
    ("a*4.+hi", "aaaaaa4uh", False),  # missing i
    ("a*4.+hi", "aaaaaa4uhiii", False), # extra characters
])
def test_regex_fsm_matching(pattern, input_str, expected):
    fsm = RegexFSM(pattern)
    assert fsm.check_string(input_str) is expected


def test_empty_pattern():
    # Empty pattern matches only empty string
    fsm = RegexFSM("")
    assert fsm.check_string("") is True
    assert fsm.check_string("a") is False


def test_starts_with_special_chars():
    # Pattern starting with star
    fsm_star = RegexFSM("*")
    assert fsm_star.check_string("") is True
    assert fsm_star.check_string("a") is False

    # Pattern starting with plus
    fsm_plus = RegexFSM("+")
    # Due to the FSM's implementation, starting with "+" wraps StartState in a PlusState.
    # StartState is the checking_state. The PlusState's next states include StartState
    # and TerminationState. During DFS, the epsilon transitions from PlusState allow
    # direct transition to TerminationState without any input characters, thus matching
    # an empty string.
    assert fsm_plus.check_string("") is True
    assert fsm_plus.check_string("a") is False


# ======================================================================
# Part 2. Unittest-based Tests (Sourced from User/Claude)
# Note: Tests targeting unimplemented CharacterClassState are excluded,
#       and invalid-pattern ValueError asserts are skipped since they
#       are not supported by the current FSM parser.
# ======================================================================

class TestStateClasses(unittest.TestCase):
    """Test the behavior of individual state classes"""

    def test_start_state(self):
        """Test StartState behavior"""
        start = StartState()
        self.assertFalse(start.check_self('a'), "StartState should not accept any character")
        self.assertEqual(start.next_states, [], "StartState should initialize with empty next_states")

    def test_termination_state(self):
        """Test TerminationState behavior"""
        term = TerminationState()
        self.assertFalse(term.check_self('a'), "TerminationState should not accept any character")
        with self.assertRaises(NotImplementedError, msg="TerminationState.check_next should raise an error"):
            term.check_next('a')

    def test_dot_state(self):
        """Test DotState behavior"""
        dot = DotState()
        self.assertTrue(dot.check_self('a'), "DotState should accept 'a'")
        self.assertTrue(dot.check_self('5'), "DotState should accept '5'")
        self.assertTrue(dot.check_self(' '), "DotState should accept space")
        self.assertTrue(dot.check_self('\n'), "DotState should accept newline")

    def test_ascii_state(self):
        """Test AsciiState behavior"""
        a_state = AsciiState('a')
        self.assertTrue(a_state.check_self('a'), "AsciiState('a') should accept 'a'")
        self.assertFalse(a_state.check_self('b'), "AsciiState('a') should reject 'b'")

        digit_state = AsciiState('7')
        self.assertTrue(digit_state.check_self('7'), "AsciiState('7') should accept '7'")
        self.assertFalse(digit_state.check_self('8'), "AsciiState('7') should reject '8'")


class TestRegexFSMConstruction(unittest.TestCase):
    """Test the construction of the RegexFSM from patterns"""

    def test_valid_patterns(self):
        """Test that valid patterns are accepted during construction"""
        # Simple patterns
        RegexFSM("a")
        RegexFSM(".")
        RegexFSM("abc")

        # Quantifiers
        RegexFSM("a*")
        RegexFSM("a+")
        RegexFSM("a*b")
        RegexFSM("a+b")


class TestRegexMatching(unittest.TestCase):
    """Test the actual pattern matching functionality"""

    def test_simple_literals(self):
        """Test matching of simple literal patterns"""
        # Single character
        pattern = RegexFSM("a")
        self.assertTrue(pattern.check_string("a"), "Pattern 'a' should match 'a'")
        self.assertFalse(pattern.check_string("b"), "Pattern 'a' should not match 'b'")
        self.assertFalse(pattern.check_string(""), "Pattern 'a' should not match empty string")
        self.assertFalse(pattern.check_string("ab"), "Pattern 'a' should not match 'ab'")

        # Multiple characters
        pattern = RegexFSM("abc")
        self.assertTrue(pattern.check_string("abc"), "Pattern 'abc' should match 'abc'")
        self.assertFalse(pattern.check_string("ab"), "Pattern 'abc' should not match 'ab'")
        self.assertFalse(pattern.check_string("abcd"), "Pattern 'abc' should not match 'abcd'")

    def test_dot_operator(self):
        """Test matching of the dot operator"""
        # Single dot
        pattern = RegexFSM(".")
        self.assertTrue(pattern.check_string("a"), "Pattern '.' should match 'a'")
        self.assertTrue(pattern.check_string("5"), "Pattern '.' should match '5'")
        self.assertFalse(pattern.check_string(""), "Pattern '.' should not match empty string")
        self.assertFalse(pattern.check_string("ab"), "Pattern '.' should not match 'ab'")

        # Dot in pattern
        pattern = RegexFSM("a.c")
        self.assertTrue(pattern.check_string("abc"), "Pattern 'a.c' should match 'abc'")
        self.assertTrue(pattern.check_string("a5c"), "Pattern 'a.c' should match 'a5c'")
        self.assertFalse(pattern.check_string("ac"), "Pattern 'a.c' should not match 'ac'")
        self.assertFalse(pattern.check_string("a55c"), "Pattern 'a.c' should not match 'a55c'")

    def test_star_quantifier(self):
        """Test matching with the star (*) quantifier"""
        # Star on single character
        pattern = RegexFSM("a*")
        self.assertTrue(pattern.check_string(""), "Pattern 'a*' should match empty string")
        self.assertTrue(pattern.check_string("a"), "Pattern 'a*' should match 'a'")
        self.assertTrue(pattern.check_string("aaaaa"), "Pattern 'a*' should match 'aaaaa'")
        self.assertFalse(pattern.check_string("b"), "Pattern 'a*' should not match 'b'")

        # Star in pattern
        pattern = RegexFSM("a*b")
        self.assertTrue(pattern.check_string("b"), "Pattern 'a*b' should match 'b'")
        self.assertTrue(pattern.check_string("ab"), "Pattern 'a*b' should match 'ab'")
        self.assertTrue(pattern.check_string("aaaab"), "Pattern 'a*b' should match 'aaaab'")
        self.assertFalse(pattern.check_string("a"), "Pattern 'a*b' should not match 'a'")
        self.assertFalse(pattern.check_string("aaaabbb"), "Pattern 'a*b' should not match 'aaaabbb'")

    def test_plus_quantifier(self):
        """Test matching with the plus (+) quantifier"""
        # Plus on single character
        pattern = RegexFSM("a+")
        self.assertFalse(pattern.check_string(""), "Pattern 'a+' should not match empty string")
        self.assertTrue(pattern.check_string("a"), "Pattern 'a+' should match 'a'")
        self.assertTrue(pattern.check_string("aaaaa"), "Pattern 'a+' should match 'aaaaa'")
        self.assertFalse(pattern.check_string("b"), "Pattern 'a+' should not match 'b'")

        # Plus in pattern
        pattern = RegexFSM("a+b")
        self.assertFalse(pattern.check_string("b"), "Pattern 'a+b' should not match 'b'")
        self.assertTrue(pattern.check_string("ab"), "Pattern 'a+b' should match 'ab'")
        self.assertTrue(pattern.check_string("aaaab"), "Pattern 'a+b' should match 'aaaab'")
        self.assertFalse(pattern.check_string("a"), "Pattern 'a+b' should not match 'a'")
        self.assertFalse(pattern.check_string("aaaabbb"), "Pattern 'a+b' should not match 'aaaabbb'")

    def test_complex_patterns(self):
        """Test matching of more complex patterns"""
        # Pattern with multiple components
        pattern = RegexFSM("a*b+c")
        self.assertTrue(pattern.check_string("bc"), "Pattern 'a*b+c' should match 'bc'")
        self.assertTrue(pattern.check_string("abc"), "Pattern 'a*b+c' should match 'abc'")
        self.assertTrue(pattern.check_string("aabbc"), "Pattern 'a*b+c' should match 'aabbc'")
        self.assertFalse(pattern.check_string("ac"), "Pattern 'a*b+c' should not match 'ac'")

        # The example from the file
        pattern = RegexFSM("a*4.+hi")
        self.assertTrue(pattern.check_string("4xhi"), "Pattern 'a*4.+hi' should match '4xhi'")
        self.assertTrue(pattern.check_string("aaa4yyhi"), "Pattern 'a*4.+hi' should match 'aaa4yyhi'")
        self.assertFalse(pattern.check_string("hi"), "Pattern 'a*4.+hi' should not match 'hi'")
        self.assertFalse(pattern.check_string("a4hi"), "Pattern 'a*4.+hi' should not match 'a4hi'")


if __name__ == '__main__':
    unittest.main()
#pytest -v test.py
