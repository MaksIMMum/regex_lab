from __future__ import annotations
from abc import ABC, abstractmethod


class State(ABC):
    @abstractmethod
    def __init__(self) -> None:
        pass

    @abstractmethod
    def check_self(self, char: str) -> bool:
        pass

    def check_next(self, next_char: str) -> State | Exception:
        for state in self.next_states:
            if state.check_self(next_char):
                return state
        raise NotImplementedError("rejected string")


class StartState(State):
    def __init__(self):
        self.next_states: list[State] = []

    def check_self(self, char: str) -> bool:
        return False


class TerminationState(State):
    def __init__(self):
        self.next_states: list[State] = []

    def check_self(self, char: str) -> bool:
        return False


class DotState(State):
    def __init__(self):
        self.next_states: list[State] = []

    def check_self(self, char: str) -> bool:
        return True


class AsciiState(State):
    def __init__(self, symbol: str) -> None:
        self.next_states: list[State] = []
        self.curr_sym: str = symbol

    def check_self(self, curr_char: str) -> bool:
        return curr_char == self.curr_sym


class StarState(State):
    def __init__(self, checking_state: State):
        self.checking_state: State = checking_state
        self.next_states: list[State] = [checking_state]

    def check_self(self, char: str) -> bool:
        return self.checking_state.check_self(char)


class PlusState(State):
    def __init__(self, checking_state: State):
        self.checking_state: State = checking_state
        self.next_states: list[State] = [checking_state]

    def check_self(self, char: str) -> bool:
        return self.checking_state.check_self(char)


class RegexFSM:
    def __init__(self, regex_expr: str) -> None:
        self.curr_state: State = StartState()

        prev_state = self.curr_state
        tmp_next_state = self.curr_state

        for char in regex_expr:
            tmp_next_state = self.__init_next_state(char, prev_state, tmp_next_state)
            prev_state.next_states.append(tmp_next_state)
            prev_state = tmp_next_state

        prev_state.next_states.append(TerminationState())

    def __init_next_state(
        self, next_token: str, prev_state: State, tmp_next_state: State
    ) -> State:
        new_state = None

        match next_token:
            case next_token if next_token == ".":
                new_state = DotState()

            case next_token if next_token == "*":
                new_state = StarState(tmp_next_state)

            case next_token if next_token == "+":
                new_state = PlusState(tmp_next_state)

            case next_token if next_token.isascii():
                new_state = AsciiState(next_token)

            case next_token if next_token in ("*", "+"):
                if isinstance(tmp_next_state, StartState):
                    raise ValueError(f"'{next_token}' cannot appear at the start of a pattern")
            case _:
                raise AttributeError("Character is not supported")

        return new_state

    def check_string(self, input_str: str) -> bool:
        def epsilon_closure(states: set[State]) -> set[State]:
            closure: set[State] = set(states)
            stack: list[State] = list(states)

            while stack:
                state = stack.pop()

                if isinstance(state, StartState):
                    eps_targets = state.next_states

                elif isinstance(state, StarState):
                    eps_targets = state.next_states

                elif isinstance(state, PlusState):
                    eps_targets = [
                        ns for ns in state.next_states
                        if ns is not state.checking_state
                    ]

                else:
                    eps_targets = []

                for ns in state.next_states:
                    if isinstance(ns, StarState) and ns not in closure:
                        closure.add(ns)
                        stack.append(ns)

                for ns in eps_targets:
                    if ns not in closure:
                        closure.add(ns)
                        stack.append(ns)

            return closure

        current: set[State] = epsilon_closure({self.curr_state})

        for char in input_str:
            next_states: set[State] = set()

            for state in current:
                if isinstance(state, (AsciiState, DotState)):
                    if state.check_self(char):
                        next_states.update(state.next_states)

                elif isinstance(state, (StarState, PlusState)):
                    if state.checking_state.check_self(char):
                        next_states.add(state)

            current = epsilon_closure(next_states)

        return any(isinstance(s, TerminationState) for s in current)


if __name__ == "__main__":
    regex_pattern = "a*4.+hi"
    regex_compiled = RegexFSM(regex_pattern)

    print(regex_compiled.check_string("aaaaaa4uhi"))
    print(regex_compiled.check_string("4uhi"))
    print(regex_compiled.check_string("meow"))
