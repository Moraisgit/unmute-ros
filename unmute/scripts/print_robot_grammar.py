"""Print the GBNF grammar enforced on the domestic-robot LLM output.

Run from the unmute-ros/ root:

    uv run python -m unmute.scripts.print_robot_grammar
"""

from unmute.llm.system_prompt import ACTIONS, _render_domestic_robot_grammar


def main() -> None:
    print(_render_domestic_robot_grammar(ACTIONS))


if __name__ == "__main__":
    main()
