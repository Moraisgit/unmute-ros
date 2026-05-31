"""Print the domestic-robot system prompt sent to the model.

Run from the unmute-ros/ root:

    uv run python -m unmute.scripts.print_robot_prompt
"""

from unmute.llm.system_prompt import DomesticRobotInstructions


def main() -> None:
    print(DomesticRobotInstructions().make_system_prompt())


if __name__ == "__main__":
    main()