"""Print the domestic-robot system prompt sent to the model.

Run from the unmute-ros/ root:

    uv run python -m unmute.scripts.print_robot_prompt
    uv run python -m unmute.scripts.print_robot_prompt --object-set sim
    uv run python -m unmute.scripts.print_robot_prompt --object-set sim \
        --rooms "bedroom,kitchen" --surfaces "bed,counter top"

--rooms/--surfaces (comma-separated) scope the place vocabulary to a specific
world, mirroring what the bridge forwards from the backend's world_vocab(); omit
them to see the static robot_world tuples.

See also print_robot_combinations.sh, which dumps every combination at once.
"""

import argparse

from unmute.llm.system_prompt import DomesticRobotInstructions


def _split(value: str | None) -> list[str] | None:
    if not value:
        return None
    return [item.strip() for item in value.split(",") if item.strip()] or None


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--object-set", choices=["sim", "real"], default=None)
    parser.add_argument("--rooms", default=None, help="comma-separated room vocabulary")
    parser.add_argument(
        "--surfaces", default=None, help="comma-separated surface vocabulary"
    )
    args = parser.parse_args()

    instructions = DomesticRobotInstructions(
        object_set=args.object_set,
        rooms=_split(args.rooms),
        surfaces=_split(args.surfaces),
    )
    print(instructions.make_system_prompt())


if __name__ == "__main__":
    main()
