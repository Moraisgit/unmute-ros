"""Print the GBNF grammar enforced on the domestic-robot LLM output.

Run from the unmute-ros/ root:

    uv run python -m unmute.scripts.print_robot_grammar
    uv run python -m unmute.scripts.print_robot_grammar --object-set sim
    uv run python -m unmute.scripts.print_robot_grammar --object-set sim \
        --rooms "bedroom,kitchen" --surfaces "bed,counter top"

--rooms/--surfaces (comma-separated) scope the place vocabulary to a specific
world, mirroring what the bridge forwards from the backend's world_vocab(); omit
them to see the static robot_world tuples.

See also print_robot_combinations.sh, which dumps every combination at once.
"""

import argparse

from unmute.llm.robot_world import objects_for
from unmute.llm.system_prompt import ACTIONS, _render_domestic_robot_grammar


def _split(value: str | None) -> tuple[str, ...] | None:
    if not value:
        return None
    items = tuple(item.strip() for item in value.split(",") if item.strip())
    return items or None


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--object-set", choices=["sim", "real"], default=None)
    parser.add_argument("--rooms", default=None, help="comma-separated room vocabulary")
    parser.add_argument(
        "--surfaces", default=None, help="comma-separated surface vocabulary"
    )
    args = parser.parse_args()

    print(
        _render_domestic_robot_grammar(
            ACTIONS,
            objects_for(args.object_set),
            _split(args.rooms),
            _split(args.surfaces),
        )
    )


if __name__ == "__main__":
    main()
