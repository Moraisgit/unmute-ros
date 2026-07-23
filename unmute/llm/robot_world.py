"""Vocabulary for the domestic-robot planner: rooms, surfaces, and objects.

Edit these tuples to extend the robot's world. The prompt's KNOWN LOCATIONS /
KNOWN OBJECTS blocks and the guided-decoding grammar are all derived from them.

Two object sets are kept deliberately separate:
  * AI2THOR_OBJECTS - the simulator's perception vocabulary.
  * REAL_OBJECTS    - what the physical robot can actually detect.
The active set is chosen by the ACTION_SIMULATOR env var (the same toggle
run_unmute_bridge.sh uses to launch the simulator): "true" -> simulator objects,
anything else -> real-life objects.
"""

import os

# --- Locations -------------------------------------------------------------
ROOMS: tuple[str, ...] = (
    "living room",
    "bedroom",
    "kitchen",
    "dining room",
    "entrance",
)
SURFACES: tuple[str, ...] = (
    "desk",
    "cabinet",
    "lower shelf",
    "middle shelf",
    "top shelf",
    "counter",
    "kitchen table",
    "dining table",
)
PLACES: tuple[str, ...] = ROOMS + SURFACES

# --- Objects ---------------------------------------------------------------
# Simulator (AI2-THOR) perception vocabulary.
AI2THOR_OBJECTS: tuple[str, ...] = (
    "alarm clock",
    "aluminum foil",
    "apple",
    "apple sliced",
    "arm chair",
    "baseball bat",
    "basket ball",
    "bathtub",
    "bathtub basin",
    "bed",
    "blinds",
    "book",
    "boots",
    "bottle",
    "bowl",
    "box",
    "bread",
    "bread sliced",
    "butter knife",
    "cabinet",
    "candle",
    "cd",
    "cell phone",
    "chair",
    "cloth",
    "coffee machine",
    "coffee table",
    "counter top",
    "credit card",
    "cup",
    "curtains",
    "desk",
    "desk lamp",
    "desktop",
    "dining table",
    "dish sponge",
    "dog bed",
    "drawer",
    "dresser",
    "dumbbell",
    "egg",
    "egg cracked",
    "faucet",
    "floor",
    "floor lamp",
    "footstool",
    "fork",
    "fridge",
    "garbage bag",
    "garbage can",
    "hand towel",
    "hand towel holder",
    "house plant",
    "kettle",
    "key chain",
    "knife",
    "ladle",
    "laptop",
    "laundry hamper",
    "lettuce",
    "lettuce sliced",
    "light switch",
    "microwave",
    "mirror",
    "mug",
    "ottoman",
    "painting",
    "pan",
    "paper towel roll",
    "pen",
    "pencil",
    "pepper shaker",
    "pillow",
    "pot",
    "remote control",
    "salt shaker",
    "scrub brush",
    "side table",
    "soap bottle",
    "sofa",
    "spoon",
    "television",
    "tissue box",
    "tomato",
    "towel holder",
    "vacuum cleaner",
    "watch",
    "window",
)

# Real-life robot object vocabulary.
REAL_OBJECTS: tuple[str, ...] = (
    "7up",
    "cola",
    "water",
    "milk",
    "orange juice",
    "tropical juice",
    "red wine",
    "red bull",
    "iced tea",
    "juice pack",
    "pringles",
    "cheezit",
    "cornflakes",
    "sugar",
    "coffee",
    "strawberry jello",
    "chocolate jello",
    "spam",
    "tomato soup",
    "mustard",
    "tuna",
    "plum",
    "pear",
    "apple",
    "lemon",
    "peach",
    "strawberry",
    "banana",
    "orange",
)


def use_simulator() -> bool:
    """Whether the simulator object set is active (driven by ACTION_SIMULATOR)."""
    return os.environ.get("ACTION_SIMULATOR", "false").lower() == "true"


def active_objects() -> tuple[str, ...]:
    """The object set selected by the ACTION_SIMULATOR env var (backend-side default)."""
    return AI2THOR_OBJECTS if use_simulator() else REAL_OBJECTS


def is_simulator_set(object_set: str | None) -> bool:
    """Resolve whether the simulator set applies for an explicit ``object_set``.

    ``"sim"``/``"real"`` are honored directly; ``None`` falls back to the
    ACTION_SIMULATOR env var (the backend-side default).
    """
    if object_set is not None:
        return object_set == "sim"
    return use_simulator()


def objects_for(object_set: str | None) -> tuple[str, ...]:
    """The object tuple for an explicit ``object_set`` (None = env-var default)."""
    return AI2THOR_OBJECTS if is_simulator_set(object_set) else REAL_OBJECTS


def choice_sets(
    objects: tuple[str, ...] | None = None,
    rooms: tuple[str, ...] | None = None,
    surfaces: tuple[str, ...] | None = None,
) -> dict[str, tuple[str, ...]]:
    """Named choice sets the grammar can pin an arg to (see ActionArg.choices).

    ``objects`` overrides the object set; when None it falls back to the
    ACTION_SIMULATOR env-var default. ``rooms``/``surfaces`` override the static
    location vocabulary with the backend's actual world (see DomesticRobot
    ``rooms``/``surfaces``); when None the static tuples are used.
    """
    rooms = rooms if rooms is not None else ROOMS
    surfaces = surfaces if surfaces is not None else SURFACES
    return {
        "rooms": rooms,
        "surfaces": surfaces,
        "places": tuple(rooms) + tuple(surfaces),
        "objects": objects if objects is not None else active_objects(),
    }
