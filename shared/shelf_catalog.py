"""Shelf IDs taken from the live rware-tiny-2ag-v2 warehouse.

The environment has shelves, not parcels. Each reset builds the same
shelves in the same order and numbers them from 1. S001 is shelf 1.
Positions do not depend on the episode seed. The request queue is a
separate random sample and is not read here.
"""

from environment.environment import make_warehouse


def shelf_code(shelf_id):
    """Integer shelf id -> S001 style code."""
    return f"S{int(shelf_id):03d}"


def parse_shelf_code(text):
    """S001, s1, or 1 -> the shelf id. None when the text is not an id."""
    raw = str(text).strip().upper().replace(" ", "")
    if raw.startswith("S"):
        raw = raw[1:]
    if not raw.isdigit():
        return None
    number = int(raw)
    if number < 1:
        return None
    return number


def read_shelves(seed=0):
    """One row per shelf currently created by the warehouse."""
    env = make_warehouse()
    try:
        env.reset(seed=int(seed))
        warehouse = env.unwrapped
        return [
            {
                "id": int(shelf.id),
                "code": shelf_code(shelf.id),
                "column": int(shelf.x),
                "row": int(shelf.y),
            }
            for shelf in warehouse.shelfs
        ]
    finally:
        env.close()


def find_shelf(warehouse, text):
    """The shelf object on this warehouse, or None when the id is absent."""
    number = parse_shelf_code(text)
    if number is None:
        return None
    for shelf in warehouse.shelfs:
        if int(shelf.id) == number:
            return shelf
    return None
