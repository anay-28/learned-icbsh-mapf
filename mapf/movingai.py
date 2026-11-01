"""Strict official .scen parsing; convert (x,y) to the solver's (row,column)."""
from dataclasses import dataclass
from pathlib import Path
import math

from mapf.validation import validate_instance


@dataclass(frozen=True)
class Scenario:
    map_name: str
    width: int
    height: int
    start: tuple
    goal: tuple
    distance: float


def parse_scen(path):
    lines = Path(path).read_text(encoding='utf-8-sig').splitlines()
    if not lines or lines[0].strip().lower() != 'version 1':
        raise ValueError('Expected MovingAI scenario version 1')
    rows = []
    for number, line in enumerate(lines[1:], 2):
        if not line.strip():
            continue
        fields = line.split()
        if len(fields) != 9:
            raise ValueError(f'Malformed .scen row {number}')
        _, name, width, height, sx, sy, gx, gy, distance = fields
        width, height, sx, sy, gx, gy = map(int, (width, height, sx, sy, gx, gy))
        distance = float(distance)
        if width <= 0 or height <= 0 or not math.isfinite(distance) or distance < 0:
            raise ValueError('Invalid scenario dimensions/distance')
        if not (0 <= sx < width and 0 <= gx < width and 0 <= sy < height and 0 <= gy < height):
            raise ValueError('Scenario coordinates out of bounds')
        rows.append(Scenario(Path(name).name, width, height, (sy, sx), (gy, gx), distance))
    if not rows:
        raise ValueError('Empty .scen file')
    return rows


def first_k(rows, k, grid, map_name):
    if len(rows) < k or k < 1:
        raise ValueError('Insufficient scenario rows')
    selected = rows[:k]
    if any(r.map_name != map_name or r.width != grid.width or r.height != grid.height for r in selected):
        raise ValueError('Scenario/map mismatch')
    starts, goals = [r.start for r in selected], [r.goal for r in selected]
    if not validate_instance(grid, starts, goals):
        raise ValueError('Invalid first-k MAPF instance')
    return starts, goals
