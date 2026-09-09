from .drawer import PROFILE as DRAWER
from .letters import PROFILE as LETTERS
from .workpiece import PROFILE as WORKPIECE

PROFILES = {p["name"]: p for p in (DRAWER, LETTERS, WORKPIECE)}
