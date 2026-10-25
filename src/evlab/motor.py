from dataclasses import dataclass


@dataclass(frozen=True)
class PMSM:
    resistance_ohm: float = 0.04
    ld_h: float = 0.0003
    lq_h: float = 0.0006
    flux_wb: float = 0.08
    pole_pairs: int = 4
    inertia_kg_m2: float = 0.08
    friction_nm_s: float = 0.001
    current_limit_a: float = 250.0

    def __post_init__(self):
        from .core import integer, positive

        integer(self.pole_pairs)
        for k in ("resistance_ohm", "ld_h", "lq_h", "flux_wb", "inertia_kg_m2", "current_limit_a"):
            positive(getattr(self, k), k)
        positive(self.friction_nm_s, allow_zero=True)
