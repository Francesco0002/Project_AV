from dataclasses import dataclass

@dataclass(frozen=True)
class ROI:
    id: int
    x: int
    y: int
    w: int
    h: int

    def contains_point(self, px: float, py: float) -> bool:
        return self.x <= px <= (self.x + self.w) and self.y <= py <= (self.y + self.h)

def bottom_center_xywh(x: float, y: float, w: float, h: float) -> tuple[float, float]:
    return (x + w / 2.0, y + h)

def denorm_roi(r: dict, W: int, H: int) -> ROI:
    return ROI(
        id=int(r["id"]),
        x=int(round(r["x"] * W)),
        y=int(round(r["y"] * H)),
        w=int(round(r["w"] * W)),
        h=int(round(r["h"] * H)),
    )
