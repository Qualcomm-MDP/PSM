import numpy as np
import shapely
import trimesh
from shapely.geometry import Polygon

from common.MeshUtils import get_base_and_height, get_corners, get_lines
from common.reporter import Reporter, get_reporter

# Foundation depth below the lowest ground point under a footprint.
EMBED_M = 0.1


def build_mesh(value, state):
    bbox = state.require_metadata("bbox")
    flagged: list[tuple[str, str]] = []
    data = extrude_buildings(
        value["osm"],
        bbox,
        reporter=get_reporter(state),
        terrain=state.get_metadata("terrain", None),
        flagged=flagged,
    )
    state.set_metadata("flagged_buildings", flagged)
    state.require_metadata("progress_monitor").next()
    return data


def ground_under(corners, terrain) -> tuple[float, float]:
    """Lowest and highest ground (m above terrain datum) under a footprint."""
    pts = np.asarray(corners, dtype=float)[:, :2]
    poly = Polygon(pts)
    if poly.is_valid and poly.area > 0:
        step = terrain.spacing
        ring = np.asarray(poly.exterior.segmentize(step).coords)[:, :2]
        minx, miny, maxx, maxy = poly.bounds
        gx, gy = np.meshgrid(np.arange(minx, maxx, step), np.arange(miny, maxy, step))
        inside = np.column_stack([gx.ravel(), gy.ravel()])
        inside = inside[shapely.contains_xy(poly, inside[:, 0], inside[:, 1])]
        pts = np.vstack([ring, inside])
    ground = terrain.ground(pts[:, 0], pts[:, 1])
    return float(ground.min()), float(ground.max())


def run_extrusion(id, buildings, element, area_bbox, terrain, tags=None) -> str | None:
    """Extrude one outline (a way, or one outer member of a relation) into buildings[id],
    added to what that id already has. tags: the height tags to use (a relation's member
    ways carry none of their own). Returns why the outline was left out, or None."""
    if "geometry" not in element:
        return "no outline in data"
    if len(element["geometry"]) < 3:
        return "outline has too few points"
    corners = get_corners(element, area_bbox)

    lines = get_lines(corners)

    path = trimesh.path.path.Path2D(
        entities=lines,
        vertices=corners,
    )

    base, top = get_base_and_height(
        {"tags": element.get("tags", {}) if tags is None else tags}
    )

    polys = path.polygons_closed
    if len(polys) == 0 or not polys[0]:
        return "outline is not a valid closed shape"

    # Meshes here are z-down (texturing and export flip z). On terrain the
    # roof sits `top` above the highest ground under the footprint and
    # the walls reach down to the lowest, so nothing floats downhill. A
    # building part with a base (min_height: e.g. a tower's upper section)
    # starts that high above the ground instead.
    bottom = base
    height = top - base
    if terrain is not None:
        ground_low, ground_high = ground_under(corners, terrain)
        bottom = ground_high + base if base > 0 else ground_low - EMBED_M
        height = ground_high + top - bottom

    height = -1 * height
    if height == 0:
        height += 1e-3

    mesh = path.extrude(height=height)

    if isinstance(mesh, list):
        mesh = trimesh.util.concatenate(
            [m.to_mesh() if hasattr(m, "to_mesh") else m for m in mesh]
        )
    else:
        if hasattr(mesh, "to_mesh"):
            mesh = mesh.to_mesh()

    if bottom:
        mesh.apply_translation([0.0, 0.0, -bottom])

    if id in buildings:
        buildings[id] = trimesh.util.concatenate([buildings[id], mesh])
    else:
        buildings[id] = mesh
    return None


def closed_ring(way) -> bool:
    """A member way that is a whole ring (first node = last node). Rings split over several
    member ways are left out: get_lines would close each piece into a stray shape."""
    g = way.get("geometry") or []
    return len(g) >= 4 and (g[0]["lat"], g[0]["lon"]) == (g[-1]["lat"], g[-1]["lon"])


def extrude_buildings(
    input_data,
    area_bbox,
    reporter: Reporter | None = None,
    terrain=None,
    flagged: list[tuple[str, str]] | None = None,
):
    """One mesh of every building in input_data (Overpass "out geom" form).

    A way is extruded from its outline. A relation (a multipolygon such as the Duderstadt
    Center) is extruded member by member: each outer member way that is a closed outline,
    at the relation's height. Inner members (courtyards) are not cut out yet, and an outer
    ring split over several member ways is left out. Outer members that are also fetched as
    buildings of their own are skipped, so they are not built twice.

    Buildings left out are appended to flagged as (osm id, reason) and reported."""
    reporter = reporter or get_reporter()
    flagged = [] if flagged is None else flagged
    buildings: dict[tuple, trimesh.Trimesh] = {}
    elements = input_data["elements"]
    way_ids = {e.get("id") for e in elements if e.get("type") == "way"}

    for _, element in reporter.progress(
        enumerate(elements),
        desc="Generating Building Mesh(s)",
        total=len(elements),
    ):
        osm = f"{element.get('type')} {element.get('id')}"
        id = (
            element.get("type"),
            element["id"],
        )  # a way and a relation can share a number
        if element.get("type") == "relation":
            outers = [
                m
                for m in element.get("members", [])
                if m.get("type", "way") == "way"
                and m.get("role", "outer") != "inner"
                and m.get("ref") not in way_ids
            ]
            reasons = [
                (
                    run_extrusion(
                        id,
                        buildings,
                        m,
                        area_bbox,
                        terrain,
                        tags=element.get("tags", {}),
                    )
                    if closed_ring(m)
                    else "outline is split over several ways (not built yet)"
                )
                for m in outers
            ]
            if id not in buildings:
                flagged.append(
                    (osm, next((r for r in reasons if r), "no outer outline in data"))
                )
            continue
        reason = run_extrusion(id, buildings, element, area_bbox, terrain)
        if reason:
            flagged.append((osm, reason))

    if flagged:
        details = "\n".join(f"  {osm}: {reason}" for osm, reason in flagged)
        reporter.warning(
            f"{len(flagged)} building(s) were left out of the mesh:\n{details}"
        )

    combined_mesh = trimesh.util.concatenate(
        [building for _, building in buildings.items()]
    )
    return combined_mesh
