"""
`GeoScript`, a gmsh `.geo` script built call by call, for the user's gmsh executable to run.

Its methods are named after the gmsh API calls they replace (`gmsh.model.occ.addSpline`,
`gmsh.model.mesh.setTransfiniteCurve`, `gmsh.write`, ...) and take the same arguments, so a builder
reads the same as it did against the API. The API returned each entity's tag; here the script assigns
it, counting up from 1 per entity kind in creation order, which is how gmsh's OpenCASCADE kernel
numbers them too.
"""

import operator
import os

_PHYSICAL_KIND = {0: "Point", 1: "Curve", 2: "Surface"}


def _num(x) -> str:
    """A float at full double precision, so the script reproduces the geometry exactly."""
    return repr(float(x))


def _tags(tags) -> str:
    return ", ".join(str(operator.index(t)) for t in tags)


class GeoScript:
    def __init__(self):
        self._lines = ['SetFactory("OpenCASCADE");']
        self._last_tag = {"point": 0, "curve": 0, "loop": 0, "surface": 0}

    def _new_tag(self, kind: str) -> int:
        self._last_tag[kind] += 1
        return self._last_tag[kind]

    # Geometry, `gmsh.model.occ.*`
    def addPoint(self, x, y, z) -> int:
        tag = self._new_tag("point")
        self._lines.append(f"Point({tag}) = {{{_num(x)}, {_num(y)}, {_num(z)}}};")
        return tag

    def addSpline(self, pointTags) -> int:
        tag = self._new_tag("curve")
        self._lines.append(f"Spline({tag}) = {{{_tags(pointTags)}}};")
        return tag

    def addLine(self, startTag, endTag) -> int:
        tag = self._new_tag("curve")
        self._lines.append(f"Line({tag}) = {{{_tags([startTag, endTag])}}};")
        return tag

    def addCircleArc(self, startTag, centerTag, endTag) -> int:
        tag = self._new_tag("curve")
        self._lines.append(f"Circle({tag}) = {{{_tags([startTag, centerTag, endTag])}}};")
        return tag

    def addCurveLoop(self, curveTags) -> int:
        tag = self._new_tag("loop")
        self._lines.append(f"Curve Loop({tag}) = {{{_tags(curveTags)}}};")
        return tag

    def addPlaneSurface(self, wireTags) -> int:
        tag = self._new_tag("surface")
        self._lines.append(f"Plane Surface({tag}) = {{{_tags(wireTags)}}};")
        return tag

    # Mesh constraints, `gmsh.model.mesh.*`
    def setTransfiniteCurve(self, tag, numNodes, meshType="Progression", coef=1.0):
        self._lines.append(
            f"Transfinite Curve{{{_tags([tag])}}} = {operator.index(numNodes)} Using {meshType} {_num(coef)};"
        )

    def setTransfiniteSurface(self, tag, arrangement="Left", cornerTags=()):
        corners = f" = {{{_tags(cornerTags)}}}" if len(cornerTags) else ""
        self._lines.append(f"Transfinite Surface{{{_tags([tag])}}}{corners} {arrangement};")

    def setRecombine(self, dim, tag):
        if dim != 2:
            raise ValueError(f"only surfaces are recombined here, got dim={dim}")
        self._lines.append(f"Recombine Surface{{{_tags([tag])}}};")

    # Model, options and output, `gmsh.model.addPhysicalGroup`, `gmsh.option.setNumber`, `gmsh.write`
    def addPhysicalGroup(self, dim, tags, tag, name):
        self._lines.append(f'Physical {_PHYSICAL_KIND[dim]}("{name}", {operator.index(tag)}) = {{{_tags(tags)}}};')

    def setNumber(self, name, value):
        self._lines.append(f"{name} = {_num(value)};")

    def write(self, fileName):
        # Absolute and forward-slashed, so it neither depends on gmsh's cwd nor needs escaping
        self._lines.append(f'Save "{os.path.abspath(fileName).replace(os.sep, "/")}";')

    def generate(self, dim):
        self._lines.append(f"Mesh {operator.index(dim)};")

    def text(self) -> str:
        return "\n".join(self._lines) + "\n"
