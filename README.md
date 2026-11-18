# FlowField
Repository to build models for 2D,3D Flow Field neural surrogates. This repo mainly investigates FNO, NIF, and AB-UPT

Generating data (airfoil geometry, meshes): see [scripts/datagen/README.md](scripts/datagen/README.md).

## Third-party tools

FlowField ships none of the programs below. Each is installed separately, under its own license, and found through
`tools.toml` ([tools.example.toml](tools.example.toml)). FlowField runs each as a separate program: it writes the
input files, starts the program, and reads back the files it writes. It never imports or links their code, so their
licenses cover those programs and not FlowField, whose own terms are in [LICENSE.md](LICENSE.md). Nor do they cover
what the programs produce: the meshes and solutions they write are FlowField's data, as a program's output carries
none of the program's own code.

| Program | Used for | License | Source |
|---|---|---|---|
| Construct2D (`c2d`) | O-grid meshes, the mesh study's mesher | GPL-3.0-or-later | [sourceforge.net/projects/construct2d](https://sourceforge.net/projects/construct2d/) |
| Gmsh | C- and O-grid meshes | GPL-2.0-or-later, with an exception | [gmsh.info](https://gmsh.info) |
| SU2 (`SU2_CFD`) | RANS solves | LGPL-2.1-or-later | [github.com/su2code/SU2](https://github.com/su2code/SU2) |
| XFoil | panel-method solves | GPL-2.0-or-later | [web.mit.edu/drela/Public/web/xfoil](https://web.mit.edu/drela/Public/web/xfoil/) |

One Python library under a copyleft license is imported directly: PyNomadBBO, the interface to the NOMAD 4 optimizer
the mesh study searches with (LGPL-3.0, [github.com/bbopt/nomad](https://github.com/bbopt/nomad)). pip installs it
on its own from [requirements.txt](requirements.txt), and it can be swapped for any other build of it.
