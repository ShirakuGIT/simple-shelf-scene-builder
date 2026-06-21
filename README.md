# Simple Shelf Scene Builder

A small graphical tool to author 3-D shelf scenes — place and orient objects from a mesh
catalogue onto a 3-tier wooden shelf, tag the hidden target, and save the layout to YAML.

A **Tkinter** control panel (object list, tier radio buttons, typed numeric pose fields) drives a
**PyBullet** 3-D view with a live translucent *ghost* preview. Object height auto-snaps to the
chosen tier board, even for laid-down objects (roll/pitch). Everything is self-contained — meshes,
shelf geometry, and the wood texture all live under `assets/`.

![shelf scenario builder](assets/output.gif)

## Install

```bash
pip install -r requirements.txt
```

Requires Python 3.9+ and a display (the GUI is interactive). Dependencies: `pybullet`,
`trimesh`, `numpy`, `pyyaml`.

## Run

```bash
python shelf_scenario_builder.py                       # new scene
python shelf_scenario_builder.py --name kitchen_demo   # set output name
python shelf_scenario_builder.py --load scenes/foo.yaml  # open an existing scene to edit
```

## Controls

| Panel | What it does |
|---|---|
| **Catalogue** | Click an object — a yellow ghost tracks your pose in the 3-D view. A label flags whether it fits a typical parallel gripper (`min-horiz ≤ 8 cm`). |
| **Tier** | Radio buttons Bottom / Mid / Top. Object z auto-snaps to that board surface. |
| **Pose** | Type or step `x`, `y`, `yaw`, `roll`, `pitch`. Quick buttons set the 3 columns and front/back depth. |
| **Place New / Update Sel / Delete Sel / Clear All** | Add a new object, or edit/remove the selected one. |
| **Placed list** | Click a row (or **left-click the object in the 3-D view**) to load its pose back into the fields for editing. |
| **Mark as Target** | Tag the selected object as the hidden target. |
| **Load / Save YAML** | Reads / writes `scenes/<name>.yaml`. |

## Scene YAML format

```yaml
scene_name: my_scene
target: ycb_rubiks_cube        # the hidden object to discover
target_tier: 2                 # 0 bottom / 1 mid / 2 top
target_x: 0.205
objects:
  - slug: black_filter_box
    x: 0.30
    y: -0.155
    level: 0                   # tier
    roll_deg: 0.0
    pitch_deg: 0.0
    yaw_deg: 0.0
  - slug: ycb_rubiks_cube
    x: 0.205
    y: -0.34
    level: 2
    roll_deg: 0.0
    pitch_deg: 30.0
    yaw_deg: 60.0
```

Positions are in the world frame (meters). `level` indexes the tier board; the loader/builder
auto-computes z from the tier surface + the object's oriented height. `roll/pitch/yaw` are degrees.

## Catalogue

The `assets/meshes/` folder holds metric `.glb` meshes (`{slug}_metric_fp.glb`, decimated for
fast loading). The mesh convention is **+Y = physical height** (the builder rotates +Y → world +Z
to stand objects upright). Drop in any `{slug}_metric_fp.glb` following that convention and it
appears in the catalogue automatically on next launch.

## Coordinate convention

- World frame, meters.
- Shelf geometry is read from `assets/shelf.yaml` (`shelves[0]`: `size`, `pose`, `levels`,
  `thickness`). Edit it to retarget a different shelf.
- The dark cube in front of the shelf marks a robot base position (reference only).

## License

Meshes under `assets/meshes/` are derived from public scanned object datasets (YCB and similar).
Check the upstream licenses before redistribution.
