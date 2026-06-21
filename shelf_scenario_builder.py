"""shelf_scenario_builder.py — Graphical 3D shelf scene authoring tool.

A Tkinter control panel (object list, tier radio buttons, typed numeric pose fields) drives a
PyBullet 3D view with a live ghost preview. Place + orient objects from the bundled mesh catalogue
onto a 3-tier wooden shelf, designate the hidden TARGET, and SAVE the scene to a YAML.

Self-contained: meshes, shelf geometry, and the wood texture all live under ./assets/. No external
data needed beyond the pip requirements (pybullet, trimesh, numpy, pyyaml).

PANEL (left, Tkinter):
  Catalogue listbox  — click an object; a yellow GHOST tracks your pose in the 3D view.
  Tier               — radio: Bottom(0) / Mid(1) / Top(2); z auto-snaps to that board.
  Pose fields        — x, y, yaw, roll, pitch: TYPE a value or use the steppers.
  x->column / y row  — quick-set buttons for 3 columns + front/back depth.
  Place / Update Sel / Delete Sel / Clear — placed-object edit.
  Placed listbox     — [TARGET] tag; click to load an object's pose back into the fields to edit.
  Mark as Target     — tag the selected placed object as the hidden goal.
  Scene name + Load/Save  — reads/writes ./scenes/<name>.yaml.

3D VIEW (right, PyBullet): a wooden shelf (boards + back/side panels) + a robot-base marker; placed
objects solid, the current selection a translucent ghost. Left-click a placed object to select it.

USAGE:
  pip install -r requirements.txt
  python shelf_scenario_builder.py [--name my_scene] [--load scenes/foo.yaml]

Needs a display (the GUI is interactive).
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path

import numpy as np
import trimesh
import yaml

import pybullet as p
import pybullet_data

import tkinter as tk
from tkinter import ttk, filedialog

# ── Paths (all bundled under ./assets) ─────────────────────────────────────────
_HERE       = Path(__file__).resolve().parent
_MESH_DIR   = _HERE / "assets" / "meshes"
_SHELF_YAML = _HERE / "assets" / "shelf.yaml"
_WOOD_PNG   = _HERE / "assets" / "wood.png"
_SCENES_DIR = _HERE / "scenes"
_CACHE_DIR  = _MESH_DIR / "_builder_cache"

# ── Geometry conventions (mesh +Y = physical height; objects authored Y-up) ─────
_R_X90 = np.array([[1, 0,  0, 0],
                   [0, 0, -1, 0],
                   [0, 1,  0, 0],
                   [0, 0,  0, 1]], dtype=np.float64)   # mesh +Y -> world +Z (upright)
_TARGET_FACES = 8_000
_PROVEN_COLS  = [0.13, 0.30, 0.47]                     # lateral columns (L / C / R)
_GRASP_LIMIT  = 0.078                                  # min-horiz <= this = parallel-gripper graspable

_COLOURS = [
    [0.80, 0.20, 0.20, 1.0], [0.20, 0.60, 0.80, 1.0], [0.20, 0.80, 0.30, 1.0],
    [0.85, 0.65, 0.10, 1.0], [0.70, 0.20, 0.80, 1.0], [0.90, 0.50, 0.10, 1.0],
    [0.20, 0.80, 0.80, 1.0], [0.80, 0.80, 0.20, 1.0], [0.50, 0.50, 0.50, 1.0],
]


class ShelfGeom:
    """Tier surfaces + interior bounds from assets/shelf.yaml shelves[0]."""

    def __init__(self, shelf_yaml: Path):
        with open(shelf_yaml, encoding="utf-8") as f:
            s = yaml.safe_load(f)["shelves"][0]
        self.cx, self.cy, self.cz = s["pose"][:3]
        self.w, self.d, self.h    = s["size"]
        self.levels    = int(s.get("levels", 3))
        self.thickness = float(s.get("thickness", 0.018))
        self._bottom   = self.cz - self.h / 2.0
        self._gap      = self.h / self.levels
        self.x_min, self.x_max = self.cx - self.w / 2.0, self.cx + self.w / 2.0
        self.y_min, self.y_max = self.cy - self.d / 2.0, self.cy + self.d / 2.0   # back .. opening

    def surface_z(self, level: int) -> float:
        return self._bottom + level * self._gap + self.thickness / 2.0


# ── Mesh prep (cached) ──────────────────────────────────────────────────────────
_mesh_cache: dict[str, tuple[str, np.ndarray]] = {}


def _prep_mesh(slug: str) -> tuple[str, np.ndarray]:
    if slug in _mesh_cache:
        return _mesh_cache[slug]
    glb = _MESH_DIR / f"{slug}_metric_fp.glb"
    raw = trimesh.load(str(glb), force="scene")
    mesh = (trimesh.util.concatenate(list(raw.geometry.values()))
            if isinstance(raw, trimesh.Scene) else raw)
    mesh = trimesh.Trimesh(vertices=np.asarray(mesh.vertices, dtype=np.float64),
                           faces=np.asarray(mesh.faces), process=False)
    mesh.apply_transform(_R_X90)
    mesh.apply_translation(-mesh.bounding_box.centroid)
    if len(mesh.faces) > _TARGET_FACES:
        try:
            mesh = mesh.simplify_quadric_decimation(target_reduction=1.0 - _TARGET_FACES / len(mesh.faces))
        except TypeError:
            mesh = mesh.simplify_quadric_decimation(face_count=_TARGET_FACES)
    _CACHE_DIR.mkdir(parents=True, exist_ok=True)
    obj_path = _CACHE_DIR / f"{slug}_builder.obj"
    mesh.export(str(obj_path))
    _mesh_cache[slug] = (str(obj_path), np.asarray(mesh.bounding_box.extents, dtype=np.float64))
    return _mesh_cache[slug]


def _oriented_height(extents: np.ndarray, rpy_deg) -> float:
    r, pi_, y = np.radians(rpy_deg)
    R = trimesh.transformations.euler_matrix(r, pi_, y, "sxyz")[:3, :3]
    return float(np.sum(np.abs(R[2, :]) * extents))


def _quat(rpy_deg):
    return p.getQuaternionFromEuler([np.radians(a) for a in rpy_deg])


# ── Builder ─────────────────────────────────────────────────────────────────────
class SceneBuilder:
    def __init__(self, name: str, load_path: Path | None):
        self.name = name
        self.geom = ShelfGeom(_SHELF_YAML)
        self.catalogue = sorted(f.name[:-len("_metric_fp.glb")]
                                for f in _MESH_DIR.glob("*_metric_fp.glb"))
        if not self.catalogue:
            raise SystemExit(f"No *_metric_fp.glb in {_MESH_DIR}")
        self.placed: list[dict] = []
        self.cur_slug = self.catalogue[0]
        self.ghost_id: int | None = None
        self._ghost_key = None

        self._build_world()
        self._build_panel()
        if load_path and Path(load_path).exists():
            self._load_scene(Path(load_path))
        self._refresh_placed_list()

    # ---- PyBullet world (wooden shelf) -----------------------------------------
    def _build_world(self):
        p.connect(p.GUI)
        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        p.setGravity(0, 0, 0)
        p.configureDebugVisualizer(p.COV_ENABLE_GUI, 0)        # hide param panel; Tk drives it
        p.configureDebugVisualizer(p.COV_ENABLE_SHADOWS, 1)
        p.loadURDF("plane.urdf", [0, 0, 0])
        wood = p.loadTexture(str(_WOOD_PNG)) if _WOOD_PNG.exists() else -1
        g = self.geom
        wood_rgba = [0.82, 0.62, 0.40, 1.0]      # warm oak tint (texture modulates this)

        def panel(pos, size, rgba=wood_rgba, tex=wood):
            bid = self._box(pos, size, rgba)
            if tex >= 0:
                p.changeVisualShape(bid, -1, textureUniqueId=tex)
            return bid

        # Boards: one per tier surface + a top cap = a real 3-shelf cabinet.
        for lvl in range(g.levels + 1):
            panel([g.cx, g.cy, g.surface_z(lvl) - g.thickness / 2.0], [g.w, g.d, g.thickness])
        # Back + side panels (full wood). Front stays open for placing/viewing.
        panel([g.cx, g.y_min, g.cz], [g.w, g.thickness, g.h])
        panel([g.x_min, g.cy, g.cz], [g.thickness, g.d, g.h])
        panel([g.x_max, g.cy, g.cz], [g.thickness, g.d, g.h])
        # Robot-base marker (dark) so the reach side (+y) is visible.
        self._box([0.5, 0.21, 0.819], [0.08, 0.08, 0.04], [0.12, 0.12, 0.12, 1.0])
        p.resetDebugVisualizerCamera(1.15, 90, -22, [g.cx, g.cy, g.surface_z(1)])

    @staticmethod
    def _box(pos, size, rgba):
        vis = p.createVisualShape(p.GEOM_BOX, halfExtents=[s / 2 for s in size], rgbaColor=rgba)
        return p.createMultiBody(0, -1, vis, basePosition=pos)

    def _spawn(self, slug, x, y, lvl, rpy, rgba, ghost=False):
        obj_path, extents = _prep_mesh(slug)
        cz = self.geom.surface_z(lvl) + _oriented_height(extents, (rpy[0], rpy[1], 0)) / 2.0
        vis = p.createVisualShape(p.GEOM_MESH, fileName=obj_path, rgbaColor=rgba, meshScale=[1, 1, 1])
        col = -1 if ghost else p.createCollisionShape(p.GEOM_BOX, halfExtents=[e / 2 for e in extents])
        return p.createMultiBody(0, col, vis, basePosition=[x, y, cz], baseOrientation=_quat(rpy))

    # ---- Tk panel --------------------------------------------------------------
    def _build_panel(self):
        self.root = tk.Tk()
        self.root.title("Simple Shelf Scene Builder")
        self.root.geometry("390x800")
        pad = dict(padx=6, pady=3)
        try:
            ttk.Style(self.root).theme_use("clam")
        except tk.TclError:
            pass

        cat = ttk.LabelFrame(self.root, text="Catalogue"); cat.pack(fill="both", expand=False, **pad)
        lbf = ttk.Frame(cat); lbf.pack(fill="both", expand=True, padx=4, pady=4)
        sb = ttk.Scrollbar(lbf, orient="vertical")
        self.cat_list = tk.Listbox(lbf, height=9, yscrollcommand=sb.set, exportselection=False)
        sb.config(command=self.cat_list.yview); sb.pack(side="right", fill="y")
        self.cat_list.pack(side="left", fill="both", expand=True)
        for s in self.catalogue:
            self.cat_list.insert("end", s)
        self.cat_list.selection_set(0)
        self.cat_list.bind("<<ListboxSelect>>", self._on_cat_select)
        self.lbl_grasp = ttk.Label(cat, text=""); self.lbl_grasp.pack(anchor="w", padx=4, pady=(0, 4))

        tier = ttk.LabelFrame(self.root, text="Tier"); tier.pack(fill="x", **pad)
        self.var_tier = tk.IntVar(value=0)
        for i, nm in enumerate(["Bottom (0)", "Mid (1)", "Top (2)"][: self.geom.levels]):
            ttk.Radiobutton(tier, text=nm, variable=self.var_tier, value=i).pack(side="left", padx=8, pady=4)

        pose = ttk.LabelFrame(self.root, text="Pose  (type a value or step)"); pose.pack(fill="x", **pad)
        g = self.geom
        self.var_x   = tk.DoubleVar(value=0.30)
        self.var_y   = tk.DoubleVar(value=-0.225)
        self.var_yaw = tk.DoubleVar(value=0.0)
        self.var_rol = tk.DoubleVar(value=0.0)
        self.var_pit = tk.DoubleVar(value=0.0)
        self._spin(pose, "x  lateral (m)", self.var_x, g.x_min + 0.02, g.x_max - 0.02, 0.005, 0)
        self._spin(pose, "y  depth (m)",   self.var_y, g.y_min + 0.02, g.y_max - 0.02, 0.005, 1)
        self._spin(pose, "yaw (deg)",   self.var_yaw, -180, 180, 15, 2)
        self._spin(pose, "roll (deg)",  self.var_rol, -180, 180, 15, 3)
        self._spin(pose, "pitch (deg)", self.var_pit, -180, 180, 15, 4)
        quick = ttk.Frame(pose); quick.grid(row=5, column=0, columnspan=2, sticky="w", padx=4, pady=4)
        ttk.Label(quick, text="col:").pack(side="left")
        for c in _PROVEN_COLS:
            ttk.Button(quick, text=f"{c}", width=4, command=lambda v=c: self.var_x.set(v)).pack(side="left", padx=1)
        ttk.Label(quick, text=" y:").pack(side="left")
        ttk.Button(quick, text="front", width=5, command=lambda: self.var_y.set(-0.155)).pack(side="left", padx=1)
        ttk.Button(quick, text="back", width=5, command=lambda: self.var_y.set(-0.325)).pack(side="left", padx=1)

        act = ttk.Frame(self.root); act.pack(fill="x", **pad)
        ttk.Button(act, text="Place New", command=self._place).pack(side="left", expand=True, fill="x", padx=2)
        ttk.Button(act, text="Update Sel", command=self._update_selected).pack(side="left", expand=True, fill="x", padx=2)
        act2 = ttk.Frame(self.root); act2.pack(fill="x", padx=6)
        ttk.Button(act2, text="Delete Sel", command=self._delete_selected).pack(side="left", expand=True, fill="x", padx=2)
        ttk.Button(act2, text="Clear All", command=self._clear).pack(side="left", expand=True, fill="x", padx=2)

        pl = ttk.LabelFrame(self.root, text="Placed  ( [TARGET] = hidden object to discover )"); pl.pack(fill="both", expand=True, **pad)
        pf = ttk.Frame(pl); pf.pack(fill="both", expand=True, padx=4, pady=4)
        sb2 = ttk.Scrollbar(pf, orient="vertical")
        self.placed_list = tk.Listbox(pf, height=7, yscrollcommand=sb2.set, exportselection=False)
        sb2.config(command=self.placed_list.yview); sb2.pack(side="right", fill="y")
        self.placed_list.pack(side="left", fill="both", expand=True)
        self.placed_list.bind("<<ListboxSelect>>", self._on_placed_select)
        tagf = ttk.Frame(pl); tagf.pack(fill="x", padx=4, pady=(0, 4))
        # Only the TARGET (hidden object to discover) is tagged. The OCCLUDER is NOT declared —
        # a POMDP shelf system discovers + removes occluders itself; declaring one defeats the point.
        ttk.Button(tagf, text="Mark as Target  (the hidden object to discover)",
                   command=self._mark_target).pack(fill="x", padx=2)

        sv = ttk.LabelFrame(self.root, text="Scene"); sv.pack(fill="x", **pad)
        row = ttk.Frame(sv); row.pack(fill="x", padx=4, pady=4)
        ttk.Label(row, text="name:").pack(side="left")
        self.var_name = tk.StringVar(value=self.name)
        ttk.Entry(row, textvariable=self.var_name).pack(side="left", expand=True, fill="x", padx=4)
        btns = ttk.Frame(sv); btns.pack(fill="x", padx=4, pady=(0, 4))
        ttk.Button(btns, text="LOAD YAML", command=self._load_dialog).pack(side="left", expand=True, fill="x", padx=2)
        ttk.Button(btns, text="SAVE YAML", command=self._save).pack(side="left", expand=True, fill="x", padx=2)
        self.lbl_status = ttk.Label(self.root, text="ready", relief="sunken", anchor="w")
        self.lbl_status.pack(fill="x", side="bottom")

        self.root.protocol("WM_DELETE_WINDOW", self._quit)
        self._on_cat_select()

    def _spin(self, parent, label, var, frm, to, inc, row):
        ttk.Label(parent, text=label).grid(row=row, column=0, sticky="w", padx=4, pady=2)
        ttk.Spinbox(parent, from_=frm, to=to, increment=inc, textvariable=var, width=10
                    ).grid(row=row, column=1, sticky="e", padx=4, pady=2)

    # ---- callbacks -------------------------------------------------------------
    def _on_cat_select(self, _=None):
        sel = self.cat_list.curselection()
        if sel:
            self.cur_slug = self.catalogue[sel[0]]
        _, ext = _prep_mesh(self.cur_slug)
        mh = min(ext[0], ext[1]); ok = mh <= _GRASP_LIMIT
        self.lbl_grasp.config(text=f"{self.cur_slug}: min-horiz {mh*100:.1f} cm  "
                                   f"{'GRASPABLE' if ok else 'WIDE > 8 cm'}",
                              foreground="#127a12" if ok else "#b35900")

    def _pose(self):
        return (float(self.var_x.get()), float(self.var_y.get()), int(self.var_tier.get()),
                (float(self.var_rol.get()), float(self.var_pit.get()), float(self.var_yaw.get())))

    def _place(self):
        x, y, lvl, rpy = self._pose()
        bid = self._spawn(self.cur_slug, x, y, lvl, rpy, _COLOURS[len(self.placed) % len(_COLOURS)])
        self.placed.append(dict(slug=self.cur_slug, x=round(x, 3), y=round(y, 3), level=lvl,
                                rpy_deg=[round(a, 1) for a in rpy], body_id=bid, is_target=False))
        self._refresh_placed_list(); self._status(f"placed {self.cur_slug}")

    def _selected_idx(self):
        sel = self.placed_list.curselection()
        return sel[0] if sel else None

    def _on_placed_select(self, _=None):
        i = self._selected_idx()
        if i is None:
            return
        o = self.placed[i]
        self.cur_slug = o["slug"]
        idx = self.catalogue.index(o["slug"])
        self.cat_list.selection_clear(0, "end"); self.cat_list.selection_set(idx); self.cat_list.see(idx)
        self.var_x.set(o["x"]); self.var_y.set(o["y"]); self.var_tier.set(o["level"])
        self.var_rol.set(o["rpy_deg"][0]); self.var_pit.set(o["rpy_deg"][1]); self.var_yaw.set(o["rpy_deg"][2])
        self._on_cat_select()

    def _update_selected(self):
        i = self._selected_idx()
        if i is None:
            self._status("select a placed object first"); return
        o = self.placed[i]; x, y, lvl, rpy = self._pose()
        p.removeBody(o["body_id"])
        o.update(slug=self.cur_slug, x=round(x, 3), y=round(y, 3), level=lvl,
                 rpy_deg=[round(a, 1) for a in rpy],
                 body_id=self._spawn(self.cur_slug, x, y, lvl, rpy, _COLOURS[i % len(_COLOURS)]))
        self._refresh_placed_list(keep=i); self._status(f"updated {self.cur_slug}")

    def _delete_selected(self):
        i = self._selected_idx()
        if i is None:
            self._status("select a placed object first"); return
        p.removeBody(self.placed.pop(i)["body_id"])
        self._refresh_placed_list(); self._status("deleted")

    def _clear(self):
        for o in self.placed:
            p.removeBody(o["body_id"])
        self.placed.clear(); self._refresh_placed_list(); self._status("cleared")

    def _mark_target(self):
        i = self._selected_idx()
        if i is None:
            self._status("select a placed object first"); return
        for o in self.placed:
            o["is_target"] = False
        self.placed[i]["is_target"] = True
        self._refresh_placed_list(keep=i)
        self._status(f"{self.placed[i]['slug']} = TARGET (hidden object to discover)")

    def _refresh_placed_list(self, keep=None):
        self.placed_list.delete(0, "end")
        for o in self.placed:
            tag = "[TARGET]" if o["is_target"] else "        "
            self.placed_list.insert("end",
                f"{tag} {o['slug']:18} x{o['x']:+.2f} y{o['y']:+.2f} t{o['level']} yaw{o['rpy_deg'][2]:.0f}")
        if keep is not None and 0 <= keep < len(self.placed):
            self.placed_list.selection_set(keep)

    # ---- save / load -----------------------------------------------------------
    def _save(self):
        if not self.placed:
            self._status("nothing to save"); return
        self.name = self.var_name.get().strip() or self.name
        tgt = next((o["slug"] for o in self.placed if o["is_target"]),
                   max(self.placed, key=lambda o: -o["y"])["slug"])     # deepest if none tagged
        tgt_obj = next(o for o in self.placed if o["slug"] == tgt)
        doc = {
            "scene_name": self.name, "target": tgt,
            "target_tier": tgt_obj["level"], "target_x": tgt_obj["x"],
            "objects": [
                {"slug": o["slug"], "x": o["x"], "y": o["y"], "level": o["level"],
                 "roll_deg": o["rpy_deg"][0], "pitch_deg": o["rpy_deg"][1], "yaw_deg": o["rpy_deg"][2]}
                for o in self.placed],
        }
        _SCENES_DIR.mkdir(parents=True, exist_ok=True)
        out = _SCENES_DIR / f"{self.name}.yaml"
        with open(out, "w", encoding="utf-8") as f:
            yaml.safe_dump(doc, f, sort_keys=False)
        self._status(f"saved {len(self.placed)} objs -> {out.name} (target={tgt})")
        print(f"[builder] SAVED -> {out}")

    def _load_scene(self, path: Path):
        with open(path, encoding="utf-8") as f:
            doc = yaml.safe_load(f)
        self.name = doc.get("scene_name", self.name)
        tgt = doc.get("target")
        for o in doc.get("objects", []):
            rpy = [o.get("roll_deg", 0.0), o.get("pitch_deg", 0.0), o.get("yaw_deg", 0.0)]
            bid = self._spawn(o["slug"], o["x"], o["y"], int(o["level"]), rpy,
                              _COLOURS[len(self.placed) % len(_COLOURS)])
            self.placed.append(dict(slug=o["slug"], x=o["x"], y=o["y"], level=int(o["level"]),
                                    rpy_deg=rpy, body_id=bid, is_target=(o["slug"] == tgt)))
        print(f"[builder] loaded {len(self.placed)} objects from {path}")

    def _load_dialog(self):
        path = filedialog.askopenfilename(
            title="Load shelf scene", initialdir=str(_SCENES_DIR),
            filetypes=[("Scene YAML", "*.yaml *.yml"), ("All files", "*.*")])
        if not path:
            return
        self._clear(); self._load_scene(Path(path)); self.var_name.set(self.name)
        self._refresh_placed_list(); self._status(f"loaded {Path(path).name}")

    # ---- 3D-view click selection ----------------------------------------------
    def _ray_from_to(self, mx, my):
        info = p.getDebugVisualizerCamera()
        width, height = info[0], info[1]
        cam_forward, horizon, vertical = info[5], info[6], info[7]
        dist, target = info[10], info[11]
        cam_pos = [target[i] - dist * cam_forward[i] for i in range(3)]
        far = 10000.0
        fwd = [target[i] - cam_pos[i] for i in range(3)]
        inv = far / math.sqrt(sum(c * c for c in fwd))
        ray_forward = [fwd[i] * inv for i in range(3)]
        d_hor = [horizon[i] / width for i in range(3)]
        d_ver = [vertical[i] / height for i in range(3)]
        center = [cam_pos[i] + ray_forward[i] for i in range(3)]
        ray_to = [center[i] - 0.5 * horizon[i] + 0.5 * vertical[i] + mx * d_hor[i] - my * d_ver[i] for i in range(3)]
        return cam_pos, ray_to

    def _poll_3d_click(self):
        try:
            events = p.getMouseEvents()
        except Exception:
            return
        for e in events:
            if e[0] == 2 and e[3] == 0 and (e[4] & p.KEY_WAS_TRIGGERED):   # left button pressed
                try:
                    rf, rt = self._ray_from_to(e[1], e[2]); hits = p.rayTest(rf, rt)
                except Exception:
                    continue
                if hits and hits[0][0] >= 0:
                    bid = hits[0][0]
                    for idx, o in enumerate(self.placed):
                        if o["body_id"] == bid:
                            self.placed_list.selection_clear(0, "end")
                            self.placed_list.selection_set(idx); self.placed_list.see(idx)
                            self._on_placed_select()
                            self._status(f"selected {o['slug']} (3D click) — edit fields, then Update Sel")
                            break

    # ---- tick + mainloop -------------------------------------------------------
    def _status(self, msg):
        self.lbl_status.config(text=msg)

    def _tick(self):
        try:
            x, y, lvl, rpy = self._pose()
        except (tk.TclError, ValueError):
            self.root.after(40, self._tick); return
        key = (self.cur_slug, round(rpy[0], 1), round(rpy[1], 1))
        if key != self._ghost_key:
            if self.ghost_id is not None:
                p.removeBody(self.ghost_id)
            self.ghost_id = self._spawn(self.cur_slug, x, y, lvl, rpy, [0.95, 0.9, 0.15, 0.45], ghost=True)
            self._ghost_key = key
        else:
            _, ext = _prep_mesh(self.cur_slug)
            cz = self.geom.surface_z(lvl) + _oriented_height(ext, (rpy[0], rpy[1], 0)) / 2.0
            p.resetBasePositionAndOrientation(self.ghost_id, [x, y, cz], _quat(rpy))
        self._poll_3d_click()
        if p.isConnected():
            p.stepSimulation()
        self.root.after(40, self._tick)

    def _quit(self):
        try:
            p.disconnect()
        except Exception:
            pass
        self.root.destroy()

    def run(self):
        print(f"[builder] {len(self.catalogue)} objects. Panel + 3D view ready.")
        self.root.after(40, self._tick)
        self.root.mainloop()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", default="my_shelf_scene")
    ap.add_argument("--load", default=None)
    args = ap.parse_args()
    SceneBuilder(args.name, Path(args.load) if args.load else None).run()


if __name__ == "__main__":
    main()
