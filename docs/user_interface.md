# The browser interface

`aquaflux-ui` is a browser interface served by your own machine; it opens on the
results of a run. Nothing is uploaded anywhere: the page is served on `127.0.0.1`, the
scene is rendered locally by VTK, and the browser is sent images of it.

## Installing and starting it

The viewer is an optional extra. Its dependencies — VTK through PyVista for
rendering, trame for the page, plotly for the convergence plot — all have wheels
for Linux, macOS and Windows, so installing it needs no compiler and no JavaScript
toolchain.

```bash
pip install "aquaflux[ui]"
aquaflux-ui results/
```

The argument is either the output directory an `aquaflux run` wrote (see
[case files](case_files.md)) or one VTK file: a `.vtu` unstructured grid, a `.pvd`
time series, a `.vtp` surface or a `.vtm` collection of them. The page opens in
the default browser; `--no-browser` prints its address instead, and `--port`
chooses the port (by default a free one). Stop it with Ctrl-C.
`python -m aquaflux_ui` is the same command.

## The page

A rail on the left switches between three sections: **Setup**, **Run** and
**Results**. The button at the left of the top bar hides or shows the side panel,
to give the view the whole window.

```bash
aquaflux-ui results/        # a run's output directory: opens on Results
aquaflux-ui case.yaml       # a case file: opens on Setup, with the case loaded
aquaflux-ui                 # nothing yet: opens on Setup
```

## Setup: editing a case file

**Open** chooses a case file through a browser of this machine's folders. The side
panel then shows the case section by section — the mesh, fluid, physics, boundary
patches, numerics, drive, sources, solver and outputs, one row per setting. A
setting the file states is shown in full weight. One it leaves out shows what
leaving it out means: its default, marked "(default)"; "Off" when leaving it out
turns the feature off; or "Not set" when the solver decides from the rest of the
case, which the setting's help (the ⓘ beside its name) explains. Emptying a box,
or its back-arrow button, puts a setting back to its default.

A setting that chooses a kind, such as a boundary patch's condition or a
turbulence model, heads a group of that kind's own settings. Each level of
nesting is marked by a guide line down its left edge and a light tint. Groups
below a section's top level start folded and show what is set inside them, with
a dot when anything is.

Every choice offered comes from the installed solver: the kinds a position can
hold (an inlet, an outlet or a wall for a boundary patch; each preconditioner the
solver knows), the values of a fixed choice, which settings are required, and
their defaults and help. The form therefore offers exactly what the solver reads,
and something added to the case file in a newer aquaflux appears in it unchanged.
Boundary patches are added by typing a name; lists such as the output writers by
choosing a kind under **Add…**.

**Save** and **Save as** write the file through aquaflux's own writer, which checks
the case first and writes nothing it would refuse: the reason is shown above the
form and the setting it concerns is marked. Saving rewrites the file from its
settings, so comments and hand formatting are not kept and a setting written out at
its default is dropped; the page says so before the first save over a file. Saved
as into another folder, the case's relative paths — the mesh's, above all — are
re-based so it still finds its mesh. **Check** checks the saved file against its
mesh, as `aquaflux check` does.

Beside the form, the case's mesh is shown as aquaflux reads it — or generates it,
for a structured grid — with its boundary patches listed by name and face count.
Changing the mesh settings marks the view out of date; **Reload** reads or generates
the mesh again from the settings as they stand, saved or not. Choosing a patch, or a
boundary entry named after a patch or a patch group, picks it out in the view once
aquaflux writes the boundary patches with the mesh. **Run** holds the place of starting and following a run: run it with
`aquaflux run case.yaml` in a terminal and open its output directory here; a run
still in progress can be followed in Results, whose **Reload** reads its history as
it grows.

## What it reads from a run

A run's output directory holds a record of the run, `run.yaml`, which lists every
file it wrote. The viewer shows the VTK files on that list and plots the
comma-separated-values file on it as the convergence history, so a run whose
outputs were given other names in its case file opens the same way. The side
panel's table summarizes `run.yaml` and `case.yaml`: the physics, the mesh, the
solver, whether it converged, in how many steps, to what residual, and the
aquaflux version and commit that ran it. A directory with no `run.yaml` — a run
stopped before it wrote one — shows every VTK file in it, and `history.csv` if
there is one. A run that did not converge wrote no fields, and the viewer says so
rather than opening an empty page.

## What the controls do

- **Dataset** — which of the snapshot's datasets is drawn. A multiblock file
  contributes each of its blocks under the block's name.
- **Colour by** — the cell (or point) field the dataset is coloured by, or
  `(None)`. For a vector, **Component** chooses its magnitude or one component.
- **Colormap**, **Log scale**, **Auto range** — the colour scale. Turned off,
  **Auto range** takes the two ends from the **Min** and **Max** boxes. The
  automatic range is taken from what the view is about: the slices and threshold
  regions when there are any, the surface only when it is drawn alone. On a
  logarithmic scale it starts at the smallest positive value, but no lower than a
  millionth of the largest, so a field that is zero over part of the domain — a
  fluence rate in full shadow — does not spread the scale over decades of nothing.
- **Surface**, **Opacity**, **Edges** — the dataset's outer surface. Beside a slice
  or a threshold region it is context and is drawn plain grey; adding the first
  slice or threshold fades an opaque surface so what is inside it can be seen.
- **Add slice** — adds a plane normal to X, Y or Z, as many as you need. Each has
  its own card: choose its axis, drag its slider, or type the coordinate it
  crosses the axis at (committed with Enter, or by leaving the box). A slice is
  made of the pieces of the cells it cuts, each coloured by its cell's value, so a
  sharp feature — a shadow edge — is shown exactly as the solution resolves it.
- **Add threshold** — adds a region of the cells whose value of a chosen field lies
  in a range, as many as you need. Each has its own field (which need not be the
  one the view is coloured by — a vector is tested by its magnitude), a range
  slider, and **Minimum** and **Maximum** boxes for typing the limits exactly.
- **Snapshot** — for a time series, which step is shown.

The buttons at the top right reset the camera and switch between a light and a
dark theme; the rendered view and the plot switch with the page.

The convergence plot below the view shows the residual, and the residual relative
to where the march began, against the step, on a logarithmic axis. **Reload**
reads the history file again, so a run still in progress can be followed.

## Large meshes

A large volume is never drawn as a volume. What is drawn is its outer surface,
extracted once when the dataset is first shown; each slice, which contains only the
cells its plane crosses; and the outer surface of the cells each threshold keeps.
Moving a slider recomputes only the piece that depends on it — changing the
colormap recomputes nothing, and moving one slice leaves the others as they were —
and a removed slice or threshold is not kept. The rendering happens on your machine and only images reach the browser,
so the size of the mesh does not affect the page.

The files aquaflux writes help: a cell bounded by six quadrilaterals is written as
a VTK hexahedron rather than a general polyhedron, which VTK reads and cuts with
much less memory.

## Without the solver

The interface imports neither aquaflux nor JAX — Results reads files with VTK — and
the solver never imports the interface. Results copied to another machine therefore
open there with only the extra's own dependencies installed.

Setup does need the solver. It asks the aquaflux installed beside the interface,
in one solver process started with the page (`aquaflux serve`), so the few seconds
that importing the solver takes are spent once, while the page loads, rather than on
every case opened or saved.
