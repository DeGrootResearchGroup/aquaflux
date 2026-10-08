---
paths:
  - "aquaflux_ui/**"
  - "tests/ui/**"
---

# Rules — `aquaflux_ui/` (the browser interface, `aquaflux-ui`)

> Comment Convention and Claude-Facing-File Reference Ban apply to `aquaflux_ui/*.py` exactly as to
> `aquaflux/` — it is shipped surface. Spell out comma-separated values (CSV) at first use per file.

A local browser interface (BUILT 2026-10-05, v1): its Results section is a read-only viewer for finished results, Setup opens / edits / saves / checks a case file, Run solves the open case and follows its march live (BUILT 2026-10-07). Renamed from `aquaflux_viz` / `aquaflux-viz` / the `viz` extra the same day, before it was ever committed (the project owner's call, once it stopped being only a viewer). PyVista/VTK renders
**server-side** (an off-screen `pv.Plotter`), trame serves the page and streams images through
trame-vtk's `VtkRemoteView`, plotly draws the convergence history. Installed by the `ui` extra;
entry point `aquaflux-ui` (`aquaflux_ui.__main__:main`). Agreed scope (project owner, 2026-10-05):
top-level package, tests run in CI, residual history output added to the solver, hexahedra written as
`VTK_HEXAHEDRON` (see `io.md`); then, the same day, a case-file editor in Setup (load, edit, save,
check) with every option generated from the solver's schema; then (2026-10-07) a Run section that starts a run and follows it live. Out of scope so far: steering a run while it goes, remote/multi-user hosting, a PyWebView wrapper (the rendering code is meant to be
reused unchanged by one).

## ⚠️ The independence rule (binding, mechanically checked)

**`aquaflux_ui` imports neither `aquaflux` nor `jax`; `aquaflux` never imports `aquaflux_ui`.** It
reads files only, so results copied to a machine without a working JAX open there, and the solver
never pays for VTK/trame. Pinned by `tests/ui/test_independence.py` — a fresh interpreter imports the
whole viewer (`app`, `__main__` included) and asserts no `aquaflux*`/`jax*` module came in; an AST scan
of `aquaflux/` asserts no import of the viewer. **This is why it is a separate top-level package, not
`aquaflux/ui/`**: `import aquaflux` enables x64 and configures JAX's compilation cache, so a subpackage
would import JAX on every launch. Tests *may* import aquaflux (they write fixtures with `write_vtu`,
`write_pvd`, `StepHistory`) — that is deliberate: it pins the writer↔reader contract between the two
packages.

## Structure

| module | holds | notes |
|---|---|---|
| `history.py` | `ConvergenceHistory.read(path)` | stdlib `csv`; a partly written last row is dropped (a live run), any other short/long row refused with its line number; `residual_columns()` prefers `residual_ratio` over `residual_norm` |
| `sources.py` | `ResultSource` ABC; `VtkFiles`; `RunDirectory`; `open_source` | the seam a live-run source would implement; `times()` / `history()` are re-asked on refresh, never cached by the page |
| `scene.py` | `View` (frozen user choices, holding `slices: tuple[Slice]` and `thresholds: tuple[Threshold]`), `Slice(axis, coordinate)`, `Threshold(field, low, high, component)`, `Layer`, `Pipeline`, `Scene`, pure helpers | `Pipeline` never renders → testable headless; `Scene` only adds/removes the actors it drew |
| `controls.py` | `view_from_state`, `new_/edit_/fit_slice`, `new_/edit_/fit_threshold`, `parse_number` | the page's entries are lists of plain dicts; every edit goes through these pure functions (no trame import) — the ONE state→`View` mapping is `view_from_state` |
| `app.py` | `Workspace` (the page shell), `Section` protocol | app bar, the Setup/Run/Results rail, the theme toggle, which section shows. Owns NO section's controls |
| `results_section.py` | `ResultsSection`, `convergence_figure` | the viewer proper: plotter, scene, its side panel and main area; routes control events to `controls` |
| `setup_section.py` | `SetupSection`, `SECTION_ICONS` | the case editor: Open / Save / Save as / Check in the side panel ABOVE the form (the project owner had the form moved there so the main area is free for the mesh view) |
| `case_form.py` | `CaseSchema`, `Row`, `form_sections`, `set_value`/`unset`/`remove_at`/`set_kind`/`add_item`/`add_entry`, `field_at`/`entry_kinds`, `setting`, `parse_input`, `plain_doc` | pure: schema + document → rows; edits return NEW documents. Lists NO kind, field or choice of its own |
| `solver_commands.py` | `SolverCommands` (`schema`/`show`/`write`/`mesh`/`check`/`plan`), `CaseDocument`, `MeshExport`, `RunPlan`, `Runner` | parses the commands' JSON; the runner is injected (tests pass a stub) and defaults to a `SolverWorker` |
| `solver_worker.py` | `SolverWorker`, `CommandResult`, `TIMEOUT` | ONE long-lived `sys.executable -m aquaflux serve` process (see `case.md`), started lazily — by the schema prefetch at page load. Requests serialized by a lock; replies read on a daemon thread into a per-process queue so the time limit works on every platform; a timeout or a death mid-request stops the process and the NEXT request starts a new one (`poll()` before writing plus a one-retry on `BrokenPipeError` — each covers the other, so a mutation of either alone passes; both together fail). Its stdin closing is how it stops, so nothing needs an exit hook. ⚠️ **Started in its own process group** (`start_new_session` / `CREATE_NEW_PROCESS_GROUP`): in the terminal's group it got Ctrl-C too and printed a `KeyboardInterrupt` traceback over the page's clean exit (project owner, 2026-10-06); pinned by `test_the_process_is_out_of_reach_of_the_terminals_ctrl_c` |
| `mesh_view.py` | `MeshView`, `mesh_layers` (pure), `patches_addressed`, `read_patches` | draws what `aquaflux mesh` wrote; reads no mesh format and generates nothing itself |
| `file_browser.py` | `list_directory` (pure), `FileBrowser` (dialog: open / save) | server-side, since a browser's picker hides the real path and a case's mesh path is relative to the case file |
| `run_section.py` | `RunSection`, `POLL_SECONDS` | the trame layout of Run and its wiring only: Run case / Stop, the status card, the case-file summary, the replace switch, the tiles, five plots, the events/log card; re-reads the history every `POLL_SECONDS` while a run goes |
| `run_monitor.py` | `RunLimits`, `RunEvent`, `run_events`, `run_tiles`, `residual_figure`, `cost_figures`, `equation_columns`, `RETRY_REASONS`, `EQUATION_PREFIX` | PURE: history + limits → figures, tiles, events (no trame import, tested headless) |
| `run_process.py` | `CaseRun`, `CONVERGED`/`NOT_CONVERGED`/`REFUSED`, `STOP_GRACE` | one `sys.executable -m aquaflux run case.yaml [--overwrite]` in its own process group, its stdout+stderr to a console file |
| `plots.py` | `base_figure` | the one plotly base every plot builds on (Results' convergence plot and Run's five) |
| `widgets.py` | `panel`, `icon_button`, `empty_state` | page pieces shared by sections |
| `theme.py` | `vuetify_config`, `RenderStyle`/`THEMES`, `PlotStyle`/`PLOT_STYLES`, `colormap_gradient`, `PAGE_CSS` | EVERY colour lives here, per light/dark theme: page (Vuetify), 3D view (background gradient, colour-bar text, grey context surface), plot. `Scene(plotter, style)` takes its `RenderStyle` injected; the theme toggle swaps page, view and plot together |
| `__main__.py` | CLI | imports `app` only after the source opened, so a bad path fails before trame loads |

- **`RunDirectory` reads what `run.yaml`'s `written` list names**, not fixed file names: VTK suffixes →
  datasets, the `.csv` → history. No `run.yaml` → every VTK file in the directory + `history.csv`.
  A non-converged run (no fields) is refused with that reason, not shown empty. It **composes** a
  `VtkFiles` (delegation), it does not re-implement reading.
- **`VtkFiles`**: non-series files are read once and the **same objects** handed to every frame —
  `Pipeline` keys its caches on `id(mesh)`, so identity is the contract that lets a snapshot change skip
  re-extracting a static dataset's surface. `.pvd` steps are read on demand, one cached per series;
  a frame at a time between steps shows the earlier step. Multiblock leaves are named by block name
  (nested with `/`), duplicates suffixed ` (2)`.
- **Any number of slices and thresholds (project owner, 2026-10-05)** — "Add slice" / "Add threshold"
  buttons, each entry a card with its own controls and a remove button; layers are named `slice N` /
  `threshold N` by position. A slice is `(axis, coordinate)` in the dataset's own coordinates (typed
  exactly, or by slider), clamped into the dataset's extent; a threshold has its OWN field (a vector by
  magnitude) and typed or slider limits that never invert. Number boxes commit on Enter/blur only (the
  `_COMMIT_EVENTS` declared through trame's `__events`), so a half-typed `-` or `0.` is never sent, and
  display 6 significant figures (`toPrecision(6)`) while the entry keeps full precision. Edits arrive as
  `(index, key, value)` calls (`update_modelValue=(self.edit_slice, "[i, 'axis', $event]")`) and the
  server REASSIGNS the list — mutating a nested dict in place is not seen by trame's change detection.
- **The outer surface is plain grey beside any slice or threshold (`View.has_focus`)** — the owner found
  a field colour at 20 % opacity read as a lavender wash carrying no information. The first slice or
  threshold added fades an opaque surface to 0.2.
- **Large-mesh policy (the brief's main performance concern):** a volume is never rendered as a
  volume. `Pipeline` draws its outer surface (extracted once per dataset), each slice (cut cells only)
  and the outer surface of each threshold's kept cells; each is cached per layer name by exactly what
  it depends on (a colour change recomputes nothing; moving one slice recomputes that one only), and
  a layer the latest view no longer draws is dropped. Pinned by
  `test_derived_geometry_is_reused_until_what_it_depends_on_changes` and `test_a_removed_slice_is_not_kept`.
- **The automatic colour range comes from what the view is LOOKING AT (`scene.focus_values`), not from
  everything drawn** — the slice and threshold layers when either is on, the surface only when drawn
  alone. Found by the project owner on the bunny room, 2026-10-05: with the range over all layers the
  ceiling faces beside the lamp (G up to 45 W/m²) set the scale while a slice through the room held
  0.003–0.06, so every slice was the bottom colour. The slice itself was verified correct (its values
  equal the cut cells' `G` exactly). Pinned by
  `test_a_slice_sets_the_automatic_range_and_the_surface_beside_it_does_not` (mutating back to all
  layers fails it).
- **Log colour scale**: `automatic_range(log_scale=True)` starts at the smallest positive value but no
  lower than `10**-LOG_DECADES` (6) of the max — a fluence rate is exactly 0 in full shadow.
- **`pyvista.trame` is NOT used**: with trame 4 / PyVista 0.49 it imports `trame_pyvista`, which is not
  installed by trame. The page drives `trame.widgets.vtk.VtkRemoteView(plotter.render_window)`
  directly, which needs one fewer dependency and no PyVista-trame version coupling.

- **Look (project owner asked for a modern style, 2026-10-05)**: the page is built on trame's bare
  `VAppLayout` (no Kitware footer), with a flat app bar (drop icon, `aquaflux` wordmark, run title, busy
  spinner, reset-camera and light/dark buttons), an accordion side panel (Run / Colour / Surface / Slices /
  Thresholds, each with an icon; slice and threshold counts as chips), the view and the plot in rounded
  cards, colormap choices with gradient swatches, FXAA anti-aliasing. Fonts are a **system stack** — the
  viewer runs offline, so no web font is fetched. Component defaults (outlined compact inputs, inset
  switches, `text-none` buttons) are set once in `vuetify_config()["defaults"]`, not per widget.

- **One interface for setup, running and results (project owner, 2026-10-05).** The page is a
  `Workspace` of three `Section`s switched from an icon rail; each section supplies `drawer()`,
  `main()`, `toolbar()` and `shown()` and is built ONCE, shown with `v_show`, so it keeps its state
  (camera, open panels, slices) while another is in front. Setup is the case editor (below); Run solves the open case (below).
  **The independence rule shapes how they are built:** both need the solver, which this package
  does not import — Setup asks it through the `aquaflux schema` / `show` / `write` / `mesh` / `check`
  commands, answered by one `aquaflux serve` process (below), and Run starts `aquaflux run` as a separate process and follows it through the files it
  flushes per step (`history.csv`, and the console it prints to; `ConvergenceHistory.read` tolerates a
  half-written last row). ⚠️ **`v_show` cannot hide an element carrying Vuetify's `d-flex`**
  (it is `display: flex !important`, beating the inline `display: none`) — put the class on a child.
- **The case editor's options come from the installed solver, never from this package (project owner,
  2026-10-05: "must be kept in sync with the file specification, ideally automatically").** `case_form`
  builds every row from `aquaflux schema` (`SettingsMapping.schema()`, read off the same annotations the
  solver validates with), so a kind or a `Literal` value added to the case file appears in the form with
  no change here. The sync is checked from the other side by
  `test_every_case_file_in_the_repository_is_shown_whole` (every committed case file: no unknown rows,
  every set value has a row, every choice/kind among its row's options) and
  `test_edits_made_through_the_form_give_a_case_the_solver_reads`. The only hand-written table is
  `SECTION_ICONS`, presentation only — an unlisted section gets a cog.
- **Widgets by position** (`Row.widget`): a nested position is a `kind` dropdown (omitted for a required
  one-form section) then its fields; `choice`; typed `integer` / `number` / `string`; `numbers` /
  `strings` (a list of scalars in one comma-separated box); `boolean` as a True/False dropdown (a switch
  cannot say "unset"); `list` (an "Add…" dropdown of kinds) with `item` headings; `table` (a name box) with
  `entry` headings; `raw` JSON for anything else. An empty box UNSETS the setting (back to its default),
  as does a set row's backspace button. A key the schema does not know is shown struck through, with a
  remove button, so an invalid file can be corrected.
- **Nesting is shown three ways, not by indent alone (project owner chose options 1+2 of three mocked,
  2026-10-06; the third, drill-in past depth 2 with a breadcrumb, was deferred).** (1) Guide lines: each row
  draws one 2-px line per enclosing group in its OWN background (`setup_section._RAIL_STYLE`; rows stack
  gapless, so the lines join into rails) plus a depth tint that stops at 2 (`.af-depth-*`), and the indent
  is `GROUP_INDENT` = 14 px. ⚠️ Position and size must be separate properties: written into
  `background-image` the whole value is invalid and the browser draws nothing, silently. Hover is an inset
  box-shadow and an invalid row sets `background-color`, so neither wipes the rails. (2) Folding: every
  group heading (`case_form.GROUP_WIDGETS`: kind/item/entry) below a section's top level starts folded; a
  heading's subtitle is its kind's one-line summary, or, folded, `Row.contents` (its directly set settings,
  `name value · …`), with a dot when `Row.set_inside` > 0 — computed in the pure `_with_group_contents`.
- **Defaults are shown, never offered as an option (project owner, 2026-10-06: the dropdown "includes
  Default as well as all of the available options, not denoting which is actually the default").** There
  is no "(Default)" item. `case_form._default_text` is the one rule: a stated default reads `<value>
  (default)` as the unset row's placeholder and marks its option in the dropdown (`_options`, the VALUE
  unmarked); `"required"` when there is none to fall back on; `OFF` ("Off") when the schema marks the
  field `off`; `NOT_SET` ("Not set") when its default is `None` and nothing resolves it. A default the
  schema RESOLVED (`resolved_default`, from the kind's `unset_resolves_to`) is shown exactly as a stated
  one (`case_form._default`). Of the 165 case-file fields whose stored default is `None` (2026-10-06), 62
  now resolve, 24 are off, and 79 stay "not set": those depend on the solve path or the rest of the case,
  or their default is an inline literal no declaration can read (see `solve.md`, "Three class-level
  declarations"), so the tooltip help is what tells a user what "Not set" does there.
- **A setting the case's physics does not read is not offered (project owner, 2026-10-06).** The schema
  publishes each field's `scope` and each physics' `reads` (`case/scopes.py`, see `case.md`);
  `CaseSchema.reading(document)` filters `fields()` to the scopes the document's physics reads, so a
  laminar case shows no `numerics.turbulence_advection`, wall `k` or inlet `turbulence`, and a flow case
  no light. A field required in its scope reads "required" when unset. Choosing a new physics (`set_kind`
  on `scopes_from`) applies `without_unread`, dropping every setting the new physics would refuse, so the
  file stays readable; a file OPENED with such a setting shows it struck through, to remove.
- **Boxes match the labels**: input and dropdown text 0.875 rem, 32 px boxes, 38 px rows (`theme.PAGE_CSS`),
  since Vuetify's compact field is still 1 rem in a 40 px box (project owner, 2026-10-06).
- ⚠️ **Every tooltip on the page drew as an empty bubble (found by the project owner, 2026-10-06), and
  the text was there all along**: a tooltip is `surface-variant` with `on-surface-variant` text, and
  Vuetify derives most `on-*` colours but keeps its own fixed `#EEEEEE` for that one — invisible on the
  light theme's `#E8EDF2`. Both themes now set it (`theme._PAGE_COLOURS`), pinned by
  `test_a_tooltips_text_is_readable_on_its_background` (WCAG contrast ≥ 4.5; Vuetify's near-white fails
  it). Diagnosed by reading the bubble's computed colours in the browser — the first guess (the
  tooltip's content slot) was wrong.
- **Saving**: through `aquaflux write --relative-to <the directory it was opened from>`, so Save as into
  another directory re-bases the mesh path (found in the browser on the first Save as: the copy pointed at
  a mesh that did not exist). The solver's refusal is shown above the form and its `at '<path>'` marks
  the matching row and section red. ⚠️ **Three template traps hit building this, all silent in Python:**
  (a) a trigger's args expression must be ONE expression — `"[a]; stmt"` inside `trigger('x', [...])` blanks
  the whole page; (b) a double-quoted JSON string in an event expression ends the HTML attribute — quote
  with `file_browser._js_string`; (c) a tuple as a widget's POSITIONAL content is not bound (`VIcon((expr,))`
  shows nothing) — use `icon=(expr,)` or `"{{ expr }}"`. A blank page is a template error: read the
  browser console.
- **The mesh view (project owner, 2026-10-05: "if it is a mesh file, it should load the mesh; if it is a
  structured mesh it should generate it… highlight boundaries when they are selected").** Setup's main
  area shows the case's mesh, read or generated by the SOLVER (`aquaflux mesh OUT --relative-to <case
  dir>` with the UNSAVED `mesh` section on stdin, so an edit can be previewed before saving), written as
  `mesh.vtu` into a per-load subdirectory of a `TemporaryDirectory`. It loads when a case opens; an edit
  under `mesh` marks it stale ("Mesh settings changed: reload"). The header lists each boundary patch with
  its face count as a chip; a chip, or a Boundaries table entry whose name is a mesh patch or GROUP (the
  command reports groups), highlights the patches it addresses (`patches_addressed`: a group's members) —
  matched by NAME against the mesh, never by assuming a section is called `boundaries`. The patches come
  from `patches.vtm`, which `aquaflux mesh` writes beside `mesh.vtu` with the solver's own
  `write_patches` (one block per boundary patch holding faces, named by it); a mesh with no boundary patch
  (fully periodic) shows no chips and says nothing. Checked in the browser on pitzDaily (2026-10-06): a
  chip highlights its patch. A 2D mesh is shown
  face on, a volume by its outer surface; edges are left out past `EDGE_LIMIT` faces. Framing is done
  server-side with a `FRAME_ZOOM` margin (the client-side `reset_camera` fit too tightly). Through the
  persistent worker, pitzDaily's (12 225 cells) first mesh load costs 2.0 s and a repeat 0.3 s
  (2026-10-06, under a load average of ~12; see `case.md` for the per-command breakdown).
- **Loading feedback (project owner, 2026-10-06: "no feedback that it is actually loading")**: `case_opening`
  (the file name) is set while a case is read — a status line under Open and a loading card in the main
  area, shown even before any case is loaded (the old `case_busy` spinner sat inside the loaded-case block
  so the first open showed nothing); the mesh card says "Loading the mesh…". The schema is fetched in the
  background at server start (`_prefetch_schema`), and an open that lands while it is in flight awaits
  THAT request (`_schema_task`) rather than asking again; a failed prefetch is retried by the next open.
- **The Run section (project owner, 2026-10-07: "a button to run the case… plot in the right panel the
  residual plot in real time… the number of cycles and details of the inner and outer solves"; the
  layout was mocked first and approved as drawn).** No settings of its own — everything is the case
  file's. The side panel: Run case / Stop, a status chip and card (step, elapsed, last step, whether the
  step was on the case's own problem, residual, stopping test), a summary of what the case file sets the
  run to do (`case_form.setting` reads each value or its default, `None` inside an `off` section), and
  the replace switch. The main area: five headline tiles, the residual (overall or per equation), four
  small plots on the same step axis, and an events/log card. **Every piece of data comes from what the
  solver records** — the history's columns (see `solve-march.md`, `StepHistory`) and the console; the
  page derives nothing about the march it cannot read there.
  - **A run reads the SAVED file**, so Run case is disabled while `case_modified`; and it replaces
    results **only with permission** (project owner, 2026-10-07): `aquaflux plan` (see `case.md`)
    reports `occupied` by the same rule `prepare_run` refuses on, and Run needs the switch on before it
    passes `--overwrite`. Until a new run starts, the earlier run's history is drawn ("Showing the
    earlier run"); a history file older than the run being followed is ignored, so an overwrite does
    not flash the old plot.
  - **The cost plot shows restart cycles (offset-corrected) and each step's hardest single solve
    against `retry.abort_above_cycles`, NOT the dual-time `cycle_budget`** — a deviation from the
    approved mockup, said in its summary: the budget is compared against the RAW summed count (lineax
    `num_steps`, +2 per solve), so a line at it over corrected cycles would compare two units.
  - **Per-equation lines use a validated palette** (`PlotStyle.equations`: six of the reference
    categorical hues in order, leaving out the orange and violet the retry/refit marks use). On the
    light surface the worst adjacent pair (green/red, CVD ΔE 7.2) is legal only with a second
    encoding, so each line is also labelled at its end; dark passes outright (2026-10-07, the dataviz
    validator). The legend and event marks take their colours from the same `PLOT_STYLES` through CSS
    variables generated per theme in `theme.py`.
  - **Stop sends SIGINT to the run's process group** (CTRL_BREAK on Windows) and kills only after
    `STOP_GRACE` = 60 s: an interrupt lands only when the solve returns to Python between compiled
    steps — on pitzDaily a 3 s wait was not enough. `aquaflux run` records an interrupted run as one
    that stopped short (`run.yaml`, exit 1). Closing the page stops a run in progress (`atexit`).
  - ⚠️ **Anything put in the page state must be plain Python** — a NumPy integer in the events list
    made trame log "Skip state value … not serializable", the variable never reached the page, and the
    template's `run_events.length` threw and took the page's connection down. `RunEvent.to_state`
    converts, pinned by a JSON round trip in `test_events_are_newest_first_and_say_what_happened`.
  - Checked by hand on a scratch copy of pitzDaily (2026-10-07): live residual, tiles, status, the
    events as refits landed, the earlier-run view, the replace switch, Stop.
- `aquaflux-ui [path]`: a results directory or VTK file opens on Results; a `.yaml` opens on Setup with
  the case loaded; no path opens on Setup (Results shows `NoResultsSection`).
- **Placeholder wording**: they say a feature "is not available in this version" and give today's
  command, never a roadmap label (Comment Convention item 2 applies to user-facing text too).

## Tests (`tests/ui/`, fast tier, CI via the `test` extra)

Each module gates on `pytest.importorskip("pyvista"|"trame")` and is declared in
`tests/unit/test_optional_dependency_skips.py`. **No test renders a frame** — a Linux CI runner has no
display, and whether VTK's wheel falls back to off-screen OSMesa/EGL there was not established — so
`Scene` is tested by its actors only. `test_workspace.py` builds the whole page (every section, an
off-screen render window, no render) and drives the Results entry methods through the page state; it
skips where `pv.system_supports_plotting()` is false (a Linux machine with no display), so on CI it
may not run — the event wiring (button clicks, typed boxes) is exercised only by hand in a browser. All 15 targeted mutations of the package (and 12 more of the multi-slice/threshold and control logic) (caching keys, identity reuse,
multiblock naming, log floor, threshold bounds, run-record listing, the jax-import guard, …) turn the
suite red (2026-10-05).

## Measured (2026-10-05)

Configuration: VTK 9.7.1, PyVista 0.49.0, trame 4.0.0, trame-vtk 2.11.17, macOS arm64 (11 cores,
19 GB), Python 3.13.
- Page checked by hand in the in-app browser on a 48 000-cell structured grid (run info, colour by,
  log scale, convergence plot from a real `StepHistory` file) and on the 2.46M-cell bunny room (two
  slices on different axes, a typed slice coordinate, a threshold with a typed minimum, grey context).
- See `io.md` for the hexahedron writer's file-size/memory measurement (1M cells: 70 vs 198 MB file,
  393 vs 1074 MB peak RSS).

## Deferred (tracked in the PR description, not built)

- Results following a run live (`ResultSource` over a running march, `times()` growing): Run follows
  the history, but Results still shows a run's fields only once it has written them, and is opened on a
  directory at start-up rather than switched to a run's output when it finishes.
- A radiation run's patch fields in Results: `PatchVtk` writes `patches.vtm` (merged 2026-10-06), which
  `RunDirectory` lists from `run.yaml` and `VtkFiles` reads as a multiblock by block name — not yet
  checked in a browser on a real radiation run.
- PyWebView wrapper; streamlines/glyphs (U is offered only as a coloured magnitude/component).
