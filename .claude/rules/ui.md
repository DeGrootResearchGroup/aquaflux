---
paths:
  - "aquaflux_ui/**"
  - "tests/ui/**"
---

# Rules — `aquaflux_ui/` (the browser interface, `aquaflux-ui`)

> Comment Convention and Claude-Facing-File Reference Ban apply to `aquaflux_ui/*.py` exactly as to
> `aquaflux/` — it is shipped surface. Spell out comma-separated values (CSV) at first use per file.

A local browser interface with three sections: **Setup** (open / edit / save / check a case file),
**Run** (solve the open case and follow its march live), **Results** (read-only viewer). PyVista/VTK renders
server-side (off-screen `pv.Plotter`), trame serves the page and streams images through trame-vtk's
`VtkRemoteView`, plotly draws plots. `ui` extra; entry point `aquaflux-ui` (`aquaflux_ui.__main__:main`).
Out of scope (project owner): steering a running march, remote/multi-user hosting, a PyWebView wrapper
(the rendering code should be reusable unchanged by one). Dated findings and measurements:
`.claude/notes/ui-log.md`.

## ⚠️ The independence rule (binding, mechanically checked)

**`aquaflux_ui` imports neither `aquaflux` nor `jax`; `aquaflux` never imports `aquaflux_ui`.** Results
open without JAX, and the solver never pays for VTK/trame. That is why it is a top-level package, not
`aquaflux/ui/` (`import aquaflux` enables x64 and configures JAX). Pinned by `tests/ui/test_independence.py`
(fresh-interpreter import of the whole package + AST scan of `aquaflux/`). Tests *may* import aquaflux
(`write_vtu`, `write_pvd`, `StepHistory`) — deliberately, to pin the writer↔reader contract.

## Structure

| module | holds | notes |
|---|---|---|
| `history.py` | `ConvergenceHistory.read(path)` | stdlib `csv`; drops a partly written last row (live run), refuses other short/long rows by line number; `residual_columns()` prefers `residual_ratio` over `residual_norm` |
| `sources.py` | `ResultSource` ABC; `VtkFiles`; `RunDirectory`; `open_source` | `times()`/`history()` re-asked on refresh, never cached by the page |
| `scene.py` | `View` (frozen; `slices`, `thresholds`), `Slice`, `Threshold`, `Layer`, `Pipeline`, `Scene` | `Pipeline` never renders (testable headless); `Scene` only adds/removes actors it drew |
| `controls.py` | `view_from_state`, `new_/edit_/fit_slice`, `new_/edit_/fit_threshold`, `parse_number` | pure, no trame; `view_from_state` is the one state→`View` mapping |
| `app.py` | `Workspace`, `Section` protocol, `NoResultsSection` | page shell (app bar, rail, theme toggle); owns no section's controls |
| `results_section.py` | `ResultsSection`, `convergence_figure` | the viewer; routes control events to `controls` |
| `setup_section.py` | `SetupSection`, `SECTION_ICONS` | case editor; Open / Save / Save as / Check above the form in the side panel (owner: main area is for the mesh view) |
| `case_form.py` | `CaseSchema`, `Row`, `form_sections`, edit functions, `setting`, `parse_input`, `plain_doc` | pure: schema + document → rows; edits return new documents; lists no kind/field/choice of its own |
| `solver_commands.py` | `SolverCommands` (`schema`/`show`/`write`/`mesh`/`check`/`plan`), `CaseDocument`, `MeshExport`, `RunPlan`, `Runner` | parses the commands' JSON; runner injected (tests stub it), default `SolverWorker` |
| `solver_worker.py` | `SolverWorker`, `CommandResult`, `TIMEOUT` | see below |
| `mesh_view.py` | `MeshView`, `mesh_layers`, `patches_addressed`, `read_patches` | draws what `aquaflux mesh` wrote; reads no mesh format itself |
| `file_browser.py` | `list_directory`, `FileBrowser` | server-side: a browser picker hides the real path, and mesh paths are relative to the case |
| `run_section.py` | `RunSection`, `POLL_SECONDS` | Run's layout and wiring; re-reads history every `POLL_SECONDS` |
| `run_monitor.py` | `RunLimits`, `RunEvent`, `run_events`, `run_tiles`, `residual_figure`, `cost_figures`, `equation_columns`, `RETRY_REASONS`, `EQUATION_PREFIX` | pure: history + limits → figures, tiles, events |
| `run_process.py` | `CaseRun`, `CONVERGED`/`NOT_CONVERGED`/`REFUSED`, `STOP_GRACE` | one `sys.executable -m aquaflux run case.yaml [--overwrite]` in its own process group, output to a console file |
| `plots.py` | `base_figure` | the one plotly base for every plot |
| `widgets.py` | `panel`, `icon_button`, `empty_state` | shared page pieces |
| `theme.py` | `vuetify_config`, `RenderStyle`/`THEMES`, `PlotStyle`/`PLOT_STYLES`, `colormap_gradient`, `PAGE_CSS` | **every colour** lives here per theme; `Scene(plotter, style)` takes its `RenderStyle` injected |
| `__main__.py` | CLI | imports `app` only after the source opened, so a bad path fails before trame loads |

`aquaflux-ui [path]`: a results directory or VTK file opens Results; a `.yaml` opens Setup with the case
loaded; no path opens Setup.

## The solver seam
Setup asks the solver through `aquaflux schema`/`show`/`write`/`mesh`/`check`/`plan`, answered by **one
long-lived `sys.executable -m aquaflux serve` process** (`SolverWorker`, see `case.md`). Run starts
`aquaflux run` separately and follows the files it flushes per step (`history.csv`, the console).
- `SolverWorker` starts lazily (schema prefetch at page load), serializes requests with a lock, reads
  replies on a daemon thread so the timeout works on every platform; a timeout or death stops the process
  and the next request starts a new one. Both `poll()` before writing and a one-retry on `BrokenPipeError`
  are needed (each covers the other). Closing its stdin stops it.
- ⚠️ **Child processes start in their own process group** (`start_new_session` /
  `CREATE_NEW_PROCESS_GROUP`), or the terminal's Ctrl-C reaches them and prints tracebacks over a clean
  exit; pinned by `test_the_process_is_out_of_reach_of_the_terminals_ctrl_c`.

## Results
- **`RunDirectory` reads what `run.yaml`'s `written` list names** (VTK suffixes → datasets, `.csv` →
  history); no `run.yaml` → every VTK file + `history.csv`. A non-converged run is refused with that
  reason. It composes a `VtkFiles`.
- **`VtkFiles` hands the same objects to every frame** for non-series files — `Pipeline` caches on
  `id(mesh)`, so identity is the contract. `.pvd` steps are read on demand, one cached per series.
  Multiblock leaves are named by block name (nested with `/`, duplicates suffixed ` (2)`).
- **Any number of slices and thresholds**, each a card; layers named `slice N`/`threshold N` by position.
  A slice is `(axis, coordinate)` in dataset coordinates, clamped to the extent; a threshold has its own
  field (vectors by magnitude) and limits that never invert. Number boxes commit on Enter/blur only
  (`_COMMIT_EVENTS`) and display `toPrecision(6)` at full stored precision. Edits arrive as
  `(index, key, value)` calls and the server **reassigns** the list — in-place mutation of a nested dict
  is not seen by trame.
- **The outer surface is plain grey beside any slice or threshold** (`View.has_focus`; the first one fades
  an opaque surface to 0.2).
- **Large-mesh policy:** never volume-render. `Pipeline` draws the outer surface (once per dataset), each
  slice (cut cells only) and each threshold's outer surface, cached per layer by exactly what it depends
  on; layers no longer drawn are dropped. Pinned by `test_derived_geometry_is_reused_until_what_it_depends_on_changes`
  and `test_a_removed_slice_is_not_kept`.
- **The automatic colour range comes from what the view is looking at** (`scene.focus_values`: slice and
  threshold layers when any is on, the surface only when alone), else a bright surface pins every slice to
  the bottom colour. Pinned by `test_slices_set_the_automatic_range_and_the_surface_beside_them_does_not`.
- **Log scale** floors at `10**-LOG_DECADES` (6) of the max — fluence is exactly 0 in full shadow.
- **`pyvista.trame` is not used** (it imports `trame_pyvista`, not installed by trame); the page drives
  `trame.widgets.vtk.VtkRemoteView(plotter.render_window)` directly.

## Page and look
- `Workspace` of three `Section`s (`drawer()`, `main()`, `toolbar()`, `shown()`), each built once and shown
  with `v_show`, so it keeps camera/panels/slices while hidden.
- Bare `VAppLayout`, flat app bar, accordion side panel, rounded cards, a **system font stack** (offline:
  no web font). Component defaults are set once in `vuetify_config()["defaults"]`, not per widget. Inputs
  0.875 rem in 32 px boxes, 38 px rows (`theme.PAGE_CSS`).
- ⚠️ **`v_show` cannot hide an element carrying `d-flex`** (`display: flex !important`) — put the class on
  a child.
- ⚠️ **Tooltip text colour must be set per theme** (`theme._PAGE_COLOURS`): Vuetify keeps a fixed
  `#EEEEEE` `on-surface-variant`, invisible on the light surface. Pinned by
  `test_a_tooltips_text_is_readable_on_its_background` (WCAG ≥ 4.5).
- ⚠️ **Template traps, all silent in Python** (a blank page is a template error: read the browser console):
  (a) a trigger's args must be ONE expression — `"[a]; stmt"` blanks the page; (b) a double-quoted JSON
  string in an event expression ends the HTML attribute — quote with `file_browser._js_string`; (c) a tuple
  as a widget's positional content is not bound (`VIcon((expr,))`) — use `icon=(expr,)` or `"{{ expr }}"`.
- ⚠️ **Page state must be plain Python** — a NumPy integer is skipped as "not serializable" and the
  template using it throws, dropping the connection. `RunEvent.to_state` converts (pinned in
  `test_events_are_newest_first_and_say_what_happened`).
- Placeholders say a feature "is not available in this version" and give today's command, never a roadmap
  label.

## Setup (the case editor)
- **Every option comes from the installed solver's schema** (`aquaflux schema` → `SettingsMapping.schema()`),
  so new kinds/`Literal` values appear with no change here (owner: "must be kept in sync … ideally
  automatically"). Checked from the other side by `test_every_case_file_in_the_repository_is_shown_whole`
  and `test_edits_made_through_the_form_give_a_case_the_solver_reads`. The only hand table is
  `SECTION_ICONS` (presentation; unlisted → cog).
- **Widgets by position** (`Row.widget`): `kind` dropdown then fields; `choice`; `integer`/`number`/
  `string`; `numbers`/`strings` (comma-separated); `boolean` as a True/False dropdown (a switch cannot say
  "unset"); `list` with `item` headings; `table` with `entry` headings; `raw` JSON otherwise. An empty box
  or the backspace button **unsets** to default. An unknown key is shown struck through with a remove
  button.
- **Nesting** (owner's choice): guide rails drawn per row in its own background
  (`setup_section._RAIL_STYLE`) plus a depth tint capped at 2, indent `GROUP_INDENT`; and folding of
  group headings (`case_form.GROUP_WIDGETS`) below a section's top level, with a summary or
  `Row.contents` subtitle and a dot when `Row.set_inside > 0` (`_with_group_contents`). ⚠️ Background
  position and size must be separate CSS properties — inside `background-image` the value is invalid and
  draws nothing. Hover is an inset box-shadow and invalid rows set `background-color`, so neither wipes
  the rails.
- **Defaults are shown, never offered as an option** (owner). `case_form._default_text` is the one rule:
  `<value> (default)` as placeholder and marked in the dropdown; `"required"`; `OFF` ("Off") when the
  schema marks the field `off`; `NOT_SET` ("Not set") when the default is `None` and unresolved. A
  schema-resolved default (`resolved_default`, from a kind's `unset_resolves_to`) shows like a stated one
  (see `solve.md`, "Three class-level declarations").
- **A setting the case's physics does not read is not offered** (owner). `CaseSchema.reading(document)`
  filters by each field's `scope` and each physics' `reads` (`case/scopes.py`, `case.md`). Choosing a new
  physics (`set_kind` on `scopes_from`) applies `without_unread`; a file opened with an unread setting
  shows it struck through.
- **Saving** goes through `aquaflux write --relative-to <the directory it was opened from>`, so Save as
  re-bases the mesh path. The solver's refusal is shown above the form; its `at '<path>'` marks the row
  and section red.
- **Mesh view.** The solver reads or generates the mesh (`aquaflux mesh OUT --relative-to <case dir>`
  with the **unsaved** `mesh` section on stdin), writing `mesh.vtu` + `patches.vtm` (`write_patches`, one
  block per patch) into a per-load temporary subdirectory. An edit under `mesh` marks it stale. Patch
  chips (and Boundaries entries naming a patch or group) highlight via `patches_addressed`, matched by
  **name** against the mesh. 2D face on, volume by outer surface, edges omitted past `EDGE_LIMIT`;
  framing server-side with `FRAME_ZOOM` (client `reset_camera` fit too tightly).
- **Loading feedback**: `case_opening` drives a status line and loading card (shown before any case is
  loaded). The schema is prefetched at server start (`_prefetch_schema`); an open during it awaits that
  request (`_schema_task`); a failed prefetch is retried by the next open.

## Run
No settings of its own — everything is the case file's (layout mocked and approved by the owner). Side
panel: Run case / Stop, status, a summary of what the case sets (`case_form.setting`, `None` inside an
`off` section), the replace switch. Main: five tiles, the residual (overall or per equation), four small
plots on one step axis, an events/log card. **Every datum comes from what the solver records** (history
columns, `solve-march.md` `StepHistory`, and the console); the page derives nothing it cannot read there.
- **A run reads the saved file** (Run disabled while `case_modified`) and **replaces results only with
  permission**: `aquaflux plan` reports `occupied` by the rule `prepare_run` refuses on, and the switch must
  be on to pass `--overwrite`. The earlier run's history is shown until a new one starts; a history file
  older than the followed run is ignored.
- **The cost plot shows offset-corrected restart cycles and each step's hardest solve against
  `retry.abort_above_cycles`, not the dual-time `cycle_budget`** — a stated deviation from the approved
  mockup: the budget compares against the raw summed count (lineax `num_steps`, +2 per solve), a
  different unit.
- **Per-equation lines** use `PlotStyle.equations` (six validated categorical hues, excluding the orange
  and violet used for retry/refit marks), each also labelled at its end (light-theme green/red needs a
  second encoding). Legend and marks take colours from `PLOT_STYLES` via per-theme CSS variables.
- **Stop sends SIGINT to the run's process group** (CTRL_BREAK on Windows) and kills only after
  `STOP_GRACE` = 60 s: an interrupt lands only between compiled steps. `aquaflux run` records it as
  stopped short (`run.yaml`, exit 1). Closing the page stops a run (`atexit`).

## Tests (`tests/ui/`, fast tier, CI via the `test` extra)
Each module gates on `pytest.importorskip("pyvista"|"trame")`, declared in
`tests/unit/test_optional_dependency_skips.py`. **No test renders a frame** (no display on CI; off-screen
fallback not established), so `Scene` is tested by its actors. `test_workspace.py` builds the whole page
and drives Results through page state, but skips where `pv.system_supports_plotting()` is false, so it may
not run on CI — event wiring is exercised only by hand in a browser.

## Deferred (not built)
- Results following a run live (`times()` growing) and switching to a run's output when it finishes.
- A radiation run's patch fields in Results (`patches.vtm` listed by `RunDirectory`, read by `VtkFiles`)
  — not yet checked in a browser on a real radiation run.
- PyWebView wrapper; streamlines/glyphs (U only as coloured magnitude/component).
