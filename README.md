# Open Channel Flow Analysis Suite

An object-oriented Python engine for open-channel hydraulics: uniform flow,
critical flow, specific energy, gradually varied flow (GVF), and hydraulic
jumps -- with three front ends (CLI, Tkinter desktop GUI, Streamlit web app)
built on the same core.

## Architecture

```
open_channel_flow/            <- core engine (import this from anywhere)
    geometry.py                 4 cross-sections: Rectangular, Triangular,
                                 Trapezoidal, Circular. Input validation +
                                 zero-division guards live here.
    core.py                     OpenChannelFlow class: normal_depth(),
                                 critical_depth(), velocity(), froude(),
                                 regime(), classify_channel().
    energy.py                   specific_energy(), energy_curve(),
                                 alternate_depth().
    gvf.py                      friction_slope(), RK4 integrator,
                                 solve_profile() (direction-aware),
                                 classify_curve() (M1-M3 / S1-S3),
                                 stitch_uniform_flow() (fills in the
                                 uniform-flow segment on the far side of
                                 a detected jump).
    jump.py                     momentum_function() (Belanger),
                                 sequent_depth(), jump_energy_loss(),
                                 belanger_rectangular() (closed-form
                                 check), locate_jump().
    control_structures.py       sluice_gate_depth(),
                                 weir_downstream_depth() -- derive the GVF
                                 boundary depth from a real structure.
    io_utils.py                 validation, tabular_report(),
                                 export_csv/json/excel().
    optimize.py                 most_economical_trapezoidal() (SLSQP),
                                 roughness_sensitivity().
    visualization.py            build_dashboard() (dual-panel Matplotlib
                                 + jump marker), plot_roughness_sensitivity().

main.py            CLI: prompts for shape/flow/boundary, prints results,
                    exports CSV/JSON/Excel, saves gvf_dashboard.png.
gui_tkinter.py      Desktop GUI: form + embedded Matplotlib dashboard.
app_streamlit.py    Interactive web app: sliders + live Plotly charts.
tests/              pytest unit tests (18 tests, geometry/core/gvf/jump).
```

## Why the code is structured this way

- **Every cross-section shares one interface** (`area`, `top_width`,
  `wetted_perimeter`, `hydraulic_radius`, `area_moment`), so the entire
  rest of the engine (Manning solver, critical depth, GVF, jump momentum
  function) works identically for a rectangle, triangle, trapezoid, or
  pipe -- swap the `CrossSection` instance and nothing else changes.
- **Physical validity guards are enforced at the geometry/core layer**,
  not scattered through the app code: negative widths/slopes raise
  `GeometryError` at construction time; zero or negative depth raises it
  on first use; solver non-convergence raises `ConvergenceError`.
- **The GVF integrator refuses to cross critical depth smoothly.** A real
  GVF curve cannot pass through `y = yc` -- that's exactly where a
  hydraulic jump belongs instead. `solve_profile()` stops there rather
  than grinding through to a numerically meaningless blow-up, and
  `locate_jump()` + `stitch_uniform_flow()` fill in the rest of the reach
  with the correct uniform-flow segment.

## Hydraulic jump detection -- what it does and doesn't cover

`locate_jump()` handles the two standard "mixed regime" textbook cases:

- **Case A (steep channel, downstream control):** a dam or weir forces a
  subcritical depth downstream of a channel whose normal flow is
  naturally supercritical. The jump sits where the backwater (M/S1) curve's
  depth matches the sequent depth of the uniform upstream flow.
- **Case B (mild channel, upstream control):** a sluice gate releases
  supercritical flow into a channel whose normal flow is subcritical. The
  jump sits where the sequent depth of the S-curve first matches the
  downstream normal depth.

It does **not** currently handle multiple jumps in one reach, jumps forced
by a mid-channel slope break (compound-slope channels), or submerged/
drowned jumps. `sequent_depth()` and `momentum_function()` are general
(work for any `CrossSection`), and are cross-checked in the test suite
against the closed-form Belanger equation for rectangular channels.

## Running it

```bash
pip install numpy scipy matplotlib pandas openpyxl streamlit plotly pytest

python main.py                    # interactive CLI
python gui_tkinter.py             # desktop GUI (needs a display)
streamlit run app_streamlit.py    # interactive web app
python -m pytest tests/ -v        # unit tests
```

## Extension lab features included

- `optimize.most_economical_trapezoidal()` -- finds the least-perimeter
  (B, y) pair for a given Q, n, S0, z via `scipy.optimize.minimize`
  (SLSQP), and checks the result against the closed-form best-hydraulic-
  section formula `B = 2y(sqrt(1+z^2) - z)`.
- `optimize.roughness_sensitivity()` -- sweeps a range of Manning's n and
  returns the resulting normal depth for each, paired with
  `visualization.plot_roughness_sensitivity()`.
- `gui_tkinter.py` -- full Tkinter desktop control dashboard.

## Known limitations / next steps

- The RK4 step size is fixed, not adaptive; very steep gradients near
  `yc` can need a smaller `dx` for full accuracy.
- Composite/multi-reach channels (different S0 or n along the same
  channel) aren't modeled -- each run assumes one uniform reach.
- `Circular.area_moment()` falls back to numerical integration (no
  closed form implemented yet), which is slightly slower but accurate.
