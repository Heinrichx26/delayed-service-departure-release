# Departure release with delayed service reports

Python implementations of joint service-information requests and airport departure
release decisions. The controller chooses when to request future takeoff counts
and how many aircraft to release, subject to response delays and a maximum number
of requests in each decision interval.

Repository: https://github.com/Heinrichx26/delayed-service-departure-release

Open derived data: https://github.com/Heinrichx26/delayed-service-departure-release/tree/main/data

## Contents

- `src/trb_latency_20261028/`: delayed responses and fixed release targets.
- `src/trb_joint_20261028/`: joint request and release optimization; fixed request schedules.
- `src/trb_precision_20261028/`: exact and interval-valued reports, request/time trade-offs.
- `src/substantive_20261028/`: interval service calendars and selective requests.
- `src/trb_independent_20261028/`: date-separated calibration and matched request ablations.
- `src/trb_six_20261028/`: gate deadlines, request comparisons, and simulated delayed reports.
- `src/problem_driven_20261028/`: immediate-response request optimization and request budgets.
- `src/trb_open_validation/`: public flight-record processing and source-document retrieval.
- `src/plotting/`: figures regenerated from saved numerical results.
- `data/`: frozen numerical cases and experiment configurations.
- `results/`: saved numerical outputs and summaries.

Directory date suffixes identify compatible experiment configurations. The package
contains code, derived numerical inputs and numerical outputs. Publication text,
submission documents and third-party articles are excluded.

## External replications and request valuation

The additional airport cases and saved comparisons are in
data/third_airport_replication/, data/temporal_transfer_replication/,
data/independent_adsb/ and data/independent_adsb_q1/, with corresponding
subdirectories in results/. The processed surveillance-derived data retain
the source archive's ODbL-1.0 license and attribution in source_manifest.json.
The raw archive remains available from the original source identified there.

The replication code is in src/stochastic_release_control/. From the repository
root, first run a monthly smoke test:

    python -m src.stochastic_release_control.build_independent_adsb_q1 smoke --airport ATL --month 1

After it passes, replace smoke with full for that airport and month.
The supplied processed data allow this comparison without downloading the raw archive.

The request valuation analysis uses saved exact waiting/request pairs and
measures group costs in aircraft-minutes. Run the smoke test before the full
analysis:

    python src/trb_objective_20261029/request_value_audit.py smoke
    python src/trb_objective_20261029/request_value_audit.py full

Outputs in results/revision_20261029/request_value/ include exact policy
intervals, corrected group-level valuation thresholds, input hashes and
airport-date bootstrap intervals. This calculation does not rerun optimization.

The travel-time sensitivity is in
`results/trb_precision_20261028/travel_time_sensitivity/`; its smoke and full
runs use the same frozen service calendars and independently replay every
retained policy.

The public ATCO2 metadata check is in `src/open_data_atco2/`. It parses the
freely distributed XML and INFO metadata and excludes the third-party audio
archive. The current archive contains 560 recordings and 877 timestamped
speech segments from seven airports. The summaries in `results/open_data_atco2/`
report communication-segment durations, speaker roles and turn gaps. These statistics describe short speech-segment durations and within-recording
gaps. Request response times and concurrent task budgets remain scenario inputs.

## Installation

Python 3.11 is the tested interpreter. The retained experiment manifests use
Windows relative paths; the archived panel runners have been tested on Windows.

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

## Verify before running a panel

Run all commands from this directory. This small smoke test compares optimization
with independent enumeration, verifies physical takeoffs and information timing,
and checks all 94 frozen evaluation cases against their hashes.

```powershell
python src/reproduce.py smoke
```

Two additional independent checks exercise more randomly generated cases:

```powershell
python src/trb_joint_20261028/validate_joint.py
python src/trb_precision_20261028/validate_band.py
```

The full panels use the included frozen configurations. The following sequence
recomputes delayed targets, joint requests and releases, and report precision.
These are longer computations; the drawing commands below use saved outputs.

```powershell
python src/trb_latency_20261028/panel.py run
python src/trb_joint_20261028/panel_joint.py run
python src/trb_precision_20261028/panel_band.py run
```

Replay records generated by these runners check each calendar's physical takeoffs
and whether identical observed information produces the same action. The stored
case data include the common travel time, release limits, training service counts,
probabilities, and evaluation service counts. The 94 evaluation windows cover
Denver and Phoenix in January 2024 and January 2025, with separate training dates.
The experiment configurations retain exact input and source hashes.

## Draw from saved results

```powershell
python src/plotting/report_precision.py
python src/plotting/waiting_reduction.py
python src/plotting/request_tradeoff.py
```

These commands read numerical outputs, make no solver calls, and save vector PDF
and editable SVG files in `results/figures/`. PDF text is embedded as text and
SVG labels remain editable. Figure outputs are ignored by Git.

## Public data and provenance

Raw flight records come from the United States Bureau of Transportation
Statistics (BTS), Airline On-Time Performance database:

- [Field definitions](https://www.transtats.bts.gov/Fields.asp?gnoyr_VQ=FGJ)
- [January 2024 archive](https://transtats.bts.gov/PREZIP/On_Time_Reporting_Carrier_On_Time_Performance_1987_present_2024_1.zip)
- [January 2025 archive](https://transtats.bts.gov/PREZIP/On_Time_Reporting_Carrier_On_Time_Performance_1987_present_2025_1.zip)

`data/public_sources.json` records the source URLs and SHA-256 checksums.
`src/download_public_data.py` downloads either monthly archive and verifies its
checksum. For example:

```powershell
python src/download_public_data.py --year 2024
```

Frozen experiment cases are supplied in `data/substantive_20261028/holdout_boxes/panel_cases/`;
the matching 94-window manifest is `data/substantive_20261028/holdout_boxes/panel_design.json`.
Their service calendars are reconstructed numerical inputs. Future service reports
are simulated in the numerical evaluations. Airport gate-count sources are
identified in `src/trb_open_validation/inventory_sources.py`; the source documents
remain available from the issuing airports and agencies.

The preprocessing utilities support tracing the construction of the flight events
and service calendars. Exact panel reproduction starts from the supplied frozen
cases. The intermediate flight database and third-party source documents are
retrieved separately when rebuilding the earlier preprocessing stages.


## License and citation

Original code is licensed under MIT. Original derived numerical inputs and outputs
are licensed under [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/).
Third-party sources retain their own terms. Cite this software using `CITATION.cff`.

The complete simulated delayed-report outputs are stored in
`results/trb_six_20261028/box_rows.json.gz`; Python's `gzip.open(..., 'rt')`
reads this JSON archive directly. `box_summary.json` gives the aggregated
comparisons. See `data/README.md` for the frozen-case fields and `LICENSE-DATA.md`
for the derived-data license.

## Time-varying travel

The ordered-arrival extension allows travel time to vary by release slot.
Its independent simulator and full release/request enumeration are in
src/trb_core_20261030/variable_travel.py. One-slot perturbations are sensitivity
inputs on the existing two-minute grid. Start with:

    python src/trb_core_20261030/variable_travel.py smoke

After it passes, run:

    python src/trb_core_20261030/variable_travel.py full
    python src/trb_core_20261030/audit_travel.py

The saved numerical results include all 200 initial window/release-limit
combinations and record the 37 combinations whose earliest releases fail to
complete under the perturbed travel profiles. The complete comparison has
163 combinations, four response/request settings and two travel information
settings. Saved nominal policies support an independent audit at the same
takeoff references. Plotting and manuscript production are separate.


## Erroneous delayed count reports

The src/trb_error_20261030 module contains exact finite-calendar decisions
under a whole-period sum of absolute report-count errors. The state records
the error already used by every retained calendar. It includes delayed
answers, a per-slot request limit and an optional hard total request cap.
The error allowance is a specified reporting-accuracy sensitivity input.
It is not an estimated distribution of tower communication errors.

Run the small checks before public-panel experiments:

    python src/trb_error_20261030/smoke.py smoke
    python src/trb_error_20261030/smoke.py audit
    python src/trb_error_20261030/validate_cover.py
    python src/trb_error_20261030/audit_budget.py smoke
    python src/trb_error_20261030/audit_budget.py audit
    python src/trb_error_20261030/public_panel.py public_smoke
    python src/trb_error_20261030/public_panel.py full
    python src/trb_error_20261030/matched_evaluation.py smoke
    python src/trb_error_20261030/matched_evaluation.py full
    python src/trb_error_20261030/matched_budget_panel.py smoke
    python src/trb_error_20261030/matched_budget_panel.py full

The full.json file stores 400 solved conditions: 200 fixed window/release-limit
combinations at each of two response times. The independent-range policy
must be compared using matched_full.json, which evaluates that policy
under the same shared error allowance as the exact policy.
The budget_full.json file additionally constrains each exact policy by the
maximum whole-period requests used by two-step lookahead in that condition.
The exact value is between the unrestricted optimum and the lookahead
value in all 400 conditions. Mean reductions from lookahead are 16.6 and
22.1 seconds per departure at two- and four-minute responses under the
same total request ceiling. All averages first take the worst permitted
history in each condition, then give conditions equal weight.

The sensitivity_public.json file retains four predetermined conditions for
error allowances zero, one and two. The summary.json file stores numeric
aggregates; detailed results retain request counts, values, physical
checks, solve status and input checksums. The independent physical
enumeration matches 384 unbudgeted and 144 request-capped conditions.
The fixed-timetable audit covers 1,344 timetables.

This numerical module has no manuscript-generation or plotting dependency.


## Longer lookahead and independent scenario optimization

The src/trb_calibrated_20261030 directory adds four- and six-step receding
request policies under the same global count-error allowance and request
limits. For each policy and condition, the exact comparison uses that
policy's maximum whole-period request count as a hard ceiling. Four-step
mean savings are 2.06 and 2.68 seconds per departure at the two response
times. Six-step policies attain the exact values in all 400 conditions.

Start with the smoke checks, then expand each experiment:

    python src/trb_calibrated_20261030/deep_lookahead.py smoke --depth 4
    python src/trb_calibrated_20261030/deep_lookahead.py full --depth 4
    python src/trb_calibrated_20261030/deep_lookahead.py smoke --depth 6
    python src/trb_calibrated_20261030/deep_lookahead.py full --depth 6
    python src/trb_calibrated_20261030/lookahead_certificate.py smoke
    python src/trb_calibrated_20261030/lookahead_certificate.py full

Finite-depth lower bounds certify 301 of the six-step policies by eight
modeled decisions. The other 99 retain a positive bound gap; their exact
values are available from the complete recursion. Certificate results
separately record candidate computation, policy evaluation and bound time.

The independent scenario mixed-integer formulation retains release and
request variables and generates nonanticipativity constraints. Waiting
has strict priority over a secondary total-request objective. Run:

    python src/trb_calibrated_20261030/mip_panel.py smoke
    python src/trb_calibrated_20261030/mip_panel.py audit
    python src/trb_calibrated_20261030/mip_panel.py public_smoke
    python src/trb_calibrated_20261030/mip_panel.py public_full
    python src/trb_calibrated_20261030/summarize_comparison.py

The 96 randomized small checks agree with exact recursion. The eight
preselected public conditions remain unresolved in the mixed-integer
formulation at its 20-second optimization allowance; all bounds and elapsed
times are retained in results/trb_calibrated_20261030/mip_request_ties.
The recursion proves all eight optima. This is a finite-scenario formulation
of the same model; it is an independently implemented comparison.

These modules use supplied numerical inputs and saved outputs. Publication
text, LaTeX sources, proofs and manuscript production scripts are excluded.

## Shared report errors in the separate surveillance source

The 2025 January ATL, CLT and DFW reconstruction supplies 35 distinct calendar/release-limit settings, evaluated at two response times (70 numerical conditions). Identical donor calendars across evaluation dates are counted once. All methods use a shared one-count error allowance and one new request per two-minute interval. Each exact comparison matches the respective receding-lookahead policy's maximum total requests.

Run the smoke stage before the full stage:

~~~shell
python src/trb_external_error_20261030/independent_panel.py smoke
python src/trb_external_error_20261030/independent_panel.py full
python src/trb_external_error_20261030/certify_independent.py smoke
python src/trb_external_error_20261030/certify_independent.py full
python src/trb_external_error_20261030/decision_trace.py
python src/trb_external_error_20261030/summarize_independent.py
~~~

Saved results retain all 70 conditions. Six-step lookahead is optimal in 67 conditions. Finite-depth lower bounds prove 60 candidates optimal; the other ten require full optimization to close the bound. A Dallas/Fort Worth trace uses identical service and reported counts to isolate the value of changing request order. Input hashes, grouping rules and represented windows are included in the design and results JSON files.

## Chronologically separated service-count ranges

The numerical extension uses January 3-18, 2025 for training, January 19-February 6 for calibration, and February 7-March 31 for evaluation. Airport-hour eligibility and group sizes use training dates. All 53 evaluation dates are kept. The saved design distinguishes 636 count vectors from 1,855 release-limit settings.

Run in this order:

~~~shell
python src/trb_transfer_20261030/service_mismatch.py
python src/trb_transfer_20261030/calibrated_service.py smoke
python src/trb_transfer_20261030/calibrated_service.py full
python src/trb_transfer_20261030/summarize_transfer.py
~~~

The 95 percent split-calibration rank assumes exchangeable calibration and next-test scores for a specified airport hour. Observed vector coverage is 617/636; the 90 percent alternative gives 605/636. The release calculation uses observed past counts, zero future-count requests and explicit physical limits. It preserves the early-release departures in 1,854/1,855 settings. The single exception, incomplete-group settings and all count-range alternatives are retained. The service-difference bound is independently checked on 110,052 trajectory pairs.

The separate finite-calendar cross-date smoke test is retained in results/trb_transfer_20261030/smoke.json. It evaluates synthetic bounded report errors against recorded future counts; it has not been expanded to a full run. Its source data, settings and outcomes are available for diagnosis.
