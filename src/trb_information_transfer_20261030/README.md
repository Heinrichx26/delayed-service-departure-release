# Delayed report selection over calibrated service ranges

The input design reuses the frozen training/calibration/test dates in data/trb_transfer_20261030. Every later date is retained. Algorithms use integer count ranges and a shared absolute report-error allowance.

Run from the repository root:

    python src/trb_information_transfer_20261030/range_release.py
    python src/trb_information_transfer_20261030/evaluate_ranges.py smoke
    python src/trb_information_transfer_20261030/evaluate_ranges.py full
    python src/trb_information_transfer_20261030/adaptive_transfer.py smoke
    python src/trb_information_transfer_20261030/adaptive_transfer.py full
    python src/trb_information_transfer_20261030/matched_ablation.py smoke
    python src/trb_information_transfer_20261030/matched_ablation.py full

Plot saved results without rerunning any experiment:

    python src/trb_information_transfer_20261030/plot_information_value.py

The range release is valid for all compatible integer calendars. Request optimization uses training calendars for its objective and the range calculation for its release constraints. All shortfalls outside the calibrated ranges remain in the results. The matched ablation fixes all three actual requests for both error rules.
