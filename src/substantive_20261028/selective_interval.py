"""Select interval confirmations using certified current release bounds.

The policy implements the all-available-information release on covered,
completing calendars. It does not claim minimum expected query cost.
"""
from functools import lru_cache

import holdout_box_smoke as physical


def choose_confirmation(active, n, caps, lag, t, previous, lead, known):
    h = len(caps)
    available = [j for j in range(t, min(h, t + lead + 1)) if j not in known]
    candidates = [j for j in available
                  if max(hi[j] for lo, hi in active) > min(lo[j] for lo, hi in active)]
    earliest = physical.acquisition.base.earliest(n, caps)
    arrived = [earliest[j - lag] if j >= lag else 0 for j in range(h)]
    cap_sum, total = [], 0
    for cap in caps:
        total += cap
        cap_sum.append(total)

    @lru_cache(None)
    def bound(service):
        done, completions = 0, []
        for arrival, capacity in zip(arrived, service):
            done = min(arrival, done + capacity)
            completions.append(done)
        return max(completions[min(v + lag, h - 1)] - cap_sum[v] + cap_sum[t]
                   for v in range(t, h))

    upper_bounds = [bound(tuple(hi)) for lo, hi in active]
    upper = max(previous, max(upper_bounds))
    lower_bounds = []
    for lo, hi in active:
        mixed = hi[:]
        for j in available:
            mixed[j] = lo[j]
        lower_bounds.append(bound(tuple(mixed)))
    lower = max(previous, min(lower_bounds))
    assert lower <= upper
    certificate = {'upper_release': upper, 'lower_release': lower,
                   'remaining_available_coordinates': len(candidates)}
    if lower == upper or not candidates:
        certificate['stop_reason'] = 'equal_release_bounds' if lower == upper else 'all_available_information_obtained'
        return None, certificate
    scores = []
    for j in candidates:
        reduction = 0
        for (lo, hi), old in zip(active, upper_bounds):
            if hi[j] == lo[j]:
                continue
            changed = hi[:]
            changed[j] = lo[j]
            reduction += max(0, old - bound(tuple(changed)))
        width = max(hi[j] for lo, hi in active) - min(lo[j] for lo, hi in active)
        scores.append((reduction, width, -j, j))
    selected = max(scores)[-1]
    certificate['selection_rule'] = 'sum of corner deadline sensitivities, then response range, then earliest slot'
    return selected, certificate


def selective_controller(case, caps, radius, lead):
    actual, h = case['actual_service'], len(caps)
    initial = physical.boxes(case, radius)
    covered = any(all(a <= v <= b for a, v, b in zip(lo, actual, hi)) for lo, hi in initial)
    active = [(lo[:], hi[:]) for lo, hi in initial
              if physical.reference(case, caps, hi)[-1] == case['n']]
    eligible_initial = len(active)
    cumulative, previous, fallback, snapshots, queries, known = [], 0, None, [], [], {}

    def intersect(boxes, j, value):
        result = []
        for lo, hi in boxes:
            if lo[j] <= value <= hi[j]:
                lo, hi = lo[:], hi[:]
                lo[j] = hi[j] = value
                if physical.reference(case, caps, hi)[-1] == case['n']:
                    result.append((lo, hi))
        return result

    for t in range(h):
        certificate = None
        if fallback is None:
            if t and t - 1 not in known:
                known[t - 1] = actual[t - 1]
                active = intersect(active, t - 1, actual[t - 1])
            if not active:
                fallback = {'decision_slot': t, 'coordinate': t - 1 if t else None,
                            'reason': 'No completing calendar remains after free observation'}
        if fallback is None:
            while True:
                j, certificate = choose_confirmation(active, case['n'], caps, case['lag'],
                                                      t, previous, lead, known)
                if j is None:
                    break
                assert j >= t and max(0, j - lead) <= t and j not in known
                known[j] = actual[j]
                queries.append({'t': t, 'j': j, 'value': actual[j]})
                active = intersect(active, j, actual[j])
                if not active:
                    fallback = {'decision_slot': t, 'coordinate': j,
                                'reason': 'No completing calendar remains after confirmation'}
                    break
        if fallback is not None:
            previous = min(case['n'], previous + caps[t])
        else:
            action = certificate['upper_release']
            assert previous <= action <= min(case['n'], previous + caps[t])
            previous = action
        cumulative.append(previous)
        snapshots.append({'t': t, 'compatible_boxes': len(active),
                          'known_coordinates': sorted(known), 'cumulative_release': previous,
                          'release_certificate': certificate})
    result = physical.evaluate(case, caps, cumulative, fallback, len(queries))
    if covered and result['actual_reference_clears']:
        assert not result['fallback_used'] and result['completion_preserving']
    result.update(radius=radius, lead=lead, method='selective_available',
                  full_calendar_covered=covered, initial_boxes=len(initial),
                  initial_completing_boxes=eligible_initial, fixed_menu_fee=None,
                  selected_menu=None, queries=queries, snapshots=snapshots)
    return result
