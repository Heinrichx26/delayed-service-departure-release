"""Summarize saved service-range results without rerunning release decisions."""
from statistics import mean
from calibrated_service import ROOT,DATA,OUT,read,save,sha


def summarize(rows):
    windows={(r['airport'],r['hour'],r['day']):r for r in rows}
    result=dict(windows=len(windows),settings=len(rows),
                raw_covered=sum(r['raw']['covered'] for r in windows.values()),
                calibrated90_covered=sum(r['calibrated90']['covered'] for r in windows.values()),
                calibrated_covered=sum(r['calibrated']['covered'] for r in windows.values()),
                raw_preserved=sum(r['raw']['takeoffs_preserved'] for r in rows),
                calibrated90_preserved=sum(r['calibrated90']['takeoffs_preserved'] for r in rows),
                calibrated_preserved=sum(r['calibrated']['takeoffs_preserved'] for r in rows),
                reference_complete=sum(r['calibrated']['reference_completes_group'] for r in rows))
    for name in ('raw','calibrated90','calibrated'):
        v=[r for r in rows if r[name]['takeoffs_preserved'] and r[name]['reference_completes_group']]
        result[name+'_complete_preserved']=len(v)
        result[name+'_holding_seconds_per_departure']=mean(120*r[name]['gateholding_aircraftslots']/r['n'] for r in v)
        assert all(r[name]['early_aircraftslots']-r[name]['postrelease_aircraftslots']==r[name]['gateholding_aircraftslots'] for r in v)
    return result


def main():
    d=read(DATA/'calibrated_design.json');full=read(OUT/'calibrated_full.json')
    assert read(OUT/'calibrated_smoke.json')['passed'] and read(OUT/'bound_smoke.json')['passed']
    assert full['passed'] and len(full['rows'])==1855
    assert len(d['training_dates'])==16 and len(d['calibration_dates'])==19 and len(d['evaluation_dates'])==53
    assert max(d['training_dates'])<min(d['calibration_dates']) and max(d['calibration_dates'])<min(d['evaluation_dates'])
    for row in full['rows']:
        assert sha(ROOT/row['input_path'])==row['input_sha256']
        for name in ('raw','calibrated90','calibrated'):
            if row[name]['covered']:
                assert row[name]['takeoffs_preserved']
    summary=dict(passed=True,all=summarize(full['rows']),
                 airports={a:summarize([r for r in full['rows'] if r['airport']==a]) for a in ('ATL','CLT','DFW')},
                 interpretation='Chronologically separated historical reconstruction; all 53 evaluation dates retained at every donor-eligible airport-hour',
                 units='Count coverage uses 636 distinct airport-date-hour vectors; takeoff preservation uses 1855 release-limit settings; holding uses group-complete settings with equal takeoffs',
                 requests=0,travel_minutes=12,report_errors='No future counts requested',
                 calibration_level=0.95,probability_condition='Exchangeability of calibration and test scores for one airport-hour',
                 residual=[dict(airport=r['airport'],hour=r['hour'],day=r['day'],n=r['n'],cap=r['cap'],
                               maximum_shortfall=r['calibrated']['maximum_shortfall'],
                               accumulated_shortfall_aircraftminutes=2*r['calibrated']['shortfall_aircraftslots'],
                               final_reference=r['calibrated']['final_reference'],final_departures=r['calibrated']['final_departures'])
                           for r in full['rows'] if not r['calibrated']['takeoffs_preserved']],
                 sha256={p.relative_to(ROOT).as_posix():sha(p) for p in [DATA/'calibrated_design.json',OUT/'calibrated_full.json',OUT/'bound_smoke.json']})
    save(OUT/'summary.json',summary)
    print(summary)


if __name__=='__main__':
    main()
