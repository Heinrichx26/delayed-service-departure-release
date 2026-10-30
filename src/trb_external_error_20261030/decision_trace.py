"""Trace an independently identified nonmonotone lookahead condition."""
from independent_panel import ROOT, OUT, read, save, prepare, BudgetedReports, TimedLookahead
from robust_reports import physical


def run():
    rows = read(OUT/'full.json')['rows']
    row = next(r for r in rows if r['airport']=='DFW' and r['hour']==18 and
               r['delay']==2 and r['cap']==5)
    c = read(ROOT/row['input_path'])
    p = prepare(c['service'], c['n'], (row['cap'],)*len(c['service'][0]), c['lag'])
    models = dict(six=TimedLookahead(p,c['lag']+2,2,1,1,depth=6,
                                   seconds_limit=30,state_limit=200000),
                  exact=BudgetedReports(p,c['lag']+2,2,1,1,total_requests=6,
                                        seconds_limit=30,state_limit=200000))
    for key, model in models.items():
        assert model.solve()['value'] == row['methods']['6'][
            'lookahead' if key=='six' else 'exact']['value']
    model = models['six']
    state = model.initial()
    trace = []
    while state[0] < model.h:
        t,e,pending,queried,u = state
        model.value(*state)
        batch = model.choice[state]
        z = model.release(t,e,u)
        children = model.transitions(t,e,pending,queried,batch)
        child = max(children,key=lambda ch:z-ch[5]+model.value(t+1,ch[2],ch[3],ch[4],z))
        due = [j for due,j in pending if due==t+1]
        trace.append(dict(t=t,requests=list(batch),due=due,answers=list(child[1]),
                          release=z,reference=child[5]))
        state = (t+1,child[2],child[3],child[4],z)
    actual = model.members(state[1])[0]
    path = p['paths'][actual]
    residuals = {j:y-path[j] for r in trace for j,y in zip(r['due'],r['answers'])}
    assert sum(abs(v) for v in residuals.values()) <= 1
    assert physical([r['release'] for r in trace],path,c['lag']) == p['refs'][actual]
    assert sum(r['release']-r['reference'] for r in trace) == 166
    exact = models['exact'];state = exact.initial();other = []
    while state[0] < exact.h:
        t,e,pending,queried,u = state
        exact.value(*state);batch=exact.choice[state];z=exact.release(t,e,u)
        due=[j for due,j in pending if due==t+1]
        answer=tuple(path[j]+residuals.get(j,0) for j in due)
        child=next(ch for ch in exact.transitions(t,e,pending,queried,batch)
                   if ch[0]==path[t] and ch[1]==answer and ch[2][actual]>=0)
        other.append(dict(t=t,requests=list(batch),due=due,answers=list(answer),
                          release=z,reference=child[5]))
        state=(t+1,child[2],child[3],child[4],z)
    assert physical([r['release'] for r in other],path,c['lag']) == p['refs'][actual]
    result=dict(passed=True,condition={k:row[k] for k in
        ['case_id','airport','day','hour','n','cap','delay','input_sha256']},
        actual_calendar_index=actual,service=list(path),
        report_residuals=residuals,six_trace=trace,exact_same_calendar_trace=other,
        six_worst_value=166,exact_worst_value=145,
        exact_same_calendar_value=sum(r['release']-r['reference'] for r in other),
        first_release_difference=next((a['t'] for a,b in zip(trace,other)
                                      if a['release']!=b['release']),None))
    save(OUT/'decision_trace.json',result)
    print(result)


if __name__ == '__main__':
    run()
