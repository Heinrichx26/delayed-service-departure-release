"""Exact nondominated waiting-request pairs; no selected exchange rate."""
from itertools import combinations
from time import perf_counter
from band_design import BandDesign, StateLimit


def prune(mapping):
    """Retain smaller waiting at each request count, then remove dominance."""
    by_count = {}
    for (waiting, count), payload in mapping.items():
        if count not in by_count or waiting < by_count[count][0]:
            by_count[count] = waiting, payload
    result = {}
    smallest = float('inf')
    for count in sorted(by_count):
        waiting, payload = by_count[count]
        if waiting < smallest:
            result[(waiting,count)] = payload
            smallest = waiting
    return result


class FrontierDesign(BandDesign):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.witness = {}

    def value(self, t, mask, pending, previous):
        state = (t,mask,pending,previous)
        if len(self.values) > self.state_limit or (
                len(self.values)%128 == 0 and perf_counter()-self.started > self.seconds_limit):
            raise StateLimit(f'frontier states={len(self.values)}; seconds={perf_counter()-self.started:.2f}')
        if t == self.h:
            self.values[state] = ((0,0),)
            self.witness[state] = {(0,0):((),())}
            return ((0,0),)
        shortcut = self.suffix(t,mask,previous)
        if shortcut is not None:
            self.values[state] = (shortcut,)
            self.witness[state] = {shortcut:((),None)}
            return (shortcut,)
        release = max(previous,self.bound(mask,t))
        assert release-previous <= self.problem['caps'][t]
        stage = self.weight(mask)*release-self.ref(mask,t)
        candidates = self.candidates(t,mask,pending)
        answers = {}
        for size in range(min(self.budget,len(candidates))+1):
            for request in combinations(candidates,size):
                partial = {(stage,self.weight(mask)*size):()}
                children = self.transition(t,mask,pending,request)
                for sub,future in children:
                    child = self.value(t+1,sub,future,release)
                    combined = {}
                    for parent,selected in partial.items():
                        for pair in child:
                            total = (parent[0]+pair[0],parent[1]+pair[1])
                            combined.setdefault(total,selected+(pair,))
                    partial = prune(combined)
                for pair,selected in partial.items():
                    answers.setdefault(pair,(request,selected))
        answers = prune(answers)
        self.values[state] = tuple(sorted(answers))
        self.witness[state] = answers
        return self.values[state]

    def solve(self):
        previous = self.problem['prefix'][-1] if self.problem['prefix'] else 0
        combined = {(0,0):()}
        groups = self.initial_groups()
        for mask in groups:
            next_values = {}
            for old,selected in combined.items():
                for new in self.value(self.t0,mask,(),previous):
                    pair = (old[0]+new[0],old[1]+new[1])
                    next_values.setdefault(pair,selected+(new,))
            combined = prune(next_values)
        self.initial_witness = combined
        return dict(optimal=True,frontier=[
            dict(weighted_waiting=a,weighted_queries=c,
                 waiting=a*self.problem['case']['delta']/self.denominator/self.problem['n'],
                 expected_queries=c/self.denominator)
            for a,c in sorted(combined)],
            states=len(self.values),max_frontier=max(map(len,self.values.values())),
            seconds=perf_counter()-self.started)

    def replay_pair(self, pair):
        chosen = self.initial_witness[pair]
        groups = self.initial_groups()
        rows = []
        for s,path in enumerate(self.paths):
            group_index = next(i for i,m in enumerate(groups) if m>>s&1)
            mask = groups[group_index]
            pending = ()
            desired = chosen[group_index]
            release = list(self.problem['prefix'])
            queries = []
            previous = release[-1] if release else 0
            shortcut = False
            for t in range(self.t0,self.h):
                if shortcut:
                    request = ()
                else:
                    request,children = self.witness[(t,mask,pending,previous)][desired]
                    shortcut = children is None
                current = max(previous,self.bound(mask,t))
                release.append(current)
                queries.extend(dict(dispatch=t,arrival=t+self.delay,slot=j) for j in request)
                transitions = self.transition(t,mask,pending,request)
                index = next(i for i,(m,p) in enumerate(transitions) if m>>s&1)
                if not shortcut:
                    desired = children[index]
                mask,pending = transitions[index]
                previous = current
            done,inventory,_ = self.problem_physical(release,path)
            rows.append(dict(s=s,queries=queries,release=release,completed=done,
                post_release=self.problem['case']['delta']*sum(inventory[self.t0:])/self.problem['n']))
        observed = (
            sum(self.weights[r['s']]*sum(r['release'][t]-r['completed'][t] for t in range(self.t0,self.h)) for r in rows),
            sum(self.weights[r['s']]*len(r['queries']) for r in rows))
        assert observed == pair
        return rows

    def problem_physical(self,release,path):
        from band_design import old
        answer=old.base.physical(release,path,self.problem['n'],self.problem['lag'])
        return answer
