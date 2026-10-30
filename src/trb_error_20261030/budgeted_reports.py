"""Exact minimax decisions with an additional hard whole-period request cap."""
from robust_reports import RobustReports

class BudgetedReports(RobustReports):
    def __init__(self,*args,total_requests,**kwargs):
        self.total_requests=total_requests
        super().__init__(*args,**kwargs)

    def batches(self,t,e,pending,queried):
        used=queried>>self.h
        for batch in super().batches(t,e,pending,queried):
            if used+len(batch)<=self.total_requests:
                yield batch

    def transitions(self,t,e,pending,queried,batch):
        used=(queried>>self.h)+len(batch)
        children=super().transitions(t,e,pending,queried,batch)
        mask=(1<<self.h)-1
        return tuple((a,y,en,pn,(q&mask)|(used<<self.h),ref)
                     for a,y,en,pn,q,ref in children)
