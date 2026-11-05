# NOTE (2026-11-05): first-search forcing and forced-answer-on-timeout are
# now unconditional defaults in backend.agent_loop.run_episode (see
# docs/EXPERIMENT_RECORD.md's "combined" ablation result -- Answer F1
# +0.0234, t=2.811, significant). experiment_episode()'s own SearchProxy/
# post-hoc forcing can no longer produce a real control-vs-treatment
# difference for 'first-search'/'final-answer'/'combined': run_episode
# applies both before SearchProxy or the post-processing ever sees the
# trajectory. These tests now assert that degenerate (identical) behavior,
# not a live ablation -- this script's 3 experiment types are obsolete for
# any future re-measurement.
from scripts.run_agent_ablation import experiment_episode

class Corpus:
    papers={'p':dict(title='T', abstract='A')}
    def __init__(self): self.queries=[]
    def search(self,p,q): self.queries.append(q); return '[p_0] (M) evidence'
    def read(self,p,q): return '[p_0] (M) full'

class Model:
    def __init__(self,outputs): self.outputs=iter(outputs); self.calls=[]
    def __call__(self,messages):
        self.calls.append([dict(m) for m in messages]); return next(self.outputs)

def run(c,m,exp,arm='treatment',rounds=2):
    return experiment_episode('q','original question','p',c,m,exp,arm,max_rounds=rounds)

def test_only_first_search_replaced():
    c=Corpus(); m=Model(['<search>rewrite</search>','<search>next</search>','<answer>yes</answer>'])
    t,meta=run(c,m,'first-search')
    assert c.queries==['original question','next']
    # run_episode overrides the first search's content before SearchProxy ever
    # sees it, so SearchProxy's own bookkeeping can no longer observe the
    # model's true original query -- both fields now read the forced value.
    assert meta['search_queries'][0]==dict(model_query='original question',executed_query='original question')
    assert t.steps[0].action.content=='original question'

def test_control_search_unchanged():
    # 'control' no longer means "unforced": run_episode forces the first
    # search unconditionally regardless of arm, so control == treatment here.
    c=Corpus(); run(c,Model(['<search>x</search>','<answer>yes</answer>']),'first-search','control')
    assert c.queries==['original question']

def test_final_answer_uses_last_tool_result_and_one_extra_call():
    # meta['extra_answer_attempted'] is set by experiment_episode()'s own
    # post-hoc forcing, which never fires now (see module note) -- not
    # checked here; what matters is run_episode's own internal forced turn
    # actually used the prior tool result and produced an answer.
    c=Corpus(); m=Model(['<search>a</search>','<search>b</search>','<answer>yes</answer>'])
    t,meta=run(c,m,'final-answer')
    assert t.final_answer=='yes' and len(t.steps)==3
    # First search is forced regardless of experiment type (see module note).
    assert c.queries==['original question','b']
    assert 'evidence' in m.calls[-1][-2]['content']

def test_extra_search_is_not_executed():
    c=Corpus(); m=Model(['<search>a</search>','<search>b</search>','<search>forbidden</search>'])
    t,meta=run(c,m,'final-answer')
    assert c.queries==['original question','b'] and t.final_answer is None
    assert t.terminated_reason=='forced_answer_invalid'

def test_no_extra_call_after_answer_or_format_failure():
    for outputs in [['<answer>yes</answer>'], ['bad','bad']]:
        m=Model(outputs); t,meta=run(Corpus(),m,'final-answer')
        assert len(m.calls)==len(outputs) and not meta['extra_answer_attempted']

def test_combined_applies_both_first_search_and_final_answer():
    # meta['extra_answer_attempted'] is set by experiment_episode()'s own
    # post-hoc forcing, which never fires now (run_episode's terminated_reason
    # is never 'max_rounds' anymore) -- no longer a meaningful check here;
    # what matters is that run_episode's own internal forced turn produced
    # an answer, which it does.
    c=Corpus(); m=Model(['<search>rewrite</search>','<search>next</search>','<answer>yes</answer>'])
    t,meta=run(c,m,'combined')
    assert c.queries==['original question','next']
    assert t.final_answer=='yes'

def test_combined_control_leaves_both_effects_off():
    # 'control' no longer suppresses either effect -- see module note.
    c=Corpus(); m=Model(['<search>x</search>','<search>y</search>','<search>z</search>'])
    t,meta=run(c,m,'combined','control')
    assert c.queries==['original question','y']
    assert t.final_answer is None
