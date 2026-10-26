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
    c=Corpus(); m=Model(['<search>rewrite</search>','<search>next</search>'])
    t,meta=run(c,m,'first-search')
    assert c.queries==['original question','next']
    assert meta['search_queries'][0]==dict(model_query='rewrite',executed_query='original question')
    assert t.steps[0].action.content=='rewrite'

def test_control_search_unchanged():
    c=Corpus(); run(c,Model(['<search>x</search>','<answer>yes</answer>']),'first-search','control')
    assert c.queries==['x']

def test_final_answer_uses_last_tool_result_and_one_extra_call():
    c=Corpus(); m=Model(['<search>a</search>','<search>b</search>','<answer>yes</answer>'])
    t,meta=run(c,m,'final-answer')
    assert t.final_answer=='yes' and len(t.steps)==3
    assert c.queries==['a','b']
    assert 'evidence' in m.calls[-1][-2]['content']
    assert meta['extra_answer_attempted']

def test_extra_search_is_not_executed():
    c=Corpus(); m=Model(['<search>a</search>','<search>b</search>','<search>forbidden</search>'])
    t,meta=run(c,m,'final-answer')
    assert c.queries==['a','b'] and t.final_answer is None
    assert t.terminated_reason=='forced_answer_invalid'

def test_no_extra_call_after_answer_or_format_failure():
    for outputs in [['<answer>yes</answer>'], ['bad','bad']]:
        m=Model(outputs); t,meta=run(Corpus(),m,'final-answer')
        assert len(m.calls)==len(outputs) and not meta['extra_answer_attempted']

def test_combined_applies_both_first_search_and_final_answer():
    c=Corpus(); m=Model(['<search>rewrite</search>','<search>next</search>','<answer>yes</answer>'])
    t,meta=run(c,m,'combined')
    assert c.queries==['original question','next']
    assert meta['search_queries'][0]==dict(model_query='rewrite',executed_query='original question')
    assert t.final_answer=='yes' and meta['extra_answer_attempted']

def test_combined_control_leaves_both_effects_off():
    c=Corpus(); m=Model(['<search>x</search>','<search>y</search>'])
    t,meta=run(c,m,'combined','control')
    assert c.queries==['x','y']
    assert t.final_answer is None and not meta['extra_answer_attempted']
