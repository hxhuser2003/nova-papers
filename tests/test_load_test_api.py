import json
from scripts.load_test_api import SSEParser, admission_counts, responded, summarize

def test_split_utf8_multiline_and_crlf():
    p = SSEParser()
    data = 'event: token\r\ndata: {"text":\r\ndata: "你好"}\r\n\r\n'.encode()
    events = []
    for byte in data:
        events += p.feed(bytes([byte]))
    assert events == [('token', '{"text":\n"你好"}')]

def test_errors_do_not_inflate_success_throughput():
    rows = [{'ok': True, 'total_s': 2}, {'ok': False, 'total_s': 8}]
    s = summarize(rows, 10, 2)
    assert s['successful_rps'] == .1
    assert s['error_rate'] == .5
    assert s['success_p95_s'] == 2

def test_admission_counts_only_counts_lines_inside_the_window(tmp_path):
    log = tmp_path / 'app_server.log'
    log.write_text('\n'.join([
        'INFO:     chat_timing ' + json.dumps({'stage': 'queue_full', 'wall_time': 100}),
        'INFO:     chat_timing ' + json.dumps({'stage': 'admission_acquired', 'wall_time': 101}),
        'INFO:     chat_timing ' + json.dumps({'stage': 'queue_full', 'wall_time': 200}),  # outside window
        'not a chat_timing line at all',
    ]))
    counts = admission_counts(str(log), wall_start=99, wall_end=150)
    assert counts == {'queue_full': 1, 'queue_timeout': 0, 'admission_acquired': 1}

def test_admission_counts_without_a_log_path_returns_empty():
    assert admission_counts(None, 0, 1) == {}

def test_admission_counts_missing_file_returns_empty():
    assert admission_counts('/no/such/file.log', 0, 1) == {}

def test_responded_is_true_when_the_agent_fails_but_the_service_answered_coherently():
    row = {'http_status': 200, 'events': [
        {'kind': 'status', 'seconds': 0, 'data': {}},
        {'kind': 'error', 'seconds': 7, 'data': {'message': 'forced_answer_invalid'}},
    ]}
    assert responded(row) is True

def test_responded_is_false_when_the_service_never_replies():
    assert responded({'http_status': None, 'events': []}) is False
    assert responded({'http_status': 502, 'events': []}) is False
    assert responded({'http_status': 200, 'events': [{'kind': 'status', 'seconds': 0, 'data': {}}]}) is False
