from scripts.load_test_api import SSEParser, summarize

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
