from resume_job_matcher import preprocess_description


def test_preprocess_description_strips_html_and_normalizes_text():
    raw = "<p>Hello&nbsp;&nbsp;world</p><img src='x.png'><p>Data\u2014Science</p>"
    assert preprocess_description(raw) == "Hello world Data-Science"


def test_preprocess_description_handles_double_encoded_html():
    assert preprocess_description("&amp;lt;b&amp;gt;Python&amp;lt;/b&amp;gt;") == "Python"


def test_preprocess_description_handles_empty_input():
    assert preprocess_description("") == ""
    assert preprocess_description(None) == ""
