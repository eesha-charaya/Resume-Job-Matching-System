import json

from resume_job_matcher.cli import main


def test_clean_command(capsys):
    assert main(["clean", "<p>Python &amp; Data Science</p>"]) == 0
    assert capsys.readouterr().out.strip() == "Python & Data Science"


def test_extract_command(capsys):
    args = [
        "extract",
        "Bachelor's degree and 2+ years of experience",
        "--title",
        "Senior Engineer",
    ]
    assert main(args) == 0
    result = json.loads(capsys.readouterr().out)
    assert result["required_degree"] == 1
    assert result["seniority_level"] == 3
    assert result["min_years_experience"] == 2


def test_metrics_command(capsys):
    assert main(
        [
            "metrics",
            "--retrieved",
            "j1,j2,j3",
            "--relevant",
            "j1,j3",
            "--k",
            "1,3",
        ]
    ) == 0
    result = json.loads(capsys.readouterr().out)
    assert result["precision"]["1"] == 1.0
    assert result["recall"]["3"] == 1.0
