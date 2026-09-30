import logging

from swlp.logging import ExtrasFormatter


def test_extras_formatter_appends_extra_fields():
    record = logging.makeLogRecord(
        {"msg": "swlp_residency_plan", "levelname": "INFO", "resident_count": 2}
    )
    line = ExtrasFormatter("%(message)s").format(record)
    assert line == "swlp_residency_plan resident_count=2"


def test_extras_formatter_plain_record_unchanged():
    record = logging.makeLogRecord({"msg": "hello"})
    assert ExtrasFormatter("%(message)s").format(record) == "hello"
