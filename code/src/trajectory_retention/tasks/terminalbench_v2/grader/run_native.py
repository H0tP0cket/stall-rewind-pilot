"""Run the task's native pytest command inside the disposable grading container.

Equivalent of the native ``uvx ... pytest --ctrf /logs/verifier/ctrf.json
/tests/test_outputs.py <flags>`` from the task's working directory, using the
pre-built verifier environment (python -I; the recorder is the only addition).
Grader hardening (disclosed): a private empty ini file, rootdir and confcutdir
fixed to /tests so worker-created ini/conftest files outside /tests are not read.
"""
import json
import sys

sys.path.insert(0, "/tb-grader")
import pytest  # noqa: E402
import tb_recorder  # noqa: E402

flags = json.loads(sys.argv[1])
args = ["-c", "/tb-grader/pytest.ini", "--rootdir=/tests", "--confcutdir=/tests",
        "--ctrf", "/logs/verifier/ctrf.json", "/tests/test_outputs.py", *flags]
code = int(pytest.main(args, plugins=[tb_recorder]))
with open("/tb-grader/out/exit.json", "w") as handle:
    json.dump({"exit_code": code, "args": args}, handle)
sys.exit(code)
