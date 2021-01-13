import math
from copy import deepcopy
from dataclasses import dataclass, field
from typing import List, Tuple

from dataclasses_json import dataclass_json

import autograde
from autograde.util import timestamp_utc_iso, prune_join, camel_case


@dataclass_json
@dataclass
class TeamMember:
    first_name: str
    last_name: str
    student_id: str


@dataclass_json
@dataclass
class UnitTestResult:
    id: str
    label: str
    target: List[str]
    score: float
    score_max: float
    messages: List[str]
    stdout: str
    stderr: str

    def pending(self) -> bool:
        return math.isnan(self.score)

    def failed(self) -> bool:
        return math.isclose(self.score, 0.)

    def partially_passed(self) -> bool:
        return not (self.pending() or self.failed() or self.passed())

    def passed(self) -> bool:
        return math.isclose(self.score, self.score_max)


@dataclass_json
@dataclass
class NotebookTestResult:
    title: str
    notebook: str
    checksum: str
    team_members: List[TeamMember]
    artifacts: List[str]
    excluded_artifacts: List[str]
    unit_test_results: List[UnitTestResult]
    applied_patches: List[Tuple[str, str, List[str]]] = field(default_factory=lambda: [])
    version: str = field(default_factory=lambda: autograde.__version__)
    timestamp: str = field(default_factory=timestamp_utc_iso)

    def __iter__(self):
        return iter(self.unit_test_results)

    def format_members(self, *args, **kwargs):
        last_names = sorted((m.last_name for m in self.team_members))
        return prune_join(map(camel_case, last_names), *args, **kwargs)

    def patch(self, patch: 'NotebookTestResult') -> 'NotebookTestResult':
        """
        Create a copy of self and patch results of given results object into it. NOTE that pending
        results are ignored.

        :param patch: results to be patched into self
        :return: patched copy
        """
        patched = deepcopy(self)
        results = {r.id: r for r in self.unit_test_results}

        if not patched.checksum == patch.checksum:
            raise ValueError('patch must not have a different origin aka checksum!')

        change_list = []
        for result in patch.unit_test_results:
            if result != results.get(result.id) and not result.pending():
                results[result.id] = result
                change_list.append(result.id)

        patched.unit_test_results = list(results.values())
        patched.applied_patches.append((patch.title, patch.timestamp, change_list))

        return patched

    def summarize(self) -> 'NotebookTestSummary':
        return NotebookTestSummary(self)


@dataclass_json
@dataclass(init=False)
class NotebookTestSummary:
    tests: int
    failed: int
    passed: int
    pending: int
    score: float
    score_max: float

    def __init__(self, results: NotebookTestResult):
        self.tests = len(results.unit_test_results)
        self.failed = sum(r.failed() for r in results.unit_test_results)
        self.passed = sum(r.passed() for r in results.unit_test_results)
        self.pending = sum(r.pending() for r in results.unit_test_results)
        self.score = sum(r.score for r in results.unit_test_results)
        self.score_max = sum(r.score_max for r in results.unit_test_results)
