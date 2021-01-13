import io
import json
import math
import re
import warnings
from copy import deepcopy
from dataclasses import dataclass, field
from functools import lru_cache
from functools import wraps
from typing import List, Tuple
from zipfile import ZipFile

from dataclasses_json import dataclass_json

import autograde
from autograde.util import timestamp_utc_iso, prune_join, camel_case, render


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
            raise ValueError('patch must not have a different origin, i.e. checksum')

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


class NotebookTestResultArchive(ZipFile):
    _id_results = 'results.json'
    _re_patches = re.compile(r'results_.*\.json')
    _required_files = ['code.py', 'notebook.ipynb', _id_results]
    _cache_size = 256

    @wraps(ZipFile.__init__)
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        files = self.namelist()
        for f in self._required_files:
            if f not in files:
                raise FileNotFoundError(f'Archive does not cointain {f}')

        self._modifications = 0

    def __repr__(self):
        return f'{type(self).__name__}(filename={self.filename}, modifications={self._modifications})'

    def __hash__(self):
        return hash(super()) ^ hash(self._modifications)

    @property
    @lru_cache(_cache_size)
    def patch_count(self) -> int:
        return len(list(filter(self._re_patches.match, self.namelist())))

    def inject_patch(self, patch: NotebookTestResult):
        """Store results as patch in mounted results archive"""

        with self.open(f'results_patch_{self.patch_count + 1:02d}.json', mode='w') as f:
            f.write(json.dumps(patch.to_dict(), indent=4).encode('utf-8'))

        self._modifications += 1

        # update report if it exists
        if 'report.html' in self.namelist():
            self._render_report()

    def _render_report(self) -> str:
        report = render('report.html', title='report', archive=self)
        with warnings.catch_warnings():
            warnings.simplefilter('ignore', UserWarning)
            with self.open('report.html', mode='w') as f:
                f.write(report.encode('utf-8'))
        return report

    @property
    @lru_cache(_cache_size)
    def results(self) -> NotebookTestResult:
        """Load results and apply patches from archive"""
        with self.open('results.json', mode='r') as f:
            results = NotebookTestResult.from_json(f.read())

        # apply patches
        for patch_path in sorted(filter(self._re_patches.match, self.namelist())):
            with self.open(patch_path, mode='r') as f:
                results = results.patch(NotebookTestResult.from_json(f.read()))

        return results

    @property
    @lru_cache(_cache_size)
    def code(self) -> str:
        with self.open('code.py', mode='r') as f, io.TextIOWrapper(f) as t:
            return t.read()

    @property
    @lru_cache(_cache_size)
    def notebook(self) -> str:
        with self.open('notebook.ipynb', mode='r') as f, io.TextIOWrapper(f) as t:
            return t.read()

    @property
    @lru_cache(_cache_size)
    def report(self) -> str:
        if 'report.html' not in self.namelist():
            return self._render_report()

        with self.open('report.html') as f:
            return f.read().decode('utf-8')
