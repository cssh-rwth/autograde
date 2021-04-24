import math
import re
from copy import deepcopy
from dataclasses import dataclass, field
from functools import lru_cache
from typing import List, Optional, Tuple, Union
from zipfile import ZipFile

from dataclasses_json import config, dataclass_json
from marshmallow import fields

import autograde
from autograde.util import logger, float_equal, prune_join, camel_case, render, DateTime, now


@dataclass_json
@dataclass
class TeamMember:
    first_name: str
    last_name: str
    student_id: str


@dataclass_json
@dataclass(eq=False)
class UnitTestResult:
    id: str
    label: str
    target: List[str]
    score: float
    score_max: float
    messages: List[str]
    stdout: str
    stderr: str

    def __post_init__(self):
        if self.score_max < self.score:
            raise ValueError('score must not be bigger than score_max')

    def __eq__(self, other: 'UnitTestResult'):
        if not isinstance(other, UnitTestResult):
            return False

        if not float_equal(self.score, other.score):
            return False

        if not float_equal(self.score_max, other.score_max):
            return False

        for attr in ['id', 'label', 'target', 'messages', 'stdout', 'stderr']:
            if not getattr(self, attr, 0) == getattr(other, attr, 1):
                return False

        return True

    def pending(self) -> bool:
        return math.isnan(self.score)

    def failed(self) -> bool:
        return math.isclose(self.score, 0.) or self.score < 0.

    def partially_passed(self) -> bool:
        return not (self.pending() or self.failed() or self.passed())

    def passed(self) -> bool:
        return math.isclose(self.score, self.score_max)


@dataclass_json
@dataclass
class NotebookTestResult:
    title: str
    checksum: str
    team_members: List[TeamMember]
    artifacts: List[str]
    excluded_artifacts: List[str]
    unit_test_results: List[UnitTestResult]
    applied_patches: List[Tuple[str, DateTime, List[str]]] = field(default_factory=lambda: [])
    version: str = field(default_factory=lambda: autograde.__version__)
    timestamp: DateTime = field(
        default_factory=now,
        metadata=config(
            encoder=DateTime.isoformat,
            decoder=DateTime.fromisoformat,
            mm_field=fields.DateTime(format='iso')
        )
    )

    def __post_init__(self):
        if self.version != autograde.__version__:
            logger.debug(f'autograde has version {autograde.__version__} '
                         f'but this notebook test result was created by version {self.version}')

    def __iter__(self):
        return iter(self.unit_test_results)

    def format_members(self, *args, **kwargs):
        last_names = sorted((m.last_name for m in self.team_members))
        return prune_join(map(camel_case, last_names), *args, **kwargs)

    def copy(self):
        return deepcopy(self)

    def patch(self, patch: 'NotebookTestResult') -> 'NotebookTestResult':
        """
        Create a copy of self and patch results of given results object into it. NOTE that pending
        results are ignored.

        :param patch: results to be patched into self
        :return: patched copy
        """
        patched = self.copy()
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
        return NotebookTestSummary.from_results(self)


@dataclass_json
@dataclass
class NotebookTestSummary:
    tests: int
    failed: int
    passed: int
    partially_passed: int
    pending: int
    score: float
    score_max: float

    def __post_init__(self):
        if self.failed + self.passed + self.partially_passed + self.pending != self.tests:
            raise ValueError(f'there is a mismatch the number of failed({self.failed})/passed({self.passed})/partially '
                             f'passed({self.partially_passed})/pending({self.pending}) tests and '
                             f'the total number of tests({self.tests})')

        if not math.isnan(self.score) and not 0 <= self.score <= self.score_max:
            raise ValueError(f'{self.score} is no valid value for score. It must either be NaN or '
                             f'0 <= score <= {self.score_max}')

    @classmethod
    def from_results(cls, results: NotebookTestResult) -> 'NotebookTestSummary':
        score = sum(r.score for r in results.unit_test_results)
        score_max = sum(r.score_max for r in results.unit_test_results)
        return cls(
            tests=len(results.unit_test_results),
            failed=sum(r.failed() for r in results.unit_test_results),
            passed=sum(r.passed() for r in results.unit_test_results),
            partially_passed=sum(r.partially_passed() for r in results.unit_test_results),
            pending=sum(r.pending() for r in results.unit_test_results),
            score=math.nan if math.isnan(score) else max(0., score),
            score_max=math.nan if math.isnan(score_max) else max(0., score_max),
        )


class NotebookTestResultArchive:
    _supported_modes = {'r', 'a'}
    _re_patches = re.compile(r'results_.*\.json')
    _re_reports = re.compile(r'report(_rev_\d+)?\.html')
    _required_files = ['code.py', 'notebook.ipynb', 'results.json']
    _cache_size = 256

    def __init__(self, file, mode: str = 'r'):
        if mode not in self._supported_modes:
            raise ValueError(f'mode "{mode}" is not supported, chose one from {self._supported_modes}')

        self._modifications = 0
        self._zipfile = ZipFile(file, mode)

        # check if contents are complete
        files = set(self._zipfile.namelist())
        for f in self._required_files:
            if f not in files:
                raise KeyError(f'Archive does not cointain {f}')

        logger.debug(f'open archive at {self.filename} (mode: {mode})')

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        logger.debug(f'close {self.filename}')
        self._zipfile.close()

    def __repr__(self):
        return f'{type(self).__name__}(filename={self._zipfile.filename}, modifications={self._modifications})'

    def __hash__(self):
        return hash(self._zipfile) ^ hash(self._modifications)

    @property
    def filename(self):
        return self._zipfile.filename

    @property
    @lru_cache(_cache_size)
    def files(self) -> List[str]:
        return sorted(self._zipfile.namelist())

    @lru_cache(_cache_size)
    def load_file(self, path: str, encoding: Optional[str] = None) -> Union[bytes, str]:
        with self._zipfile.open(name=path, mode='r') as f:
            if enc := encoding:
                return f.read().decode(enc)
            return f.read()

    @property
    @lru_cache(_cache_size)
    def patch_count(self) -> int:
        return len(list(filter(self._re_patches.match, self.files)))

    @property
    @lru_cache(_cache_size)
    def report_count(self) -> int:
        return len(list(filter(self._re_reports.match, self.files)))

    def inject_patch(self, patch: NotebookTestResult):
        """Store results as patch in mounted results archive"""
        # add patch
        patch_name = f'results_patch_{self.patch_count + 1:02d}.json'
        with self._zipfile.open(patch_name, mode='w') as f:
            logger.debug(f'add {patch_name} to {self._zipfile.filename}')
            f.write(patch.to_json(indent=4).encode('utf-8'))
            f.flush()

        # add report revision
        if 'report.html' in self.files:
            self.inject_report()

        self._modifications += 1

    def inject_report(self):
        """Store results as patch in mounted results archive"""
        revision_name = 'report.html'
        if revision_name in self.files:
            revision_name = f'report_rev_{self.report_count:02d}.html'

        report = self._render_report()
        with self._zipfile.open(revision_name, mode='w') as f:
            logger.debug(f'add {revision_name} to {self._zipfile.filename}')
            f.write(report.encode('utf-8'))
            f.flush()

        self._modifications += 1

    @property
    @lru_cache(_cache_size)
    def results(self) -> NotebookTestResult:
        """Load results and apply patches from archive"""
        results = NotebookTestResult.from_json(self.load_file('results.json'))

        # apply patches
        for patch_path in sorted(filter(self._re_patches.match, self.files)):
            results = results.patch(NotebookTestResult.from_json(self.load_file(patch_path)))

        return results

    @property
    @lru_cache(_cache_size)
    def code(self) -> str:
        return self.load_file('code.py', encoding='utf-8')

    @property
    @lru_cache(_cache_size)
    def notebook(self) -> str:
        return self.load_file('notebook.ipynb', encoding='utf-8')

    def _render_report(self) -> str:
        return render('report.html', title='report', archive=self)

    @property
    @lru_cache(_cache_size)
    def report(self) -> str:
        return self._render_report()
