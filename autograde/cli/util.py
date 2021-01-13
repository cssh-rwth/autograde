#!/usr/bin/env python3
import base64
import io
import json
import math
import re
from difflib import SequenceMatcher
from itertools import combinations
from pathlib import Path
from typing import List, Dict
from zipfile import ZipFile

# ensure matplotlib uses the right backend (this has to be done BEFORE import of pyplot!)
import matplotlib as mpl

mpl.use('Agg')

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from jinja2 import Environment, PackageLoader, select_autoescape
from scipy.linalg import LinAlgError
from scipy.stats import norm

import autograde
from autograde.test_result import NotebookTestResult
from autograde.static import CSS, FAVICON
from autograde.util import logger, timestamp_utc_iso

# Globals and constants variables.
FAVICON = base64.b64encode(FAVICON).decode('utf-8')
JINJA_ENV = Environment(
    loader=PackageLoader('autograde', 'templates'),
    autoescape=select_autoescape(['html', 'xml']),
    trim_blocks=True,
    lstrip_blocks=True
)


def b64str(data) -> str:
    """Convert bytes like in base64 encoded utf-8 string"""
    return base64.b64encode(data).decode('utf-8')


def list_results(path='.', prefix='results') -> List[Path]:
    """List all results archives at given location"""
    path = Path(path).expanduser().absolute()

    if path.is_file():
        return [path]

    return sorted(path.rglob(f'{prefix}_*.zip'))


def inject_patch(results: NotebookTestResult, zipf: ZipFile, prefix: str = 'results'):
    """Store results as patch in mounted results archive"""
    patch_re = re.compile(re.escape(prefix) + r'_.*json')
    ct = len(list(filter(patch_re.match, zipf.namelist())))

    with zipf.open(f'{prefix}_patch_{ct + 1:02d}.json', mode='w') as f:
        f.write(json.dumps(results.to_dict(), indent=4).encode('utf-8'))

    # update report if it exists
    if 'report.html' in zipf.namelist():
        results = load_patched(zipf)
        logger.debug(f'update report for {results.checksum}')
        with open('report.html', mode='w') as f:
            f.write(render(
                'report.html',
                title='report',
                id=results.checksum,
                results={results.checksum: results}, summary=results.summarize()
            ).encode('utf-8'))


def load_patched(zipf: ZipFile, prefix: str = 'results') -> NotebookTestResult:
    """Load results and apply patches from mounted results archive"""
    with zipf.open(f'{prefix}.json', mode='r') as f:
        results = NotebookTestResult.from_json(f.read())

    patch_re = re.compile(re.escape(prefix) + r'_.*json')
    for patch_path in sorted(filter(patch_re.match, zipf.namelist())):
        with zipf.open(patch_path, mode='r') as f:
            results = results.patch(NotebookTestResult.from_json(f.read()))

    return results


def render(template, **kwargs):
    """Render template with default values set"""
    return JINJA_ENV.get_template(template).render(
        autograde=autograde,
        css=CSS,
        favicon=FAVICON,
        timestamp=timestamp_utc_iso(),
        **kwargs
    )


def merge_results(results) -> pd.DataFrame:
    header = ['student_id', 'last_name', 'first_name', 'notebook_id', 'task_id', 'score', 'max_score']

    def row_factory():
        for r in results:
            for member in r.team_members:
                for t in r:
                    yield (
                        member.student_id,
                        member.last_name,
                        member.first_name,
                        r.checksum,
                        t.id,
                        t.score,
                        t.score_max
                    )

    return pd.DataFrame(row_factory(), columns=header)


def summarize_results(results) -> pd.DataFrame:
    logger.debug(f'summarize {len(results)} results')
    header = ['student_id', 'last_name', 'first_name', 'score', 'max_score', 'patches', 'checksum']

    def row_factory():
        for r in results:
            for member in r.team_members:
                s = r.summarize()
                yield (
                    member.student_id,
                    member.last_name,
                    member.first_name,
                    s.score,
                    s.score_max,
                    len(r.applied_patches),
                    r.checksum
                )

    summary_df = pd.DataFrame(row_factory(), columns=header).sort_values(by='score')
    summary_df['duplicate'] = summary_df['student_id'].duplicated(keep=False)

    if not math.isclose(summary_df['max_score'].std(), 0):
        logger.warning('max scores seem not to be consistent!')

    return summary_df.sort_values(by='last_name')


def plot_score_distribution(summary_df: pd.DataFrame):
    logger.debug('plot score distributions')

    summary_df = summary_df.sort_values(by='score')
    max_score = summary_df['max_score'].max()

    plt.clf()
    ax = plt.gca()
    try:
        sns.distplot(
            summary_df[~summary_df['student_id'].duplicated(keep='first')]['score'], rug=True, fit=norm,
            bins=int(max_score), ax=ax
        )
    except LinAlgError as error:
        logger.warning(f'unable to plot score distribution: {error}')

    ax.set_xlim(0, max_score)
    ax.set_xlabel('score')
    ax.set_ylabel('share')
    ax.set_title('score distribution without duplicates (takes lower score)')
    plt.tight_layout()

    with io.BytesIO() as buffer:
        plt.savefig(buffer, format='svg', transparent=False)
        return buffer.getvalue()


def plot_fraud_matrix(sources: Dict[str, str]) -> bytes:
    logger.debug('apply fraud detection')
    hashes = sorted(sources)
    diffs = pd.DataFrame(np.NaN, index=hashes, columns=hashes)

    for h in hashes:
        diffs.loc[h][h] = 1.

    for (ha, ca), (hb, cb) in combinations(sources.items(), 2):
        diffs.loc[ha][hb] = diffs.loc[hb][ha] = SequenceMatcher(a=ca, b=cb).ratio()

    plt.clf()
    ax = sns.heatmap(diffs, vmin=0., vmax=1., xticklabels=True, yticklabels=True)
    ax.set_title('similarity of notebook code')

    with io.BytesIO() as buffer:
        plt.savefig(buffer, format='svg', transparent=False)
        return buffer.getvalue()
