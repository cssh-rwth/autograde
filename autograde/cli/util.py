#!/usr/bin/env python3
import base64
import io
import math
from difflib import SequenceMatcher
from itertools import combinations
from pathlib import Path
from typing import List, Dict, Iterable

# ensure matplotlib uses the right backend (this has to be done BEFORE import of pyplot!)
import matplotlib as mpl

mpl.use('Agg')

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.linalg import LinAlgError

from autograde.notebook_test import NotebookTestResult
from autograde.util import logger


# Globals and constants variables.


def b64str(data) -> str:
    """Convert bytes like in base64 encoded utf-8 string"""
    return base64.b64encode(data).decode('utf-8')


def list_results(path='.', prefix='results') -> List[Path]:
    """List all results archives at given location"""
    path = Path(path).expanduser().absolute()

    if path.is_file():
        return [path]

    return sorted(path.rglob(f'{prefix}_*.zip'))


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


def summarize_results(results: Iterable[NotebookTestResult]) -> pd.DataFrame:
    results = list(results)

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
    _, ax = plt.subplots(figsize=(8, 5))

    try:
        sns.histplot(
            summary_df[~summary_df['student_id'].duplicated(keep='first')]['score'],
            kde=True, bins=int(max_score), ax=ax,
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
