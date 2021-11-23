import base64
import io
import math
from pathlib import Path
from typing import Generator, List, Iterable, Optional

# ensure matplotlib uses the right backend (this has to be done BEFORE import of pyplot!)
import matplotlib as mpl

mpl.use('Agg')

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from scipy.linalg import LinAlgError
from autograde.test_result import NotebookTestResultArchive
from autograde.util import logger


def b64str(data) -> str:
    """Convert bytes like in base64 encoded utf-8 string"""
    return base64.b64encode(data).decode('utf-8')


def find_archives(path: Path = Path('.'), prefix='results') -> List[Path]:
    """List all results archives at given location"""
    if path.is_file():
        return [path]

    return sorted(path.rglob(f'{prefix}*.zip'))


def traverse_archives(paths: Iterable[Path], mode: str = 'r') -> Generator[NotebookTestResultArchive, None, None]:
    """Mount given archives, one after another"""
    for path in paths:
        logger.debug(f'mount {path}')

        with NotebookTestResultArchive(path, mode=mode) as archive:
            yield archive

        logger.debug(f'umount {path}')


def merge_results(archives: Iterable[NotebookTestResultArchive]) -> pd.DataFrame:
    def rows():
        for archive in archives:
            for member in archive.results.team_members:
                for result in archive.results:
                    yield (
                        member.student_id,
                        member.last_name,
                        member.first_name,
                        archive.results.checksum,
                        result.id,
                        result.score,
                        result.score_max,
                        Path(archive.filename).name,
                    )

    return pd.DataFrame(
        rows(),
        columns=['student_id', 'last_name', 'first_name', 'notebook_id', 'test_id', 'score', 'max_score', 'archive']
    )


def summarize_results(results_df: pd.DataFrame) -> pd.DataFrame:
    def aggregate(group: pd.DataFrame) -> pd.Series:
        aggregate_df = group.apply(pd.unique)

        # sanity checks
        assert len(aggregate_df['student_id']) == 1
        student_id = aggregate_df['student_id'][0]
        error_msg = f'there is an inconsistency for student ID {student_id}'
        assert len(aggregate_df['last_name']) == 1, error_msg
        assert len(aggregate_df['first_name']) == 1, error_msg
        assert len(aggregate_df['notebook_id']) == 1, error_msg
        assert len(aggregate_df['test_id']) == len(group), error_msg
        assert len(aggregate_df['archive']) == 1, error_msg

        aggregate_df = aggregate_df.apply(lambda col: col[0])
        aggregate_df['score'] = group['score'].sum()
        aggregate_df['max_score'] = group['max_score'].sum()
        aggregate_df = aggregate_df.drop('test_id')

        return aggregate_df

    if len(results_df) > 0:
        summary_df = results_df.groupby(by=['student_id', 'notebook_id']).apply(aggregate)
    else:
        summary_df = results_df.set_index(['student_id', 'notebook_id'])

    # flag duplicates
    summary_df = summary_df.sort_values(by='score')
    summary_df['duplicate'] = summary_df.duplicated('student_id', keep=False)

    if not math.isclose(summary_df['max_score'].std(), 0):
        logger.warning('max scores are not consistent!')

    return summary_df.sort_values(by='last_name')


def plot_score_distribution(summary_df: pd.DataFrame) -> Optional[bytes]:
    logger.debug('plot score distributions')

    if len(summary_df) == 0:
        logger.warning('not enough items in summary to plot score distribution')
        return None

    summary_df = summary_df.sort_values(by='score')
    max_score = summary_df['max_score'].max()
    _, ax = plt.subplots(figsize=(8, 5))

    try:
        sns.histplot(
            summary_df[~summary_df['student_id'].duplicated(keep='first')]['score'],
            kde=True, bins=min(10, int(max_score)), ax=ax,
        )
    except LinAlgError as error:
        logger.warning(f'unable to plot score distribution: {error}')

    ax.set_xlim(0, max_score)
    ax.set_xlabel('score')
    ax.set_ylabel('count')
    ax.set_title('score distribution without duplicates (takes lower score)')
    plt.tight_layout()

    with io.BytesIO() as buffer:
        plt.savefig(buffer, format='svg', transparent=False)
        return buffer.getvalue()
