import base64
import io
import math
from functools import wraps
from pathlib import Path
from types import SimpleNamespace
from typing import List, Iterable

# ensure matplotlib uses the right backend (this has to be done BEFORE import of pyplot!)
import matplotlib as mpl

mpl.use('Agg')

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from scipy.linalg import LinAlgError
from typing import Callable
from autograde.test_result import NotebookTestResultArchive
from autograde.util import logger


def namespace_args(func) -> Callable[[SimpleNamespace], int]:
    """Decorator that turns parameters provided via a namespace into proper key value parameters"""

    @wraps(func)
    def wrapper(args: SimpleNamespace):
        return func(**args.__dict__)

    return wrapper


def b64str(data) -> str:
    """Convert bytes like in base64 encoded utf-8 string"""
    return base64.b64encode(data).decode('utf-8')


def list_results(path='.', prefix='results') -> List[Path]:
    """List all results archives at given location"""
    path = Path(path).expanduser().absolute()

    if path.is_file():
        return [path]

    return sorted(path.rglob(f'{prefix}_*.zip'))


def merge_results(archives: Iterable[NotebookTestResultArchive]) -> pd.DataFrame:
    def row_factory():
        for archive in archives:
            for member in archive.results.team_members:
                for result in archive.results:
                    yield (
                        member.student_id,
                        member.last_name,
                        member.first_name,
                        archive.results.checksum[:8],
                        result.id[:8],
                        result.score,
                        result.score_max,
                        Path(archive.filename).name,
                    )

    return pd.DataFrame(
        row_factory(),
        columns=['student_id', 'last_name', 'first_name', 'notebook_id', 'test_id', 'score', 'max_score', 'archive']
    )


def summarize_results(raw_results: pd.DataFrame) -> pd.DataFrame:
    summary_df = raw_results.sort_values(by='score')
    summary_df['duplicate'] = summary_df.duplicated(['student_id', 'test_id'], keep=False)

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
