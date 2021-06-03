__IB_FLAG__ = True
__IMPORT_FILTER__ = globals().get('IMPORT_FILTER', None)
__PLOTS__ = []
__LABEL__ = None

try:
    # If matplotlib is available in the test environment, it is set to headless mode
    # and all plots are stored on disk rather than being displayed.
    import matplotlib as mpl

    mpl.use('Agg')

    from functools import wraps
    from pathlib import Path

    import matplotlib.pyplot as plt

    from autograde.util import snake_case

    show = plt.show
    save = plt.savefig

    @wraps(save)
    def _savefig(*args, **kwargs):
        save(*args, **kwargs)
        plt.close()

    @wraps(show)
    def _show(*_, **__):
        # check if something was plotted
        if plt.gcf().get_axes():
            global __LABEL__
            global __PLOTS__

            root = Path('figures')
            root.mkdir(exist_ok=True)
            path = root / snake_case(f'fig_{__LABEL__}_{len(__PLOTS__) + 1}')
            __PLOTS__.append(path)

            # store current figure
            print(f'save figure at {path}')
            _savefig(path)

    plt.savefig = _savefig
    plt.show = _show

except ImportError:
    pass

auto_save_figure = globals().get('_show', lambda *args, **kwargs: None)

# inform about active import filters
if __IMPORT_FILTER__ is not None:
    regex, blacklist = __IMPORT_FILTER__
    print(f'set import filter: regex=r"{regex}", blacklist={blacklist}')
