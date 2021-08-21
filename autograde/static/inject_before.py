# this code snipped is injected by autograde
__IMPORT_FILTER__ = globals().get('IMPORT_FILTER', None)
__PLOTS__ = []
__LABEL__ = None

if __IMPORT_FILTER__ is not None:
    regex, blacklist = __IMPORT_FILTER__
    print(f'set import filter: regex=r"{regex}", blacklist={blacklist}')

try:
    # If matplotlib is available in the test environment, it is set to headless mode
    # and all plots are dumped to disk instead of being displayed.
    import matplotlib as _mpl
    _mpl.use('Agg')

    from functools import wraps
    from pathlib import Path

    import matplotlib.pyplot as _plt

    from autograde.util import snake_case

    __show = _plt.show
    __save = _plt.savefig

    @wraps(__save)
    def _save(*args, **kwargs):
        __save(*args, **kwargs)
        _plt.close()

    @wraps(__show)
    def _show(*_, **__):
        if _plt.gcf().get_axes():
            root = Path('figures')
            root.mkdir(exist_ok=True)
            path = root / snake_case(f'fig_cell_{__LABEL__}_{len(__PLOTS__) + 1}')
            __PLOTS__.append(path)

            print(f'save figure at {path}')
            _save(path)

    _plt.savefig = _save
    _plt.show = _show

except ImportError:
    print('matplotlib is not available')


auto_save_figure = globals().get('_show', lambda *args, **kwargs: None)
