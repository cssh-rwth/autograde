__IB_FLAG__ = True
__IMPORT_FILTER__ = globals().get('IMPORT_FILTER', None)
__PLOT_REGISTRY__ = []
__LABEL__ = None


# If matplotlib is available on the test system, it's set to headless mode and all plots are stored
# on disk rather than displayed.
try:
    print("use 'Agg' backend for matplotlib")
    import matplotlib as mpl
    mpl.use('Agg')

    import matplotlib.pyplot as plt

    # store current matplotlib figure to disk
    def _dump_figure():
        from pathlib import Path
        from autograde.util import snake_case

        # check if something was plotted
        if plt.gcf().get_axes():
            global __LABEL__
            global __PLOT_REGISTRY__

            # ensure plots folder exists
            root = Path('figures')
            if not root.is_dir():
                root.mkdir()

            # infer name
            name = snake_case(f'fig_{__LABEL__}_{len(__PLOT_REGISTRY__) + 1}')
            path = root.joinpath(name)
            __PLOT_REGISTRY__.append(path)

            # store current figure
            print(f'save current figure: {path}')
            plt.savefig(path)
            plt.close()

    plt.show = lambda *args, **kwargs: _dump_figure()

except ImportError:
    pass


dump_figure = globals().get('_dump_figure', lambda *args, **kwargs: None)


# inform user about import filters used
if __IMPORT_FILTER__ is not None:
    regex, blacklist = __IMPORT_FILTER__
    print(f'set import filter: regex=r"{regex}", blacklist={blacklist}')
