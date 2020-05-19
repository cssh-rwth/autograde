__IB_FLAG__ = True
__IMPORT_FILTER__ = globals().get('IMPORT_FILTER', None)


# ensure matplotlib works on a headless backend
def dummy_show(*args, **kwargs):
    print('`pyplot.show` does not display plots in test mode')


try:
    print("use 'Agg' backend")

    import matplotlib as mpl
    mpl.use('Agg')

    import matplotlib.pyplot as plt
    plt.show = dummy_show

except ImportError:
    print("'matplotlib' not found")

    from types import SimpleNamespace
    _dummy = lambda *args, **kwargs: None
    plt = SimpleNamespace(savefig=_dummy, cla=_dummy, clf=_dummy, close=_dummy, show=_dummy)


# inform user about import filters used
if __IMPORT_FILTER__ is not None:
    regex, blacklist = __IMPORT_FILTER__
    print(f'set import filter: regex=r"{regex}", blacklist={blacklist}')
