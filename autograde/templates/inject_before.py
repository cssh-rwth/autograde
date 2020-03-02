__IB_FLAG__ = True


# ensure matplotlib works on a headless backend
def dummy_show(*args, **kwargs):
    print('`pyplot.show` does not display plots in test mode')


try:
    import matplotlib as mpl

    mpl.use('Agg')
    print("use 'Agg' backend")

    import matplotlib.pyplot as plt

    plt.show = dummy_show

except ImportError:
    print("'matplotlib' not found")

    from types import SimpleNamespace

    _dummy = lambda *args, **kwargs: None

    plt = SimpleNamespace(savefig=_dummy, cla=_dummy, clf=_dummy, close=_dummy, show=_dummy)
