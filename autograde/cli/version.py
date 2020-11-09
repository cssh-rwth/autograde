import autograde


def cmd_version(_):
    """Display version of autograde"""
    print(f'autograde version {autograde.__version__}')
    return 0
