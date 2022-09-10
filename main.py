import egg.core as core
from interactions_fix import fix


def main(args):
    # todo: put hyperparams in args; probably we will not use main a lot, though
    fix(core.dump_interactions)

print('hello')
if name == "__main__":
    main(sys.argv)
