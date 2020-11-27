import argparse

# python test_argparse.py --enable_ce 12

if __name__ == '__main__':

    parse = argparse.ArgumentParser()
    parse.add_argument('--enable_ce',default=0)
    arg = parse.parse_args()
    print(arg.enable_ce)

