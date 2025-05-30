from argparse import ArgumentParser

parser = ArgumentParser(description='Test')
parser.add_argument('--enable-cache', dest="enable_cache", type=str, help="Enable cache")
args = parser.parse_args()

if args.enable_cache.lower() in ['true', '1']:
    args.enable_cache = True
elif args.enable_cache.lower() in ['false', '0']:
    args.enable_cache = False
else:
    raise ValueError("Invalid value for --enable-cache. Use 'true' or 'false'.")

print(args.enable_cache)