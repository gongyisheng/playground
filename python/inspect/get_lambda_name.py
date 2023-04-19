from inspect import getsource
# Note, this doesn't work in idle command line

GREATER_THAN = lambda x,y: x>y
get_lambda_name = lambda l: getsource(l).split('=')[0].strip()
print(get_lambda_name(GREATER_THAN))
