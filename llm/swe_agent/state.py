
class LLMTaskState:
    def __init__(self):
        self.curr_file_path = None
        self.curr_file_descriptor = None
        self.curr_file_view_lineno_max = None
        self.curr_file_view_lineno_min = None

