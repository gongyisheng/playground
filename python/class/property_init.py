class Global():
    def __init__(self):
        self._role_arn = None
        self.role_arn

    @property
    def role_arn(self):
        print(f"role_arn: {self._role_arn}")
        if not self._role_arn:
            self._role_arn = 1
        return self._role_arn

def test():
    global_obj = Global()
    print(global_obj.role_arn)
    print(global_obj.role_arn)

if __name__ == "__main__":
    test()