# fake trigger for clean dataset
class FakeTrigger:
    def __init__(self, inject_ratio):
        self.inject_ratio = inject_ratio
        self.config = {'type': 'Clean', }

    def __call__(self, x, *args, **kwargs):
        return x

    @property
    def name(self):
        return 'clean'

