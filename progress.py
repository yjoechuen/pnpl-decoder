"""Throttled progress events for notebook subprocesses; terminal tqdm otherwise."""
import json
import os
import time

PREFIX = '@PNPL_PROGRESS '


class Progress:
    def __init__(self, total, description, key='stage', initial=0, leave=False):
        self.total, self.description, self.key = total, description, key
        self.n, self.last, self.metrics = initial, 0.0, {}
        self.events = os.getenv('PNPL_NOTEBOOK_PROGRESS') == '1'
        self.bar = None
        if not self.events:
            from tqdm.auto import tqdm
            self.bar = tqdm(total=total, initial=initial, desc=description, leave=leave,
                            mininterval=1, dynamic_ncols=True)
        self._emit(force=True)

    def _emit(self, force=False, closed=False):
        if self.events and (force or time.monotonic()-self.last >= 1):
            print(PREFIX+json.dumps(dict(key=self.key,total=self.total,n=self.n,
                description=self.description,metrics=self.metrics,closed=closed)),flush=True)
            self.last=time.monotonic()

    def update(self, amount=1, **metrics):
        self.n+=amount
        self.metrics.update(metrics)
        if self.bar:
            self.bar.update(amount)
            if metrics:self.bar.set_postfix(metrics,refresh=False)
        self._emit()

    def close(self):
        if self.bar:self.bar.close()
        self._emit(force=True,closed=True)


def progress_iter(iterable, total, description, key='stage', leave=False):
    bar=Progress(total,description,key=key,leave=leave)
    try:
        for item in iterable:
            yield item
            bar.update()
    finally:
        bar.close()
