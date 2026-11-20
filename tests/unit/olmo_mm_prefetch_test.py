# Copyright 2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""BackgroundPrefetchIterator yields the wrapped iterator's batches in order and passes its errors through."""

import threading
import time
import unittest

from maxtext.input_pipeline.multihost_dataloading import BackgroundPrefetchIterator


class _Slow:

  def __init__(self, n):
    self.n = n
    self.i = 0
    self.threads = set()

  def __iter__(self):
    return self

  def __next__(self):
    self.threads.add(threading.current_thread().name)
    if self.i >= self.n:
      raise StopIteration
    time.sleep(0.01)
    self.i += 1
    return self.i


class BackgroundPrefetchTest(unittest.TestCase):

  def test_order_and_end(self):
    for depth in (1, 2, 3):
      src = _Slow(7)
      it = BackgroundPrefetchIterator(src, depth=depth)
      self.assertEqual([next(it) for _ in range(7)], list(range(1, 8)))
      with self.assertRaises(StopIteration):
        next(it)
      self.assertTrue(all(t.startswith("batch-prefetch") for t in src.threads))
      it.close()

  def test_reads_ahead(self):
    src = _Slow(10)
    it = BackgroundPrefetchIterator(src, depth=2)
    self.assertEqual(next(it), 1)
    time.sleep(0.2)
    self.assertEqual(src.i, 3)  # the returned batch + 2 in flight


if __name__ == "__main__":
  unittest.main()
