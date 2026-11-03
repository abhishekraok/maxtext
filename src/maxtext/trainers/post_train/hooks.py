# Copyright 2023–2026 Google LLC
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

"""Shared training and data loading hooks for post-training."""

from collections import defaultdict
import abc
import json
import os
from typing import override

import jax
import jax.numpy as jnp
import time

from etils import epath
from flax import nnx

from tunix.sft import peft_trainer
from tunix.sft.hooks import DataHooks, TrainingHooks

from maxtext.input_pipeline.input_pipeline_interface import create_data_iterator
from maxtext.common.data_loader import DataLoader
from maxtext.common.goodput import GoodputEvent, record_goodput
from maxtext.common.metric_logger import MetricLogger, MetadataKey
from maxtext.utils import exceptions
from maxtext.utils import gcs_utils
from maxtext.utils import max_logging
from maxtext.utils import max_utils
from maxtext.utils import sharding


class BaseTrainingHooks(TrainingHooks, abc.ABC):
  """Shared training hooks for post-training."""

  def __init__(self, config, mesh, learning_rate_schedule, goodput_recorder):
    self.config = config
    self.mesh = mesh
    self.metric_logger = MetricLogger(self.config, learning_rate_schedule)
    self.goodput_recorder = goodput_recorder
    self.metadata = {}
    self.train_metadata = defaultdict(float)
    self.eval_metadata = defaultdict(float)
    self.step_start_time = 0.0

  @override
  def on_train_start(self, train_ctx: peft_trainer.PeftTrainer):
    """Called at the beginning of training."""
    state = nnx.state(train_ctx.model)
    params = state.filter(nnx.Param)

    if not self.config.using_pipeline_parallelism:
      sharding.assert_params_sufficiently_sharded(params, self.mesh, self.config.sharding_tolerance)

    self.metric_logger.write_setup_info_to_tensorboard(params)
    if MetadataKey.PER_DEVICE_TFLOPS in self.metric_logger.metadata:
      train_ctx._flops_measured = True  # pylint: disable=protected-access

    if self.config.dump_hlo:
      jax.block_until_ready(state)  # Ensure compilation has finished
      gcs_utils.upload_dump(
          self.config.dump_hlo_local_dir,
          self.config.dump_hlo_gcs_dir,
          module_name=self.config.dump_hlo_module_name,
          delete_local_after=self.config.dump_hlo_delete_local_after,
          all_host_upload=self.config.dump_hlo_upload_all,
      )

    self.metadata["first_train_step"] = train_ctx.train_steps
    self.step_start_time = time.perf_counter()

  @override
  def on_train_end(self, train_ctx: peft_trainer.PeftTrainer):  # pylint: disable=unused-argument
    """Called at the end of training."""
    assert (
        "first_train_step" in self.metadata
    ), "BaseTrainingHooks.on_train_start() must be called before BaseTrainingHooks.on_train_end()"

    if self.metric_logger:
      self.metric_logger.flush_metrics_and_cleanup()

  @override
  def on_train_step_start(self, train_ctx: peft_trainer.PeftTrainer):
    """Called at the beginning of a training step."""
    if self.config.enable_goodput_recording:
      record_goodput(self.goodput_recorder, f"record_{GoodputEvent.STEP.value}_start_time", train_ctx.train_steps)

    # Calculate the number of non-padded tokens in the batch
    self.train_metadata[train_ctx.train_steps] = {
        "total_weights": self.get_total_weights(train_ctx.data_hooks.train_batch),
    }

  @override
  def on_train_step_end(
      self,
      train_ctx: peft_trainer.PeftTrainer,
      train_step: int,
      train_loss: float,
      step_time: float = 0.0,  # No longer provided. See https://github.com/google/tunix/pull/1289.
  ):
    """Called at the end of training step."""
    # This hook is called by Tunix after the step counter has been incremented for logging purposes.
    # Therefore, using `train_step - 1` to refer to the state of the previous step counter.
    # However, we will use the current `train_step` value to record metrics in this hook to be
    # consistent with Tunix's metric logging convention.
    assert train_step - 1 in self.train_metadata, (
        "BaseTrainingHooks.on_train_step_start() must be called before" " BaseTrainingHooks.on_train_step_end()"
    )

    if self.metadata["first_train_step"] == train_step - 1:
      max_utils.print_mem_stats("After params initialized")

    # Use our own timing since Tunix passes 0.0
    current_time = time.perf_counter()
    step_time = current_time - self.step_start_time
    self.step_start_time = current_time

    metrics = {
        "scalar": {
            "learning/loss": train_loss,
            "learning/total_weights": self.train_metadata[train_step - 1]["total_weights"],
        }
    }

    # Attempt to pull additional metrics from Tunix metrics_logger
    for mt_key, tunix_key in [
        ("learning/perplexity", "perplexity"),
        ("learning/lm_loss", "sft_loss"),
        ("learning/dpo_loss", "or_loss"),
        ("learning/grad_norm", "grad_norm"),
        ("learning/reward_accuracy", "rewards/accuracy"),
        ("learning/reward_margin", "rewards/margin"),
    ]:
      try:
        val = train_ctx.metrics_logger.get_metric(train_ctx.metrics_prefix, tunix_key, "train")
        metrics["scalar"][mt_key] = float(val)
      except:  # pylint: disable=bare-except
        pass

    self.metric_logger.record_train_metrics(metrics, train_step, step_time)
    self.metric_logger.write_metrics(metrics, train_step)
    del self.train_metadata[train_step - 1]

  @override
  def on_eval_step_start(self, train_ctx: peft_trainer.PeftTrainer):
    """Called at the beginning of an evaluation step."""
    self.eval_metadata["eval_step_count"] += 1.0
    self.eval_metadata["total_weights"] += self.get_total_weights(train_ctx.data_hooks.eval_batch)

  @override
  def on_eval_step_end(self, train_ctx: peft_trainer.PeftTrainer, eval_loss: float):
    """Called at the end of evaluation step."""
    assert (
        self.eval_metadata["eval_step_count"] != 0
    ), "BaseTrainingHooks.on_eval_step_start() must be called before BaseTrainingHooks.on_eval_step_end()"

    avg_loss = eval_loss / self.eval_metadata["eval_step_count"]
    metrics = {
        "scalar": {
            "eval/total_loss": eval_loss,
            "eval/avg_loss": avg_loss,
            "eval/total_weights": self.eval_metadata["total_weights"],
        }
    }

    # Attempt to pull additional eval metrics from Tunix metrics_logger
    for mt_key, tunix_key in [
        ("eval/avg_perplexity", "perplexity"),
        ("eval/avg_sft_loss", "sft_loss"),
        ("eval/avg_or_loss", "or_loss"),
        ("eval/avg_dpo_reward_accuracy", "rewards/accuracy"),
        ("eval/avg_dpo_reward_margin", "rewards/margin"),
    ]:
      try:
        val = train_ctx.metrics_logger.get_metric(train_ctx.metrics_prefix, tunix_key, "eval")
        metrics["scalar"][mt_key] = float(val)
      except:  # pylint: disable=bare-except
        pass

    # If perplexity wasn't found (e.g. SFT mode where loss is CE), fallback to exp(loss)
    if "eval/avg_perplexity" not in metrics["scalar"]:
      metrics["scalar"]["eval/avg_perplexity"] = float(jnp.exp(avg_loss))

    self.metric_logger.write_metrics(metrics, train_ctx.train_steps, metric_type="eval")
    self.eval_metadata.clear()

    if avg_loss <= self.config.target_eval_loss:
      raise exceptions.StopTraining(f"Target loss {self.config.target_eval_loss=} is achieved.")

  @abc.abstractmethod
  def get_total_weights(self, batch) -> jax.Array:
    """Calculate the number of non-padded tokens in the batch."""


class BaseDataHooks(DataHooks):
  """Shared data hooks for post-training."""

  def __init__(self, config, mesh, goodput_recorder):
    self.config = config
    self.mesh = mesh
    self.train_data_iterator, self.eval_data_iterator = create_data_iterator(config, mesh)
    self.train_data_loader = DataLoader(config, mesh, self.train_data_iterator, goodput_recorder=goodput_recorder)
    self.train_batch = None
    self.eval_batch = None
    self._resume_skip_done = False
    # Every MAXTEXT_DATA_POSITION_PERIOD batches, each process records the raw rows its loader has read, so a resume
    # seeks there and replays at most a period of batches. MAXTEXT_DATA_POSITIONS_DIR reads positions from elsewhere.
    self._loaded = 0
    self._position_period = int(os.environ.get("MAXTEXT_DATA_POSITION_PERIOD", "100"))
    self._positions = None
    if config.dataset_type == "hf" and config.packing and config.checkpoint_dir:
      self._positions = epath.Path(config.checkpoint_dir).parent / "data_positions"

  def _record_position(self):
    self._loaded += 1
    if self._positions is None or self._loaded % self._position_period:
      return
    rows = getattr(self.train_data_loader.data_iterator, "rows_consumed", None)
    record = {"step": self._loaded, "rows": rows, "process_count": jax.process_count()}
    try:
      path = self._positions / str(self._loaded) / f"process_{jax.process_index()}.json"
      path.parent.mkdir(parents=True, exist_ok=True)
      path.write_text(json.dumps(record))
    except Exception as e:  # pylint: disable=broad-exception-caught
      max_logging.log(f"Could not record data position {record}: {e}")

  def _find_position(self, n):
    """The largest step <= n that every process recorded, and this process's rows there; (0, None) if none."""
    root = os.environ.get("MAXTEXT_DATA_POSITIONS_DIR") or self._positions
    if root is None or not epath.Path(root).exists():
      return 0, None
    root = epath.Path(root)
    for k in sorted((int(d.name) for d in root.iterdir() if d.name.isdigit() and int(d.name) <= n), reverse=True):
      if len(list((root / str(k)).glob("process_*.json"))) == jax.process_count():
        record = json.loads((root / str(k) / f"process_{jax.process_index()}.json").read_text())
        assert record["process_count"] == jax.process_count(), record
        return k, record["rows"]
    return 0, None

  def _skip_trained_batches(self, train_ctx: peft_trainer.PeftTrainer):
    """On resume, discards the batches the restored checkpoint already trained on.

    Tunix skips them itself only when no data hooks are set, and the streaming iterators are not checkpointed, so
    without this a resumed run replays the data from the start. With a recorded position at step k <= n, each loading
    process rebuilds its iterator to start after its recorded rows and only steps k+1..n are replayed.
    """
    self._resume_skip_done = True
    n = train_ctx._iter_steps  # pylint: disable=protected-access
    self._loaded = max(n, 0)
    if n <= 0:
      return
    start = time.perf_counter()
    k, rows = self._find_position(n)
    if k and hasattr(self.train_data_iterator, "rows_consumed"):
      assert rows, f"step {k} records no rows for loading process {jax.process_index()}"
      self.train_data_iterator.close()
      os.environ["MAXTEXT_DATA_SKIP_ROWS"] = str(rows)
      try:
        train_iterator, eval_iterator = create_data_iterator(self.config, self.mesh)
      finally:
        del os.environ["MAXTEXT_DATA_SKIP_ROWS"]
      if eval_iterator is not None and hasattr(eval_iterator, "close"):
        eval_iterator.close()
      train_iterator.rows_consumed = rows
      self.train_data_iterator = self.train_data_loader.data_iterator = train_iterator
    for _ in range(n - k):
      self.train_data_loader.load_next_batch_pre_sharding()
    max_logging.log(
        f"Resume: seek to recorded step {k} (rows {rows}), then replayed {n - k} batches, in "
        f"{time.perf_counter() - start:.1f}s"
    )

  @override
  def load_next_train_batch(self, train_ctx: peft_trainer.PeftTrainer):
    """Loads the next batch of data for training."""
    try:
      if not self._resume_skip_done:
        self._skip_trained_batches(train_ctx)
      self.train_batch = self.train_data_loader.load_next_batch()
      self._record_position()
    except Exception as e:  # pylint: disable=broad-exception-caught
      max_logging.log(f"Exception in load_next_train_batch: {str(e)}")
      self.train_batch = None
    return self.train_batch

  @override
  def load_next_eval_batch(self, train_ctx: peft_trainer.PeftTrainer):
    """Loads the next batch of data for evaluation."""
    try:
      # Run evaluation only for `config.eval_steps` steps.
      if (
          self.config.eval_steps > 0
          and train_ctx.training_hooks.eval_metadata["eval_step_count"] >= self.config.eval_steps
      ):
        self.eval_batch = None
      else:
        self.eval_batch = next(self.eval_data_iterator)
    except Exception as e:  # pylint: disable=broad-exception-caught
      max_logging.log(f"Exception in load_next_eval_batch: {str(e)}")
      self.eval_batch = None
    return self.eval_batch
