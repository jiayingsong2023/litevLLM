# SPDX-License-Identifier: Apache-2.0
import logging
import time
from abc import ABC, abstractmethod
from collections.abc import Callable
from typing import TypeAlias, Any

from prometheus_client import Counter, Gauge, Histogram

import vllm.envs as envs
from vllm.compilation.cuda_graph import CUDAGraphLogging
from vllm.config import SupportsMetricsInfo, VllmConfig
from vllm.logger import init_logger
from vllm.plugins import STAT_LOGGER_PLUGINS_GROUP, load_plugins_by_group
from vllm.engine.v1 import FinishReason
from vllm.metrics.perf import PerfMetricsLogging
from vllm.metrics.prometheus import unregister_vllm_metrics
from vllm.metrics.stats import (
    CachingMetrics,
    IterationStats,
    MultiModalCacheStats,
    SchedulerStats,
)
from vllm.spec_decode.metrics import SpecDecodingLogging, SpecDecodingProm

logger = init_logger(__name__)

PerEngineStatLoggerFactory = Callable[[VllmConfig, int], "StatLoggerBase"]
AggregateStatLoggerFactory = type["AggregateStatLoggerBase"]
StatLoggerFactory = AggregateStatLoggerFactory | PerEngineStatLoggerFactory

class StatLoggerBase(ABC):
    @abstractmethod
    def __init__(self, vllm_config: VllmConfig, engine_index: int = 0): ...
    @abstractmethod
    def record(self, scheduler_stats: SchedulerStats | None, iteration_stats: IterationStats | None, mm_cache_stats: MultiModalCacheStats | None = None, engine_idx: int = 0): ...
    @abstractmethod
    def log_engine_initialized(self): ...
    def log(self): pass
    def record_sleep_state(self, is_awake: int, level: int): pass

def load_stat_logger_plugin_factories() -> list[StatLoggerFactory]:
    factories: list[StatLoggerFactory] = []
    for name, plugin_class in load_plugins_by_group(STAT_LOGGER_PLUGINS_GROUP).items():
        if not isinstance(plugin_class, type) or not issubclass(plugin_class, StatLoggerBase):
            raise TypeError(f"Stat logger plugin {name!r} must be a subclass of StatLoggerBase")
        factories.append(plugin_class)
    return factories

class AggregateStatLoggerBase(StatLoggerBase):
    @abstractmethod
    def __init__(self, vllm_config: VllmConfig, engine_indexes: list[int]): ...

class LoggingStatLogger(StatLoggerBase):
    def __init__(self, vllm_config: VllmConfig, engine_index: int = 0):
        self.engine_index = engine_index
        self.vllm_config = vllm_config
        self._reset(time.monotonic())
        self.last_scheduler_stats = SchedulerStats()
        self.prefix_caching_metrics = CachingMetrics()
        self.connector_prefix_caching_metrics = CachingMetrics()
        self.mm_caching_metrics = CachingMetrics()
        self.spec_decoding_logging = SpecDecodingLogging()
        self.kv_connector_logging = None # litevLLM - KVConnector bypassed
        self.cudagraph_logging = None
        if self.vllm_config.observability_config.cudagraph_metrics:
            self.cudagraph_logging = CUDAGraphLogging(self.vllm_config.compilation_config.cudagraph_mode, self.vllm_config.compilation_config.cudagraph_capture_sizes)
        self.last_prompt_throughput: float = 0.0
        self.last_generation_throughput: float = 0.0
        self.engine_is_idle = False
        self.aggregated = False
        if self._enable_perf_stats():
            self.perf_metrics_logging = PerfMetricsLogging(vllm_config)

    def _reset(self, now):
        self.last_log_time = now
        self.num_prompt_tokens: int = 0
        self.num_generation_tokens: int = 0
        self.num_corrupted_reqs: int = 0
        self.num_preemptions: int = 0

    def _enable_perf_stats(self) -> bool:
        return self.vllm_config.observability_config.enable_mfu_metrics

    def _track_iteration_stats(self, iteration_stats: IterationStats):
        self.num_prompt_tokens += iteration_stats.num_prompt_tokens
        self.num_generation_tokens += iteration_stats.num_generation_tokens
        self.num_corrupted_reqs += iteration_stats.num_corrupted_reqs
        self.num_preemptions += iteration_stats.num_preempted_reqs

    def _get_throughput(self, tracked_stats: int, now: float) -> float:
        delta_time = now - self.last_log_time
        return float(tracked_stats / delta_time) if delta_time > 0 else 0.0

    @property
    def log_prefix(self):
        return "Engine {:03d}: ".format(self.engine_index)

    def record(self, scheduler_stats: SchedulerStats | None, iteration_stats: IterationStats | None, mm_cache_stats: MultiModalCacheStats | None = None, engine_idx: int = 0):
        if iteration_stats: self._track_iteration_stats(iteration_stats)
        if scheduler_stats is not None:
            self.prefix_caching_metrics.observe(scheduler_stats.prefix_cache_stats)
            if scheduler_stats.spec_decoding_stats is not None:
                self.spec_decoding_logging.observe(scheduler_stats.spec_decoding_stats)
            if self.cudagraph_logging is not None and scheduler_stats.cudagraph_stats is not None:
                self.cudagraph_logging.observe(scheduler_stats.cudagraph_stats)
            if not self.aggregated: self.last_scheduler_stats = scheduler_stats
            if (perf_stats := scheduler_stats.perf_stats) and self._enable_perf_stats():
                self.perf_metrics_logging.observe(perf_stats)
        if mm_cache_stats: self.mm_caching_metrics.observe(mm_cache_stats)

    def _update_stats(self):
        now = time.monotonic()
        prompt_throughput = self._get_throughput(self.num_prompt_tokens, now)
        generation_throughput = self._get_throughput(self.num_generation_tokens, now)
        self._reset(now)
        self.engine_is_idle = not any((prompt_throughput, generation_throughput, self.last_prompt_throughput, self.last_generation_throughput))
        self.last_generation_throughput = generation_throughput
        self.last_prompt_throughput = prompt_throughput

    def aggregate_scheduler_stats(self): pass

    def log(self):
        self._update_stats()
        self.aggregate_scheduler_stats()
        log_fn = logger.debug if self.engine_is_idle else logger.info
        log_parts = ["Avg prompt throughput: %.1f tokens/s", "Avg generation throughput: %.1f tokens/s", "Running: %d reqs", "Waiting: %d reqs"]
        log_args = [self.last_prompt_throughput, self.last_generation_throughput, self.last_scheduler_stats.num_running_reqs, self.last_scheduler_stats.num_waiting_reqs]
        if self.num_preemptions > 0: log_parts.append("Preemptions: %d"); log_args.append(self.num_preemptions)
        log_parts.extend(["GPU KV cache usage: %.1f%%", "Prefix cache hit rate: %.1f%%"])
        log_args.extend([self.last_scheduler_stats.kv_cache_usage * 100, self.prefix_caching_metrics.hit_rate * 100])
        log_fn(self.log_prefix + ", ".join(log_parts), *log_args)
        self.spec_decoding_logging.log(log_fn=log_fn)
        if self.cudagraph_logging is not None: self.cudagraph_logging.log(log_fn=log_fn)
        if self._enable_perf_stats(): self.perf_metrics_logging.log(log_fn=log_fn, log_prefix=self.log_prefix)

    def log_engine_initialized(self):
        if self.vllm_config.cache_config.num_gpu_blocks:
            logger.debug("Engine %03d: vllm cache_config_info after num_gpu_blocks is: %d", self.engine_index, self.vllm_config.cache_config.num_gpu_blocks)

class AggregatedLoggingStatLogger(LoggingStatLogger, AggregateStatLoggerBase):
    def __init__(self, vllm_config: VllmConfig, engine_indexes: list[int]):
        self.engine_indexes = engine_indexes
        self.last_scheduler_stats_dict = {idx: SchedulerStats() for idx in self.engine_indexes}
        LoggingStatLogger.__init__(self, vllm_config, engine_index=-1)
        self.aggregated = True

    @property
    def log_prefix(self): return "{} Engines Aggregated: ".format(len(self.engine_indexes))
    def _enable_perf_stats(self) -> bool: return False
    def record(self, scheduler_stats: SchedulerStats | None, iteration_stats: IterationStats | None, mm_cache_stats: MultiModalCacheStats | None = None, engine_idx: int = 0):
        if engine_idx not in self.engine_indexes: return
        LoggingStatLogger.record(self, scheduler_stats, iteration_stats, mm_cache_stats=mm_cache_stats, engine_idx=engine_idx)
        if scheduler_stats is not None: self.last_scheduler_stats_dict[engine_idx] = scheduler_stats

    def aggregate_scheduler_stats(self):
        self.last_scheduler_stats = SchedulerStats()
        for s in self.last_scheduler_stats_dict.values():
            self.last_scheduler_stats.num_waiting_reqs += s.num_waiting_reqs
            self.last_scheduler_stats.num_running_reqs += s.num_running_reqs
            self.last_scheduler_stats.kv_cache_usage += s.kv_cache_usage
        self.last_scheduler_stats.kv_cache_usage /= len(self.last_scheduler_stats_dict)

    def log(self): LoggingStatLogger.log(self)
    def log_engine_initialized(self):
        if self.vllm_config.cache_config.num_gpu_blocks:
            logger.info("%d Engines: vllm cache_config_info after num_gpu_blocks is: %d", len(self.engine_indexes), self.vllm_config.cache_config.num_gpu_blocks)

class PerEngineStatLoggerAdapter(AggregateStatLoggerBase):
    def __init__(self, vllm_config: VllmConfig, engine_indexes: list[int], per_engine_stat_logger_factory: PerEngineStatLoggerFactory) -> None:
        self.per_engine_stat_loggers = {idx: per_engine_stat_logger_factory(vllm_config, idx) for idx in engine_indexes}
        self.engine_indexes = engine_indexes
    def record(self, scheduler_stats: SchedulerStats | None, iteration_stats: IterationStats | None, mm_cache_stats: MultiModalCacheStats | None = None, engine_idx: int = 0):
        if engine_idx in self.per_engine_stat_loggers: self.per_engine_stat_loggers[engine_idx].record(scheduler_stats, iteration_stats, mm_cache_stats=mm_cache_stats, engine_idx=engine_idx)
    def log(self):
        for l in self.per_engine_stat_loggers.values(): l.log()
    def log_engine_initialized(self):
        for l in self.per_engine_stat_loggers.values(): l.log_engine_initialized()

class PrometheusStatLogger(AggregateStatLoggerBase):
    def __init__(self, vllm_config: VllmConfig, engine_indexes: list[int] | None = None):
        if engine_indexes is None: engine_indexes = [0]
        self.engine_indexes = engine_indexes
        unregister_vllm_metrics()
        self.vllm_config = vllm_config
        self.show_hidden_metrics = vllm_config.observability_config.show_hidden_metrics
        self.kv_cache_metrics_enabled = vllm_config.observability_config.kv_cache_metrics
        labelnames = ["model_name", "engine"]
        model_name = vllm_config.model_config.served_model_name
        per_engine_labelvalues = {idx: [model_name, str(idx)] for idx in engine_indexes}
        self.spec_decoding_prom = SpecDecodingProm(vllm_config.speculative_config, labelnames, per_engine_labelvalues)
        self.kv_connector_prom = None # litevLLM - KVConnector bypassed
        
        # Scheduler state
        self.gauge_scheduler_running = make_per_engine(Gauge("vllm:num_requests_running", "Number of requests running", labelnames=labelnames), engine_indexes, model_name)
        self.gauge_scheduler_waiting = make_per_engine(Gauge("vllm:num_requests_waiting", "Number of requests waiting", labelnames=labelnames), engine_indexes, model_name)
        self.gauge_kv_cache_usage = make_per_engine(Gauge("vllm:kv_cache_usage_perc", "KV-cache usage", labelnames=labelnames), engine_indexes, model_name)
        
        # Counters
        self.counter_prompt_tokens = make_per_engine(Counter("vllm:prompt_tokens", "Prefill tokens processed", labelnames=labelnames), engine_indexes, model_name)
        self.counter_generation_tokens = make_per_engine(Counter("vllm:generation_tokens", "Generation tokens processed", labelnames=labelnames), engine_indexes, model_name)
        
        self.counter_request_success = {}
        counter_request_success_base = Counter("vllm:request_success", "Successfully processed requests", labelnames=labelnames + ["finished_reason"])
        for reason in FinishReason:
            self.counter_request_success[reason] = {idx: counter_request_success_base.labels(model_name, str(idx), str(reason)) for idx in engine_indexes}

    def record(self, scheduler_stats: SchedulerStats | None, iteration_stats: IterationStats | None, mm_cache_stats: MultiModalCacheStats | None = None, engine_idx: int = 0):
        if scheduler_stats is not None:
            self.gauge_scheduler_running[engine_idx].set(scheduler_stats.num_running_reqs)
            self.gauge_scheduler_waiting[engine_idx].set(scheduler_stats.num_waiting_reqs)
            self.gauge_kv_cache_usage[engine_idx].set(scheduler_stats.kv_cache_usage)
            if scheduler_stats.spec_decoding_stats is not None:
                self.spec_decoding_prom.observe(scheduler_stats.spec_decoding_stats, engine_idx)
        if iteration_stats:
            self.counter_prompt_tokens[engine_idx].inc(iteration_stats.num_prompt_tokens)
            self.counter_generation_tokens[engine_idx].inc(iteration_stats.num_generation_tokens)
            for finished_request in iteration_stats.finished_requests:
                self.counter_request_success[finished_request.finish_reason][engine_idx].inc()

    def log_engine_initialized(self): pass

def make_per_engine(metric, engine_idxs, model_name):
    return {idx: metric.labels(model_name, str(idx)) for idx in engine_idxs}

class StatLoggerManager:
    def __init__(self, vllm_config: VllmConfig, engine_idxs: list[int] | None = None, custom_stat_loggers: list[StatLoggerFactory] | None = None, enable_default_loggers: bool = True, aggregate_engine_logging: bool = False, client_count: int = 1):
        self.engine_indexes = engine_idxs if engine_idxs else [0]
        self.stat_loggers = []
        factories = list(custom_stat_loggers or [])
        if enable_default_loggers: factories.append(AggregatedLoggingStatLogger if aggregate_engine_logging else LoggingStatLogger)
        for f in factories:
            if isinstance(f, type) and issubclass(f, AggregateStatLoggerBase):
                self.stat_loggers.append(f(vllm_config=vllm_config, engine_indexes=self.engine_indexes))
            else:
                self.stat_loggers.append(PerEngineStatLoggerAdapter(vllm_config=vllm_config, engine_indexes=self.engine_indexes, per_engine_stat_logger_factory=f))
        if not any(isinstance(l, PrometheusStatLogger) for l in self.stat_loggers):
            self.stat_loggers.append(PrometheusStatLogger(vllm_config, self.engine_indexes))

    def record(self, scheduler_stats: SchedulerStats | None, iteration_stats: IterationStats | None, mm_cache_stats: MultiModalCacheStats | None = None, engine_idx: int | None = None):
        idx = engine_idx if engine_idx is not None else 0
        for l in self.stat_loggers: l.record(scheduler_stats, iteration_stats, mm_cache_stats=mm_cache_stats, engine_idx=idx)
    def log(self):
        for l in self.stat_loggers: l.log()
    def log_engine_initialized(self):
        for l in self.stat_loggers: l.log_engine_initialized()
    def record_sleep_state(self, sleep: int = 0, level: int = 0):
        for l in self.stat_loggers: l.record_sleep_state(sleep, level)
