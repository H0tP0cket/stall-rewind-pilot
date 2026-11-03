"""External trajectory observers: T extraction, audits, summaries, renderings.

Entry points (all read recorded runs only through prefix.PrefixView):

  pipeline.observe_run(store, run_id, client, config)            primary incremental T chain
  pipeline.audit_checkpoint(store, run_id, checkpoint_id, ...)   independent full-prefix reference + comparison
  pipeline.freeze_packets(store, run_id, checkpoint_id, ...)     typed/prose/filtered packets + matched summary

Treatment construction lives in trajectory_retention.experiments; outcome
analysis in trajectory_retention.analysis. Prompts, filters and renderer
revisions are development versions until frozen through calibration.
"""
