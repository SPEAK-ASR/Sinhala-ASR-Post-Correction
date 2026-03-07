from dataclasses import dataclass


@dataclass
class TrainingConfig:
    """
    Configuration class for model training parameters.
    
    This class encapsulates all training-related arguments used by the Trainer.
    It provides fine-grained control over the training loop, evaluation, logging,
    checkpointing, and optimization strategies.
    """
    
    # Output and Basic Training Settings
    output_dir: str | None = "checkpoints/sinhala-asr-correction"
    """
    The output directory where model predictions and checkpoints will be written.
    
    >>> do_train: bool = False
    ...     Whether to run training. 
    ...     Not directly used by Trainer - intended for training scripts.
    >>> do_eval: bool = False
    ...     Whether to run evaluation on the validation set. 
    ...     Will be set to True if eval_strategy is not "no".
    >>> do_predict: bool = False
    ...     Whether to run predictions on the test set. 
    ...     Not directly used by Trainer - intended for evaluation scripts.
    """
    
    # Evaluation Settings
    eval_strategy: transformers.trainer_utils.IntervalStrategy | str = "epoch"
    """
    The evaluation strategy to adopt during training. Possible values:
    - "no": No evaluation during training
    - "steps": Evaluation every eval_steps
    - "epoch": Evaluation at the end of each epoch
    """
    
    prediction_loss_only: bool = False
    """When performing evaluation and predictions, only returns the loss."""
    
    # Batch Size Settings
    per_device_train_batch_size: int = 32
    """
    The batch size per device (GPU/TPU/CPU) for training.
    Global batch size = per_device_train_batch_size * number_of_devices * gradient_accumulation_steps
    """
    
    per_device_eval_batch_size: int = 16
    """The batch size per device for evaluation."""
    
    gradient_accumulation_steps: int = 1
    """
    Number of update steps to accumulate gradients before performing a backward/update pass.
    Logging, evaluation, and save will be conducted every gradient_accumulation_steps * xxx_step examples.
    """
    
    # eval_accumulation_steps: int | None = None
    """
    Number of prediction steps to accumulate output tensors before moving results to CPU.
    If unset, whole predictions are accumulated on device (faster but requires more memory).
    """
    
    # eval_delay: float = 0
    """Number of epochs or steps to wait before the first evaluation, depending on eval_strategy."""
    
    # torch_empty_cache_steps: int | None = None
    """
    Number of steps to wait before calling torch.<device>.empty_cache().
    Helps avoid CUDA OOM errors by lowering peak VRAM usage at ~10% performance cost.
    If unset or None, cache will not be emptied.
    """
    
    # Optimization Settings
    learning_rate: float = 5e-5
    """The initial learning rate for the AdamW optimizer."""
    
    weight_decay: float = 0.01
    """Weight decay to apply to all layers except bias and LayerNorm weights in AdamW optimizer."""
    
    # adam_beta1: float = 0.9
    """The beta1 hyperparameter for the AdamW optimizer."""
    
    # adam_beta2: float = 0.999
    """The beta2 hyperparameter for the AdamW optimizer."""
    
    # adam_epsilon: float = 1e-08
    """The epsilon hyperparameter for the AdamW optimizer."""
    
    # max_grad_norm: float = 1.0
    """Maximum gradient norm for gradient clipping."""
    
    # Training Duration Settings
    num_train_epochs: float = 10
    """
    Total number of training epochs to perform.
    If not an integer, will perform the decimal part of the last epoch before stopping.
    """
    
    # max_steps: int = -1
    """
    If set to positive number, total number of training steps to perform.
    Overrides num_train_epochs. Training continues through dataset until max_steps is reached.
    """
    
    # Learning Rate Scheduler Settings
    # lr_scheduler_type: transformers.trainer_utils.SchedulerType | str = 'linear'
    """The learning rate scheduler type to use. See SchedulerType for all possible values."""
    
    # lr_scheduler_kwargs: dict | str | None = None
    """Extra arguments for the lr_scheduler. See documentation of each scheduler for possible values."""
    
    # warmup_ratio: float | None = None
    """Ratio of total training steps used for linear warmup from 0 to learning_rate."""
    
    warmup_steps: float = 250
    """
    Number of steps for linear warmup from 0 to learning_rate.
    If < 1, interpreted as ratio of total training steps.
    """
    
    # Logging Settings
    # log_level: str = 'passive'
    """
    Logger log level on main process. Choices: 'debug', 'info', 'warning', 'error', 'critical'.
    'passive' doesn't set anything and keeps current log level (default "warning").
    """
    
    # log_level_replica: str = 'warning'
    """Logger log level on replica processes. Same choices as log_level."""
    
    # log_on_each_node: bool = True
    """In multinode distributed training, whether to log using log_level once per node or only on main node."""
    
    # logging_dir: str | None = None
    """Directory for storing logs. Defaults to output_dir/runs."""
    
    logging_strategy: transformers.trainer_utils.IntervalStrategy | str = 'steps'
    """
    The logging strategy to adopt during training. Possible values:
    - "no": No logging during training
    - "epoch": Logging at the end of each epoch
    - "steps": Logging every logging_steps
    """
    
    logging_first_step: bool = True
    """Whether to log the first global_step."""
    
    logging_steps: float = 50
    """
    Number of update steps between two logs if logging_strategy="steps".
    If < 1, interpreted as ratio of total training steps.
    """
    
    # logging_nan_inf_filter: bool = True
    """
    Whether to filter nan and inf losses for logging.
    If True, nan/inf step loss is filtered and average loss of logging window is used instead.
    Only influences logging, not gradient computation or application.
    """
    
    # Checkpoint Saving Settings
    save_strategy: transformers.trainer_utils.SaveStrategy | str = "epoch"
    """
    The checkpoint save strategy to adopt during training. Possible values:
    - "no": No save during training
    - "epoch": Save at the end of each epoch
    - "steps": Save every save_steps
    - "best": Save whenever a new best_metric is achieved
    Saving is also performed at the very end of training.
    """
    
    save_steps: float = 3000
    """
    Number of update steps between two checkpoint saves if save_strategy="steps".
    If < 1, interpreted as ratio of total training steps.
    """
    
    save_total_limit: int | None = 3
    """
    Limit the total number of checkpoints. Deletes older checkpoints in output_dir.
    When load_best_model_at_end is enabled, the "best" checkpoint is always retained
    in addition to the most recent ones.
    """
    
    # enable_jit_checkpoint: bool = False
    """
    Whether to enable Just-In-Time (JIT) checkpointing on SIGTERM signal.
    Useful for shared clusters with preemptible workloads.
    Configure graceful shutdown period: Kubernetes (terminationGracePeriodSeconds),
    Slurm (--signal=USR1@<seconds>). Calculate as: iteration_time + checkpoint_save_time.
    """
    
    # save_on_each_node: bool = False
    """
    In multi-node distributed training, whether to save on each node or only on main.
    Don't activate when nodes share the same storage.
    """
    
    # save_only_model: bool = False
    """
    When checkpointing, whether to save only model or also optimizer, scheduler & rng state.
    If True, can't resume training from checkpoint but saves storage.
    Can only load model with from_pretrained when this is True.
    """
    
    # restore_callback_states_from_checkpoint: bool = False
    """
    Whether to restore callback states from checkpoint.
    If True, will override callbacks passed to Trainer if they exist in checkpoint.
    """
    
    # Hardware and Precision Settings
    # use_cpu: bool = False
    """Whether to use CPU. If False, will use available torch device/backend."""
    
    seed: int = 42
    """
    Random seed set at beginning of training.
    For reproducibility, use model_init function to instantiate model
    if it has randomly initialized parameters.
    """
    
    data_seed: int | None = 42
    """
    Random seed for data samplers. If not set, data sampling uses same seed as 'seed'.
    Ensures reproducibility of data sampling independent of model seed.
    """
    
    bf16: bool = True
    """
    Whether to use bf16 16-bit (mixed) precision training instead of 32-bit.
    Requires Ampere or higher NVIDIA architecture, Intel XPU, CPU, or Ascend NPU.
    """
    
    fp16: bool = False
    """Whether to use fp16 16-bit (mixed) precision training instead of 32-bit. Set False if on CPU or older GPU."""
    
    # bf16_full_eval: bool = False
    """
    Whether to use full bfloat16 evaluation instead of 32-bit.
    Faster and saves memory but can harm metric values.
    """
    
    # fp16_full_eval: bool = False
    """
    Whether to use full float16 evaluation instead of 32-bit.
    Faster and saves memory but can harm metric values.
    """
    
    # tf32: bool | None = None
    """
    Whether to enable TF32 mode (available in Ampere and newer GPU architectures).
    Default depends on PyTorch's torch.backends.cuda.matmul.allow_tf32.
    Experimental API that may change.
    """
    
    # local_rank: int = -1
    """Local rank for distributed training. Set automatically by launch script."""
    
    # Distributed Training Settings
    # ddp_backend: str | None = None
    """The backend to use for distributed training. Must be one of: "nccl", "mpi", "ccl", "gloo", "hccl"."""
    
    # debug: str | list[transformers.debug_utils.DebugOption] = ''
    """
    Enable debug features (experimental). Possible options (separated by whitespace):
    - "underflow_overflow": Detects overflow in model's input/outputs and reports last frames
    - "tpu_metrics_debug": Print debug metrics on TPU
    """
    
    # DataLoader Settings
    # dataloader_drop_last: bool = False
    """Whether to drop the last incomplete batch if dataset length isn't divisible by batch size."""
    
    eval_steps: float | None = 3000
    """
    Number of update steps between two evaluations if eval_strategy="steps".
    Defaults to same value as logging_steps if not set.
    If < 1, interpreted as ratio of total training steps.
    """
    
    dataloader_num_workers: int = 2
    """Number of subprocesses for data loading (PyTorch only). 0 means data loaded in main process."""
    
    dataloader_prefetch_factor: int | None = 2
    """
    Number of batches loaded in advance by each worker.
    2 means 2 * num_workers batches prefetched across all workers.
    """
    
    # run_name: str | None = None
    """A descriptor for the run. Used for wandb, mlflow, comet, swanlab, trackio logging."""
    
    # disable_tqdm: bool | None = None
    """
    Whether to disable tqdm progress bars and metrics table in Jupyter Notebooks.
    Defaults to True if log level is warn or lower, False otherwise.
    """
    
    remove_unused_columns: bool = True
    """Whether to automatically remove columns unused by model forward method."""
    
    label_names: list[str] | None = field(default_factory=lambda: ["labels"])
    """
    List of keys in input dictionary that correspond to labels.
    Defaults to keys containing "label" (except XxxForQuestionAnswering models,
    which also include ["start_positions", "end_positions"]).
    Only specify for custom label names or multiple label tensors.
    """
    
    # Best Model Loading Settings
    load_best_model_at_end: bool = True
    """
    Whether to load best model found during training at end of training.
    When enabled, best checkpoint is always saved.
    Requires save_strategy == eval_strategy, and save_steps must be
    a round multiple of eval_steps if strategy is "steps".
    """
    
    metric_for_best_model: str | None = "bleu"
    """
    Metric to use to compare models with load_best_model_at_end.
    Must be name of metric returned by evaluation with or without "eval_" prefix.
    Defaults to "loss" when load_best_model_at_end == True or lr_scheduler_type == REDUCE_ON_PLATEAU.
    """
    
    greater_is_better: bool | None = True
    """
    Whether better models should have greater metric (use with load_best_model_at_end).
    Defaults to True if metric_for_best_model doesn't end in "loss", False otherwise.
    """
    
    # ignore_data_skip: bool = False
    """
    When resuming training, whether to skip epochs/batches to get data loading at same stage.
    If True, training begins faster but won't yield same results as interrupted training.
    """
    
    # Advanced Distributed Training Settings
    # fsdp: list[transformers.trainer_utils.FSDPOption] | str | None = None
    """
    PyTorch Fully Sharded Data Parallel Training options (distributed training only):
    - "full_shard": Shard parameters, gradients and optimizer states
    - "shard_grad_op": Shard optimizer states and gradients
    - "hybrid_shard": FULL_SHARD within node, replicate parameters across nodes
    - "hybrid_shard_zero2": SHARD_GRAD_OP within node, replicate parameters across nodes
    - "offload": Offload parameters and gradients to CPUs (with "full_shard" or "shard_grad_op")
    - "auto_wrap": Automatically wrap layers with FSDP using default_auto_wrap_policy
    """
    
    # fsdp_config: dict[str, Any] | str | None = None
    """
    Config for FSDP (PyTorch Distributed Parallel Training).
    Either path to FSDP json config file or loaded json dict.
    Options include: fsdp_version, min_num_params, transformer_layer_cls_to_wrap,
    backward_prefetch, forward_prefetch, limit_all_gathers, use_orig_params,
    sync_module_states, cpu_ram_efficient_loading, activation_checkpointing, xla, etc.
    """
    
    # accelerator_config: dict | str | None = None
    """
    Config for internal Accelerator implementation.
    Path to accelerator json config, loaded json dict, or AcceleratorConfig instance.
    Options: split_batches, dispatch_batches, even_batches, use_seedable_sampler,
    use_configured_state.
    """
    
    # parallelism_config: accelerate.parallelism_config.ParallelismConfig | None = None
    """
    Parallelism configuration for the training run. 
    - Requires Accelerate 1.10.1+
    """
    
    # deepspeed: dict | str | None = None
    """
    Use DeepSpeed (experimental). Either location of DeepSpeed json config file
    or loaded json dict. For Zero-init, ensure model is not initialized until
    after TrainingArguments initialization.
    """
    
    # label_smoothing_factor: float = 0.0
    """
    Label smoothing factor. 0 means no smoothing.
    Otherwise, onehot labels change from 0s/1s to:
    label_smoothing_factor/num_labels and 1 - label_smoothing_factor + label_smoothing_factor/num_labels
    """
    
    # Optimizer Settings
    optim: transformers.training_args.OptimizerNames | str = 'adamw_torch_fused'
    """
    Optimizer to use: "adamw_torch", "adamw_torch_fused", "adamw_anyprecision", "adafactor", etc.
    See OptimizerNames in training_args.py for full list.
    """
    
    # optim_args: str | None = None
    """Optional arguments supplied to optimizers like AnyPrecisionAdamW, AdEMAMix, and GaLore."""
    
    # group_by_length: bool = False
    """
    Whether to group samples of roughly same length in training dataset.
    Minimizes padding and improves efficiency. Only useful with dynamic padding.
    """
    
    # length_column_name: str = 'length'
    """
    Column name for precomputed lengths. If exists, grouping by length uses these values
    rather than computing them at train startup. Ignored unless group_by_length is True
    and dataset is an instance of Dataset.
    """
    
    # Reporting and Tracking Settings
    report_to: None | str | list[str] = field(default_factory=lambda: ["wandb"])
    """
    Integrations to report results and logs to.
    Supported: "azure_ml", "clearml", "codecarbon", "comet_ml", "dagshub", "dvclive",
    "flyte", "mlflow", "swanlab", "tensorboard", "trackio", "wandb".
    Use "all" for all installed integrations, "none" for no integrations.
    """
    
    # project: str = 'huggingface'
    """The project name for logging. Currently only used by Trackio."""
    
    # trackio_space_id: str | None = 'trackio'
    """
    Hugging Face Space ID for Trackio deployment.
    Format: 'username/reponame', 'orgname/reponame', or 'reponame'.
    If None, logs to local directory. Space is public unless hub_private_repo=True.
    """
    
    # ddp_find_unused_parameters: bool | None = None
    """
    In distributed training, value of find_unused_parameters flag passed to DistributedDataParallel.
    Defaults to False if gradient checkpointing used, True otherwise.
    """
    
    # ddp_bucket_cap_mb: int | None = None
    """In distributed training, value of bucket_cap_mb flag passed to DistributedDataParallel."""
    
    # ddp_broadcast_buffers: bool | None = None
    """
    In distributed training, value of broadcast_buffers flag passed to DistributedDataParallel.
    Defaults to False if gradient checkpointing used, True otherwise.
    """
    
    dataloader_pin_memory: bool = True
    """Whether to pin memory in data loaders."""
    
    dataloader_persistent_workers: bool = True
    """
    If True, data loader won't shut down worker processes after dataset consumed once.
    Maintains workers Dataset instances alive. Can speed up training but increases RAM usage.
    """
    
    # skip_memory_metrics: bool = True
    """
    Whether to skip memory profiler reports in metrics.
    Skipped by default as it slows down training and evaluation.
    """
    
    # Hub Settings
    push_to_hub: bool = True
    """
    Whether to push model to Hub every time model is saved.
    output_dir will be git directory synced with repo (determined by hub_model_id).
    Content pushed each time save is triggered (per save_strategy). save_model() also triggers push.
    If output_dir exists, must be local clone of repository to push to.
    """
    
    # resume_from_checkpoint: str | None = None
    """
    Path to folder with valid checkpoint for model.
    Not directly used by Trainer - intended for training/evaluation scripts.
    """
    
    hub_model_id: str | None = HF_MODEL_ID
    """
    Name of repository to keep in sync with output_dir.
    Can be simple model ID (pushed in your namespace) or full repo name "user_name/model".
    Defaults to user_name/output_dir_name.
    """
    
    hub_strategy: transformers.trainer_utils.HubStrategy | str = 'checkpoint'
    """
    Defines scope of what is pushed to Hub and when:
    - "end": Push model, config, tokenizer, and model card on save_model()
    - "every_save": Push on each model save (asynchronous)
    - "checkpoint": Like "every_save" + push latest checkpoint in "last-checkpoint" subfolder
    - "all_checkpoints": Like "checkpoint" but all checkpoints are pushed
    """
    
    # hub_token: str | None = None
    """Token to push model to Hub. Defaults to token from cache folder (hf auth login)."""
    
    # hub_private_repo: bool | None = None
    """
    Whether to make repo private. If None (default), repo is public unless
    organization's default is private. Ignored if repo already exists.
    Also applies to Trackio Spaces deployment.
    """
    
    # hub_always_push: bool = False
    """Unless True, Trainer will skip pushing checkpoint when previous push isn't finished."""
    
    # hub_revision: str | None = None
    """The revision to use when pushing to Hub. Can be branch name, tag, or commit hash."""
    
    # Gradient Checkpointing Settings
    # gradient_checkpointing: bool = False
    """If True, use gradient checkpointing to save memory at expense of slower backward pass."""
    
    # gradient_checkpointing_kwargs: dict[str, Any] | str | None = None
    """Keyword arguments passed to gradient_checkpointing_enable method."""
    
    # Metrics and Evaluation Settings
    # include_for_metrics: list = field(default_factory=list)
    """
    Include additional data in compute_metrics function.
    Options: "inputs" (input data for input-dependent metrics),
    "loss" (loss values for loss-dependent metrics).
    """
    
    # eval_do_concat_batches: bool = True
    """
    Whether to recursively concat inputs/losses/labels/predictions across batches.
    If False, stores them as lists with each batch kept separate.
    """
    
    auto_find_batch_size: bool = True
    """
    Whether to find batch size that fits into memory automatically through exponential decay,
    avoiding CUDA OOM errors. Requires accelerate (pip install accelerate).
    """
    
    # full_determinism: bool = False
    """
    If True, enable_full_determinism() called instead of set_seed() for reproducible
    results in distributed training. WARNING: Negatively impacts performance.
    """
    
    # ddp_timeout: int = 1800
    """
    Timeout for torch.distributed.init_process_group calls (seconds).
    Avoids GPU socket timeouts during slow operations in distributed training.
    """
    
    # PyTorch Compilation Settings
    # torch_compile: bool = False
    """
    Whether to compile model using PyTorch 2.0 torch.compile.
    Uses best defaults for torch.compile API. Customize with torch_compile_backend
    and torch_compile_mode. Experimental - subject to change in future releases.
    """
    
    # torch_compile_backend: str | None = None
    """
    Backend to use in torch.compile. If set, torch_compile will be True.
    Experimental - subject to change in future releases.
    """
    
    # torch_compile_mode: str | None = None
    """
    Mode to use in torch.compile. If set, torch_compile will be True.
    Experimental - subject to change in future releases.
    """
    
    # Token Tracking Settings
    # include_num_input_tokens_seen: str | bool = 'no'
    """
    Whether to track number of input tokens seen.
    Must be one of ["all", "non_padding", "no"] or boolean (maps to "all" or "no").
    May be slower in distributed training due to gather operations.
    """
    
    # Training Techniques
    neftune_noise_alpha: float | None = 5.0
    """
    If not None, activates NEFTune noise embeddings. Can drastically improve
    instruction fine-tuning performance. Supports PreTrainedModel and PeftModel.
    Original paper used values in range [5.0, 15.0].
    """
    
    # optim_target_modules: None | str | list[str] = None
    """
    Target modules to optimize (module names to train).
    Used for GaLore and APOLLO algorithms. Must pass valid GaLore or APOLLO optimizer
    ("apollo_adamw", "galore_adamw", "galore_adamw_8bit", "galore_adafactor").
    Target modules must be nn.Linear modules only.
    """
    
    batch_eval_metrics: bool = True
    """
    If True, evaluation calls compute_metrics at end of each batch to accumulate statistics
    rather than saving all eval logits in memory. Must pass compute_metrics function that
    takes boolean compute_result argument to trigger final global summary statistics.
    """
    
    # eval_on_start: bool = False
    """Whether to perform evaluation step (sanity check) before training to ensure validation works."""
    
    # use_liger_kernel: bool = False
    """
    Whether to enable Liger Kernel for LLM training. Increases multi-GPU throughput by ~20%
    and reduces memory usage by ~60%. Works with flash attention, PyTorch FSDP, and DeepSpeed.
    Supports llama, mistral, mixtral, and gemma models.
    """
    
    # liger_kernel_config: dict[str, bool] | None = None
    """
    Configuration for Liger Kernel. When use_liger_kernel=True, this dict is passed
    to _apply_liger_kernel_to_instance. Options: 'rope', 'swiglu', 'cross_entropy',
    'fused_linear_cross_entropy', 'rms_norm', etc. If None, uses default kernel configurations.
    """
    
    # eval_use_gather_object: bool = False
    """
    Whether to recursively gather objects in nested list/tuple/dictionary from all devices.
    Only enable if not just returning tensors. Actively discouraged by PyTorch.
    """
    
    # average_tokens_across_devices: bool = True
    """
    Whether to average tokens across devices. If enabled, uses all_reduce to synchronize
    num_tokens_in_batch for precise loss calculation.
    """
    
    use_cache: bool = True
    """
    Whether to enable cache for the model. Usually not needed for training
    except some PEFT methods that use past_key_values.
    """
    
    # sortish_sampler: bool = False
    """Whether to use sortish sampler for training (groups similar length samples)."""
    
    # Generation Settings
    predict_with_generate: bool = True
    """Whether to use generate to calculate generative metrics (ROUGE, BLEU)."""
    
    generation_max_length: int | None = 256
    """
    The max_length to use on each evaluation loop when predict_with_generate=True.
    Defaults to model configuration's max_length value.
    """
    
    generation_num_beams: int | None = 4
    """
    The num_beams to use on each evaluation loop when predict_with_generate=True.
    Defaults to model configuration's num_beams value.
    """
    
    # generation_config: str | pathlib.Path | transformers.generation.configuration_utils.GenerationConfig | None = None
    """
    Allows loading GenerationConfig from from_pretrained method. Can be:
    - String: model id of pretrained model configuration on huggingface.co
    - Path: directory containing configuration file saved using save_pretrained()
    - GenerationConfig object
    """

    # ── Fields below are NOT passed to Seq2SeqTrainingArguments ─────────────
    early_stopping_patience: int = 3
    use_wandb: bool = False     # set True to log to Weights & Biases
    wandb_project: str = "sinhala-asr-correction"