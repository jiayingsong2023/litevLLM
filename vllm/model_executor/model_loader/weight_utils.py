# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
    if "HF_HUB_ENABLE_HF_TRANSFER" not in os.environ:
        try:
            # enable hf hub transfer if available
            import hf_transfer  # type: ignore # noqa

            huggingface_hub.constants.HF_HUB_ENABLE_HF_TRANSFER = True
        except ImportError:
            pass

enable_hf_transfer()

class DisabledTqdm(tqdm):
    def __init__(self, *args, **kwargs):
        kwargs["disable"] = True
        super().__init__(*args, **kwargs)

def get_lock(model_name_or_path: str | Path, cache_dir: str | None = None):
    lock_dir = cache_dir or temp_dir
    model_name_or_path = str(model_name_or_path)
    os.makedirs(os.path.dirname(lock_dir), exist_ok=True)
    model_name = model_name_or_path.replace("/", "-")
    hash_name = hashlib.sha256(model_name.encode()).hexdigest()
    # add hash to avoid conflict with old users' lock files
    lock_file_name = hash_name + model_name + ".lock"
    # mode 0o666 is required for the filelock to be shared across users
    lock = filelock.FileLock(os.path.join(lock_dir, lock_file_name), mode=0o666)
    return lock

@contextmanager
def atomic_writer(
    filepath: str | Path, mode: str = "w", encoding: str | None = None
) -> Generator[IO]:
    # Create a temporary file in the same directory as the target file
    # to ensure it's on the same filesystem for an atomic replace.
    temp_dir = os.path.dirname(filepath)
    temp_fd, temp_path = tempfile.mkstemp(dir=temp_dir)

    try:
        # Open the temporary file for writing
        with os.fdopen(temp_fd, mode=mode, encoding=encoding) as temp_file:
            yield temp_file

        # If the 'with' block completes successfully,
        # perform the atomic replace.
        os.replace(temp_path, filepath)

    except Exception:
        logger.exception(
            "Error during atomic write. Original file '%s' not modified", filepath
        )
        raise
    finally:
        # Clean up the temporary file if it still exists.
        if os.path.exists(temp_path):
            os.remove(temp_path)

def maybe_download_from_modelscope(
    model: str,
    revision: str | None = None,
    download_dir: str | None = None,
    ignore_patterns: str | list[str] | None = None,
    allow_patterns: list[str] | str | None = None,
) -> str | None:
    if envs.VLLM_USE_MODELSCOPE:
        # download model from ModelScope hub,
        # lazy import so that modelscope is not required for normal use.
        # pylint: disable=C.
        from modelscope.hub.snapshot_download import snapshot_download

        # Use file lock to prevent multiple processes from
        # downloading the same model weights at the same time.
        with get_lock(model, download_dir):
            if not os.path.exists(model):
                model_path = snapshot_download(
                    model_id=model,
                    cache_dir=download_dir,
                    local_files_only=huggingface_hub.constants.HF_HUB_OFFLINE,
                    revision=revision,
                    ignore_file_pattern=ignore_patterns,
                    allow_patterns=allow_patterns,
                )
            else:
                model_path = model
        return model_path
    return None

def _shared_pointers(tensors):
    ptrs = defaultdict(list)
    for k, v in tensors.items():
        ptrs[v.data_ptr()].append(k)
    failing = []
    for _, names in ptrs.items():
        if len(names) > 1:
            failing.append(names)
    return failing

def convert_bin_to_safetensor_file(
    pt_filename: str,
    sf_filename: str,
) -> None:
    loaded = torch.load(pt_filename, map_location="cpu", weights_only=True)
    if "state_dict" in loaded:
        loaded = loaded["state_dict"]
    shared = _shared_pointers(loaded)
    for shared_weights in shared:
        for name in shared_weights[1:]:
            loaded.pop(name)

    # For tensors to be contiguous
    loaded = {k: v.contiguous() for k, v in loaded.items()}

    dirname = os.path.dirname(sf_filename)
    os.makedirs(dirname, exist_ok=True)
    save_file(loaded, sf_filename, metadata={"format": "pt"})

    # check file size
    sf_size = os.stat(sf_filename).st_size
    pt_size = os.stat(pt_filename).st_size
    if (sf_size - pt_size) / pt_size > 0.01:

    # check if the tensors are the same
    reloaded = load_file(sf_filename)
    for k in loaded:
        pt_tensor = loaded[k]
        sf_tensor = reloaded[k]
        if not torch.equal(pt_tensor, sf_tensor):
            raise RuntimeError(f"The output tensors do not match for key {k}")

# TODO(woosuk): Move this to other place.
def get_quant_config(
    model_config: ModelConfig, load_config: LoadConfig
) -> QuantizationConfig:
    quant_cls = get_quantization_config(model_config.quantization)

    # GGUF doesn't have config file
    if model_config.quantization == "gguf":
        return quant_cls()

    # Read the quantization config from the HF model config, if available.
    hf_quant_config = getattr(model_config.hf_config, "quantization_config", None)
    # some vision model may keep quantization_config in their text_config
    hf_text_config = getattr(model_config.hf_config, "text_config", None)
    if hf_quant_config is None and hf_text_config is not None:
        hf_quant_config = getattr(hf_text_config, "quantization_config", None)
    if hf_quant_config is None:
        # compressed-tensors uses a compressions_config
        hf_quant_config = getattr(model_config.hf_config, "compression_config", None)

    # Pipe information about heads to enable TP-aware loading of attn_head scales
    if (
        hf_quant_config is not None
        and hf_quant_config.get("quant_method") == "compressed-tensors"
    ):
        if hf_text_config is not None:
            n_heads = getattr(hf_text_config, "num_attention_heads", None)
            n_kv_heads = getattr(hf_text_config, "num_key_value_heads", None)
        else:
            n_heads = getattr(model_config.hf_config, "num_attention_heads", None)
            n_kv_heads = getattr(model_config.hf_config, "num_key_value_heads", None)

        hf_quant_config["total_num_heads"] = n_heads
        hf_quant_config["total_num_kv_heads"] = (
            n_kv_heads if n_kv_heads is not None else n_heads
        )

    if hf_quant_config is not None:
        return quant_cls.from_config(hf_quant_config)

    # if hf_quant_config is None, we will try to get config from
    # hf_overrides
    hf_overrides = model_config.hf_overrides
    quantization_config_file = hf_overrides.get("quantization_config_file", None)
    if quantization_config_file is not None:
        if hasattr(quant_cls, "from_config_file"):
            return quant_cls.from_config_file(quantization_config_file)
        else:
            raise NotImplementedError(
                "from_config_file is specified in hf_override config, "
                "but quant_cls.from_config_file is not implemented in "
                f"{quant_cls}"
            )
    quantization_config_json = hf_overrides.get("quantization_config_dict_json", None)
    if quantization_config_json is not None:
        if hasattr(quant_cls, "from_config_dict_json"):
            return quant_cls.from_config_dict_json(quantization_config_json)
        else:
            raise NotImplementedError(
                "from_config_dict_json is specified in hf_override config, "
                "but quant_cls.from_config_dict_json is not implemented in "
                f"{quant_cls}"
            )

    # Inflight BNB quantization
    if model_config.quantization == "bitsandbytes":
        return quant_cls.from_config({})
    model_name_or_path = (
        maybe_download_from_modelscope(
            model_config.model,
            revision=model_config.revision,
            download_dir=load_config.download_dir,
            allow_patterns=["*.json"],
        )
        or model_config.model
    )
    is_local = os.path.isdir(model_name_or_path)
    if not is_local:
        # Download the config files.
        with get_lock(model_config.model, load_config.download_dir):
            hf_folder = snapshot_download(
                model_config.model,
                revision=model_config.revision,
                allow_patterns="*.json",
                cache_dir=load_config.download_dir,
                local_files_only=huggingface_hub.constants.HF_HUB_OFFLINE,
                tqdm_class=DisabledTqdm,
            )
    else:
        hf_folder = model_name_or_path

    possible_config_filenames = quant_cls.get_config_filenames()

    # If the quantization config is not found, use the default config.
    if not possible_config_filenames:
        return quant_cls()

    config_files = glob.glob(os.path.join(hf_folder, "*.json"))

    quant_config_files = [
        f for f in config_files if any(f.endswith(x) for x in possible_config_filenames)
    ]
    if len(quant_config_files) == 0:
        raise ValueError(f"Cannot find the config file for {model_config.quantization}")
    if len(quant_config_files) > 1:
        raise ValueError(
            f"Found multiple config files for {model_config.quantization}: "
            f"{quant_config_files}"
        )

    quant_config_file = quant_config_files[0]
    with open(quant_config_file) as f:
        config = json.load(f)

        if model_config.quantization == "bitsandbytes":
            config["adapter_name_or_path"] = model_config.model
        elif model_config.quantization == "modelopt":
            if config["producer"]["name"] == "modelopt":
                return quant_cls.from_config(config)
            else:
                raise ValueError(
                    f"Unsupported quantization config"
                    f" found for {model_config.quantization} in {f}."
                )

    return quant_cls.from_config(config)

def get_sparse_attention_config(
    model_config: ModelConfig,
    load_config: LoadConfig,
    sparse_attention_config_filename: str = "sparse_attention_config.json",
) -> dict[str, Any]:
    model_name_or_path = model_config.model
    is_local = os.path.isdir(model_name_or_path)
    if not is_local:
        # Download the config files.
        with get_lock(model_name_or_path, load_config.download_dir):
            hf_folder = snapshot_download(
                model_name_or_path,
                revision=model_config.revision,
                allow_patterns="*.json",
                cache_dir=load_config.download_dir,
                local_files_only=huggingface_hub.constants.HF_HUB_OFFLINE,
                tqdm_class=DisabledTqdm,
            )
    else:
        hf_folder = model_name_or_path

    config_file = os.path.join(hf_folder, sparse_attention_config_filename)
    if not os.path.exists(config_file):
        return {}

    # Load the sparse attention config.
    with open(config_file) as f:
        config = json.load(f)
    logger.info("Loaded sparse attention config from %s", config_file)

    return config

def download_gguf(
    repo_id: str,
    quant_type: str,
    cache_dir: str | None = None,
    revision: str | None = None,
    ignore_patterns: str | list[str] | None = None,
) -> str:
    # Use patterns that snapshot_download can handle directly
    # Patterns to match:
    # - *-{quant_type}.gguf (root)
    # - *-{quant_type}-*.gguf (root sharded)
    # - */*-{quant_type}.gguf (subdir)
    # - */*-{quant_type}-*.gguf (subdir sharded)
    allow_patterns = [
        f"*-{quant_type}.gguf",
        f"*-{quant_type}-*.gguf",
        f"*/*-{quant_type}.gguf",
        f"*/*-{quant_type}-*.gguf",
    ]

    # Use download_weights_from_hf which handles caching and downloading
    folder = download_weights_from_hf(
        model_name_or_path=repo_id,
        cache_dir=cache_dir,
        allow_patterns=allow_patterns,
        revision=revision,
        ignore_patterns=ignore_patterns,
    )

    # Find the downloaded file(s) in the folder
    local_files = []
    for pattern in allow_patterns:
        # Convert pattern to glob pattern for local filesystem
        glob_pattern = os.path.join(folder, pattern)
        local_files.extend(glob.glob(glob_pattern))

    if not local_files:
        raise ValueError(
            f"Downloaded GGUF files not found in {folder} for quant_type {quant_type}"
        )

    # Sort to ensure consistent ordering (prefer non-sharded files)
    local_files.sort(key=lambda x: (x.count("-"), x))
    return local_files[0]

def download_weights_from_hf(
    model_name_or_path: str,
    cache_dir: str | None,
    allow_patterns: list[str],
    revision: str | None = None,
    ignore_patterns: str | list[str] | None = None,
) -> str:
    assert len(allow_patterns) > 0
    local_only = huggingface_hub.constants.HF_HUB_OFFLINE
    if not local_only:
        # Attempt to reduce allow_patterns to a single pattern
        # so we only have to call snapshot_download once.
        try:
            fs = HfFileSystem()
            file_list = fs.ls(model_name_or_path, detail=False, revision=revision)

            # If downloading safetensors and an index file exists, use the
            # specific file names from the index to avoid downloading
            # unnecessary files (e.g., from subdirectories like "original/").
            index_file = f"{model_name_or_path}/{SAFE_WEIGHTS_INDEX_NAME}"
            if "*.safetensors" in allow_patterns and index_file in file_list:
                index_path = hf_hub_download(
                    repo_id=model_name_or_path,
                    filename=SAFE_WEIGHTS_INDEX_NAME,
                    cache_dir=cache_dir,
                    revision=revision,
                )
                with open(index_path) as f:
                    weight_map = json.load(f)["weight_map"]
                if weight_map:
                    # Extra [] so that weight_map files are treated as a
                    # single allow_pattern in the loop below
                    allow_patterns = [list(set(weight_map.values()))]  # type: ignore[list-item]
                else:
                    allow_patterns = ["*.safetensors"]
            else:
                # Use the first pattern found in the HF repo's files.
                for pattern in allow_patterns:
                    if fnmatch.filter(file_list, pattern):
                        allow_patterns = [pattern]
                        break
        except Exception as e:
            logger.warning(
                "Failed to get file list for '%s'. Trying each pattern in "
                "allow_patterns individually until weights have been "
                "downloaded. Error: %s",
                model_name_or_path,
                e,
            )

    logger.debug("Using model weights format %s", allow_patterns)
    # Use file lock to prevent multiple processes from
    # downloading the same model weights at the same time.
    with get_lock(model_name_or_path, cache_dir):
        start_time = time.perf_counter()
        for allow_pattern in allow_patterns:
            hf_folder = snapshot_download(
                model_name_or_path,
                allow_patterns=allow_pattern,
                ignore_patterns=ignore_patterns,
                cache_dir=cache_dir,
                tqdm_class=DisabledTqdm,
                revision=revision,
                local_files_only=local_only,
            )
            # If we have downloaded weights for this allow_pattern,
            # we don't need to check the rest.
            # allow_pattern can be a list (from weight_map) or str (glob)
            if isinstance(allow_pattern, list):
                break
            if any(Path(hf_folder).glob(allow_pattern)):
                break
        time_taken = time.perf_counter() - start_time
        if time_taken > 0.5:
            logger.info(
                "Time spent downloading weights for %s: %.6f seconds",
                model_name_or_path,
                time_taken,
            )
    return hf_folder

def download_safetensors_index_file_from_hf(
    model_name_or_path: str,
    index_file: str,
    cache_dir: str | None,
    revision: str | None = None,
) -> None:
    # Use file lock to prevent multiple processes from
    # downloading the same model weights at the same time.
    with get_lock(model_name_or_path, cache_dir):
        try:
            # Download the safetensors index file.
            hf_hub_download(
                repo_id=model_name_or_path,
                filename=index_file,
                cache_dir=cache_dir,
                revision=revision,
                local_files_only=huggingface_hub.constants.HF_HUB_OFFLINE,
            )
        # If file not found on remote or locally, we should not fail since
        # only some models will have index_file.
        except huggingface_hub.utils.LocalEntryNotFoundError:
            logger.info("No %s found in local cache.", index_file)
        except huggingface_hub.utils.EntryNotFoundError:
            logger.info("No %s found in remote.", index_file)

# For models like Mistral-7B-v0.3, there are both sharded
# safetensors files and a consolidated safetensors file.
# Passing both of these to the weight loader functionality breaks.
# So, we use the index_file to
# look up which safetensors files should be used.
def filter_duplicate_safetensors_files(
    hf_weights_files: list[str], hf_folder: str, index_file: str
) -> list[str]:
    # model.safetensors.index.json is a mapping from keys in the
    # torch state_dict to safetensors file holding that weight.
    index_file_name = os.path.join(hf_folder, index_file)
    if not os.path.isfile(index_file_name):
        return hf_weights_files

    # Iterate through the weight_map (weight_name: safetensors files)
    # to identify weights that we should use.
    with open(index_file_name) as f:
        weight_map = json.load(f)["weight_map"]
    weight_files_in_index = set()
    for weight_name in weight_map:
        weight_files_in_index.add(os.path.join(hf_folder, weight_map[weight_name]))
    # Filter out any fields that are not found in the index file.
    hf_weights_files = [f for f in hf_weights_files if f in weight_files_in_index]
    return hf_weights_files

def filter_files_not_needed_for_inference(hf_weights_files: list[str]) -> list[str]:
    blacklist = [
        "training_args.bin",
        "optimizer.bin",
        "optimizer.pt",
        "scheduler.pt",
        "scaler.pt",
    ]
    hf_weights_files = [
        f for f in hf_weights_files if not any(f.endswith(x) for x in blacklist)
    ]
    return hf_weights_files

# explicitly use pure text format, with a newline at the end
# this makes it impossible to see the animation in the progress bar
# but will avoid messing up with ray or multiprocessing, which wraps
# each line of output with some prefix.
_BAR_FORMAT = "{desc}: {percentage:3.0f}% Completed | {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]\n"  # noqa: E501

def enable_tqdm(use_tqdm_on_load: bool):
    return use_tqdm_on_load and (
        not torch.distributed.is_initialized() or torch.distributed.get_rank() == 0
    )

def np_cache_weights_iterator(
    model_name_or_path: str,
    cache_dir: str | None,
    hf_folder: str,
    hf_weights_files: list[str],
    use_tqdm_on_load: bool,
) -> Generator[tuple[str, torch.Tensor], None, None]:
    # Convert the model weights from torch tensors to numpy arrays for
    # faster loading.
    np_folder = os.path.join(hf_folder, "np")
    os.makedirs(np_folder, exist_ok=True)
    weight_names_file = os.path.join(np_folder, "weight_names.json")
    # Use file lock to prevent multiple processes from
    # dumping the same model weights to numpy at the same time.
    with get_lock(model_name_or_path, cache_dir):
        if not os.path.exists(weight_names_file):
            weight_names: list[str] = []
            for bin_file in tqdm(
                hf_weights_files,
                desc="Loading np_cache checkpoint shards",
                disable=not enable_tqdm(use_tqdm_on_load),
                bar_format=_BAR_FORMAT,
            ):
                state = torch.load(bin_file, map_location="cpu", weights_only=True)
                for name, param in state.items():
                    param_path = os.path.join(np_folder, name)
                    with open(param_path, "wb") as f:
                        np.save(f, param.cpu().detach().numpy())
                    weight_names.append(name)
            with open(weight_names_file, "w") as f:
                json.dump(weight_names, f)

    with open(weight_names_file) as f:
        weight_names = json.load(f)

    for name in weight_names:
        param_path = os.path.join(np_folder, name)
        with open(param_path, "rb") as f:
            param = np.load(f)
        yield name, torch.from_numpy(param)

def safetensors_weights_iterator(
    hf_weights_files: list[str],
    use_tqdm_on_load: bool,
    safetensors_load_strategy: str = "lazy",
) -> Generator[tuple[str, torch.Tensor], None, None]:

    def _load_file(st_file: str):
        result = load_file(st_file, device="cpu")
        return result

    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = [executor.submit(_load_file, st_file) for st_file in hf_weights_files]
        futures_iter = tqdm(
            concurrent.futures.as_completed(futures),
            total=len(hf_weights_files),
            desc="Multi-thread loading shards",
            disable=not enable_tqdm(use_tqdm_on_load),
            bar_format=_BAR_FORMAT,
        )

        for future in futures_iter:
            state_dict = future.result()
            yield from state_dict.items()

def runai_safetensors_weights_iterator(
    hf_weights_files: list[str],
    use_tqdm_on_load: bool,
    is_distributed: bool = False,
) -> Generator[tuple[str, torch.Tensor], None, None]:
    for bin_file in tqdm(
        hf_weights_files,
        desc="Loading pt checkpoint shards",
        disable=not enable_tqdm(use_tqdm_on_load),
        bar_format=_BAR_FORMAT,
    ):
        state = torch.load(
            bin_file, map_location=pt_load_map_location, weights_only=True
        )
        yield from state.items()
        del state

def multi_thread_pt_weights_iterator(
    hf_weights_files: list[str],
    use_tqdm_on_load: bool,
    pt_load_map_location: str | dict[str, str] = "cpu",
    max_workers: int = 4,
) -> Generator[tuple[str, torch.Tensor], None, None]:
    Return GGUF mapped weight's name and its quant type
    Iterate over the quant weights in the model gguf files and convert
    them to torch tensors.
    Be careful of the order of yielding weight types and weights data,
    we have to yield all weight types first before yielding any weights.
    Otherwise it would cause issue when loading weights with for packed
    layer with different quant types.

    PySafeSlice object supports indexing, which is done before loading the
    actual tensor and can reduce the amount of memory being read into the
    memory. However, it does not support more advanced functionalities
    like `.view()` or `.t()`. Therefore, if we need to modify the loaded
    tensor with these more complicated operators, we need to convert to
    tensor first.
    try:
        if param.numel() == 1 and loaded_weight.numel() == 1:
            # Sometimes scalar values aren't considered tensors with shapes
            # so if both param and loaded_weight are a scalar,
            # "broadcast" instead of copy
            param.data.fill_(loaded_weight.item())
        else:
            assert param.size() == loaded_weight.size(), (
                f"Attempted to load weight ({loaded_weight.size()}) "
                f"into parameter ({param.size()})"
            )

            param.data.copy_(loaded_weight)
    except Exception:
        # NOTE: This exception is added for the purpose of setting breakpoint to
        # debug weight loading issues.
        raise

def row_parallel_weight_loader(
    param: torch.Tensor, loaded_weight: torch.Tensor
) -> None:

    def loader(param: torch.Tensor, loaded_weight: torch.Tensor) -> None:
        tp_rank = get_tensor_model_parallel_rank()

        shard_size = param.data.shape[shard_axis]
        start_idx = tp_rank * shard_size
        loaded_weight = loaded_weight.narrow(shard_axis, start_idx, shard_size)

        return default_weight_loader(param, loaded_weight)

    return loader

def composed_weight_loader(
    loader: LoaderFunction, fn: Callable[[torch.Tensor], torch.Tensor]
) -> LoaderFunction:

    The model weights must be randomly initialized for accurate performance
    measurements. Additionally, the model weights should not cause NaNs in the
    forward pass. We empirically found that initializing the weights with
    values between -1e-3 and 1e-3 works well for most models.

    We use per-parameter random seed, so that dummy weights are consistent,
    even if the model is partitioned across multiple devices. When the seed
    is fixed, the random values generated by this function only depends on
    the parameter's number of elements and its data type.

    This function handles the remapping of FP8 k/v_scale parameter names.
    It detects if the given name ends with a suffix and attempts to remap
    it to the expected name format in the model. If the remapped name is not
    found in the params_dict, a warning is printed and None is returned.

    Args:
        name (str): The original loaded checkpoint parameter name.
        params_dict (dict): Dictionary containing the model's named parameters.

    Returns:
        str: The remapped parameter name if successful, or the original name
             if no remapping is needed.
        None: If the remapped name is not found in params_dict.
