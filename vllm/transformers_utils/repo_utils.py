# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
    file_path = try_get_local_file(model=model, file_name=file_name, revision=revision)

    if file_path is None:
        hf_hub_file = hf_hub_download(model, file_name, revision=revision)
        file_path = Path(hf_hub_file)

    if file_path is not None and file_path.is_file():
        with open(file_path, "rb") as file:
            return file.read()

    return None

def try_get_local_file(
    model: str | Path, file_name: str, revision: str | None = "main"
) -> Path | None:
    file_path = Path(model) / file_name
    if file_path.is_file():
        return file_path
    else:
        try:
            cached_filepath = try_to_load_from_cache(
                repo_id=model, filename=file_name, revision=revision
            )
            if isinstance(cached_filepath, str):
                return Path(cached_filepath)
        except ValueError:
            ...
    return None

def get_hf_file_to_dict(
    file_name: str, model: str | Path, revision: str | None = "main"
):

    file_path = try_get_local_file(model=model, file_name=file_name, revision=revision)

    if file_path is None:
        try:
            hf_hub_file = hf_hub_download(model, file_name, revision=revision)
        except huggingface_hub.errors.OfflineModeIsEnabled:
            return None
        except (
            RepositoryNotFoundError,
            RevisionNotFoundError,
            EntryNotFoundError,
            LocalEntryNotFoundError,
        ) as e:
            logger.debug("File or repository not found in hf_hub_download", e)
            return None
        except HfHubHTTPError as e:
            logger.warning(
                "Cannot connect to Hugging Face Hub. Skipping file download for '%s':",
                file_name,
                exc_info=e,
            )
            return None
        file_path = Path(hf_hub_file)

    if file_path is not None and file_path.is_file():
        with open(file_path) as file:
            return json.load(file)

    return None
