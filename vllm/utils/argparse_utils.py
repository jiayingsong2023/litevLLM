# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

    def _split_lines(self, text, width):
        # The patterns also include whitespace after the newline
        single_newline = re.compile(r"(?<!\n)\n(?!\n)\s*")
        multiple_newlines = re.compile(r"\n{2,}\s*")
        text = single_newline.sub(" ", text)
        lines = re.split(multiple_newlines, text)
        return sum([textwrap.wrap(line, width) for line in lines], [])

    def add_arguments(self, actions):
        actions = sorted(actions, key=lambda x: x.option_strings)
        super().add_arguments(actions)

class FlexibleArgumentParser(ArgumentParser):
            return match.group(0).replace("_", "-")

        # Everything between the first -- and the first .
        pattern = re.compile(r"(?<=--)[^\.]*")

        # Convert underscores to dashes and vice versa in argument names
        processed_args = list[str]()
        for i, arg in enumerate(args):
            if arg.startswith("--help="):
                FlexibleArgumentParser._search_keyword = arg.split("=", 1)[-1].lower()
                processed_args.append("--help")
            elif arg.startswith("--"):
                if "=" in arg:
                    key, value = arg.split("=", 1)
                    key = pattern.sub(repl, key, count=1)
                    processed_args.append(f"{key}={value}")
                else:
                    key = pattern.sub(repl, arg, count=1)
                    processed_args.append(key)
            elif arg.startswith("-O") and arg != "-O":
                # allow -O flag to be used without space, e.g. -O3 or -Odecode
                # also handle -O=<optimization_level> here
                optimization_level = arg[3:] if arg[2] == "=" else arg[2:]
                processed_args += ["--optimization-level", optimization_level]
            elif (
                arg == "-O"
                and i + 1 < len(args)
                and args[i + 1] in {"0", "1", "2", "3"}
            ):
                # Convert -O <n> to --optimization-level <n>
                processed_args.append("--optimization-level")
            else:
                processed_args.append(arg)

        def create_nested_dict(keys: list[str], value: str) -> dict[str, Any]:
            nested_dict: Any = value
            for key in reversed(keys):
                nested_dict = {key: nested_dict}
            return nested_dict

        def recursive_dict_update(
            original: dict[str, Any],
            update: dict[str, Any],
        ) -> set[str]:
            duplicates = set[str]()
            for k, v in update.items():
                if isinstance(v, dict) and isinstance(original.get(k), dict):
                    nested_duplicates = recursive_dict_update(original[k], v)
                    duplicates |= {f"{k}.{d}" for d in nested_duplicates}
                elif isinstance(v, list) and isinstance(original.get(k), list):
                    original[k] += v
                else:
                    if k in original:
                        duplicates.add(k)
                    original[k] = v
            return duplicates

        delete = set[int]()
        dict_args = defaultdict[str, dict[str, Any]](dict)
        duplicates = set[str]()
        # Track regular arguments (non-dict args) for duplicate detection
        regular_args_seen = set[str]()
        for i, processed_arg in enumerate(processed_args):
            if i in delete:  # skip if value from previous arg
                continue

            if processed_arg.startswith("--") and "." not in processed_arg:
                if "=" in processed_arg:
                    arg_name = processed_arg.split("=", 1)[0]
                else:
                    arg_name = processed_arg

                if arg_name in regular_args_seen:
                    duplicates.add(arg_name)
                else:
                    regular_args_seen.add(arg_name)
                continue

            if processed_arg.startswith("-") and "." in processed_arg:
                if "=" in processed_arg:
                    processed_arg, value_str = processed_arg.split("=", 1)
                    if "." not in processed_arg:
                        # False positive, '.' was only in the value
                        continue
                else:
                    value_str = processed_args[i + 1]
                    delete.add(i + 1)

                if processed_arg.endswith("+"):
                    processed_arg = processed_arg[:-1]
                    value_str = json.dumps(list(value_str.split(",")))

                key, *keys = processed_arg.split(".")
                try:
                    value = json.loads(value_str)
                except json.decoder.JSONDecodeError:
                    value = value_str

                # Merge all values with the same key into a single dict
                arg_dict = create_nested_dict(keys, value)
                arg_duplicates = recursive_dict_update(dict_args[key], arg_dict)
                duplicates |= {f"{key}.{d}" for d in arg_duplicates}
                delete.add(i)
        # Filter out the dict args we set to None
        processed_args = [a for i, a in enumerate(processed_args) if i not in delete]
        if duplicates:
            logger.warning("Found duplicate keys %s", ", ".join(duplicates))

        # Add the dict args back as if they were originally passed as JSON
        for dict_arg, dict_value in dict_args.items():
            processed_args.append(dict_arg)
            processed_args.append(json.dumps(dict_value))

        return super().parse_args(processed_args, namespace)

    def check_port(self, value):
        try:
            value = int(value)
        except ValueError:
            msg = "Port must be an integer"
            raise ArgumentTypeError(msg) from None

        if not (1024 <= value <= 65535):
            raise ArgumentTypeError("Port must be between 1024 and 65535")

        return value

    def _pull_args_from_config(self, args: list[str]) -> list[str]:
        assert args.count("--config") <= 1, "More than one config file specified!"

        index = args.index("--config")
        if index == len(args) - 1:
            raise ValueError(
                "No config file specified! Please check your command-line arguments."
            )

        file_path = args[index + 1]

        config_args = self.load_config_file(file_path)

        # 0th index might be the sub command {serve,chat,complete,...}
        # optionally followed by model_tag (only for serve)
        # followed by config args
        # followed by rest of cli args.
        # maintaining this order will enforce the precedence
        # of cli > config > defaults
        if args[0].startswith("-"):
            # No sub command (e.g., api_server entry point)
            args = config_args + args[0:index] + args[index + 2 :]
        elif args[0] == "serve":
            model_in_cli = len(args) > 1 and not args[1].startswith("-")
            model_in_config = any(arg == "--model" for arg in config_args)

            if not model_in_cli and not model_in_config:
                raise ValueError(
                    "No model specified! Please specify model either "
                    "as a positional argument or in a config file."
                )

            if model_in_cli:
                # Model specified as positional arg, keep CLI version
                args = (
                    [args[0]]
                    + [args[1]]
                    + config_args
                    + args[2:index]
                    + args[index + 2 :]
                )
            else:
                # No model in CLI, use config if available
                args = [args[0]] + config_args + args[1:index] + args[index + 2 :]
        else:
            args = [args[0]] + config_args + args[1:index] + args[index + 2 :]

        return args

    def load_config_file(self, file_path: str) -> list[str]:
        extension: str = file_path.split(".")[-1]
        if extension not in ("yaml", "yml"):
            raise ValueError(
                f"Config file must be of a yaml/yml type. {extension} supplied"
            )

        # Supports both flat configs and nested dicts
        processed_args: list[str] = []

        config: dict[str, Any] = {}
        try:
            with open(file_path) as config_file:
                config = yaml.safe_load(config_file)
        except Exception as ex:
            logger.error(
                "Unable to read the config file at %s. Check path correctness",
                file_path,
            )
            raise ex

        for key, value in config.items():
            if isinstance(value, bool):
                if value:
                    processed_args.append("--" + key)
            elif isinstance(value, list):
                if value:
                    processed_args.append("--" + key)
                    for item in value:
                        processed_args.append(str(item))
            elif isinstance(value, dict):
                # Convert nested dicts to JSON strings so they can be parsed
                # by the existing JSON argument parsing machinery.
                processed_args.append("--" + key)
                processed_args.append(json.dumps(value))
            else:
                processed_args.append("--" + key)
                processed_args.append(str(value))

        return processed_args
