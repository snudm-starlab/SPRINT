import random
import itertools
import collections
import gc

import torch

import logging
import numpy as np

import lm_eval.api
import lm_eval.models
import lm_eval.api.metrics
import lm_eval.api.registry

from lm_eval.tasks import (
    get_task_dict,
    TaskManager
)
from lm_eval.utils import (
    positional_deprecated,
    run_task_tests,
    get_git_commit_hash,
    simple_parse_args_string,
    eval_logger
)


@positional_deprecated
def simple_evaluate(
    model,
    model_args=None,
    tasks=None,
    num_fewshot=None,
    batch_size=None,
    max_batch_size=None,
    device=None,
    use_cache=None,
    limit=None,
    bootstrap_iters: int = 100000,
    check_integrity: bool = False,
    decontamination_ngrams_path=None,
    write_out: bool = False,
    log_samples: bool = True,
    gen_kwargs: str = None,
    task_manager: TaskManager = None,
    verbosity: str = "INFO",
    predict_only: bool = False,
    compressed_model = None,
):
    """Instantiate and evaluate a model on a list of tasks.

    :param model: Union[str, LM]
        Name of model or LM object, see lm_eval.models.get_model
    :param model_args: Optional[str]
        String arguments for each model class, see LM.create_from_arg_string.
        Ignored if `model` argument is a LM object.
    :param tasks: list[Union[str, dict, Task]]
        List of task names or Task objects. Task objects will be taken to have name task.EVAL_HARNESS_NAME if defined and type(task).__name__ otherwise.
    :param num_fewshot: int
        Number of examples in few-shot context
    :param batch_size: int or str, optional
        Batch size for model
    :param max_batch_size: int, optional
        Maximal batch size to try with automatic batch size detection
    :param device: str, optional
        PyTorch device (e.g. "cpu" or "cuda:0") for running models
    :param use_cache: str, optional
        A path to a sqlite db file for caching model responses. `None` if not caching.
    :param limit: int or float, optional
        Limit the number of examples per task (only use this for testing), If <1, limit is a percentage of the total number of examples.
    :param bootstrap_iters:
        Number of iterations for bootstrap statistics
    :param check_integrity: bool
        Whether to run the relevant part of the test suite for the tasks
    :param write_out: bool
        If True, write out an example document and model input for checking task integrity
    :param log_samples: bool
        If True, write out all model outputs and documents for per-sample measurement and post-hoc analysis
    :param gen_kwargs: str
        String arguments for model generation
        Ignored for all tasks with loglikelihood output_type
    :param predict_only: bool
        If true only model outputs will be generated and returned. Metrics will not be evaluated

    :return
        Dictionary of results
    """
    random.seed(0)
    np.random.seed(1234)
    torch.manual_seed(
        1234
    )  # TODO: this may affect training runs that are run with evaluation mid-run.

    eval_logger.setLevel(getattr(logging, f"{verbosity}"))

    if tasks is None:
        tasks = []
    assert (
        tasks != []
    ), "No tasks specified, or no tasks found. Please verify the task names."

    if gen_kwargs is not None:
        gen_kwargs = simple_parse_args_string(gen_kwargs)
        eval_logger.warning(
            "generation_kwargs specified through cli, these settings will update set parameters in yaml tasks. Ensure 'do_sample=True' for non-greedy decoding!"
        )
        if gen_kwargs == "":
            gen_kwargs = None

    if isinstance(model, str):
        if model_args is None:
            model_args = ""
        lm = lm_eval.api.registry.get_model(model).create_from_arg_string(
            model_args,
            {
                "pretrained": compressed_model,
                "batch_size": batch_size,
                "max_batch_size": max_batch_size,
                "device": device,
            },
        )
        torch.cuda.empty_cache()
        gc.collect()
            
    else:
        assert isinstance(model, lm_eval.api.model.LM)
        lm = model

    if use_cache is not None:
        print(f"Using cache at {use_cache + '_rank' + str(lm.rank) + '.db'}")
        lm = lm_eval.api.model.CachingLM(
            lm,
            use_cache
            # each rank receives a different cache db.
            # necessary to avoid multiple writes to cache at once
            + "_rank"
            + str(lm.rank)
            + ".db",
        )

    if task_manager is None:
        task_manager = TaskManager(verbosity)

    eval_logger.info(
        "get_task_dict has been updated to accept an optional argument, `task_manager`"
        "Read more here: https://github.com/EleutherAI/lm-evaluation-harness/blob/recursive-groups/docs/interface.md#external-library-usage"
        )
    task_dict = get_task_dict(tasks, task_manager)
    for task_name in task_dict.keys():
        task_obj = task_dict[task_name]
        if isinstance(task_obj, tuple):
            _, task_obj = task_obj
            if task_obj is None:
                continue

        if task_obj.get_config("output_type") == "generate_until":
            if gen_kwargs is not None:
                task_obj.override_config(
                    key="generation_kwargs", value=gen_kwargs, update=True
                )

            if predict_only:
                log_samples = True
                eval_logger.info(
                    f"Processing {task_name} in output-only mode. Metrics will not be calculated!"
                )
                # we have to change the class properties post-hoc. This is pretty hacky.
                task_obj.override_metric(metric_name="bypass")

        if num_fewshot is not None:
            if (default_num_fewshot := task_obj.get_config("num_fewshot")) == 0:
                eval_logger.info(
                    f"num_fewshot has been set to 0 for {task_name} in its config. Manual configuration will be ignored."
                )
            else:
                eval_logger.warning(
                    f"Overwriting default num_fewshot of {task_name} from {default_num_fewshot} to {num_fewshot}"
                )
                task_obj.override_config(key="num_fewshot", value=num_fewshot)

    if check_integrity:
        run_task_tests(task_list=tasks)

    results = evaluate(
        lm=lm,
        task_dict=task_dict,
        limit=limit,
        bootstrap_iters=bootstrap_iters,
        decontamination_ngrams_path=decontamination_ngrams_path,
        write_out=write_out,
        log_samples=log_samples,
        verbosity=verbosity,
    )

    if lm.rank == 0:
        if isinstance(model, str):
            model_name = model
        elif hasattr(model, "config") and hasattr(model.config, "_name_or_path"):
            model_name = model.config._name_or_path
        else:
            model_name = type(model).__name__

        # add info about the model and few shot config
        results["config"] = {
            "model": model_name,
            "model_args": model_args,
            "batch_size": batch_size,
            "batch_sizes": list(lm.batch_sizes.values())
            if hasattr(lm, "batch_sizes")
            else [],
            "device": device,
            "use_cache": use_cache,
            "limit": limit,
            "bootstrap_iters": bootstrap_iters,
            "gen_kwargs": gen_kwargs,
        }
        results["git_hash"] = get_git_commit_hash()
        return results
    else:
        return None


decontaminate_suffix = "_decontaminate"


@positional_deprecated
def evaluate(
    lm,
    task_dict,
    limit=None,
    bootstrap_iters: int = 100000,
    decontamination_ngrams_path=None,
    write_out: bool = False,
    log_samples: bool = True,
    verbosity: str = "INFO",
):
    """Instantiate and evaluate a model on a list of tasks.

    :param lm: obj
        Language Model
    :param task_dict: dict[str, Task]
        Dictionary of tasks. Tasks will be taken to have name type(task).config.task .
    :param limit: int, optional
        Limit the number of examples per task (only use this for testing)
    :param bootstrap_iters:
        Number of iterations for bootstrap statistics
    :param write_out: bool
        If True, write out an example document and model input for checking task integrity
    :param log_samples: bool
        If True, write out all model outputs and documents for per-sample measurement and post-hoc analysis
    :return
        Dictionary of results
    """

    eval_logger.setLevel(getattr(logging, f"{verbosity}"))
    # decontaminate = decontamination_ngrams_path is not None

    for task_name, task in task_dict.items():
        if isinstance(task, tuple):
            _, task = task
        if not log_samples:
            assert (
                "bypass" not in getattr(task, "_metric_fn_list", {}).keys()
            ), f"log_samples must be True for 'bypass' only tasks: {task_name}"

    # stores the final result for each task, for each metric/filter pair.
    results = collections.defaultdict(dict)
    # Tracks each task's version.
    versions = collections.defaultdict(dict)
    # Tracks the YAML configs of all chosen tasks.
    configs = collections.defaultdict(dict)
    # logs info about each document evaluated.
    samples = collections.defaultdict(list)
    # tracks all Instances/requests a model must generate output on.
    requests = collections.defaultdict(list)
    # Aggregated task scores presented with groups
    results_agg = collections.defaultdict(dict)
    # Aggregated groups scores only
    groups_agg = collections.defaultdict(dict)
    # stores the amount to pad out reqs per req. type so that
    # number of fwd passes per distributed rank is equal
    padding_requests = collections.defaultdict(int)
    # store the hierarchy to do proper ordering
    task_hierarchy = collections.defaultdict(list)
    # store num-fewshot value per task
    num_fewshot = collections.defaultdict(int)

    # get lists of each type of request
    for task_name, task in task_dict.items():
        if isinstance(task, tuple):
            group_name, task = task
            task_hierarchy[group_name].append(task_name)
            versions[group_name] = "N/A"

        else:
            group_name = None
            task_hierarchy[task_name] = []

        if task is None:
            continue

        versions[task_name] = task.VERSION
        configs[task_name] = dict(task.dump_config())

        if "num_fewshot" in configs[task_name]:
            if configs[task_name]["metadata"]:
                n_shot = configs[task_name]["metadata"].get("num_fewshot", None)
            if not n_shot:
                n_shot = configs[task_name]["num_fewshot"]
        else:
            n_shot = 0 # TODO: is this always right?
        num_fewshot[task_name] = n_shot

        if "task_alias" in configs[task_name]:
            results[task_name]["alias"] = configs[task_name]["task_alias"]

        if (
            ("group_alias" in configs[task_name])
            and (group_name not in results)
            and (group_name is not None)
        ):
            results[group_name]["alias"] = configs[task_name]["group_alias"]

        if limit is not None:
            if task.has_test_docs():
                task_docs = task.test_docs()
            elif task.has_validation_docs():
                task_docs = task.validation_docs()
            else:
                raise RuntimeError("Task has neither test_docs nor validation_docs")
            limit = int(len(task_docs) * limit) if limit < 1.0 else int(limit)

        task.build_all_requests(limit=limit, rank=lm.rank, world_size=lm.world_size)

        eval_logger.debug(
            f"Task: {task_name}; number of requests on this rank: {len(task.instances)}"
        )

        if write_out:
            for inst in task.instances:
                # print the prompt for the first few documents
                if inst.doc_id < 1:
                    eval_logger.info(
                        f"Task: {task_name}; document {inst.doc_id}; context prompt (starting on next line):\
\n{inst.args[0]}\n(end of prompt on previous line)\ntarget string or answer choice index (starting on next line):\n{task.doc_to_target(inst.doc)}\n(end of target on previous line)"
                    )
                    eval_logger.info(f"Request: {str(inst)}")

        # aggregate Instances by LM method requested to get output.
        for instance in task.instances:
            reqtype = instance.request_type
            requests[reqtype].append(instance)

        if lm.world_size > 1:
            instances_rnk = torch.tensor(len(task._instances), device=lm.device)
            gathered_item = (
                lm.accelerator.gather(instances_rnk).cpu().detach().numpy().tolist()
            )

            # compute number of pseudobatches to pad with (FSDP/DDP require even batches among ranks)
            numpad = max(gathered_item) - gathered_item[lm.rank]
            padding_requests[task.OUTPUT_TYPE] += numpad

    ### Run LM on inputs, get all outputs ###
    # execute each type of request
    for reqtype, reqs in requests.items():
        eval_logger.info(f"Running {reqtype} requests")
        # create `K` copies of each request `req` based off `K = req.repeats`
        cloned_reqs = []
        for req in reqs:
            cloned_reqs.extend([req] * req.repeats)

        if (lm.world_size > 1) and (padding_requests[reqtype] > 0):
            for _ in range(padding_requests[reqtype]):
                cloned_reqs.extend([req] * req.repeats)

        # run requests through model
        resps = getattr(lm, reqtype)(cloned_reqs)

        # put responses from model into a list of length K for each request.
        for x, req in zip(resps, cloned_reqs):
            req.resps.append(x)

        if lm.world_size > 1:
            lm.accelerator.wait_for_everyone()

    ### Postprocess outputs ###
    # TODO: del model here, maybe (idea: allow user to specify device of e.g. reward model separately)
    for task_name, task in task_dict.items():
        if isinstance(task, tuple):
            group, task = task
            if task is None:
                continue
        task.apply_filters()

    ### Collect values of metrics on all datapoints ###
    vals = collections.defaultdict(list)

    # unpack results and sort back in order and return control to Task
    for task_name, task in task_dict.items():
        if isinstance(task, tuple):
            group, task = task
            if task is None:
                continue
        # TODO: make it possible to use a different metric per filter
        # iterate over different filters used
        for key in task.instances[0].filtered_resps.keys():
            doc_iterator = (
                itertools.islice(
                    enumerate(task.test_docs()), lm.rank, limit, lm.world_size
                )
                if task.has_test_docs()
                else itertools.islice(
                    enumerate(task.validation_docs()), lm.rank, limit, lm.world_size
                )
            )
            for doc_id, doc in doc_iterator:
                # subset instances to only this document id ; sort by idx
                requests = list(filter(lambda x: x.doc_id == doc_id, task.instances))
                requests.sort(key=lambda x: x.idx)
                metrics = task.process_results(
                    doc, [req.filtered_resps[key] for req in requests]
                )
                if log_samples:
                    target = task.doc_to_target(doc)
                    example = {
                        "doc_id": doc_id,
                        "doc": doc,
                        "target": target,
                        "arguments": [req.args for req in requests],
                        "resps": [req.resps for req in requests],
                        "filtered_resps": [req.filtered_resps[key] for req in requests],
                    }
                    example.update(metrics)
                    samples[task_name].append(example)
                for metric, value in metrics.items():
                    vals[(task_name, key, metric)].append(value)

    if lm.world_size > 1:
        # if multigpu, then gather data across all ranks
        # first gather logged samples across all ranks
        for task_name, task_samples in list(samples.items()):
            full_samples = [None] * lm.world_size
            torch.distributed.all_gather_object(full_samples, task_samples)

            samples[task_name] = list(itertools.chain.from_iterable(full_samples))

        # then collect metrics across all ranks
        vals_torch = collections.defaultdict(list)
        for (task_name, key, metric), items in vals.items():
            numitem = 0
            if isinstance(items[0], tuple):
                numitem = len(items[0])

            if isinstance(items[0], (str, list, tuple)):
                # handle the string case
                gathered_items = [None] * lm.accelerator.num_processes
                torch.distributed.all_gather_object(gathered_items, items)

                gathered_item = list(itertools.chain.from_iterable(gathered_items))
            else:
                # distributed gather requires all ranks to have same dimensions
                # so we pad out with float32 min value
                pad_value = torch.finfo(torch.float32).min
                metrics_tensor = torch.tensor(items, device=lm.device)

                original_dtype = metrics_tensor.dtype  # store original dtype
                torch_device_tensor = lm.accelerator.pad_across_processes(
                    metrics_tensor.to(torch.float32), pad_index=pad_value
                )
                gathered_item = lm.accelerator.gather(torch_device_tensor)

                if numitem > 0:
                    gathered_filtered = gathered_item[gathered_item[:, 0] != pad_value]
                else:
                    gathered_filtered = gathered_item[gathered_item != pad_value]

                gathered_item = (
                    gathered_filtered.to(original_dtype).cpu().detach().numpy().tolist()
                )
                # reconvert if we were passed a tuple of values
                if numitem > 0:
                    gathered_item = [tuple(g) for g in gathered_item]

            if lm.rank == 0:
                vals_torch[(task_name, key, metric)] = gathered_item

        vals = vals_torch

    if lm.rank == 0:

        ### Aggregate results over all datapoints ###
        # aggregate results ; run bootstrap CIs
        for (task_name, key, metric), items in vals.items():
            task = task_dict[task_name]
            metric_key = metric + "," + key

            if isinstance(task, tuple):
                group_name, task = task
            else:
                group_name = None

            agg_fn = task.aggregation()[metric]
            results[task_name][metric_key] = agg_fn(items)
            results[task_name]["samples"] = len(items)

            # hotfix: bleu, chrf, ter seem to be really expensive to bootstrap
            # so we run them less iterations. still looking for a cleaner way to do this
            if bootstrap_iters > 0:
                stderr = lm_eval.api.metrics.stderr_for_metric(
                    metric=task.aggregation()[metric],
                    bootstrap_iters=min(bootstrap_iters, 100)
                    if metric in ["bleu", "chrf", "ter"]
                    else bootstrap_iters,
                )

                if stderr is not None and len(items) > 1:
                    results[task_name][metric + "_stderr" + "," + key] = stderr(items)
                else:
                    results[task_name][metric + "_stderr" + "," + key] = "N/A"

        if bool(results):
            for group, task_list in reversed(task_hierarchy.items()):
                if task_list == []:
                    # TODO: No samples when bypass
                    total_size = results[group].get("samples", 999)
                else:
                    total_size = 0

                    for task in task_list:
                        metrics = results[task].copy()

                        if "alias" in metrics:
                            metrics.pop("alias")

                        current_size = metrics.pop("samples")

                        all_stderr = []
                        for metric in [
                            key for key in metrics.keys() if "_stderr" not in key
                        ]:
                            stderr = "_stderr,".join(metric.split(","))
                            stderr_score = results[task][stderr]
                            if stderr_score == "N/A":
                                var_score = "N/A"
                            else:
                                var_score = stderr_score**2
                                all_stderr.append(stderr)

                            metric_score = results[task][metric]

                            if metric in results[group]:
                                results[group][metric] = (
                                    results[group][metric] * total_size
                                    + metric_score * current_size
                                ) / (total_size + current_size)
                                # $$s_z^2 = \frac{(n-1) s_x^2 + (m-1) s_y^2}{n+m-1} + \frac{nm(\bar x - \bar y)^2}{(n+m)(n+m-1)}.$$
                                if var_score == "N/A" or results[group][stderr] == "N/A":
                                    results[group][stderr] = "N/A"
                                else:
                                    results[group][stderr] = (
                                        (total_size - 1) * results[group][stderr]
                                        + (current_size - 1) * var_score
                                    ) / (
                                        total_size + current_size - 1
                                    ) + total_size * current_size / (
                                        (total_size + current_size)
                                        * (total_size + current_size - 1)
                                    ) * (
                                        results[group][metric] - metric_score
                                    ) ** 2
                            else:
                                results[group][metric] = metric_score
                                results[group][stderr] = var_score

                        total_size += current_size

                    for stderr in all_stderr:
                        results[group][stderr] = np.sqrt(results[group][stderr])

                results[group]["samples"] = total_size

        def print_tasks(task_hierarchy, results, tab=0):
            results_agg = collections.defaultdict(dict)
            groups_agg = collections.defaultdict(dict)

            (group_name, task_list), *_ = task_hierarchy.items()
            task_list = sorted(task_list)

            results_agg[group_name] = results[group_name].copy()
            # results_agg[group_name]["tab"] = tab
            if "samples" in results_agg[group_name]:
                results_agg[group_name].pop("samples")

            tab_string = " " * tab + "- " if tab > 0 else ""

            if "alias" in results_agg[group_name]:
                results_agg[group_name]["alias"] = (
                    tab_string + results_agg[group_name]["alias"]
                )
            else:
                results_agg[group_name]["alias"] = tab_string + group_name

            if len(task_list) > 0:
                groups_agg[group_name] = results[group_name].copy()
                # groups_agg[group_name]["tab"] = tab
                if "samples" in groups_agg[group_name]:
                    groups_agg[group_name].pop("samples")

                if "alias" in groups_agg[group_name]:
                    groups_agg[group_name]["alias"] = (
                        tab_string + groups_agg[group_name]["alias"]
                    )
                else:
                    groups_agg[group_name]["alias"] = tab_string + group_name

                for task_name in task_list:
                    if task_name in task_hierarchy:
                        _task_hierarchy = {
                            **{task_name: task_hierarchy[task_name]},
                            **task_hierarchy,
                        }
                    else:
                        _task_hierarchy = {
                            **{task_name: []},
                            **task_hierarchy,
                        }

                    _results_agg, _groups_agg = print_tasks(
                        _task_hierarchy, results, tab + 1
                    )
                    results_agg = {**results_agg, **_results_agg}
                    groups_agg = {**groups_agg, **_groups_agg}

            return results_agg, groups_agg

        results_agg = collections.defaultdict(dict)
        groups_agg = collections.defaultdict(dict)
        all_tasks_list = list(task_hierarchy.keys())
        left_tasks_list = []
        while True:
            add_tasks_list = list(k for k in results_agg.keys())
            left_tasks_list = sorted(list(set(all_tasks_list) - set(add_tasks_list)))
            if len(left_tasks_list) == 0:
                break

            _task_hierarchy = {
                k: v for k, v in task_hierarchy.items() if k in left_tasks_list
            }
            _results_agg, _groups_agg = print_tasks(_task_hierarchy, results)

            results_agg = {**results_agg, **_results_agg}
            groups_agg = {**groups_agg, **_groups_agg}

        for group_name, task_list in task_hierarchy.items():
            if task_list != []:
                num_fewshot[group_name] = num_fewshot[task_list[0]] # TODO: validate this

        results_dict = {
            "results": dict(results_agg.items()),
            **({"groups": dict(groups_agg.items())} if bool(groups_agg) else {}),
            "configs": dict(sorted(configs.items())),
            "versions": dict(sorted(versions.items())),
            "n-shot": dict(sorted(num_fewshot.items())),
        }
        if log_samples:
            results_dict["samples"] = dict(samples)

        return results_dict

    else:
        return None
