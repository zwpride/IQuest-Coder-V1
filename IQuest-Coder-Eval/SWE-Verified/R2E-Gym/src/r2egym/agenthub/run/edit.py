# editagent_script.py

import openai
import re
import yaml
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional
from datetime import datetime
import json
import concurrent.futures
import threading
import docker
import os

from r2egym.agenthub.runtime.docker import DockerRuntime, cleanup_docker_client_pool, get_docker_pool_stats
from r2egym.agenthub.environment.env import EnvArgs, RepoEnv
from r2egym.agenthub.agent.agent import AgentArgs, Agent
# from r2egym.agenthub.agent.context_agent import AgentArgs, Agent

from r2egym.docker_bash_utils.docker_list_tags import fetch_docker_tags

MAX_TURN_RETRY = int(os.environ.get("MAX_TURN_RETRY", 5))
def safe_read_jsonl(file_path):
    """Safely read JSONL file, skipping lines with parsing errors"""
    data = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for idx, line in enumerate(f):
            try:
                data.append(json.loads(line))
            except json.JSONDecodeError as e:
                print(f"Error decoding JSON on line {idx+1}: {e}")
    return data
from r2egym.agenthub.utils.log import get_logger
from r2egym.logging import setup_logging, INFO
from r2egym.agenthub.utils.utils import get_parsed_commit

from fire import Fire
from r2egym.agenthub.utils.utils import match_dockerimage_to_repo
from r2egym.agenthub import SUPPORTED_REPOS
from datasets import load_dataset
from r2egym.agenthub.trajectory import TrajectoryStep, Trajectory
import time

##############################################################################
# Initialize Logger
##############################################################################
logger = get_logger(__name__)  # Initialize the logger

##############################################################################
# Initialize File Lock for Thread-Safe Writing
##############################################################################
file_lock = threading.Lock()


##############################################################################
# Utility Function
##############################################################################
def get_docker_images(repo_name) -> List[str]:
    """
    Fetches the list of Docker images available for the base image.

    Returns:
        A list of Docker image tags.
    """
    base_image = f"namanjain12/{repo_name}new"
    tags = fetch_docker_tags(base_image)
    docker_image_list = [f"{base_image}:{x['name']}" for x in tags]
    return docker_image_list


def prepull_docker_image(docker_image: str) -> bool:
    """
    Prepulls a single Docker image.
    
    Args:
        docker_image: The Docker image name to pull
        
    Returns:
        True if successful, False otherwise
    """
    try:
        client = docker.from_env()
        logger.info(f"Pulling Docker image: {docker_image}")
        client.images.pull(docker_image)
        logger.info(f"Successfully pulled Docker image: {docker_image}")
        return True
    except Exception as e:
        logger.error(f"Failed to pull Docker image {docker_image}: {e}")
        return False


def prepull_docker_images(ds_selected: List[Dict], max_workers: Optional[int] = None) -> None:
    """
    Prepulls all Docker images in parallel before starting the main execution.
    
    Args:
        ds_selected: List of dataset entries containing docker_image keys
        max_workers: Maximum number of threads for parallel pulling
    """
    # Extract unique Docker images
    docker_images = list(set([ds_entry["docker_image"] for ds_entry in ds_selected]))
    logger.info(f"Starting parallel prepull of {len(docker_images)} unique Docker images...")
    
    # Use ThreadPoolExecutor for I/O bound operations like Docker pulls
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        # Submit all pull tasks
        future_to_image = {
            executor.submit(prepull_docker_image, docker_image): docker_image
            for docker_image in docker_images
        }
        
        # Track results
        successful_pulls = []
        failed_pulls = []
        
        for future in concurrent.futures.as_completed(future_to_image):
            docker_image = future_to_image[future]
            try:
                success = future.result()
                if success:
                    successful_pulls.append(docker_image)
                else:
                    failed_pulls.append(docker_image)
            except Exception as e:
                logger.error(f"Exception during prepull of {docker_image}: {e}")
                failed_pulls.append(docker_image)
    
    logger.info(f"Prepull completed. Success: {len(successful_pulls)}, Failed: {len(failed_pulls)}")
    if failed_pulls:
        logger.warning(f"Failed to pull images: {failed_pulls}")


def fetch_repos_from_docker_images(ds_selected, docker_image_file):
    """
    Fetches the list of repositories from the provided Docker images.

    Args:
        ds_selected: dataset containing docker_image keys
        docker_image_file: Path to the file containing valid docker images
    Returns:
        A dataset of dataset entries with valid docker images
    """
    # Validate and load valid docker images from json file safely
    if not isinstance(docker_image_file, str):
        logger.warning(
            f"docker_image_file is not a string ({type(docker_image_file)}). Skip docker image filtering."
        )
        return ds_selected

    docker_image_file = docker_image_file.strip()
    if docker_image_file == "":
        logger.info("docker_image_file is empty. Skip docker image filtering.")
        return ds_selected

    if not os.path.exists(docker_image_file):
        logger.warning(
            f"docker_image_file does not exist: {docker_image_file}. Skip docker image filtering."
        )
        return ds_selected

    try:
        with open(docker_image_file, 'r', encoding='utf-8') as f:
            loaded = json.load(f)
    except Exception as e:
        logger.error(f"Failed to read docker_image_file {docker_image_file}: {e}. Skip filtering.")
        return ds_selected

    # Accept list directly, or dict with a common key
    if isinstance(loaded, list):
        valid_docker_images = loaded
    elif isinstance(loaded, dict):
        for key in ["valid_docker_images", "docker_images", "images"]:
            if key in loaded and isinstance(loaded[key], list):
                valid_docker_images = loaded[key]
                break
        else:
            logger.error(
                f"Unsupported JSON structure in {docker_image_file}. Expected list or dict with a list under known keys. Skip filtering."
            )
            return ds_selected
    else:
        logger.error(
            f"Unsupported JSON type in {docker_image_file}: {type(loaded)}. Skip filtering."
        )
        return ds_selected

    logger.info(f"Loaded {len(valid_docker_images)} valid docker images from {docker_image_file}")
    # Filter ds_selected to only include entries with docker_image in valid_docker_images, using dataset's filter method
    filtered_ds = ds_selected.filter(lambda x: x["docker_image"] in valid_docker_images)
    # filtered_ds = [ds_entry for ds_entry in ds_selected if ds_entry["docker_image"] in valid_docker_images]

    
    logger.info(f"Filtered dataset size: {len(filtered_ds)} out of {len(ds_selected)}")
    return filtered_ds


##############################################################################
# editagent Functions
##############################################################################
def run_agent_with_restarts(
    agent,
    env,
    max_steps=40,
    num_restarts=1,
    temperature=0.0,
    max_steps_absolute=50,
    use_fn_calling: bool = True,
    max_iterations: int = 1,
    scaffold: str = "r2egym",
    max_tokens: int = 65536,
):
    """
    Iterative eval protocol:
    - normally run the agent
    - run for maximum num_iterations = 3 times
    - stop if trajectory.exit_reason == "agent"
    - otherwise continue iteratively till maximum iterations
    - finally choose the trajectory with the lowest number of steps
    - note restarts and iterative_evals are different (so just use one of them | add an assert flag)
    - also if original is at temp = 0, then we do next with 0.1 and 0.2 and so on (max 0.2)
    """
    steps_per_agent = max_steps // num_restarts
    logger.warning(f"running {steps_per_agent} steps per agent")

    # only one of restarts > 1 and iterative_eval can be True
    iterative_eval = max_iterations > 1
    assert not (num_restarts > 1 and iterative_eval), "only one of restarts > 1 and iterative_eval can be True"
    logger.warning(f"Using iterations: {max_iterations}, using iterative protocol: {iterative_eval}")

    # if original is at temp = 0, then we do next with 0.1 and 0.2 and so on (max 0.2)
    # if temperature is 0, create list of increasing temperatures up to 0.2
    if temperature == 0:
        temperatures = [0.0 + 0.1 * i for i in range(max_iterations)]
        temperatures = [min(t, 0.2) for t in temperatures]  # cap at 0.2
    else:
        temperatures = [temperature] * max_iterations
    logger.warning(f"Using temperatures: {temperatures}")

    # run the agent in iterative protocol
    trajectories = []
    for iteration in range(max_iterations):
        for idx in range(num_restarts):
            logger.warning(f"running agent at idx: {idx+1}")
            trajectory = agent.run(
                env,
                max_steps=steps_per_agent,
                temperature=temperatures[iteration],
                max_steps_absolute=max_steps_absolute,
                use_fn_calling=use_fn_calling,
                scaffold=scaffold,
                max_token_limit=max_tokens,
            )
            # remove reproduce.py
            # env.runtime.run('rm reproduce_issue.py')
        if trajectory.exit_reason == "agent":
            logger.warning(f"agent self-finished at iteration: {iteration}")
            return trajectory
        # otherwise continue iteratively
        trajectories.append(trajectory)
        # reset the env
        # env.reset()

    # choose the trajectory with the lowest number of steps
    trajectory = min(trajectories, key=lambda x: x.num_steps)
    return trajectory

def runagent(
    ds,
    exp_name: Optional[str] = None,
    max_steps=40,
    num_restarts=1,
    max_steps_absolute=50,
    llm_name="gpt-4o",
    temperature=0,
    use_fn_calling: bool = True,
    backend: str = "kubernetes", # "kubernetes" or "docker"
    max_reward_calc_time: int = 300,
    max_iterations: int = 1,
    scaffold: str = "r2egym",
    max_tokens: int = 65536,
) -> Optional[str]:
    """
    Runs the editagent agent on a specified Docker image.

    Args:
        docker_image: The Docker image to use for the environment.
        traj_dir: Directory to save trajectories.
        jsonl_file: Path to the JSONL file to save results. If not provided, generated using traj_dir and exp_name.
        exp_name: Experiment name. Used if jsonl_file is not provided. If not provided, a unique name is generated.
    """

    max_steps_absolute = max_steps_absolute
    max_steps = max_steps
    logger = setup_logging(
        name=ds["instance_id"].replace("/", "_"),
        log_file=f"run_logs/{exp_name}/{ds['docker_image'].replace('/', '_')}.log",
        console=True,
        level=INFO,
    )
    logger.info(f"Starting editagent on Docker image: {ds['docker_image']}")
    logger.info(f"Using LLM: {llm_name}")
    logger.info(f"Max Steps: {max_steps}")

    assert scaffold in ["r2egym", "sweagent", "openhands","mopenhands","contexthands"], f"Scaffold is {scaffold}, must be one of [r2egym, sweagent, openhands]"
    # Generate a unique experiment name if not provided
    if exp_name is None:
        exp_name = datetime.now().strftime("%Y%m%d_%H%M%S")

    max_retries = MAX_TURN_RETRY
    retry_num = 0
    while retry_num < max_retries:
        # Initialize environment arguments
        env_args = EnvArgs(ds=ds)

        # Initialize the RepoEnv
        env = RepoEnv(env_args, logger=logger, backend=backend)
        logger.info("RepoEnv initialized")
        # set agent args
        if use_fn_calling:
            assert scaffold != "sweagent", "SWEagent scaffold does not support fn calling"
            agent_args = AgentArgs.from_yaml(
                Path(f"./src/r2egym/agenthub/config/{scaffold}/edit_fn_calling.yaml")
            )
        else:
            agent_args = AgentArgs.from_yaml(
                Path(f"./src/r2egym/agenthub/config/{scaffold}/edit_non_fn_calling.yaml")
            )
        agent_args.llm_name = llm_name

        print(f"{agent_args=}")
        # Initialize the agent
        agent = Agent(name="EditAgent", args=agent_args, logger=logger)

        # run agent editagent
        try:
            trajectory = run_agent_with_restarts(
                agent,
                env,
                max_steps=max_steps,
                num_restarts=num_restarts,
                temperature=temperature,
                max_steps_absolute=max_steps_absolute,
                use_fn_calling=use_fn_calling,
                max_iterations=max_iterations,
                scaffold=scaffold,
                max_tokens=max_tokens,
            )
        except Exception as e:
            import traceback
            
            # Get complete error stack trace
            error_traceback = traceback.format_exc()
            
            # Print detailed error information
            logger.error(
                f"Error during agent run for Docker image {ds['docker_image']}: {e}"
            )
            logger.error(f"Error type: {type(e).__name__}")
            logger.error(f"Error details: {str(e)}")
            logger.error(f"Full traceback:\n{error_traceback}")
            
            # Print dataset-related information for debugging
            logger.error(f"Dataset info - repo: {ds.get('repo', 'N/A')}, instance_id: {ds.get('instance_id', 'N/A')}")
            
            # Print environment-related information
            if hasattr(env, 'runtime') and hasattr(env.runtime, 'docker_image'):
                logger.error(f"Runtime docker image: {env.runtime.docker_image}")
            
            # Try to close environment and runtime
            try:
                env.close()
                logger.info("Environment closed successfully after error.")
            except Exception as close_error:
                logger.error(f"Error closing environment: {close_error}")
            
            # Force flush and close all logger handlers before returning
            for handler in logger.handlers[:]:
                try:
                    handler.flush()
                    handler.close()
                    logger.removeHandler(handler)
                except Exception as flush_error:
                    pass
            
            return None

        try:
            # also get the gt outputs
            if ds["tag"] == "swemul":
                reward = 1.0
                test_output = "Success"
                reward_calc_time = 0
            else:
                reward_calc_time = time.time()
                reward, test_output = env.runtime._calculate_reward(get_test_output=True, timeout=max_reward_calc_time)
                reward_calc_time = time.time() - reward_calc_time
            
            # Close the environment and runtime
            logger.info("edit Stopping and removing container/pod/sandbox...")
            env.close()
            if test_output == "" and retry_num < max_retries - 1:
                logger.error("Empty test output received, retrying...")
                retry_num += 1
                continue
            if trajectory.exit_reason != "agent" and retry_num < max_retries - 1:
                logger.error("After retries, reward is 0.0 and exit reason is not agent, retrying...")
                logger.info(f"trajectory.exit_reason: {trajectory.exit_reason}")
                retry_num += 1
                if retry_num == max_retries - 2:
                    max_steps_absolute = 500
                    max_steps = 500
                continue
            if test_output == "":
                logger.error("After 5 retries empty test output received")
        # except KeyboardInterrupt:
        #     logger.warning("Edit agent reward calculation interrupted by user (KeyboardInterrupt).")
        #     try:
        #         env.close()
        #         logger.info("Environment closed successfully after interruption.")
        #     except Exception as close_error:
        #         logger.error(f"Error closing environment after interruption: {close_error}")
        #     return None
        except Exception as e:
            logger.error(f"Error during reward calculation or environment closure: {e}")
            reward = 0.0
            env.close()

        # update the trajectory object
        trajectory.reward = reward
        trajectory.test_output = test_output
        trajectory.ds = ds
        trajectory.exp_name = exp_name
        trajectory.reward_calc_time = reward_calc_time # time taken to calculate reward
        logger.warning(f"time taken to calculate reward in seconds: {reward_calc_time:.2f}")
        logger.warning(f"reward: {reward}")

        logger.info(f"editagent completed for Docker image: {ds['docker_image']}")
        # close env and docker runtime
        logger.info(f"Closing environment for Docker image: {ds['docker_image']}")

        try:
            result = trajectory.model_dump_json()
            logger.info(f"runagent completed successfully for {ds['docker_image']}, result length: {len(result)}")
            
            # Force flush and close all logger handlers before returning
            logger.info(f"About to return result for {ds['docker_image']}")
            for handler in logger.handlers[:]:  # Use slice to avoid modification during iteration
                try:
                    handler.flush()
                    handler.close()
                    logger.removeHandler(handler)
                except Exception as flush_error:
                    pass  # Ignore flush errors
            
            return result
        except Exception as e:
            import traceback
            error_traceback = traceback.format_exc()
            logger.error(f"Failed to serialize trajectory for {ds['docker_image']}: {e}")
            logger.error(f"Serialization error traceback:\n{error_traceback}")
            
            # Force flush and close on error too
            for handler in logger.handlers[:]:
                try:
                    handler.flush()
                    handler.close()
                    logger.removeHandler(handler)
                except Exception as flush_error:
                    pass
            
            return None


def runagent_multiple(
    dataset: str,
    split: str,
    k: int = 1,
    traj_dir: str = "./traj",
    exp_name: Optional[str] = None,
    start_idx=0,
    max_steps=40,
    num_restarts=1,
    max_steps_absolute=50,
    max_workers: Optional[int] = None,
    llm_name="gpt-4o",
    use_existing: bool = True,
    skip_existing: bool = False,
    temperature: float = 0,
    use_fn_calling: bool = True,
    backend: str = "kubernetes", # "kubernetes" or "docker"
    max_reward_calc_time: int = 300,
    max_iterations: int = 1,
    scaffold: str = "r2egym",
    prepull_images: bool = False,
    max_tokens: int = 65536,
    docker_image_file: Optional[str] = None,
):
    """
    Runs the editagent agent on the first k Docker images.

    Args:
        k: The number of Docker images to process.
        traj_dir: Directory to save trajectories.
        exp_name: Experiment name for the JSONL file. If not provided, a unique name is generated.
        start_idx: The starting index in the Docker images list.
        max_steps: Maximum steps for the agent run.
        max_workers: Maximum number of threads to use.
        prepull_images: Whether to prepull Docker images in parallel before starting execution.
    """
    # Load the dataset
    ds = load_dataset(dataset, split=split)
    # Unify field names: if dataset contains Smith, copy image_name field to docker_image
    if "smith" in dataset:
        ds = ds.map(lambda x: {"docker_image": x["image_name"]})
        ds = ds.map(lambda x: {"tag": "smith"})
    elif "Verified" in dataset or "Lite" in dataset:
        ds = ds.map(lambda x: {"tag": "default"})
    elif "Live" in dataset:
        print("Live dataset")
        ds = ds.map(lambda x: {"docker_image": f"starryzhang/sweb.eval.x86_64.{x['instance_id'].lower()}:latest"})
        # Replace "__" with "_1776_"
        ds = ds.map(lambda x: {"docker_image": x["docker_image"].replace("__", "_1776_")})
        # Replace repo with repo_name field
        # ds = ds.map(lambda x: {"repo_name": x["repo"]})
        ds = ds.map(lambda x: {"tag": "live"})
    elif "SWE-bench" in dataset:
        ds = ds.map(lambda x: {"docker_image": f"swebench/sweb.eval.x86_64.{x['instance_id'].lower()}"})
        # Replace "__" with "_1776_"
        ds = ds.map(lambda x: {"docker_image": x["docker_image"].replace("__", "_1776_")})
        ds = ds.map(lambda x: {"repo_name": x["repo"]})
        # Add a new tag column with value swemul
        if "Multilingual" in dataset:
            ds = ds.map(lambda x: {"tag": "swemul"})
        else:
            ds = ds.map(lambda x: {"tag": "default"})
    elif "SWE-rebench" in dataset:
        ds = ds.map(lambda x: {"repo_name": x["repo"]})
        # Add a new tag column with value swerebench
        ds = ds.map(lambda x: {"tag": "swerebench"})
    else:
        ds = ds.map(lambda x: {"tag": "default"})
    
    # Uniformly prepend agi-code-agent-cn-beijing.cr.volces.com/ to docker_image
    # if "SWE-rebench" not in dataset:
    #     ds = ds.map(lambda x: {"docker_image": f"agi-code-agent-cn-beijing.cr.volces.com/{x['docker_image']}"})

    # If x["ds"]["instance_id"] does not exist, use x["ds"]["docker_image"] field instead of x["ds"]["instance_id"]
    def fill_instance_id(example):
        if "instance_id" not in example or example["instance_id"] is None or example["instance_id"] == "":
            example["instance_id"] = example.get("docker_image", "")
        return example

    ds = ds.map(fill_instance_id)

    # If x["ds"]["instance_id"] does not exist, use x["ds"]["docker_image"] field instead of x["ds"]["instance_id"]
    def fill_instance_id(example):
        if "instance_id" not in example or example["instance_id"] is None or example["instance_id"] == "":
            example["instance_id"] = example.get("docker_image", "")
        return example

    ds = ds.map(fill_instance_id)



    logger.info(f"{len(ds)}, {k}, {start_idx}")
    # shuffle the dataset
    ds = ds.shuffle(seed=42)

    # get selected idxs
    selected_idx = range(start_idx, start_idx + k)
    # ds_selected = [ds[i] for i in selected_idx] Keep as dataset format
    ds_selected = ds.select(selected_idx)

    # print ds_selected stats
    logger.info(
        f"Dataset: {dataset}, Split: {split}, Num_total: {len(ds)}, Start Index: {start_idx}, k: {k}"
    )
    logger.info(f"Starting editagent on {len(ds_selected)} Docker images.")

    # Filter out data where problem_statement is None or empty string
    ds_selected = ds_selected.filter(lambda x: x["problem_statement"] is not None and x["problem_statement"] != "")


    # Generate a unique experiment name if not provided
    if exp_name is None:
        exp_name = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Ensure traj_dir exists
    traj_dir_path = Path(traj_dir)
    traj_dir_path.mkdir(parents=True, exist_ok=True)

    # Generate a filename for the JSONL file
    jsonl_file = traj_dir_path / f"{exp_name}.jsonl"

    if use_existing:
        if jsonl_file.exists():
            with open(jsonl_file) as f:
                existing_dockers = []
                # for line in f.readlines():
                #     try:
                #         existing_dockers.append(
                #             Trajectory.load_from_model_dump_json(line).ds[
                #                 "docker_image"
                #             ]
                #         )
                #     except:
                #         print("error in jsonl file")
                existing_dockers = []
                for loadline in safe_read_jsonl(jsonl_file):
                    if "ds" in loadline:
                        if "instance_id" in loadline["ds"]:
                            existing_dockers.append(loadline["ds"]["instance_id"])
                        # elif "docker_image" in loadline["ds"]:
                        #     existing_dockers.append(loadline["ds"]["docker_image"])
            print(f"existing_dockers: {len(existing_dockers)}")
            ds_selected = [
                ds_entry
                for ds_entry in ds_selected
                if ds_entry["instance_id"] not in existing_dockers
            ]

    if skip_existing:
        # old_jsonl_files_glob = f"{exp_name[:-1]}*"
        # for old_jsonl_file in traj_dir_path.glob(old_jsonl_files_glob):
        #     print(f"old_jsonl_file: {old_jsonl_file}")
        #     existing_dockers = []
        #     for loadline in safe_read_jsonl(old_jsonl_file):
        #         if ("ds" in loadline and 
        #             "instance_id" in loadline["ds"] and
        #             loadline.get("reward") == 1):
        #             existing_dockers.append(loadline["ds"]["instance_id"])
        #     print(f"existing_dockers: {len(existing_dockers)}")
        #     ds_selected = [
        #         ds_entry
        #         for ds_entry in ds_selected
        #         if ds_entry["instance_id"] not in existing_dockers
        #     ]

        # Get pass_instance_filepath from environment variable
        pass_instance_filepath = os.getenv("PASS_INSTANCE_FILEPATH")
        # Read JSON files in the folder
        import glob
        json_files = glob.glob(os.path.join(pass_instance_filepath, "*.json"))
        instance_ids = []
        for json_file in json_files:
            with open(json_file, "r") as f:
                items = json.load(f)
                instance_ids.extend([item["instance_id"] for item in items])
                print(f"existing num is {len(instance_ids)}")
        ds_selected = [
            ds_entry
            for ds_entry in ds_selected
            if ds_entry["instance_id"] not in instance_ids
        ]

    # Filter out data where docker_image is not in the list
    logger.info(f"docker_image_file: {docker_image_file}")
    # Only apply filtering when a non-empty string path is provided
    if isinstance(docker_image_file, str) and docker_image_file.strip():
        ds_selected = fetch_repos_from_docker_images(ds_selected, docker_image_file)
    logger.info(f"Dataset after filtering invalid docker images: {len(ds_selected)}")
    logger.info(
        f"Starting editagent on {len(ds_selected)} Docker images after filtering."
    )

    # Prepull all Docker images in parallel before starting main execution
    if len(ds_selected) > 0 and prepull_images:
        logger.info("Prepulling Docker images before starting main execution...")
        prepull_docker_images(ds_selected, max_workers=max_workers)
        logger.info("Docker image prepull completed.")

    # Check if original run_logs/exp_name/ directory exists, delete if it does
    run_logs_exp_dir = Path(f"run_logs/{exp_name}")
    if run_logs_exp_dir.exists() and run_logs_exp_dir.is_dir():
        import shutil
        shutil.rmtree(run_logs_exp_dir)
        logger.info(f"Deleted existing log directory: {run_logs_exp_dir}")

    # with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
    # with concurrent.futures.ProcessPoolExecutor(max_workers=max_workers) as executor:
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        # Submit all tasks to the executor using keyword arguments
        future_to_image = {
            executor.submit(
                runagent,
                ds=ds_entry,
                exp_name=exp_name,
                max_steps=max_steps,
                num_restarts=num_restarts,
                max_steps_absolute=max_steps_absolute,
                llm_name=llm_name,
                temperature=temperature,
                use_fn_calling=use_fn_calling,
                backend=backend,
                max_reward_calc_time=max_reward_calc_time,
                max_iterations=max_iterations,
                scaffold=scaffold,
                max_tokens=max_tokens,
            ): ds_entry[
                "docker_image"
            ]  # <-- store the docker_image from ds_entry here
            for ds_entry in ds_selected
        }

        with open(jsonl_file, "a") as f:
            completed = 0
            total = len(future_to_image)
            logger.info(f"[MAIN] Waiting for {total} tasks to complete...")
            
            for future in concurrent.futures.as_completed(future_to_image):
                docker_image = future_to_image[
                    future
                ]  # <-- retrieve that stored docker_image
                
                completed += 1
                logger.info(f"[MAIN] Task {completed}/{total} completed for {docker_image}")
                logger.info(f"[MAIN] Calling future.result() to retrieve data...")
                
                try:
                    result = future.result(timeout=120)  # Add 30s timeout to detect hanging
                    logger.info(f"[MAIN] future.result() returned for {docker_image}, type={type(result)}, is_none={result is None}")
                    
                    if result is not None:
                        logger.info(f"[MAIN] Retrieved non-None result (length {len(result)}) for {docker_image}. Writing to file.")
                        with file_lock:
                            f.write(result + "\n")
                            f.flush()  # Usually auto-flushes, but can explicitly call to ensure immediate write
                        logger.info(f"[MAIN] Successfully wrote {len(result)} bytes for {docker_image}")
                    else:
                        logger.warning(f"[MAIN] Received None result for {docker_image}. Check subprocess logs for details.")
                except concurrent.futures.TimeoutError:
                    logger.error(f"[MAIN] TIMEOUT waiting for result from {docker_image} after 30 seconds!")
                    logger.error(f"[MAIN] This suggests the subprocess is hanging during return or cleanup.")
                except Exception as e:
                    # Use docker_image from above when logging
                    logger.error(f"[MAIN] Exception for Docker image {docker_image}: {e}")
                    import traceback
                    full_traceback = traceback.format_exc()
                    
                    # Log detailed exception information
                    # logger.error(f"Exception for Docker image {docker_image}: {e}")
                    logger.error(f"[MAIN] Full traceback for {docker_image}:")
                    logger.error(full_traceback)
            
            logger.info(f"[MAIN] All {completed} tasks processed.")

    logger.info(f"editagent completed on {len(ds_selected)} Docker images.")
    
    # High concurrency optimization: output client pool statistics and cleanup
    try:
        pool_stats = get_docker_pool_stats()
        logger.info(f"Docker client pool stats: {pool_stats}")
        cleanup_docker_client_pool()
        logger.info("Docker client pool cleaned up")
    except Exception as e:
        logger.warning(f"Error during client pool cleanup: {e}")


if __name__ == "__main__":
    # Expose functions via Fire
    Fire(
        {
            "runagent": runagent,
            "runagent_multiple": runagent_multiple,
        }
    )
