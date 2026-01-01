import os, sys
import json
from time import sleep
import time
import uuid
import tempfile
import docker
from docker.models.containers import Container
from swesmith.profiles import registry
import threading
from functools import lru_cache

import volcenginesdkcore
import volcenginesdkvefaas
from volcenginesdkcore.rest import ApiException
from volcenginesdkvefaas import (
    VEFAASApi,
    InstanceImageInfoForCreateSandboxInput,
    CreateSandboxRequest, 
    CreateSandboxResponse,
    KillSandboxRequest,
    )

# Kubernetes proxy configuration
os.environ.setdefault('HTTP_PROXY', '')
os.environ.setdefault('HTTPS_PROXY', '')
os.environ.setdefault('NO_PROXY', 'localhost,127.0.0.1,kubernetes.default.svc')

from r2egym.repo_analysis.execution_log_parser import parse_log_fn, decolor_dict_keys
from r2egym.agenthub.runtime.base import (
    ExecutionEnvironment,
)
import base64
import subprocess
import datetime
import hashlib
import shutil
import uuid

import docker
import kubernetes
import tarfile
import io
import os
from r2egym.agenthub.utils.log import get_logger
import re
from r2egym.agenthub.utils.utils import match_dockerimage_to_repo
from r2egym.agenthub import SUPPORTED_REPOS, SKIP_FILES, SKIP_FILES_NEW, CMD_TIMEOUT
import concurrent.futures

from r2egym.agenthub.trajectory.swebench_utils import (
    make_test_spec,
    swebench_parse,
    TestSpec,
)
from r2egym.agenthub.utils.utils import get_logger
from r2egym.commit_models.diff_classes import ParsedCommit
from r2egym.swesmith.utils import get_test_command
from typing import Callable, TypeVar, Optional, Sequence, Type, Tuple

from kubernetes import client, config, watch

# For Kubernetes exec.
from kubernetes.stream import stream


T = TypeVar("T")
DEFAULT_NAMESPACE = "default"
DOCKER_PATH = "/root/.venv/bin:/root/.local/bin:/root/.cargo/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin"


TOTAL_NETWORKS = 30
NETWORK_NAMES = [f"perf-net-{i}" for i in range(TOTAL_NETWORKS)]

# Global Docker client pool for high concurrency scenarios
_docker_clients_lock = threading.Lock()
_docker_clients = {}

class DockerClientPool:
    """
    Docker client pool for connection reuse in high concurrency scenarios
    Supports grouping clients by docker_host
    """
    
    def __init__(self, max_clients_per_host=10):
        self.max_clients_per_host = max_clients_per_host
        self.clients = {}
        self.client_usage = {}
        self.lock = threading.Lock()
    
    def get_client(self, docker_host=None):
        """Get Docker client with connection reuse support"""
        host_key = docker_host or "local"
        
        with self.lock:
            if host_key not in self.clients:
                self.clients[host_key] = []
                self.client_usage[host_key] = []
            
            # Find available client (one with least usage)
            if self.clients[host_key]:
                min_usage_idx = min(range(len(self.client_usage[host_key])), 
                                  key=lambda i: self.client_usage[host_key][i])
                self.client_usage[host_key][min_usage_idx] += 1
                return self.clients[host_key][min_usage_idx]
            
            # If no available client and not at limit, create new client
            if len(self.clients[host_key]) < self.max_clients_per_host:
                try:
                    if docker_host:
                        client = docker.DockerClient(base_url=docker_host, timeout=600)
                    else:
                        client = docker.from_env(timeout=600)
                    
                    self.clients[host_key].append(client)
                    self.client_usage[host_key].append(1)
                    return client
                except Exception as e:
                    # If creation fails, return None for caller to handle
                    return None
            
            # At limit, return client with least usage
            min_usage_idx = min(range(len(self.client_usage[host_key])), 
                              key=lambda i: self.client_usage[host_key][i])
            self.client_usage[host_key][min_usage_idx] += 1
            return self.clients[host_key][min_usage_idx]
    
    def release_client(self, client, docker_host=None):
        """Release client (decrease usage count)"""
        host_key = docker_host or "local"
        
        with self.lock:
            if host_key in self.clients:
                try:
                    client_idx = self.clients[host_key].index(client)
                    self.client_usage[host_key][client_idx] = max(0, 
                        self.client_usage[host_key][client_idx] - 1)
                except (ValueError, IndexError):
                    pass  # Client not in pool, ignore
    
    def cleanup(self):
        """Clean up all client connections"""
        with self.lock:
            for host_clients in self.clients.values():
                for client in host_clients:
                    try:
                        client.close()
                    except:
                        pass
            self.clients.clear()
            self.client_usage.clear()

# Global client pool instance
_global_docker_pool = DockerClientPool(max_clients_per_host=10)

def cleanup_docker_client_pool():
    """Clean up global Docker client pool"""
    _global_docker_pool.cleanup()

def get_docker_pool_stats():
    """Get Docker client pool statistics"""
    with _global_docker_pool.lock:
        stats = {}
        for host, clients in _global_docker_pool.clients.items():
            stats[host] = {
                'client_count': len(clients),
                'usage_counts': _global_docker_pool.client_usage.get(host, []).copy()
            }
        return stats

from swebench.harness.constants import (
    APPLY_PATCH_FAIL,
    END_TEST_OUTPUT,
    FAIL_TO_FAIL,
    FAIL_TO_PASS,
    KEY_INSTANCE_ID,
    KEY_PREDICTION,
    MAP_REPO_VERSION_TO_SPECS,
    PASS_TO_FAIL,
    PASS_TO_PASS,
    RESET_FAILED,
    START_TEST_OUTPUT,
    TESTS_ERROR,
    TESTS_TIMEOUT,
    EvalType,
    ResolvedStatus,
    TestStatus,
)
from swebench.harness.test_spec.test_spec import TestSpec
from swebench.harness.log_parsers import MAP_REPO_TO_PARSER, get_eval_type
from swebench.harness.grading import get_eval_tests_report, get_resolution_status
from swesmith.harness.grading import  get_eval_tests_report as swesmith_get_eval_tests_report
from swesmith.harness.grading import get_resolution_status as swesmith_get_resolution_status


##############################################################################
# Docker runtime
##############################################################################
class DockerRuntime(ExecutionEnvironment):
    """
    docker runtime is responsible for the interacting with the docker environment.
    In particular, it should allow for accomodating the features of the particualr docker envs used for r2e-edits
    - collect files
    - list files excluding test files etc
    """

    def __init__(
        self,
        ds,  # dataset entry: defaulting to this (required for all dockers moving forward)
        repo_path: str = "/testbed",  # main repo path
        alt_path: str = "/root",  # used for keeping useful scripts to be hidden from the agent
        docker_image: str = None,  # docker image to use (if not provided, will be inferred from ds)
        command: str = "/bin/bash",
        logger=None,
        backend="docker",
        **docker_kwargs,
    ):
        # check if ds is provided (required for all dockers moving forward)
        assert ds, f"Dataset not provided for docker image: {docker_image}"
        assert backend in ["docker", "kubernetes", "sandbox"], f"Invalid backend: {backend}"
        self.docker_host = os.getenv("DOCKER_HOST")
        # swebench specific setup
        self.ds = ds
        self.backend = backend
        ds_image = None
        if "docker_image" in self.ds:
            ds_image = self.ds["docker_image"]
        elif "image_name" in self.ds:
            ds_image = self.ds["image_name"]
        else:
            raise ValueError(f"No docker image found in ds: {self.ds}")
        self.docker_image = ds_image if not docker_image else docker_image
        self.swemul = self.ds["tag"] == "swemul"
        self.swebench_verified = "swebench" in self.docker_image and self.ds["tag"] != "swemul"
        self.swesmith = "swesmith" in self.docker_image
        self.live = "starryzhang" in self.docker_image
        self.swerebench = self.ds["tag"] == "swerebench"


        # Define container connection retry parameter mechanism to prevent delays from overwhelming the machine
        self.start_retries = int(docker_kwargs.pop("start_retries", 5))
        self.start_retry_backoff = float(docker_kwargs.pop("start_retry_backoff", 2.0))
        self.attach_retry_wait = float(docker_kwargs.pop("start_retry_attach_wait", 1.0))
        self.operation_retry_attempts = max(1, int(docker_kwargs.pop("operation_retry_attempts", 3)))
        self.operation_retry_initial_delay = max(0.1, float(docker_kwargs.pop("operation_retry_initial_delay", 1.0)))
        self.operation_retry_backoff = max(1.0, float(docker_kwargs.pop("operation_retry_backoff", 2.0)))
        self.operation_retry_max_delay = max(
            self.operation_retry_initial_delay,
            float(docker_kwargs.pop("operation_retry_max_delay", 30.0)),
        )
        self.operation_retry_timeout = max(1.0, float(docker_kwargs.pop("operation_retry_timeout", 120.0)))
        self.exec_timeout_buffer = max(0.0, float(docker_kwargs.pop("exec_timeout_buffer", 5.0)))

        # Initialize logger (must be called before _execute_with_retry, as it uses self.logger)
        if logger is None:
            if backend == "docker":
                logger_name = "DockerRuntime"
            elif backend == "kubernetes":
                logger_name = "KubernetesRuntime"
            elif backend == "sandbox":
                logger_name = "SandboxRuntime"
            else:
                raise ValueError(f"Invalid backend: {backend}")
            self.logger = get_logger(logger_name)  # Pass the module name for clarity
        else:
            self.logger = logger
        
        if self.swesmith:
            # print("this is swesmith")
            # image_name = self.ds['image_name'].replace('__', '_1776_')
            self.swebench_verified = False
            # self.docker_image = f'jyangballin/{image_name}:latest'
            # self.docker_image = self.ds['image_name']
            # self.docker_image = self.ds['image_name']
            print(f"this is swesmith: {self.docker_image}")
            self.rp = registry.get_from_inst(self.ds)
        
        if self.swebench_verified:
            # also create a test spec for swebench verified dockers (useful for grading)
            # self.test_spec = make_test_spec(self.ds)
            self.test_spec = self._execute_with_retry(
                lambda: make_test_spec(self.ds),
                op_name="Create Test Spec"
                )
        
        if self.swerebench:
            from swerebench.harness.test_spec.test_spec import make_test_spec as make_test_spec_swerebench
            self.test_spec = self._execute_with_retry(
                lambda: make_test_spec_swerebench(self.ds),
                op_name="Create Test Spec"
                )

        if self.live:
            from swebench_live.harness.test_spec.test_spec import make_test_spec as make_test_spec_live
            self.test_spec = make_test_spec_live(self.ds)
        print("test_spec: ")
        # set runtime params
        self.repo_path = repo_path
        self.alt_path = alt_path
        self.command = command
        self.repo_name = (
            self.ds["repo"] if self.swebench_verified or self.swesmith or self.live else self.ds["repo_name"]
        )
        # if not self.swesmith and not self.live:
        #     self.commit_json = (
        #         self.ds["parsed_commit"]
        #         if self.swebench_verified
        #         else self.ds["parsed_commit_content"]
        #     )
        #     self.commit = ParsedCommit(**json.loads(self.commit_json))
        self.docker_kwargs = docker_kwargs

        if self.backend == "docker":
            # High concurrency optimization: use global client pool
            use_client_pool = os.getenv("USE_DOCKER_CLIENT_POOL", "true").lower() == "true"
            
            if use_client_pool:
                self.client = _global_docker_pool.get_client(self.docker_host)
                self._uses_pool = True
                
                if self.client is None:
                    # Pool acquisition failed, fall back to independent client
                    self.logger.warning("Failed to get client from pool, falling back to independent client")
                    if self.docker_host:
                        self.client = docker.DockerClient(base_url=self.docker_host, timeout=300)
                    else:
                        self.client = docker.from_env(timeout=300)
                    self._uses_pool = False
            else:
                # Traditional way: independent client
                if self.docker_host:
                    self.client = docker.DockerClient(base_url=self.docker_host, timeout=300)
                else:
                    self.client = docker.from_env(timeout=300)
                self._uses_pool = False
        elif self.backend == "kubernetes":
            # Try in-cluster config first, fallback to kubeconfig
            try:
                config.load_incluster_config()
            except Exception:
                try:
                    config.load_kube_config()
                except Exception as e:
                    self.logger.warning(f"Failed to load Kubernetes config: {e}. Falling back to Docker backend.")
                    self.backend = "docker"
                    self.client = docker.from_env(timeout=300)
                    return
            
            try:
                self.client = client.CoreV1Api()
                # Test the connection with a simple API call
                self.client.list_namespaced_pod(namespace=DEFAULT_NAMESPACE, limit=1)
            except Exception as e:
                self.logger.warning(f"Failed to connect to Kubernetes API: {e}. Falling back to Docker backend.")
                self.backend = "docker"
                self.client = docker.from_env(timeout=300)
                return
        elif self.backend == "sandbox":
            # Configure backend
            configuration = volcenginesdkcore.Configuration()
            configuration.ak = os.getenv("VE_FAAS_AK")
            configuration.sk = os.getenv("VE_FAAS_SK")
            configuration.region = "xxx-xxx"
            configuration.client_side_validation = True
            volcenginesdkcore.Configuration.set_default(configuration)

            self.client = volcenginesdkvefaas.VEFAASApi(volcenginesdkcore.ApiClient(configuration))
            self.function_id = os.getenv("VE_FAAS_FUNCTION_ID")
            self.sandbox_network = os.getenv("VE_FAAS_SANDBOX_NETWORK")

        
        # Start the container
        self.container = None
        self.container_name = self._get_container_name(self.docker_image)

        if self.backend == "kubernetes":
            # Generate a random UUID and truncate to 30 characters
            self.container_name = str(uuid.uuid4())
        elif self.backend == "sandbox":
            self.container_name = None
        self.start_container(
            self.docker_image, command, self.container_name, **docker_kwargs
        )

        # Initialize the environment
        self.setup_env()
        if self.backend == "kubernetes":
            self.logger.info("Kubernetes environment initialized")
        elif self.backend == "sandbox":
            self.logger.info("Sandbox environment initialized")
        else:
            self.logger.info("Docker environment initialized")
        self.logger.info("repo name: %s", self.repo_name)
        self.logger.info("Docker image: %s", self.docker_image)
        if self.backend == "docker":
            self.logger.info("Container ID: %s", self.container.id)
        elif self.backend == "kubernetes":
            # Assuming self.container is a V1Pod object after creation/retrieval
            pod_name = (
                self.container.metadata.name
                if self.container and self.container.metadata
                else "N/A"
            )
            self.logger.info("Pod Name: %s", pod_name)
        elif self.backend == "sandbox":
            self.logger.info("Sandbox Name: %s", self.container_name)

    @staticmethod
    def _get_container_name(image_name: str) -> str:
        """Return name of container"""
        process_id = str(os.getpid())
        current_time = str(datetime.datetime.now())
        unique_string = current_time + process_id
        hash_object = hashlib.sha256(unique_string.encode())
        image_name_sanitized = image_name.replace("/", "-")
        image_name_sanitized = image_name_sanitized.replace(":", "-")
        return f"{image_name_sanitized}-{hash_object.hexdigest()[:10]}"

    def _execute_with_retry(
        self,
        func: Callable[[], T],
        *,
        op_name: str,
        max_attempts: Optional[int] = None,
        timeout: Optional[float] = None,
        retry_exceptions: Optional[Sequence[type[BaseException]]] = None,
        initial_delay: Optional[float] = None,
        backoff: Optional[float] = None,
        max_delay: Optional[float] = None,
    ) -> T:
        attempts = max_attempts or self.operation_retry_attempts
        delay = initial_delay if initial_delay is not None else self.operation_retry_initial_delay
        backoff_factor = backoff if backoff is not None else self.operation_retry_backoff
        max_sleep = max_delay if max_delay is not None else self.operation_retry_max_delay
        op_timeout = timeout if timeout is not None else self.operation_retry_timeout
        exceptions: Tuple[type[BaseException], ...]
        if retry_exceptions is None:
            exceptions = (Exception,)
        else:
            exceptions = tuple(retry_exceptions)

        last_exc: Optional[BaseException] = None
        for attempt in range(1, attempts + 1):
            try:
                if op_timeout is not None:
                    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
                        future = executor.submit(func)
                        return future.result(timeout=op_timeout)
                return func()
            except exceptions as exc:  # type: ignore[misc]
                if isinstance(exc, (KeyboardInterrupt, SystemExit)):
                    raise
                last_exc = exc
                if attempt >= attempts:
                    raise
                self.logger.warning(
                    "%s failed (attempt %d/%d): %s",
                    op_name,
                    attempt,
                    attempts,
                    repr(exc),
                )
                time.sleep(delay)
                delay = min(delay * backoff_factor, max_sleep)
            except BaseException as exc:
                # Non-retryable exception, re-raise immediately
                raise

        assert last_exc is not None
        raise last_exc

    def _start_kubernetes_pod(
        self, docker_image: str, command: str, pod_name: str, **docker_kwargs
    ):
        """
        Starts or connects to a Kubernetes pod with the specified configuration.

        If a pod with the given name already exists, it attempts to connect to it.
        Otherwise, it creates a new pod based on the provided image, command,
        and environment variables, then waits for it to reach the 'Running' state.

        Args:
            docker_image: The Docker image to use for the pod's container.
            command: The command to run inside the container.
            pod_name: The desired name for the Kubernetes pod.
            **docker_kwargs: Additional keyword arguments. Currently used to extract
                             'environment' variables for the pod spec.

        Raises:
            kubernetes.client.ApiException: If there's an error interacting with the
                                           Kubernetes API (other than 404 Not Found
                                           when checking existence).
            RuntimeError: If the pod fails to reach the 'Running' state after creation.
        """
        not_found_error = None
        try:
            # Check if the pod already exists
            self.container = self.client.read_namespaced_pod(
                name=pod_name, namespace=DEFAULT_NAMESPACE, _request_timeout=60,
            )
            self.logger.info(f"Found existing Kubernetes pod: {pod_name}")
            return
        except client.ApiException as e:
            not_found_error = e

        if not_found_error.status != 404:
            self.logger.error(
                f"Error checking Kubernetes pod '{pod_name}' status: {not_found_error}. Check Kubernetes configuration and permissions."
            )
            raise not_found_error

        env_vars = {"PATH": DOCKER_PATH, **docker_kwargs.get("environment", {})}
        env_spec = [{"name": k, "value": str(v)} for k, v in env_vars.items()]
        pod_body = {
            "apiVersion": "v1",
            "kind": "Pod",
            "metadata": {"name": pod_name},
            "spec": {
                "restartPolicy": "Never",
                "containers": [
                    {
                        "name": pod_name,
                        "image": docker_image,
                        "command": ["/bin/sh", "-c"],
                        "args": [command] if isinstance(command, str) else command,
                        "stdin": True,
                        "tty": True,
                        "env": env_spec,
                        "resources": {
                            "requests": {"cpu": "1", "memory": "1Gi"},
                        },
                        "imagePullPolicy": "Never",
                    }
                ],
                "imagePullSecrets": [{"name": "dockerhub-pro"}],
                # "nodeSelector": {"karpenter.sh/nodepool": "bigcpu-standby"},
                "tolerations": [
                    {
                        "key": "node.kubernetes.io/disk-pressure",
                        "operator": "Exists",
                        "effect": "NoExecute",
                        "tolerationSeconds": 10800
                    }
                ],
            },
        }

        # Create the Pod with retry logic & efficiently monitor with K8 Watch
        max_retries = 5
        backoff = 5  # seconds
        pod = None
        for attempt in range(1, max_retries + 1):
            try:
                pod = self.client.create_namespaced_pod(
                    namespace=DEFAULT_NAMESPACE, body=pod_body, _request_timeout=300,
                )
                break  # success
            except client.ApiException as e:
                # Retry on API-server throttling or transient errors
                if e.status in (409, 429, 500, 503):
                    self.logger.warning(
                        f"Transient Kubernetes error {e.status} while creating pod "
                        f"'{pod_name}' (attempt {attempt}/{max_retries}); "
                        f"retrying in {backoff}s"
                    )
                    time.sleep(backoff)
                    backoff = min(backoff * 2, 60)
                    continue
                # Non-retryable error → propagate
                self.logger.error(f"Failed to create Kubernetes pod '{pod_name}': {e}")
                raise
        else:
            raise RuntimeError(
                f"Exceeded retry limit ({max_retries}) while creating pod '{pod_name}'."
            )

        try:
            rv = pod.metadata.resource_version
            w = watch.Watch()
            stream = w.stream(
                self.client.list_namespaced_pod,
                namespace=DEFAULT_NAMESPACE,
                field_selector=f"metadata.name={pod_name}",
                resource_version=rv,
                timeout_seconds=1200,  # 10 minutes timeout instead of 1 hour
            )
            start_time = time.time()
            for event in stream:
                obj = event["object"]
                phase = obj.status.phase
                if time.time() - start_time > 1200:
                    w.stop()
                    raise RuntimeError(f"Kubernetes pod '{pod_name}' timed out after 1200 seconds.")
                # self.logger.info(f"Event {event['type']} → pod.phase={phase}")
                if phase == "Running":
                    self.logger.info(f"Kubernetes pod '{pod_name}' is Running.")
                    w.stop()
                    break
                if phase in ["Failed", "Succeeded", "Unknown"]:
                    w.stop()
                    raise RuntimeError(
                        f"Kubernetes pod '{pod_name}' entered terminal phase '{phase}'."
                    )
            self.container = pod
        except client.ApiException as create_error:
            self.logger.error(
                f"Failed to create Kubernetes pod '{pod_name}': {create_error}"
            )
            raise create_error
        except Exception as e:
            # Handle watch timeout or other errors
            self.logger.error(f"Error waiting for pod to start: {e}")
            # Check pod status directly as fallback
            try:
                pod_status = self.client.read_namespaced_pod(
                    name=pod_name, namespace=DEFAULT_NAMESPACE, _request_timeout=60,
                )
                if pod_status.status.phase == "Running":
                    self.logger.info(f"Pod '{pod_name}' is running (verified after watch error)")
                    self.container = pod_status
                else:
                    self.logger.warning(f"Pod '{pod_name}' is in state {pod_status.status.phase}")
                    raise RuntimeError(f"Pod '{pod_name}' failed to reach Running state: {pod_status.status.phase}")
            except Exception as status_error:
                self.logger.error(f"Failed to check pod status after watch error: {status_error}")
                raise RuntimeError(f"Failed to verify pod status: {status_error}")
   

    def ListSandboxImages(self):
        api_instance = self.client
        list_sandbox_images_request = volcenginesdkvefaas.ListSandboxImagesRequest(
            image_type="private",
            page_size=1000,
        )
        
        try:
            # Copy code example, please print API return value yourself
            output = api_instance.list_sandbox_images(list_sandbox_images_request)
            # print(output)
            return output
        except ApiException as e:
            # Copy code example, please print API error information yourself
            # print("Exception when calling api: %s\n" % e)
            raise e

    def PrecacheSandboxImages(self,image_url_list: list[str]):
        # First get the list of existing images
        api_instance = self.client
        
        # Get list of precached images
        try:
            while True:
                existing_images_response = self.ListSandboxImages()
                
                # Extract list of existing image URLs
                existing_image_urls = set()
                cacheing_image_urls = set() # List of image URLs currently being precached
                if hasattr(existing_images_response, 'images') and existing_images_response.images:
                    for image in existing_images_response.images:
                        if hasattr(image, 'image_url') and image.image_url and image.precache_status == 'success':
                            existing_image_urls.add(image.image_url)
                        if hasattr(image, 'image_url') and image.image_url and image.precache_status == 'caching':
                            cacheing_image_urls.add(image.image_url)

                self.logger.info(f"Number of existing images: {len(existing_image_urls)}")
                self.logger.info(f"Number of images being precached: {len(cacheing_image_urls)}")
                # self.logger.info(f"Existing image URLs: {list(existing_image_urls)}")

                # Filter out already existing images
                images_to_precache = [url for url in image_url_list if url not in existing_image_urls]

                if not images_to_precache:
                    self.logger.info("All images are already precached, no need to precache again")
                    return
                
                # Filter out already existing/currently precaching images
                images_to_precache = [url for url in images_to_precache if url not in cacheing_image_urls]
                if not images_to_precache:
                    self.logger.info("All images are already precached or being precached, no need to precache again")
                    continue

                self.logger.info(f"Images to precache: {images_to_precache}")
                # self.logger.info(f"Image URLs to precache: {images_to_precache}")
                
                # Precache filtered images
                precache_sandbox_images_request = volcenginesdkvefaas.PrecacheSandboxImagesRequest(
                    image_urls=images_to_precache,
                )
                api_instance.precache_sandbox_images(precache_sandbox_images_request)
                self.logger.info("PrecacheSandboxImages succeeded!")
            
        except ApiException as e:
            self.logger.error(f"Exception when calling api: %s\n" % e)
            raise e

    def _start_sandbox(self, docker_image: str):

        # Precache image    
        # self.PrecacheSandboxImages([docker_image])
        """Create Sandbox, specify instance TOS Bucket remote directory and image"""
        try:
            tos_mount_point = volcenginesdkvefaas.TosMountPointForCreateSandboxInput(
                # bucket_path="/agi-data/users/yzli/liusk-workspace/swe_sandbox/test",
                bucket_path="/sandbox_tmp",
                local_mount_path="/mnt/tos",
            )
            instance_tos_mount_config = volcenginesdkvefaas.InstanceTosMountConfigForCreateSandboxInput(
                enable=True,
                tos_mount_points=[tos_mount_point]  # Must be a list
            )

            req_instance_image_info = volcenginesdkvefaas.InstanceImageInfoForCreateSandboxInput(
                command = "mkdir -p /sandbox_dir/sandbox_env && tar -xf /mnt/tos/sandbox_env.tar -C /sandbox_dir/sandbox_env && source /sandbox_dir/sandbox_env/bin/activate && tar -xf /mnt/tos/sandboxfusion.tar -C /sandbox_dir && mkdir -p /sandbox_dir/SandboxFusion/docs/build && bash /sandbox_dir/SandboxFusion/scripts/run.sh",
                image=docker_image,
                port=8080,
            )

            create_sandbox_request = volcenginesdkvefaas.CreateSandboxRequest(
                # Sandbox application ID, obtained from: "vefaas frontend console - Cloud Sandbox - Sandbox Application - Basic Info - ID"
                function_id=self.function_id,
                instance_tos_mount_config=instance_tos_mount_config,
                instance_image_info = req_instance_image_info,
                cpu_milli = 2000,
                memory_mb = 4096,
                request_timeout=900,
                timeout=240,
            )
            response = self.client.create_sandbox(create_sandbox_request)
            self.container_name = response.sandbox_id
            self.logger.info(f"Sandbox created successfully! Response: {response}")
        except Exception as e:
            self.logger.error(f"[5] Failed to create with specified TOS Bucket Path: {e}")
            raise e

    # todo: add faas sandbox creation
    def start_container(
        self, docker_image: str, command: str, ctr_name: str, **docker_kwargs
    ):
        # Start or reuse a container
        try:
            if self.backend == "docker":
                containers = self.client.containers.list(
                    all=True, filters={"name": ctr_name}
                )
                if containers:
                    self.container = containers[0]
                    if self.container.status != "running":
                        self.container.start()
                else:
                    import random
                    # chosen_network = random.choice(NETWORK_NAMES)
                    self.container = self._execute_with_retry(
                        lambda: self.client.containers.run(
                            docker_image,
                            command,
                            name=ctr_name,
                            detach=True,
                            tty=True,
                            stdin_open=True,
                            # network=chosen_network,
                            **docker_kwargs,
                        ),
                        op_name="Create Container with Network",
                    )
                    # self.client.containers.run(
                    #     docker_image,
                    #     command,
                    #     name=ctr_name,
                    #     detach=True,
                    #     tty=True,
                    #     stdin_open=True,
                    #     network=chosen_network,
                    #     # environment={"PATH": "/commands"},
                    #     **docker_kwargs,
                    # )
            elif self.backend == "kubernetes":
                self._start_kubernetes_pod(
                    docker_image, command, ctr_name, **docker_kwargs
                )
            elif self.backend == "sandbox":
                # self._start_sandbox(
                #     docker_image
                # )
                import random
                import time
                # Random wait 1-5 seconds
                time.sleep(random.randint(1, 5))
                self._execute_with_retry(
                    lambda: self._start_sandbox(docker_image),
                    op_name="Create Sandbox"
                )
        except Exception as e:
            print("Container start error:", repr(e))
            self.stop_container()
            # return
            raise e

    def _stop_sandbox(self):
        self.logger.info(f"Stopping and removing sandbox {self.container_name}...")
        if self.container_name:
            response = self.client.kill_sandbox(
                KillSandboxRequest(
                    function_id=self.function_id,
                    sandbox_id=self.container_name
                )
            )
            self.logger.info(f"Kill Sandbox response: {response}")

    def _stop_kubernetes_pod(self):
        try:
            self.client.delete_namespaced_pod(
                name=self.container_name,
                namespace=DEFAULT_NAMESPACE,
                body=kubernetes.client.V1DeleteOptions(grace_period_seconds=0),
                _request_timeout=60,
            )

            w = watch.Watch()
            stream = w.stream(
                self.client.list_namespaced_pod,
                namespace=DEFAULT_NAMESPACE,
                field_selector=f"metadata.name={self.container_name}",
                timeout_seconds=60,  # 1 minute timeout instead of indefinite
            )

            deletion_confirmed = False
            for event in stream:
                if event["type"] == "DELETED":
                    self.logger.info(f"Kubernetes pod {self.container_name} deleted.")
                    deletion_confirmed = True
                    w.stop()
                    break
            
            # If watch times out without seeing deletion, verify pod is gone
            if not deletion_confirmed:
                try:
                    # Check if pod still exists
                    self.client.read_namespaced_pod(
                        name=self.container_name, namespace=DEFAULT_NAMESPACE
                    )
                    self.logger.warning(
                        f"Watch timed out but pod {self.container_name} still exists. Forcing deletion."
                    )
                    # Try deleting again with force
                    self.client.delete_namespaced_pod(
                        name=self.container_name,
                        namespace=DEFAULT_NAMESPACE,
                        body=kubernetes.client.V1DeleteOptions(
                            grace_period_seconds=0,
                            force=True
                        ),
                    )
                except kubernetes.client.rest.ApiException as e:
                    if e.status == 404:
                        # Pod is gone, which is what we want
                        self.logger.info(f"Confirmed pod {self.container_name} is deleted.")
                    else:
                        # Some other API error
                        self.logger.error(f"Error checking pod status after timeout: {e}")
        except kubernetes.client.rest.ApiException as e:
            if e.status == 404:
                # Pod already deleted, ignore
                self.logger.info(
                    f"Kubernetes pod '{self.container_name}' not found, likely already deleted."
                )
            else:
                # Log other K8s API errors during deletion
                self.logger.error(
                    f"Error deleting Kubernetes pod '{self.container_name}': {e}"
                )
                raise e  # Re-raise unexpected errors
    # todo: add faas sandbox kill
    def stop_container(self):
        self.logger.info(f"Stopping and removing container/pod/sandbox {self.container_name}...")
        try:
            if self.container:
                if self.backend == "docker":
                    self.container.stop()
                    self.container.remove()
                elif self.backend == "kubernetes":
                    self._stop_kubernetes_pod()
            if self.backend == "sandbox" and self.container_name:
                self._stop_sandbox()
        except Exception as e:
            print("Container stop/delete error:", repr(e))
    
    def reset_swesmith_tests(self):
        """
        Reorganize SWE-smith test flow:
        1. First stash save current AI fix code
        2. Switch to parent commit (bug state)
        3. Apply AI fix code from stash
        4. Restore changed test files
        """
        # print("=== Starting SWE-smith test reset process ===")
        
        # Step 1: First stash save current AI fix code
        # print("Step 1: Stashing current AI fixes...")
        
        # Check if there are uncommitted changes (AI fix code)
        uncommitted_changes, error_code = self.run("git status --porcelain")
        # print(f"--- Git status --porcelain (error code: {error_code}) ---")
        # print(uncommitted_changes)
        
        has_ai_fixes = False
        if error_code == "0" and uncommitted_changes.strip():
            # print("Found uncommitted changes (AI fixes), stashing them...")
            
            # Check changes in staging area and working directory
            staged_files, staged_error = self.run("git diff --cached --name-only")
            working_files, working_error = self.run("git diff --name-only")
            
            has_staged_changes = staged_error == "0" and staged_files.strip()
            has_working_changes = working_error == "0" and working_files.strip()
            
            if has_staged_changes or has_working_changes:
                # print("Modified files detected, stashing AI fixes...")
                try:
                    # First stash current changes (including staging area)
                    stash_output, stash_error = self.run("git stash push --include-untracked -m 'AI fixes before test'")
                    #print(f"Stash push result (error code: {stash_error}):")
                    #print(stash_output)
                    
                    if stash_error == "0":
                        # print("AI fixes stashed successfully")
                        has_ai_fixes = True
                    else:
                        print(f"Warning: Failed to stash changes: {stash_error}")
                        # print(f"Stash output: {stash_output}")
                        
                except Exception as e:
                    print(f"Error stashing AI fixes: {e}")
                    import traceback
                    print(f"Traceback: {traceback.format_exc()}")
            else:
                print("No actual file modifications found")
        else:
            print("No AI-generated fixes found")
        
        # Step 2: Switch to parent commit (bug state)
        #print("Step 2: Switching to parent commit (bug state)...")
        parent_commit, error_code = self.run("git rev-parse HEAD~1")
        if error_code != "0":
            #print(f"Warning: Failed to get parent commit: {error_code}")
            # If no parent commit, try to revert to specified commit_id
            commit_id = self.ds.get('instance_id')
            if commit_id:
                # print(f"Falling back to specified commit: {commit_id}")
                self.run(f"git checkout {commit_id}")
            else:
                # print("No parent commit or fallback commit available")
                return
        else:
            parent_commit = parent_commit.strip()
            #print(f"Switching to parent commit: {parent_commit}")
            self.run(f"git checkout {parent_commit}")
        
        # Step 3: Apply AI fix code from stash
        if has_ai_fixes:
            # print("Step 3: Applying AI fixes from stash...")
            try:
                # Apply stash
                apply_output, apply_error = self.run("git stash pop")
                # print(f"Stash pop result (error code: {apply_error}):")
                # print(apply_output)
                
                if apply_error == "0":
                    print("AI fixes applied successfully from stash")
                else:
                    # print(f"Warning: Failed to apply stash: {apply_error}")
                   # print(f"Stash apply output: {apply_output}")
                    # If stash pop fails, try reset --hard
                    #print("Attempting git reset --hard HEAD...")
                    reset_output, reset_error = self.run("git reset --hard HEAD")
                    #print(f"Reset result (error code: {reset_error}): {reset_output}")
                    
            except Exception as e:
                print(f"Error applying AI fixes: {e}")
                import traceback
                print(f"Traceback: {traceback.format_exc()}")
        else:
            print("Step 3: No AI fixes to apply")
        
        # Step 4: Restore changed test files
        # print("Step 4: Restoring test files...")
        f2p_files, p2p_files = self.rp.get_test_files(self.ds)
        all_test_files = list(set(f2p_files + p2p_files))
        # print(f"Test files to restore: {all_test_files}")
        
        if all_test_files:
            # Restore test files to original state
            test_files_str = " ".join(all_test_files)
            command = f"git checkout HEAD -- {test_files_str}"
            output, error_code = self.run(command)
            if error_code == "0":
                print(f"Test files restored successfully: {test_files_str}")
            else:
                print(f"Warning: Failed to restore some test files: {error_code}")
                # print(f"Restore output: {output}")
        else:
            print("No test files to restore")
        
        # Clean up stash (if exists)
        stash_list, stash_error = self.run("git stash list")
        if stash_error == "0" and stash_list.strip():
            print(f"Cleaning up stash: {stash_list}")
            self.run("git stash drop")
            print("Stash cleaned up")
        else:
            print("No stash to clean up")
        
        # print("=== SWE-smith test reset process completed ===")

    def setup_env_swesmith(self):
        try:
            #commit_id = self.ds['base_commit']
            commit_id = self.ds['instance_id']
            self.run("git fetch")
            self.run(f"git checkout {commit_id}")
            print(f"git checkout {commit_id}")
            # Setup the run_test.sh script for subsequent testing.  
            # test_command, _ = get_test_command(self.ds)
            test_command,_ = self.rp.get_test_cmd(self.ds)
            print(f"test_command: {test_command}")
            eval_script_content = "\n".join(
                [
                    "#!/bin/bash",
                    "set -uxo pipefail",
                    "source /opt/miniconda3/bin/activate",
                    f"conda activate testbed",
                    f"cd testbed/",
                    f": '>>>>> Start Test Output'",
                    test_command,
                    f": '>>>>> End Test Output'",
                ]
            ) + "\n"
            
            with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.sh') as temp_file:
                temp_file.write(eval_script_content)
                temp_file.flush()  # Ensure content is written to disk
                temp_file_path = temp_file.name
            
            # Copy the file to container and clean up
            self.copy_to_container(temp_file_path, "/run_tests.sh")
            os.unlink(temp_file_path)  # Clean up the temporary file
            
            self.run("chmod +x /run_tests.sh")

            # Ensure can call and execute the tools in /usr/local/bin.
            self.run(f"ln -s /opt/miniconda3/envs/testbed /root/.venv")
            self.run('echo \'export PATH="/usr/local/bin:$PATH"\' >> ~/.bashrc')
            self.run("/root/.venv/bin/python -m pip install chardet -i https://pypi.tuna.tsinghua.edu.cn/simple some-package")
            # self.run("/root/.venv/bin/python -m pip install pytest -i https://pypi.tuna.tsinghua.edu.cn/simple some-package")
        except Exception as e:
            self.logger.error(f"Error setting up environment: {repr(e)}")

    def setup_env_swebench(self):
        try:
            # make the run_tests.sh executable
            if "swebench/" in self.docker_image or self.ds["tag"] == "swerebench": # If it's a native swebench image
                with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.sh') as temp_file:
                    temp_file.write(self.test_spec.eval_script)
                    temp_file.flush()  # Ensure content is written to disk
                    temp_file_path = temp_file.name
            
                # Copy the file to container and clean up
                self.copy_to_container(temp_file_path, "/run_tests.sh")
                os.unlink(temp_file_path)  # Clean up the temporary file

            self.run("chmod +x /run_tests.sh")

            # # move all skip files (if present) to /root
            # for skip_file in SKIP_FILES:
            #     self.run(f"mv {self.repo_path}/{skip_file} {self.alt_path}/{skip_file}")
            self.alt_path = (
                "/"  # the run_test is in the "/" directory for swebench dockers
            )

            # make symlink of conda env to /root/.venv
            self.run(f"ln -s /opt/miniconda3/envs/testbed /root/.venv")

            # install required packages TODO: check if working
            # self.run(
            #     "python -m pip install tree-sitter==0.20.4 tree_sitter_languages==1.10.2"
            # )

            # self.run_with_retry("/root/.venv/bin/python -m pip install chardet -i https://pypi.tuna.tsinghua.edu.cn/simple some-package")
            # self.run_with_retry("/root/.venv/bin/python -m pip install pytest -i https://pypi.tuna.tsinghua.edu.cn/simple some-package")
            self.run_with_retry("/root/.venv/bin/python -m pip install chardet")
            self.run_with_retry("/root/.venv/bin/python -m pip install pytest")          
            # sudo apt-get install patchutils
            # self.run("apt-get update")
            # self.run("apt-get install -y patchutils")
        except Exception as e:
            self.logger.error(
                f"Error setting up environment: {repr(e)} @ {self.docker_image}"
            )
    
    def setup_env_swebench_live(self):
        try:
            # make the run_tests.sh executable
            with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.sh') as temp_file:
                temp_file.write(self.test_spec.eval_script)
                temp_file.flush()  # Ensure content is written to disk
                temp_file_path = temp_file.name
            
            # Copy the file to container and clean up
            self.copy_to_container(temp_file_path, "/run_tests.sh")
            os.unlink(temp_file_path)  # Clean up the temporary file
            
            self.run("chmod +x /run_tests.sh")

            # # move all skip files (if present) to /root
            # for skip_file in SKIP_FILES:
            #     self.run(f"mv {self.repo_path}/{skip_file} {self.alt_path}/{skip_file}")
            self.alt_path = (
                "/"  # the run_test is in the "/" directory for swebench dockers
            )

            # make symlink of conda env to /root/.venv
            self.run(f"ln -s /usr/local /root/.venv")

            # install required packages TODO: check if working
            # self.run(
            #     "python -m pip install tree-sitter==0.20.4 tree_sitter_languages==1.10.2"
            # )
            self.run_with_retry("/root/.venv/bin/python -m pip install chardet")
            self.run_with_retry("/root/.venv/bin/python -m pip install pytest")   
            # sudo apt-get install patchutils
            # self.run("apt-get update")
            # self.run("apt-get install -y patchutils")
        except Exception as e:
            self.logger.error(
                f"Error setting up environment: {repr(e)} @ {self.docker_image}"
            )
    # not change
    def setup_env(self):
        if self.swebench_verified or self.swerebench:
            return self.setup_env_swebench()
        elif self.swesmith:
            return self.setup_env_swesmith()
        elif self.live:
            return self.setup_env_swebench_live()
        elif self.swemul:
            # Setup Python environment for SWE-bench Multilingual
            try:
                # Install Python and pip
                self.logger.info("Installing Python environment for SWE-bench Multilingual...")
                
                # Update package lists and install Python
                self.run("apt-get update -y")
                self.run("bash -c 'DEBIAN_FRONTEND=noninteractive apt-get install -y python3 python3-pip'")
                self.run("mkdir -p /root/.venv/bin")
                self.run("ln -s /usr/bin/python3 /root/.venv/bin/python")
                self.run("apt install python3-chardet")
                #self.run(f"ln -s /usr/bin/python3 /usr/bin/python")
                # Create symlink to make Python available at /root/.venv/bin/python
                #self.run(f"ln -s /usr/local /root/.venv")
                
                self.logger.info("Python environment setup completed for SWE-bench Multilingual")
            except Exception as e:
                self.logger.error(f"Error setting up Python environment for SWE-bench Multilingual: {repr(e)}")
            return

        try:
            # setup venv
            # modify the repo path to a common path
            # self.run(f"cp -r {self.repo_path} /workspace")

            # create a symlink from repo_path/.venv to /root/.venv
            self.logger.info(f"Setting up symlink for virtual environment...")
            self.run(f"ln -s {self.repo_path}/.venv {self.alt_path}/.venv")

            self.run(
                f"ln -s {self.repo_path}/.venv/bin/python {self.alt_path}/.local/bin/python"
            )
            self.run(
                f"ln -s {self.repo_path}/.venv/bin/python {self.alt_path}/.local/bin/python3"
            )
            self.run(
                f"find {self.repo_path}/.venv/bin -type f -executable -exec ln -sf {{}} {self.alt_path}/.local/bin/ \\;"
            )
            # print(self.run(f"ls -l {self.alt_path}/.local/bin"))

            # self.run(f"mv {self.repo_path} /workspace")
            # self.repo_path = "/workspace"

            # install required packages
            # self.run("uv pip install tree_sitter_languages") # remove since already installed in new dockers

            # self.run("uv pip install chardet")
            self.run(f"/root/.venv/bin/python -m ensurepip && timeout 1800 /root/.venv/bin/python -m pip install chardet -i https://pypi.tuna.tsinghua.edu.cn/simple some-package")

            self.run("find . -name '*.pyc' -delete")

            self.run("find . -name '__pycache__' -exec rm -rf {} +")

            # also delete pycache and pyc from /r2e_tests
            self.run("find /r2e_tests -name '*.pyc' -delete")
            self.run("find /r2e_tests -name '__pycache__' -exec rm -rf {} +")

            # move all skip files (if present) to /root
            for skip_file in SKIP_FILES_NEW:
                self.run(f"mv {self.repo_path}/{skip_file} {self.alt_path}/{skip_file}")

            # r2e_tests are in the / directory, move them to /root
            self.run(f"mv /r2e_tests {self.alt_path}/r2e_tests")

            # make a softlink for /root/r2e_tests (if present)
            self.run(f"ln -s {self.alt_path}/r2e_tests {self.repo_path}/r2e_tests")
            # self.run(f"ln -s /r2e_tests {self.repo_path}/r2e_tests")
        except Exception as e:
            self.logger.error(f"Error setting up environment: {repr(e)}")

    def get_task_instruction(self) -> str:
        # try getting the content inside of [ISSUE] [/ISSUE] using regex tags for ds['problem_statement'] else return ds['problem_statement']
        try:
            content = self.ds["problem_statement"]
            return re.search(r"\[ISSUE\](.*)\[/ISSUE\]", content, re.DOTALL).group(1)
        except Exception as e:
            return self.ds["problem_statement"]

    def _run_kubernetes(
        self,
        code: str,
        timeout: int = CMD_TIMEOUT,
        args: str = "",
        workdir: str = "",
    ) -> tuple[str, str]:
        """
        Kubernetes-specific method to execute code or commands in the pod, with a timeout.
        Mirrors the logic of the original Docker `run` method using Kubernetes API.
        """
        # Command includes 'timeout' and potentially 'cd <workdir> &&' from the main run method
        command = ""
        if workdir:
            # Use '&&' so that failure to change directory aborts the command
            command += f"cd {workdir} && "
        command += f"timeout {timeout} {code} {args}"
        full_command = ["/bin/sh", "-c", command]
        try:
            # Define the exec function call within a lambda for the executor
            def execute_command():
                resp = stream(
                    self.client.connect_get_namespaced_pod_exec,
                    self.container_name,
                    DEFAULT_NAMESPACE,
                    command=full_command,
                    stderr=True,
                    stdin=False,
                    stdout=True,
                    tty=False,  # Match docker exec_run settings
                    _preload_content=False,  # Important for streaming
                )
                # Read until the command exits, accumulating each channel
                combined_chunks = []
                stdout_chunks = []
                stderr_chunks = []
                while resp.is_open():
                    resp.update(timeout=1)  # wait for data
                    if resp.peek_stdout():
                        chunk = resp.read_stdout()
                        stdout_chunks.append(chunk)
                        combined_chunks.append(chunk)
                    if resp.peek_stderr():
                        chunk = resp.read_stderr()
                        stderr_chunks.append(chunk)
                        combined_chunks.append(chunk)
                resp.close()
                exit_code = resp.returncode
                combined_output = "".join(combined_chunks)
                return combined_output, exit_code

            # Execute with an overall timeout slightly larger than the command's timeout
            with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
                future = executor.submit(execute_command)
                # Use timeout+10 as a buffer for k8s comms
                combined_output, exit_code = future.result(timeout=timeout + 5)

            # Process results - combined_output already preserves inter-leaved stdout/stderr
            output = combined_output

            if exit_code is None:  # Should not happen if command finished
                self.logger.error("Kubernetes exec: Exit code not found.")
                return output, "-1"  # Unknown error state

            if exit_code == 124:
                self.logger.error(f"Internal Timeout via 'timeout' command: {timeout}s")
                return f"The command took too long to execute (>{timeout}s)", "-1"

            if exit_code != 0:
                # Log format matches the docker version's error logging
                self.logger.error(
                    f"Kubernetes exec Error: Exit code {exit_code}\nError Message: {output}"
                )
                # Return combined output and error code string
                return output, f"Error: Exit code {exit_code}"

            # Remove ANSI escape codes and \r characters from the combined output
            output = re.sub(r"\x1b\[[0-9;]*m|\r", "", output)
            return output, str(exit_code)
        except concurrent.futures.TimeoutError:
            self.logger.error(f"Kubernetes exec Overall Timeout: {timeout + 5}s")
            return f"The command took too long to execute (>{timeout}s)", "-1"
        except client.ApiException as e:
            self.logger.error(f"Kubernetes API Error during exec: {e}")
            return f"Error executing command in pod: {repr(e)}", "-1"
        except Exception as e:
            self.logger.error(f"Unexpected error during Kubernetes exec: {repr(e)}")
            return f"Error: {repr(e)}", "-1"

    def _run_sandbox(
        self,
        code: str,
        timeout: int = CMD_TIMEOUT,
        args: str = "",
        workdir: str = "",
        task_type: str = "general",
    ) -> tuple[str, str]:
        """
        Sandbox-specific method to execute code or commands in the sandbox, with a timeout.
        """
        timeout = max(CMD_TIMEOUT, timeout)  # Enforce maximum timeout

        # Command includes 'timeout' and potentially 'cd <workdir> &&' from the main run method
        command = ""
        if workdir:
            # Use '&&' so that failure to change directory aborts the command
            command += f"cd {workdir} && export PATH=\"/root/.venv/bin:$PATH\" && "
        command += f"timeout {timeout} {code} {args}"
        # full_command = "/bin/sh -c '" + command + "'"
        full_command = command
        print(f"container_name: {self.container_name}, function_id: {self.function_id}")
        try:
            def execute_command_curl_fetch_files():
                import requests
                command_file = "rm -f /tmp/output.log && " + full_command + " > /tmp/output.log 2>&1"
                url = f"{self.sandbox_network}/run_code"
                headers = {
                    'accept': 'application/json, text/plain, */*',
                    'accept-language': 'zh-CN,zh;q=0.9',
                    'content-type': 'application/json',
                    'x-faas-instance-name': self.container_name
                }
                data = {
                    "code": command_file,
                    "language": "bash",
                    "run_timeout": timeout,
                    "compile_timeout": timeout,
                    "fetch_files": ["/tmp/output.log"],  # Return base64 encoded full content through files field
                }
                try:
                    # requests automatically converts the 'json' parameter into the request body
                    # and sets the 'Content-Type' header to 'application/json'.
                    response = requests.post(url, headers=headers, json=data,timeout=timeout + 5)

                    # Check if the request was successful
                    response.raise_for_status() 
                    response_dict = response.json()
                    if 'files' in response_dict and '/tmp/output.log' in response_dict['files']:
                        import base64
                        output_log_base64 = response_dict['files']['/tmp/output.log']
                        output_log = base64.b64decode(output_log_base64).decode('utf-8', errors='ignore')
                        response_dict['run_result']['stdout'] = output_log
                        response_dict['run_result']['stderr'] = ""

                    return response_dict["run_result"]

                except requests.exceptions.RequestException as e:
                    self.logger.error(f"Run code error occurred: {e}")

            def execute_command_curl():
                import requests
                url = f"{self.sandbox_network}/run_code"
                headers = {
                    'accept': 'application/json, text/plain, */*',
                    'accept-language': 'zh-CN,zh;q=0.9',
                    'content-type': 'application/json',
                    'x-faas-instance-name': self.container_name
                }
                data = {
                    "code": full_command,
                    "language": "bash",
                    "run_timeout": timeout,
                    "compile_timeout": timeout
                }
                try:
                    # requests automatically converts the 'json' parameter into the request body
                    # and sets the 'Content-Type' header to 'application/json'.
                    response = requests.post(url, headers=headers, json=data,timeout=timeout + 5)

                    # Check if the request was successful
                    response.raise_for_status() 

                    # Print the response details
                    # print(f"Status Code: {response.status_code}")
                    
                    # Try to print the JSON response content, if available
                    # try:
                    #     print("Response Body (JSON):")
                    #     print(json.dumps(response.json(), indent=4, ensure_ascii=False))
                    # except requests.exceptions.JSONDecodeError:
                    #     print("Response Body (Text):")
                    #     print(response.text)

                    # result_dict = json.loads(response)
                    # return result_dict["run_result"]

                    response_dict = response.json()


                    return response_dict["run_result"]

                except requests.exceptions.RequestException as e:
                    print(f"Run code error occurred: {e}")                

            # Define the exec function call within a lambda for the executor
            def execute_command():
                # client = self.client
                # data = {
                #     "code": full_command,
                #     "language": "bash",
                #     "run_timeout" : timeout + 5,
                #     "compile_timeout" : timeout + 5
                # }
                # data = json.dumps(data)
                # response = self.client.run_code(volcenginesdkvefaas.RunCodeRequest(
                #         function_id=self.function_id,
                #         sandbox_id=self.container_name,
                #         data=data,
                #     ),
                #     _request_timeout=timeout + 5  # Set API request timeout to 30 seconds
                # )
                # 1. Parse outer dictionary
                # result_str = response.result

                # # 2. Parse inner JSON string
                # result_dict = json.loads(result_str)

                # # 3. Extract run result
                # run_result = result_dict.get('run_result', {}) # Use .get() to avoid key errors

                # if task_type == "general":
                #     run_result = execute_command_curl()
                # else:
                run_result = execute_command_curl_fetch_files()

                self.logger.info(f"Sandbox run_code command: {full_command}")
                self.logger.info(f"Sandbox exec run_result: {run_result}")
                # 4. Extract required information
                stdout = run_result.get('stdout', '')
                stderr = run_result.get('stderr', '')
                exit_code = run_result.get('return_code', -1) # If not found, give a default value of -1
                combined_output = stdout + '\n' +stderr
                return combined_output, exit_code

            # Execute with an overall timeout slightly larger than the command's timeout
            with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
                future = executor.submit(execute_command)
                # Use timeout+10 as a buffer for sandbox comms
                combined_output, exit_code = future.result(timeout=timeout + 5)

            # Process results - combined_output already preserves inter-leaved stdout/stderr
            output = combined_output

            if exit_code is None:  # Should not happen if command finished
                self.logger.error("Sandbox exec: Exit code not found.")
                return output, "-1"  # Unknown error state

            if exit_code == 124:
                self.logger.error(f"Internal Timeout via 'timeout' command: {timeout}s")
                return f"The command took too long to execute (>{timeout}s)", "-1"

            if exit_code != 0:
                # Log format matches the docker version's error logging
                self.logger.error(
                    f"Sandbox exec Error: Exit code {exit_code}\nError Message: {output}"
                )
                # Return combined output and error code string
                return output, f"Error: Exit code {exit_code}"

            # Remove ANSI escape codes and \r characters from the combined output
            output = re.sub(r"\x1b\[[0-9;]*m|\r", "", output)
            return output, str(exit_code)
        except concurrent.futures.TimeoutError:
            self.logger.error(f"Sandbox exec Overall Timeout: {timeout + 5}s")
            return f"The command took too long to execute (>{timeout}s)", "-1"
        except client.ApiException as e:
            self.logger.error(f"Sandbox API Error during exec: {e}")
            return f"Error executing command in pod: {repr(e)}", "-1"
        except Exception as e:
            self.logger.error(f"Unexpected error during sandbox exec: {repr(e)}")
            return f"", "-1"


    def run_with_retry(self, code: str, timeout: int = CMD_TIMEOUT, args: str = "", workdir=None, type: str = None) -> tuple[str, str]:
        max_retries = 5
        retry_delay = 5
        for attempt in range(max_retries):
            output,exit_code = self.run(code, timeout, args, workdir, type)
            if exit_code == "0":
                return output, exit_code
            else:
                time.sleep(retry_delay)
                retry_delay *= 2
                retry_delay = min(retry_delay, 60)
        
        return output, "-1"
    # todo: add faas command execution
    def run(
        self,
        code: str,
        timeout: int = CMD_TIMEOUT,
        args: str = "",
        workdir=None,
        type: str = None,
    ) -> tuple[str, str]:
        """
        General method to execute code or commands in the container, with a timeout.

        :param code: The code or command to execute.
        :param args: Arguments to pass to the code/script.
        :param workdir: The working directory inside the container (optional).
        :return: A tuple containing (output, error_message). If no error, error_message is the exit code (str).
        """
        exec_code = code
        exec_workdir = self.repo_path if workdir is None else workdir

        if self.backend == "kubernetes":
            return self._run_kubernetes(exec_code, timeout, args, workdir=exec_workdir)
        elif self.backend == "sandbox":
            return self._run_sandbox(exec_code, timeout, args, workdir=exec_workdir)

        command = f"timeout {timeout} {exec_code} {args}"
        try:
            with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
                # Notice we do NOT set tty=True here
                future = executor.submit(
                    self.container.exec_run,
                    cmd=["/bin/sh", "-c", command],
                    # cmd=command,
                    workdir=exec_workdir,
                    stdout=True,
                    stderr=True,
                    environment={"PATH": DOCKER_PATH},
                )
                exec_result = future.result(timeout=timeout + 5)

            # Retrieve output and exit code
            output = exec_result.output.decode("utf-8", errors="replace")
            error_code = exec_result.exit_code

            if error_code == 124:
                self.logger.error(f"Internal Timeout: {timeout}s")
                return f"The command took too long to execute (>{timeout}s)", "-1"

            if error_code != 0:
                self.logger.error(
                    f"Error: Exit code {error_code} \nError Message: {output}"
                )
                return output, f"Error: Exit code {error_code}"

            # Remove ANSI escape codes and \r characters
            output = re.sub(r"\x1b\[[0-9;]*m|\r", "", output)
            return output, str(error_code)

        ## timeout
        except concurrent.futures.TimeoutError:
            self.logger.error(f"Timeout: {timeout}s")
            return f"The command took too long to execute (>{timeout}s)", "-1"

        except Exception as e:
            return f"Error: {repr(e)}", "-1"

    def demux_run(
        self, code: str, timeout: int = CMD_TIMEOUT, args: str = "", workdir=None
    ) -> tuple[str, str]:
        command = f"timeout {timeout} {code} {args}"
        try:
            with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
                # Set demux=True to get separate stdout and stderr streams
                future = executor.submit(
                    self.container.exec_run,
                    cmd=command,
                    workdir=self.repo_path if workdir is None else workdir,
                    demux=True,  # This is the key change
                    environment={"PATH": DOCKER_PATH},
                )
                exec_result = future.result(timeout=timeout + 5)

            # Unpack the result - when demux=True, output is a tuple of (stdout_data, stderr_data)
            output_data, error_data = exec_result.output
            error_code = exec_result.exit_code

            # Handle None cases and decode the outputs
            stdout = (
                output_data.decode("utf-8", errors="replace") if output_data else ""
            )
            stderr = error_data.decode("utf-8", errors="replace") if error_data else ""

            if error_code != 0:
                self.logger.error(
                    f"Error: Exit code {error_code} \nStdout Message: {stdout}, \nError Message: {stderr}"
                )
                return stdout, stderr, f"Error: Exit code {error_code}"

            return stdout, stderr, str(error_code)
        except Exception as e:
            return f"Error: {repr(e)}", f"Error: {repr(e)}", "-1"

    def _copy_to_container_sandbox(self, src_path: str, dest_path: str):
        """
        Copy a file or directory from host into Sandbox using tar over exec.
        """

        def copy_to_container():
            with open(src_path, 'rb') as f:
                content = f.read()
            base64_content = base64.b64encode(content).decode('utf-8')

            upload_data = {
                'code': f'print(open("{dest_path}").read())',
                'language': 'python',
                'files': {f'{dest_path}': base64_content}
            }
            data = json.dumps(upload_data)
            response = self.client.run_code(volcenginesdkvefaas.RunCodeRequest(
                    function_id=self.function_id,
                    sandbox_id=self.container_name,
                    data=data,
                ),
                _request_timeout=300  # Set API request timeout to 300 seconds
            )
            response = json.loads(response.result)
            return response

        # Retry with exponential backoff
        max_retries = 5
        retry_delay = 5  # Initial delay in seconds
        for attempt in range(max_retries):
            try:
                response = copy_to_container()  
                if response["status"] == "Success":
                    self.logger.info(f"Successfully copied file to Sandbox {self.container_name}")
                    break  # Success, exit the retry loop
                else:
                    self.logger.error(f"Failed to copy file to Sandbox {self.container_name}: {response['message']}")
                    raise RuntimeError(f"Failed to copy file after {max_retries} attempts.")
            except Exception as e:
                if attempt < max_retries - 1:
                    self.logger.warning(f"Copy to container failed (attempt {attempt+1}/{max_retries}): {str(e)}")
                    time.sleep(retry_delay)
                    retry_delay *= 2  # Exponential backoff
                    retry_delay = min(retry_delay, 60)
                else:
                    self.logger.error(f"Copy to container failed after {max_retries} attempts: {str(e)}")
                    raise

    def _copy_to_container_kubernetes(self, src_path: str, dest_path: str):
        """
        Copy a file or directory from host into Kubernetes pod using tar over exec.
        """
        # Calculate destination directory and prepare in-memory tarball
        dest_dir = os.path.dirname(dest_path)
        tar_stream = io.BytesIO()
        with tarfile.open(fileobj=tar_stream, mode="w") as tar:
            tar.add(src_path, arcname=os.path.basename(dest_path))
        tar_stream.seek(0)

        # Retry with exponential backoff
        max_retries = 5
        retry_delay = 5  # Initial delay in seconds
        for attempt in range(max_retries):
            try:
                # Exec into pod to untar into the destination directory
                exec_command = ["tar", "xmf", "-", "-C", dest_dir]
                
                self.logger.info(f"Attempting to copy file to Kubernetes pod {self.container_name} (attempt {attempt+1}/{max_retries})")
                
                resp = stream(
                    self.client.connect_get_namespaced_pod_exec,
                    self.container_name,
                    DEFAULT_NAMESPACE,
                    command=exec_command,
                    stderr=True,
                    stdin=True,
                    stdout=True,
                    tty=False,
                    _preload_content=False,
                )
                # Stream the tar binary data into the pod
                resp.write_stdin(tar_stream.read())
                resp.close()
                self.logger.info(f"Successfully copied file to Kubernetes pod {self.container_name}")
                break  # Success, exit the retry loop
            except kubernetes.client.exceptions.ApiException as e:
                if e.status == 403:
                    self.logger.error(f"Permission denied (403) when copying to Kubernetes pod. This might be a proxy/authentication issue.")
                    if attempt < max_retries - 1:
                        self.logger.warning(f"Retrying in {retry_delay}s...")
                        time.sleep(retry_delay)
                        retry_delay *= 2
                        retry_delay = min(retry_delay, 60)
                        tar_stream.seek(0)
                    else:
                        raise RuntimeError(f"Failed to copy file after {max_retries} attempts due to permission issues. Consider using Docker backend instead.")
                else:
                    self.logger.error(f"Kubernetes API error (status {e.status}): {e}")
                    raise
            except Exception as e:
                if attempt < max_retries - 1:
                    self.logger.warning(f"Copy to container failed (attempt {attempt+1}/{max_retries}): {str(e)}")
                    time.sleep(retry_delay)
                    retry_delay *= 2  # Exponential backoff
                    retry_delay = min(retry_delay, 60)
                    tar_stream.seek(0)  # Reset the stream for the next attempt
                else:
                    self.logger.error(f"Copy to container failed after {max_retries} attempts: {str(e)}")
                    raise


    # todo: add faas file copy
    def copy_to_container(self, src_path: str, dest_path: str):
        """
        Copies a file or directory from the host into the container (Docker or Kubernetes).
        """
        if self.backend == "docker":
            tar_stream = io.BytesIO()
            with tarfile.open(fileobj=tar_stream, mode="w") as tar:
                tar.add(src_path, arcname=os.path.basename(dest_path))
            tar_stream.seek(0)
            self.container.put_archive(os.path.dirname(dest_path), tar_stream.read())
        elif self.backend == "kubernetes":
            # Kubernetes pod copy
            return self._copy_to_container_kubernetes(src_path, dest_path)
        elif self.backend == "sandbox":
            return self._copy_to_container_sandbox(src_path, dest_path)
        # local -> tos -> container path
        # 1. Mount tos 
        # 2. Write code to download from tos wget error code?

    @DeprecationWarning  # TODO: remove dependency on this method with new dockers
    def read_file(self, rel_file_path: str) -> str:
        output, _ = self.run(f"cat /{self.alt_path}/{rel_file_path}")
        return output

    def run_tests(self, timeout: int = 300) -> tuple[str, str]:
        output, error_code = self.run(f"bash {self.alt_path}/run_tests.sh", timeout=timeout)
        # Remove ANSI escape codes and \r characters
        output = re.sub(r"\x1b\[[0-9;]*m|\r", "", output)
        return output, error_code

    def demux_run_tests(self) -> tuple[str, str, str]:
        stdout, stderr, error_code = self.demux_run(
            f"bash {self.alt_path}/run_tests.sh"
        )
        # Remove ANSI escape codes and \r characters
        stdout = re.sub(r"\x1b\[[0-9;]*m|\r", "", stdout)
        stderr = re.sub(r"\x1b\[[0-9;]*m|\r", "", stderr)
        return stdout, stderr, error_code

    def checkout(self, commit_hash: str) -> tuple[str, str]:
        output, error_code = self.run(f"git checkout {commit_hash}")
        return output, error_code

    def get_patch(self) -> str:
        """
        Get the diff of the current state of the repository.
        """
        # git add -A && git diff --cached
        # self.run("git add -A")
        output, _ = self.run("git add -A && git diff --cached")
        # output, _ = self.run("git diff")
        return output

    def create_file(self, file_path: str, content: str) -> tuple[str, str]:
        # create a local file with the content
        uuid_ = uuid.uuid4()
        file_path_ = f"{file_path}_{uuid_}"
        file_path__ = os.path.join("/tmp", file_path_)
        with open(file_path__, "w") as f:
            f.write(content)
        # copy the file to the container
        self.copy_to_container(file_path__, f"/testbed/{file_path_}")
        self.run(f"mv /testbed/{file_path_} /{file_path}")

    def apply_patch(self, patch: str) -> tuple[str, str]:
        # store the patch locally in a file identifiable by docker container id and timestamp
        # must contain unique patch name with both timestamp and docker image name
        uuid_ = uuid.uuid4()
        patch_path = f"{self.container_name}_{uuid_}.patch"
        patch_path = os.path.join("/tmp", patch_path)
        with open(patch_path, "w") as f:
            f.write(patch)
        # copy the patch to / of the container
        self.copy_to_container(patch_path, f"/{patch_path}")
        # apply the patch
        output, error_code = self.run(f"git apply --whitespace=fix /{patch_path}")
        return output, error_code

    def reverse_patch(self, patch: str) -> tuple[str, str]:
        # store the patch locally in a file identifiable by docker container id and timestamp
        patch_path = f"{self.container_name}_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.patch"
        patch_path = os.path.join("/tmp", patch_path)
        with open(patch_path, "w") as f:
            f.write(patch)
        # copy the patch to / of the container
        self.copy_to_container(patch_path, f"/{patch_path}")
        # apply the patch
        output, error_code = self.run(f"git apply -R /{patch_path}")
        return output, error_code

    def get_logs_eval(
        self, test_spec: TestSpec, content: str
    ) -> tuple[dict[str, str], bool]:
        """
        Retrieve evaluation results for a task instance from its corresponding log file

        Args:
            log_fp (str): path to log file
        Returns:
            bool: whether the patch applied successfully
            dict: status map

        modified from swebench/harness/grading.py
        """
        repo = test_spec.repo
        version = test_spec.version
        log_parser = MAP_REPO_TO_PARSER[repo]
        test_cmd = MAP_REPO_VERSION_TO_SPECS[repo][version]["test_cmd"]
        if isinstance(test_cmd, list):
            test_cmd = test_cmd[-1]

        # with open(log_fp) as f:
        # # TODO fix constant here
        bad_codes = list(
            filter(
                lambda x: x in content,
                [
                    APPLY_PATCH_FAIL,
                    RESET_FAILED,
                    TESTS_ERROR,
                    TESTS_TIMEOUT,
                ],
            )
        )
        if bad_codes:
            self.logger.error(f"Bad code found in log: {bad_codes}")
            return {}, False

        # elif not (START_TEST_OUTPUT in content and END_TEST_OUTPUT in content):
        #     # Test patch did not apply (should not happen at all)
        #     self.logger.error("Test patch did not apply")
        #     return {}, False

        # Get status map of evaluation results
        content = content.split(test_cmd)[-1]
        self.logger.info(f"using swebench log_parser for repo: {repo}")
        return log_parser(content, test_spec), True

    # def get_logs_eval_live(
    #     self, test_spec: TestSpec, content: str
    # ) -> tuple[dict[str, str], bool]:
    #     """
    #     Retrieve evaluation results for a task instance from its corresponding log file
    #     Specifically for swebench_live tasks

    #     Args:
    #         test_spec: test specification for the live task
    #         content (str): log content to parse
    #     Returns:
    #         tuple: (status map, whether the patch applied successfully)

    #     Modified from swebench/harness/grading.py for live tasks
    #     """
    #     repo = test_spec.repo
    #     version = test_spec.version
        
    #     # Import live-specific mappings and parsers
    #     from swebench_live.harness.log_parsers import MAP_REPO_TO_PARSER as MAP_REPO_TO_PARSER_LIVE
    #     from swebench_live.harness.constants import MAP_REPO_VERSION_TO_SPECS as MAP_REPO_VERSION_TO_SPECS_LIVE
        
    #     # Get the appropriate log parser for the repo
    #     log_parser = MAP_REPO_TO_PARSER_LIVE[repo]
        
    #     # Get test command from live specs
    #     test_cmd = MAP_REPO_VERSION_TO_SPECS_LIVE[repo][version]["test_cmd"]
    #     if isinstance(test_cmd, list):
    #         test_cmd = test_cmd[-1]

    #     # Check for bad codes in the log content
    #     # Import live-specific constants
    #     from swebench_live.harness.constants import (
    #         APPLY_PATCH_FAIL as APPLY_PATCH_FAIL_LIVE,
    #         RESET_FAILED as RESET_FAILED_LIVE,
    #         TESTS_ERROR as TESTS_ERROR_LIVE,
    #         TESTS_TIMEOUT as TESTS_TIMEOUT_LIVE,
    #     )
        
    #     bad_codes = list(
    #         filter(
    #             lambda x: x in content,
    #             [
    #                 APPLY_PATCH_FAIL_LIVE,
    #                 RESET_FAILED_LIVE,
    #                 TESTS_ERROR_LIVE,
    #                 TESTS_TIMEOUT_LIVE,
    #             ],
    #         )
    #     )
        
    #     if bad_codes:
    #         self.logger.error(f"Bad code found in live log: {bad_codes}")
    #         return {}, False

    #     # Extract the test output section after the test command
    #     content = content.split(test_cmd)[-1]
    #     self.logger.info(f"using swebench_live log_parser for repo: {repo}")
        
    #     # Parse the log content using the live parser
    #     return log_parser(content, test_spec), True


    def parse_log_smith(self, log_output: str) -> dict:
        """
        Parser for test logs generated with Sympy framework

        Args:
            log_output (str): log content
        Returns:
            dict: test case to test status mapping
        """
        if log_output is None:
            return {}
        test_status_map = {}
        if "short test summary info" in log_output:
            log_output = log_output.split("short test summary info")[0]
        log_output = log_output.strip()
        log_output = log_output.split("\n")
        for line in log_output:
            if "PASSED" in line:
                line = line.split(" ")[0]
                test_name = ".".join(line.split("::")[1:])
                test_status_map[test_name] = "PASSED"
            elif "FAILED" in line:
                line = line.split(" ")[0]
                test_name = ".".join(line.split("::")[1:]).split(" - ")[0]
                test_status_map[test_name] = "FAILED"
            elif "ERROR" in line:
                try:
                    line = line.split(" ")[0]
                    test_name = ".".join(line.split("::")[1:])
                except IndexError:
                    test_name = line
                test_name = test_name.split(" - ")[0]
                test_status_map[test_name] = "ERROR"
        return test_status_map

    def parse_logs(self, log_output: str) -> dict:
        if self.swebench_verified:
            parsed_output, patch_apply_success = self.get_logs_eval(
                self.test_spec, log_output
            )
            return parsed_output
        elif self.live:
            return self.parse_log_swebench_live(log_output)
        elif self.swesmith:
            return self.parse_log_smith(log_output)
        else:
            return parse_log_fn(f"{self.repo_name}")(log_output)
    
    def _calculate_reward_swesmith(self, get_test_output=False, timeout: int = 300) -> float | tuple[float, str]:
        self.reset_swesmith_tests()
        output, error_msg = self.run("/run_tests.sh", timeout=timeout)
        # print(f"output: {output}")
        start_sep = f"+ : '>>>>> Start Test Output'"
        end_sep = f"+ : '>>>>> End Test Output'"
        start_idx = output.find(start_sep)
        end_idx = output.find(end_sep)
        if start_idx > end_idx:
            raise ValueError(
                "Invalid test output - Start and end markers are not in correct order"
            )
        output = output[start_idx:end_idx][len(start_sep) :]
        print(f"new_output: {output}")
        test_status_map = self.rp.log_parser(output)
        report = swesmith_get_eval_tests_report(test_status_map, self.ds)
        if swesmith_get_resolution_status(report) == ResolvedStatus.FULL.value:
            reward = 1.0
        else:
            reward = 0.0
        if get_test_output:
            return reward, output
        return reward
        # parse = self.parse_logs(output)
        # print(f"parse: {parse}")
        
        # # if len(parse) == 0:
        # #     if get_test_output:
        # #         return 1.0, output
        # #     return 1.0
        
        # fail2pass = [ ".".join(line.split("::")[1:]) for line in self.ds['FAIL_TO_PASS']]
        # pass2pass = [ ".".join(line.split("::")[1:]) for line in self.ds['PASS_TO_PASS']]
        # # @(Naman, Jas): Parse the output and return the reward. This implementation is a hack rn.
        # if not parse:
        #     if get_test_output:
        #         return 0.0, output
        #     return 0.0
        
        # # Add debugging information below
        # # Check fail2pass
        # for test_name in fail2pass:
        #     if test_name not in parse:
        #         # Check if test_name is substring of any key
        #         matching_key = next((k for k in parse.keys() if test_name in k), None)
        #         if matching_key is None:
        #             if get_test_output:
        #                 print("fail2pass not in parse")
        #                 return 0.0, output
        #             return 0.0
        #         if parse[matching_key] != 'PASSED':
        #             if get_test_output:
        #                 print("fail2pass not passed")
        #                 return 0.0, output
        #             return 0.0
        #         test_name = matching_key
        #     if parse[test_name] != 'PASSED':
        #         if get_test_output:
        #             print("fail2pass not passed 2")
        #             return 0.0, output
        #         return 0.0
        
        # # Check pass2pass
        # for test_name in pass2pass:
        #     if test_name not in parse:
        #         # Check if test_name is substring of any key
        #         matching_key = next((k for k in parse.keys() if test_name in k), None)
        #         if matching_key is None:
        #             if get_test_output:
        #                 print("pass2pass not in parse")
        #                 return 0.0, output
        #             return 0.0
        #         test_name = matching_key
        #     if parse[test_name] != 'PASSED':
        #         print("fail2pass not passed 3")
        #         if get_test_output:
        #             print("fail2pass not passed 4")
        #             return 0.0, output
        #         return 0.0
        
        # # If the caller wants the test output as well, return (reward, output)
        # if get_test_output:
        #     return 1.0, output
        # return 1.0


    def _calculate_reward_swebench(self, get_test_output=False, timeout: int = 300) -> float:
        # gt_test_patch = self.commit.get_patch(test_file=True,non_test_file=False)
        # self.apply_patch(gt_test_patch)
        out, _ = self.run(
            "/run_tests.sh", timeout=timeout
        )  # run the tests after applying the patch
        eval_status_map, found = self.get_logs_eval(self.test_spec, out)
        eval_ref = {
            KEY_INSTANCE_ID: self.test_spec.instance_id,
            FAIL_TO_PASS: self.test_spec.FAIL_TO_PASS,
            PASS_TO_PASS: self.test_spec.PASS_TO_PASS,
        }
        report = get_eval_tests_report(
            eval_status_map, eval_ref, eval_type=get_eval_type(self.test_spec)
        )
        success = get_resolution_status(report) == ResolvedStatus.FULL.value
        if get_test_output:
            return success, out
        return int(success)
    
    def _calculate_reward_swebench_live(self, get_test_output=False, timeout: int = 300) -> float:
        """
        Calculate reward for swebench_live tasks
        Modified from _calculate_reward_swebench for live tasks
        """
        out, _ = self.run(
            "/run_tests.sh", timeout=timeout
        )  # run the tests after applying the patch
        
        from swebench_live.harness.grading import get_logs_eval as get_logs_eval_live
        # Use live-specific evaluation function
        # Write the out result to a temporary file first
        with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.txt') as temp_file:
            temp_file.write(out)
            temp_file.flush()  # Ensure content is written to disk
            temp_file_path = temp_file.name
        eval_status_map, found = get_logs_eval_live(self.test_spec, temp_file_path)
        os.unlink(temp_file_path)
        
        # Import live-specific constants and functions
        from swebench_live.harness.constants import (
            KEY_INSTANCE_ID as KEY_INSTANCE_ID_LIVE,
            FAIL_TO_PASS as FAIL_TO_PASS_LIVE,
            PASS_TO_PASS as PASS_TO_PASS_LIVE,
            FAIL_ONLY_REPOS as FAIL_ONLY_REPOS_LIVE,
            EvalType as EvalType_LIVE,
        )
        from swebench_live.harness.grading import (
            get_eval_tests_report as get_eval_tests_report_live,
            get_resolution_status as get_resolution_status_live,
        )
        from swebench_live.harness.constants import (
            ResolvedStatus as ResolvedStatus_LIVE,
        )
        
        eval_ref = {
            KEY_INSTANCE_ID_LIVE: self.test_spec.instance_id,
            FAIL_TO_PASS_LIVE: self.test_spec.FAIL_TO_PASS,
            PASS_TO_PASS_LIVE: self.test_spec.PASS_TO_PASS,
        }
        
        eval_type = EvalType_LIVE.FAIL_ONLY if self.test_spec.repo in FAIL_ONLY_REPOS_LIVE \
        else EvalType_LIVE.PASS_AND_FAIL
        
        report = get_eval_tests_report_live(
            eval_status_map, eval_ref, eval_type=eval_type
        )
        
        success = get_resolution_status_live(report) == ResolvedStatus_LIVE.FULL.value
        
        if get_test_output:
            return success, out
        return int(success)


    def _calculate_reward_swerebench(self, get_test_output=False, timeout: int = 300) -> float:
        """
        Calculate reward for swerebench tasks
        Modified from _calculate_reward_swerebench for swerebench tasks
        """
        out, _ = self.run(
            "/run_tests.sh", timeout=timeout
        )  # run the tests after applying the patch
        
        from swerebench.harness.grading import get_logs_eval as get_logs_eval_swerebench
        # Use swerebench-specific evaluation function
        # Write the out result to a temporary file first
        with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.txt') as temp_file:
            temp_file.write(out)
            temp_file.flush()  # Ensure content is written to disk
            temp_file_path = temp_file.name
        eval_status_map, found = get_logs_eval_swerebench(self.test_spec, temp_file_path)
        os.unlink(temp_file_path)
        
        # Import swerebench-specific constants and functions
        from swerebench.harness.constants import (
            KEY_INSTANCE_ID as KEY_INSTANCE_ID_LIVE,
            FAIL_TO_PASS as FAIL_TO_PASS_LIVE,
            PASS_TO_PASS as PASS_TO_PASS_LIVE,
            FAIL_ONLY_REPOS as FAIL_ONLY_REPOS_LIVE,
            EvalType as EvalType_LIVE,
        )
        from swerebench.harness.grading import (
            get_eval_tests_report as get_eval_tests_report_live,
            get_resolution_status as get_resolution_status_live,
        )
        from swerebench.harness.constants import (
            ResolvedStatus as ResolvedStatus_LIVE,
        )
        
        eval_ref = {
            KEY_INSTANCE_ID_LIVE: self.test_spec.instance_id,
            FAIL_TO_PASS_LIVE: self.test_spec.FAIL_TO_PASS,
            PASS_TO_PASS_LIVE: self.test_spec.PASS_TO_PASS,
        }
        
        eval_type = EvalType_LIVE.FAIL_ONLY if self.test_spec.repo in FAIL_ONLY_REPOS_LIVE \
        else EvalType_LIVE.PASS_AND_FAIL
        
        report = get_eval_tests_report_live(
            eval_status_map, eval_ref, eval_type=eval_type
        )
        
        success = get_resolution_status_live(report) == ResolvedStatus_LIVE.FULL.value
        
        if get_test_output:
            return success, out
        return int(success)

    def _calculate_reward_r2e(self, get_test_output=False, timeout: int = 300) -> float:
        # calculate reward based for r2e-edit dockers
        output, error_code = self.run_tests(timeout=timeout)
        # print(output)x
        parse = self.parse_logs(output)
        parse = decolor_dict_keys(parse)
        try:
            expected_json = self.ds["expected_output_json"]
        except Exception as e:
            expected_json = self.read_file("expected_test_output.json")

        expected: dict = json.loads(expected_json)
        expected = decolor_dict_keys(expected)
        parse = {k.split(" - ")[0]: parse[k] for k in sorted(parse.keys())}
        expected = {k.split(" - ")[0]: expected[k] for k in sorted(expected.keys())}

        # Compare
        if len(parse) != len(expected):
            reward = 0.0
        else:
            # If ANY mismatch, reward = 0.0, else = 1.0
            match = True
            for k in parse.keys():
                if not k:
                    continue
                if k not in expected:
                    match = False
                    break
                if parse[k] != expected[k]:
                    match = False
                    break
            reward = 1.0 if match else 0.0
        # If the caller wants the test output as well, return (reward, output)
        if get_test_output:
            return reward, output
        return reward

    def _calculate_reward(self, get_test_output=False, timeout: int = 300) -> float | tuple[float, str]:
        if self.swebench_verified:
            return self._calculate_reward_swebench(get_test_output=get_test_output, timeout=timeout)
        elif self.live:
            return self._calculate_reward_swebench_live(get_test_output=get_test_output, timeout=timeout)
        elif self.swesmith:
            return self._calculate_reward_swesmith(get_test_output=get_test_output, timeout=timeout)
        elif self.swerebench:
            return self._calculate_reward_swerebench(get_test_output=get_test_output, timeout=timeout)
        else:
            return self._calculate_reward_r2e(get_test_output=get_test_output, timeout=timeout)

    def reset(self):
        self.stop_container()
        self.start_container(
            self.docker_image, self.command, self.container_name, **self.docker_kwargs
        )

    def close(self):
        self.stop_container()
        if self.backend == "docker":
            if hasattr(self, '_uses_pool') and self._uses_pool:
                # If using client pool, release client instead of closing
                _global_docker_pool.release_client(self.client, self.docker_host)
            else:
                # Independent client, close directly
                try:
                    self.client.close()
                except:
                    pass  # Ignore exceptions when closing

    def run_swebv_regression(
        self, run_tests_regression: str | None = None, timeout: int = 300
    ) -> dict[str, str]:
        # run the regression tests for swebench verified dockers
        # copy the 'run_tests_regression' thing from ds into the container at /run_tests_regression.sh
        if run_tests_regression is None:
            run_tests_regression = self.ds["run_tests_regression"]

        with tempfile.NamedTemporaryFile("w") as f:
            f.write(run_tests_regression)
            f.flush()
            self.copy_to_container(f.name, "/run_tests_regression.sh")
        # make the script executable
        self.run("chmod +x /run_tests_regression.sh")

        # run the regression tests
        output, error_code = self.run("/run_tests_regression.sh", timeout=timeout)
        return output
        # return swebench_parse(self.ds, output)

    def start_new_branch(self, branch_name: str = "exp") -> tuple[str, str]:
        # ## save current branch-name
        # output, error_code = self.run("git branch --show-current")
        # self.current_branch = output.strip()
        # # new branch
        # output, error_code = self.run(f"git checkout -b {branch_name}")
        # # save commit hash

        output, error_code = self.run(
            "git config --global user.email 'you@example.com'"
        )
        output, error_code = self.run("git config --global user.name 'Your Name'")
        output, error_code = self.run("git rev-parse HEAD")
        self.current_commit = output.strip()
        return output, error_code

    def commit_after_step(self, step_idx: int) -> tuple[str, str]:
        # commit
        output, error_code = self.run("git add .")
        output, error_code = self.run(f"git commit -m '{step_idx}'")
        return output, error_code

    def undo_last_commit(self) -> tuple[str, str]:
        # undo last commit
        output, error_code = self.run("git reset --hard HEAD~1")
        return output, error_code

    def get_current_commit_hash(self) -> str:
        output, _ = self.run("git rev-parse HEAD")
        return output.strip()

    def soft_git_reset(self) -> tuple[str, str]:
        # soft reset to saved commit
        output, error_code = self.run(f"git reset --soft {self.current_commit}")

        # # checkout to saved branch
        # output, error_code = self.run(f"git checkout {self.current_branch}")

        return output, error_code
