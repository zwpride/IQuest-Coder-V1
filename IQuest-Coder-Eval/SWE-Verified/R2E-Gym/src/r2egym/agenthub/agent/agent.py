import os
import re
import copy
import yaml
import json
import time
from pathlib import Path
from dataclasses import dataclass, asdict
from typing import Any, Dict, List, Optional, Tuple
from pydantic import BaseModel

import litellm
from openai import OpenAI

from r2egym.agenthub.action import Action
from r2egym.agenthub.utils.log import get_logger
from r2egym.agenthub.environment.env import RepoEnv
from r2egym.agenthub.runtime.docker import DockerRuntime
from r2egym.agenthub.trajectory import TrajectoryStep, Trajectory
from anthropic import Anthropic, AnthropicVertex  # Add Anthropic Vertex import
from r2egym.agenthub.tools import (
    r2egym_bash_execute_tool,
    search_tool,
    file_editor,
    finish_tool,
    str_replace_editor_tool,
    execute_bash_tool,
    submit_tool,
)
import traceback
logger = get_logger(__name__)  # Logger for this module
MAX_CONTEXT_TOKENS = int(os.environ.get("MODEL_MAX_LEN", 65536))
##############################################################################
# AgentArgs Dataclass
##############################################################################
@dataclass
class AgentArgs:
    system_prompt: str
    instance_prompt: str
    command_files: List[Path]
    llm_name: str
    llm_base_url: Optional[str] = "http://localhost:8000/v1"  # None
    demo_file: Optional[Path] = None
    use_demo: Optional[bool] = False
    other_args: Optional[Dict[str, Any]] = None  # To handle extra configurations

    @classmethod
    def from_yaml(cls, yaml_path: Path) -> "AgentArgs":
        with open(yaml_path, "r") as file:
            config = yaml.safe_load(file)
        return cls(**config)


##############################################################################
# Agent Class
##############################################################################
class Agent:
    """Agent handles the behavior of the model and how it interacts with the environment."""

    def __init__(self, name: str, args: AgentArgs, logger=None):
        self.name = name
        self.args = args
        # self.trajectory_steps: List[TrajectoryStep] = []
        if logger is None:
            self.logger = get_logger(name)  # initialize logger from the agent name
        else:
            self.logger = logger
        self.llm_name = args.llm_name

        self.llm_base_url = (
            # "http://localhost:8000/v1"
            os.environ.get("LLM_BASE_URL", "http://localhost:8000/v1")
            if ("openai/" in self.llm_name) or ("hosted_vllm" in self.llm_name)
            else None
        )
        self.system_prompt_template = args.system_prompt
        self.instance_prompt_template = args.instance_prompt
        self.command_files = args.command_files
        self.other_args = args.other_args or {}
        self.logger.info(f"Initialized Agent: {name} with LLM: {args.llm_name}")
        self.max_retries = self.other_args.get("max_retries", 3)
        self.llm_timeout = self.other_args.get("timeout", 3000)

        import ast
        self.extra_headers = ast.literal_eval(os.environ.get("EXTRA_HEADERS", "{}")) if os.environ.get("EXTRA_HEADERS", "{}") else {}   
        self.extra_body = ast.literal_eval(os.environ.get("EXTRA_BODY", "{}")) if os.environ.get("EXTRA_BODY", "{}") else {}
        self.extra_query = ast.literal_eval(os.environ.get("EXTRA_QUERY", "{}")) if os.environ.get("EXTRA_QUERY", "{}") else {}
        self.logger.info(f"Extra headers: {self.extra_headers}")
        self.logger.info(f"Extra body: {self.extra_body}")
        self.logger.info(f"Extra query: {self.extra_query}")        

    def prepare_system_message(
        self, problem_statement: str, structure: str, command_docs: str, demo: str
    ) -> str:
        """Prepare the system prompt by filling in placeholders."""
        system_prompt = self.system_prompt_template.format(
            # problem_statement=problem_statement,
            # structure=structure,
            command_docs=command_docs,
            demo=demo,
        )
        return system_prompt

    def prepare_instance_prompt(
        self, agent_history: str, command_docs: str, steps_remaining: int
    ) -> str:
        """Prepare the instance prompt by filling in placeholders."""
        instance_prompt = self.instance_prompt_template.format(
            agent_history=agent_history,
            command_docs=command_docs,
        )
        # self.logger.info(isinstance(steps_remaining, int))
        # Add steps remaining message
        if steps_remaining > 0:
            stepcount_message = f"Steps Remaining: {steps_remaining}"
        else:
            stepcount_message = "You have reached the maximum number of steps. Please submit your answer NOW."
        instance_prompt += f"\n{stepcount_message}"
        self.logger.info(stepcount_message)  # Log the steps remaining message
        return instance_prompt

    def prepare_history_message(self, include_all_obs=False) -> str:
        """Prepare the agent's message history as a string."""
        history = ""
        for idx, step in enumerate(self.trajectory_steps):
            thought = step.thought
            action = step.action
            observation = step.observation
            # history += f'THOUGHT:\n```\n{thought}\n```\n'
            # history += f'ACTION:\n```\n{action}\n```\n'
            action_template = """
            {thought}
            ```
            {action}
            ```
            """
            history += action_template.format(thought=thought, action=action)
            if idx == len(self.trajectory_steps) - 1 or include_all_obs:
                history += f"\nOBSERVATION:\n```\n{observation}\n```\n"
            # add a separator
            history += "-" * 50 + "\n"
        return history

    def reset(self):
        """Reset the agent's trajectory."""
        self.trajectory_steps = []
        self.history = []

    def _count_tokens(self, messages: List[Dict[str, str]]) -> int:
        """
        Counts the tokens for a list of messages using the litellm library.
        Adjust as needed depending on the model and library.
        """
        token_count = litellm.token_counter(model=self.llm_name, messages=messages)
        
        # Filter out unsupported fields for token counting
        # filtered_messages = []
        # for message in messages:
        #     filtered_message = {}
        #     for key, value in message.items():
        #         # Skip cache_control and other non-string fields that litellm doesn't support
        #         if key in ["cache_control"] or not isinstance(value, str):
        #             continue
        #         filtered_message[key] = value
        #     filtered_messages.append(filtered_message)
        
        # token_count = litellm.token_counter(model=self.llm_name, messages=filtered_messages)
        self.logger.info(f"Total tokens in conversation: {token_count}")
        return token_count
    
    def delete_messages(self, messages: List[Dict[str, str]]) -> List[Dict[str, str]]:
        """Delete oldest user/assistant messages until token limit is satisfied"""
        
        if len(messages) <= 1:
            return messages
        
        # Preserve system and first user message
        preserved_messages = messages[:2] if messages[1]["role"] == "user" else [messages[0]]
        removable_messages = messages[2:] if messages[1]["role"] == "user" else messages[1:]
        
        # Try deleting from the oldest messages (from front to back)
        for i in range(len(removable_messages)):
            # Keep messages from index i onwards (i.e., delete the first i messages)
            temp_messages = preserved_messages + removable_messages[i:]
            token_count = self._count_tokens(temp_messages)
            
            if token_count < MAX_CONTEXT_TOKENS * 0.8:
                return temp_messages
        
        # If still not satisfied after deleting all removable messages, return the minimum preserved messages
        return preserved_messages

    def should_condense_with_llm(self, messages: List[Dict[str, str]], max_size: int = 50, token_threshold: float = 0.5) -> bool:
        """
        Check if LLMSummarizingCondenser compression is needed
        
        Args:
            messages: Message list
            max_size: Maximum message count threshold (default 50)
            token_threshold: Token usage rate threshold (default 0.75, i.e., 75%)
        
        Returns:
            bool: Returns True if compression is needed
        """
        # Check message count
        # if len(messages) > max_size:
        #     self.logger.info(f"Message count ({len(messages)}) exceeds max_size ({max_size}), should condense")
        #     return True

        # Check token count
        try:
            token_count = self._count_tokens(messages)
            if token_count > MAX_CONTEXT_TOKENS * token_threshold:
                self.logger.info(f"Token count ({token_count}) exceeds threshold ({MAX_CONTEXT_TOKENS * token_threshold}), should condense")
                return True
        except Exception as e:
            self.logger.warning(f"Failed to count tokens: {e}")
        
        return False

    def _truncate_content(self, content: str, max_chars: int = 10000) -> str:
        """Truncate content to fit maximum length limit"""
        if len(content) <= max_chars:
            return content
        return content[:max_chars] + "\n... [content truncated]"

    def generate_llm_summary(
        self, 
        messages: List[Dict[str, str]], 
        keep_first: int = 2,
        keep_last: int = 8,
        max_event_length: int = 5000
    ) -> str:
        """
        Generate structured summary using LLM (referencing LLMSummarizingCondenser)
        
        Args:
            messages: Complete message list
            keep_first: Keep first N messages (system + first user)
            keep_last: Keep last N messages (current progress)
            max_event_length: Maximum length of a single event
        
        Returns:
            str: Generated summary
        """
        if len(messages) <= keep_first + keep_last:
            return ""
        
        # Separate head, middle (to be compressed), and tail
        head_messages = messages[:keep_first]
        tail_messages = messages[-keep_last:] if keep_last > 0 else []
        middle_messages = messages[keep_first:-keep_last] if keep_last > 0 else messages[keep_first:]
        
        # Find previous summary
        previous_summary = ""
        for msg in reversed(head_messages[1:]):  # Skip system message
            content = msg.get("content", "")
            if msg.get("role") == "user" and "[LLM Summary]" in content:
                # Extract summary content
                previous_summary = content.replace("[LLM Summary]", "").strip()
                break
        
        # Build compression prompt (referencing LLMSummarizingCondenser)
        prompt = """You are maintaining a concise context-aware summary for an interactive agent solving software engineering tasks.
You will summarize past conversation history to maintain critical information while reducing context size.

SUMMARIZATION RULES:
1. Preserve exact task IDs and status if any are mentioned
2. Preserve code file paths, function names, and key variable names
3. Preserve test results and error messages
4. Preserve git/version control status (branch, commits)
5. Distinguish between COMPLETED and PENDING subtasks
6. Keep all critical information for task continuity

SUMMARY FORMAT:
USER_GOAL: [Brief user objective]
COMPLETED_WORK: [List of completed steps with results]
CURRENT_STATE: [Current progress, files modified, tests status]
PENDING_TASKS: [What still needs to be done]
KEY_FINDINGS: [Important discoveries, errors encountered]

Previous Summary (if any):
{previous_summary}

Events to summarize (from conversation history):
{events_text}

Now generate a concise summary preserving all critical information for task continuity."""
        
        # Build event text
        events_text = ""
        for i, msg in enumerate(middle_messages):
            role = msg.get("role", "unknown")
            content = msg.get("content", "")[:max_event_length]
            events_text += f"\n[{i}] {role}: {content}"
        
        prompt = prompt.format(
            previous_summary=previous_summary if previous_summary else "No previous summary",
            events_text=events_text
        )
        
        # Call LLM to generate summary
        try:
            summary_messages = [{"role": "user", "content": prompt}]
            completion_kwargs = {}
            if self.extra_headers:
                completion_kwargs["extra_headers"] = self.extra_headers
            if self.extra_body:
                completion_kwargs["extra_body"] = self.extra_body
            if self.extra_query:
                completion_kwargs["extra_query"] = self.extra_query

            model = os.environ.get("SUMMARY_MODEL_NAME", self.llm_name)
            api_base = os.environ.get("SUMMARY_API_URL", self.llm_base_url)
            api_key = os.environ.get("SUMMARY_API_KEY", os.environ.get("OPENAI_API_KEY", "EMPTY"))

            response = litellm.completion(
                model=model,
                messages=summary_messages,
                temperature=0.1,  # Low temperature for consistent summaries
                timeout=3600,
                api_base=api_base,
                api_key=api_key,
                max_tokens=16384,
                **completion_kwargs,
            )
            
            summary = response.choices[0].message.content
            self.logger.info(f"✓ LLM generated summary: {len(summary)} characters, {len(middle_messages)} events condensed")
            return summary
        except Exception as e:
            self.logger.error(f"Failed to generate LLM summary: {e}")
            # Fallback: Simple statistical summary
            return f"Summary: Condensed {len(middle_messages)} messages from conversation history."

    def _save_condensation_log(self, condensation_data: Dict[str, Any], instance_id: str) -> None:
        """
        Save condensation log to JSON file
        
        Args:
            condensation_data: All data related to condensation
            instance_id: Instance ID for creating log directory
        """
        try:
            import os
            from datetime import datetime
            
            # Create log directory
            log_dir = f"tmp/condensation_logs_{instance_id}"
            os.makedirs(log_dir, exist_ok=True)
            
            # Generate timestamp as log file name
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]
            log_file = f"{log_dir}/{timestamp}_condensation.json"
            
            # Save log
            with open(log_file, 'w', encoding='utf-8') as f:
                json.dump(condensation_data, f, indent=2, ensure_ascii=False)
            
            self.logger.info(f"✓ Condensation log saved to {log_file}")
        except Exception as e:
            self.logger.warning(f"Failed to save condensation log: {e}")

    def apply_llm_summarizing_condensation(
        self, 
        messages: List[Dict[str, str]], 
        keep_first: int = 2,
        keep_last: int = 12,
        instance_id: str = "default"
    ) -> List[Dict[str, str]]:
        """
        Apply LLMSummarizingCondenser compression logic
        
        Similar to LLMSummarizingCondenser:
        - Keep head messages (system + initial user)
        - Keep tail messages (recent progress)
        - Middle messages are replaced with summary
        
        Args:
            messages: Complete message list
            keep_first: Keep first N messages
            keep_last: Keep last N messages
            instance_id: Instance ID for saving logs
        
        Returns:
            List[Dict[str, str]]: Compressed message list
        """

        if len(messages) <= keep_first + keep_last:
            self.logger.info(f"Messages count ({len(messages)}) <= threshold ({keep_first + keep_last}), no condensation needed")
            return messages
        
        self.logger.info(f"Applying LLM Summarizing Condensation: {len(messages)} messages -> condensing middle section")
        
        # Record state before compression
        messages_before = copy.deepcopy(messages)
        token_count_before = self._count_tokens(messages)
        
        # Generate summary
        summary = self.generate_llm_summary(messages, keep_first, keep_last)
        
        # Build compressed message list
        head_messages = messages[:keep_first]
        tail_messages = messages[-keep_last:] if keep_last > 0 else []
        
        # Separate middle messages for logging
        middle_messages = messages[keep_first:-keep_last] if keep_last > 0 else messages[keep_first:]
        
        # Build compressed history
        condensed_messages = head_messages.copy()
        
        # Add summary as assistant message (simulating compression process)
        condensed_messages.append({
            "role": "assistant",
            "content": "[LLM-based context condensation applied]"
        })
        
        # Add summary as user observation message
        condensed_messages.append({
            "role": "user",
            "content": f"[LLM Summary]\n{summary}"
        })
        
        # Add tail messages
        condensed_messages.extend(tail_messages)
        
        token_count_after = self._count_tokens(condensed_messages)
        compression_ratio = round((1 - token_count_after / token_count_before) * 100, 1)
        
        # Build detailed log data
        condensation_log = {
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()),
            "compression_stats": {
                "messages_count_before": len(messages),
                "messages_count_after": len(condensed_messages),
                "token_count_before": token_count_before,
                "token_count_after": token_count_after,
                "compression_ratio": f"{compression_ratio}%",
                "middle_events_count": len(middle_messages),
            },
            "structure": {
                "head_messages": len(head_messages),
                "middle_messages_condensed": len(middle_messages),
                "tail_messages": len(tail_messages),
                "summary_message_added": 1,
            },
            "messages_before": [
                {
                    "index": i,
                    "role": msg.get("role", "unknown"),
                    "content_length": len(msg.get("content", "")),
                    "content_preview": msg.get("content", "")[:200] + ("..." if len(msg.get("content", "")) > 200 else "")
                }
                for i, msg in enumerate(messages_before)
            ],
            "messages_after": [
                {
                    "index": i,
                    "role": msg.get("role", "unknown"),
                    "content_length": len(msg.get("content", "")),
                    "content_preview": msg.get("content", "")[:200] + ("..." if len(msg.get("content", "")) > 200 else "")
                }
                for i, msg in enumerate(condensed_messages)
            ],
            "summary_generated": {
                "length": len(summary),
                # "preview": summary[:300] + ("..." if len(summary) > 300 else "")
                "preview": summary
            },
            "detailed_breakdown": {
                "head_section": {
                    "count": len(head_messages),
                    "messages": [
                        {
                            "index": i,
                            "role": msg.get("role", "unknown"),
                            "content_length": len(msg.get("content", ""))
                        }
                        for i, msg in enumerate(head_messages)
                    ]
                },
                "middle_section": {
                    "count": len(middle_messages),
                    "messages": [
                        {
                            "index": keep_first + i,
                            "role": msg.get("role", "unknown"),
                            "content_length": len(msg.get("content", ""))
                        }
                        for i, msg in enumerate(middle_messages)
                    ]
                },
                "tail_section": {
                    "count": len(tail_messages),
                    "messages": [
                        {
                            "index": len(messages) - len(tail_messages) + i,
                            "role": msg.get("role", "unknown"),
                            "content_length": len(msg.get("content", ""))
                        }
                        for i, msg in enumerate(tail_messages)
                    ]
                }
            }
        }
        
        # Save log
        # self._save_condensation_log(condensation_log, instance_id)
        
        self.logger.info(
            f"✓ LLM Condensation complete: {len(messages)} messages -> {len(condensed_messages)} messages | "
            f"Tokens: {token_count_before} -> {token_count_after} ({compression_ratio}% reduction)"
        )
        
        # Replace the observation of the last element in self.trajectory_steps with summary
        if self.trajectory_steps:
            self.trajectory_steps[-1].observation += f"[LLM Summary]\n{summary}"
            self.logger.info(f"✓ Updated last trajectory step observation with LLM summary ({len(summary)} chars)")
        
        return condensed_messages, token_count_after

    def apply_direct_deletion_condensation(
        self, 
        messages: List[Dict[str, str]], 
        keep_first: int = 2,
        keep_last: int = 12,
        target_compression_ratio: float = 80.0,
        max_iterations: int = 3
    ) -> Tuple[List[Dict[str, str]], int]:
        """
        Compress by directly deleting middle conversations (without using model to generate summary)
        
        - Keep head messages (system + initial user)
        - Keep tail messages (recent progress)
        - Directly delete middle messages
        - If compression ratio is below target, reduce keep_last and recompress, up to max_iterations times
        
        Args:
            messages: Complete message list
            keep_first: Keep first N messages
            keep_last: Keep last N messages (initial value, will be adjusted during iteration)
            target_compression_ratio: Target compression ratio (percentage, default 80%)
            max_iterations: Maximum number of iterations (default 3)
        
        Returns:
            Tuple[List[Dict[str, str]], int]: (Compressed message list, token count)
        """
        if len(messages) <= keep_first + keep_last:
            self.logger.info(
                f"Messages count ({len(messages)}) <= threshold ({keep_first + keep_last}), "
                f"no condensation needed"
            )
            token_count = self._count_tokens(messages)
            return messages, token_count
        
        # Record state before compression
        messages_before = copy.deepcopy(messages)
        token_count_before = self._count_tokens(messages)
        
        self.logger.info(
            f"Applying Direct Deletion Condensation: {len(messages)} messages, "
            f"keep_first={keep_first}, keep_last={keep_last} (initial)"
        )
        
        current_keep_last = keep_last
        condensed_messages = None
        token_count_after = None
        compression_ratio = 0.0
        
        # Iteratively compress until target compression ratio is reached or max iterations reached
        for iteration in range(1, max_iterations + 1):
            # Check if there are enough messages to delete
            if len(messages) <= keep_first + current_keep_last:
                self.logger.info(
                    f"Iteration {iteration}: Messages count ({len(messages)}) <= "
                    f"threshold ({keep_first + current_keep_last}), cannot compress further"
                )
                break
            
            # Build compressed message list (directly delete middle messages)
            head_messages = messages[:keep_first]
            tail_messages = messages[-current_keep_last:] if current_keep_last > 0 else []
            
            # Directly concatenate head and tail messages (without adding summary)
            condensed_messages = head_messages.copy()
            condensed_messages.extend(tail_messages)
            
            # Calculate token count after compression
            token_count_after = self._count_tokens(condensed_messages)
            compression_ratio = round((1 - token_count_after / token_count_before) * 100, 1)
            
            self.logger.info(
                f"Iteration {iteration}: {len(messages)} messages -> {len(condensed_messages)} messages | "
                f"Tokens: {token_count_before} -> {token_count_after} ({compression_ratio}% reduction) | "
                f"keep_last={current_keep_last}"
            )
            
            # Check if target compression ratio is reached
            if compression_ratio >= target_compression_ratio:
                self.logger.info(
                    f"✓ Target compression ratio ({target_compression_ratio}%) achieved "
                    f"after {iteration} iteration(s)"
                )
                break
            
            # If target not reached and there are more iterations, reduce keep_last
            if iteration < max_iterations:
                # Reduce keep_last (decrease by about 1/3 each time, but keep at least 1)
                new_keep_last = max(1, int(current_keep_last * 0.7))
                if new_keep_last >= current_keep_last:
                    new_keep_last = max(1, current_keep_last - 1)
                
                if new_keep_last < current_keep_last:
                    self.logger.info(
                        f"Compression ratio ({compression_ratio}%) < target ({target_compression_ratio}%), "
                        f"reducing keep_last: {current_keep_last} -> {new_keep_last}"
                    )
                    current_keep_last = new_keep_last
                else:
                    self.logger.info(
                        f"Cannot reduce keep_last further (current: {current_keep_last}), "
                        f"stopping iterations"
                    )
                    break
            else:
                self.logger.warning(
                    f"Reached max iterations ({max_iterations}) with compression ratio "
                    f"({compression_ratio}%) < target ({target_compression_ratio}%)"
                )
        
        # If target not reached after last iteration, use the last result
        if condensed_messages is None:
            # This should theoretically not happen, but as a safety measure
            head_messages = messages[:keep_first]
            tail_messages = messages[-current_keep_last:] if current_keep_last > 0 else []
            condensed_messages = head_messages.copy()
            condensed_messages.extend(tail_messages)
            token_count_after = self._count_tokens(condensed_messages)
            compression_ratio = round((1 - token_count_after / token_count_before) * 100, 1)
        
        # Calculate deleted middle messages
        middle_messages = messages[keep_first:-current_keep_last] if current_keep_last > 0 else messages[keep_first:]
        
        # Build log information
        self.logger.info(
            f"✓ Direct Deletion Condensation complete: "
            f"{len(messages)} messages -> {len(condensed_messages)} messages | "
            f"Tokens: {token_count_before} -> {token_count_after} ({compression_ratio}% reduction) | "
            f"Deleted {len(middle_messages)} middle messages"
        )
        
        return condensed_messages, token_count_after

    def model_query(
        self, messages: List[Dict[str, str]], temperature: float = 0, instance_id=None) -> Dict[str, Any]:
        """Query the LLM with the messages and measure execution time."""
        response = None
        retries = 0
        tools = None

        if self.use_fn_calling:
            if self.scaffold == "r2egym":
                tools = [search_tool, file_editor, r2egym_bash_execute_tool, finish_tool]
            elif self.scaffold == "openhands" or self.scaffold == "sweagent":
                tools = [str_replace_editor_tool, execute_bash_tool, submit_tool]

        # Start timer
        start_time = time.time()
        # check if using locally hosted models
        using_local = "openai/" in self.llm_name or "hosted" in self.llm_name
        # if using_local:
        #     litellm.api_key = None
        # else:
        litellm.api_key = os.environ.get("OPENAI_API_KEY", "EMPTY")
        if not litellm.api_key:
            raise ValueError("OPENAI_API_KEY environment variable is not set.")
        
        messages_ = copy.deepcopy(messages)
        
        # query the model with retries
        while retries < self.max_retries:
            try:
                # kwargs = {
                #     "tool_choice": "none",
                #     "function_call": None,
                # }
                kwargs={}
                if tools:
                    kwargs = {}
                # if "o3" not in self.llm_name and "o4" not in self.llm_name:
                #     kwargs["temperature"] = temperature

                # 1. Temporary fix for 'NoneType' error: only pass extra parameters when non-empty
                completion_kwargs = kwargs.copy()
                if self.extra_headers:
                    completion_kwargs["extra_headers"] = self.extra_headers
                if self.extra_body or instance_id:
                    extra_body = self.extra_body.copy() if self.extra_body else {}
                    if instance_id:
                        extra_body["instance_id"] = instance_id
                    completion_kwargs["extra_body"] = extra_body
                if self.extra_query:
                    completion_kwargs["extra_query"] = self.extra_query
                self.logger.info(f"Completion kwargs: {completion_kwargs}")

                response = litellm.completion(
                    model=self.llm_name,
                    tools=tools,
                    messages=messages_,
                    # timeout=self.llm_timeout,
                    temperature=temperature,
                    timeout=3600,
                    api_base=self.llm_base_url,
                    api_key=os.environ.get("OPENAI_API_KEY", "EMPTY"),
                    max_tokens=16384,
                    **completion_kwargs,
                )
                self.logger.warning(f"Querying LLM complete")
                break
            except Exception as e:
                self.logger.error(f"LLM query failed @ {retries}: {e}")
                retries += 1
                if "RateLimitError" in str(e):
                    time.sleep(60)
                elif "context length" in str(e) or "Total tokens" in str(e) or "Please reduce the length of the input messages" in str(e) or "ContextWindowExceededError" in str(e):    
                    self.logger.error(f"Context length exceeded. Current messages length: {len(messages_)}")
                    if self._count_tokens(messages_) > 2 * MAX_CONTEXT_TOKENS:
                        # Usually because obs returned abnormal environment information, directly give up
                        raise e
                    else:
                        self.logger.warning(f"Deleting messages to fit token limit")
                        # messages_ = self.delete_messages(messages_)
                        messages_,_ = self.apply_llm_summarizing_condensation(messages_, keep_first=2, keep_last=60, instance_id=instance_id)
                        self.history = copy.deepcopy(messages_)
                        self.logger.info(f"✓ Condensation successful, messages reduced to {len(messages_)}")
                        if len(messages_) < 10:
                            raise e
                if retries >= self.max_retries:
                    raise e

        # End timer, calculate total execution time, and include in response
        exec_time = time.time() - start_time
        return response, exec_time

    def parse_response(self, response: Dict[str, Any]) -> Tuple[str, Action]:
        """
        Parse the response from the LLM.
        """
        """
        Extracts:
        - thought: first thing in <think>...</think> block
        - action: the entire first <function=...></function> block
        Returns (thought, action).
        """
        # Regex to match (non-greedily) from `<think>` up to the first `</think>`
        pattern_thought = re.compile(r"(?s)(<think>.*?</think>)")
        pattern_action = re.compile(r"(?s)(<function=.*?</function>)")
        match_thought = pattern_thought.search(response)
        match_action = pattern_action.search(response)

        if match_thought:
            thought = match_thought.group(1)  # The entire <think>...</think> block
        else:
            thought = ""
        if match_action:
            action = match_action.group(1)  # The entire <function=...></function> block
        else:
            action = ""
        # Strip leading/trailing whitespace
        thought = thought.strip()
        action = action.strip()

        # convert action to Action object
        action = Action.from_string(action)

        return thought, action

    def parse_response_v2(self, response_text: str) -> Tuple[str, Action]:
        """
        Extracts:
        - thought: everything before the first <function=...> block
        - action: the entire first <function=...></function> block
        Returns (thought, action).
        """
        # Regex to match (non-greedily) from `<function=` up to the first `</function>`
        pattern = re.compile(r"(?s)(<function=.*?</function>)")
        match = pattern.search(response_text)

        if match:
            action = match.group(1)  # The entire <function=...></function> block
            thought = response_text[: match.start()]  # Everything before the block
        else:
            # If no match, treat entire text as "thought"
            thought = response_text
            action = ""

        # Strip leading/trailing whitespace
        thought = thought.strip()
        action = action.strip()

        # convert action to Action object
        action = Action.from_string(action)

        return thought, action

    def custom_parser(self, response):
        thought = response.choices[0].message.content
        if not thought:
            thought = ""

        try:
            function_name = response.choices[0].message.tool_calls[0].function.name
            parameters = json.loads(
                response.choices[0].message.tool_calls[0].function.arguments
            )
            action = Action(function_name=function_name, parameters=parameters)
        except:
            action = Action(function_name="", parameters={})

        return thought, action

    def run(
        self,
        env: "RepoEnv",  # env: RepoEnv
        use_fn_calling: bool = True,
        # step limits TODO: maybe add these limits in the agent args
        max_steps: int = 10,
        max_steps_absolute: int = 50,
        # token limits
        max_token_limit: int = 65536,  # 64k tokens
        # time limits
        max_exec_time: int = 90,  # 5 mins per env execution
        max_total_time: int = 50000,  # 20 minutes overall agent run limit
        max_llm_time: int = 7200,  # 2 mins per LLM timeout (note this is per query exlcuding retries | not enforcing hard limit since llm might hit rate limits etc)
        # temperature
        temperature=0,
        # additional metadata e.g. for hints / additional inputs etc
        metadata: Optional[Dict[str, Any]] = {},
        scaffold: str = "r2egym",
    ):
        assert scaffold in ["r2egym", "openhands", "sweagent","mopenhands"], "Scaffold must be either r2egym or openhands or sweagent"
        self.scaffold = scaffold
        # get the start time
        start_time = time.time()
        self.llm_timeout = max_llm_time

        # if self.llm_name is not gpt or sonnet, disable fn calling
        support_fn_calling = (
            "gpt" in self.llm_name
            or "sonnet" in self.llm_name
            or "o3" in self.llm_name
            or "o4" in self.llm_name
            or "deepseek" in self.llm_name
            or "Qwen" in self.llm_name
            # and "qwen" not in self.llm_name
        )
        self.use_fn_calling = use_fn_calling and support_fn_calling
        self.logger.warning(f"Using fn calling: {self.use_fn_calling}")

        # Log the environment and agent
        self.logger.info(f"Running agent {self.name} in environment {env}.")

        # Reset the environment and the agent
        # env.reset()
        env.add_commands(self.command_files)
        self.reset()

        instance_id = env.args.ds['instance_id']
        # Prepare problem_statement and structure from the environment
        problem_statement = env.runtime.get_task_instruction()
        self.logger.info(f"Problem Statement: {problem_statement}")
        if hasattr(env.runtime, 'commit'):
            gt_patch = env.runtime.commit.get_patch(test_file=True, non_test_file=False)
        else:
            gt_patch = ""

        # get system and instance prompts
        system_prompt = self.system_prompt_template
        user_prompt = self.instance_prompt_template.format(
            problem_statement=problem_statement,
            gt_patch=gt_patch,
            working_dir='/testbed',
            # base_commit=env.runtime.ds['base_commit'],
            test_patch_hint=metadata.get("test_patch_hint", ""),
            candidate_patch=metadata.get("candidate_patch", ""),
            candidate_patch_correctness=(
                "correct"
                if metadata.get("candidate_patch_correctness", False)
                else "incorrect"
            ),
        )
        self.logger.info(f"System Prompt: {system_prompt}")
        self.logger.info(f"User Prompt: {user_prompt}")

        if self.args.use_demo:
            with open(self.args.demo_file, "r") as file:
                demo = file.read()
            user_prompt = f"{demo}\n\n{user_prompt}"
        self.logger.info(f"User Prompt with demo: {user_prompt}")

        # initialize the history
        self.history = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]

        # initialize the parameters
        obs = None
        done = False
        step_count = 0
        total_time_traj = 0
        total_llm_time = 0
        total_env_time = 0
        self.trajectory_steps: List[TrajectoryStep] = []

        # agent loop
        while not done:
            # Prepare the agent's message history
            # self.logger.info(isinstance(steps_remaining, int))
            # Add steps remaining message
            steps_remaining = max_steps - step_count
            if steps_remaining > 0:
                stepcount_message = f"Steps Remaining: {steps_remaining}"
            else:
                stepcount_message = "You have reached the maximum number of steps. Please submit your answer NOW."
            self.history[-1][
                "content"
            ] += f"\n{stepcount_message}"  # postpend stepcount message
            self.logger.info(stepcount_message)

            # Query the LLM
            messages = copy.deepcopy(self.history)
            try:
                response, llm_exec_time = self.model_query(messages, temperature,instance_id=instance_id)
            except Exception as e:
                self.logger.error(f"Error querying LLM: {e}")
                self.logger.error(f"Error querying LLM: {traceback.format_exc()}")
                done = True
                exit_reason = "llm_query_error"
                # If it's because context length exceeded, exit; otherwise raise exception
                # if "is longer than the model's context length" in str(e) or "Total tokens" in str(e) or "Please reduce the length of the input messages" in str(e) or "ContextWindowExceededError" in str(e):
                #     break
                # else:
                # raise e
                break

            # Log total tokens in the response
            if hasattr(response, "usage"):
                usage = response.usage
                prompt_tokens = getattr(usage, "prompt_tokens", 0)
                completion_tokens = getattr(usage, "completion_tokens", 0)
                total_tokens = getattr(usage, "total_tokens", 0)

                prompt_tokens_details = getattr(usage, "prompt_tokens_details", None)
                self.logger.warning(f"Prompt Token Details: {prompt_tokens_details}")
                self.logger.info(
                    f"Prompt Tokens: {prompt_tokens}\nCompletion Tokens: {completion_tokens}\nTotal Tokens: {total_tokens}"
                )
            else:
                completion_tokens = -1
                prompt_tokens = -1
                total_tokens = -1
                total_tokens =  self._count_tokens(messages)
                self.logger.warning(
                    "No token usage information available in the response."
                )

            # Parse the LLM response to get 'thought' and 'action'
            self.response = response  # for debugging
            assistant_message = response.choices[0].message.content
            self.logger.info(f"Assistant's message:\n{assistant_message}\n")

            if self.use_fn_calling:
                thought, action = self.custom_parser(response)
            else:
                thought, action = self.parse_response_v2(assistant_message)

            action_str = action.to_xml_string()
            self.logger.info(f"THOUGHT:\n{thought}\n")
            self.logger.info(f"ACTION:\n{action.to_bashcmd()}\n")

            # Send the action to the environment
            try:
                obs, reward, done, info = env.step(action, timeout=max_exec_time)
                # env.runtime.commit_after_step(step_count)
            except Exception as e:
                obs = str(e)
                self.logger.error(f"Error during environment step: {obs}")

            env_exec_time = info["total_time"]
            total_step_time = llm_exec_time + env_exec_time
            total_llm_time += llm_exec_time
            total_env_time += env_exec_time
            total_time_traj += total_step_time
            step_count += 1  # Increment the step count
            self.logger.info(f"Total step time: {total_step_time}, LLM time: {llm_exec_time}, Env time: {env_exec_time}")
            self.logger.info(f"Total LLM time: {total_llm_time}, Total env time: {total_env_time}")

            if self.use_fn_calling:
                assistant_response = response.choices[0].message.dict()
                if assistant_response.get("tool_calls", None):
                    assistant_response["tool_calls"] = assistant_response["tool_calls"][
                        :1
                    ]  # only keep the first tool call
                self.history.append(assistant_response)
                # add tool response / user response to history
                try:
                    function_name = (
                        response.choices[0].message.tool_calls[0].function.name
                    )
                    function_id = response.choices[0].message.tool_calls[0].id
                    self.history.append(
                        {
                            "role": "tool",
                            "content": str(obs),
                            "name": function_name,
                            "tool_call_id": function_id,
                        }
                    )
                    self.logger.warning("logging fn response as a tool call")
                    self.logger.warning(
                        f"number of fn calls: {len(response.choices[0].message.tool_calls)}"
                    )
                except Exception as e:
                    self.logger.error(f"Error logging tool response: {e}")
                    self.logger.warning("fallback: logging fn response as a tool call")
                    self.history.append({"role": "user", "content": str(obs)})
            else:
                self.logger.warning("logging fn response as a user message")
                assistant_message = f"{thought}\n\n{action.to_xml_string()}"
                # assistant_message = f"{thought}\n\n{original_xml_str}"
                self.history.append({"role": "assistant", "content": assistant_message})
                self.history.append({"role": "user", "content": str(obs)})

            # Log the thought, action, and observation
            self.logger.info(f"OBSERVATION:\n{obs}\n")
            self.logger.info("-" * 50)

            # Check if the agent has reached limits or done
            # check if agent has finished naturally i.e. the agent uses the finish tool
            if done:
                if steps_remaining > 0:
                    self.logger.info(
                        f"Agent has finished naturally before step limit. current step count: {step_count}. max steps: {max_steps}."
                    )
                    exit_reason = "agent"
                elif steps_remaining == 0:
                    self.logger.info(
                        f"Agent finised on reaching the maximum number of steps: {max_steps}. current step count: {step_count}."
                    )
                    exit_reason = "max_step_limit"
                else:
                    self.logger.info(
                        f"Agent has finished after continuing past the max steps: {max_steps}. current step count: {step_count}."
                    )
                    exit_reason = "agent_max_step_limit"
            # check for token limit
            # elif total_tokens >= max_token_limit:
            #     self.logger.info(
            #         f"Agent reached max tokens: {max_token_limit}. Current token count: {total_tokens}. Exiting."
            #     )
            #     exit_reason = "token_limit"
            #     done = True
            # check for absolute step limit | note that the max steps is just indicative but the absolute step limit is the hard limit
            elif step_count >= max_steps_absolute:
                self.logger.info(
                    f"Agent reached max steps: {max_steps_absolute}. Exiting."
                )
                exit_reason = "abs_step_limit"
                done = True

            elif total_time_traj >= max_total_time:
                self.logger.info(f"Agent reached max time: {max_total_time}. Exiting.")
                exit_reason = "traj_time_limit"
                done = True

            # Create a TrajectoryStep object and append to the list
            trajectory_step = TrajectoryStep(
                # key parts
                step_idx=step_count - 1,
                thought=thought,
                action=action.to_xml_string(),
                observation=str(obs),
                done=done,
                info=info,  # also store the info to be safe
                # user_message=user_prompt,
                assistant_message=assistant_message,
                # tokens
                token_usage_prompt=prompt_tokens,
                token_usage_completion=completion_tokens,
                token_usage_total=total_tokens,
                # metadata (current step stats)
                llm_exec_time=llm_exec_time,
                env_exec_time=env_exec_time,
                total_step_time=total_step_time,
                total_time_traj=total_time_traj,
                step_count=step_count,
            )
            self.trajectory_steps.append(trajectory_step)

        # get the output patch
        # output_patch, _ = env.runtime.run(f"git diff {initial_commit} HEAD")
        # output_patch, _ = env.runtime.run(f"git diff {initial_commit} HEAD -- . ':(exclude)pyproject.toml'")
        # env.runtime.soft_git_reset()

        # compute output patch cummulatively from the start using git diff from the initial commit
        output_patch = env.runtime.get_patch()

        # Create a Trajectory object
        self.trajectory = Trajectory(
            trajectory_steps=[
                traj_step.model_dump() for traj_step in self.trajectory_steps
            ],
            problem_statement=problem_statement,
            docker_image=env.runtime.docker_image,
            agent_args=asdict(self.args),
            env_args=asdict(env.args),
            max_steps=max_steps,
            max_steps_absolute=max_steps_absolute,
            max_token_limit=max_token_limit,
            max_llm_time=max_llm_time,
            max_exec_time=max_exec_time,
            max_total_time=max_total_time,
            total_llm_time=total_llm_time,
            total_env_time=total_env_time,
            exit_reason=exit_reason,  # reason for exiting. must be one of the [agent, max_step_limit, agent_max_step_limit, abs_step_limit, token_limit, traj_time_limit, llm_query_error]
            output_patch=output_patch,
        )

        self.logger.info(f"Agent completed in {time.time() - start_time} seconds.")
        return self.trajectory
