from typing import List, Optional
from pathlib import Path
from bfcl._llm_response_generation import get_args, main as generation_main
from bfcl.eval_checker.eval_runner import main as evaluation_main
from bfcl.constants.eval_config import (
    DOTENV_PATH,
    PROJECT_ROOT,
    RESULT_PATH,
    SCORE_PATH,
)
from dotenv import load_dotenv
from types import SimpleNamespace
import wandb
#from config_singleton import WandbConfigSingleton

def run(
    model: List[str] = ["gorilla-openfunctions-v2"],
    test_category: List[str] = ["all"],
    # Generation specific options
    temperature: float = 0.001,
    include_input_log: bool = False,
    exclude_state_log: bool = False,
    num_gpus: int = 1,
    num_threads: int = 1,
    gpu_memory_utilization: float = 0.9,
    backend: str = "vllm",
    skip_server_setup: bool = False,
    local_model_path: Optional[str] = None,
    result_dir: str = RESULT_PATH,
    allow_overwrite: bool = False,
    run_ids: bool = False,
    # Evaluation specific options
    score_dir: str = SCORE_PATH,
    # Control flags
    skip_generation: bool = False,
    skip_evaluation: bool = False,
    # Sampling options
    samples_per_category: Optional[int] = None,
):
    """
    Run both generation and evaluation for the specified models and test categories.

    Args:
        model: List of model names to generate/evaluate
        test_category: List of test categories to run
        temperature: Temperature parameter for the model
        include_input_log: Include input log in inference log
        exclude_state_log: Exclude state log in inference log
        num_gpus: Number of GPUs to use
        num_threads: Number of threads to use
        gpu_memory_utilization: GPU memory utilization
        backend: Backend to use for the model
        skip_server_setup: Skip server setup
        local_model_path: Path to local model
        result_dir: Path to result directory
        allow_overwrite: Allow overwriting existing results
        run_ids: Run test entry IDs
        score_dir: Path to score directory
        skip_generation: Skip generation step
        skip_evaluation: Skip evaluation step
        samples_per_category: Number of samples to take from each category. If None, use all samples.
    """
    # Load environment variables
    load_dotenv(dotenv_path=DOTENV_PATH, verbose=True, override=True)

    # Convert model and test_category to the format expected by the original implementation
    if isinstance(model, str):
        model = [model]
    if isinstance(test_category, str):
        test_category = [test_category]

    if not skip_generation:
        # Create args namespace for generation
        gen_args = SimpleNamespace(
            model=model,
            test_category=test_category,
            temperature=temperature,
            include_input_log=include_input_log,
            exclude_state_log=exclude_state_log,
            num_gpus=num_gpus,
            num_threads=num_threads,
            gpu_memory_utilization=gpu_memory_utilization,
            backend=backend,
            skip_server_setup=skip_server_setup,
            local_model_path=local_model_path,
            result_dir=result_dir,
            allow_overwrite=allow_overwrite,
            run_ids=run_ids,
            samples_per_category=samples_per_category,
        )
        generation_main(gen_args)

    if not skip_evaluation:
        # Call evaluation_main with required arguments
        evaluation_main(
            model=model,
            test_categories=test_category,
            result_dir=result_dir,
            score_dir=score_dir,
            samples_per_category=samples_per_category
        )

if __name__ == "__main__":
    # Example usage
    #instance = WandbConfigSingleton.get_instance()
    #run = instance.run
    #cfg = instance.config
    #llm = instance.llm
    load_dotenv(dotenv_path=DOTENV_PATH, verbose=True, override=True) # will be removed when integrating with Nejumi Leaderboard

    with wandb.init():
        run(
            model=["gorilla-openfunctions-v2"],
            test_category=["all"],
            temperature=0.001,
            num_gpus=1,
            num_threads=1,
            samples_per_category=1,  # Take 1 sample from each category
        )

