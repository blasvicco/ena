"""Visualization module for ENA Experiments"""

# General imports
from typing import Callable, TextIO
from queue import Empty, Queue
import threading
import time

# Libs imports
import psutil
from rich.console import Console
from rich.text import Text
from rich.progress import (
	BarColumn,
	Progress,
	SpinnerColumn,
	TextColumn,
	TimeElapsedColumn,
)

# Setup visualization
# Global UI components
console = Console()

# pylint: disable=too-few-public-methods
class ProgressReporter:
	"""Centralized progress reporter for parallel experiments."""

	def __init__(self, queue: Queue | None, total_experiments: int, min_interval: float = 60.0) -> None:
		"""Initialize ProgressReporter."""
		# Off a terminal (a redirected log, a remote run) the live bar never reaches the file, so each status change is also printed as a plain line,
		# at most one per replicate per min_interval seconds (a replicate's end always).
		self.queue = queue
		self.total_experiments = total_experiments
		self.stop_event = threading.Event()
		self.min_interval = min_interval
		self._last_plain = {}

	def _plain(self, experiment_id, description: str, force: bool = False) -> None:
		"""Print one plain, timestamped status line when not on a terminal (throttled per replicate)."""
		if console.is_terminal:
			return
		now = time.monotonic()
		if not force and now - self._last_plain.get(experiment_id, -float("inf")) < self.min_interval:
			return
		self._last_plain[experiment_id] = now
		print(f"{time.strftime('%Y-%m-%d %H:%M:%S')} {Text.from_markup(description).plain}", flush=True)

	def run(self) -> None:  # pylint: disable=too-many-locals
		"""Main loop for the reporter thread."""
		with Progress(
			SpinnerColumn(),
			TextColumn("[progress.description]{task.description}"),
			BarColumn(),
			TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
			TimeElapsedColumn(),
			console=console,
		) as progress:
			# Main task for all experiments
			main_task = progress.add_task(
				"[bold blue]Total Experiments Progress",
				total=self.total_experiments * 100,
			)

			completed_count = 0
			exp_tasks = {}
			exp_progress = {}
			finished_ids = set()

			while completed_count < self.total_experiments and not self.stop_event.is_set():
				memory_info = psutil.virtual_memory()
				memory_string = f"Mem: {memory_info.used / (1024**3):.1f}GB ({memory_info.percent}%)"
				progress.update(
					main_task,
					description=f"[bold blue]Total Experiments Progress[/bold blue] | [magenta]{memory_string}[/magenta]",
				)

				try:
					# Use timeout to allow checking stop_event
					data = self.queue.get(timeout=1.0)
					if data is None:
						break  # Sentinel

					experiment_id = data.get("exp_id")
					status = data.get("status")
					progress_inc = data.get("progress", 0)
					done = data.get("done", False)

					if experiment_id is not None:
						if experiment_id not in exp_tasks:
							exp_tasks[experiment_id] = progress.add_task(f"Experiment {experiment_id + 1}: Starting...", total=100)

						description = None
						if status:
							description = f"Experiment {experiment_id + 1}: {status}"
						elif done:
							description = f"[green]Experiment {experiment_id + 1}: Done[/green]"

						if description:
							progress.update(
								exp_tasks[experiment_id],
								description=description,
							)
							self._plain(experiment_id, description, force=done)

						if progress_inc > 0:
							progress.advance(exp_tasks[experiment_id], progress_inc)
							progress.advance(main_task, progress_inc)
							exp_progress[experiment_id] = exp_progress.get(experiment_id, 0) + progress_inc

						if done and experiment_id not in finished_ids:
							current_p = exp_progress.get(experiment_id, 0)
							remaining = 100 - current_p

							if remaining > 0:
								progress.advance(main_task, remaining)
								exp_progress[experiment_id] = 100

							progress.update(
								exp_tasks[experiment_id],
								completed=100,
							)

							finished_ids.add(experiment_id)
							completed_count += 1
				except Empty:
					continue  # no update within the timeout: check stop_event again
				except (EOFError, BrokenPipeError):
					break  # the study's manager has shut its queue down


def vprint(message: str, file: TextIO | None = None) -> None:
	"""Print to stdout if no file is provided, or only to the file if provided."""
	if file is None:
		console.print(message)
	else:
		print(message, file=file, flush=True)  # a run's outputs.txt is readable while it goes


def post_update(experiment_id: int, queue: Queue | None, status: str = None, progress: int = 0, done: bool = False) -> None:
	"""Post update to the reporting queue."""
	if queue:
		queue.put(
			{
				"done": done,
				"exp_id": experiment_id,
				"progress": progress,
				"status": status,
			}
		)


def format_duration(seconds: float) -> str:
	"""Format a duration in seconds as e.g. '1h 55m 31s', '6m 22s', or '42s'."""
	hours, remainder = divmod(int(seconds), 3600)
	minutes, secs = divmod(remainder, 60)
	if hours:
		return f"{hours}h {minutes}m {secs}s"
	if minutes:
		return f"{minutes}m {secs}s"
	return f"{secs}s"


def make_eta_reporter(experiment_id: int, queue: Queue | None, label: str, agent_name: str, start_time: float) -> Callable[[int, int], None]:
	"""Build a play() on_episode callback that reports a live ETA via the status queue."""
	# Estimated from the average time per episode elapsed so far this phase.

	def _report(episode_num: int, total_episodes: int) -> None:
		"""Post the current ETA, skipping the first and last episode (no rate yet / done)."""
		if episode_num <= 0 or episode_num >= total_episodes:
			return
		elapsed = time.time() - start_time
		remaining = (elapsed / episode_num) * (total_episodes - episode_num)
		eta_str = format_duration(remaining)
		post_update(experiment_id, queue, status=f"{label} {agent_name} (ETA {eta_str})")

	return _report
