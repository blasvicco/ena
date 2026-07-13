"""Visualization module for ENA Experiments"""

# General imports
import threading

# Libs imports
import psutil
from rich.console import Console
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

	def __init__(self, queue, total_experiments):
		"""Initialize ProgressReporter"""
		self.queue = queue
		self.total_experiments = total_experiments
		self.stop_event = threading.Event()

	def run(self):
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

			while (
				completed_count < self.total_experiments
				and not self.stop_event.is_set()
			):
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
							exp_tasks[experiment_id] = progress.add_task(
								f"Experiment {experiment_id + 1}: Starting...", total=100
							)

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
				except:  # pylint: disable=bare-except
					continue


def vprint(message, file=None):
	"""Print to stdout if no file is provided, or only to the file if provided."""
	if file is None:
		console.print(message)
	else:
		print(message, file=file)
