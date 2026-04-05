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
				total=self.total_experiments,
			)

			completed_count = 0
			exp_tasks = {}
			finished_ids = set()

			while (
				completed_count < self.total_experiments
				and not self.stop_event.is_set()
			):
				mem = psutil.virtual_memory()
				mem_str = f"Mem: {mem.used / (1024**3):.1f}GB ({mem.percent}%)"
				progress.update(
					main_task,
					description=f"[bold blue]Total Experiments Progress[/bold blue] | [magenta]{mem_str}[/magenta]",
				)

				try:
					# Use timeout to allow checking stop_event
					data = self.queue.get(timeout=1.0)
					if data is None:
						break  # Sentinel

					_exp_id = data.get("exp_id")
					status = data.get("status")
					progress_inc = data.get("progress", 0)
					done = data.get("done", False)

					if _exp_id is not None:
						if _exp_id not in exp_tasks:
							exp_tasks[_exp_id] = progress.add_task(
								f"Experiment {_exp_id + 1}: Starting...", total=100
							)

						description = None
						if status:
							description = f"Experiment {_exp_id + 1}: {status}"
						elif done:
							description = f"[green]Experiment {_exp_id + 1}: Done[/green]"

						if description:
							progress.update(
								exp_tasks[_exp_id],
								description=description,
							)

						if progress_inc > 0:
							progress.advance(exp_tasks[_exp_id], progress_inc)

						if done and _exp_id not in finished_ids:
							progress.update(
								exp_tasks[_exp_id],
								completed=100,
							)
							progress.advance(main_task)
							finished_ids.add(_exp_id)
							completed_count += 1
				except:  # pylint: disable=bare-except
					continue


def vprint(message, file=None):
	"""Print to stdout if no file is provided, or only to the file if provided."""
	if file is None:
		console.print(message)
	else:
		print(message, file=file)
