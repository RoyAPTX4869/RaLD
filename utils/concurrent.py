import multiprocessing as mp
from multiprocessing.pool import AsyncResult
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from queue import Empty
from functools import partial
from os import PathLike
from queue import Queue
import os
from os.path import isfile
from os.path import isfile
from typing import List, Any, Iterable, Union, Literal
from subprocess import Popen, PIPE, STDOUT, DEVNULL, run
import shlex
import logging
import logging.handlers
import traceback
from tqdm import tqdm



def shell_cmd(command_line: str, log_to: Union[PathLike, str] = Literal['stdout'], maxBytes: int = 10**9, timeout: float = None, **kwargs):
	command_line_args = shlex.split(command_line)
	if isinstance(log_to, str): 
		if log_to == 'stdout':
			exitcode = run(command_line_args, timeout=timeout, **kwargs).returncode
		else:
			raise NotImplementedError
	elif isinstance(log_to, PathLike):
		# create a file logger with max byte limit
		test_logger = logging.getLogger('SHELL_CMD')
		# overwrite manually https://stackoverflow.com/a/77223050
		if isfile(log_to): os.remove(log_to)
		maybe_backup_file = str(log_to) + '.1'
		if isfile(maybe_backup_file): os.remove(maybe_backup_file)
		# log cmd to file
		with open(log_to, 'w') as f: f.write(command_line + '\n')
		# limit bytes https://stackoverflow.com/a/3999638 & https://stackoverflow.com/a/48095853
		handler = logging.handlers.RotatingFileHandler(log_to, maxBytes=maxBytes, backupCount=1)
		test_logger.setLevel(logging.INFO)
		test_logger.addHandler(handler)
		# run command, merge stderr to stdout and redirect to pipe https://stackoverflow.com/a/21978778
		process = Popen(command_line_args, stdout=PIPE, stderr=STDOUT, **kwargs)
		with process.stdout:
			for line in iter(process.stdout.readline, b''): # b'\n'-separated lines
				test_logger.info(line.strip().decode('utf-8'))
		exitcode = process.wait(timeout=timeout) # 0 means success
		test_logger.removeHandler(handler)
		handler.close() # release resources
	else: # silent
		process = Popen(command_line_args, stdout=DEVNULL, stderr=DEVNULL, **kwargs)
		exitcode = process.wait(timeout=timeout) # 0 means success
	return exitcode

class AsyncWorkerPool:
	def __init__(self, worker_num: int, worker_type: Literal['process', 'thread'] = 'process', mp_method: Literal['fork', 'spawn', 'forkserver'] = 'spawn', **kwargs): # TODO
		"""Create a new asyncornized process pool.

		Args:
			process_num (int): number of processes
			worker_type (Literal['process', 'thread'], optional): type of worker. Defaults to 'process'.
			kwargs (dict): additional arguments for ProcessPoolExecutor or ThreadPoolExecutor
		"""
		self.process_num = worker_num
		if worker_type == 'process':
			self.pool = mp.get_context(mp_method).Pool(processes=self.process_num, **kwargs)
			self.worker_type = 'process'
		else:
			self.pool = ThreadPoolExecutor(max_workers=self.process_num, **kwargs)
			self.worker_type = 'thread'
		self.results = []

	@staticmethod
	def _proc_err_callback(e: Exception):
		# print exception stack
		traceback.print_exception(type(e), e, e.__traceback__)
		# pass

	def add_task(self, func: callable, *args, **kwargs) -> AsyncResult:
		"""Add a new task to the task queue."""
		if self.worker_type == 'process':
			async_result = self.pool.apply_async(func, args=args, kwds=kwargs, error_callback=self._proc_err_callback)
		else:
			async_result = self.pool.submit(func, *args, **kwargs)
		self.results.append(async_result)
		return async_result

	def wait_for_results(self) -> List[Any]:
		"""Get all results and close the processes."""  
		if self.worker_type == 'process':
			results = [async_result.get() for async_result in self.results]
		else:
			results = [async_result.result() for async_result in self.results]
		self.results = []
		return results

	def wait_for_task_result(self, async_result: AsyncResult) -> Any:
		if self.worker_type == 'process':
			res = async_result.get()
		else:
			res = async_result.result()
		self.results.remove(async_result)
		return res

	def get_one_result(self, block: bool = True) -> Any:
		while True:
			for async_result in self.results:
				done = async_result.ready() if self.worker_type == 'process' else async_result.done()
				if done:
					self.results.remove(async_result)
					return async_result, async_result.get() if self.worker_type == 'process' else async_result.result()
			else:
				if not block: raise ValueError('no ready result')

	def close(self) -> None:
		"""Close the process pool."""
		if self.worker_type == 'process':
			self.pool.close()
			self.pool.join()
			self.pool.terminate()
		else:
			self.pool.shutdown()


def clear_queue(q: Union[Queue, mp.Queue]):
	# https://stackoverflow.com/a/18873213/14792316
	# with q.mutex:
	# 	q.queue.clear()
	# 	q.all_tasks_done.notify_all()
	# 	q.unfinished_tasks = 0
	try:
		while True:
			q.get_nowait()
	except Empty:
		pass


def clear_pipe(conn):
	while True:
		# Use select to check if there is data available for reading
		if conn.poll():
			_ = conn.recv()
		else:
			# If no data is available, break the loop
			break


def imap_tqdm(  function, 
				iterable: Iterable, 
				processes: int, 
				chunksize: int = 1, 
				desc: bool = None, 
				disable: bool = False, 
				maxtasksperchild: int = None,
				mp_method: Literal['fork', 'spawn', 'forkserver'] = 'fork',
			  **kwargs) -> List[Any]:
	"""
	https://stackoverflow.com/a/73635314/14792316
	Run a function in parallel with a tqdm progress bar and an arbitrary number of arguments.
	Results are always ordered and the performance should be the same as of Pool.map.
	:param function: The function that should be parallelized.
	:param iterable: The iterable passed to the function.
	:param processes: The number of processes used for the parallelization.
	:param chunksize: The iterable is based on the chunk size chopped into chunks and submitted to the process pool as separate tasks.
	:param desc: The description displayed by tqdm in the progress bar.
	:param disable: Disables the tqdm progress bar.
	:param kwargs: Any additional arguments that should be passed to the function.
	"""
	if kwargs:
		function_wrapper = partial(_wrapper, function=function, **kwargs)
	else:
		function_wrapper = partial(_wrapper, function=function)

	results = [None] * len(iterable)
	with mp.get_context(mp_method).Pool(processes=processes, maxtasksperchild=maxtasksperchild) as p:
	# with Pool(processes=processes, maxtasksperchild=maxtasksperchild) as p:
		with tqdm(desc=desc, total=len(iterable), disable=disable) as pbar:
			for i, result in p.imap_unordered(function_wrapper, enumerate(iterable), chunksize=chunksize):
				results[i] = result
				pbar.update()
	return results


def _wrapper(enum_iterable, function, **kwargs):
	i = enum_iterable[0]
	result = function(enum_iterable[1], **kwargs)
	return i, result


def _error_handle_wrapper(function, params):
	try:
		return function(params)
	except Exception as e:
		print(traceback.format_exc())
		raise e

def imap_tqdm_open3d(  function, 
						iterable: Iterable, 
						processes: int, 
						maxtasksperchild: int = None,
						**kwargs) -> List[Any]:
	ctx = mp.get_context('forkserver')
	with ProcessPoolExecutor(max_workers=processes, mp_context=ctx, max_tasks_per_child=maxtasksperchild, **kwargs) as p:
		results = list(p.map(_error_handle_wrapper, [function] * len(iterable), iterable))
	return results