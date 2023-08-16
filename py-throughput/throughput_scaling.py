from __future__ import annotations

import functools
import os
import sys
import typing as t
import time

import mpi4py as MPI
import numpy as np
import smartredis

if t.TYPE_CHECKING:
    import numpy.typing as npt

    PR = t.ParamSpec("PR")
    T = t.TypeVar("T")


def get_iterations() -> int:
    try:
        return int(os.getenv("SS_ITERATIONS", "100"))
    except ValueError:
        return 100


def get_cluster_flag() -> bool:
    try:
        return bool(int(os.getenv("SS_CLUSTER", "0")))
    except ValueError:
        return False


def my_timeit(fn: t.Callable[PR, T], /) -> t.Callable[PR, tuple[float, T]]:
    @functools.wraps(fn)
    def wrapper(*a: PR.args, **kw: PR.kwargs) -> tuple[float, T]:
        # start = MPI.Wtime()
        start = time.monotonic()
        ret = fn(*a, **kw)
        # delta = MPI.Wtime() - start
        delta = time.monotonic() - start
        return delta, ret
    return wrapper


# hooks to replace w/ alt backend >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>

if t.TYPE_CHECKING:
    TClient: t.TypeAlias = smartredis.Client


@my_timeit
def construct_client() -> TClient:
    cluster = get_cluster_flag()
    return smartredis.Client(cluster=cluster)


@my_timeit
def put_tensor(client: TClient, key: str, array: npt.NDArray[t.Any]) -> None:
    client.put_tensor(key, array)


@my_timeit
def get_tensor(client: TClient, key: str) -> npt.NDArray[t.Any]:
    return client.get_tensor(key)


# <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<


@my_timeit
def run_throughput(timing_file: t.TextIO, n_bytes: int) -> None:
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()

    if rank == 0:
        print("Connecting clients", flush=True)

    delta_t, client = construct_client()
    timing_file.write(f"{rank},clinet(),{delta_t}\n")

    comm.Barrier()

    sizeof_int = 4
    n_values = n_bytes // sizeof_int
    array = np.array([float(i) for i in range(n_values)])

    put_tensor_times: list[float] = []
    get_tensor_times: list[float] = []

    iterations = get_iterations()

    comm.Barrier()

    # TODO: Move this loop to use the mpi timing dec
    # loop_start = MPI.Wtime()
    loop_start = time.monotonic()

    # Keys are overwritten in order to help
    # ensure that the database does not run out of memory
    # for large messages.
    for i in range(iterations):
        key = f"throughput_rank_{rank}"
        delta_t, _ = put_tensor(client, rank, array)
        put_tensor_times.append(delta_t)

        delta_t, _ = get_tensor(client, key)
        get_tensor_times.append(delta_t)

    # loop_end = MPI.Wtime()
    loop_end = time.monotonic()
    loop_t = loop_end - loop_start

    for put_t, get_t in zip(put_tensor_times, get_tensor_times):
        timing_file.write(f"{rank},put_tensor,{put_t}\n")
        timing_file.write(f"{rank},unpack_tensor,{get_t}\n")

    timing_file.write(f"{rank},loop_time,{loop_t}\n")
    timing_file.flush()


def main() -> int:
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()

    if len(sys.argv) == 1:
        raise RuntimeError(
            "The number tensor size in "
            "bytes must be provided as "
            "a command line argument."
        )
    n_bytes = int(sys.argv[1])

    if rank == 0:
        print(f"Running throughput scaling test with tensor size of {n_bytes} bytes")

    with open(f"rank_{rank}_timing.csv", "w") as timing_file:
        delta_t, _ = run_throughput(timing_file, n_bytes)
        if rank == 0:
            print("Finished throughput test", flush=True)
        timing_file.write(f"{rank},main(),{delta_t}\n")
        timing_file.flush()

    MPI.Finalize()
    return 0


if __name__ == "__main__":
    sys.exit(main())
