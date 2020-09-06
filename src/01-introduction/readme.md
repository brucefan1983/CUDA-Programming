# Chapter 1: Introduction to GPU and CUDA 

## 1.1 Introduction to GPU 

GPU means `graphics processing unit`, which is usually compared to CPU (central processing unit). While a typical CPU has a few relatively fast cores, a typical GPU has hundreds or thousands of relatively slow cores. In a CPU, more transistors are devoted to cache and control; in a GPU, more transistors are devoted to data processing. 

GPU computing is `heterogeneous computing`. It involves both CPU and GPU, which are usually referred to as `host` and `device`, respectively. Both CPU and non-embedded GPU have their own DRAM (dynamic random-access memory), and they are usually connected by a PCIe（peripheral component interconnect express）bus.

We only consider GPUs from Nvidia, because CUDA programming only supports these GPUs. There are a few series of Nvidia GPUs:
* Tesla series: good for scientific computing but expensive.
* GeForce series: cheaper but less professional. 
* Quadro series: kind of between the above two.
* Jetson series: embedded device (I have never used these).

Every GPU has a version number `X.Y` to indicate its **compute capability**. Here, `X` is a major version number, and `Y` is a minor version number. A major version number corresponds to a major GPU architecture and is also named after a famous scientist. See the following table.

| Major compute capability  | architecture name |   release year  |
|:------------|:---------------|:--------------|
| `X=1` | Tesla | 2006 |
| `X=2` | Fermi | 2010 |
| `X=3` | Kepler | 2012 |
| `X=5` | Maxwell | 2014 |
| `X=6` | Pascal | 2016 |
| `X=7` | Volta | 2017 |
| `X.Y=7.5` | Turing | 2018 |
| `X=8` | Ampere | 2020 |

GPUs older than Pascal will become deprecated soon. We will focus on GPUs no older than Pascal.

The compute capability of a GPU is not directly related to its performance. The following table lists the major metrics regarding performance for a few selected GPUs.

| GPU  | compute capability |  memory capacity  |  memory bandwidth  |  double-precision peak FLOPs | single-precision peak FLOPs |
|:------------|:---------------|:--------------|:-----------------|:------------|:------------------|
| Tesla P100         | 6.0 | 16 GB | 732 GB/s | 4.7 TFLOPs | 9.3 TFLOPs|
| Tesla V100         | 7.0 | 32 GB | 900 GB/s | 7 TFLOPs  | 14 TFLOPs |
| GeForce RTX 2070   | 7.5 | 8 GB  | 448 GB/s | 0.2 TFLOPs| 6.5 TFLOPs|
| GeForce RTX 2080ti | 7.5 | 11 GB | 732 GB/s | 0.4 TFLOPs| 13 TFLOPs|

We notice that the double precision performance of a GeForce GPU is only 1/32 of its single-precision performance.


## 1.2 Introduction to CUDA 

There are a few tools for GPU computing, including CUDA, OpenCL, and OpenACC, but we only consider CUDA in this book. We also only consider CUDA based on C++, which is called CUDA C++ for short. We will not consider CUDA FORTRAN.

CUDA provides two APIs (Application Programming Interfaces) for developers: the **CUDA driver API** and the **CUDA runtime API**. The CUDA driver API is more fundamental (low-level) and more flexible. The CUDA runtime API is constructed based on the CUDA driver API and is easier to use. We only consider the CUDA runtime API.

There are also many CUDA versions, which can also be represented as `X.Y`. The following table lists a few recent CUDA versions and the supported compute capabilities.

| CUDA versions | supported GPUs |
|:------------|:---------------|
|CUDA 11.0 |  Compute capability 3.5-8.0 (Kepler to Ampere) |
|CUDA 10.0-10.2 | Compute capability 3.0-7.5 (Kepler to Turing) |
|CUDA 9.0-9.2 | Compute capability 3.0-7.2  (Kepler to Volta) | 
|CUDA 8.0     | Compute capability 2.0-6.2  (Fermi to Pascal) | 

## 1.3 Installing CUDA 

For Linux, check this manual: https://docs.nvidia.com/cuda/cuda-installation-guide-linux

For Windows, one needs to install both CUDA and Visual Studio:

* Installing Visual Studio. Go to https://visualstudio.microsoft.com/free-developer-offers/ and download a free Visual Studio (Community version). For the purpose of this book, we only need to install `Desktop development with C++` within the many components of Visual Studio. 

* Installing CUDA. Go to https://developer.nvidia.com/cuda-downloads and choose a Windows CUDA version and install it. You can choose the highest version that supports your GPU.

* After installing both Visual Studio and CUDA (The `ProgramData` folder might be hiden and you can enable to show it), go to the following folder
```
    C:\ProgramData\NVIDIA Corporation\CUDA Samples\v10.1\1_Utilities\deviceQuery  
```
and use Visual Studio to open the solution `deviceQuery_vs2019.sln`. Then build the solution and run the executable. If you see `Result = PASS` at the end of the output, congratulations! If you encountered problems, you can check the manual carefully: https://docs.nvidia.com/cuda/cuda-installation-guide-microsoft-windows.

In this book, we will not use the Visual Studio IDE to develop CUDA programs. Instead, we use the **command line** (called **terminal** in Linux and **command prompt** in Windows) and, if needed, the `make` program. In Windows, to make the MSVC (Microsoft Visual C++ Compiler) `cl.exe` available, one can follow the following steps to open a command prompt:
```
Windows start -> Visual Studio 2019 -> x64 Native Tools Command Prompt for VS 2019
```
In some cases, we need to have administrator rights, which can be achieved by right clicking `x64 Native Tools Command Prompt for VS 2019` and choosing `more` and then `run as administrator`.


## 1.4 Using the `nvidia-smi` program

After installing CUDA, one should be able to use the program (an executable) `nvidia-smi`（**Nvidia's system management interface**）from the command line. Simply type the name of this program:
```
$ nvidia-smi
```
it will show the information regarding the GPU(s) in the system. Here is an example from my laptop:
```
    +-----------------------------------------------------------------------------+
    | NVIDIA-SMI 426.00       Driver Version: 426.00       CUDA Version: 10.1     |
    |-------------------------------+----------------------+----------------------+
    | GPU  Name            TCC/WDDM | Bus-Id        Disp.A | Volatile Uncorr. ECC |
    | Fan  Temp  Perf  Pwr:Usage/Cap|         Memory-Usage | GPU-Util  Compute M. |
    |===============================+======================+======================|
    |   0  GeForce RTX 207... WDDM  | 00000000:01:00.0 Off |                  N/A |
    | N/A   38C    P8    12W /  N/A |    161MiB /  8192MiB |      0%      Default |
    +-------------------------------+----------------------+----------------------+

    +-----------------------------------------------------------------------------+
    | Processes:                                                       GPU Memory |
    |  GPU       PID   Type   Process name                             Usage      |
    |=============================================================================|
    |  No running processes found                                                 |
    +-----------------------------------------------------------------------------+
```

Here are some useful information from the above outputs:
* From line 1 we can see the Nvidia driver version (426.00) and the CUDA version (10.1).
* There is only one GPU in the system, which is a GeForce RTX 2070. It has a device ID of 0. If there are more GPUs, they will be labeled starting from 0. You can use the following command in the command line to select to use device 1 before running a CUDA program:
```
$ export CUDA_VISIBLE_DEVICES=1        
```
* This GPU is on the **WDDM（windows display driver model）** mode. Another possible mode is **TCC（tesla compute cluster)**, but it is only avaible for GPUs in the Tesla, Quadro, and Titan series. One can use the following commands to choose the mode（In Windows, one needs to have administrator rights and `sudo` below should be removed):
```
$ sudo nvidia-smi -g GPU_ID -dm 0 # set device GPU_ID to the WDDM mode
$ sudo nvidia-smi -g GPU_ID -dm 1 # set device GPU_ID to the TCC mode
```
* `Compute M`. refers to compute mode. Here the compute mode is `Default`, which means that multiple computing process are allowed to be run with the GPU. Another possible mode is `E. Process`，which means exclusive process mode. The `E. Process` mode is not available for GPUs in the WDDM mode。One can use the following commands to choose the mode（In Windows, one needs to have administrator rights and `sudo` below should be removed):
```
$ sudo nvidia-smi -i GPU_ID -c 0 # set device GPU_ID to Default mode
$ sudo nvidia-smi -i GPU_ID -c 1 # set device GPU_ID to E. Process mode
```

For more details of the `nvidia-smi` program, see the following official manual: https://developer.nvidia.com/nvidia-system-management-interface

