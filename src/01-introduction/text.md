Note: I am writing a simplified English version based on the Chinese version. This is the first chapter.

# Chapter 1: Introduction to GPU and CUDA 

## Introduction to GPU 

GPU means graphics processing unit, which is usually compared to CPU (central processing unit). While a typical CPU has a few relatively fast cores, a typical GPU has hundreds or thousands of relatively slow cores. In a CPU, more transistors are devoted to cache and control; in a GPU, more transistors are devoted to data processing. 

GPU computing is heterogeneous computing, which involves both CPU and GPU, which are usually referred to as host and device, respectively. Both CPU and non-embedded GPU have their own DRAM（dynamic random-access memory), and they are usually connected by a PCIe（peripheral component interconnect express）bus.

We only consider GPUs from Nvidia, becasue CUDA programming only supports these GPUs. There are a few series of Nvidia GPUs:
* Tesla series: good for scientific computing but expensive.
* GeForce series: cheaper but less professional. 
* Quadro series: kind of between the above two.
* Jetson series: embeded device (I have never used these).

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

The computate capability of a GPU is not directly related to its performance. The following table lists the major metrics regarding performance for a few selected GPUs.

| GPU  | compute capability |  memory capacity  |  memory bandwidth  |  double-precision peak FLOPs | single-precision peak FLOPs |
|:------------|:---------------|:--------------|:-----------------|:------------|:------------------|
| Tesla P100         | 6.0 | 16 GB | 732 GB/s | 4.7 FLOPs | 9.3 FLOPs|
| Tesla V100         | 7.0 | 32 GB | 900 GB/s | 7 FLOPs  | 14 FLOPs |
| GeForce RTX 2070   | 7.5 | 8 GB  | 448 GB/s | 0.2 FLOPs| 6.5 FLOPs|
| GeForce RTX 2080ti | 7.5 | 11 GB | 732 GB/s | 0.4 FLOPs| 13 FLOPs|

We notice that the double precision performane of a GeForce GPU is only 1/32 of its single-precision performance.


## Introduction to CUDA 

There are a few tools for GPU computing, including CUDA, OpenCL, and OpenACC, but we only consider CUDA in this book. We also only consider CUDA based on C++, which is called CUDA C++ for short. We will not consider CUDA Fortran.

CUDA provides two APIs Application Programming Interfaces) for devolopers: the CUDA driver API and the CUDA runtime API. The CUDA driver API is more fundamental (low-level) and more flexible. The CUDA runtime API is constructed based on top of the CUDA driver API and is easier to use. We only consider the CUDA runtime API.

There are also many CUDA versions, which can also be represented as `X.Y`. The following table lists the recent CUDA versions and the  the supported compute capabilites.

| CUDA versions | supported GPUs |
|:------------|:---------------|
|CUDA 11.0 |  Compute capability 3.5-8.0 (Kepler to Ampere) |
|CUDA 10.0-10.2 | Compute capability 3.0-7.5 (Kepler to Turing) |
|CUDA 9.0-9.2 | Compute capability 3.0-7.2  (Kepler to Volta) | 
|CUDA 8.0     | Compute capability 2.0-6.2  (Fermi to Pascal) | 

# Installing a CUDA devolopment envirionment

## Linux

Check this manual: https://docs.nvidia.com/cuda/cuda-installation-guide-linux

## Windows 10

* Installing Visual Studio. Go to https://visualstudio.microsoft.com/free-developer-offers/ and download a free Visual Studio (Community version). For our purpose, you only need to install `Desktop development with C++` within the many components of Visual Studio. Of course, you can install more compoents too.

* Installing CUDA. Go to https://developer.nvidia.com/ and choose a Windows CUDA verison and install it. You can choose the highest version that support your GPU.

* After installing both Visual Studio and CUDA (ProgramData folder might be hiden and you can enable to show it), go to the following folder
```
    C:\ProgramData\NVIDIA Corporation\CUDA Samples\v10.1\1_Utilities\deviceQuery  
```
and use Visual Studio to open the solution `deviceQuery_vs2019.sln`. Then build the solution and run the executable. If you see `Result = PASS` at the end of the output, congratulations! If you encountered problems, you can check the manual carefully: https://docs.nvidia.com/cuda/cuda-installation-guide-microsoft-windows.

In this book, we will not use the Visual Studio IDE to develop CUDA programs. Instead, we use the command line (called **terminal** in Linux and **command prompt** in Windows) and, if needed, the `make` program. In Windows, to make the MSVC (Microsoft Visual C++ Compiler) `cl.exe` available, one can follow the following steps to open a command prompt:
```
Windows start -> Visual Studio 2019 -> x64 Native Tools Command Prompt for VS 2019
```
In some cases, we need to have administrator rights, which can be achived by right clicking `x64 Native Tools Command Prompt for VS 2019` and choosing `more` and then `run as administrator`.


# Using the `nvidia-smi` program

After installing CUDA, one should be able to use the program (an executable) `nvidia-smi`（Nvidia's system management interface）from the command line. Simply type the name of this program:
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


\item 作者所用计算机中有一型号为~GeForce RTX 2070~的~GPU。该~GPU~的设备号是~0。该计算机仅有一个~GPU。如果有多个~GPU，会将各个~GPU~从~0~开始编号。如果读者的系统中有多个~GPU，而且只需要使用某个特定的~GPU（比如两个之中更强大的那个），则可以通过设置环境变量~\verb"CUDA_VISIBLE_DEVICES" ~的值在运行~CUDA~程序之前选定一个~GPU。假如读者的系统中有编号为~0~和~1~的两个~GPU，而读者想在~1~号~GPU~运行~CUDA~程序，则可以用如下命令设置环境变量：
\begin{verbatim}
    $ export CUDA_VISIBLE_DEVICES=1        
\end{verbatim}
这样设置的环境变量在当前~shell session~及其子进程中有效。
\item 该~GPU~处于~WDDM（windows display driver model ）模式。另一个可能的模式是~TCC（tesla compute cluster），但它仅在~Tesla、Quadro~和~Titan~系列的~GPU~中可选。可用如下方式选择（在~Windows~中需要用管理员身份打开~Command Prompt~并去掉命令中的~sudo）：
\begin{verbatim}
    $ sudo nvidia-smi -g GPU_ID -dm 0 # 设置为 WDDM 模式
    $ sudo nvidia-smi -g GPU_ID -dm 1 # 设置为 TCC 模式
\end{verbatim}
这里，\verb"GPU_ID"~是~GPU~的编号。
\item 该~GPU~当前的温度为~38~摄氏度。GPU~在满负荷运行时，温度会高一些。
\item 这是~GeForce~系列的~GPU，没有~ECC~内存，故~Uncorr. ECC~为~N/A，代表不适用（not applicable）或者不存在（not available）。
\item Compute M.~指计算模式（compute mode）。该~GPU~的计算模式是~Default。在默认模式中，同一个~GPU~中允许存在多个计算进程，但每个计算进程对应程序的运行速度一般来说会降低。还有一种模式为~E. Process，指的是独占进程模式（exclusive process mode），但不适用于处于~WDDM~模式的~GPU。在独占进程模式下，只能运行一个计算进程独占该~GPU。可以用如下命令设置计算模式（在~Windows~中需要用管理员身份打开~Command Prompt~并去掉命令中的~sudo）：
\begin{verbatim}
    $ sudo nvidia-smi -i GPU_ID -c 0 # 默认模式
    $ sudo nvidia-smi -i GPU_ID -c 1 # 独占进程模式
\end{verbatim}
这里，\verb"-i GPU_ID" 的意思是希望该设置仅仅作用于编号为~\verb"GPU_ID"~的~GPU；如果去掉该选项，该设置将会作用于系统中所有的~GPU。
\end{itemize}

关于~\verb"nvidia-smi"~程序更多的介绍，请参考如下官方文档：
\url{https://developer.nvidia.com/nvidia-system-management-interface}。




# Other learning resources

* The official documentation: https://docs.nvidia.com/cuda
* https://devblogs.nvidia.com
* https://stackoverflow.com


