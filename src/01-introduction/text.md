Note: I am writing a simplified version of my Chinese CUDA book in English. This is the first chapter.

# Chapter 1: GPU and CUDA 

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
| Tesla P100         | 6.0 | 16 GB | 732 GB/s | 4.7 | 9.3 |
| Tesla V100         | 7.0 | 32 GB | 900 GB/s | 7   | 14  |
| GeForce RTX 2070   | 7.5 | 8 GB  | 448 GB/s | 0.2 | 6.5 |
| GeForce RTX 2080ti | 7.5 | 11 GB | 732 GB/s | 0.4 | 13  |

We notice that the double precision performane of a GeForce GPU is only 1/32 of its single-precision performance.


## Introduction to CUDA 

There are a few tools for GPU computing, including CUDA, OpenCL, OpenACC, but we only consider CUDA in this book. We also only consider CUDA based on C++, which is called CUDA C++ for short. We will not consider CUDA Fortran.


CUDA~提供了两层~API（Application Programming Interface，应用程序编程接口）给程序员使用，即
CUDA 驱动（driver） API和CUDA 运行时（runtime） API。其中，CUDA~驱动~API~是更加底层的~API，它为程序员提供了更为灵活的编程接口；CUDA~运行时~API~是在~CUDA~驱动~API~的基础上构建的一个更为高级的~API，更容易使用。这两种~API~在性能上几乎没有差别。从程序的可读性来看，使用~CUDA~运行时~API~是更好的选择。在其他编程语言中使用~CUDA~的时候，驱动~API~很多时候是必需的。因为作者没有使用驱动~API~的经验，故本书只涉及~CUDA~运行时~API。

图~\ref{figure:cuda_tools}~展示了~CUDA~开发环境的主要组件。开发的应用程序是以主机（CPU）为出发点的。应用程序可以调用~CUDA~运行时~API、CUDA~驱动~API及一些已有的~CUDA~库。所有这些调用都将利用设备（GPU）的硬件资源。对~CUDA~运行时~API~的介绍是本书大部分章节的重点内容；第~\ref{chapter:lib}~章将介绍若干常用的~CUDA~库。

\begin{figure}[ht]
  \centering
  \captionsetup{font=small}
  \includegraphics[width=\columnwidth]{CUDA.pdf}\\
  \caption{CUDA~编程开发环境概览。}
  \label{figure:cuda_tools}
\end{figure}

CUDA~版本也由形如~X.Y~的两个数字表示，但它并不等同于GPU~的计算能力。可以这样理解：CUDA~版本是~GPU~软件开发平台的版本，而计算能力对应着~GPU~硬件架构的版本。

最早的~CUDA 1.0~于~2007~年发布。当前（笔者交稿之日）最新的版本是~CUDA 10.2。CUDA~版本与~GPU~的最高计算能力都在逐年上升。虽然它们之间没有严格的对应关系，但一个具有较高计算能力的~GPU~通常需要一个较高的~CUDA~版本才能支持。最近的几个~CUDA~版本对计算能力的支持情况见表~\ref{table:cuda-versions}。一般来说，建议安装一个支持所用~GPU~的较新的~CUDA~工具箱。本书中的所有示例程序都可以在~CUDA~9.0-10.2~中进行测试。目前最新版本的~CUDA 10.2~有两个值得注意的地方。第一，它是最后一个支持~macOS~系统的~CUDA~版本；第二，它将~CUDA C 改名为~CUDA C++，用以强调~CUDA C++~是基于~C++~的扩展。


\begin{table}[htb]
\centering
\captionsetup{font=small}
\caption{最近的几个~CUDA~版本对~GPU~计算能力的支持情况。}
\begin{tabular}{ccc}
\hline
CUDA~版本 &  所支持~GPU~的计算能力 & 架构 \\
\hline
\hline
10.0-10.2 &  3.0-7.5 & 从开普勒到图灵 \\
\hline
9.0-9.2 &  3.0-7.2  & 从开普勒到伏特 \\
\hline
8.0 &  2.0-6.2  & 从费米到帕斯卡 \\
\hline
7.0-7.5 &  2.0-5.3 & 从费米到麦克斯韦 \\
\hline
\hline
\end{tabular}
\label{table:cuda-versions}
\end{table}


\section{CUDA~开发环境搭建示例}

下面叙述作者最近在装有~GeForce RTX 2070~的便携式计算机（以下简称计算机）中搭建~CUDA~开发环境的大致过程。因为作者的计算机中预装了~Windows 10~操作系统，所以我们以~Windows 10~操作系统为例进行讲解。因为~Linux~发行版有多种，故本书不列出在~Linux~中安装~CUDA~开发环境的步骤。读者可参阅~Nvidia~的官方文档：\url{https://docs.nvidia.com/cuda/cuda-installation-guide-linux}。

我们说过，GPU~计算实际上是~CPU+GPU（主机+设备）的异构计算。在~CUDA C++~程序中，既有运行于主机的代码，也有运行于设备的代码。其中，运行于主机的代码需要由主机的~C++~编译器编译和链接。所以，除了安装~CUDA~工具箱，还需要安装一个主机的~C++~编译器。在~Windows~中，最常用的~C++~编译器是~Microsoft Visual C++ （MSVC），它目前集成在~Visual Studio~中，所以我们首先安装~Visual Studio。作者安装了最高版本的~Visual Studio 2019 16.x。因为这是个人使用的，故选择了免费的~Community~版本。下载地址为
~\url{https://visualstudio.microsoft.com/free-developer-offers/}。对于~CUDA C++程序开发来说，只需要选择安装~Desktop development with C++~即可。当然，读者也可以选择安装更多的组件。

关于~CUDA，作者选择安装~2019~年8月发布的~CUDA Toolkit 10.1 update2。首先，进入网址~\url{https://developer.nvidia.com/cuda-10.1-download-archive-update2}。然后根据提示，做如下选择：Operating System~项选择~Windows；Architecture~项选择~\verb"x86_64"；Version~项选择操作系统版本，我们这里是~10；Installer Type~项可以选择~exe (network)~或者~exe (local)，分别代表一边下载一边安装和下载完毕后安装。接着，运行安装程序，根据提示一步一步安装即可。该版本的~CUDA~工具箱包含一个对应版本的~Nvidia driver，故不需要再单独安装~Nvidia driver。

安装好~Visual Studio~和~CUDA~后，进入到如下目录（读者如果找不到~C~盘下的~ProgramData~目录，可能是因为没有选择显示一些隐藏的文件）：
\begin{small}
\begin{verbatim}
    C:\ProgramData\NVIDIA Corporation\CUDA Samples\v10.1\1_Utilities\deviceQuery    
\end{verbatim}
\end{small}
然后，用~Visual Studio 2019~打开文件~\verb"deviceQuery_vs2019.sln"。接下来，编译（构建）、运行。若输出内容的最后部分为~\verb"Result = PASS"，则说明已经搭建好~Windows~中的~CUDA~开发环境。若有疑问，请参阅~Nvidia~的官方文档：\url{https://docs.nvidia.com/cuda/cuda-installation-guide-microsoft-windows}。

在上面的测试中，我们是直接用~Visual Studio~打开一个已有的解决方案（solution），然后直接构建并运行。本书不介绍~Visual Studio~的使用，而是选择用命令行解释器编译与运行程序。这里的命令行解释器指的是~Linux~中的~terminal~或者~Windows~中的~command prompt~程序。在~Windows~中使用~MSVC~作为~C++~程序的编译器时，需要单独设置相应的环境变量，或者从~Windows~的开始（start）菜单中找到~Visual Studio 2019~文件夹，然后单击其中的“x64 Native Tools Command Prompt for VS 2019”，而从打开一个加载了~MSVC~环境变量的命令行解释器。在本书的某些章节，需要有管理员的权限来使用~nvprof~性能分析器。此时，可以右击“x64 Native Tools Command Prompt for VS 2019”，然后选择“更多”，接着选择“以管理员身份运行”。

用命令行解释器编译与运行~CUDA~程序的方式在~Windows~和~Linux~系统几乎没有区别，但为了简洁起见，本书后面主要以~Linux~开发环境为例进行讲解。虽然如此，Windows~和~Linux~中的~CUDA~编程功能还是稍有差别。我们将在后续章节中适当的地方指出这些差别。

\section{用~nvidia-smi~检查与设置设备}

可以通过~\verb"nvidia-smi"（Nvidia's system management interface）程序检查与设置设备。它包含在~CUDA~开发工具套装内。该程序最基本的用法就是在命令行解释器中使用不带任何参数的命令~\verb"nvidia-smi"。在作者的计算机中使用该命令，得到如下文本形式的输出：
\begin{small}
\begin{verbatim}
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
\end{verbatim}
\end{small}

从中可以看出一些比较有用的信息：
\begin{itemize}
\item 第一行可以看到~Nvidia driver~的版本及~CUDA~工具箱的版本。
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



\section{其他学习资料}

本书将循序渐进地带领读者学习~CUDA C++~编程的基础知识。虽然本书力求自给自足，但读者在阅读本书的过程中同时参考一些其他的学习资料也是有好处的。

任何关于~CUDA 编程的书籍都不可能替代官方提供的手册等资料。以下是几个重要的官方文档，请读者在有一定的基础之后务必查阅。限于作者水平，本书难免存在谬误。当读者觉得本书中的个别论断与官方资料有冲突时，当以官方资料为标准（官方手册的网址为~\url{https://docs.nvidia.com/cuda}）。在这个网站，包括但不限于以下几个方面的文档：
\begin{itemize}
\item 安装指南（installation guides）。读者遇到与~CUDA~安装有关的问题时，应该仔细阅读此处的文档。
\item 编程指南（programming guides）。该部分有很多重要的文档：
\begin{itemize}
    \item 最重要的文档是《CUDA C++ Programming Guide》，见以下网址：
    \url{https://docs.nvidia.com/cuda/cuda-c-programming-guide}。
    \item 另一个值得一看的文档是《CUDA C++ Best Practices Guide》，见以下网址：
    \url{https://docs.nvidia.com/cuda/cuda-c-best-practices-guide}。
    \item 针对最近的几个~GPU~架构进行优化的指南，包括以下网址：
    \begin{itemize}
        \item \url{https://docs.nvidia.com/cuda/kepler-tuning-guide}。
        \item \url{https://docs.nvidia.com/cuda/maxwell-tuning-guide}。
        \item \url{https://docs.nvidia.com/cuda/pascal-tuning-guide}。
        \item \url{https://docs.nvidia.com/cuda/volta-tuning-guide}。
        \item \url{https://docs.nvidia.com/cuda/turing-tuning-guide}。
    \end{itemize}
    这几个简短的文档可以帮助有经验的用户迅速了解一个新的架构。
\end{itemize}
\item CUDA API 手册（CUDA API references）。这里有：
\begin{itemize}
\item CUDA~运行时~API~的手册：\url{https://docs.nvidia.com/cuda/cuda-runtime-api}。
\item CUDA~驱动~API~的手册：\url{https://docs.nvidia.com/cuda/cuda-driver-api}。
\item CUDA~数学函数库~API~的手册：\url{https://docs.nvidia.com/cuda/cuda-math-api}
\item 其他若干~CUDA~库的手册。
\end{itemize}

\end{itemize}

为明确起见，在撰写本书时，作者参考的是与~CUDA 10.2~对应的官方手册。

在学习~CUDA~编程的过程中如果遇到了某些难以解决的问题，可以考虑去论坛或者交流群求助。以下是几个比较有用的学习资源：

\begin{itemize}
\item 国内方面，作者推荐如下网站与交流群：
\begin{itemize}
    \item 《GPU~世界》论坛： \url{https://bbs.gpuworld.cn}。
    \item 《GPU~编程开发技术》QQ~群：62833093。
    \item 《CUDA 100\%》QQ~群：195055206。
    \item 《CUDA Professional》QQ~群：45157483。
\end{itemize}
\item 国际方面，作者推荐如下网站：
\begin{itemize}
    \item 英伟达官方的开发者博客： \url{https://devblogs.nvidia.com}。
    \item Stack Overflow 问答网站：
    \url{https://stackoverflow.com}。
\end{itemize}
\end{itemize}


