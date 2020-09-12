\chapter{获得~GPU~加速的关键\label{chapter:speedup}}

前几章主要关注程序的正确性，没有强调程序的性能（执行速度）。从本章起，我们开始关注~CUDA~程序的性能。在开发~CUDA~程序时往往要验证某些改变是否提高了程序的性能，这就需要对程序进行比较精确的计时。所以，下面我们就从给主机和设备函数的计时讲起。

\section{用CUDA事件计时\label{section:timing}}

在~C++~中，有多种可以对一段代码进行计时的方法，包括使用~GCC~和~MSVC~都有的~\verb"clock"~函数和与头文件~\verb"<chrono>"~对应的时间库、GCC~中的~\verb"gettimeofday"~函数及~MSVC~中的~\verb"QueryPerformanceCounter"~和~\verb"QueryPerformanceFrequency"~函数等。CUDA~提供了一种基于~CUDA~事件（CUDA event）的计时方式，可用来给一段~CUDA~代码（可能包含了主机代码和设备代码）计时。为简单起见，我们这里仅介绍基于~CUDA~事件的计时方法。Listing~\ref{listing:timing}~给出了使用~CUDA~事件对一段代码进行计时的方式。

\begin{lstlisting}[language=C++,caption={本书中常用的计时方式。},label={listing:timing}]
cudaEvent_t start, stop;
CHECK(cudaEventCreate(&start));
CHECK(cudaEventCreate(&stop));
CHECK(cudaEventRecord(start));
cudaEventQuery(start); // 此处不能用 CHECK 宏函数（见第 4 章的讨论）

需要计时的代码块

CHECK(cudaEventRecord(stop));
CHECK(cudaEventSynchronize(stop));
float elapsed_time;
CHECK(cudaEventElapsedTime(&elapsed_time, start, stop));
printf("Time = %g ms.\n", elapsed_time);

CHECK(cudaEventDestroy(start));
CHECK(cudaEventDestroy(stop));
\end{lstlisting}

下面是对该计时方式的解释：
\begin{itemize}
\item 第1行定义了两个CUDA事件类型（\verb"cudaEvent_t"）的变量~\verb"start"~和~\verb"stop"，第2行和第3行用cudaEventCreate函数初始化它们。
\item 第~4~行将~\verb"start"~传入~cudaEventRecord~函数，在需要计时的代码块之前记录一个代表开始的事件。
\item 第~5~行对处于~TCC~驱动模式的~GPU~来说可以省略，但对处于~WDDM~驱动模式的~GPU~来说必须保留。这是因为，在处于~WDDM~驱动模式的~GPU~中，一个~CUDA~流（CUDA stream）中的操作（如这里的~cudaEventRecord~函数）并不是直接提交给~GPU~执行，而是先提交到一个软件队列，需要添加一条对该流的~cudaEventQuery~操作（或者~cudaEventSynchronize）刷新队列，才能促使前面的操作在~GPU~执行。关于~CUDA~流，会在第~\ref{chapter:cuda-stream}~章详细讨论，读者暂时不必对此深究。
\item 第~7~行代表一个需要计时的代码块，它可以是一段主机代码（如对一个主机函数的调用），也可以是一段设备代码（如对一个核函数的调用），还可以是一段混合代码。
\item 第~9~行将~\verb"stop"~传入~cudaEventRecord~函数，在需要计时的代码块之后记录一个代表结束的事件。
\item 第~10~行的~cudaEventSynchronize~函数让主机等待事件~\verb"stop"~被记录完毕。
\item 第~11-13~行调用~cudaEventElapsedTime~函数计算~\verb"start"~和~\verb"stop"~这两个事件之间的时间差（单位是ms）并输出到屏幕。
\item 第~15-16~行调用~cudaEventDestroy~函数销毁~\verb"start"~和~\verb"stop"~这两个~CUDA~事件。这是本书中唯一使用~CUDA~事件的地方，故这里不对~CUDA~事件做进一步讨论。下面，我们对前两章讨论过的数组相加程序进行计时。
\end{itemize}


\subsection{为~C++~程序计时}

先考虑~C++~版本的程序。本章的程序~\verb"add1cpu.cu"~是在第~\ref{chapter:framework}~章的程序~\verb"add.cpp"~的基础上改写的，主要有如下3个方面的改动：
\begin{enumerate}
\item 即使该程序中没有使用核函数，我们也将源文件的后缀名改成了~\verb".cu"，这样就不用包含一些~CUDA~头文件了。若用~\verb".cpp"~后缀，用~\verb"nvcc"~编译时需要明确地增加一些头文件的包含，用~\verb"g++"~编译时还要明确地链接一些~CUDA~库。
\item 从本章起，我们用条件编译的方式选择程序中所用浮点数的精度。在程序的开头部分，有如下几行代码：
\begin{verbatim}
#ifdef USE_DP
    typedef double real;
    const real EPSILON = 1.0e-15;
#else
    typedef float real;
    const real EPSILON = 1.0e-6f;
#endif    
\end{verbatim}
当宏~\verb"USE_DP"~有定义时，程序中的~\verb"real"~就代表~\verb"double"，否则代表~\verb"float"。该宏可以通过编译选项定义（具体见后面的编译命令）。
\item 我们用CUDA事件对该程序中函数~\verb"add"~的调用进行了计时，而且重复了11次。我们忽略第一次测得的时间，因为第一次计算时，机器（无论是CPU还是GPU）都可能处于预热状态，测得的时间往往偏大。我们根据后10次测试的时间计算一个平均值。具体细节见本章的程序~\verb"add1cpu.cu"。
\end{enumerate}

我们依然用~\verb"nvcc"~编译程序。这里，有几个编译选项值得注意。首先，C++~程序的性能显著地依赖于优化选项。我们将总是用~\verb"-O3"~选项。然后，正如前面提到过的，我们可以用条件编译的方式来选择程序中浮点数的精度。具体地说，如果将~\verb"-DUSE_DP"~加入编译选项，程序中的宏~\verb"USE_DP"~将有定义，从而使用双精度浮点数，否则使用单精度浮点数。最后，对本例来说~\verb"GPU"~架构的指定是无关紧要的，但还是可以指定一个具体的架构选项。

我们首先用如下命令编译程序：
\begin{verbatim}
    $ nvcc -O3 -arch=sm_75 add1cpu.cu
\end{verbatim}
这将得到一个使用单精度浮点数的可执行文件。运行该可执行文件，程序将输出其中的~\verb"add"~函数所花的时间。在作者的计算机中，该主机函数耗时约60 ms。然后，我们用如下命令编译程序：
\begin{verbatim}
    $ nvcc -O3 -arch=sm_75 -DUSE_DP add1cpu.cu
\end{verbatim}
这将得到一个使用双精度浮点数的可执行文件。在该版本中，\verb"add"~函数耗时约120 ms。我们看到，双精度版本的~\verb"add"~函数所用时间大概是单精度版本的~\verb"add"~函数所用时间的2倍，这对于这种访存主导的函数来说是合理的。本章后面会继续讨论这一点。

\subsection{为~CUDA~程序计时}

类似地，我们在第~\ref{chapter:error-check}~章的~\verb"check1api.cu"~程序的基础上进行修改，用~CUDA~事件对其中的核函数~\verb"add"~进行计时，从而得到本章的~\verb"add2gpu.cu"~程序。我们用命令
\begin{verbatim}
    $ nvcc -O3 -arch=sm_75 add2gpu.cu
\end{verbatim}
编译出使用单精度浮点数的可执行文件，用命令
\begin{verbatim}
    $ nvcc -O3 -arch=sm_75 -DUSE_DP add2gpu.cu
\end{verbatim}
编译出使用双精度浮点数的可执行文件。在装有~GeForce RTX 2070~的计算机中测试，使用单精度浮点数时核函数~\verb"add"~所用时间约为3.3 ms，使用双精度浮点数时核函数~\verb"add"~所用时间约为6.8 ms。这两个时间的比值也约为2。作者也用其他一些GPU进行了测试，结果见表~\ref{table:add-timing}。可以看到，这个时间比值对每一款~GPU~都是基本适用的。从表~\ref{table:add-timing}~中可以看出，该比值与单、双精度浮点数运算峰值的比值没有关系。这是因为，对于数组相加的问题，其执行速度是由显存带宽决定的，而不是由浮点数运算峰值决定的。

我们还可以计算数组相加问题在~GPU~中达到的有效显存带宽（effective memory bandwidth），并与表~\ref{table:add-timing}~中的理论显存带宽（theoretical memory bandwidth）进行比较。有效显存带宽定义为GPU在单位时间内访问设备内存的字节数。以作者计算机中的GeForce RTX 2070和使用单精度浮点数的情形为例，根据表中的数据，其有效显存带宽为
\begin{equation}
\frac{3 \times 10^8 \times 4 ~\rm{B}}{3.3\times 10^{-3} ~\rm{s}} \approx 360 ~\rm{GB/s}
\end{equation}
可见，有效显存带宽略小于理论显存带宽，进一步说明该问题是访存主导的，即该问题中的浮点数运算所占比例可以忽略不计。

\begin{table}[htb]
\captionsetup{font=small}
\centering
\caption{数组相加程序中的核函数在若干~GPU~中的耗时。在“浮点数运算峰值”和“核函数耗时”这两栏中，括号前的数字对应于双精度浮点数版本，括号中的数字对应于单精度浮点数版本。}
\begin{tabular}{ccccc}
\hline
GPU型号  & 计算能力 & 显存带宽 & 浮点数运算峰值 & 核函数耗时  \\
\hline
\hline
Tesla K40 & 3.5 &  288 GB/s & 1.4 (4.3) TFLOPS & 13 (6.5) ms \\
\hline
Tesla P100 & 6.0 &  732 GB/s & 4.7 (9.3) TFLOPS & 4.3 (2.2) ms \\
\hline
Tesla V100 & 7.0 & 900 GB/s & 7 (14) TFLOPS & 3.0 (1.5) ms \\
\hline
GeForce RTX 2070（笔记本） & 7.5  & 448 GB/s & 0.2 (6.5) TFLOPS & 6.8 (3.3) ms \\
\hline
GeForce RTX 2080ti & 7.5  & 616 GB/s & 0.4 (13) TFLOPS & 4.3 (2.1) ms \\
\hline
\hline
\end{tabular}
\label{table:add-timing}
\end{table}

在程序~\verb"add2gpu.cu"~中，我们仅仅对核函数进行了计时。因为我们的~CUDA~程序相对于~C++~程序多了数据复制的操作，所以我们也尝试将数据复制的操作加入被计时的代码段。由此得到的程序为~\verb"add3memcpy.cu"。我们仅用~GeForce RTX 2070~进行测试：使用单精度时，数据复制和核函数调用共耗时~180~毫秒；使用双精度时，它们共耗时~360~毫秒。

从上述测试得到的数据可以看到一个令人惊讶的结果：核函数的运行时间不到数据复制时间的~$2\%$。如果将CPU与GPU之间的数据传输时间也计入，CUDA~程序相对于~C++~程序得到的不是性能提升，而是性能降低。总之，如果一个程序的计算任务仅仅是将来自主机端的两个数组相加，并且要将结果传回主机端，使用GPU就不是一个明智的选择。那么，什么样的计算任务能够用GPU获得加速呢？本章下面的内容将回答这个问题。

在CUDA工具箱中有一个称为~\verb"nvprof"~的可执行文件，可用于对CUDA程序进行更多的性能剖析。在使用~\verb"nvprof"~时，可将它置于原来的程序执行命令之前，得到如下的运行命令：
\begin{verbatim}
    $ nvprof ./a.out
\end{verbatim}
如果用上述命令时遇到了类似如下的错误提示：
\begin{verbatim}
    Unable to profile application. Unified Memory profiling failed
\end{verbatim}
则可以尝试将运行命令换为
\begin{verbatim}
    $ nvprof --unified-memory-profiling off ./a.out
\end{verbatim}
对程序~\verb"add3memcpy.cu"~来说，在~GeForce RTX 2070~中使用上述命令，得到部分结果如下（单精度浮点数版本）：
\begin{verbatim}
    Time(%) Time     Calls Avg      Min      Max      Name
    47.00%  134.38ms 2     67.191ms 62.854ms 71.527ms [CUDA memcpy HtoD] 
    40.13%  114.74ms 1     114.74ms 114.74ms 114.74ms [CUDA memcpy DtoH] 
    12.86%  36.778ms 11    3.3435ms 3.3424ms 3.3501ms add()
\end{verbatim}
为排版方便起见，我们将~\verb"add()"~函数中的参数类型省去了，而在原始的输出中函数的参数类型是保留的。这里的第一列是此处列出的每类操作所用时间的百分比，第二列是每类操作用的总时间，第三列是每类操作被调用的次数，第四列是每类操作单次调用所用时间的平均值，第五列是每类操作单次调用所用时间的最小值，第六列是每类操作单次调用所用时间的最大值，第七列是每类操作的名称。从这里的输出可以看出核函数的执行时间及数据传输所用时间，它们和用CUDA事件获得的结果是一致的。


\section{几个影响~GPU 加速的关键因素}

\subsection{数据传输的比例}

从第~\ref{section:timing}~节的讨论我们知道，如果一个程序的目的仅仅是计算两个数组的和，那么用~GPU~可能比用~CPU~还要慢。这是因为，花在数据传输（CPU~与~GPU~之间）上的时间比计算（求和）本身还要多很多。GPU~计算核心和设备内存之间数据传输的峰值理论带宽要远高于~GPU~和~CPU~之间数据传输的带宽。参看表~\ref{table:add-timing}，典型~GPU~的显存带宽理论值为几百吉字节每秒，而常用的连接GPU和CPU内存的PCIe x16 Gen3仅有16 GB/s的带宽。它们相差几十倍。要获得可观的GPU加速，就必须尽量缩减数据传输所花时间的比例。有时候，即使有些计算在GPU中的速度并不高，也要尽量在GPU中实现，避免过多的数据经由PCIe传递。这是CUDA编程中较重要的原则之一。

假设计算任务不是做一次数组相加的计算，而是做~$10000$~次数组相加的计算，而且只需要在程序的开始和结束部分进行数据传输，那么数据传输所占的比例将可以忽略不计。此时，整个CUDA程序的性能就大为提高。在第~\ref{chapter:md}~章的分子动力学模拟程序中，仅仅在程序的开始部分将一些数据从主机复制到设备，然后在程序的中间部分偶尔将一些在GPU中计算的数据复制到主机。对这样的计算，用CUDA就有可能获得可观的加速。本书其他部分的程序都是一些简短的例子，其中数据传输部分都可能占主导，但我们将主要关注核函数优化。

\subsection{算术强度}

从前面测试的数据我们可以看到，在作者的装有GeForce RTX 2070的计算机中，数组相加的核函数比对应的C++函数快20倍左右（这是在没有对C++程序进行深度优化的情况下得到的结果，但本书不讨论对C++程序的深度优化）。这是一个可观的加速比，但远远没有达到极限。其实，对于很多计算问题，能够得到的加速比更高。数组相加的问题之所以很难得到更高的加速比，是因为该问题的算术强度（arithmetic intensity）不高。一个计算问题的算术强度指的是其中算术操作的工作量与必要的内存操作的工作量之比。例如，在数组相加的问题中，在对每一对数据进行求和时需要先将一对数据从设备内存中取出来，然后对它们实施求和计算，最后再将计算的结果存放到设备内存。这个问题的算术强度其实是不高的，因为在取两次数据、存一次数据的情况下只做了一次求和计算。在~CUDA~中，设备内存的读、写都是代价高昂（比较耗时）的。

对设备内存的访问速度取决于~GPU~的显存带宽。以~GeForce RTX 2070~为例，其显存带宽理论值为~448 GB/s。相比之下，该~GPU~的单精度浮点数计算的峰值性能为~$6.5$ TFLOPS，意味着该~GPU~的理论寄存器带宽（只考虑浮点数运算，不考虑能同时进行的整数运算）为
\begin{equation}
    \frac{4~\rm{B} \times 4~(\rm{number~of~operands~per~FMA})}{2~ (\rm{number~of~operations~per~FMA})} \times 6.5 \times 10^{12} /\rm{s}
    = 52~\rm{TB/s}.
\end{equation}
这里，FMA~指~fused multiply–add~指令，即涉及4个操作数和2个浮点数操作的运算~$d=a\times b+c$。由此可见，对单精度浮点数来说，该GPU中的数据存取比浮点数计算慢100多倍。如果考虑双精度浮点数，该比例将缩小32倍左右。对其他GPU也可以做类似的分析。

如果一个问题中需要的不仅仅是简单的单次求和操作，而是更为复杂的浮点数运算，那么就有可能得到更高的加速比。为了得到较高的算术强度，我们将之前程序（包括~C++~和~CUDA~的两个版本）中的数组相加函数进行修改。Listing~\ref{listing:arithmetic}~给出了修改后的主机函数和核函数。


\begin{lstlisting}[language=C++,caption={本章程序~arithmetic1cpu.cu~和~arithmetic2gpu.cu~中的~arithmetic~函数。},label={listing:arithmetic}]
const real x0 = 100.0;

void arithmetic(real *x, const real x0, const int N)
{
    for (int n = 0; n < N; ++n)
    {
        real x_tmp = x[n];
        while (sqrt(x_tmp) < x0)
        {
            ++x_tmp;
        }
        x[n] = x_tmp;
    }
}

void __global__ arithmetic(real *d_x, const real x0, const int N)
{
    const int n = blockDim.x * blockIdx.x + threadIdx.x;
    if (n < N)
    {
        real x_tmp = d_x[n];
        while (sqrt(x_tmp) < x0)
        {
            ++x_tmp;
        }
        d_x[n] = x_tmp;
    }
}
\end{lstlisting}

也就是说，核函数中的计算不再是一次相加的计算，而是一个~10000~次的循环，而且循环条件中还使用了数学函数~\verb"sqrt"（本章最后一节将介绍~CUDA~中的数学函数）。程序~\verb"arithmetic1cpu.cu"~可以用如下方式编译和运行（这是用单精度浮点数的情形，如果要用双精度浮点数，需要在编译选项中加上~\verb"-DUSE_DP"）：
\begin{verbatim}
    $ nvcc -O3 -arch=sm_75 -arithmetic1cpu.cu
    $ ./a.out
\end{verbatim}
程序~\verb"arithmetic2gpu.cu"~可以用如下方式编译和运行：
\begin{verbatim}
    $ nvcc -O3 -arch=sm_75 -arithmetic2gpu.cu
    $ ./a.out N
\end{verbatim}
注意，这里~CUDA~版本的可执行文件在运行时需要提供一个命令行参数~\verb"N"。该参数将赋值给程序中的变量~\verb"N"，相关代码如下：
\begin{verbatim}
    if (argc != 2) 
    {
        printf("usage: %s N\n", argv[0]);
        exit(1);
    }
    const int N = atoi(argv[1]);
\end{verbatim}

继续在装有GeForce RTX 2070的计算机中测试：当数组长度为~$10^4$时，主机函数的执行时间是320 ms（单精度）和450 ms（双精度）；当数组长度为~$10^6$~时核函数的执行时间是28 ms（单精度）和1000 ms（双精度）。因为核函数和主机函数处理的数组长度相差100倍，故在使用单精度浮点数和双精度浮点数时，GPU相对于CPU的加速比分别为
\begin{equation}
\frac{320 ~\rm{ms} \times 100}{28 ~\rm{ms}} \approx 1100
\end{equation}
和
\begin{equation}
\frac{450 ~\rm{ms} \times 100}{1000 ~\rm{ms}} = 45.
\end{equation}
可见，提高算术强度能够显著地提高~GPU~相对于~CPU~的加速比。另外值得注意的是，当算术强度很高时，GeForce~系列~GPU~的单精度浮点数的运算能力就能更加充分地发挥出来。在我们的例子中，单精度版本的核函数是双精度版本核函数的36倍之快，接近于理论比值32，进一步说明该问题是计算主导的，而不是访存主导的。用~GeForce RTX 2080ti~测试程序~\verb"arithmetic2gpu.cu"，使用单精度浮点数和双精度浮点数时核函数的执行时间分别是15 ms和450 ms，相差30倍。用Tesla V100测试，使用单精度浮点数和双精度浮点数时核函数的执行时间分别是11 ms和28 ms，只相差2倍多。可见，对于算术强度很高的问题，在使用双精度浮点数时Tesla系列的GPU相对于GeForce系列的GPU有很大的优势，而在使用单精度浮点数时前者没有显著的优势。对于算术强度不高的问题（如前面的数组相加问题），Tesla系列的GPU在使用单精度
浮点数或双精度浮点数时都没有显著的优势。在使用单精度浮点数时，GeForce系列的GPU具有更高的性价比。

\subsection{并行规模}

另一个影响~CUDA~程序性能的因素是并行规模。并行规模可用~GPU~中总的线程数目来衡量。从硬件的角度来看，一个~GPU~由多个流多处理器（streaming multiprocessor，SM）构成，而每个~SM~中有若干~CUDA~核心。每个~SM~是相对独立的。从开普勒架构到伏特架构，一个~SM~中最多能驻留（reside）的线程个数是~2048。对于图灵架构，该数目是~1024。一块~GPU~中一般有几个到几十个~SM（取决于具体的型号）。所以，一块~GPU~一共可以驻留几万到几十万个线程。如果一个核函数中定义的线程数目远小于这个数的话，就很难得到很高的加速比。

\begin{figure}[ht]
  \centering
  \captionsetup{font=small}
  \includegraphics[width=\columnwidth]{data_size.eps}\\
  \caption{（a）核函数~arithmetic~的执行时间随数组元素个数（也就是线程数目）的变化关系。（b）核函数~arithmetic~相对于对应的主机函数的加速比随数组元素个数的变化关系。测试用的GPU为GeForce RTX 2070。GPU~版本和~CPU~版本的程序都采用单精度浮点数。}
  \label{figure:data_size}
\end{figure}

为了验证这个论断，我们将~\verb"arithmetic2gpu.cu"~程序中的数组元素个数~\verb"N"~从~$10^3$~以10倍的间隔增加到~$10^8$，分别测试核函数的执行时间，结果展示在图~\ref{figure:data_size}（a）中。因为~CPU~中的计算时间基本上与数据量成正比，所以我们可以根据之前的结果计算~\verb"N"~取不同值时~GPU~程序相对于~CPU~程序的加速比，结果显示在图~\ref{figure:data_size}（b）中。

由图~\ref{figure:data_size}（a） 可知，在数组元素个数~\verb"N"~很大时，核函数的计算时间正比于~\verb"N"；在~\verb"N"~很小时，核函数的计算时间不依赖于~\verb"N"~ 的值，保持为常数。这两个极限情况都是容易理解的。当~\verb"N" 足够大时，GPU~是满负荷工作的，增加一倍的工作量就会增加一倍的计算时间。反之，当~\verb"N"~不够大时，GPU~中是有空闲的计算资源的，增加~\verb"N"~的值并不会增加计算时间。若要让~GPU~满负荷工作，则核函数中定义的线程总数要不少于某个值，该值在一般情况下和~GPU~中能够驻留的线程总数相当，但也有可能更小。只有在~GPU~满负荷工作的情况下，GPU~中的计算资源才能充分地发挥作用，从而获得较高的加速比。

因为我们的~CPU~程序中的计算是串行的，其性能基本上与数组长度无关，所以~GPU~程序相对于~CPU~程序的加速比在小~\verb"N"~的极限下几乎是正比于~\verb"N"~的。在大~\verb"N"~的极限下，GPU~程序相对于~CPU~程序的加速比接近饱和。总之，对于数据规模很小的问题，用~GPU~很难得到可观的加速。

\subsection{总结}

通过本节的例子，我们看到，一个~CUDA~程序能够获得高性能的必要（但不充分）条件有如下几点：
\begin{itemize}
    \item 数据传输比例较小。
    \item 核函数的算术强度较高。
    \item 核函数中定义的线程数目较多。
\end{itemize}
所以，在编写与优化~CUDA~程序时，一定要想方设法（主要是指仔细设计算法）做到以下几点：
\begin{itemize}
    \item 减少主机与设备之间的数据传输。
    \item 提高核函数的算术强度。
    \item 增大核函数的并行规模。
\end{itemize}


\section{CUDA~中的数学函数库}

在前面的例子中，我们在核函数中使用了求平方根的数学函数。在~CUDA~数学库中，还有很多类似的数学函数，如幂函数、三角函数、指数函数、对数函数等。这些函数可以在如下网站查询：\url{http://docs.nvidia.com/cuda/cuda-math-api}。建议读者浏览该文档，了解~CUDA~的数学函数库都提供了哪些数学函数。这样，在需要使用时就容易想起来。

CUDA~数学库中的函数可以归纳如下：
\begin{enumerate}
    \item 单精度浮点数内建函数和数学函数（single precision intrinsics and math functions）。使用该类函数时不需要包含任何额外的头文件。
    \item 双精度浮点数内建函数和数学函数（double precision intrinsics and math functions）。使用该类函数时不需要包含任何额外的头文件。
    \item 半精度浮点数内建函数和数学函数（half precision intrinsics and math functions）。使用该类函数时需要包含头文件~\verb"<cuda_fp16.h>"。本书不涉及此类函数。
    \item 整数类型的内建函数（integer intrinsics）。使用该类函数时不需要包含任何额外的头文件。本书不涉及此类函数。
    \item 类型转换内建函数（type casting intrinsics）。使用该类函数时不需要包含任何额外的头文件。本书不涉及此类函数。
    \item 单指令-多数据内建函数（SIMD intrinsics）。使用该类函数时不需要包含任何额外的头文件。本书不涉及此类函数。
\end{enumerate}

本书将仅涉及单精度浮点数和双精度浮点数类型的数学函数和内建函数。其中数学函数（math functions）都是经过重载的。例如，求平方根的函数具有如下3种原型：
\begin{verbatim}
    double sqrt(double x);
    float sqrt(float x);
    float sqrtf(float x);
\end{verbatim}
所以，当~\verb"x"~是双精度浮点数时，我们只可以用~\verb"sqrt(x)"；当~\verb"x"~是单精度浮点数时，我们可以用~\verb"sqrt(x)"，也可以用~\verb"sqrtf(x)"。那么综合起来，我们可统一地用双精度函数的版本处理单精度浮点数和双精度浮点数类型的变量。

内建函数指的是一些准确度较低，但效率较高的函数。例如，有如下版本的求平方根的内建函数：
\begin{verbatim}
__device__​ float __fsqrt_rd (float  x); // round-down mode
__device__​ float __fsqrt_rn (float  x); // round-to-nearest-even mode
__device__​ float __fsqrt_ru (float  x); // round-up mode
__device__​ float __fsqrt_rz (float  x); // round-towards-zero mode
__device__​ double __fsqrt_rd (double  x); // round-down mode
__device__​ double __fsqrt_rn (double  x); // round-to-nearest-even mode
__device__​ double __fsqrt_ru (double  x); // round-up mode
__device__​ double __fsqrt_rz (double  x); // round-towards-zero mode   
\end{verbatim}

在开发~CUDA~程序时，浮点数精度的选择及数学函数和内建函数之间的选择都要视应用程序的要求而定。例如，在作者开发的分子动力学模拟程序~GPUMD（\url{https://github.com/brucefan1983/GPUMD}）中，绝大部分的代码使用了双精度浮点数，只在极个别的地方使用了单精度浮点数，而且没有使用内建函数；在作者开发的经验势拟合程序~GPUGA（\url{https://github.com/brucefan1983/GPUGA}）中，统一使用了单精度浮点数，而且使用了内建函数。之所以这样选择，是因为前者对计算精度要求较高，后者对计算精度要求较低。

\chapter{CUDA~的内存组织\label{chapter:memory}}

前一章讨论了几个获得~GPU~加速的必要但不充分条件。在满足那些条件之后，要获得尽可能高的性能，还有很多需要注意的方面，其中最重要的是合理地使用各种设备内存。本章从整体上介绍~CUDA~中的内存组织，为后续章节的讨论打好理论基础。
