\section{CUDA~的内存组织简介}

现代计算机中的内存往往存在一种组织结构（hierarchy）。在这种结构中，含有多种类型的内存，每种内存分别具有不同的容量和延迟（latency，可以理解为处理器等待内存数据的时间）。一般来说，延迟低（速度高）的内存容量小，延迟高（速度低）的内存容量大。当前被处理的数据一般存放于低延迟、低容量的内存中；当前没有被处理但之后将要被处理的大量数据一般存放于高延迟、高容量的内存中。相对于不用分级的内存，用这种分级的内存可以降低延迟，提高计算效率。

CPU~和~GPU~中都有内存分级的设计。相对于~CPU~编程来说，CUDA~编程模型向程序员提供更多的控制权。因此，对~CUDA~编程来说，熟悉其内存的分级组织是非常重要的。

\begin{table}[htb]
\centering
\captionsetup{font=small}
\caption{CUDA~中设备内存的分类与特征}
\begin{tabular}{ccccc}
\hline
内存类型 & 物理位置  &  访问权限  & 可见范围  & 生命周期 \\
\hline
\hline
全局内存 & 在芯片外  &  可读可写 & 所有线程和主机端 & 由主机分配与释放\\
\hline
常量内存 & 在芯片外  &  仅可读 & 所有线程和主机端 & 由主机分配与释放\\
\hline
纹理和表面内存 & 在芯片外  &  一般仅可读 & 所有线程和主机端 & 由主机分配与释放\\
\hline
寄存器内存 & 在芯片内  &  可读可写 & 单个线程 & 所在线程 \\
\hline
局部内存 & 在芯片外  &  可读可写 & 单个线程 & 所在线程\\
\hline
共享内存 & 在芯片内  &  可读可写 & 单个线程块 & 所在线程块\\ 
\hline
\hline
\end{tabular}
\label{table:memory}
\end{table}

表~\ref{table:memory}~列出了~CUDA~中的几种内存和它们的主要特征。这些特征包括物理位置、设备的访问权限、可见范围及对应变量的生命周期。图~\ref{figure:memory}~可以进一步帮助理解。本章仅简要介绍~CUDA~中的各种内存，更多细节在后续几章会有进一步讨论。本章会频繁引用前几章讨论过的数组相加的例子。


\begin{figure}[ht]
  \centering
  \captionsetup{font=small}
  \includegraphics[width=\columnwidth]{memory.pdf}\\
  \caption{CUDA~中的内存组织示意图。请结合表~\ref{table:memory}~理解此图。图中箭头的方向表示数据可以移动的方向。}
  \label{figure:memory}
\end{figure}

\section{CUDA~中不同类型的内存}

\subsection{全局内存}

这里“全局内存”（global memory）的含义是核函数中的所有线程都能够访问其中的数据，和~C++~中的“全局变量”不是一回事。我们已经用过这种内存，在数组相加的例子中，指针~\verb"d_x"、\verb"d_y"~和~\verb"d_z"~都是指向全局内存的。全局内存由于没有存放在~GPU~的芯片上，因此具有较高的延迟和较低的访问速度。然而，它的容量是所有设备内存中最大的。其容量基本上就是显存容量。第~\ref{chapter:GPU-and-CUDA}~章的表~\ref{table:gpus}~列出了几款~GPU~的显存容量。

全局内存的主要角色是为核函数提供数据，并在主机与设备及设备与设备之间传递数据。首先，我们用~\verb"cudaMalloc"~函数为全局内存变量分配设备内存。然后，可以直接在核函数中访问分配的内存，改变其中的数据值。我们说过，要尽量减少主机与设备之间的数据传输，但有时是不可避免的。可以用~\verb"cudaMemcpy"~函数将主机的数据复制到全局内存，或者反过来。在前几章数组相加的例子中，语句
\begin{verbatim}
    cudaMemcpy(d_x, h_x, M, cudaMemcpyHostToDevice);
\end{verbatim}
将中~\verb"M"~字节的数据从主机复制到设备，而语句
\begin{verbatim}
    cudaMemcpy(h_z, d_z, M, cudaMemcpyDeviceToHost);
\end{verbatim}
将中~\verb"M"~字节的数据从设备复制到主机。还可以将一段全局内存中的数据复制到另一段全局内存中去。例如：
\begin{verbatim}
    cudaMemcpy(d_x, d_y, M, cudaMemcpyDeviceToDevice);
\end{verbatim}
的作用就是将首地址为~\verb"d_y"~的全局内存中~\verb"M"~字节的数据复制到首地址为~\verb"d_x"~的全局内存中去。注意，这里必须将数据传输的方向指定为~\verb"cudaMemcpyDeviceToDevice"~或~\verb"cudaMemcpyDefault"。


全局内存可读可写。在数组相加的例子中，语句
\begin{verbatim}
    d_z[n] = d_x[n] + d_y[n];
\end{verbatim}
同时体现了全局内存的可读性和可写性。对于线程~\verb"n"~来说，该语句将变量~\verb"d_x"~和~\verb"d_y"~所指全局内存缓冲区的第~\verb"n"~个元素读出，相加后将结果写入变量~\verb"d_z"~所指全局内存缓冲区的第~\verb"n"~个元素。

全局内存对整个网格的所有线程可见。也就是说，一个网格的所有线程都可以访问（读或写）传入核函数的设备指针所指向的全局内存中的全部数据。在上面的语句中，第~\verb"n"~个线程刚好访问全局内存缓冲区的第~\verb"n"~个元素，但并不是非要这样。如有需要，第~\verb"n"~个线程可以访问全局内存缓冲区中的任何一个元素。

全局内存的生命周期（lifetime）不是由核函数决定的，而是由主机端决定的。在数组相加的例子中，由指针~\verb"d_x"、\verb"d_y"~和~\verb"d_z"~所指向的全局内存缓冲区的生命周期就是从主机端用~\verb"cudaMalloc"~对它们分配内存开始，到主机端用~\verb"cudaFree"~释放它们的内存结束。在这期间，可以在相同的或不同的核函数中多次访问这些全局内存中的数据。

在处理逻辑上的两维或三维问题时，可以用~\verb"cudaMallocPitch"~和~\verb"cudaMalloc3D"~函数分配内存，用~\verb"cudaMemcpy2D"~和~\verb"cudaMemcpy3D"~复制数据，释放时依然用~\verb"cudaFree"~函数。本书不讨论这种内存分配函数及相应的数据复制函数。

以上所有的全局内存都称为线性内存（linear memory）。在CUDA中还有一种内部构造对用户不透明的（not transparent）全局内存，称为CUDA Array。CUDA Array使用英伟达公司不对用户公开的数据排列方式，专为纹理拾取服务，本书不讨论。

我们前面介绍的全局内存变量都是动态地分配内存的。在~CUDA~中允许使用静态全局内存变量，其所占内存数量是在编译期间就确定的。而且，这样的静态全局内存变量必须在所有主机与设备函数外部定义，所以是一种“全局的静态全局内存变量”。这里，第一个“全局”的含义与~C++~中全局变量的含义相同，指的是对应的变量对从其定义之处开始、一个翻译单元内的所有设备函数直接可见。如果采用所谓的分离编译（separate compiling），还可以将可见范围进一步扩大，但本书不讨论分离编译。

静态全局内存变量由以下方式在任何函数外部定义：
\begin{verbatim}
    __device__ T x; // 单个变量
    __device__ T y[N]; // 固定长度的数组
\end{verbatim}
其中，修饰符~\verb"__device__"~说明该变量是设备中的变量，而不是主机中的变量；\verb"T"~是变量的类型；\verb"N"~是一个整型常数。Listing \ref{listing:static.cu}~展示了静态全局内存变量的使用方式。该程序将输出：
\begin{verbatim}
    d_x = 1, d_y[0] = 11, d_y[1] = 21.
    h_y[0] = 11, h_y[1] = 21.
\end{verbatim}
在核函数中，可直接对静态全局内存变量进行访问，并不需要将它们以参数的形式传给核函数。不可在主机函数中直接访问静态全局内存变量，但可以用~\verb"cudaMemcpyToSymbol"~函数和~\verb"cudaMemcpyFromSymbol"~函数在静态全局内存与主机内存之间传输数据。这两个~CUDA~运行时~API~函数的原型如下：
\begin{verbatim}
    cudaError_t cudaMemcpyToSymbol
    (
        const void* symbol, // 静态全局内存变量名
        const void* src, // 主机内存缓冲区指针
        size_t count, // 复制的字节数
        size_t offset = 0, // 从 symbol 对应设备地址开始偏移的字节数
        cudaMemcpyKind kind = cudaMemcpyHostToDevice // 可选参数
    );   
    cudaError_t cudaMemcpyFromSymbol
    (
        void* dst, // 主机内存缓冲区指针
        const void* symbol, // 静态全局内存变量名
        size_t count, // 复制的字节数
        size_t offset = 0, // 从 symbol 对应设备地址开始偏移的字节数
        cudaMemcpyKind kind = cudaMemcpyDeviceToHost // 可选参数
    );
\end{verbatim}
这两个函数的参数~\verb"symbol"~可以是静态全局内存变量的变量名，也可以是下面要介绍的常量内存变量的变量名。第~16~行调用~\verb"cudaMemcpyToSymbol"~函数将主机数组~\verb"h_y"~中的数据复制到静态全局内存数组~\verb"d_y"，第~21~行调用~\verb"cudaMemcpyFromSymbol"~函数将静态全局内存数组~\verb"d_y"~中的数据复制到主机数组~\verb"h_y"。这里只是展示静态全局内存的使用方法，我们将在第~\ref{chapter:warp}~章讨论一种利用静态全局内存加速程序的技巧。

\begin{lstlisting}[language=C++,caption={本章程序~static.cu~的全部代码。},label={listing:static.cu}]
#include "error.cuh"
#include <stdio.h>
__device__ int d_x = 1;
__device__ int d_y[2];

void __global__ my_kernel(void)
{
    d_y[0] += d_x;
    d_y[1] += d_x;
    printf("d_x = %d, d_y[0] = %d, d_y[1] = %d.\n", d_x, d_y[0], d_y[1]);
}

int main(void)
{
    int h_y[2] = {10, 20};
    CHECK(cudaMemcpyToSymbol(d_y, h_y, sizeof(int) * 2));
    
    my_kernel<<<1, 1>>>();
    CHECK(cudaDeviceSynchronize());
    
    CHECK(cudaMemcpyFromSymbol(h_y, d_y, sizeof(int) * 2));
    printf("h_y[0] = %d, h_y[1] = %d.\n", h_y[0], h_y[1]);
    
    return 0;
}
\end{lstlisting}


\subsection{常量内存}

常量内存（constant memory）是有常量缓存的全局内存，数量有限，一共仅有~64 KB。它的可见范围和生命周期与全局内存一样。不同的是，常量内存仅可读、不可写。由于有缓存，常量内存的访问速度比全局内存高，但得到高访问速度的前提是一个线程束中的线程（一个线程块中相邻的~32~个线程）要读取相同的常量内存数据。

一个使用常量内存的方法是在核函数外面用~\verb"__constant__"~ 定义变量，并用前面介绍的~CUDA~运行时~API~函数~\verb"cudaMemcpyToSymbol"~将数据从主机端复制到设备的常量内存后供核函数使用。当计算能力不小于~2.0~时，给核函数传递的参数（传值，不是像全局变量那样传递指针）就存放在常量内存中，但给核函数传递参数最多只能在一个核函数中使用4 KB常量内存。

所以，我们其实已经用过了常量内存。在数组相加的例子中，核函数的参数 ~\verb"const int N"~就是在主机端定义的变量，并通过传值的方式传送给核函数中的线程使用。在核函数中的代码段~
\verb"if (n < N)"~中，这个参数~\verb"N"~就被每一个线程使用了。所以，核函数中的每一个线程都知道该变量的值，而且对它的访问比对全局内存的访问要快。除给核函数传递单个的变量外，还可以传递结构体，同样也是使用常量内存。结构体中可以定义单个的变量，也可以定义固定长度的数组。第~\ref{chapter:md}~章的程序将涉及常量内存的使用。

\subsection{纹理内存和表面内存}

纹理内存（texture memory）和表面内存（surface memory）类似于常量内存，也是一种具有缓存的全局内存，有相同的可见范围和生命周期，而且一般仅可读（表面内存也可写）。不同的是，纹理内存和表面内存容量更大，而且使用方式和常量内存也不一样。

对于计算能力不小于3.5的GPU来说，将某些只读全局内存数据用~\verb"__ldg()"~函数通过只读数据缓存（read-only data cache）读取，既可达到使用纹理内存的加速效果，又可使代码简洁。该函数的原型为
\begin{verbatim}
    T __ldg(const T* address);
\end{verbatim}
其中，\verb"T"~是需要读取的数据的类型；\verb"address"~是数据的地址。对帕斯卡架构和更高的架构来说，全局内存的读取在默认情况下就利用了~\verb"__ldg()"~函数，所以不需要明显地使用它。我们在第~\ref{chapter:global}~章就会讨论该函数的使用。

\subsection{寄存器}

在核函数中定义的不加任何限定符的变量一般来说就存放于寄存器（register）中。核函数中定义的不加任何限定符的数组有可能存放于寄存器中，但也有可能存放于局部内存中。另外，以前提到过的各种内建变量，如~\verb"gridDim"、\verb"blockDim"、\verb"blockIdx"、\verb"threadIdx"~及~\verb"warpSize"~都保存在特殊的寄存器中。在核函数中访问这些内建变量是很高效的。

我们已经使用过寄存器变量。在数组求和的例子中，我们在核函数中有如下语句：
\begin{verbatim}
    const int n = blockDim.x * blockIdx.x + threadIdx.x;
\end{verbatim}
这里的~\verb"n"~就是一个寄存器变量。寄存器可读可写。上述语句的作用就是定义一个寄存器变量~\verb"n"~并将赋值号右边计算出来的值赋给它（写入）。在稍后的语句
\begin{verbatim}
    z[n] = x[n] + y[n];
\end{verbatim}
中，寄存器变量~\verb"n"~的值被使用（读出）。

寄存器变量仅仅被一个线程可见。也就是说，每一个线程都有一个变量~\verb"n"~的副本。虽然在核函数的代码中用了这同一个变量名，但是不同的线程中该寄存器变量的值是可以不同的。每个线程都只能对它的副本进行读写。寄存器的生命周期也与所属线程的生命周期一致，从定义它开始，到线程消失时结束。


寄存器内存在芯片上（on-chip），是所有内存中访问速度最高的，但是其数量也很有限。表~\ref{table:compute_capability_memory}~列出了几个不同计算能力的GPU中与寄存器和后面要介绍的共享内存有关的技术指标。该表只包含少数几个计算能力，更完整的列表见《CUDA C++ Programming Guide》的附录H。一个寄存器占有32 bit（4~字节）的内存。所以，一个双精度浮点数将使用两个寄存器。这是在估算寄存器使用量时要注意的。

\begin{table}[htb]
\centering
\captionsetup{font=small}
\caption{几个计算能力的技术指标}
\begin{tabular}{lllll}
\hline
计算能力 & 3.5 & 6.0 & 7.0 & 7.5\\
\hline
\hline
GPU~代表 & Tesla K40 & Tesla P100 & Tesla V100 & Geforce RTX 2080 \\
\hline
SM~寄存器数上限 & $64\times1024$   & $64\times1024$ & $64\times1024$ & $64\times1024$ \\
\hline
单个线程块寄存器数上限 & $64\times1024$ & $64\times1024$ & $64\times1024$ & $64\times1024$ \\
\hline
单个线程寄存器数上限 & 255 & 255 & 255 & 255  \\
\hline
SM~共享内存上限 & 48 KB & 64 KB & 96 KB & 64 KB \\
\hline
单个线程块共享内存上限 & 48 KB & 48 KB & 96 KB & 64 KB \\
\hline
\hline
\end{tabular}
\label{table:compute_capability_memory}
\end{table}


\subsection{局部内存}

我们还没有用过局部内存（local memory），但从用法上看，局部内存和寄存器几乎一样。核函数中定义的不加任何限定符的变量有可能在寄存器中，也有可能在局部内存中。寄存器中放不下的变量，以及索引值不能在编译时就确定的数组，都有可能放在局部内存中。这种判断是由编译器自动做的。对于数组相加例子中的变量~\verb"n"~来说，作者可以肯定它在寄存器中，而不是局部内存中，因为核函数所用寄存器数量还远远没有达到上限。

虽然局部内存在用法上类似于寄存器，但从硬件来看，局部内存只是全局内存的一部分。所以，局部内存的延迟也很高。每个线程最多能使用高达~512 KB~的局部内存，但使用过多会降低程序的性能。

\subsection{共享内存}

我们还没有使用过共享内存（shared memory）。共享内存和寄存器类似，存在于芯片上，具有仅次于寄存器的读写速度，数量也有限。表~\ref{table:compute_capability_memory}~列出了与几个计算能力对应的共享内存数量指标。

不同于寄存器的是，共享内存对整个线程块可见，其生命周期也与整个线程块一致。也就是说，每个线程块拥有一个共享内存变量的副本。共享内存变量的值在不同的线程块中可以不同。一个线程块中的所有线程都可以访问该线程块的共享内存变量副本，但是不能访问其他线程块的共享内存变量副本。共享内存的主要作用是减少对全局内存的访问，或者改善对全局内存的访问模式。这些将在第~\ref{chapter:shared}~章详细地讨论。

\subsection{L1~和~L2~缓存}

从费米架构开始，有了~SM~层次的~L1~缓存（一级缓存）和设备（一个设备有多个~SM）层次的~L2~缓存（二级缓存）。它们主要用来缓存全局内存和局部内存的访问，减少延迟。

从硬件的角度来看，开普勒架构中的L1缓存和共享内存使用同一块物理芯片；在麦克斯韦架构和帕斯卡架构中，L1缓存和纹理缓存统一起来，而共享内存是独立的；在伏特架构和图灵架构中，L1缓存、纹理缓存及共享内存三者统一起来。从编程的角度来看，共享内存是可编程的缓存（共享内存的使用完全由用户操控），而L1和L2缓存是不可编程的缓存（用户最多能引导编译器做一些选择）。

对某些架构来说，还可以针对单个核函数或者整个程序改变~L1~缓存和共享内存的比例。具体地说：
\begin{itemize}
    \item 计算能力~3.5：L1~缓存和共享内存共有64 KB，可以将共享内存上限设置成~16 KB、32 KB~和~48 KB，其余的归~L1~缓存。默认情况下有48KB共享内存。
    \item 计算能力~3.7：L1缓存和共享内存共有128 KB，可以将共享内存上限设置成~80 KB、96 KB~和~112 KB，其余的归~L1~缓存。默认情况下有~112 KB~共享内存。
    \item 麦克斯韦架构和帕斯卡架构不允许调整共享内存的上限。
    \item 伏特架构：统一的（L1/纹理/共享内存）缓存共有~128 KB，共享内存上限可调整为0 KB、8 KB、16 KB、32 KB、64 KB或96 KB。
    \item 图灵架构：统一的（L1/纹理/共享内存）缓存共有96 KB，共享内存上限可调整为32 KB或64 KB。
\end{itemize}
由于以上关于共享内存比例的设置不是很通用，本书不对它们做进一步讨论。感兴趣的读者可阅读《CUDA C++ Programming Guide》和其他资料进一步学习。

\section{SM及其占有率}

\subsection{SM的构成}

我们在第~\ref{chapter:speedup}~章讨论并行规模对~CUDA~程序性能的影响时提到了流多处理器SM。一个GPU是由多个SM构成的。一个SM包含如下资源：
\begin{itemize}
\item 一定数量的寄存器（参见表~\ref{table:compute_capability_memory}）。
\item 一定数量的共享内存（参见表~\ref{table:compute_capability_memory}）。
\item 常量内存的缓存。
\item 纹理和表面内存的缓存。
\item L1缓存。
\item 两个（计算能力6.0）或4个（其他计算能力）线程束调度器（warp scheduler），用于在不同线程的上下文之间迅速地切换，以及为准备就绪的线程束发出执行指令。
\item 执行核心，包括：
\begin{itemize}
\item 若干整型数运算的核心（INT32）。
\item 若干单精度浮点数运算的核心（FP32）。
\item 若干双精度浮点数运算的核心（FP64）。
\item 若干单精度浮点数超越函数（transcendental functions）的特殊函数单元（Special Function Units，SFUs）。
\item 若干混合精度的张量核心（tensor cores，由伏特架构引入，适用于机器学习中的低精度矩阵计算，本书不讨论）。
\end{itemize}
\end{itemize}

\subsection{SM~的占有率}

因为一个~SM~中的各种计算资源是有限的，那么有些情况下一个~SM~中驻留的线程数目就有可能达不到理想的最大值。此时，我们说该~SM~的占有率小于~$100\%$。获得~$100\%$~的占有率并不是获得高性能的必要或充分条件，但一般来说，要尽量让~SM~的占有率不小于某个值，比如~$25\%$，才有可能获得较高的性能。


在第~\ref{chapter:speedup}~章，我们讨论了并行规模。当并行规模较小时，有些~SM~可能就没有被利用，占有率为零。这是导致程序性能低下的原因之一。当并行规模足够大时，也有可能得到非~100\%~的占有率，这就是下面要讨论的情形。

在表~\ref{table:compute_capability_memory}~中，我们列举了一个~SM、一个线程块及一个线程中能够使用的寄存器和共享内存的上限。在第~\ref{chapter:thread}~章，我们还提到了，一个线程块（无论几维的）中的线程数不能超过~1024。要分析~SM~的理论占有率（theoretical occupancy），还需要知道两个指标：
\begin{itemize}
\item 一个~SM~中最多能拥有的线程块个数为~$N_{\rm b}=16$（开普勒架构和图灵架构）或者~$N_{\rm b}=32$（麦克斯韦架构、帕斯卡架构和伏特架构）；
\item 一个~SM~中最多能拥有的线程个数为~$N_{\rm t}=2048$（从开普勒架构到伏特架构）或者~$N_{\rm t}=1024$（图灵架构）。
\end{itemize}

下面在并行规模足够大（即核函数执行配置中定义的总线程数足够多）的前提下分几种情况来分析~SM~的理论占有率：
\begin{itemize}
\item 寄存器和共享内存使用量很少的情况。此时，SM~的占有率完全由执行配置中的线程块大小决定。关于线程块大小，读者也许注意到我们之前总是用~128。这是因为，SM~中线程的执行是以线程束为单位的，所以最好将线程块大小取为线程束大小（32~个线程）的整数倍。例如，假设将线程块大小定义为~100，那么一个线程块中将有~3~个完整的线程束（一共~96~个线程）和一个不完整的线程束（只有~4~个线程）。在执行核函数中的指令时，不完整的线程束花的时间和完整的线程束花的时间一样，这就无形中浪费了计算资源。所以，建议将线程块大小取为~32~的整数倍。在该前提下，任何不小于~$N_{\rm t}/N_{\rm b}$~而且能整除~$N_{\rm t}$~的线程块大小都能得到~$100\%$~的占有率。根据我们列出的数据，线程块大小不小于~$ 128$~时开普勒架构能获得~100\%~的占有率；线程块大小不小于~$64$~时其
他架构能获得~$100\%$~的占有率。作者近几年都用一块开普勒架构的~Tesla K40~开发程序，所以习惯了在一般情况下都用~128~的线程块大小。
\item 有限的寄存器个数对占有率的约束情况。我们只针对表~\ref{table:compute_capability_memory}~中列出的几个计算能力进行分析，读者可类似地分析其他未列出的计算能力。对于表~\ref{table:compute_capability_memory}~中列出的所有计算能力，一个~SM~最多能使用的寄存器个数为~64 K（$64\times 1024$）。除了图灵架构，如果我们希望在一个~SM~中驻留最多的线程（2048~个），核函数中的每个线程最多只能用~32~个寄存器。当每个线程所用寄存器个数大于~64~时，SM~的占有率将小于~$50\%$；当每个线程所用寄存器个数大于~128~时，SM~的占有率将小于~$25\%$。对于图灵架构，同样的占有率允许使用更多的寄存器。
\item 有限的共享内存对占有率的约束情况。因为共享内存的数量随着计算能力的上升没有显著的变化规律，所以我们这里仅针对一个3.5的计算能力进行分析，对其他计算能力可以类似地分析。如果线程块大小为~128，那么每个~SM~要激活~16~个线程块才能有~2048~个线程，达到~$100\%$~的占有率。此时，一个线程块最多能使用~3 KB~的共享内存。在不改变线程块大小的情况下，要达到~$50\%$~的占有率，一个线程块最多能使用~6 KB~的共享内存；要达到~$25\%$~的占有率，一个线程块最多能使用~12 KB~共享内存。最后，如果一个线程块使用了超过~48 KB~的共享内存，会直接导致核函数无法运行。对其他线程块大小可类似地分析。
\item 以上单独分析了线程块大小、寄存器数量及共享内存数量对~SM~占有率的影响。一般情况下，需要综合以上三点分析。在~CUDA~工具箱中，有一个名为~\verb"CUDA_Occupancy_Calculator.xls"~的~Excel~文档，可用来计算各种情况下的~SM~占有率，感兴趣的读者可以去尝试使用。
\end{itemize}

值得一提的是，用编译器选项~\verb"--ptxas-options=-v"~可以报道每个核函数的寄存器使用数量。CUDA~还提供了核函数的~\verb"__launch_bounds__()"~修饰符和~\verb"--maxrregcount="~编译选项来让用户分别对一个核函数和所有核函数中寄存器的使用数量进行控制。本书不对此展开讨论，感兴趣的读者可查阅其他资料进一步学习。

\section{用CUDA运行时API函数查询设备}

在第~\ref{chapter:GPU-and-CUDA}~章，我们介绍了如何利用~nvidia-smi~程序对设备进行某些方面的查询与设置。本节介绍用~CUDA~运行时~API~函数查询所用~GPU~的规格。Listing~\ref{listing:query.cu}~所示程序可用来查询一些到目前为止我们介绍过的~GPU~规格。

\begin{lstlisting}[language=C++,caption={本章程序~query.cu~的全部代码。},label={listing:query.cu}]
#include "error.cuh"
#include <stdio.h>

int main(int argc, char *argv[])
{
    int device_id = 0;
    if (argc > 1) device_id = atoi(argv[1]);
    CHECK(cudaSetDevice(device_id));

    cudaDeviceProp prop;
    CHECK(cudaGetDeviceProperties(&prop, device_id));

    printf("Device id:                                 %d\n",
        device_id);
    printf("Device name:                               %s\n",
        prop.name);
    printf("Compute capability:                        %d.%d\n",
        prop.major, prop.minor);
    printf("Amount of global memory:                   %g GB\n",
        prop.totalGlobalMem / (1024.0 * 1024 * 1024));
    printf("Amount of constant memory:                 %g KB\n",
        prop.totalConstMem  / 1024.0);
    printf("Maximum grid size:                         %d %d %d\n",
        prop.maxGridSize[0], 
        prop.maxGridSize[1], prop.maxGridSize[2]);
    printf("Maximum block size:                        %d %d %d\n",
        prop.maxThreadsDim[0], prop.maxThreadsDim[1], 
        prop.maxThreadsDim[2]);
    printf("Number of SMs:                             %d\n",
        prop.multiProcessorCount);
    printf("Maximum amount of shared memory per block: %g KB\n",
        prop.sharedMemPerBlock / 1024.0);
    printf("Maximum amount of shared memory per SM:    %g KB\n",
        prop.sharedMemPerMultiprocessor / 1024.0);
    printf("Maximum number of registers per block:     %d K\n",
        prop.regsPerBlock / 1024);
    printf("Maximum number of registers per SM:            %d K\n",
        prop.regsPerMultiprocessor / 1024);
    printf("Maximum number of threads per block:       %d\n",
        prop.maxThreadsPerBlock);
    printf("Maximum number of threads per SM:          %d\n",
        prop.maxThreadsPerMultiProcessor);

    return 0;
}
\end{lstlisting}

程序的第10行定义了一个CUDA中定义好的结构体类型~\verb"cudaDeviceProp"~的变量~\verb"prop"。在第11行，利用~CUDA~运行时~API~函数~\verb"cudaGetDeviceProperties"~得到了编号为~\verb"device_id = 0"~的设备的性质，存放在结构体变量~\verb"prop"~中。从第13行开始，将变量~\verb"prop"~中的某些成员的值打印出来。在装有~GeForce RTX 2070~的计算机中得到如下输出：
\begin{small}
\begin{verbatim}
    Device id:                                 0
    Device name:                               GeForce RTX 2070 with Max-Q Design
    Compute capability:                        7.5
    Amount of global memory:                   8 GB
    Amount of constant memory:                 64 KB
    Maximum grid size:                         2147483647 65535 65535
    Maximum block size:                        1024 1024 64
    Number of SMs:                             36
    Maximum amount of shared memory per block: 48 KB
    Maximum amount of shared memory per SM:    64 KB
    Maximum number of registers per block:     64 K
    Maximum number of registers per SM:        64 K
    Maximum number of threads per block:       1024
    Maximum number of threads per SM:          1024
\end{verbatim}
\end{small}


读者可以尝试在自己的系统中运行该程序，确保能够理解每一个规格的含义。在本例中，我们选择查询编号为~0~的设备。如果读者的系统中有不止一块~GPU，而且不想查询第~0~号设备，则可以修改第~6~行的设备编号。值得说明的是，如果读者想用编号为~1~的~GPU~执行程序，而不想用默认的编号为~0~的~GPU，则可以在调用任何~CUDA~运行时~API~函数之前写下如下语句：
\begin{verbatim}
    CHECK(cudaSetDevice(1));
\end{verbatim}
另外，读者还可以回顾一下在第~\ref{chapter:GPU-and-CUDA}~章介绍过的用~nvidia-smi~程序在命令行选择~GPU~的方法。在~CUDA~工具箱中，有一个名为~\verb"deviceQuery.cu"~的程序，可以输出更多的信息。

