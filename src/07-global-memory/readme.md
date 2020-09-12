
\chapter{全局内存的合理使用\label{chapter:global}}

在第~\ref{chapter:memory}~章，我们抽象地介绍了~CUDA~中的各种内存。从本章开始，我们将通过实例讲解各种内存的合理使用。在各种设备内存中，全局内存具有最低的访问速度（最高的延迟），往往是一个~CUDA~程序性能的瓶颈，所以值得特别地关注。本章讨论全局内存的合理使用。

\section{全局内存的合并与非合并访问\label{section:coalesing}}

对全局内存的访问将触发内存事务（memory
transaction），也就是数据传输（data transfer）。在第~\ref{chapter:memory}~章我们提到，从费米架构开始，有了~SM~层次的~L1~缓存和设备层次的~L2~缓存，可以用于缓存全局内存的访问。在启用了~L1~缓存的情况下，对全局内存的读取将首先尝试经过~L1~缓存；如果未中，则接着尝试经过~L2~缓存；如果再次未中，则直从~DRAM~读取。一次数据传输处理的数据量在默认情况下是~32~字节。

关于全局内存的访问模式，有合并（coalesced）与非合并（uncoalesced）之分。合并访问指的是一个线程束对全局内存的一次访问请求（读或者写）导致最少数量的数据传输，否则称访问是非合并的。定量地说，可以定义一个合并度（degree of coalesing），它等于线程束请求的字节数除以由该请求导致的所有数据传输处理的字节数。如果所有数据传输中处理的数据都是线程束所需要的，那么合并度就是~100\%，即对应合并访问。所以，也可以将合并度理解为一种资源利用率。利用率越高，核函数中与全局内存访问有关的部分的性能就更好；利用率低则意味着对显存带宽的浪费。本节后面主要探讨合并度与全局内存访问模式之间的关系。

为简单起见，我们主要以全局内存的读取和仅使用~L2~缓存的情况为例进行下述讨论。在此情况下，一次数据传输指的就是将~32~字节的数据从全局内存（DRAM）通过~32~字节的~L2~缓存片段（cache sector）传输到~SM。考虑一个线程束访问单精度浮点数类型的全局内存变量的情形。因为一个单精度浮点数占有4个字节，故该线程束将请求~128~字节的数据。在理想情况下（即合并度为~100\%~的情况），这将仅触发~$128/32=4$~次用~L2~缓存的数据传输。那么，在什么情况下会导致多于~4~次数据传输呢？

为了回答这个问题，我们首先需要了解数据传输对数据地址的要求：在一次数据传输中，从全局内存转移到L2缓存的一片内存的首地址一定是一个最小粒度（这里是32字节）的整数倍。例如，一次数据传输只能从全局内存读取地址为0-31字节、32-63字节、64-95字节、96-127字节等片段的数据。如果线程束请求的全局内存数据的地址刚好为0-127字节或128-255字节等，就能与4次数据传输所处理的数据完全吻合。这种情况下的访问就是合并访问。

读者也许会问：如何保证一次数据传输中内存片段的首地址为最小粒度的整数倍呢？或者问：如何控制所使用的全局内存的地址呢？答案是：使用CUDA运行时API函数（如我们常用的~\verb"cudaMalloc"）分配的内存的首地址至少是256字节的整数倍。

下面我们通过几个具体的核函数列举几种常见的内存访问模式及其合并度。
\begin{enumerate}
\item 顺序的合并访问。我们考察如下的核函数和相应的调用：
\begin{verbatim}
    void __global__ add(float *x, float *y, float *z)
    {
        int n = threadIdx.x + blockIdx.x * blockDim.x;
        z[n] = x[n] + y[n];
    }
    add<<<128, 32>>>(x, y, z);
\end{verbatim}
其中，\verb"x"、\verb"y"~和~\verb"z"~是由~\verb"cudaMalloc()"~分配全局内存的指针。很容易看出，核函数中对这几个指针所指内存区域的访问都是合并的。例如，第一个线程块中的线程束将访问数组~\verb"x"~中第~0-31~个元素，对应~128~字节的连续内存，而且首地址一定是~256~字节的整数倍。这样的访问只需要~4~次数据传输即可完成，所以是合并访问，合并度为~100\%。
\item 乱序的合并访问。将上述核函数稍做修改：
\begin{verbatim}
    void __global__ add_permuted(float *x, float *y, float *z)
    {
        int tid_permuted = threadIdx.x ^ 0x1;
        int n = tid_permuted + blockIdx.x * blockDim.x;
        z[n] = x[n] + y[n];
    }
    add_permuted<<<128, 32>>>(x, y, z);
\end{verbatim}
其中，\verb"threadIdx.x ^ 0x1"~是某种置换操作，作用是将0-31的整数做某种置换（交换两个相邻的数）。第一个线程块中的线程束将依然访问数组~\verb"x"~中第~0-31~个元素，只不过线程号与数组元素指标不完全一致而已。这样的访问是乱序的（或者交叉的）合并访问，合并度也为~100\%。
\item 不对齐的非合并访问。将第一个核函数稍做修改：
\begin{verbatim}
    void __global__ add_offset(float *x, float *y, float *z)
    {
        int n = threadIdx.x + blockIdx.x * blockDim.x + 1;
        z[n] = x[n] + y[n];
    }
    add_offset<<<128, 32>>>(x, y, z);
\end{verbatim}
第一个线程块中的线程束将访问数组~\verb"x"~中第~1-32~个元素。假如数组~\verb"x"~的首地址为~256字节，该线程束将访问设备内存的260-387字节。这将触发5次数据传输，对应的内存地址分别是256-287字节、288-319字节、320-351字节、352-383字节和384-415字节。这样的访问属于不对齐的非合并访问，合并度为~$4/5=80\%$。
\item 跨越式的非合并访问。将第一个核函数改写如下：
\begin{verbatim}
    void __global__ add_stride(float *x, float *y, float *z)
    {
        int n = blockIdx.x + threadIdx.x * gridDim.x;
        z[n] = x[n] + y[n];
    }
    add_stride<<<128, 32>>>(x, y, z);
\end{verbatim}
第一个线程块中的线程束将访问数组~\verb"x"~中指标为~0、128、256、384~等的元素。因为这里的每一对数据都不在一个连续的~32~字节的内存片段，故该线程束的访问将触发~$32$~次数据传输。这样的访问属于跨越式的非合并访问，合并度为~$4/32=12.5\%$。
\item 广播式的非合并访问。将第一个核函数改写如下：
\begin{verbatim}
    void __global__ add_broadcast(float *x, float *y, float *z)
    {
        int n = threadIdx.x + blockIdx.x * blockDim.x;
        z[n] = x[0] + y[n];
    }
    add_broadcast<<<128, 32>>>(x, y, z);
\end{verbatim}
第一个线程块中的线程束将一致地访问数组~\verb"x"~中的第~0~个元素。这只需要一次数据传输（处理~32~字节的数据），但由于整个线程束只使用了~4~字节的数据，故合并度为~$4/32=12.5\%$。这样的访问属于广播式的非合并访问。这样的访问（如果是读数据的话）适合采用第~\ref{chapter:memory}~章提到的常量内存。具体的例子见第~\ref{chapter:md}~章。
\end{enumerate}

\section{例子：矩阵转置}

本节将通过一个矩阵转置的例子讨论全局内存的合理使用。矩阵转置是线性代数中一个基本的操作。我们这里仅考虑行数与列数相等的矩阵，即方阵。学完本节后，读者可以思考如何在~CUDA~中对非方阵进行转置。

假设一个矩阵~$A$~的矩阵元为~$A_{ij}$，则其转置矩阵~$B=A^{T}$~的矩阵元为
\begin{equation}
B_{ij}=\left(A^{T}\right)_{ij} = A_{ji}.
\end{equation}
例如，取
\begin{equation}
A=\left(
\begin{array}{cccc}
0 &1 &2 &3 \\
4 &5 &6 &7\\
8 &9 &10 &11\\
12 &13 &14 &15\\
\end{array}
\right),
\end{equation}
则其转置矩阵为
\begin{equation}
B=A^{T}=\left(
\begin{array}{cccc}
0 &4 &8 &12 \\
1 &5 &9 &13\\
2 &6 &10 &14\\
3 &7 &11 &15\\
\end{array}
\right).
\end{equation}


\subsection{矩阵复制\label{section:matrix_copy}}

在讨论矩阵转置之前，我们先考虑一个更简单的问题：矩阵复制，即形如~$B=A$~的计算。Listing \ref{listing:copy}~给出了矩阵复制核函数~\verb"copy"~的定义和调用。

\begin{lstlisting}[language=C++,caption={本章程序~matrix.cu~中的~copy~函数及其调用。},label={listing:copy}]
__global__ void copy(const real *A, real *B, const int N)
{
    const int nx = blockIdx.x * TILE_DIM + threadIdx.x;
    const int ny = blockIdx.y * TILE_DIM + threadIdx.y;
    const int index = ny * N + nx;
    if (nx < N && ny < N)
    {
        B[index] = A[index];
    }
}

const int grid_size_x = (N + TILE_DIM - 1) / TILE_DIM;
const int grid_size_y = grid_size_x;
const dim3 block_size(TILE_DIM, TILE_DIM);
const dim3 grid_size(grid_size_x, grid_size_y);
copy<<<grid_size, block_size>>>(d_A, d_B, N);
\end{lstlisting}

首先，我们说明一下，在核函数中可以直接使用在函数外部由~\verb"#define"~或~\verb"const"~定义的常量，包括整型常量和浮点型常量，但是在使用微软的编译器（MSVC）时有一个限制，即不能在核函数中使用在函数外部由~\verb"const"~定义的浮点型常量。在本例中，\verb"TILE_DIM"~是一个整型常量，在文件的开头定义：
\begin{verbatim}
    const int TILE_DIM = 32; // C++ 风格
\end{verbatim}
它可以等价地写为
\begin{verbatim}
    #define TILE_DIM 32 // C 风格
\end{verbatim}
可以在核函数中直接使用该常量的值，但要记住不能在核函数中使用这种常量的引用或地址。

再看核函数~\verb"copy"~的执行配置。在调用核函数~\verb"copy"~时，我们用了二维的网格和线程块。在该问题中，并不是一定要使用二维的网格和线程块，因为矩阵中的数据排列本质上依然是一维的。然而，在后面的矩阵转置问题中，使用二维的网格和线程块更为方便。为了保持一致（我们将对比程序中几个核函数的性能），我们这里也用二维的网格和线程块。如上所述，程序中的~\verb"TILE_DIM"~是一个整型常量，取值为~32，指的是一片（tile）矩阵的维度（dimension，即行数）。我们将一片一片地处理一个大矩阵。其中的一片是一个~$32\times 32$~的矩阵。每一个二维的线程块将处理一片矩阵。线程块的维度和一片矩阵的维度一样大，如第~14~行所示。和线程块一致，网格也用二维的，维度为待处理矩阵的维度~\verb"N"~除以线程块维度，如第12-13行和第15行所示。
例如，假如~\verb"N"~为128，则~\verb"grid_size_x"和~\verb"grid_size_y"~都是~$128/32=4$。也就是说，核函数所用网格维度为~$4\times4$，线程块维度为~$32\times32$。此时在核函数~\verb"copy"~中的~\verb"gridDim.x"~和~\verb"gridDim.y"~都等于~4，而~\verb"blockDim.x"~和~\verb"blockDim.y"~都等于~32。读者应该注意到，一个线程块中总的线程数目为~1024，刚好为所允许的最大值。

最后看核函数~\verb"copy"~的实现。第~3~行将矩阵的列指标~\verb"nx"~与带~\verb".x"~后缀的内建变量联系起来，而第~4~行将矩阵的行指标~\verb"ny"~与带~\verb".y"~后缀的内建变量联系起来。第~5~行将上述行指标与列指标结合起来转化成一维指标~\verb"index"。第~6-9~行在行指标和列指标都不越界的情况下将矩阵~\verb"A"~的第~\verb"index"~个元素复制给矩阵~\verb"B"~的第~\verb"index"~个元素。

我们来分析一下核函数中对全局内存的访问模式。在第~\ref{chapter:thread}~章我们介绍过，对于多维数组，\verb"x"~维度的线程指标~\verb"threadIdx.x"~是最内层的（变化最快），所以相邻的~\verb"threadIdx.x"~对应相邻的线程。从核函数中的代码可知，相邻的~\verb"nx"~对应相邻的线程，也对应相邻的数组元素（对~\verb"A"~和~\verb"B"~都成立）。所以，在核函数中，相邻的线程访问了相邻的数组元素，在没有内存不对齐的情况下属于~\ref{section:coalesing}~节介绍的顺序的合并访问。我们取~\verb"N = 10000"~并在GeForce RTX 2080ti中进行测试。采用单精度浮点数，核函数的执行时间为1.6 ms。根据该执行时间，有效的显存带宽为~$500~\rm{GB/s}$，略小于该GPU的理论显存带宽~$616~\rm{GB/s}$。测试矩阵复制计算的性能是为后面讨论矩阵转置核函数的性能确定一个可以比较的基准。第~\ref{chapter:speedup}~章讨论过有效显存带宽的计算，读者可以去回顾一下。

\subsection{使用全局内存进行矩阵转置}

在~\ref{section:matrix_copy}~节我们讨论了矩阵复制的计算。本小节我们讨论矩阵转置的计算。为此，我们回顾一下~\ref{section:matrix_copy}~节矩阵复制核函数中的如下语句：
\begin{verbatim}
    const int index = ny * N + nx;
    if (nx < N && ny < N) B[index] = A[index];
\end{verbatim}
为了便于理解，我们首先将这两条语句写成一条语句：
\begin{verbatim}
    if (nx < N && ny < N) B[ny * N + nx] = A[ny * N + nx];
\end{verbatim}
从数学的角度来看，这相当于做了~$B_{ij}=A_{ij}$~的操作。如果要实现矩阵转置，即~$B_{ij}=A_{ji}$~的操作，可以将上述代码换成
\begin{verbatim}
    if (nx < N && ny < N) B[nx * N + ny] = A[ny * N + nx];
\end{verbatim}
或者
\begin{verbatim}
    if (nx < N && ny < N) B[ny * N + nx] = A[nx * N + ny];
\end{verbatim}
以上两条语句都能实现矩阵转置，但是它们将带来不同的性能。与它们对应的核函数分别为~\verb"transpose1"~和~\verb"transpose2"，分别列于~Listing~\ref{listing:transpose1}~和~Listing~\ref{listing:transpose2}。可以看出，在核函数~\verb"transpose1"~中，对矩阵~\verb"A"~中数据的访问（读取）是顺序的，但对矩阵~\verb"B"~中数据的访问（写入）不是顺序的。在核函数~\verb"transpose2"~中，对矩阵~\verb"A"~中数据的访问（读取）不是顺序的，但对矩阵~\verb"B"~中数据的访问（写入）是顺序的。在不考虑数据是否对齐的情况下，我们可以说核函数~\verb"transpose1"~对矩阵~\verb"A"~和~\verb"B"~的访问分别是合并的和非合并的，而核函数~\verb"transpose2"~对矩阵~\verb"A"~和~\verb"B"~的访问分别是非合并的和合并的。

\begin{lstlisting}[language=C++,caption={本章程序~matrix.cu~中的~transpose1~核函数。},label={listing:transpose1}]
__global__ void transpose1(const real *A, real *B, const int N)
{
    const int nx = blockIdx.x * blockDim.x + threadIdx.x;
    const int ny = blockIdx.y * blockDim.y + threadIdx.y;
    if (nx < N && ny < N)
    {
        B[nx * N + ny] = A[ny * N + nx];
    }
}
\end{lstlisting}


继续用~GeForce RTX 2080ti~测试相关核函数的执行时间（采用单精度浮点数计算）。核函数~\verb"transpose1"~的执行时间为~5.3 ms，而核函数~\verb"transpose2"~的执行时间为~2.8 ms。以上两个核函数中都有一个合并访问和一个非合并访问，但为什么性能差别那么大呢？这是因为，在核函数~\verb"transpose2"中，读取操作虽然是非合并的，但利用了第~\ref{chapter:memory}~章提到的只读数据缓存的加载函数~\verb"__ldg()"。从帕斯卡架构开始，如果编译器能够判断一个全局内存变量在整个核函数的范围都只可读（如这里的矩阵~\verb"A"），则会自动用函数~\verb"__ldg()"~读取全局内存，从而对数据的读取进行缓存，缓解非合并访问带来的影响。对于全局内存的写入，则没有类似的函数可用。这就是以上两个核函数性能差别的根源。所以，在不能同时满足读取和写入都是合并的情况下，一般来说应当尽量做到合并的写入。

\begin{lstlisting}[language=C++,caption={本章程序~matrix.cu~中的~transpose2~核函数。},label={listing:transpose2}]
__global__ void transpose2(const real *A, real *B, const int N)
{
    const int nx = blockIdx.x * blockDim.x + threadIdx.x;
    const int ny = blockIdx.y * blockDim.y + threadIdx.y;
    if (nx < N && ny < N)
    {
        B[ny * N + nx] = A[nx * N + ny];
    }
}
\end{lstlisting}

对于开普勒架构和麦克斯韦架构，默认情况下不会使用~\verb"__ldg()"~函数。用~Tesla K40~测试（采用单精度浮点数计算），核函数~\verb"transpose1"~的执行时间短一些，为12 ms，而核函数~\verb"transpose2"~的执行时间长一些，为23 ms。这与用GeForce RTX 2080ti测试的结果是相反的。在使用开普勒架构和麦克斯韦架构的GPU时，需要明显地使用~\verb"__ldg()"~函数。例如，可将核函数~\verb"transpose2"~改写为核函数~\verb"transpose3"，其中使用~\verb"__ldg()"~函数的代码行为
\begin{verbatim}
    if (nx < N && ny < N) B[ny * N + nx] = __ldg(&A[nx * N + ny]);
\end{verbatim}
完整的核函数见Listing~\ref{listing:transpose3}。该版本的核函数在Tesla K40中的执行时间为8 ms，比核函数~\verb"transpose1"~的执行时间短了一些。

\begin{lstlisting}[language=C++,caption={本章程序~matrix.cu~中的~transpose3~核函数。},label={listing:transpose3}]
__global__ void transpose3(const real *A, real *B, const int N)
{
    const int nx = blockIdx.x * blockDim.x + threadIdx.x;
    const int ny = blockIdx.y * blockDim.y + threadIdx.y;
    if (nx < N && ny < N)
    {
        B[ny * N + nx] = __ldg(&A[nx * N + ny]);
    }
}
\end{lstlisting}

除了利用只读数据缓存加速非合并的访问，有时还可以利用共享内存将非合并的全局内存访问转化为合并的。我们将在第~\ref{chapter:shared}~章讨论这个问题。
