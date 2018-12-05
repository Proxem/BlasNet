-------------------------------
------  Proxem.BlasNet --------
-------------------------------

Base operations on array used for higher level functions in NumNet arrays.
Blas Provider uses either 'mkl_rt.dll' ('libmkl_rt.so' on Linux) or 'mklml.dll' ('libmklml.so' in Linux) for those base operations
A default provider is implemented in C# but we do not recommend using it as it is way slower.

The provider is initialized with its default value. On windows one must use StartProvider.LaunchMkl or StartProvider.LaunchMklMl to use MKL
and provide the path where the MKL '.dll' are stored in argument. On Linux OS, providing the path is not necessary but the libraries need to be in a folder
specified in LD_LIBRARY_PATH (/usr/lib/ for instance) 
