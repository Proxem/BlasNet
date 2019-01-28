# BlasNet
BlasNet is an optimized library for basic linear algebra operations on arrays written in C\# and developed at [Proxem](https://proxem.com).

## Table of contents

* [Requirements](#requirements)
   * [Intel MKL](#intel-mkl)    
* [Nuget Package](#nuget-package)
* [Contact](#contact) 
* [License](#license)

## Requirements

BlasNet is developed in .Net Standard 2.0 and is compatible with both .Net Framework and .Net Core thus working on Windows and Linux platform.
For Mac OS users there shouldn't be any problem but we didn't test extensively.

### Intel MKL

Intel's math kernel library is used as a backend to optimize arrays operations when available. 
In order to use MKL you must launch the provider in your code. This step depends of your operating system :

#### Windows Users

You first need to download [Intel's MKL](https://software.intel.com/en-us/mkl/choose-download/windows) libraries for Windows. Check that the folder contains `mkl_rt.dll` which is the main library used by BlasNet.
To set the provider to use `mkl_rt.dll` for the low level array operations place the following line at the beginning of your code :

```
using BlasNet;

public namespace Test
{
    public static void Main(string[] args)
    {
        StartProvider.LaunchMklRt(max, path);
        // rest of the code;
    }
}
```

where ```max``` is the max number of threads used by MKL (-1 to use the max number) and `path` is the path to `mkl_rt.dll`.

#### Linux Users

You first need to download [Intel's MKL](https://software.intel.com/en-us/mkl) libraries for Linux. Check that the folder contains `libmkl_rt.so` which is the main library used by BlasNet.
To set the provider to use `libmkl_rt.so` for the low level array you first need to make the libraries visible through the `LD_LIBRARY_PATH` variable. 
To do so, add the path to Intel's libraries by running the following commands : 

```
$  echo "/path/to/mkl/lib/" >> /etc/ld.so.conf.d/intel.conf
$  ldconfig
```

Now, to set the provider to use `libmkl_rt.so` for the low level operations place the following line at the beginning of your code :

```
using BlasNet;

public namespace Test
{
    public static void Main(string[] args)
    {
        StartProvider.LaunchMklRt(max);
        // rest of the code;
    }
}
```
where `max` is the max number of threads used by MKL (-1 to use the max number).


## Nuget Package

We provide a Nuget Package of **BlasNet** to facilitate its use. It's available on [Nuget.org](https://www.nuget.org/packages/Proxem.BlasNet/). 
Symbols are also available to facilitate debugging inside the package.

## Contact

If you can't make **BlasNet** work on your computer or if you have any tracks of improvement drop us an e-mail at one of the following address:
- thp@proxem.com
- joc@proxem.com

## License

BlasNet is Licensed to the Apache Software Foundation (ASF) under one or more contributor license agreements.
See the NOTICE file distributed with this work for additional information regarding copyright ownership.
The ASF licenses this file to you under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with the License.
You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and limitations under the License.
