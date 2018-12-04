/*
 * Licensed to the Apache Software Foundation (ASF) under one
 * or more contributor license agreements.  See the NOTICE file
 * distributed with this work for additional information
 * regarding copyright ownership.  The ASF licenses this file
 * to you under the Apache License, Version 2.0 (the
 * "License"); you may not use this file except in compliance
 * with the License.  You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.

 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace Proxem.BlasNet
{
    /// <summary>
    /// Start the provider for Blas using the either mkl_rt, mklml or C# implementations 
    /// for basic array functions
    /// </summary>
    public static class StartProvider
    {
        /// <summary>
        /// Set Blas Provider as a DefaultBlas (pure C# implementation)
        /// This will be way slower than when using MKL and Lapack functions won't be available
        /// </summary>
        public static void LaunchDefault()
        {
            Blas.Provider = new DefaultBlas();
        }

        /// <summary>
        /// Creates a new DynMkl with the given args and sets it as the default Blas implementation.
        /// </summary>
        public static void LaunchMklRt(int max = -1, string path = null)
        {
            if (IsLinux)
            {
                Blas.Provider = new DynMklRtLinux(max);
                Lapack.Provider = new LapackRtLinux();
            }
            else
            {
                Blas.Provider = new DynMklRtWindows(max, path);
                Lapack.Provider = new LapackRtWindows();
            }
        }

        /// <summary>
        /// Creates a new MklMl with the given args and sets it as the default Blas implementation.
        /// </summary>
        public static void LaunchMklMl(int max = -1, string path = null)
        {
            if (IsLinux)
            {
                Blas.Provider = new DynMklMlLinux(max);
                Lapack.Provider = new LapackMlLinux();
            }
            else
            {
                Blas.Provider = new DynMklMlWindows(max, path);
                Lapack.Provider = new LapackMlWindows();
            }
        }

        /// <summary>
        /// Creates a new Acml with the given args and sets it as the default Blas implementation.
        /// </summary>
        public static void LaunchAcml(int max = -1, string path = null)
        {
            if (IsLinux)
            {
                Blas.Provider = new DynAcmlLinux(max);
                Lapack.Provider = null; // TODO : Lapack with Acml? remove Acml?
            }
            else
            {
                Blas.Provider = new DynAcmlWindows(max, path);
                Lapack.Provider = null;
            }
        }

        private static bool IsLinux
        {
            get
            {
                int p = (int)Environment.OSVersion.Platform;
                return (p == 4) || (p == 6) || (p == 128);
            }
        }
    }
}
