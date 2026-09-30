"""CUDA transpose for three-dimensional real stencil matrices."""

import numpy as np

_KERNEL = None

_SOURCE = r'''
extern "C" __global__
void stencil_transpose_3d(const double* M, double* Mt,
 const int s0,const int s1,const int s2,
 const int p0,const int p1,const int p2,
 const int a0,const int a1,const int a2,
 const int so0,const int so1,const int so2,
 const int eo0,const int eo1,const int eo2,
 const int po0,const int po1,const int po2,
 const int q0,const int q1,const int q2,
 const long ms0,const long ms1,const long ms2,const long ms3,const long ms4,const long ms5,
 const long ts0,const long ts1,const long ts2,const long ts3,const long ts4,const long ts5)
{
 long long tid=(long long)blockIdx.x*blockDim.x+threadIdx.x;
 const int nr0=eo0-so0+1,nr1=eo1-so1+1,nr2=eo2-so2+1;
 const long long total=(long long)nr0*nr1*nr2*q0*q1*q2;
 if(tid>=total)return;
 int d2=tid%q2; tid/=q2;
 int d1=tid%q1; tid/=q1;
 int d0=tid%q0; tid/=q0;
 int il2=tid%nr2; tid/=nr2;
 int il1=tid%nr1; int il0=tid/nr1;
 const int nd0=(il0==nr0-1)?2*p0+a0:2*p0+1;
 const int nd1=(il1==nr1-1)?2*p1+a1:2*p1+1;
 const int nd2=(il2==nr2-1)?2*p2+a2:2*p2+1;
 if(d0>=nd0||d1>=nd1||d2>=nd2)return;
 const int i0=so0+il0,i1=so1+il1,i2=so2+il2;
 const int j0=i0-p0+d0,j1=i1-p1+d1,j2=i2-p2+d2;
 const int jl0=j0-s0,jl1=j1-s1,jl2=j2-s2;
 const long mi=(long)(p0+jl0)*ms0+(long)(p1+jl1)*ms1+(long)(p2+jl2)*ms2
              +(long)(po0+i0-j0)*ms3+(long)(po1+i1-j1)*ms4+(long)(po2+i2-j2)*ms5;
 const long ti=(long)(po0+il0)*ts0+(long)(po1+il1)*ts1+(long)(po2+il2)*ts2
              +(long)d0*ts3+(long)d1*ts4+(long)d2*ts5;
 Mt[ti]=M[mi];
}
'''


def supports(matrix, out, conjugate=False):
    return (
        not conjugate
        and matrix.ndim == 6
        and out.ndim == 6
        and matrix.dtype == np.float64
        and out.dtype == np.float64
    )


def device_transpose_3d(matrix, out, **args):
    """Transpose ``matrix`` into ``out`` using current stencil metadata."""
    import cupy as cp

    global _KERNEL
    if _KERNEL is None:
        _KERNEL = cp.RawKernel(_SOURCE, "stencil_transpose_3d")

    def ints(name):
        value = args[name]
        if hasattr(value, "get"):
            value = value.get()
        return tuple(int(x) for x in np.asarray(value).reshape(-1))

    s_in, p_in, add = ints("s_in"), ints("p_in"), ints("add")
    s_out, e_out, p_out = ints("s_out"), ints("e_out"), ints("p_out")
    nrows = tuple(e - s + 1 for s, e in zip(s_out, e_out))
    ndiags = tuple(max(2 * p + 1, 2 * p + a) for p, a in zip(p_in, add))
    total = int(np.prod(nrows) * np.prod(ndiags))
    strides_m = tuple(s // matrix.itemsize for s in matrix.strides)
    strides_t = tuple(s // out.itemsize for s in out.strides)
    out.fill(0.0)
    threads = 256
    kernel_args = (
        matrix, out,
        *s_in, *p_in, *add, *s_out, *e_out, *p_out, *ndiags,
        *strides_m, *strides_t,
    )
    _KERNEL(((total + threads - 1) // threads,), (threads,), kernel_args)
    return out
