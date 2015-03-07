using MPI

module ScaLAPACK

using Base.LinAlg: BlasFloat, BlasReal

using MPI
# , DistributedArrays

if myid() > 1
    MPI.Initialized() || MPI.Init()
end

immutable ScaLAPACKException <: Exception
    info::Int32
end

const libscalapack = "/usr/local/lib/libscalapack.dylib"
# const libscalapack = "/usr/lib/libscalapack-openmpi.so"

include("blacs.jl")
include("scalapackWrappers.jl")
include("convenience.jl")

end # module
