using MPI

module ScaLAPACK

using Compat

using Base.LinAlg: BlasFloat, BlasReal

using MPI, DistributedArrays

import DistributedArrays: DArray, defaultdist
# this should only be a temporary solution until procs returns a type that encodes more information about the processes
DArray(init, dims, manager::MPIManager, args...) = DArray(init, dims, collect(values(manager.mpi2j))[sortperm(collect(keys(manager.mpi2j)))], args...)
function defaultdist(sz::Int, nc::Int)
    if sz >= nc
        d, r = divrem(sz, nc)
        if r == 0
            return vcat(1:d:sz+1)
        end
        return vcat(vcat(1:d+1:sz+1), [sz+1])
    else
        return vcat(vcat(1:(sz+1)), zeros(Int, nc-sz))
    end
end


if myid() > 1
    MPI.Initialized() || MPI.Init()
end

immutable ScaLAPACKException <: Exception
    info::Int32
end

# const libscalapack = "/Users/andreasnoack/Downloads/scalapack-2.0.2/build/lib/libscalapack.dylib"
const libscalapack = "/usr/local/lib/libscalapack.dylib"
# const libscalapack = "/usr/lib/libscalapack-openmpi.so"

include("blacs.jl")
include("scalapackWrappers.jl")
include("convenience.jl")

end # module
