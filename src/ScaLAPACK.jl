module ScaLAPACK

# const libscalapack = "/usr/local/lib/libscalapack.dylib"
const libscalapack = "/usr/lib/libscalapack-openmpi.so"

include("blacs.jl")
include("scalapackWrappers.jl")
include("convenience.jl")

end # module
