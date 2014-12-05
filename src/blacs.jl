# the BLACS routines
function blacs_pinfo()
    mypnum, nprocs = Array(Int32, 1), Array(Int32, 1)
    ccall((:blacs_pinfo_, libscalapack), Void,
        (Ptr{Int32}, Ptr{Int32}),
        mypnum, nprocs)
    return mypnum[1], nprocs[1]
end

function blacs_gridinfo()
    nprow = Array(Int32, 1)
    npcol = Array(Int32, 1)
    myprow = Array(Int32, 1)
    mypcol = Array(Int32, 1)
    ccall((:blacs_gridinfo_, libscalapack), Void,
        (Ptr{Int32}, Ptr{Int32}, Ptr{Int32}, Ptr{Int32}, Ptr{Int32}),
        ictxt, nprow, npcol, myprow, mypcol)
    return nprow[1], npcol[1], myprow[1], mypcol[1]
end

blacs_gridexit() = ccall((:blacs_gridexit_, libscalapack), Void, (Ptr{Int32},), ictxt)

blacs_exit(cont = 0) = ccall((:blacs_exit_, libscalapack), Void, (Ptr{Int},), &cont)

# end BLACS
