import Base.LinAlg: svdvals!

function svdvals!{T<:BlasFloat}(A::DArray{T,2}, nb = min(ceil(Int32, size(A, 1)/size(A.chunks, 1)), ceil(Int32, size(A, 2)/size(A.chunks, 2))))

    # problem size
    m, n = size(A)
    m == n || throw(DimensionMismatch("Non-square problems not handled yet"))
    mGrid, nGrid = size(A.chunks)

    # Check that grid is feasible for ScaLAPACK (notice that
    # nb == length(A.indexes[1][1]) == length(A.indexes[1][2]) || throw(DimensionMismatch(""))
    # mb = ceil(Int32, m/mGrid)
    # nb = ceil(Int32, n/nGrid)
    # mb, nb = 1, 1

    # mb == nb || throw(DimensionMismatch("solver requires row and column block sizes to be the same"))

    vals = RemoteRef[]

    @sync for p in MPI.workers()
        # initialize grid
        valsp = @spawnat p begin

            id, nprocs = BLACS.pinfo()
            ic = BLACS.gridinit(BLACS.get(0, 0), 'c', mGrid, nGrid)

            # who am I?
            nprow, npcol, myrow, mycol = BLACS.gridinfo(ic)
            np = numroc(m, nb, myrow, 0, nprow)
            # nq = numroc(n, nb, mycol, 0, npcol)

            # print("myrow: $myrow, mycol: $mycol, nb: $nb, np: $np, nq: $nq\n")
            # display(localpart(A))

            if nprow >= 0 && npcol >= 0
                # Get DArray info
                dA = descinit(m, n, nb, nb, 0, 0, ic, np)

                # calculate DSVD
                V, s, U = pxgesvd!('N', 'N', n, n, localpart(A), 1, 1, dA, Array(typeof(real(one(T))), n), Array(T, 0, 0), 0, 0, dA, Array(T, 0, 0), 0, 0, dA)

                # show result
                # myrow == 0 && mycol == 0 && println(s[1:3])

                # clean up
                BLACS.gridexit(ic)
            end
            s
        end
        push!(vals, valsp)
    end
    # end
    return fetch(vals[1])
end