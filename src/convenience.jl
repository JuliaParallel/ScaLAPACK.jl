import Base.LinAlg: svdvals!

function svdvals!{T<:BlasFloat}(A::DArray{T,2}, mbB::Integer)

    # problem size
    m, n = size(A)
    mGrid, nGrid = size(A.chunks)
    mbA = ceil(Int32, size(A, 1)/size(A.chunks, 1))
    nbA = ceil(Int32, size(A, 2)/size(A.chunks, 2))

    # Check that array distribution is feasible for ScaLAPACK
    if !all(diff(diff(A.cuts[1]))[1:end-1] .== 0) || !all(diff(diff(A.cuts[2]))[1:end-1] .== 0)
        throw(ArgumentError("the distributions of the array does not fit to the requirements of ScaLAPACK. All blocks must have the same size except for the last in row and column"))
    end

    # mb == nb || throw(DimensionMismatch("solver requires row and column block sizes to be the same"))

    vals = RemoteRef[]

    @sync for p in MPI.workers()
        # initialize grid
        valsp = @spawnat p begin

            id, nprocs = BLACS.pinfo()
            ic = BLACS.gridinit(BLACS.get(0, 0), 'c', mGrid, nGrid)

            # who am I?
            nprow, npcol, myrow, mycol = BLACS.gridinfo(ic)
            npA = numroc(m, mbA, myrow, 0, nprow)
            nqA = numroc(n, nbA, mycol, 0, npcol)

            npB = numroc(m, mbB, myrow, 0, nprow)
            nqB = numroc(n, mbB, mycol, 0, npcol)

            # print("myrow: $myrow, mycol: $mycol, mbB: $mbB, npB: $npB, nqB: $nqB\n")

            if nprow >= 0 && npcol >= 0
                # Get DArray info
                dA = descinit(m, n, mbA, nbA, 0, 0, ic, npA)
                dB = descinit(m, n, mbB, mbB, 0, 0, ic, npB)

                B = Array(T, npB, nqB)
                pxgemr2d!(m, n, localpart(A), 1, 1, dA, B, 1, 1, dB, ic)
                # display(B)

                # calculate DSVD
                V, s, U = pxgesvd!('N', 'N', m, n, B, 1, 1, dB, Array(typeof(real(one(T))), min(m, n)), Array(T, 0, 0), 0, 0, dB, Array(T, 0, 0), 0, 0, dB)

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