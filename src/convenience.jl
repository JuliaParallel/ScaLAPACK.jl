import Base.LinAlg: svdvals!

function check_distribution{T}(A::DArray{T,2})
    if !all(diff(diff(A.cuts[1]))[1:end-1] .== 0) || !all(diff(diff(A.cuts[2]))[1:end-1] .== 0)
        throw(ArgumentError("the distributions of the array does not fit to the requirements of ScaLAPACK. All blocks must have the same size except for the last in row and column"))
    end
    true
end

function A_mul_B!{T<:BlasFloat}(α::T, A::DArray{T,2,Matrix{T}}, B::DArray{T,2,Matrix{T}}, β::T, C::DArray{T,2,Matrix{T}}, blocksize1 = max(round(Integer, size(C, 1)/100), 10), blocksize2 = max(round(Integer, size(C, 2)/100), 10))

    # extract
    mA, nA = size(A)
    mB, nB = size(B)
    mC, nC = size(C)
    k = nA
    mGrid, nGrid = size(A.chunks)
    mbA = ceil(Int32, mA/size(A.chunks, 1))
    nbA = ceil(Int32, nA/size(A.chunks, 2))
    mbB = ceil(Int32, mB/size(B.chunks, 1))
    nbB = ceil(Int32, nB/size(B.chunks, 2))
    mbC = ceil(Int32, mC/size(C.chunks, 1))
    nbC = ceil(Int32, nC/size(C.chunks, 2))

    # check
    if mA != mC || nA != mB || nB != nC
        throw(DimensionMismatch("shapes don't fit"))
    end
    check_distribution(A)
    check_distribution(B)
    check_distribution(C)

    @sync for p in MPI.workers()
        # initialize grid
        @spawnat p begin

            id, nprocs = BLACS.pinfo()
            ic = BLACS.gridinit(BLACS.get(0, 0), 'c', mGrid, nGrid)

            # who am I?
            nprow, npcol, myrow, mycol = BLACS.gridinfo(ic)
            npA = numroc(mA, mbA, myrow, 0, nprow)
            nqA = numroc(nA, nbA, mycol, 0, npcol)
            npB = numroc(mB, mbB, myrow, 0, nprow)
            nqB = numroc(nB, nbB, mycol, 0, npcol)
            npC = numroc(mC, mbC, myrow, 0, nprow)
            nqC = numroc(nC, nbC, mycol, 0, npcol)

            npAnew = numroc(mA, blocksize1, myrow, 0, nprow)
            nqAnew = numroc(nA, blocksize2, mycol, 0, npcol)
            npBnew = numroc(mB, blocksize1, myrow, 0, nprow)
            nqBnew = numroc(nB, blocksize2, mycol, 0, npcol)
            npCnew = numroc(mC, blocksize1, myrow, 0, nprow)
            nqCnew = numroc(nC, blocksize2, mycol, 0, npcol)

            # print("myrow: $myrow, mycol: $mycol, npAnew: $npAnew, npCnew: $npCnew, npCnew: $npCnew\n")

            if nprow >= 0 && npcol >= 0
                # Get DArray info
                dA    = descinit(mA, nA, mbA, nbA, 0, 0, ic, npA)
                dAnew = descinit(mA, nA, blocksize1, blocksize2, 0, 0, ic, npAnew)
                dB    = descinit(mB, nB, mbB, nbB, 0, 0, ic, npB)
                dBnew = descinit(mB, nB, blocksize1, blocksize2, 0, 0, ic, npBnew)
                dC    = descinit(mC, nC, mbC, nbC, 0, 0, ic, npC)
                dCnew = descinit(mC, nC, blocksize1, blocksize2, 0, 0, ic, npCnew)

                # redistribute
                Anew = Array(T, npAnew, nqAnew)
                pxgemr2d!(mA, nA, localpart(A), 1, 1, dA, Anew, 1, 1, dAnew, ic)
                Bnew = Array(T, npBnew, nqBnew)
                pxgemr2d!(mB, nB, localpart(B), 1, 1, dB, Bnew, 1, 1, dBnew, ic)
                Cnew = Array(T, npCnew, nqCnew)
                pxgemr2d!(mC, nC, localpart(C), 1, 1, dC, Cnew, 1, 1, dCnew, ic)


                # calculate
                pdgemm!('N', 'N', mC, nC, k, α, Anew, 1, 1, dAnew, Bnew, 1, 1, dBnew, β, Cnew, 1, 1, dCnew)

                # move result back to C
                pxgemr2d!(mC, nC, Cnew, 1, 1, dCnew, localpart(C), 1, 1, dC, ic)

                # cleanup
                BLACS.gridexit(ic)
            end
        end
    end
    C
end

function eigvals!{T<:BlasReal}(d::Vector{T}, e::Vector{T}, blocksize = max(div(length(d), 100), 10))

    # Extract
    n = length(d)

    # Check
    if length(e) != n - 1
        throw(DimensionMismatch("off diagonal vector must have length $(n-1) but had length $(length(e))"))
    end

    @sync for p in MPI.workers()
        # initialize grid
        @spawnat p begin

            id, nprocs = BLACS.pinfo()
            ic = BLACS.gridinit(BLACS.get(0, 0), 'c', mGrid, nGrid)

            # who am I?
            nprow, npcol, myrow, mycol = BLACS.gridinfo(ic)
            np = numroc(m, blocksize, myrow, 0, nprow)
            nq = numroc(n, blocksize, mycol, 0, npcol)

            # print("myrow: $myrow, mycol: $mycol, mbB: $mbB, npB: $npB, nqB: $nqB\n")

            if nprow >= 0 && npcol >= 0

                dQ = descinit(n, n, blocksize, blocksize, 0, 0, ic, npB)

                # calculate DSVD
                λ, V = pxstdnc!('N', n, d, e, Array(T, 0, 0), 1, 1)

                # show result
                # myrow == 0 && mycol == 0 && println(s[1:3])

                # clean up
                BLACS.gridexit(ic)
            end
        end
    end
    return d
end

function svdvals!{T<:BlasFloat}(A::DArray{T,2,Matrix{T}}, blocksize::Integer = max(10, round(Integer, minimum(size(A))/10)))

    # problem size
    m, n = size(A)
    mGrid, nGrid = size(A.chunks)
    mbA = ceil(Int32, size(A, 1)/size(A.chunks, 1))
    nbA = ceil(Int32, size(A, 2)/size(A.chunks, 2))
    mbB = blocksize

    # Check that array distribution is feasible for ScaLAPACK
    check_distribution(A)

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

                # calculate DSVD
                U, s, Vt = pxgesvd!('N', 'N', m, n, B, 1, 1, dB, Array(typeof(real(one(T))), min(m, n)), Array(T, 0, 0), 0, 0, dB, Array(T, 0, 0), 0, 0, dB)

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
