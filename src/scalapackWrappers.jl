function sl_init(nprow::Integer, npcol::Integer)
    ictxt = Array(Int32, 1)
    ccall((:sl_init_, libscalapack), Void,
        (Ptr{Int32}, Ptr{Int32}, Ptr{Int32}),
        ictxt, &nprow, &npcol)
    return ictxt[1]
end

function numroc(n::Integer, nb::Integer, iproc::Integer, isrcproc::Integer, nprocs::Integer)
    ccall((:numroc_, libscalapack), Int32,
        (Ptr{Int32}, Ptr{Int32}, Ptr{Int32}, Ptr{Int32}, Ptr{Int32}),
        &n, &nb, &iproc, &isrcproc, &nprocs)
end

function descinit(m::Integer, n::Integer, mb::Integer, nb::Integer, irsrc::Integer, icsrc::Integer, ictxt::Integer, lld::Integer)

    # extract values
    nprow, npcol, myrow, mycol = blacs_gridinfo(ictxt)
    locrm = numroc(m, mb, myrow, irsrc, nprow)

    # checks
    m >= 0 || throw(ArgumentError("first dimension must be non-negative"))
    n >= 0 || throw(ArgumentError("second dimension must be non-negative"))
    mb > 0 || throw(ArgumentError("first dimension blocking factor must be positive"))
    nb > 0 || throw(ArgumentError("second dimension blocking factor must be positive"))
    0 <= irsrc < nprow || throw(ArgumentError("process row must be positive and less that grid size"))
    0 <= irsrc < nprow || throw(ArgumentError("process column must be positive and less that grid size"))
    lld >= locrm || throw(ArgumentError("leading dimension of local array is too small"))

    # allocation
    desc = Array(Int32, 9)
    info = Array(Int32, 1)

    # ccall
    ccall((:descinit_, libscalapack), Void,
        (Ptr{Int32}, Ptr{Int32}, Ptr{Int32}, Ptr{Int32},
         Ptr{Int32}, Ptr{Int32}, Ptr{Int32}, Ptr{Int32},
         Ptr{Int32}, Ptr{Int32}),
        desc, &m, &n, &mb,
        &nb, &irsrc, &icsrc, &ictxt,
        &lld, info)

    info[1] == 0 || error("input argument $(info[1]) has illegal value")

    return desc
end

function pdgesvd!(jobu::Char, jobvt::Char, m::Integer, n::Integer, A::StridedMatrix{Float64}, ia::Integer, ja::Integer, desca::Vector{Int32}, s::StridedVector{Float64}, U::StridedMatrix{Float64}, iu::Integer, ju::Integer, descu::Vector{Int32}, Vt::Matrix{Float64}, ivt::Integer, jvt::Integer, descvt::Vector{Int32})
    # extract values

    # check

    # allocate
    info = Array(Int32, 1)
    work = Array(Float64, 1)
    lwork = -1

    # ccall
    for i = 1:2
        ccall((:pdgesvd_, libscalapack), Void,
            (Ptr{Uint8}, Ptr{Uint8}, Ptr{Int32}, Ptr{Int32},
             Ptr{Float64}, Ptr{Int32}, Ptr{Int32}, Ptr{Int32},
             Ptr{Float64}, Ptr{Float64}, Ptr{Int32}, Ptr{Int32},
             Ptr{Int32}, Ptr{Float64}, Ptr{Int32}, Ptr{Int32},
             Ptr{Int32}, Ptr{Float64}, Ptr{Int32}, Ptr{Int32}),
            &jobu, &jobvt, &m, &n,
            A, &ia, &ja, desca,
            s, U, &iu, &ju,
            descu, Vt, &ivt, &jvt,
            descvt, work, &lwork, info)
        if i == 1
            lwork = int(work[1])
            work = Array(Float64, lwork)
        end
    end

    info[1] > 0 && error("ScaLAPACK error code $(info[1])")

    return U, s, Vt
end
