typealias ScaInt Int32 # Fixme! Have to find a way of detecting if this is always the case

#############
# Auxiliary #
#############

# Initialize
function sl_init(nprow::Integer, npcol::Integer)
    ictxt = Array(ScaInt, 1)
    ccall((:sl_init_, libscalapack), Void,
        (Ptr{ScaInt}, Ptr{ScaInt}, Ptr{ScaInt}),
        ictxt, &nprow, &npcol)
    return ictxt[1]
end

# Calculate size of local array
function numroc(n::Integer, nb::Integer, iproc::Integer, isrcproc::Integer, nprocs::Integer)
    ccall((:numroc_, libscalapack), ScaInt,
        (Ptr{ScaInt}, Ptr{ScaInt}, Ptr{ScaInt}, Ptr{ScaInt}, Ptr{ScaInt}),
        &n, &nb, &iproc, &isrcproc, &nprocs)
end

# Array descriptor
function descinit(m::Integer, n::Integer, mb::Integer, nb::Integer, irsrc::Integer, icsrc::Integer, ictxt::Integer, lld::Integer)

    # extract values
    nprow, npcol, myrow, mycol = BLACS.gridinfo(ictxt)
    locrm = numroc(m, mb, myrow, irsrc, nprow)

    # checks
    m >= 0 || throw(ArgumentError("first dimension must be non-negative"))
    n >= 0 || throw(ArgumentError("second dimension must be non-negative"))
    mb > 0 || throw(ArgumentError("first dimension blocking factor must be positive"))
    nb > 0 || throw(ArgumentError("second dimension blocking factor must be positive"))
    0 <= irsrc < nprow || throw(ArgumentError("process row must be positive and less that grid size"))
    0 <= irsrc < nprow || throw(ArgumentError("process column must be positive and less that grid size"))
    # lld >= locrm || throw(ArgumentError("leading dimension of local array is too small"))

    # allocation
    desc = Array(ScaInt, 9)
    info = Array(ScaInt, 1)

    # ccall
    ccall((:descinit_, libscalapack), Void,
        (Ptr{ScaInt}, Ptr{ScaInt}, Ptr{ScaInt}, Ptr{ScaInt},
         Ptr{ScaInt}, Ptr{ScaInt}, Ptr{ScaInt}, Ptr{ScaInt},
         Ptr{ScaInt}, Ptr{ScaInt}),
        desc, &m, &n, &mb,
        &nb, &irsrc, &icsrc, &ictxt,
        &lld, info)

    info[1] == 0 || error("input argument $(info[1]) has illegal value")

    return desc
end

# Redistribute arrays
for (fname, elty) in ((:psgemr2d_, :Float32),
                      (:pdgemr2d_, :Float64),
                      (:pcgemr2d_, :Complex64),
                      (:pzgemr2d_, :Complex128))
    @eval begin
        function pxgemr2d!(m::Integer, n::Integer, A::Matrix{$elty}, ia::Integer, ja::Integer, desca::Vector{ScaInt}, B::Matrix{$elty}, ib::Integer, jb::Integer, descb::Vector{ScaInt}, ictxt::Integer)

            ccall(($(string(fname)), libscalapack), Void,
                (Ptr{ScaInt}, Ptr{ScaInt}, Ptr{$elty}, Ptr{ScaInt},
                 Ptr{ScaInt}, Ptr{ScaInt}, Ptr{$elty}, Ptr{ScaInt},
                 Ptr{ScaInt}, Ptr{ScaInt}, Ptr{ScaInt}),
                &m, &n, A, &ia,
                &ja, desca, B, &ib,
                &jb, descb, &ictxt)
        end
    end
end

##################
# Linear Algebra #
##################

# Matmul
for (fname, elty) in ((:psgemm_, :Float32),
                      (:pdgemm_, :Float64),
                      (:pcgemm_, :Complex64),
                      (:pzgemm_, :Complex128))
    @eval begin
        function pdgemm!(transa::Char, transb::Char, m::Integer, n::Integer, k::Integer, α::$elty, A::Matrix{$elty}, ia::Integer, ja::Integer, desca::Vector{ScaInt}, B::Matrix{$elty}, ib::Integer, jb::Integer, descb::Vector{ScaInt}, β::$elty, C::Matrix{$elty}, ic::Integer, jc::Integer, descc::Vector{ScaInt})

            ccall(($(string(fname)), libscalapack), Void,
                (Ptr{UInt8}, Ptr{UInt8}, Ptr{ScaInt}, Ptr{ScaInt},
                 Ptr{ScaInt}, Ptr{$elty}, Ptr{$elty}, Ptr{ScaInt},
                 Ptr{ScaInt}, Ptr{ScaInt}, Ptr{$elty}, Ptr{ScaInt},
                 Ptr{ScaInt}, Ptr{ScaInt}, Ptr{$elty}, Ptr{$elty},
                 Ptr{ScaInt}, Ptr{ScaInt}, Ptr{ScaInt}),
                &transa, &transb, &m, &n,
                &k, &α, A, &ia,
                &ja, desca, B, &ib,
                &jb, descb, &β, C,
                &ic, &jc, descc)
        end
    end
end

# Eigensolves
for (fname, elty) in ((:psstedc_, :Float32),
                      (:pdstedc_, :Float64))
    @eval begin
        function pxstedc!(compz::Char, n::Integer, d::Vector{$elty}, e::Vector{$elty}, Q::Matrix{$elty}, iq::Integer, jq::Integer, descq::Vector{ScaInt})


            work    = $elty[0]
            lwork   = convert(ScaInt, -1)
            iwork   = ScaInt[0]
            liwork  = convert(ScaInt, -1)
            info    = ScaInt[0]

            for i = 1:2
                ccall(($(string(fname)), libscalapack), Void,
                    (Ptr{UInt8}, Ptr{UInt8}, Ptr{$elty}, Ptr{$elty},
                     Ptr{$elty}, Ptr{ScaInt}, Ptr{ScaInt}, Ptr{ScaInt},
                     Ptr{$elty}, Ptr{ScaInt}, Ptr{ScaInt}, Ptr{ScaInt},
                     Ptr{$ScaInt}),
                    &compz, &n, d, e,
                    Q, &iq, &jq, descq,
                    work, &lwork, iwork, &liwork,
                    info)

                if i == 1
                    lwork = convert(ScaInt, work[1])
                    work = Array($elty, lwork)
                    liwork = convert(ScaInt, iwork[1])
                    iwork = Array(ScaInt, liwork)
                end
            end

            return d, Q
        end
    end
end

# SVD solver
for (fname, elty) in ((:psgesvd_, :Float32),
                      (:pdgesvd_, :Float64))
    @eval begin
        function pxgesvd!(jobu::Char, jobvt::Char, m::Integer, n::Integer, A::StridedMatrix{$elty}, ia::Integer, ja::Integer, desca::Vector{ScaInt}, s::StridedVector{$elty}, U::StridedMatrix{$elty}, iu::Integer, ju::Integer, descu::Vector{ScaInt}, Vt::Matrix{$elty}, ivt::Integer, jvt::Integer, descvt::Vector{ScaInt})
            # extract values

            # check

            # allocate
            info = Array(ScaInt, 1)
            work = Array($elty, 1)
            lwork = -1

            # ccall
            for i = 1:2
                ccall(($(string(fname)), libscalapack), Void,
                    (Ptr{UInt8}, Ptr{UInt8}, Ptr{ScaInt}, Ptr{ScaInt},
                     Ptr{$elty}, Ptr{ScaInt}, Ptr{ScaInt}, Ptr{ScaInt},
                     Ptr{$elty}, Ptr{$elty}, Ptr{ScaInt}, Ptr{ScaInt},
                     Ptr{ScaInt}, Ptr{$elty}, Ptr{ScaInt}, Ptr{ScaInt},
                     Ptr{ScaInt}, Ptr{$elty}, Ptr{ScaInt}, Ptr{ScaInt}),
                    &jobu, &jobvt, &m, &n,
                    A, &ia, &ja, desca,
                    s, U, &iu, &ju,
                    descu, Vt, &ivt, &jvt,
                    descvt, work, &lwork, info)
                if i == 1
                    lwork = convert(ScaInt, work[1])
                    work = Array($elty, lwork)
                end
            end

            if 0 < info[1] <= min(m,n)
                throw(ScaLAPACKException(info[1]))
            end

            return U, s, Vt
        end
    end
end
for (fname, elty, relty) in ((:pcgesvd_, :Complex64, :Float32),
                             (:pzgesvd_, :Complex128, :Float64))
    @eval begin
        function pxgesvd!(jobu::Char, jobvt::Char, m::Integer, n::Integer, A::Matrix{$elty}, ia::Integer, ja::Integer, desca::Vector{ScaInt}, s::Vector{$relty}, U::Matrix{$elty}, iu::Integer, ju::Integer, descu::Vector{ScaInt}, Vt::Matrix{$elty}, ivt::Integer, jvt::Integer, descvt::Vector{ScaInt})
            # extract values

            # check

            # allocate
            info = Array(ScaInt, 1)
            work = Array($elty, 1)
            rwork = Array($relty, 1 + 4*max(m, n))
            lwork = -1

            # ccall
            for i = 1:2
                ccall(($(string(fname)), libscalapack), Void,
                    (Ptr{UInt8}, Ptr{UInt8}, Ptr{ScaInt}, Ptr{ScaInt},
                     Ptr{$elty}, Ptr{ScaInt}, Ptr{ScaInt}, Ptr{ScaInt},
                     Ptr{$relty}, Ptr{$elty}, Ptr{ScaInt}, Ptr{ScaInt},
                     Ptr{ScaInt}, Ptr{$elty}, Ptr{ScaInt}, Ptr{ScaInt},
                     Ptr{ScaInt}, Ptr{$elty}, Ptr{ScaInt}, Ptr{$relty},
                     Ptr{ScaInt}),
                    &jobu, &jobvt, &m, &n,
                    A, &ia, &ja, desca,
                    s, U, &iu, &ju,
                    descu, Vt, &ivt, &jvt,
                    descvt, work, &lwork, rwork,
                    info)
                if i == 1
                    lwork = convert(ScaInt, work[1])
                    work = Array($elty, lwork)
                end
            end

            info[1] > 0 && throw(ScaLAPACKException(info[1]))

            return U, s, Vt
        end
    end
end

