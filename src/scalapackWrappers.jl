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

# SVD solver
for (fname, elty) in ((:psgesvd_, :Float32),
                      (:pdgesvd_, :Float64))
    @eval begin
        function pxgesvd!(jobu::Char, jobvt::Char, m::Integer, n::Integer, A::StridedMatrix{$elty}, ia::Integer, ja::Integer, desca::Vector{Int32}, s::StridedVector{$elty}, U::StridedMatrix{$elty}, iu::Integer, ju::Integer, descu::Vector{Int32}, Vt::Matrix{$elty}, ivt::Integer, jvt::Integer, descvt::Vector{Int32})
            # extract values

            # check

            # allocate
            info = Array(Int32, 1)
            work = Array($elty, 1)
            lwork = -1

            # ccall
            for i = 1:2
                ccall(($(string(fname)), libscalapack), Void,
                    (Ptr{Uint8}, Ptr{Uint8}, Ptr{Int32}, Ptr{Int32},
                     Ptr{$elty}, Ptr{Int32}, Ptr{Int32}, Ptr{Int32},
                     Ptr{$elty}, Ptr{$elty}, Ptr{Int32}, Ptr{Int32},
                     Ptr{Int32}, Ptr{$elty}, Ptr{Int32}, Ptr{Int32},
                     Ptr{Int32}, Ptr{$elty}, Ptr{Int32}, Ptr{Int32}),
                    &jobu, &jobvt, &m, &n,
                    A, &ia, &ja, desca,
                    s, U, &iu, &ju,
                    descu, Vt, &ivt, &jvt,
                    descvt, work, &lwork, info)
                if i == 1
                    lwork = convert(Int32, work[1])
                    work = Array($elty, lwork)
                end
            end

            info[1] > 0 && throw(ScaLAPACKException(info[1]))

            return U, s, Vt
        end
    end
end
for (fname, elty, relty) in ((:pcgesvd_, :Complex64, :Float32),
                             (:pzgesvd_, :Complex128, :Float64))
    @eval begin
        function pxgesvd!(jobu::Char, jobvt::Char, m::Integer, n::Integer, A::Matrix{$elty}, ia::Integer, ja::Integer, desca::Vector{Int32}, s::Vector{$relty}, U::Matrix{$elty}, iu::Integer, ju::Integer, descu::Vector{Int32}, Vt::Matrix{$elty}, ivt::Integer, jvt::Integer, descvt::Vector{Int32})
            # extract values

            # check

            # allocate
            info = Array(Int32, 1)
            work = Array($elty, 1)
            rwork = Array($relty, 1 + 4*max(m, n))
            lwork = -1

            # ccall
            for i = 1:2
                ccall(($(string(fname)), libscalapack), Void,
                    (Ptr{Uint8}, Ptr{Uint8}, Ptr{Int32}, Ptr{Int32},
                     Ptr{$elty}, Ptr{Int32}, Ptr{Int32}, Ptr{Int32},
                     Ptr{$relty}, Ptr{$elty}, Ptr{Int32}, Ptr{Int32},
                     Ptr{Int32}, Ptr{$elty}, Ptr{Int32}, Ptr{Int32},
                     Ptr{Int32}, Ptr{$elty}, Ptr{Int32}, Ptr{$relty},
                     Ptr{Int32}),
                    &jobu, &jobvt, &m, &n,
                    A, &ia, &ja, desca,
                    s, U, &iu, &ju,
                    descu, Vt, &ivt, &jvt,
                    descvt, work, &lwork, rwork,
                    info)
                if i == 1
                    lwork = convert(Int32, work[1])
                    work = Array($elty, lwork)
                end
            end

            info[1] > 0 && throw(ScaLAPACKException(info[1]))

            return U, s, Vt
        end
    end
end

for (fname, elty) in ((:psgemr2d_, :Float32),
                      (:pdgemr2d_, :Float64),
                      (:pcgemr2d_, :Complex64),
                      (:pzgemr2d_, :Complex128))
    @eval begin
        function pxgemr2d!(m::Integer, n::Integer, A::Matrix{$elty}, ia::Integer, ja::Integer, desca::Vector{Int32}, B::Matrix{$elty}, ib::Integer, jb::Integer, descb::Vector{Int32}, ictxt::Integer)

            ccall(($(string(fname)), libscalapack), Void,
                (Ptr{Int32}, Ptr{Int32}, Ptr{$elty}, Ptr{Int32},
                 Ptr{Int32}, Ptr{Int32}, Ptr{$elty}, Ptr{Int32},
                 Ptr{Int32}, Ptr{Int32}, Ptr{Int32}),
                &m, &n, A, &ia,
                &ja, desca, B, &ib,
                &jb, descb, &ictxt)
        end
    end
end

