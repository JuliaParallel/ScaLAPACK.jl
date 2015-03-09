using ScaLAPACK
using MPI

MPI.Init()

Base.disable_threaded_libs()

# problem size
m = 8
n = 5
# bf = 3
# nb = div(m, bf)
nb = 2

# initialize grid
id, nprocs = ScaLAPACK.BLACS.pinfo()
ic = ScaLAPACK.sl_init(trunc(Integer, sqrt(nprocs)), div(nprocs, trunc(Integer, sqrt(nprocs))))

# who am I?
nprow, npcol, myrow, mycol = ScaLAPACK.BLACS.gridinfo(ic)
np = ScaLAPACK.numroc(m, nb, myrow, 0, nprow)
nq = ScaLAPACK.numroc(n, nb, mycol, 0, npcol)
print("myrow: $myrow, mycol: $mycol, nb: $nb, np: $np, nq: $nq\n")

if nprow >= 0 && npcol >= 0
    # Get DArray info
    dA = ScaLAPACK.descinit(m, n, nb, nb, 0, 0, ic, np)

    # allocate local array
    A = randn(Int(np), Int(nq))
    # A = float32(randn(Int(np), Int(nq)))
    # A = complex(randn(Int(np), Int(nq)), randn(Int(np), Int(nq)))
    # A = complex64(complex(randn(Int(np), Int(nq)), randn(Int(np), Int(nq))))

    # calculate DSVD
    V, s, U = ScaLAPACK.pxgesvd!('N', 'N', m, n, A, 1, 1, dA, Array(typeof(real(A[1])), n), Array(eltype(A), 1, 1), 0, 0, dA, Array(eltype(A), 1, 1), 0, 0, dA)

    # show result
    if myrow == 0 && mycol == 0
        println(s[1:3])
    end

    # clean up
    tmp = ScaLAPACK.BLACS.gridexit(ic)

end
ScaLAPACK.BLACS.exit()
