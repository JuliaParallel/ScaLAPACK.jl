using ScaLAPACK
using MPI

MPI.Init()

Base.disable_threaded_libs()

# problem size
n = 5000
bf = 100
nb = div(n, bf)

# initialize grid
id, nprocs = ScaLAPACK.blacs_pinfo()
ic = ScaLAPACK.sl_init(itrunc(sqrt(nprocs)), div(nprocs, itrunc(sqrt(nprocs))))

# who am I?
nprow, npcol, myrow, mycol = ScaLAPACK.blacs_gridinfo(ic)
np = ScaLAPACK.numroc(n, nb, myrow, 0, nprow)
nq = ScaLAPACK.numroc(n, nb, mycol, 0, npcol)
print("myrow: $myrow, mycol: $mycol, nb: $nb, np: $np, nq: $nq\n")

if nprow >= 0 && npcol >= 0
    # Get DArray info
    dA = ScaLAPACK.descinit(n, n, nb, nb, 0, 0, ic, np)

    # allocate local array
    A = randn(int(np), int(nq))

    # calculate DSVD
    V, s, U = ScaLAPACK.pdgesvd!('N', 'N', n, n, A, 1, 1, dA, Array(Float64, n), Array(Float64, 0, 0), 0, 0, dA, Array(Float64, 0, 0), 0, 0, dA)

    # show result
    myrow == 0 && mycol == 0 && println(s[1:3])

    # clean up
    ScaLAPACK.blacs_gridexit(ic)
end

ScaLAPACK.blacs_exit()
