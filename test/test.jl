using ScaLAPACK

# problem size
n = 5000
nb = div(n, 2)

# initialize grid
ScaLAPACK.sl_init(2, 2)

# who am I?
nprow, npcol, myrow, mycol = ScaLAPACK.blacs_gridinfo()

# Get DArray info
dA = ScaLAPACK.descinit(n, n, nb, nb, 0, 0, nb)

# allocate local array
A = randn(nb, nb)

# calculate DSVD
V, s, U = ScaLAPACK.pdgesvd!('N', 'N', n, n, A, 1, 1, dA, Array(Float64, n), Array(Float64, 0, 0), 0, 0, dA, Array(Float64, 0, 0), 0, 0, dA)

# show result
myrow == 0 && mycol == 0 && println(s[1:3])

# clean up
ScaLAPACK.blacs_gridexit()
ScaLAPACK.blacs_exit()
