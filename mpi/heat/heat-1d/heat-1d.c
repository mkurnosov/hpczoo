/*
 * heat-1d.c: MPI implementation of Laplace equation solver by Jacobi iteration method.
 * 
 * 2D Laplace equation: 
 *   \Delta u = 0
 *   \frac{\partial^2 u}{\partial x^2} + \frac{\partial^2 u}{\partial y^2} = 0
 * 
 * Domain: x in [0, 1],  y in [0, 1]
 * Boundary conditions: 
 *   u(x, 0) = sin(pi * x)
 *   u(x, 1) = sin(pi * x) * exp(-pi)
 *   u(0, y) = u(1, y) = 0
 * Initial value for interior points is 0
 * Analytical solution: 
 *   u(x, y) = sin(pi * x) * exp(-pi * y)
 * 
 * Parallel implementation: 
 * 1D domain decomposition of grid [0..rows - 1][0..cols -1]
 * Each process is assigned a strip of rows ~ O(rows / nprocs)
 *
 * Input parameters: rows, cols, EPS
 *
 * Usage: mpiexec -np <p> ./heat-1d <rows> <cols>
 *
 * (C) Mikhail Kurnosov, 2015
 */
 
#include <stdio.h>
#include <stdlib.h>
#include <inttypes.h>
#include <math.h>
#include <mpi.h>

#define EPS 0.001
#define PI 3.14159265358979323846

#define NELEMS(x) (sizeof((x)) / sizeof((x)[0]))
#define IND(i, j) ((i) * cols + (j))

void *xcalloc(size_t nmemb, size_t size)
{
    void *p = calloc(nmemb, size);
    if (p == NULL) {
        fprintf(stderr, "No enough memory\n");
        MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
    }
    return p;    
}

int get_block_size(int n, int rank, int nprocs)
{
    int s = n / nprocs;
    if (n % nprocs > rank)
        s++;
    return s;
}

int main(int argc, char *argv[]) 
{
    int commsize, rank;
    MPI_Init(&argc, &argv);
    double ttotal = -MPI_Wtime();
    
    MPI_Comm_size(MPI_COMM_WORLD, &commsize);   
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);               
    
    if (commsize < 2) {
        fprintf(stderr, "Invalid number of processes %d: must be greater than 1\n", commsize);
        MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
    }
    
    int rows, cols;
           
    // Broadcast command line arguments
    if (rank == 0) {
        rows = (argc > 1) ? atoi(argv[1]) : commsize * 100;
        cols = (argc > 2) ? atoi(argv[2]) : 100;        
        if (rows < commsize) {
            fprintf(stderr, "Number of rows %d less then number of processes %d\n", rows, commsize);
            MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
        }
        int args[2] = {rows, cols};
        MPI_Bcast(&args, NELEMS(args), MPI_INT, 0, MPI_COMM_WORLD);
    } else {
        int args[2];
        MPI_Bcast(&args, NELEMS(args), MPI_INT, 0, MPI_COMM_WORLD);
        rows = args[0];
        cols = args[1];
    }    
    
    // Allocate memory for local 1D subgrids with 2 halo rows [0..ny + 1][0..cols - 1]
    int ny = get_block_size(rows, rank, commsize);
    double *local_grid = xcalloc((ny + 2) * cols, sizeof(*local_grid));
    double *local_newgrid = xcalloc((ny + 2) * cols, sizeof(*local_newgrid));

    // Fill boundary points: 
    //   - left and right borders are zero filled
    //   - top border: u(x, 0) = sin(pi * x)
    //   - bottom border: u(x, 1) = sin(pi * x) * exp(-pi)    
    double dx = 1.0 / (cols - 1.0); 
    if (rank == 0) {
        // Initialize top border: u(x, 0) = sin(pi * x)
        for (int j = 0; j < cols; j++) {
            int ind = IND(0, j);
            local_newgrid[ind] = local_grid[ind] = sin(PI * dx * j);
        }
    }
    if (rank == commsize - 1) {
        // Initialize bottom border: u(x, 1) = sin(pi * x) * exp(-pi)
        for (int j = 0; j < cols; j++) {
            int ind = IND(ny + 1, j);
            local_newgrid[ind] = local_grid[ind] = sin(PI * dx * j) * exp(-PI);
        }
    }   
    
    // Neighbours
    int top = (rank > 0) ? rank - 1 : MPI_PROC_NULL;
    int bottom = (rank < commsize - 1) ? rank + 1 : MPI_PROC_NULL;
    
    // Top and bottom borders type
    MPI_Datatype row;        
    MPI_Type_contiguous(cols, MPI_DOUBLE, &row);
    MPI_Type_commit(&row);

    MPI_Request reqs[4];
    double thalo = 0;
    double treduce = 0;

    int niters = 0;
    for (;;) {
        niters++;
        
        // Update interior points
        for (int i = 1; i <= ny; i++) {
            for (int j = 1; j < cols - 1; j++) {
                local_newgrid[IND(i, j)] = 
                    (local_grid[IND(i - 1, j)] + local_grid[IND(i + 1, j)] +
                     local_grid[IND(i, j - 1)] + local_grid[IND(i, j + 1)]) * 0.25;
            }
        }        
                        
        // Check termination condition
        double maxdiff = 0;
        for (int i = 1; i <= ny; i++) {
            for (int j = 1; j < cols - 1; j++) {
                int ind = IND(i, j);
                maxdiff = fmax(maxdiff, fabs(local_grid[ind] - local_newgrid[ind]));
            }
        }        
        // Swap grids (after termination local_grid will contain result)
        double *p = local_grid;
        local_grid = local_newgrid;
        local_newgrid = p;    

        treduce -= MPI_Wtime();
        MPI_Allreduce(MPI_IN_PLACE, &maxdiff, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);
        treduce += MPI_Wtime();               
        if (maxdiff < EPS)
            break;      
        
        // Halo exchange: T = 4 * (a + b * cols)
        thalo -= MPI_Wtime();                
        MPI_Irecv(&local_grid[IND(0, 0)], 1, row, top, 0, MPI_COMM_WORLD, &reqs[0]);         // top
        MPI_Irecv(&local_grid[IND(ny + 1, 0)], 1, row, bottom, 0, MPI_COMM_WORLD, &reqs[1]); // bottom
        MPI_Isend(&local_grid[IND(1, 0)], 1, row, top, 0, MPI_COMM_WORLD, &reqs[2]);         // top
        MPI_Isend(&local_grid[IND(ny, 0)], 1, row, bottom, 0, MPI_COMM_WORLD, &reqs[3]);     // bottom
        MPI_Waitall(4, reqs, MPI_STATUS_IGNORE);
        thalo += MPI_Wtime();
    }
    MPI_Type_free(&row);
       
    free(local_newgrid);
    free(local_grid);

    ttotal += MPI_Wtime();

    if (rank == 0)
        printf("# Heat 1D (mpi): grid: rows %d, cols %d, procs %d\n", rows, cols, commsize);

    int namelen;
    char procname[MPI_MAX_PROCESSOR_NAME];
    MPI_Get_processor_name(procname, &namelen);    
    printf("# P %4d on %s: grid ny %d nx %d, total %.6f, mpi %.6f (%.2f) = allred %.6f (%.2f) + halo %.6f (%.2f)\n", 
           rank, procname, ny, cols, ttotal, treduce + thalo, (treduce + thalo) / ttotal, 
           treduce, treduce / (treduce + thalo), thalo, thalo / (treduce + thalo)); 
        
    double prof[3] = {ttotal, treduce, thalo};    
    if (rank == 0) {
        MPI_Reduce(MPI_IN_PLACE, prof, NELEMS(prof), MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
        printf("# procs %d : grid %d %d : niters %d : total time %.6f : mpi time %.6f : allred %.6f : halo %.6f\n", 
               commsize, rows, cols, niters, prof[0], prof[1] + prof[2], prof[1], prof[2]);
    } else {
        MPI_Reduce(prof, NULL, NELEMS(prof), MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
    }

    MPI_Finalize();    
    return 0;
}

