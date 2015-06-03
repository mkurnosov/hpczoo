/*
 * heat.c: Serial implementation of Laplace equation solver by Jacobi iteration method.
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
 * Serial implementation
 *
 * Input parameters: rows, cols, EPS
 *
 * Usage: mpiexec -np <p> ./heat <rows> <cols>
 *
 * (C) Mikhail Kurnosov, 2015
 */
 
#include <stdio.h>
#include <stdlib.h>
#include <inttypes.h>
#include <math.h>
#include <sys/time.h>

#define EPS 0.001
#define PI 3.14159265358979323846

#define IND(i, j) ((i) * nx + (j))

double wtime()
{
    struct timeval t;
    gettimeofday(&t, NULL);
    return (double)t.tv_sec + (double)t.tv_usec * 1E-6;
}

void *xcalloc(size_t nmemb, size_t size)
{
    void *p = calloc(nmemb, size);
    if (p == NULL) {
        fprintf(stderr, "No enough memory\n");
        exit(EXIT_FAILURE);
    }
    return p;    
}

int main(int argc, char *argv[]) 
{
    double ttotal = -wtime();
    int rows = (argc > 1) ? atoi(argv[1]) : 100;
    int cols = (argc > 2) ? atoi(argv[2]) : 100;
    const char *filename = (argc > 3) ? argv[3] : NULL;    
    if (cols < 1 || rows < 1) {
        fprintf(stderr, "Invalid size of grid: rows %d, cols %d\n", rows, cols);
        exit(EXIT_FAILURE);
    }
        
    // Allocate memory for grids [0..ny - 1][0..nx - 1]
    int ny = rows;
    int nx = cols;
    double *local_grid = xcalloc(ny * nx, sizeof(*local_grid));
    double *local_newgrid = xcalloc(ny * nx, sizeof(*local_newgrid));

    // Fill boundary points: 
    //   - left and right borders are zero filled
    //   - top border: u(x, 0) = sin(pi * x)
    //   - bottom border: u(x, 1) = sin(pi * x) * exp(-pi)    
    double dx = 1.0 / (nx - 1.0); 
    // Initialize top border: u(x, 0) = sin(pi * x)
    for (int j = 0; j < nx; j++) {
        int ind = IND(0, j);
        local_newgrid[ind] = local_grid[ind] = sin(PI * dx * j);
    }
    // Initialize bottom border: u(x, 1) = sin(pi * x) * exp(-pi)
    for (int j = 0; j < nx; j++) {
        int ind = IND(ny - 1, j);
        local_newgrid[ind] = local_grid[ind] = sin(PI * dx * j) * exp(-PI);
    }
      
    int niters = 0;
    for (;;) {
        niters++;
        
        // Update interior points
        for (int i = 1; i < ny - 1; i++) {
            for (int j = 1; j < nx - 1; j++) {
                local_newgrid[IND(i, j)] = 
                    (local_grid[IND(i - 1, j)] + local_grid[IND(i + 1, j)] +
                     local_grid[IND(i, j - 1)] + local_grid[IND(i, j + 1)]) * 0.25;
            }
        }        
                        
        // Check termination condition
        double maxdiff = 0;
        for (int i = 1; i < ny - 1; i++) {
            for (int j = 1; j < nx - 1; j++) {
                int ind = IND(i, j);
                maxdiff = fmax(maxdiff, fabs(local_grid[ind] - local_newgrid[ind]));
            }
        }        
        // Swap grids (after termination local_grid will contain result)
        double *p = local_grid;
        local_grid = local_newgrid;
        local_newgrid = p;    

        if (maxdiff < EPS)
            break;              
    }
    ttotal += wtime();
    
    printf("# Heat 2D (serial): grid: rows %d, cols %d\n", rows, cols);    
    printf("# niters %d, total time %.6f\n", niters, ttotal);

    // Save grid
    if (filename) {
        FILE *fout = fopen(filename, "w");
        if (!fout) {
            perror("Can't open file");
            exit(EXIT_FAILURE);
        }
        for (int i = 0; i < ny; i++) {
            for (int j = 0; j < nx; j++)
                fprintf(fout, "%.4f ", local_grid[IND(i, j)]);
            fprintf(fout, "\n");
        }                
        fclose(fout);
    }
            
    return 0;
}

